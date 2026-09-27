"""FastAPI + WebSocket server for the mission-console dashboard.

Serves the built React app (``ui/dist``) and a WebSocket at ``/ws`` that relays bus topics to
the browser as JSON (``to_jsonable``), and relays the browser's ``Answer`` / ``Command`` messages
back onto the bus. Video does not go over the WebSocket (``FramePacket`` is not JSON-encodable);
it is served as an MJPEG fallback at ``/stream/mjpeg``, with WebRTC (via MediaMTX, T-IO) as the
primary path the frontend tries first.

Standalone / demo use, so this lane never waits on the others::

    uv run python -m actus.app.server --replay correct

That feeds a P0 :mod:`actus.capture.replay` scenario onto the bus and serves the dashboard
against it. ``/api/demo/*`` (also backed by the replay source) lets the *browser* pick a
scenario instead; both paths only ever produce data watermarked SIMULATED.

When the integrator (T-INT) wires the real pipeline it calls :func:`create_app` with its own
``Bus`` and passes ``enable_demo=False``, so the dashboard can't inject synthetic data onto a
live bus.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import threading
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from actus.core.bus import Bus, BusClosed
from actus.core.config import Config, load_config, load_yaml, resolve_path
from actus.core.contracts import (
    Answer,
    AnswerValue,
    Command,
    CommandName,
    InputVia,
    Topic,
    to_jsonable,
)
from actus.protocol.schema import Protocol, load_protocol

try:
    # A real (not TYPE_CHECKING-only) import, because FastAPI resolves a route handler's
    # annotations (e.g. `websocket: WebSocket`) via `typing.get_type_hints` against the
    # *module's* globals at route-registration time — a name visible only under TYPE_CHECKING,
    # or bound solely inside create_app's local scope, resolves to nothing and FastAPI silently
    # mistreats `websocket: WebSocket` as a plain (required) query parameter named "websocket".
    # Guarded so the base install (no `ui` extra) can still import this module.
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import StreamingResponse
except ImportError:  # pragma: no cover - exercised by CI's base (non-`ui`) install
    if not TYPE_CHECKING:
        FastAPI = WebSocket = StreamingResponse = None

logger = logging.getLogger("actus.app.server")

# Topics relayed to the dashboard over the WebSocket as JSON.
JSON_TOPICS: tuple[Topic, ...] = (
    Topic.DETECTIONS,
    Topic.HANDS,
    Topic.POSE,
    Topic.RACK,
    Topic.PREDICATES,
    Topic.EVIDENCE,
    Topic.ACTIONS,
    Topic.EVENTS,
    Topic.ALERTS,
    Topic.PROMPTS,
    Topic.ASKS,
    Topic.ANSWERS,
    Topic.COMMANDS,
    Topic.STATS,
)

# Frame-carrying topics; these go over MJPEG / WebRTC, never the JSON WebSocket.
FRAME_TOPICS: tuple[Topic, ...] = (Topic.ANNOTATED, Topic.FRAMES)


def create_app(
    bus: Bus,
    *,
    config: Config | None = None,
    protocol: Protocol | None = None,
    enable_demo: bool = True,
) -> FastAPI:
    """Build the FastAPI app around an already-running ``bus``.

    ``protocol`` defaults to loading ``config.protocol.path``; pass it explicitly to avoid a
    second parse when the caller already has one loaded. ``enable_demo`` mounts ``/api/demo/*``,
    which feeds :mod:`actus.capture.replay` scenarios onto ``bus`` on request — leave it True for
    a standalone dashboard, and set it False once a real pipeline owns ``bus``.
    """
    from fastapi import FastAPI, WebSocketDisconnect
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    config = config if config is not None else load_config()
    if protocol is None:
        protocol = _try_load_protocol(config)

    app = FastAPI(title="ACTUS mission console")
    app.state.bus = bus
    app.state.config = config
    app.state.protocol = protocol
    app.state.demo_stop = None
    app.state.demo_thread = None

    @app.get("/api/protocol")
    def get_protocol() -> Any:
        if protocol is None:
            return {"available": False}
        return _protocol_summary(protocol)

    @app.get("/api/rack")
    def get_rack() -> Any:
        rack_path = config.get("rack.config", "configs/rack.yaml")
        try:
            data = load_yaml(rack_path)
        except OSError:
            return {"available": False}
        return {"available": True, **data}

    @app.get("/api/report")
    def get_report() -> Any:
        path = resolve_path("runs/eval/report.json")
        if not path.exists():
            return {"available": False}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            return {"available": False, "error": str(e)}
        return {**data, "available": True}

    @app.get("/api/bus_stats")
    def get_bus_stats() -> Any:
        return bus.stats()

    if enable_demo:
        _mount_demo_routes(app, bus, config, protocol)

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        sub = bus.subscribe(JSON_TOPICS, maxsize=512, policy="drop_oldest", name="ui-ws")
        send_task = asyncio.create_task(_relay(websocket, sub))
        try:
            while True:
                data = await websocket.receive_json()
                _handle_incoming(bus, data)
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("ws_endpoint: unexpected error")
        finally:
            send_task.cancel()
            sub.unsubscribe()

    @app.get("/stream/mjpeg")
    async def mjpeg_stream() -> StreamingResponse:
        return StreamingResponse(
            _mjpeg_frames(bus), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    static_dir = resolve_path(config.get("ui.static_dir", "ui/dist"))
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
    else:

        @app.get("/")
        def not_built() -> Any:
            return JSONResponse(
                {
                    "error": f"UI not built: {static_dir} does not exist.",
                    "hint": "cd ui && npm install && npm run build",
                },
                status_code=503,
            )

    return app


def _try_load_protocol(config: Config) -> Protocol | None:
    path = config.get("protocol.path")
    if not path:
        return None
    try:
        return load_protocol(path)
    except (OSError, ValueError) as e:
        logger.warning("could not load protocol %r: %s", path, e)
        return None


def _protocol_summary(protocol: Protocol) -> dict[str, Any]:
    order = "strict" if protocol.order == "strict" else [list(e) for e in protocol.order]
    return {
        "available": True,
        "id": protocol.id,
        "title": protocol.title,
        "objects": {
            name: {"label": spec.label, "classes": list(spec.classes)}
            for name, spec in protocol.objects.items()
        },
        "zones": {name: [list(pt) for pt in pts] for name, pts in protocol.zones.items()},
        "steps": [
            {
                "id": s.id,
                "text": s.text,
                "pre": dict(s.effective_pre),
                "post": dict(s.post),
                "requires_hoi": s.requires_hoi,
                "ask": s.ask,
            }
            for s in protocol.steps
        ],
        "order": order,
        "timing": {
            "dwell_s": protocol.timing.dwell_s,
            "timeout_s": protocol.timing.timeout_s,
            "min_duration_s": dict(protocol.timing.min_duration_s),
        },
        "decision": {
            "p_enter": protocol.decision.p_enter,
            "p_exit": protocol.decision.p_exit,
            "ask_band": list(protocol.decision.ask_band),
            "ask_after_s": protocol.decision.ask_after_s,
        },
    }


def _mount_demo_routes(app: FastAPI, bus: Bus, config: Config, protocol: Protocol | None) -> None:
    @app.get("/api/scenarios")
    def get_scenarios() -> Any:
        from actus.capture.replay import SCENARIOS

        return [
            {"name": s.name, "description": s.description, "duration_s": s.duration_s}
            for s in SCENARIOS.values()
        ]

    @app.post("/api/demo/start")
    def start_demo(scenario: str = "correct") -> Any:
        from fastapi import HTTPException

        from actus.capture.replay import ReplaySource, get_scenario

        try:
            get_scenario(scenario)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        _stop_demo(app)
        stop = threading.Event()
        src = ReplaySource(
            scenario,
            protocol=protocol,
            render=True,
            fps=config.get("replay.fps", 15.0),
            seed=config.get("replay.seed", 0),
            noise=config.get("replay.noise", 0.04),
        )
        thread = threading.Thread(
            target=src.run,
            kwargs={"bus": bus, "realtime": True, "stop": stop},
            daemon=True,
            name=f"demo-replay-{scenario}",
        )
        app.state.demo_stop = stop
        app.state.demo_thread = thread
        thread.start()
        return {"started": scenario, "simulated": True}

    @app.post("/api/demo/stop")
    def stop_demo() -> Any:
        _stop_demo(app)
        return {"stopped": True}


def _stop_demo(app: FastAPI) -> None:
    stop: threading.Event | None = app.state.demo_stop
    thread: threading.Thread | None = app.state.demo_thread
    if stop is not None:
        stop.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=2.0)
    app.state.demo_stop = None
    app.state.demo_thread = None


async def _relay(websocket: WebSocket, sub: Any) -> None:
    from starlette.concurrency import run_in_threadpool

    try:
        while True:
            env = await run_in_threadpool(sub.get, 0.5)
            if env is None:
                continue
            await websocket.send_json(
                {
                    "topic": env.topic.value,
                    "seq": env.seq,
                    "t_pub": env.t_pub,
                    "payload": to_jsonable(env.payload),
                }
            )
    except BusClosed:
        return
    except asyncio.CancelledError:
        return
    except Exception:
        logger.exception("_relay: unexpected error")


def _handle_incoming(bus: Bus, data: Mapping[str, Any]) -> None:
    """Turn a browser message into an ``Answer`` or ``Command`` published on the bus.

    ``t`` is stamped here (the receiver), since the browser has no pipeline clock; this mirrors
    the contract's own convention for a receiver-stamped ``Answer.frame_idx``. Voice answers do
    not come through here — T-VOICE's ASR publishes them straight onto the bus — so ``via`` is
    always forced to ``gui``.
    """
    kind = data.get("type")
    now = time.monotonic()
    try:
        if kind == "Answer":
            answer = Answer(
                question_id=str(data["question_id"]),
                value=AnswerValue(data["value"]),
                via=InputVia.GUI,
                t=now,
            )
            bus.publish(Topic.ANSWERS, answer)
        elif kind == "Command":
            command = Command(name=CommandName(data["name"]), via=InputVia.GUI, t=now)
            bus.publish(Topic.COMMANDS, command)
        else:
            logger.warning("ws: ignoring unknown incoming message type %r", kind)
    except (KeyError, ValueError, BusClosed) as e:
        logger.warning("ws: bad incoming message %r: %s", data, e)


async def _mjpeg_frames(bus: Bus):
    import cv2
    from starlette.concurrency import run_in_threadpool

    sub = bus.subscribe(FRAME_TOPICS, maxsize=2, policy="drop_oldest", name="ui-mjpeg")
    try:
        while True:
            env = await run_in_threadpool(sub.get, 1.0)
            if env is None:
                continue
            packet = env.payload
            ok, buf = cv2.imencode(".jpg", packet.image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            body = buf.tobytes()
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(body)).encode()
                + b"\r\n\r\n"
                + body
                + b"\r\n"
            )
    except BusClosed:
        return
    finally:
        sub.unsubscribe()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the ACTUS mission-console server.")
    ap.add_argument("--config", default=None, help="override YAML on top of configs/default.yaml")
    ap.add_argument("--override", action="append", default=[], help="key.path=value")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument(
        "--replay",
        default=None,
        help="feed a P0 replay scenario (e.g. 'correct') instead of waiting for a real pipeline",
    )
    ap.add_argument(
        "--no-demo-api",
        action="store_true",
        help="disable /api/demo/* and /api/scenarios (use once a real pipeline owns the bus)",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config, overrides=args.override)
    bus = Bus()
    app = create_app(bus, config=config, enable_demo=not args.no_demo_api)

    stop = threading.Event()
    replay_thread: threading.Thread | None = None
    if args.replay:
        from actus.capture.replay import ReplaySource

        src = ReplaySource(
            args.replay,
            fps=config.get("replay.fps", 15.0),
            seed=config.get("replay.seed", 0),
            noise=config.get("replay.noise", 0.04),
            render=config.get("replay.render", True),
        )
        replay_thread = threading.Thread(
            target=src.run,
            kwargs={"bus": bus, "realtime": True, "stop": stop},
            daemon=True,
            name=f"replay-{args.replay}",
        )
        replay_thread.start()
        logger.info("feeding replay scenario %r onto the bus", args.replay)

    import uvicorn

    host = args.host or config.get("ui.host", "127.0.0.1")
    port = args.port or config.get("ui.port", 8000)
    try:
        uvicorn.run(app, host=host, port=port)
    finally:
        stop.set()
        if replay_thread is not None:
            replay_thread.join(timeout=2.0)
        bus.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
