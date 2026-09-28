"""Tests for actus/app/server.py (T-UI): the FastAPI + WebSocket dashboard backend.

Uses ``fastapi.testclient.TestClient`` throughout, so no real socket or browser is needed. The
WebSocket relay is exercised against the P0 replay source per CLAUDE.md's convention of testing
against replay rather than a camera.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from actus.app.server import _handle_incoming, create_app
from actus.capture.replay import REQUIRED_SCENARIOS, ReplaySource
from actus.core.bus import Bus, BusClosed
from actus.core.config import load_config
from actus.core.contracts import (
    Answer,
    AnswerValue,
    InputVia,
    Predicates,
    Topic,
    from_jsonable,
)
from actus.protocol.schema import load_protocol

PROTOCOL_PATH = "protocols/bas_boxes.yaml"


def _make_client(**kwargs) -> tuple[TestClient, Bus]:
    bus = Bus()
    app = create_app(bus, **kwargs)
    return TestClient(app), bus


# -- REST endpoints --------------------------------------------------------------------------


def test_protocol_endpoint_returns_bas_boxes():
    client, _bus = _make_client()
    resp = client.get("/api/protocol")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["id"] == "bas_boxes_v1"
    step_ids = [s["id"] for s in body["steps"]]
    assert step_ids == ["S1", "S2", "S3", "S4", "S5", "S6"]
    assert body["steps"][1]["requires_hoi"] == "red"
    assert body["decision"]["p_enter"] == pytest.approx(0.8)


def test_rack_endpoint_reports_unmeasured_placeholder():
    client, _bus = _make_client()
    resp = client.get("/api/rack")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["measured"] is False  # H-PROPS hasn't measured the board yet
    assert body["board"]["width"] == pytest.approx(0.90)


def test_report_endpoint_honest_when_no_eval_has_run():
    client, _bus = _make_client()
    resp = client.get("/api/report")
    assert resp.status_code == 200
    assert resp.json() == {"available": False}


def test_bus_stats_endpoint():
    client, bus = _make_client()
    bus.publish(Topic.PREDICATES, Predicates(0, 0.0, {"outer.open": 0.1}, 1.0))
    resp = client.get("/api/bus_stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["published"]["predicates"] == 1


def test_root_reports_ui_not_built_when_dist_is_missing(tmp_path):
    # Point ui.static_dir at a directory that can't exist, regardless of whether this checkout
    # happens to have a built ui/dist sitting around (gitignored; present after `npm run build`).
    cfg = load_config(overrides={"ui.static_dir": str(tmp_path / "not-built")})
    client, _bus = _make_client(config=cfg)
    resp = client.get("/")
    assert resp.status_code == 503
    assert "npm run build" in resp.json()["hint"]


# -- demo (/api/scenarios, /api/demo/*) -------------------------------------------------------


def test_scenarios_endpoint_lists_the_five_required_scenarios():
    client, _bus = _make_client()
    resp = client.get("/api/scenarios")
    assert resp.status_code == 200
    names = {s["name"] for s in resp.json()}
    assert set(REQUIRED_SCENARIOS) <= names


def test_demo_start_stop_feeds_the_bus():
    client, bus = _make_client()
    sub = bus.subscribe(Topic.PREDICATES, maxsize=256, name="probe")
    resp = client.post("/api/demo/start", params={"scenario": "skip_S2"})
    assert resp.status_code == 200
    assert resp.json() == {"started": "skip_S2", "simulated": True}

    env = sub.get(timeout=5.0)
    assert env is not None
    assert env.topic == Topic.PREDICATES

    resp = client.post("/api/demo/stop")
    assert resp.status_code == 200
    assert resp.json() == {"stopped": True}


def test_demo_start_rejects_unknown_scenario():
    client, _bus = _make_client()
    resp = client.post("/api/demo/start", params={"scenario": "not_a_scenario"})
    assert resp.status_code == 404


def test_demo_routes_absent_when_disabled():
    client, _bus = _make_client(enable_demo=False)
    assert client.get("/api/scenarios").status_code == 404
    # A GET to an unregistered path falls through to the "/" static mount as a plain 404; POST
    # falls through the same way, but StaticFiles answers a non-GET/HEAD method with 405 rather
    # than 404. Either way, the point holds: the real /api/demo/start handler never runs.
    assert client.post("/api/demo/start").status_code in (404, 405)


# -- WebSocket relay ---------------------------------------------------------------------------


def test_websocket_relays_a_bus_message():
    client, bus = _make_client()
    with client.websocket_connect("/ws") as ws:
        bus.publish(Topic.PREDICATES, Predicates(3, 1.5, {"red.in_zone_A": 0.9}, 0.97))
        msg = ws.receive_json()
    assert msg["topic"] == "predicates"
    payload = msg["payload"]
    assert payload["type"] == "Predicates"
    rebuilt = from_jsonable(payload)
    assert isinstance(rebuilt, Predicates)
    assert rebuilt.frame_idx == 3
    assert rebuilt.values["red.in_zone_A"] == pytest.approx(0.9)


def test_websocket_does_not_relay_frame_topics():
    """FramePacket isn't JSON-encodable; it must never be pushed down the JSON socket."""
    import numpy as np

    from actus.core.contracts import FramePacket

    client, bus = _make_client()
    with client.websocket_connect("/ws") as ws:
        packet = FramePacket(0, 0.0, np.zeros((4, 4, 3), dtype="uint8"), "cam0")
        bus.publish(Topic.FRAMES, packet)
        bus.publish(Topic.PREDICATES, Predicates(0, 0.0, {"outer.open": 0.05}, 0.9))
        msg = ws.receive_json()  # the only thing that can arrive is the predicates message
    assert msg["topic"] == "predicates"


def test_websocket_answer_from_browser_is_published_as_gui():
    client, bus = _make_client()
    sub = bus.subscribe(Topic.ANSWERS, maxsize=8, name="probe")
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "Answer", "question_id": "q-S3", "value": "yes"})
        # give the server a moment to process the inbound message before disconnecting
        env = sub.get(timeout=5.0)
    assert env is not None
    answer = env.payload
    assert isinstance(answer, Answer)
    assert answer.question_id == "q-S3"
    assert answer.value == AnswerValue.YES
    assert answer.via == InputVia.GUI


def test_websocket_survives_a_malformed_message():
    """Invalid JSON or a non-object is skipped; the next good message still gets through on the
    same connection."""
    client, bus = _make_client()
    sub = bus.subscribe(Topic.ANSWERS, maxsize=8, name="probe")
    with client.websocket_connect("/ws") as ws:
        ws.send_text("not json {")
        ws.send_text("[1, 2]")
        ws.send_json({"type": "Answer", "question_id": "q-S3", "value": "no"})
        env = sub.get(timeout=5.0)
    assert env is not None and env.payload.value == AnswerValue.NO


def test_handle_incoming_ignores_malformed_messages(caplog):
    bus = Bus()
    sub = bus.subscribe([Topic.ANSWERS, Topic.COMMANDS], maxsize=8, name="probe")
    _handle_incoming(bus, {"type": "Answer", "question_id": "q1"})  # missing "value"
    _handle_incoming(bus, {"type": "Nonsense"})
    with pytest.raises(BusClosed):
        bus.close()
        sub.get(timeout=0.01)
    assert sub.delivered == 0


# -- the five required replay scenarios over the socket ---------------------------------------


@pytest.mark.parametrize("scenario", REQUIRED_SCENARIOS)
def test_dashboard_runs_from_replay_for_required_scenarios(scenario):
    """Build plan §5 T-UI 'done when': the dashboard runs from replay for all 5 scenarios."""
    client, bus = _make_client()
    protocol = load_protocol(PROTOCOL_PATH)
    src = ReplaySource(scenario, protocol=protocol, fps=2.0, render=False)
    n_frames = len(src)
    n_steps = len(protocol.steps)
    expected = n_frames * (1 + n_steps + 1 + 1)  # predicates + evidence*steps + actions + dets

    with client.websocket_connect("/ws") as ws:
        published = src.run(bus, realtime=False)
        assert published == n_frames

        topics_seen: set[str] = set()
        for _ in range(expected):
            msg = ws.receive_json()
            topics_seen.add(msg["topic"])
            json.dumps(msg)  # already-decoded, but confirms it round-trips as plain JSON

    assert topics_seen == {"predicates", "evidence", "actions", "detections"}
