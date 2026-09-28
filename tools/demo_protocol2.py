"""T-PROTO2 demo run: the second protocol, end to end, from its YAML alone. SIMULATED scenes.

    uv run python tools/demo_protocol2.py --list
    uv run python tools/demo_protocol2.py correct
    uv run python tools/demo_protocol2.py skip_cup
    uv run python tools/demo_protocol2.py occluded_S5            # the ask path, answered "yes"
    uv run python tools/demo_protocol2.py all --no-video
    uv run python tools/demo_protocol2.py correct --verify-determinism

The build plan's T-PROTO2 card asks for a second protocol "from YAML only" plus a demo run. The P0
replay source only simulates the box world, so this tool brings its own small scene for the
cup / bottle / book world: a scripted operator carries each object to its place, the objects are
drawn on a board with real ArUco markers, and each frame goes through the real pipeline core
(``actus.app.pipeline.PipelineCore``): rack frame solved from the rendered markers, boxes projected
onto the rack plane, the world-state filter, hand proximity, step evidence, the protocol engine,
the overlay, the hash-chained event log and the downlink packet.

What is simulated is only the camera and the neural perception: the boxes and hand keypoints are
generated from the script (with position noise and confidence jitter) instead of coming from
YOLO and MediaPipe. Every frame is watermarked SIMULATED and ``run.json`` says so. Nothing here
is a number for the report card. It shows that the sequencing, alerts, asks and logs work on a
procedure nobody wrote code for. The same protocol on a real take (Task B footage) is:

    uv run python tools/run_offline.py data/raw/<take>.mp4 --config configs/sample_mixing.yaml

Outputs go to ``runs/<run_id>/``: ``events.jsonl``, ``downlink.json``, ``run.json`` (with the
scene's ground truth) and ``video/annotated_NNN.mp4``.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from actus.app.pipeline import (
    PipelineClock,
    PipelineCore,
    Runner,
    RunSummary,
    SourceFrame,
    new_run_id,
    outcome_line,
    run_dir_for,
    write_downlink_for,
    write_json,
)
from actus.capture.replay import IMG_H, IMG_W, MARKER_M, PX_PER_M, RACK_MARKERS, rack_to_px
from actus.core.config import Config, load_config
from actus.core.contracts import (
    Alert,
    Detection,
    DetectionSet,
    FramePacket,
    HandObs,
    HandSet,
    StepEvent,
)
from actus.io.eventlog import EventLog, iso_at, verify_log
from actus.io.recorder import SegmentWriter
from actus.protocol.engine import describe
from actus.protocol.schema import ZONE_PRED_RE, Protocol, load_protocol
from actus.spatial.zones import zones_from_protocol

CONFIG_PATH = "configs/sample_mixing.yaml"
FPS = 15.0  # the scene's own declared rate; t = frame_idx / FPS
SOURCE_ID = "sim-scene"

# -- the scene ---------------------------------------------------------------------------------
# Rack metres (configs/rack.yaml frame). The layout is chosen so that no reach or carry path
# brings the hand near an object it isn't handling, and every start/end lies well inside its zone
# (or, for the cup's start, in none).

OBJECTS = ("cup", "bottle", "book")  # COCO classes, also the protocol's object names
TRACK_ID = {"cup": 1, "bottle": 2, "book": 3}
REST = (0.45, 0.78)  # where the hand waits between actions, clear of every object
START = {"cup": (0.22, 0.72), "bottle": (0.60, 0.52), "book": (0.74, 0.64)}
DEST = {"cup": (0.20, 0.45), "bottle": (0.37, 0.45), "book": (0.675, 0.225)}
SIZE_M = {"cup": (0.07, 0.08), "bottle": (0.07, 0.16), "book": (0.14, 0.10)}  # width, height
COLOUR = {"cup": (200, 140, 60), "bottle": (80, 170, 70), "book": (60, 90, 170)}  # BGR
# Ground-truth label of the first / second half of an action (T-REC annotation convention).
ACT_LABELS = {"cup": ("S1", "S1"), "bottle": ("S2", "S3"), "book": ("S4", "S5")}
EDGE_OF_R = (0.55, 0.225)  # the record area's left edge: a placement no one can call

POS_NOISE_M = 0.003  # detection / hand position noise (std)
CONF_MEAN, CONF_STD = 0.90, 0.03

LEAD_S, REACH_S, GRAB_S, WITHDRAW_S, GAP_S, TAIL_S = 1.5, 0.7, 0.3, 0.7, 1.0, 4.0

_ANG = np.linspace(0.0, 2.0 * np.pi, 21, endpoint=False)
_RAD = np.linspace(0.004, 0.02, 21)
_HAND_SHAPE_M = np.stack([np.cos(_ANG) * _RAD, np.sin(_ANG) * _RAD], axis=1)  # 21 points, 2 cm


@dataclass(frozen=True)
class Act:
    """The operator carries ``obj`` to ``dest`` (default: its protocol place) and lets go."""

    obj: str
    dest: tuple[float, float] | None = None
    carry_s: float = 2.0


@dataclass(frozen=True)
class Wait:
    s: float


@dataclass(frozen=True)
class Blackout:
    """The view is blocked ``delay_s`` after the previous action's arrival, for ``dur_s``.
    Detections and hands vanish and the frame shows an occluder over the work area."""

    delay_s: float = 0.4
    dur_s: float = 6.0


Item = Act | Wait | Blackout


@dataclass(frozen=True)
class Scenario:
    name: str
    summary: str
    items: tuple[Item, ...]
    answer: str | None = None  # default scripted operator answer to any Ask


SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in (
        Scenario(
            "correct",
            "Cup, bottle, book in protocol order. Expect S1..S5 verified, no alerts, no asks.",
            (Act("cup"), Act("bottle"), Act("book")),
        ),
        Scenario(
            "skip_cup",
            "The cup is never placed. Expect an out_of_order alert when the bottle is taken "
            "(S2), then S1 marked skipped retroactively with one skip alert.",
            (Act("bottle"), Act("book")),
        ),
        Scenario(
            "bottle_first",
            "The bottle is staged before the cup. Expect 'Out of sequence: bottle moved out of "
            "Zone S before cup', a re-prompt for the cup, then the run completes; no skip.",
            (Act("bottle"), Act("cup"), Act("book")),
        ),
        Scenario(
            "unsure_S5",
            "The book is left exactly on the edge of the record area. The evidence for S5 "
            "hovers near 0.5: the engine must not verify it. It asks if the evidence sits in "
            "its ask band long enough.",
            (Act("cup"), Act("bottle"), Act("book", dest=EDGE_OF_R)),
        ),
        Scenario(
            "occluded_S5",
            "A helper blocks the view as the book is placed. Expect a blindness Ask about S5 "
            "('Did you place it?') and, with the scripted 'yes', S5 operator_confirmed.",
            (Act("cup"), Act("bottle"), Act("book"), Blackout()),
            answer="yes",
        ),
    )
}


def get_scenario(name: str | Scenario) -> Scenario:
    if isinstance(name, Scenario):
        return name
    try:
        return SCENARIOS[name]
    except KeyError:
        raise KeyError(f"unknown scenario {name!r}; choose from {sorted(SCENARIOS)}") from None


Point = tuple[float, float]


@dataclass(frozen=True)
class Timeline:
    """Piecewise-linear object and hand paths, plus the blackout windows and truth labels."""

    obj_keys: Mapping[str, tuple[tuple[float, Point], ...]]
    hand_keys: tuple[tuple[float, Point], ...]
    blackouts: tuple[tuple[float, float], ...]
    labels: tuple[tuple[float, float, str], ...]  # (t0, t1, label); idle elsewhere
    duration_s: float

    @staticmethod
    def _at(keys: Sequence[tuple[float, Point]], t: float) -> Point:
        ts = [k[0] for k in keys]
        return (
            float(np.interp(t, ts, [k[1][0] for k in keys])),
            float(np.interp(t, ts, [k[1][1] for k in keys])),
        )

    def objects_at(self, t: float) -> dict[str, Point]:
        return {o: self._at(self.obj_keys[o], t) for o in OBJECTS}

    def hand_at(self, t: float) -> Point:
        return self._at(self.hand_keys, t)

    def blocked(self, t: float) -> bool:
        return any(a <= t < b for a, b in self.blackouts)

    def label_at(self, t: float) -> str:
        for a, b, name in self.labels:
            if a <= t < b:
                return name
        return "idle"


def build_timeline(items: Sequence[Item]) -> Timeline:
    """Lay the items out one after another: reach, grab, carry, let go, withdraw, pause."""
    pos = dict(START)
    obj_keys: dict[str, list[tuple[float, Point]]] = {o: [(0.0, START[o])] for o in OBJECTS}
    hand_keys: list[tuple[float, Point]] = [(0.0, REST)]
    blackouts: list[tuple[float, float]] = []
    labels: list[tuple[float, float, str]] = []
    t = LEAD_S
    last_arrive = t
    for item in items:
        if isinstance(item, Wait):
            t += item.s
        elif isinstance(item, Blackout):
            t0 = last_arrive + item.delay_s
            blackouts.append((t0, t0 + item.dur_s))
            t = max(t, t0 + item.dur_s + GAP_S)
        else:
            if item.obj not in pos:
                raise ValueError(f"unknown object {item.obj!r}; choose from {OBJECTS}")
            src, dst = pos[item.obj], item.dest or DEST[item.obj]
            t_reach = t + REACH_S
            t_grab = t_reach + GRAB_S
            t_arrive = t_grab + item.carry_s
            t_home = t_arrive + WITHDRAW_S
            hand_keys += [(t, REST), (t_reach, src), (t_grab, src), (t_arrive, dst), (t_home, REST)]
            obj_keys[item.obj] += [(t_grab, src), (t_arrive, dst)]
            first, second = ACT_LABELS[item.obj]
            t_mid = t_grab + item.carry_s / 2
            labels += [(t, t_mid, first), (t_mid, t_home, second)]
            pos[item.obj] = dst
            last_arrive = t_arrive
            t = t_home + GAP_S
    return Timeline(
        {o: tuple(k) for o, k in obj_keys.items()},
        tuple(hand_keys),
        tuple(blackouts),
        tuple(labels),
        t + TAIL_S,
    )


# -- rendering ---------------------------------------------------------------------------------

_MARKERS: dict[int, np.ndarray] = {}


def _marker(marker_id: int, px: int) -> np.ndarray:
    import cv2

    if marker_id not in _MARKERS:
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        _MARKERS[marker_id] = cv2.aruco.generateImageMarker(d, marker_id, px)
    return _MARKERS[marker_id]


def _px(x: float, y: float) -> tuple[int, int]:
    px, py = rack_to_px(x, y)
    return round(px), round(py)


def render_frame(
    zones: Mapping[str, Sequence[Point]],
    objs: Mapping[str, Point],
    hand: Point,
    blocked: bool,
    caption: str,
) -> np.ndarray:
    """One BGR frame: the board with its four ArUco markers, the zones, the objects and hand
    (or an occluder), a SIMULATED watermark. Same camera map as the P0 replay."""
    import cv2

    img = np.full((IMG_H, IMG_W, 3), (38, 34, 30), np.uint8)
    cv2.rectangle(img, _px(0.0, 0.0), _px(0.9, 0.8), (205, 205, 200), -1)  # the rack board
    size = round(MARKER_M * PX_PER_M)
    for mid, (mx, my) in RACK_MARKERS.items():
        x0, y0 = _px(mx - MARKER_M / 2, my - MARKER_M / 2)
        img[y0 : y0 + size, x0 : x0 + size] = _marker(mid, size)[:, :, None]
    for name, pts in zones.items():
        (x1, y1), (x2, y2) = np.min(pts, axis=0), np.max(pts, axis=0)
        cv2.rectangle(img, _px(x1, y1), _px(x2, y2), (60, 140, 60), 2)
        cv2.putText(img, f"Zone {name}", _px(x1 + 0.01, y1 + 0.04), 0, 0.5, (60, 140, 60), 1)
    if blocked:
        cv2.rectangle(img, _px(0.10, 0.10), _px(0.80, 0.70), (20, 20, 20), -1)
        cv2.putText(img, "VIEW BLOCKED (simulated)", _px(0.18, 0.40), 0, 0.7, (90, 90, 230), 2)
    else:
        for o in OBJECTS:
            (cx, cy), (w, h) = objs[o], SIZE_M[o]
            top_left, bottom_right = _px(cx - w / 2, cy - h / 2), _px(cx + w / 2, cy + h / 2)
            cv2.rectangle(img, top_left, bottom_right, COLOUR[o], -1)
            cv2.putText(img, o, _px(cx - w / 2, cy - h / 2 - 0.01), 0, 0.45, (20, 20, 20), 1)
        cv2.circle(img, _px(*hand), 12, (150, 180, 220), -1)  # a "hand"
    cv2.putText(img, "SIMULATED", (IMG_W - 170, 30), 0, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    # Below the board (as in the P0 replay): at IMG_H - 70 the text ran over ArUco marker 3,
    # which the rack solve needs.
    cv2.putText(img, caption, (8, IMG_H - 10), 0, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# -- the source and the hand stage -------------------------------------------------------------


class SceneSource:
    """A ``FrameSource`` for the pipeline's Runner: rendered frames + scripted detections.

    Detections carry pixel boxes only (no rack positions), so the spatial lane has to project them
    through the rack it solves from the rendered markers, as it does for a real detector.
    """

    kind = "scene"
    live = False
    publishes_frames = False
    simulated = True

    def __init__(
        self,
        scenario: str | Scenario,
        *,
        fps: float = FPS,
        seed: int = 0,
        protocol: Protocol | None = None,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self.scenario = get_scenario(scenario)
        self.timeline = build_timeline(self.scenario.items)
        self.fps = self.fps_hint = float(fps)
        self.seed = int(seed)
        self.protocol = protocol or load_protocol(load_config(CONFIG_PATH).protocol.path)
        self._hands: HandSet | None = None

    def __len__(self) -> int:
        return int(self.timeline.duration_s * self.fps) + 1

    @property
    def dropped(self) -> int:
        return 0

    def close(self) -> None:
        pass

    def hands_for(self, frame_idx: int) -> HandSet:
        if self._hands is None or self._hands.frame_idx != frame_idx:
            raise LookupError(f"no simulated hands for frame {frame_idx}")
        return self._hands

    def __iter__(self) -> Iterator[SourceFrame]:
        rng = np.random.default_rng(self.seed)
        tl = self.timeline
        for fi in range(len(self)):
            t = fi / self.fps
            blocked = tl.blocked(t)
            objs, hand = tl.objects_at(t), tl.hand_at(t)
            label = tl.label_at(t)
            dets = self._detections(fi, t, objs, blocked, rng)
            self._hands = self._hand_set(fi, t, hand, blocked, rng)
            caption = f"{self.protocol.id} / {self.scenario.name}  f{fi} t={t:.2f}s truth={label}"
            image = render_frame(self.protocol.zones, objs, hand, blocked, caption)
            yield SourceFrame(FramePacket(fi, t, image, SOURCE_ID), detections=dets, label=label)

    def _detections(self, fi, t, objs, blocked, rng) -> DetectionSet:
        out = []
        for o in OBJECTS:  # always draw, so the noise stream doesn't depend on the blackout
            dx, dy = rng.normal(0.0, POS_NOISE_M, 2)
            conf = float(np.clip(CONF_MEAN + rng.normal(0.0, CONF_STD), 0.3, 0.99))
            if blocked:
                continue
            (cx, cy), (w, h) = objs[o], SIZE_M[o]
            x1, y1 = rack_to_px(cx + dx - w / 2, cy + dy - h / 2)
            x2, y2 = rack_to_px(cx + dx + w / 2, cy + dy + h / 2)
            box = (max(0.0, x1), max(0.0, y1), min(IMG_W - 1.0, x2), min(IMG_H - 1.0, y2))
            out.append(Detection(TRACK_ID[o], o, conf, box))
        return DetectionSet(fi, t, out, "simulated")

    def _hand_set(self, fi, t, hand, blocked, rng) -> HandSet:
        dx, dy = rng.normal(0.0, POS_NOISE_M, 2)
        if blocked:  # the hand model ran and saw nothing
            return HandSet(fi, t, [], 0)
        centre = np.array([hand[0] + dx, hand[1] + dy])
        pts = np.array([rack_to_px(*(centre + shape)) for shape in _HAND_SHAPE_M])
        return HandSet(fi, t, [HandObs("right", pts, None, 0.95)], 0)


class SceneHands:
    """The pipeline's ``handpose`` stage for a ``SceneSource``: returns the scripted hands."""

    backend_name = "simulated-hands"

    def __init__(self, source: SceneSource) -> None:
        self.source = source

    def process(self, packet: FramePacket) -> tuple[HandSet, None]:
        return self.source.hands_for(packet.frame_idx), None


# -- running ------------------------------------------------------------------------------------

_SCENARIO_ANSWER: Any = object()  # sentinel: use the scenario's own scripted answer


@dataclass
class DemoResult:
    scenario: str
    run_id: str
    run_dir: Path
    summary: RunSummary
    events_path: Path
    events_sha256: str
    downlink_bytes: int
    chain_ok: bool
    protocol: Protocol
    truth: list[tuple[float, float, str]] = field(default_factory=list)

    @property
    def statuses(self) -> dict[str, str]:
        return {r["step_id"]: r["status"] for r in self.summary.outcomes}


def check_scene_protocol(protocol: Protocol) -> None:
    """The scene is choreographed for ``sample_mixing_v1``: its objects, and destinations that
    lie inside the zones the protocol's steps name. Refuse any other protocol, and any edit to
    the YAML's zones that the scene's ``DEST`` no longer matches, instead of running a demo whose
    scenarios silently stop meaning what their summaries say."""
    missing = [o for o in OBJECTS if o not in protocol.object_names]
    if missing:
        raise ValueError(
            f"protocol {protocol.id!r} has no object(s) {missing}: the SIMULATED scene only knows "
            f"{list(OBJECTS)}. Run it with --config {CONFIG_PATH}."
        )
    zones = zones_from_protocol(protocol)
    for step in protocol.steps:
        for pred, value in step.post.items():
            obj, _, name = pred.partition(".")
            m = ZONE_PRED_RE.match(name)
            if not (value and m and obj in DEST):
                continue
            zone = zones.get(m["zone"])
            if zone is None or not zone.contains(DEST[obj]):
                raise ValueError(
                    f"{step.id} needs {pred}, but the scene carries {obj} to {DEST[obj]}, outside "
                    f"zone {m['zone']} of {protocol.id!r}: update DEST in tools/demo_protocol2.py "
                    "to follow the YAML"
                )


def run_scenario(
    scenario: str | Scenario,
    cfg: Config | None = None,
    *,
    run_id: str | None = None,
    seed: int = 0,
    fps: float = FPS,
    answer: Any = _SCENARIO_ANSWER,
    answer_delay_s: float = 1.0,
    video: bool = True,
    force: bool = False,
) -> DemoResult:
    """Run one scenario through the real pipeline core. Deterministic for a given seed."""
    sc = get_scenario(scenario)
    cfg = cfg if cfg is not None else load_config(CONFIG_PATH)
    protocol = load_protocol(cfg.protocol.path)
    check_scene_protocol(protocol)
    source = SceneSource(sc, fps=fps, seed=seed, protocol=protocol)
    core = PipelineCore.from_config(
        cfg,
        protocol=protocol,
        detector=None,  # the scene supplies the detections
        handpose=SceneHands(source),
        models=None,  # HOI / temporal weights belong to the box protocol
        simulated=True,
    )
    core.overlay = bool(video)
    if answer is _SCENARIO_ANSWER:
        answer = sc.answer

    run_id = str(run_id or f"proto2-{sc.name}-{new_run_id()}")
    run_dir = run_dir_for(cfg, run_id)
    events_name = cfg.get("log.events_file", "events.jsonl")
    if run_dir.exists() and any(run_dir.iterdir()):
        if not force:
            raise FileExistsError(f"{run_dir} is not empty (pass force=True / --force)")
        for name in (events_name, "downlink.json", "run.json"):
            (run_dir / name).unlink(missing_ok=True)
        shutil.rmtree(run_dir / "video", ignore_errors=True)

    clock = PipelineClock(0.0)
    log = EventLog(run_dir, events_file=events_name, catch_signals=False, clock=clock)
    writer = None
    if video:
        segment_s = float(cfg.get("record.segment_s", 60))
        writer = SegmentWriter(run_dir / "video", "annotated", fps=fps, segment_s=segment_s)
    runner = Runner(
        core,
        source,
        eventlog=log,
        answer=answer,
        answer_delay_s=answer_delay_s,
        clock=clock,
        annotated_writer=writer,
    )
    try:
        summary = runner.run()
    except BaseException:
        log.close(reason="error")  # a partial log still ends in run_end and verifies
        raise
    finally:
        if writer is not None:
            writer.close()
    log.close(reason=summary.reason)
    size = write_downlink_for(
        run_dir,
        run_id=run_id,
        protocol=protocol,
        summary=summary,
        eventlog=log,
        started_at=iso_at(0.0),
        ended_at=iso_at(clock()),
        max_bytes=int(cfg.get("log.downlink_max_bytes", 4096)),
    )
    sha = hashlib.sha256(log.path.read_bytes()).hexdigest()
    truth = list(source.timeline.labels)
    write_json(
        run_dir / "run.json",
        {
            "run_id": run_id,
            "mode": "demo_protocol2",
            "simulated": True,
            "note": "SIMULATED scene: scripted detections and hands, real pipeline core",
            "scenario": sc.name,
            "protocol": protocol.id,
            "seed": seed,
            "fps": fps,
            "log_clock": "1970-01-01T00:00:00Z + synthetic scene t",
            "stages": core.status(),
            "answer_policy": answer if isinstance(answer, str) else None,
            "frames": summary.frames,
            "reason": summary.reason,
            "outcomes": summary.outcomes,
            "events_sha256": sha,
            "log_hash": log.final_hash,
            "downlink_bytes": size,
            "truth": truth,
            "blackouts": list(source.timeline.blackouts),
            "video": [s.path.name for s in writer.segments] if writer is not None else [],
        },
    )
    return DemoResult(
        sc.name,
        run_id,
        run_dir,
        summary,
        log.path,
        sha,
        size,
        verify_log(log.path).ok,
        protocol,
        truth,
    )


def format_outputs(outputs: Sequence[Any]) -> list[str]:
    """One ASCII line per engine output: frame, time, kind, timing, message."""
    lines = []
    for o in outputs:
        text = o.message if isinstance(o, Alert) else getattr(o, "text", "")
        extra = ""
        if isinstance(o, StepEvent) and o.t_end is not None:
            t0 = f"{o.t_start:.2f}" if o.t_start is not None else "?"
            extra = f" [{t0}-{o.t_end:.2f}s] conf={o.confidence:.2f}"
        line = f"f{o.frame_idx:5d} t={o.t:6.2f}  {describe(o):30s}{extra}  {text}"
        lines.append(line.encode("ascii", "replace").decode("ascii"))
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scenario", nargs="?", help="scenario name, or 'all' (see --list)")
    ap.add_argument("--list", action="store_true", help="list the scenarios and exit")
    ap.add_argument("--config", default=CONFIG_PATH, help=f"config overlay (default {CONFIG_PATH})")
    ap.add_argument("--override", action="append", default=[], help="key.path=value")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--answer", choices=["yes", "no"], help="scripted answer to every Ask")
    ap.add_argument("--answer-delay", type=float, default=1.0, help="seconds after the Ask")
    ap.add_argument("--run-id", help="single scenario only")
    ap.add_argument("--force", action="store_true", help="overwrite an existing run dir")
    ap.add_argument("--no-video", action="store_true", help="skip the annotated MP4")
    ap.add_argument(
        "--verify-determinism",
        action="store_true",
        help="run each scenario twice and compare events.jsonl byte for byte",
    )
    args = ap.parse_args(argv)

    if args.list or not args.scenario:
        for s in SCENARIOS.values():
            print(f"{s.name:14s} {s.summary}")
        return 0 if args.list else 2
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    if args.run_id and len(names) > 1:
        ap.error("--run-id needs a single scenario")
    cfg = load_config(args.config, overrides=args.override)
    status = 0
    for name in names:
        sc = get_scenario(name)
        answer = args.answer if args.answer else _SCENARIO_ANSWER
        res = run_scenario(
            sc,
            cfg,
            run_id=args.run_id,
            seed=args.seed,
            fps=args.fps,
            answer=answer,
            answer_delay_s=args.answer_delay,
            video=not args.no_video,
            force=args.force,
        )
        print(f"# {res.protocol.id} / {sc.name} [SIMULATED]: {sc.summary}")
        print("\n".join(format_outputs(res.summary.outputs)))
        print(f"# outcomes: {outcome_line(res.summary.outcomes)}")
        print(f"# log: {res.events_path} ({'chain OK' if res.chain_ok else 'CHAIN BROKEN'})")
        print(f"# downlink: {res.downlink_bytes} bytes; events sha256 {res.events_sha256[:16]}")
        if not res.chain_ok:
            status = 1
        if args.verify_determinism:
            again = run_scenario(
                sc,
                cfg,
                run_id=res.run_id + "-rerun",
                seed=args.seed,
                fps=args.fps,
                answer=answer,
                answer_delay_s=args.answer_delay,
                video=False,
                force=True,
            )
            same = again.events_sha256 == res.events_sha256
            print(f"# rerun {again.run_id}: {'IDENTICAL' if same else 'DIFFERENT'} events.jsonl")
            if not same:
                status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
