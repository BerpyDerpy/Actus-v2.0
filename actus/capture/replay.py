"""Synthetic replay source: scripted scenarios → noisy sensor streams, no camera needed.

Every downstream lane tests against this. A scenario is a script of actions (protocol steps,
idle, or non-step "other" actions) that change a true world state over time, plus modifiers
(occlusion, an unsure predicate). From the true state the simulator produces, per frame:

* ``Predicates``     noisy, lagged probabilities for every world-state predicate;
* ``StepEvidence``   one per protocol step. It is prior-free: it rises only after the step's
                     transition (effective pre → post) actually happened in the script. Occlusion
                     pulls it toward 0.5;
* ``ActionScores``   a noisy stand-in for the temporal model;
* ``DetectionSet``   object boxes with rack-plane positions, backend ``"replay"``;
* ``FramePacket``    optional rendered frame, watermarked SIMULATED (with real ArUco markers).

Everything is deterministic for a given (scenario, fps, seed, noise). ``t`` is synthetic seconds,
``t0 + frame_idx / fps``, with the replay's own declared fps.

    python -m actus.capture.replay --scenario skip_S2
    python -m actus.capture.replay --scenario occluded --format jsonl --stream evidence
    python -m actus.capture.replay --list
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from actus.core.bus import Bus
from actus.core.contracts import (
    IDLE,
    OTHER,
    ActionScores,
    Detection,
    DetectionSet,
    FramePacket,
    Predicates,
    StepEvidence,
    Topic,
    to_jsonable,
)
from actus.protocol.schema import Protocol, load_protocol

DEFAULT_PROTOCOL = "protocols/bas_boxes.yaml"

# --------------------------------------------------------------------------------------------
# Scenario model
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Action:
    """``label`` is a step id, ``"idle"`` or ``"other"``. ``changes`` are (fraction of the
    action's duration, predicate updates) applied to the true world state."""

    label: str
    duration_s: float
    changes: tuple[tuple[float, Mapping[str, bool]], ...] = ()
    name: str = ""  # descriptive name for non-step actions, e.g. "return_red"


@dataclass(frozen=True)
class Modifier:
    kind: Literal["occlude", "unsure"]
    t0: float
    t1: float
    level: float  # visibility for "occlude"; forced probability for "unsure"
    predicate: str | None = None  # "unsure" only


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    actions: tuple[Action, ...]
    modifiers: tuple[Modifier, ...] = ()
    # What actually happened, for tests and docs. The engine lane owns the exact event lists.
    truth: Mapping[str, list[str]] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return sum(a.duration_s for a in self.actions)

    def spans(self) -> list[tuple[Action, float, float]]:
        out, t = [], 0.0
        for a in self.actions:
            out.append((a, t, t + a.duration_s))
            t += a.duration_s
        return out

    def span_of(self, label: str, occurrence: int = 0) -> tuple[float, float]:
        hits = [(t0, t1) for a, t0, t1 in self.spans() if a.label == label or a.name == label]
        return hits[occurrence]


# ---- BAS box-transfer action library ---------------------------------------------------------

STEP_S = 2.5  # default step duration, seconds

BAS_INITIAL: dict[str, bool] = {
    "outer.open": False,
    "red.in_outer": True,
    "red.held": False,
    "red.in_zone_A": False,
    "red.in_zone_B": False,
    "yellow.in_outer": True,
    "yellow.held": False,
    "yellow.in_zone_A": False,
    "yellow.in_zone_B": False,
}

_BAS_STEP_CHANGES: dict[str, tuple[tuple[float, dict[str, bool]], ...]] = {
    "S1": ((0.7, {"outer.open": True}),),
    "S2": ((0.25, {"red.held": True}), (0.6, {"red.in_outer": False})),
    "S3": ((0.8, {"red.in_zone_A": True, "red.held": False}),),
    "S4": ((0.25, {"yellow.held": True}), (0.6, {"yellow.in_outer": False})),
    "S5": ((0.8, {"yellow.in_zone_B": True, "yellow.held": False}),),
    "S6": ((0.7, {"outer.open": False}),),
}


def step(step_id: str, d: float = STEP_S) -> Action:
    return Action(step_id, d, _BAS_STEP_CHANGES[step_id])


def idle(d: float) -> Action:
    return Action(IDLE, d)


def return_red(d: float = STEP_S) -> Action:
    """Non-step action: the red box is picked up from Zone A and put back into the outer box."""
    return Action(
        OTHER,
        d,
        (
            (0.2, {"red.held": True, "red.in_zone_A": False}),
            (0.8, {"red.in_outer": True, "red.held": False}),
        ),
        name="return_red",
    )


def _correct(*, gap_s: float = 1.0, long_gap_before: str | None = None, long_gap_s: float = 0.0):
    acts = [idle(2.0)]
    order = ["S1", "S2", "S3", "S4", "S5", "S6"]
    for i, s in enumerate(order):
        if long_gap_before == s:
            acts.append(idle(long_gap_s))
        elif i > 0 and s in ("S2", "S4", "S6"):
            acts.append(idle(gap_s))
        acts.append(step(s))
    acts.append(idle(2.0))
    return tuple(acts)


def _build_scenarios() -> dict[str, Scenario]:
    sc: dict[str, Scenario] = {}

    sc["correct"] = Scenario(
        "correct",
        "All six steps in order.",
        _correct(),
        truth={"performed": ["S1", "S2", "S3", "S4", "S5", "S6"]},
    )

    sc["skip_S2"] = Scenario(
        "skip_S2",
        "The red box is never taken out (S2, and so S3, skipped); yellow steps and close follow.",
        (
            idle(2.0),
            step("S1"),
            idle(1.0),
            step("S4"),
            step("S5"),
            idle(1.0),
            step("S6"),
            idle(2.0),
        ),
        truth={"performed": ["S1", "S4", "S5", "S6"], "skipped": ["S2", "S3"]},
    )

    sc["S4_before_S2"] = Scenario(
        "S4_before_S2",
        "Yellow box handled before the red box (S4, S5 before S2, S3), then close.",
        (
            idle(2.0),
            step("S1"),
            idle(1.0),
            step("S4"),
            step("S5"),
            idle(1.0),
            step("S2"),
            step("S3"),
            idle(1.0),
            step("S6"),
            idle(2.0),
        ),
        truth={"performed": ["S1", "S4", "S5", "S2", "S3", "S6"], "out_of_order": ["S4", "S5"]},
    )

    base = Scenario("_", "", _correct())
    s3_t0, s3_t1 = base.span_of("S3")
    s3_place = s3_t0 + 0.8 * (s3_t1 - s3_t0)
    sc["unsure_S3"] = Scenario(
        "unsure_S3",
        "Correct order, but the red box ends up on the Zone A boundary: red.in_zone_A hovers "
        "around 0.62 from placement to the end (inside the ask band).",
        base.actions,
        (Modifier("unsure", s3_place, math.inf, 0.62, "red.in_zone_A"),),
        truth={"performed": ["S1", "S2", "S3", "S4", "S5", "S6"], "unsure": ["S3"]},
    )

    sc["occluded"] = Scenario(
        "occluded",
        "Correct order, but the view is blocked from mid-S3 until 2 s after it (visibility 0.15).",
        base.actions,
        (Modifier("occlude", s3_t0 + 0.3 * (s3_t1 - s3_t0), s3_t1 + 2.0, 0.15),),
        truth={"performed": ["S1", "S2", "S3", "S4", "S5", "S6"], "occluded": ["S3"]},
    )

    sc["unexpected"] = Scenario(
        "unexpected",
        "After S3 the red box is put back into the outer box (a transition not in the protocol), "
        "then S2 and S3 are redone and the run finishes normally.",
        (
            idle(2.0),
            step("S1"),
            idle(1.0),
            step("S2"),
            step("S3"),
            idle(1.0),
            return_red(),
            idle(1.0),
            step("S2"),
            step("S3"),
            idle(1.0),
            step("S4"),
            step("S5"),
            idle(1.0),
            step("S6"),
            idle(2.0),
        ),
        truth={
            "performed": ["S1", "S2", "S3", "S2", "S3", "S4", "S5", "S6"],
            "unexpected": ["return_red"],
        },
    )

    sc["timeout"] = Scenario(
        "timeout",
        "Correct order with a 65 s pause before S4 (protocol timeout is 60 s).",
        _correct(long_gap_before="S4", long_gap_s=65.0),
        truth={"performed": ["S1", "S2", "S3", "S4", "S5", "S6"], "timeout": ["S4"]},
    )
    return sc


SCENARIOS: dict[str, Scenario] = _build_scenarios()
REQUIRED_SCENARIOS = ("correct", "skip_S2", "S4_before_S2", "unsure_S3", "occluded")


def get_scenario(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError:
        raise KeyError(f"unknown scenario {name!r}; choose from {sorted(SCENARIOS)}") from None


# --------------------------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------------------------

P_TRUE, P_FALSE = 0.93, 0.05
VIS_NORMAL = 0.95
TAU_PRED_S = 0.15  # detector lag
TAU_VIS_S = 0.2
TAU_POS_S = 0.4  # object motion smoothing

# Rack-plane layout (metres; matches configs/rack.yaml placeholders and bas_boxes zones).
OUTER_XY = (0.275, 0.425)
IN_OUTER_XY = {"red": (0.24, 0.42), "yellow": (0.31, 0.43)}
HOME_ZONE = {"red": "A", "yellow": "B"}
ELSEWHERE_XY = (0.45, 0.72)
OBJECTS = ("red", "yellow")

# Replay camera: fixed orthographic map rack metres → 640×480 pixels.
IMG_W, IMG_H = 640, 480
PX_PER_M, PX_OFF = 520.0, (60.0, 25.0)


def rack_to_px(x: float, y: float) -> tuple[float, float]:
    return PX_OFF[0] + PX_PER_M * x, PX_OFF[1] + PX_PER_M * y


@dataclass(frozen=True)
class ReplayTick:
    frame_idx: int
    t: float
    label: str  # ground-truth action label (step id, "idle", "other")
    truth_state: Mapping[str, bool]
    predicates: Predicates
    evidence: tuple[StepEvidence, ...]
    actions: ActionScores
    detections: DetectionSet
    frame: FramePacket | None


def _zone_center(zones: Mapping[str, list], name: str) -> tuple[float, float]:
    pts = np.asarray(zones[name], dtype=float)
    return tuple(pts.mean(axis=0))  # type: ignore[return-value]


class ReplaySource:
    """Deterministic synthetic stream for one scenario. Iterate it, or ``run(bus)``."""

    def __init__(
        self,
        scenario: str | Scenario = "correct",
        *,
        protocol: Protocol | str | Path | None = None,
        fps: float = 15.0,
        seed: int = 0,
        noise: float = 0.04,
        render: bool = False,
        t0: float = 0.0,
        source_id: str = "replay",
    ) -> None:
        self.scenario = get_scenario(scenario) if isinstance(scenario, str) else scenario
        if protocol is None:
            protocol = DEFAULT_PROTOCOL
        self.protocol = protocol if isinstance(protocol, Protocol) else load_protocol(protocol)
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self.fps = float(fps)
        self.seed = seed
        self.noise = float(noise)
        self.render = render
        self.t0 = float(t0)
        self.source_id = source_id
        self.n_frames = round(self.scenario.duration_s * self.fps)
        self.predicate_names = sorted(set(BAS_INITIAL) | set(self.protocol.predicates))
        missing = set(self.protocol.predicates) - set(BAS_INITIAL)
        if missing:
            raise ValueError(
                f"replay scenarios simulate the BAS box world; protocol {self.protocol.id} "
                f"uses predicates it cannot simulate: {sorted(missing)}"
            )

    # -- ground truth -------------------------------------------------------------------------

    def _events(self) -> list[tuple[int, dict[str, bool]]]:
        """(frame at which it applies, predicate updates), sorted by frame."""
        out = []
        for a, s0, s1 in self.scenario.spans():
            for frac, upd in a.changes:
                out.append((round((s0 + frac * (s1 - s0)) * self.fps), dict(upd)))
        return sorted(out, key=lambda e: e[0])

    def labels(self) -> list[str]:
        """Ground-truth action label for every frame."""
        out = []
        for a, s0, s1 in self.scenario.spans():
            f0, f1 = round(s0 * self.fps), round(s1 * self.fps)
            out.extend([a.label] * (f1 - f0))
        return out[: self.n_frames] + [IDLE] * max(0, self.n_frames - len(out))

    def segments(self) -> list[tuple[str, int, int]]:
        """Ground-truth (label, first_frame, end_frame_exclusive) runs."""
        segs: list[tuple[str, int, int]] = []
        for i, lab in enumerate(self.labels()):
            if segs and segs[-1][0] == lab and segs[-1][2] == i:
                segs[-1] = (lab, segs[-1][1], i + 1)
            else:
                segs.append((lab, i, i + 1))
        return segs

    def annotation_rows(self) -> list[tuple[int, float, str]]:
        """Label changes in the T-REC sidecar CSV convention: (frame_idx, t, label)."""
        return [(f0, self.t0 + f0 / self.fps, lab) for lab, f0, _ in self.segments()]

    def write_annotations(self, path: str | Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx", "t", "label"])
            for fi, t, lab in self.annotation_rows():
                w.writerow([fi, f"{t:.4f}", lab])

    # -- stream -------------------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_frames

    def __iter__(self) -> Iterator[ReplayTick]:
        rng = np.random.default_rng(self.seed)
        dt = 1.0 / self.fps
        a_pred = 1.0 - math.exp(-dt / TAU_PRED_S)
        a_vis = 1.0 - math.exp(-dt / TAU_VIS_S)
        a_pos = 1.0 - math.exp(-dt / TAU_POS_S)
        sigma = self.noise

        state = dict(BAS_INITIAL)
        state.update({k: v for k, v in self.protocol.initial.items() if k in state})
        base = {k: (P_TRUE if v else P_FALSE) for k, v in state.items()}
        ar = dict.fromkeys(state, 0.0)
        vis = VIS_NORMAL
        events = self._events()
        ev_i = 0
        labels = self.labels()

        steps = self.protocol.steps
        armed = {s.id: _holds(state, s.effective_pre) and not _holds(state, s.post) for s in steps}
        achieved = dict.fromkeys(armed, False)

        pos = {o: IN_OUTER_XY[o] for o in OBJECTS}
        zones = self.protocol.zones
        step_ids = self.protocol.step_ids

        for fi in range(self.n_frames):
            t_rel = fi * dt
            t = self.t0 + t_rel
            while ev_i < len(events) and events[ev_i][0] <= fi:
                state.update(events[ev_i][1])
                ev_i += 1

            # transition bookkeeping on the TRUE state (prior-free: order is never consulted)
            for s in steps:
                post_ok = _holds(state, s.post)
                if post_ok and armed[s.id]:
                    achieved[s.id], armed[s.id] = True, False
                elif not post_ok:
                    achieved[s.id] = False
                    armed[s.id] = armed[s.id] or _holds(state, s.effective_pre)

            # visibility (occlusion modifiers)
            vis_target = VIS_NORMAL
            for m in self.scenario.modifiers:
                if m.kind == "occlude" and m.t0 <= t_rel < m.t1:
                    vis_target = min(vis_target, m.level)
            vis += a_vis * (vis_target - vis)
            vis_obs = float(np.clip(vis + rng.normal(0, sigma / 2), 0.0, 1.0))

            # predicates: lagged base + AR(1) noise, unsure overrides, occlusion regression
            clean: dict[str, float] = {}
            obs: dict[str, float] = {}
            for k in self.predicate_names:
                target = P_TRUE if state[k] else P_FALSE
                forced = None
                for m in self.scenario.modifiers:
                    if m.kind == "unsure" and m.predicate == k and m.t0 <= t_rel < m.t1:
                        forced = m.level
                base[k] += a_pred * ((forced if forced is not None else target) - base[k])
                s_k = sigma * (1.5 if k.endswith(".held") else 1.0)
                ar[k] = 0.8 * ar[k] + rng.normal(0, s_k * 0.6)
                p = float(np.clip(base[k] + ar[k] + rng.normal(0, s_k * 0.4), 0.001, 0.999))
                clean[k] = p
                obs[k] = vis * p + (1 - vis) * 0.5
            predicates = Predicates(fi, t, obs, vis_obs)

            # step evidence
            evidence = []
            for s in steps:
                if achieved[s.id]:
                    e = min(clean[k] if v else 1.0 - clean[k] for k, v in s.post.items())
                else:
                    e = float(np.clip(P_FALSE + abs(rng.normal(0, sigma)), 0.001, 0.999))
                p = float(np.clip(vis * e + (1 - vis) * 0.5, 0.0, 1.0))
                evidence.append(StepEvidence(fi, t, s.id, p, {"replay": p}))

            # action scores
            label = labels[fi]
            actions = ActionScores(fi, t, _action_scores(label, step_ids, rng, sigma))

            # object positions (rack plane) and detections
            for o in OBJECTS:
                target = _object_target(o, state, zones)
                pos[o] = (
                    pos[o][0] + a_pos * (target[0] - pos[o][0]),
                    pos[o][1] + a_pos * (target[1] - pos[o][1]),
                )
            dets = [] if vis < 0.5 else _detections(state, pos, rng, sigma)
            detections = DetectionSet(fi, t, dets, "replay")

            frame = None
            if self.render:
                img = _render(self, fi, t, label, state, pos, vis, dets)
                frame = FramePacket(fi, t, img, self.source_id)

            yield ReplayTick(
                fi, t, label, dict(state), predicates, tuple(evidence), actions, detections, frame
            )

    def run(
        self,
        bus: Bus,
        *,
        realtime: bool = True,
        stop: threading.Event | None = None,
        close_bus: bool = False,
    ) -> int:
        """Publish every tick to the bus (frames, detections, predicates, evidence, actions).
        Returns the number of frames published."""
        start = time.monotonic()
        n = 0
        for tick in self:
            if stop is not None and stop.is_set():
                break
            if realtime:
                delay = start + tick.frame_idx / self.fps - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            if tick.frame is not None:
                bus.publish(Topic.FRAMES, tick.frame)
            bus.publish(Topic.DETECTIONS, tick.detections)
            bus.publish(Topic.PREDICATES, tick.predicates)
            for ev in tick.evidence:
                bus.publish(Topic.EVIDENCE, ev)
            bus.publish(Topic.ACTIONS, tick.actions)
            n += 1
        if close_bus:
            bus.close()
        return n


def _holds(state: Mapping[str, bool], conds: Mapping[str, bool]) -> bool:
    return all(state.get(k) == v for k, v in conds.items())


def _action_scores(label: str, step_ids: list[str], rng: np.random.Generator, sigma: float):
    keys = [*step_ids, IDLE]
    raw = {k: float(rng.uniform(0.01, 0.06)) for k in keys}
    if label == OTHER:
        raw[IDLE] = 0.35 + float(rng.normal(0, sigma))
        for k in rng.choice(step_ids, size=min(2, len(step_ids)), replace=False):
            raw[str(k)] += 0.2
    else:
        raw[label if label in raw else IDLE] = 0.75 + float(rng.normal(0, sigma))
    raw = {k: max(v, 1e-3) for k, v in raw.items()}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def _object_target(o: str, state: Mapping[str, bool], zones: Mapping[str, list]):
    if state[f"{o}.held"]:
        # carried: halfway between the outer box and its home zone, where the hand would be
        zx, zy = _zone_center(zones, HOME_ZONE[o])
        return ((OUTER_XY[0] + zx) / 2, (OUTER_XY[1] + zy) / 2 - 0.08)
    if state[f"{o}.in_outer"]:
        return IN_OUTER_XY[o]
    for z in ("A", "B"):
        if state.get(f"{o}.in_zone_{z}") and z in zones:
            return _zone_center(zones, z)
    return ELSEWHERE_XY


def _box_px(cx: float, cy: float, half_m: float) -> tuple[float, float, float, float]:
    x, y = rack_to_px(cx, cy)
    h = half_m * PX_PER_M
    return (max(0.0, x - h), max(0.0, y - h), min(IMG_W - 1.0, x + h), min(IMG_H - 1.0, y + h))


def _detections(state, pos, rng: np.random.Generator, sigma: float) -> list[Detection]:
    def conf() -> float:
        return float(np.clip(0.9 + rng.normal(0, sigma), 0.3, 0.99))

    out = [
        Detection(
            1,
            "outer_box_open" if state["outer.open"] else "outer_box_closed",
            conf(),
            _box_px(*OUTER_XY, 0.125),
            OUTER_XY,
        )
    ]
    for tid, o in ((2, "red"), (3, "yellow")):
        if state[f"{o}.in_outer"] and not state["outer.open"]:
            continue  # inside a closed box: not visible
        out.append(Detection(tid, f"{o}_box", conf(), _box_px(*pos[o], 0.035), pos[o]))
    return out


# --------------------------------------------------------------------------------------------
# Rendering (optional; OpenCV)
# --------------------------------------------------------------------------------------------

_MARKER_CACHE: dict[int, np.ndarray] = {}


def _marker(marker_id: int, px: int) -> np.ndarray | None:
    import cv2

    if marker_id not in _MARKER_CACHE:
        try:
            d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            _MARKER_CACHE[marker_id] = cv2.aruco.generateImageMarker(d, marker_id, px)
        except AttributeError:  # very old OpenCV without the aruco module
            return None
    return _MARKER_CACHE[marker_id]


RACK_MARKERS = {0: (0.05, 0.05), 1: (0.85, 0.05), 2: (0.85, 0.75), 3: (0.05, 0.75)}
MARKER_M = 0.06


def _render(src: ReplaySource, fi, t, label, state, pos, vis, dets) -> np.ndarray:
    import cv2

    img = np.full((IMG_H, IMG_W, 3), (38, 34, 30), np.uint8)
    p = lambda x, y: tuple(round(v) for v in rack_to_px(x, y))  # noqa: E731
    cv2.rectangle(img, p(0, 0), p(0.9, 0.8), (205, 205, 200), -1)  # the rack board
    size = round(MARKER_M * PX_PER_M)
    for mid, (mx, my) in RACK_MARKERS.items():
        m = _marker(mid, size)
        x0, y0 = p(mx - MARKER_M / 2, my - MARKER_M / 2)
        if m is not None:
            img[y0 : y0 + size, x0 : x0 + size] = m[:, :, None]
    for name, pts in src.protocol.zones.items():
        (x1, y1), (x2, y2) = np.min(pts, axis=0), np.max(pts, axis=0)
        cv2.rectangle(img, p(x1, y1), p(x2, y2), (60, 140, 60), 2)
        cv2.putText(
            img,
            f"Zone {name}",
            p(x1 + 0.01, y1 + 0.04),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (60, 140, 60),
            1,
            cv2.LINE_AA,
        )
    ox, oy = OUTER_XY
    cv2.rectangle(img, p(ox - 0.125, oy - 0.125), p(ox + 0.125, oy + 0.125), (60, 100, 150), -1)
    if state["outer.open"]:
        cv2.rectangle(img, p(ox - 0.11, oy - 0.11), p(ox + 0.11, oy + 0.11), (40, 70, 110), -1)
        cv2.rectangle(img, p(ox - 0.125, oy - 0.30), p(ox + 0.125, oy - 0.15), (80, 125, 175), -1)
    for o, color in (("red", (40, 40, 210)), ("yellow", (40, 210, 230))):
        if state[f"{o}.in_outer"] and not state["outer.open"]:
            continue
        cx, cy = pos[o]
        cv2.rectangle(img, p(cx - 0.035, cy - 0.035), p(cx + 0.035, cy + 0.035), color, -1)
        if state[f"{o}.held"]:
            cv2.circle(img, p(cx + 0.03, cy + 0.05), 14, (150, 180, 220), -1)  # a "hand"
    if vis < 0.9:  # occluder
        alpha = float(np.clip((0.95 - vis) / 0.8, 0, 1))
        over = img.copy()
        cv2.rectangle(over, (IMG_W // 5, IMG_H // 8), (IMG_W - 10, IMG_H - 10), (20, 20, 20), -1)
        img = cv2.addWeighted(over, alpha, img, 1 - alpha, 0)
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.bbox_xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
    cv2.putText(
        img,
        "SIMULATED",
        (IMG_W - 170, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"replay:{src.scenario.name} f{fi} t={t:.2f}s truth={label}",
        (8, IMG_H - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return img


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def _abbrev(name: str) -> str:
    obj, pred = name.split(".", 1)
    pred = {"in_outer": "in", "held": "hd", "open": "open"}.get(pred, pred.replace("in_zone_", "z"))
    return f"{obj[0]}.{pred}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Print a synthetic replay stream for a scenario.")
    ap.add_argument("--scenario", default="correct")
    ap.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.04)
    ap.add_argument("--format", choices=["table", "jsonl"], default="table")
    ap.add_argument(
        "--stream",
        choices=["predicates", "evidence", "actions", "detections", "all"],
        default="predicates",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=None,
        help="print every Nth frame (default 5 for table, 1 for jsonl)",
    )
    ap.add_argument("--realtime", action="store_true", help="pace output at --fps")
    ap.add_argument("--truth-csv", type=Path, help="also write ground-truth labels (T-REC CSV)")
    ap.add_argument("--list", action="store_true", help="list scenarios and exit")
    args = ap.parse_args(argv)

    if args.list:
        for name, s in SCENARIOS.items():
            print(f"{name:14s} {s.duration_s:6.1f}s  {s.description}")
        return 0

    src = ReplaySource(
        args.scenario, protocol=args.protocol, fps=args.fps, seed=args.seed, noise=args.noise
    )
    if args.truth_csv:
        src.write_annotations(args.truth_csv)
    stride = args.stride or (5 if args.format == "table" else 1)
    out = sys.stdout
    start = time.monotonic()

    if args.format == "table":
        sc = src.scenario
        out.write(f"# scenario {sc.name}: {sc.description}\n")
        out.write(
            f"# {len(src)} frames @ {src.fps:g} fps, seed {src.seed}, noise {src.noise:g}. "
            "SIMULATED data.\n"
        )
        if args.stream in ("predicates", "all"):
            cols = src.predicate_names
            out.write("# " + "  ".join(f"{_abbrev(c)}={c}" for c in cols) + "\n")
            out.write(
                f"{'frame':>5} {'t':>6} {'vis':>4} {'truth':>6} | "
                + " ".join(f"{_abbrev(c):>6}" for c in cols)
                + "\n"
            )
        elif args.stream == "evidence":
            out.write(
                f"{'frame':>5} {'t':>6} {'truth':>6} | "
                + " ".join(f"{s:>5}" for s in src.protocol.step_ids)
                + "\n"
            )

    for tick in src:
        if args.realtime:
            delay = start + tick.frame_idx / src.fps - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        if tick.frame_idx % stride:
            continue
        if args.format == "jsonl":
            msgs = {
                "predicates": [tick.predicates],
                "evidence": list(tick.evidence),
                "actions": [tick.actions],
                "detections": [tick.detections],
            }
            chosen = (
                [m for v in msgs.values() for m in v] if args.stream == "all" else msgs[args.stream]
            )
            for m in chosen:
                out.write(json.dumps(to_jsonable(m), separators=(",", ":")) + "\n")
        elif args.stream in ("predicates", "all"):
            pr = tick.predicates
            out.write(
                f"{tick.frame_idx:5d} {tick.t:6.2f} {pr.visibility:4.2f} {tick.label:>6} | "
                + " ".join(f"{pr.values[c]:6.2f}" for c in src.predicate_names)
                + "\n"
            )
        elif args.stream == "evidence":
            out.write(
                f"{tick.frame_idx:5d} {tick.t:6.2f} {tick.label:>6} | "
                + " ".join(f"{e.p_complete:5.2f}" for e in tick.evidence)
                + "\n"
            )
        elif args.stream == "actions":
            top, p = tick.actions.top()
            out.write(
                f"{tick.frame_idx:5d} {tick.t:6.2f} truth={tick.label:>6} top={top:>6} {p:.2f}\n"
            )
        else:
            out.write(
                f"{tick.frame_idx:5d} {tick.t:6.2f} "
                + ", ".join(
                    f"{d.cls}@({d.bbox_rack_xy[0]:.2f},{d.bbox_rack_xy[1]:.2f})"
                    for d in tick.detections.detections
                    if d.bbox_rack_xy
                )
                + "\n"
            )
        out.flush()

    if args.format == "table":
        out.write(
            "# truth segments: "
            + ", ".join(f"{lab}[{a}-{b - 1}]" for lab, a, b in src.segments())
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
