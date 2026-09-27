"""Tracked object-state estimator: where is each object, as a probability, frame by frame.

Vocabulary (read from the protocol's predicates)
------------------------------------------------
* **containers**: objects that other objects can be *in* (``red.in_outer`` makes ``outer`` one)
  or that can be ``open``. A container has a binary open/closed belief and a footprint on the
  rack plane (``configs/rack.yaml: outer_box_footprint``, or ``spatial.footprints``) that
  follows its detections.
* **movables**: everything else. Each carries a categorical *location belief* over
  ``in_<container>``, ``held``, ``zone_<Z>`` (one per protocol zone) and ``elsewhere``. Its
  reported ``location`` is the most likely state, or ``unknown`` when none reaches
  ``location_min_p`` (e.g. never observed and no ``initial`` prior).

Supported predicates: ``<movable>.held``, ``<movable>.in_zone_<Z>``, ``<movable>.in_<container>``
and ``<container>.open``. Anything else in a protocol is rejected at construction.

The filter (an input-output HMM per object)
-------------------------------------------
Hands are *inputs* that gate transitions; detections, misses, motion and HOI are *emissions*.

* Transitions (rates per second, turned into probabilities with the real ``dt``):
  ``X → held`` needs a hand near the object (``pick_rate × p(hand near)``; out of a container
  also needs it open). ``held → X`` needs the object (or the hand carrying it) to be over X
  and the hand to have let go (``place_rate × membership × (1 − p(hand near))``). With the hand
  model running but no hand in view, both use the small ``hand_unseen_gate``: a blinded camera
  does not move objects. A small ``correction_rate`` connects every pair, so a wrong belief can
  still be corrected. Objects do not teleport between regions without being held.
* Emissions: a detection's rack position scores each state by soft zone/footprint membership;
  motion above ``static_speed_mps`` is only explained by ``held``; a visible hand far from a
  detected object argues against ``held``; a *missing* detection is expected for an object in
  a closed container, plausible when held or in an open container, and unlikely for an object
  resting in view — scaled by scene visibility, so a blinded camera is uninformative.
* Evidence is tempered by ``min(1, dt / obs_tau_s)``: it accumulates per second, not per frame,
  so the output does not depend on the frame rate and consecutive frames are not treated as
  independent.

Consequences the tests pin down: an object that vanishes while held stays ``held`` (it can only
leave ``held`` towards where it was last seen or carried); an object that vanishes over an open
container's footprint, once the hand withdraws, becomes ``in_<container>``.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from actus.core.contracts import Detection, DetectionSet
from actus.spatial.zones import Zone, as_polygon, zones_from_protocol

HELD = "held"
ELSEWHERE = "elsewhere"
UNKNOWN = "unknown"
IN_RE = re.compile(r"^in_(?P<c>[A-Za-z][A-Za-z0-9_]*)$")
ZONE_PRED_RE = re.compile(r"^in_zone_(?P<z>[A-Za-z0-9_]+)$")

# Hand landmarks used for proximity: wrist, the four MCP knuckles and three fingertips.
HAND_KEYPOINTS = (0, 4, 5, 8, 9, 12, 13, 17)


# --------------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SpatialParams:
    """Tuning for the world-state filter and the predicate/evidence layer.

    Set from the free-form ``spatial`` section of the config; unknown keys are rejected.
    """

    # geometry
    pos_sigma_m: float = 0.02  # position noise for soft zone membership
    hand_near_m: float = 0.08  # hand–object distance at which p(near) = 0.5
    hand_near_scale_m: float = 0.02  # logistic width of p(near)
    bbox_hand_margin: float = 0.2  # a hand landmark inside the object's bbox grown by this = near
    carry_radius_m: float = 0.3  # an unseen held object follows a hand within this distance
    default_footprint_m: float = 0.25  # square footprint for a container with none configured
    footprint_follow_tau_s: float = 0.5  # footprint re-centres on the container's detections
    footprints: Mapping[str, Any] = field(default_factory=dict)  # {container: polygon}
    # transitions (per second)
    pick_rate: float = 2.0
    place_rate: float = 1.0
    open_rate: float = 1.0
    correction_rate: float = 0.02
    hand_unknown_gate: float = 0.5  # hand gate when no hand model runs
    hand_unseen_gate: float = 0.1  # hand gate when the hand model runs but sees no hand
    # emissions
    miss_visible: float = 0.1  # P(no detection | resting in view)
    miss_held_hand_seen: float = 0.35  # P(no detection | held, a hand is visible)
    miss_held: float = 0.6  # P(no detection | held, no hand visible)
    miss_in_open: float = 0.5
    miss_in_closed: float = 0.98
    held_pos_like: float = 0.5  # position likelihood of a detection under "held" (anywhere)
    hand_far_like: float = 0.15  # L(held | hands visible, none near the detected object)
    hand_unseen_like: float = 0.6  # L(held | the hand model sees no hand)
    static_speed_mps: float = 0.05  # speed noise of a resting object
    moving_floor: float = 0.05  # min L(resting | moving)
    held_static_like: float = 0.4  # L(held | not moving): a held object is rarely still
    max_speed_mps: float = 3.0  # faster = association jump, ignored
    speed_window_s: float = 0.25
    obs_tau_s: float = 0.2  # evidence decorrelation time
    hoi_clip: float = 0.02
    vis_tau_s: float = 0.3
    initial_violation: float = 0.2  # prior weight of a state contradicting `initial`
    location_min_p: float = 0.5
    max_dt_s: float = 5.0
    # predicates / evidence (predicates.py)
    arm_memory_s: float = 5.0  # how long an observation supports arming a step
    arm_dwell_s: float = 0.25  # a pre-state must hold this long to arm a step
    report_regress: bool = True  # reported p = v·p + (1-v)·0.5 (as the replay source does)
    p_clip: tuple[float, float] = (0.01, 0.99)
    complete_hi: float = 0.8  # latch reset: post was above this ...
    complete_lo: float = 0.2  # ... and fell below this (the step was undone)
    temperatures: Mapping[str, float] = field(default_factory=dict)  # predicate or step id → T
    # body
    world_landmarks_rotated_back: bool = False

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any] | None) -> SpatialParams:
        m = dict(m or {})
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(m) - names)
        if unknown:
            raise KeyError(f"unknown spatial parameter(s): {unknown}; known: {sorted(names)}")
        if "p_clip" in m:
            m["p_clip"] = tuple(float(v) for v in m["p_clip"])
        return cls(**m)


# --------------------------------------------------------------------------------------------
# Inputs and outputs
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HandInput:
    """Hands for one frame. ``observed`` is False when no hand model ran (no information)."""

    observed: bool
    points_px: tuple[np.ndarray, ...] = ()  # per hand, (K, 2) key landmarks in image pixels
    points_rack: tuple[np.ndarray, ...] | None = None  # per hand, (K, 2) rack metres

    @property
    def n(self) -> int:
        return len(self.points_px)


NO_HANDS_INFO = HandInput(observed=False)


@dataclass(frozen=True)
class ObjectState:
    name: str
    kind: str  # "movable" | "container"
    location: str  # movable: a state name or "unknown"; container: "open" | "closed" | "unknown"
    belief: Mapping[str, float]
    pos: tuple[float, float] | None  # last detected rack position (metres)
    detected: bool  # detected this frame
    last_seen_t: float | None
    supported_t: float | None  # last time an observation supported the belief
    speed: float | None = None  # m/s, None when unknown
    p_open: float | None = None  # containers
    footprint: tuple[tuple[float, float], ...] | None = None  # containers, rack metres


@dataclass(frozen=True)
class WorldSnapshot:
    frame_idx: int
    t: float
    objects: Mapping[str, ObjectState]
    predicates: Mapping[str, float]  # raw (uncalibrated, un-regressed) beliefs
    visibility: float
    rack_state: str  # "fresh" | "held" | "invalid" | "none" (no rack frame supplied)

    def location(self, name: str) -> str:
        return self.objects[name].location


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def parse_vocabulary(protocol) -> tuple[list[str], list[str]]:
    """(containers, movables) implied by a protocol's predicates and declared objects."""
    containers: set[str] = set()
    movables: set[str] = set()
    bad: list[str] = []
    for name in protocol.predicates:
        obj, pred = name.split(".", 1)
        if pred == "open":
            containers.add(obj)
        elif pred == HELD or ZONE_PRED_RE.match(pred):
            movables.add(obj)
        elif m := IN_RE.match(pred):
            containers.add(m["c"])
            movables.add(obj)
        else:
            bad.append(name)
    if bad:
        raise ValueError(
            f"spatial cannot estimate predicate(s) {sorted(bad)}; supported forms are "
            "<obj>.held, <obj>.in_zone_<Z>, <obj>.in_<container>, <container>.open"
        )
    if protocol.objects:
        undeclared = sorted(c for c in containers if c not in protocol.objects)
        if undeclared:
            raise ValueError(f"container(s) {undeclared} are not declared in objects")
    movables |= {o for o in protocol.objects if o not in containers}
    both = sorted(containers & movables)
    if both:
        raise ValueError(f"object(s) {both} are used both as container and as movable")
    return sorted(containers), sorted(movables)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def expm(A: np.ndarray) -> np.ndarray:
    """Matrix exponential by scaling and squaring (small generator matrices)."""
    n = float(np.abs(A).sum(axis=1).max()) if A.size else 0.0
    s = max(0, math.ceil(math.log2(n)) + 1) if n > 0.5 else 0
    A = A / (2**s)
    E = np.eye(len(A))
    term = np.eye(len(A))
    for k in range(1, 12):
        term = term @ A / k
        E = E + term
    for _ in range(s):
        E = E @ E
    return E


def _is_lid(cls: str) -> bool:
    return cls == "lid" or cls.endswith("_lid")


def _norm(b: np.ndarray) -> np.ndarray:
    b = np.clip(b, 1e-9, None)
    return b / b.sum()


# --------------------------------------------------------------------------------------------
# Tracks
# --------------------------------------------------------------------------------------------


class _Container:
    def __init__(self, name: str, classes: Iterable[str], p_open: float, home: Zone | None):
        self.name = name
        self.box_classes = {c for c in classes if not _is_lid(c)}
        self.lid_classes = {c for c in classes if _is_lid(c)}
        self.p_open = p_open
        self.home = home
        self.offset = np.zeros(2)
        self.det: Detection | None = None
        self.det_xy: np.ndarray | None = None
        self.pos: np.ndarray | None = None
        self.track_id: int | None = None
        self.last_seen_t: float | None = None
        self.ever_seen = False

    def footprint(self) -> Zone | None:
        return None if self.home is None else self.home.translated(*self.offset)


class _Movable:
    def __init__(self, name: str, classes: Iterable[str], states: list[str], b: np.ndarray):
        self.name = name
        self.classes = set(classes)
        self.states = states
        self.idx = {s: i for i, s in enumerate(states)}
        self.b = b
        self.pos: np.ndarray | None = None
        self.carry: np.ndarray | None = None
        self.hist: deque[tuple[float, np.ndarray]] = deque()
        self.speed: float | None = None
        self.det: Detection | None = None
        self.det_xy: np.ndarray | None = None
        self.track_id: int | None = None
        self.last_seen_t: float | None = None
        self.supported_t: float | None = None

    def p(self, state: str) -> float:
        return float(self.b[self.idx[state]])


# --------------------------------------------------------------------------------------------
# World model
# --------------------------------------------------------------------------------------------


class WorldModel:
    """Location beliefs for every protocol object. Call ``update`` once per processed frame."""

    def __init__(
        self,
        protocol,
        params: SpatialParams | None = None,
        *,
        footprints: Mapping[str, Any] | None = None,
    ) -> None:
        self.protocol = protocol
        self.params = params or SpatialParams()
        self.zones = zones_from_protocol(protocol)
        self.container_names, self.movable_names = parse_vocabulary(protocol)
        fps = {**(footprints or {}), **dict(self.params.footprints)}
        unknown_fp = sorted(set(self.params.footprints) - set(self.container_names))
        if unknown_fp:
            raise ValueError(f"spatial.footprints names non-container(s) {unknown_fp}")
        self._footprints_cfg = {k: as_polygon(v) for k, v in fps.items()}
        self.states = (
            [f"in_{c}" for c in self.container_names]
            + [HELD]
            + [f"zone_{z}" for z in self.zones]
            + [ELSEWHERE]
        )
        self.reset()

    # -- setup ----------------------------------------------------------------------------------

    def _classes(self, obj: str) -> list[str]:
        spec = self.protocol.objects.get(obj)
        return list(spec.classes) if spec and spec.classes else [obj]

    def reset(self) -> None:
        init = dict(self.protocol.initial)
        viol = self.params.initial_violation
        self.containers: dict[str, _Container] = {}
        for c in self.container_names:
            p_open = {True: 1 - viol, False: viol}.get(init.get(f"{c}.open"), 0.5)  # type: ignore[arg-type]
            fp = self._footprints_cfg.get(c)
            self.containers[c] = _Container(
                c, self._classes(c), p_open, None if fp is None else Zone(c, fp)
            )
        self.movables: dict[str, _Movable] = {}
        for o in self.movable_names:
            # a flat, modest weight (not compounded per contradicted fact), so the first clear
            # observations overturn a wrong `initial` within a frame or two
            prior = np.ones(len(self.states))
            for i, s in enumerate(self.states):
                for pred, val in init.items():
                    obj, name = pred.split(".", 1)
                    if obj == o and self._state_satisfies(s, name) != bool(val):
                        prior[i] = viol
            self.movables[o] = _Movable(o, self._classes(o), self.states, _norm(prior))
        self.visibility = 1.0
        self._t: float | None = None
        self.frame_idx = 0

    @staticmethod
    def _state_satisfies(state: str, pred: str) -> bool:
        """Does location ``state`` make predicate ``pred`` (e.g. ``in_zone_A``) true?"""
        if pred == HELD:
            return state == HELD
        if m := ZONE_PRED_RE.match(pred):
            return state == f"zone_{m['z']}"
        if m := IN_RE.match(pred):
            return state == f"in_{m['c']}"
        return False

    @property
    def predicate_names(self) -> list[str]:
        names = [f"{c}.open" for c in self.container_names]
        for o in self.movable_names:
            names.append(f"{o}.{HELD}")
            names += [f"{o}.in_{c}" for c in self.container_names]
            names += [f"{o}.in_zone_{z}" for z in self.zones]
        return sorted(names)

    # -- geometry helpers -----------------------------------------------------------------------

    def _near_point(self, hands: HandInput, xy, bbox) -> float | None:
        """p(a hand is at the object): rack distance and/or image bbox; None if unmeasurable."""
        pr = self.params
        near: float | None = None
        if xy is not None and hands.points_rack:
            d = min(float(np.linalg.norm(p - xy, axis=1).min()) for p in hands.points_rack)
            near = _sigmoid((pr.hand_near_m - d) / pr.hand_near_scale_m)
        if bbox is not None and hands.points_px:
            x1, y1, x2, y2 = bbox
            mx, my = (x2 - x1) * pr.bbox_hand_margin, (y2 - y1) * pr.bbox_hand_margin
            inside = any(
                bool(
                    (
                        (p[:, 0] >= x1 - mx)
                        & (p[:, 0] <= x2 + mx)
                        & (p[:, 1] >= y1 - my)
                        & (p[:, 1] <= y2 + my)
                    ).any()
                )
                for p in hands.points_px
            )
            if inside:
                near = max(near or 0.0, 0.9)
            elif near is None:
                near = 0.1
        return near

    def _near_zone(self, hands: HandInput, zone: Zone | None) -> float | None:
        if zone is None or not hands.points_rack:
            return None
        pr = self.params
        d = min(
            float(np.min(np.maximum(0.0, -np.asarray(zone.signed_distance(p)))))
            for p in hands.points_rack
        )
        return _sigmoid((pr.hand_near_m - d) / pr.hand_near_scale_m)

    def _gate(self, hands: HandInput, near: float | None) -> float:
        pr = self.params
        if not hands.observed:
            return pr.hand_unknown_gate
        if hands.n == 0:
            return pr.hand_unseen_gate
        return pr.hand_unknown_gate if near is None else max(near, 0.02)

    def _release_gate(self, hands: HandInput, near: float | None) -> float:
        """Letting go is a hand action too: a hand still at the object keeps it held."""
        pr = self.params
        if not hands.observed:
            return pr.hand_unknown_gate
        if hands.n == 0:
            return pr.hand_unseen_gate
        return pr.hand_unknown_gate if near is None else max(1.0 - near, 0.02)

    def _memberships(self, refs: Sequence[np.ndarray]) -> dict[str, float]:
        """Per state, how plausible it is that the object sits in that region (max over refs)."""
        pr = self.params
        out = dict.fromkeys(self.states, 0.0)
        out[HELD] = 1.0
        if not refs:
            for s in self.states:
                out[s] = 1.0 if s in (HELD, ELSEWHERE) else 0.0
            return out
        for ref in refs:
            regions = []
            for z, zone in self.zones.items():
                m = zone.membership(ref, pr.pos_sigma_m)
                out[f"zone_{z}"] = max(out[f"zone_{z}"], m)
                regions.append(m)
            for c, ct in self.containers.items():
                fp = ct.footprint()
                m = 0.0 if fp is None else fp.membership(ref, pr.pos_sigma_m) * ct.p_open
                out[f"in_{c}"] = max(out[f"in_{c}"], m)
                regions.append(m)
            out[ELSEWHERE] = max(out[ELSEWHERE], 1.0 - max(regions, default=0.0))
        return out

    def _pos_likelihood(self, xy: np.ndarray | None) -> np.ndarray:
        pr = self.params
        L = np.ones(len(self.states))
        for i, s in enumerate(self.states):
            if s.startswith("in_"):
                ct = self.containers[s[3:]]
                fp = ct.footprint()
                if xy is None:
                    L[i] = ct.p_open + 0.02
                else:
                    m = 0.0 if fp is None else fp.membership(xy, pr.pos_sigma_m)
                    L[i] = m * ct.p_open + 0.02
            elif s == HELD:
                L[i] = pr.held_pos_like if xy is not None else 1.0
            elif xy is None:
                continue
            elif s == ELSEWHERE:
                m = self._memberships([xy])
                L[i] = max(m[ELSEWHERE], 0.02)
            else:
                L[i] = self.zones[s[5:]].membership(xy, pr.pos_sigma_m)
        return L

    def _miss(self, s: str, hands_seen: bool) -> float:
        pr = self.params
        if s == HELD:
            return pr.miss_held_hand_seen if hands_seen else pr.miss_held
        if s.startswith("in_"):
            po = self.containers[s[3:]].p_open
            return po * pr.miss_in_open + (1 - po) * pr.miss_in_closed
        return pr.miss_visible

    # -- association ----------------------------------------------------------------------------

    @staticmethod
    def _pick(dets, classes, pos, track_id, taken: set[int]):
        best, best_score = None, -1.0
        for k, (d, xy) in enumerate(dets):
            if k in taken or d.cls not in classes:
                continue
            score = d.conf
            if pos is not None and xy is not None:
                score *= math.exp(-float(np.sum((xy - pos) ** 2)) / (2 * 0.15**2))
            if track_id is not None and d.track_id == track_id:
                score += 0.1
            if score > best_score:
                best, best_score = k, score
        return best

    # -- update ---------------------------------------------------------------------------------

    def update(
        self,
        frame_idx: int,
        t: float,
        *,
        detections: DetectionSet | None = None,
        hands: HandInput | None = None,
        hoi: Mapping[str, float] | None = None,
        rack_state: str = "none",
    ) -> WorldSnapshot:
        """Advance to ``t`` and fold in this frame's observations.

        ``detections`` must already carry ``bbox_rack_xy`` where a rack frame exists (None:
        the detector did not run this frame). ``hands``: None/``NO_HANDS_INFO`` when no hand
        model runs. ``hoi``: optional ``{object: p(holding)}`` from the HOI classifier.
        """
        pr = self.params
        hands = hands or NO_HANDS_INFO
        dt = 0.0 if self._t is None else min(max(0.0, t - self._t), pr.max_dt_s)
        w = 1.0 if self._t is None else min(1.0, dt / pr.obs_tau_s)
        self._t, self.frame_idx = t, frame_idx
        ran = detections is not None
        dets = []
        if detections is not None:
            dets = [
                (d, None if d.bbox_rack_xy is None else np.asarray(d.bbox_rack_xy, float))
                for d in detections.detections
            ]
        taken: set[int] = set()

        # containers first: the movables' likelihoods depend on open/closed and footprints
        for ct in self.containers.values():
            self._update_container(ct, dets, taken, hands, dt, w, t)
        hands_seen = hands.observed and hands.n > 0
        for mv in self.movables.values():
            self._update_movable(mv, dets, taken, hands, hoi, dt, w, t, ran, hands_seen)

        if ran:
            self._update_visibility(dt, rack_state, hands_seen)
        return self.snapshot(rack_state)

    def _update_container(self, ct: _Container, dets, taken, hands, dt, w, t) -> None:
        pr = self.params
        k = self._pick(dets, ct.box_classes, ct.pos, ct.track_id, taken)
        ct.det, ct.det_xy = (None, None) if k is None else dets[k]
        lid = None
        if k is not None:
            taken.add(k)
            ct.track_id = ct.det.track_id  # type: ignore[union-attr]
            ct.last_seen_t, ct.ever_seen = t, True
        if ct.lid_classes:
            kl = self._pick(dets, ct.lid_classes, None, None, taken)
            if kl is not None:
                taken.add(kl)
                lid = dets[kl]
        # footprint follows the container's detections (snap on first sight)
        if ct.det is not None and ct.det_xy is not None and ct.det.conf >= 0.4:
            if ct.home is None:
                h = pr.default_footprint_m / 2
                x, y = ct.det_xy
                ct.home = Zone(ct.name, [[x - h, y - h], [x + h, y + h]])
            target = ct.det_xy - np.asarray(ct.home.centroid)
            a = 1.0 if ct.pos is None else 1 - math.exp(-dt / pr.footprint_follow_tau_s)
            ct.offset = ct.offset + a * (target - ct.offset)
            ct.pos = ct.det_xy
        # transition: opening/closing needs a hand at the container
        gate = self._gate(hands, self._near_zone(hands, ct.footprint()))
        r = pr.open_rate * gate + pr.correction_rate
        ct.p_open = 0.5 + (ct.p_open - 0.5) * math.exp(-2 * r * dt)
        # emission
        lo, lc = 1.0, 1.0
        if ct.det is not None:
            c = ct.det.conf
            if ct.det.cls.endswith("_open"):
                lo, lc = c * 0.95 + (1 - c) * 0.5, c * 0.05 + (1 - c) * 0.5
            elif ct.det.cls.endswith("_closed"):
                lo, lc = c * 0.05 + (1 - c) * 0.5, c * 0.95 + (1 - c) * 0.5
        if lid is not None:
            d, xy = lid
            fp = ct.footprint()
            if xy is not None and fp is not None:
                off = fp.distance_outside(xy) > 0.03  # a lid lying off the box → open
                c = d.conf
                lo *= c * (0.8 if off else 0.4) + (1 - c) * 0.5
                lc *= c * (0.2 if off else 0.6) + (1 - c) * 0.5
        po = ct.p_open * lo**w
        pc = (1 - ct.p_open) * lc**w
        ct.p_open = min(max(po / (po + pc), 1e-6), 1 - 1e-6)

    def _update_movable(self, mv: _Movable, dets, taken, hands, hoi, dt, w, t, ran, hands_seen):
        pr = self.params
        k = self._pick(dets, mv.classes, mv.pos, mv.track_id, taken)
        mv.det, mv.det_xy = (None, None) if k is None else dets[k]
        if k is not None:
            taken.add(k)
            mv.track_id = mv.det.track_id  # type: ignore[union-attr]
            mv.last_seen_t = mv.supported_t = t
        xy = mv.det_xy

        # position, speed, carried position
        mv.speed = None
        if xy is not None:
            while mv.hist and t - mv.hist[0][0] > 3 * pr.speed_window_s:
                mv.hist.popleft()
            ref = next(
                ((tt, p) for tt, p in reversed(mv.hist) if t - tt >= pr.speed_window_s), None
            )
            if ref is not None and t > ref[0]:
                v = float(np.linalg.norm(xy - ref[1])) / (t - ref[0])
                mv.speed = v if v <= pr.max_speed_mps else None
            mv.hist.append((t, xy))
            mv.pos = mv.carry = xy
        elif mv.det is None:
            if mv.carry is not None and hands.points_rack and mv.p(HELD) > 0.5:
                centres = [p.mean(axis=0) for p in hands.points_rack]
                dists = [float(np.linalg.norm(c - mv.carry)) for c in centres]
                j = int(np.argmin(dists))
                if dists[j] <= pr.carry_radius_m:
                    mv.carry = centres[j]

        # a hand inside an open container supports the belief about what is inside it
        for c, ct in self.containers.items():
            if (
                mv.p(f"in_{c}") > 0.5
                and ct.p_open > 0.5
                and (self._near_zone(hands, ct.footprint()) or 0.0) > 0.5
            ):
                mv.supported_t = t

        # transition
        refs = [xy] if xy is not None else [p for p in (mv.pos, mv.carry) if p is not None]
        memb = self._memberships(refs)
        near = self._near_point(
            hands, xy if xy is not None else mv.pos, None if mv.det is None else mv.det.bbox_xyxy
        )
        g_pick = self._gate(hands, near)
        g_place = self._release_gate(hands, near)
        n = len(self.states)
        Q = np.full((n, n), pr.correction_rate)
        h = mv.idx[HELD]
        for s, i in mv.idx.items():
            if s == HELD:
                continue
            open_ok = self.containers[s[3:]].p_open if s.startswith("in_") else 1.0
            Q[i, h] += pr.pick_rate * g_pick * open_ok
            Q[h, i] += pr.place_rate * g_place * memb[s]
        np.fill_diagonal(Q, 0.0)
        np.fill_diagonal(Q, -Q.sum(axis=1))
        b = mv.b @ expm(Q * dt) if dt > 0 else mv.b

        # emission
        L = np.ones(n)
        if ran:
            if mv.det is not None:
                c = mv.det.conf
                L *= c * self._pos_likelihood(xy) + (1 - c) * 0.5
                if mv.speed is not None:
                    rest = math.exp(-0.5 * (mv.speed / pr.static_speed_mps) ** 2)
                    L[np.arange(n) != h] *= max(rest, pr.moving_floor)
                    L[h] *= pr.held_static_like + (1 - pr.held_static_like) * (1 - rest)
                if hands.observed and hands.n == 0:
                    L[h] *= pr.hand_unseen_like
                elif hands.observed and near is not None:
                    L[h] *= near + (1 - near) * pr.hand_far_like
            else:
                v = self.visibility
                L *= np.array([v * self._miss(s, hands_seen) + (1 - v) for s in self.states])
        if hoi and mv.name in hoi:
            p = min(max(float(hoi[mv.name]), pr.hoi_clip), 1 - pr.hoi_clip)
            L *= np.where(np.arange(n) == h, p, 1 - p)
        mv.b = _norm(b * L**w)

    def _update_visibility(self, dt: float, rack_state: str, hands_seen: bool) -> None:
        pr = self.params
        expected, observed = 0.0, 0
        for ct in self.containers.values():
            if ct.ever_seen:
                expected += 1 - pr.miss_visible
                observed += ct.det is not None
        for mv in self.movables.values():
            expected += sum(
                float(mv.b[i]) * (1 - self._miss(s, hands_seen)) for s, i in mv.idx.items()
            )
            observed += mv.det is not None
        ratio = min(1.0, (observed + 0.5) / (expected + 0.5))
        rack = {"fresh": 1.0, "none": 1.0, "held": 0.8, "invalid": 0.3}.get(rack_state, 1.0)
        a = 1.0 if dt <= 0 else 1 - math.exp(-dt / pr.vis_tau_s)
        self.visibility += a * (ratio * rack - self.visibility)
        self.visibility = min(max(self.visibility, 0.0), 1.0)

    # -- output ---------------------------------------------------------------------------------

    def predicate_values(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for c, ct in self.containers.items():
            out[f"{c}.open"] = ct.p_open
        for o, mv in self.movables.items():
            out[f"{o}.{HELD}"] = mv.p(HELD)
            for c in self.container_names:
                out[f"{o}.in_{c}"] = mv.p(f"in_{c}")
            for z in self.zones:
                out[f"{o}.in_zone_{z}"] = mv.p(f"zone_{z}")
        return out

    def snapshot(self, rack_state: str = "none") -> WorldSnapshot:
        pr = self.params
        objs: dict[str, ObjectState] = {}
        for c, ct in self.containers.items():
            fp = ct.footprint()
            loc = "open" if ct.p_open >= 0.5 else "closed"
            if max(ct.p_open, 1 - ct.p_open) < pr.location_min_p:
                loc = UNKNOWN
            objs[c] = ObjectState(
                c,
                "container",
                loc,
                {"open": ct.p_open, "closed": 1 - ct.p_open},
                None if ct.pos is None else (float(ct.pos[0]), float(ct.pos[1])),
                ct.det is not None,
                ct.last_seen_t,
                ct.last_seen_t,
                p_open=ct.p_open,
                footprint=None if fp is None else tuple(map(tuple, fp.to_list())),  # type: ignore[arg-type]
            )
        for o, mv in self.movables.items():
            i = int(np.argmax(mv.b))
            loc = mv.states[i] if mv.b[i] >= pr.location_min_p else UNKNOWN
            objs[o] = ObjectState(
                o,
                "movable",
                loc,
                {s: float(mv.b[j]) for s, j in mv.idx.items()},
                None if mv.pos is None else (float(mv.pos[0]), float(mv.pos[1])),
                mv.det is not None,
                mv.last_seen_t,
                mv.supported_t,
                speed=mv.speed,
            )
        return WorldSnapshot(
            self.frame_idx,
            self._t if self._t is not None else 0.0,
            objs,
            self.predicate_values(),
            self.visibility,
            rack_state,
        )
