"""The feature spec shared by offline training and online inference (T-HOI-TEMP).

One definition, two callers: ``tools/extract_features.py`` runs it over recorded takes to build
the training cache, and ``actus.models.runtime.ModelStack`` runs the *same* code per frame in the
live pipeline. If these ever diverge the models silently break, so everything either model sees
is built here and nowhere else.

What goes in
------------
The real perception stack's per-frame output: a ``RackFrame``, a ``DetectionSet``, a ``HandSet``
and a ``PoseSet``. Nothing from the protocol engine, the world state or the annotations — the
features must not be able to encode "which step is expected next", or a model trained on them
would smuggle the protocol prior into what is supposed to be prior-free evidence.

Design rules
------------
* **Rack coordinates, not pixels.** Positions are metres in the rack frame, divided by the board
  size, so a camera rolled 90° or 180° yields the same numbers. Pixel-space features appear only
  where they are genuinely scale-relative (a hand landmark inside a grown bounding box).
* **Frame index is the clock, but rates are per second.** Every velocity is a metres/second
  difference over ``vel_window_s`` of wall time, never a per-frame delta, so a model trained on
  30 fps takes behaves the same at 15 fps live.
* **Missing is a value, not a zero.** Every block carries an explicit presence flag and its
  payload is zeroed when absent. A consumer that wants to abstain reads the flag
  (``FrameFeatures.pair_valid``); it never has to guess whether 0.0 means "at the origin" or
  "not seen".

Layout
------
``FrameFeatures.frame`` is a flat ``(F_FRAME,)`` vector for the temporal action model.
``FrameFeatures.pair`` is ``(n_hands, n_objects, F_PAIR)`` for the HOI classifier, one row per
(hand, object) pair, with ``pair_valid`` marking the rows where both were actually observed.
``feature_names()`` / ``pair_feature_names()`` return the matching names, and ``SPEC_VERSION``
is written into every cache and model file: a model whose spec version differs from the running
code is refused rather than fed features it never saw.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from actus.core.contracts import DetectionSet, HandSet, PoseSet, RackFrame
from actus.spatial.rack_frame import img_to_rack
from actus.spatial.zones import Zone, as_polygon, zones_from_protocol

SPEC_VERSION = 1

HAND_SIDES: tuple[str, ...] = ("left", "right")

# Hand landmarks used for object proximity: wrist, the four MCP knuckles and three fingertips.
# Same set as actus.spatial.world_state.HAND_KEYPOINTS, kept here so the feature layout is
# readable on its own.
HAND_KEYPOINTS: tuple[int, ...] = (0, 4, 5, 8, 9, 12, 13, 17)
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, PINKY_TIP = 0, 4, 8, 12, 20
INDEX_MCP, PINKY_MCP = 5, 17

# MediaPipe Pose world-landmark indices for the upper body.
POSE_JOINTS: tuple[int, ...] = (11, 12, 13, 14, 15, 16)  # shoulders, elbows, wrists
L_SHOULDER, R_SHOULDER = 11, 12

EPS = 1e-9


@dataclass(frozen=True)
class FeatureParams:
    """Tuning shared by extraction and runtime. Written into the cache and the model files."""

    vel_window_s: float = 0.25  # wall time a velocity is differenced over
    hist_s: float = 1.0  # how much per-object/per-hand history to keep
    zone_sigma_m: float = 0.05  # softness of zone membership
    bbox_margin: float = 0.2  # bbox grown by this fraction counts as "hand inside"
    max_speed_mps: float = 3.0  # faster than this is an association jump, reported as missing


@dataclass(frozen=True)
class FrameFeatures:
    """One frame's features. ``frame`` feeds the temporal model, ``pair`` feeds HOI."""

    frame_idx: int
    t: float
    frame: np.ndarray  # (F_FRAME,) float32
    pair: np.ndarray  # (n_hands, n_objects, F_PAIR) float32
    pair_valid: np.ndarray  # (n_hands, n_objects) bool -- both hand and object observed
    rack_valid: bool

    def pairs(self) -> Iterable[tuple[int, int, np.ndarray]]:
        """(hand index, object index, feature row) for the valid pairs only."""
        for h in range(self.pair.shape[0]):
            for o in range(self.pair.shape[1]):
                if self.pair_valid[h, o]:
                    yield h, o, self.pair[h, o]


# ----------------------------------------------------------------------------------------------
# Feature names. The order here *is* the layout; the builders below append in the same order.
# ----------------------------------------------------------------------------------------------


def _container_names(c: str) -> list[str]:
    return [f"{c}.present", f"{c}.x", f"{c}.y", f"{c}.area", f"{c}.conf_open", f"{c}.conf_closed",
            f"{c}.conf_lid"]  # fmt: skip


def _object_names(o: str, zones: Sequence[str]) -> list[str]:
    return [
        f"{o}.present",
        f"{o}.conf",
        f"{o}.x",
        f"{o}.y",
        f"{o}.speed",
        f"{o}.area",
        f"{o}.in_footprint",
        *[f"{o}.in_zone_{z}" for z in zones],
    ]


def _hand_names(h: str, zones: Sequence[str]) -> list[str]:
    return [
        f"hand_{h}.present",
        f"hand_{h}.conf",
        f"hand_{h}.wrist_x",
        f"hand_{h}.wrist_y",
        f"hand_{h}.palm_x",
        f"hand_{h}.palm_y",
        f"hand_{h}.speed",
        f"hand_{h}.aperture",
        f"hand_{h}.curl",
        f"hand_{h}.spread",
        f"hand_{h}.in_footprint",
        f"hand_{h}.dist_footprint",
        *[f"hand_{h}.dist_zone_{z}" for z in zones],
    ]


def _pose_names() -> list[str]:
    names = ["pose.present", "pose.conf", "pose.shoulder_width"]
    names += [f"pose.torso_{ax}" for ax in "xyz"]
    for j in POSE_JOINTS:
        names += [f"pose.j{j}_{ax}" for ax in "xyz"]
    return names


class FeatureLayout:
    """Names and sizes for one protocol's objects and zones."""

    def __init__(self, containers: Sequence[str], objects: Sequence[str], zones: Sequence[str]):
        self.containers = tuple(containers)
        self.objects = tuple(objects)
        self.zones = tuple(zones)
        self.hands = HAND_SIDES

        names: list[str] = ["rack.valid", "rack.held", "rack.markers", "det.ran"]
        for c in self.containers:
            names += _container_names(c)
        for o in self.objects:
            names += _object_names(o, self.zones)
        for h in self.hands:
            names += _hand_names(h, self.zones)
        for h in self.hands:
            for o in self.objects:
                names += [
                    f"x.{h}_{o}.min_dist",
                    f"x.{h}_{o}.d_min_dist",
                    f"x.{h}_{o}.co_motion",
                ]
        names += ["hands.separation"]
        names += _pose_names()
        self.frame_names = tuple(names)

        pair: list[str] = ["hand_present", "obj_present", "rack_valid"]
        pair += [f"kp{k}_dist" for k in HAND_KEYPOINTS]
        pair += [
            "min_dist",
            "mean_dist",
            "frac_inside_bbox",
            "dist_over_diag",
            "dx",
            "dy",
            "d_min_dist",
            "hand_speed",
            "obj_speed",
            "rel_speed",
            "co_motion",
            "aperture",
            "curl",
            "spread",
            "obj_conf",
            "obj_area",
            "co_disp_corr",
        ]
        self.pair_names = tuple(pair)

    @property
    def n_frame(self) -> int:
        return len(self.frame_names)

    @property
    def n_pair(self) -> int:
        return len(self.pair_names)

    @classmethod
    def from_protocol(cls, protocol) -> FeatureLayout:
        from actus.spatial.world_state import parse_vocabulary

        containers, objects = parse_vocabulary(protocol)
        return cls(containers, objects, sorted(protocol.zones))


# ----------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------


def _finite(a: np.ndarray | None) -> np.ndarray | None:
    if a is None:
        return None
    a = np.asarray(a, dtype=np.float64)
    return a if np.all(np.isfinite(a)) else None


class _Track:
    """Position history on one moving thing, for per-second velocities."""

    def __init__(self, params: FeatureParams) -> None:
        self.params = params
        self.hist: deque[tuple[float, np.ndarray]] = deque()
        self.pos: np.ndarray | None = None

    def update(self, t: float, pos: np.ndarray | None) -> None:
        self.pos = pos
        if pos is None:
            return
        while self.hist and t - self.hist[0][0] > self.params.hist_s:
            self.hist.popleft()
        self.hist.append((t, np.asarray(pos, dtype=np.float64)))

    def _ref(self, t: float) -> tuple[float, np.ndarray] | None:
        w = self.params.vel_window_s
        return next(((tt, p) for tt, p in reversed(self.hist) if t - tt >= w), None)

    def velocity(self, t: float) -> np.ndarray | None:
        """Metres per second over ``vel_window_s``; None when there is not enough history."""
        if self.pos is None:
            return None
        ref = self._ref(t)
        if ref is None or t <= ref[0]:
            return None
        v = (np.asarray(self.pos, dtype=np.float64) - ref[1]) / (t - ref[0])
        return None if float(np.linalg.norm(v)) > self.params.max_speed_mps else v

    def speed(self, t: float) -> float | None:
        v = self.velocity(t)
        return None if v is None else float(np.linalg.norm(v))

    def displacement(self, t: float) -> np.ndarray | None:
        ref = self._ref(t)
        if self.pos is None or ref is None:
            return None
        return np.asarray(self.pos, dtype=np.float64) - ref[1]


class _Scalar:
    """Scalar history, for d(distance)/dt."""

    def __init__(self, params: FeatureParams) -> None:
        self.params = params
        self.hist: deque[tuple[float, float]] = deque()

    def update(self, t: float, v: float | None) -> float | None:
        """Push ``v`` and return its per-second rate of change, or None."""
        if v is None:
            self.hist.clear()
            return None
        while self.hist and t - self.hist[0][0] > self.params.hist_s:
            self.hist.popleft()
        w = self.params.vel_window_s
        ref = next(((tt, p) for tt, p in reversed(self.hist) if t - tt >= w), None)
        self.hist.append((t, float(v)))
        if ref is None or t <= ref[0]:
            return None
        return (float(v) - ref[1]) / (t - ref[0])


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < EPS or nb < EPS:
        return None
    return float(np.dot(a, b) / (na * nb))


def _bbox_centre(bbox: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float64)


def _grow(bbox: tuple[float, float, float, float], margin: float):
    x1, y1, x2, y2 = bbox
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    return x1 - dx, y1 - dy, x2 + dx, y2 + dy


# ----------------------------------------------------------------------------------------------
# The featurizer
# ----------------------------------------------------------------------------------------------


@dataclass
class _ObjSlot:
    track: _Track
    pos: np.ndarray | None = None  # rack metres, normalised
    pos_px: np.ndarray | None = None
    bbox: tuple[float, float, float, float] | None = None
    conf: float = 0.0
    area: float = 0.0
    present: bool = False


@dataclass
class _HandSlot:
    palm: _Track
    present: bool = False
    conf: float = 0.0
    pts_rack: np.ndarray | None = None  # (21, 2) normalised rack
    pts_px: np.ndarray | None = None  # (21, 2)
    wrist: np.ndarray | None = None
    palm_xy: np.ndarray | None = None
    aperture: float = 0.0
    curl: float = 0.0
    spread: float = 0.0


class FrameFeaturizer:
    """Stateful per-frame featurizer. One instance per stream; call ``update`` in frame order.

    It keeps short position histories (``FeatureParams.hist_s``) to difference velocities over
    wall time, so it must see frames in order. ``reset()`` clears that history between takes.
    """

    def __init__(
        self,
        protocol,
        *,
        params: FeatureParams | None = None,
        layout: FeatureLayout | None = None,
        board_size: tuple[float, float] = (1.0, 1.0),
        outer_footprint: Mapping[str, Sequence[Sequence[float]]] | None = None,
    ) -> None:
        self.protocol = protocol
        self.params = params or FeatureParams()
        self.layout = layout or FeatureLayout.from_protocol(protocol)
        self.board_w = float(board_size[0]) or 1.0
        self.board_h = float(board_size[1]) or 1.0
        self.zones: dict[str, Zone] = zones_from_protocol(protocol)
        self._footprints: dict[str, Zone] = {}
        for name, poly in (outer_footprint or {}).items():
            self._footprints[name] = Zone(name, as_polygon(poly))
        self._classes = {
            o: set(getattr(protocol.objects.get(o), "classes", ()) or ())
            for o in (*self.layout.containers, *self.layout.objects)
        }
        self.reset()

    # -- state ---------------------------------------------------------------------------------

    def reset(self) -> None:
        p = self.params
        self._objs = {o: _ObjSlot(_Track(p)) for o in self.layout.objects}
        self._conts = {c: _ObjSlot(_Track(p)) for c in self.layout.containers}
        self._hands = {h: _HandSlot(_Track(p)) for h in self.layout.hands}
        self._dist = {(h, o): _Scalar(p) for h in self.layout.hands for o in self.layout.objects}
        self._t: float | None = None

    # -- geometry ------------------------------------------------------------------------------

    def _norm(self, xy: np.ndarray | None) -> np.ndarray | None:
        if xy is None:
            return None
        return np.array([xy[0] / self.board_w, xy[1] / self.board_h], dtype=np.float64)

    def _to_rack(self, rack: RackFrame | None, pts_px) -> np.ndarray | None:
        if rack is None or not rack.valid:
            return None
        out = img_to_rack(rack, pts_px)
        return _finite(out)

    # -- per frame -----------------------------------------------------------------------------

    def update(
        self,
        frame_idx: int,
        t: float,
        *,
        rack: RackFrame | None = None,
        detections: DetectionSet | None = None,
        hands: HandSet | None = None,
        pose: PoseSet | None = None,
    ) -> FrameFeatures:
        """Fold one frame of perception into a feature row.

        ``detections`` may already carry ``bbox_rack_xy`` (the spatial lane fills it in); when it
        does not and ``rack`` is valid, the box centres are projected here. A missing input is
        "not observed", never a zero.
        """
        self._t = t
        rack_ok = bool(rack is not None and rack.valid)
        self._read_detections(t, rack, detections)
        self._read_hands(t, rack, hands)
        pose_block = self._pose_block(pose, rack)

        frame = self._frame_vector(t, rack, detections is not None, pose_block)
        pair, valid = self._pair_matrix(t, rack_ok)
        return FrameFeatures(int(frame_idx), float(t), frame, pair, valid, rack_ok)

    # -- inputs --------------------------------------------------------------------------------

    def _read_detections(self, t: float, rack: RackFrame | None, dets: DetectionSet | None) -> None:
        best: dict[str, object] = {}
        if dets is not None:
            for d in dets.detections:
                for name, classes in self._classes.items():
                    if d.cls in classes:
                        cur = best.get(name)
                        if cur is None or d.conf > cur.conf:  # type: ignore[union-attr]
                            best[name] = d
        # containers keep the per-class confidences too (open / closed / lid)
        for name, slot in (*self._objs.items(), *self._conts.items()):
            d = best.get(name)
            slot.present = d is not None
            slot.conf = float(d.conf) if d is not None else 0.0
            slot.bbox = d.bbox_xyxy if d is not None else None
            slot.pos_px = _bbox_centre(d.bbox_xyxy) if d is not None else None
            xy = None
            if d is not None:
                if d.bbox_rack_xy is not None:
                    xy = np.asarray(d.bbox_rack_xy, dtype=np.float64)
                else:
                    proj = self._to_rack(rack, [slot.pos_px])
                    xy = None if proj is None else proj[0]
            slot.pos = self._norm(_finite(xy))
            slot.area = 0.0
            if d is not None:
                x1, y1, x2, y2 = d.bbox_xyxy
                slot.area = float((x2 - x1) * (y2 - y1))
            slot.track.update(t, slot.pos)
        self._class_conf = {
            c: {
                cls: max(
                    (d.conf for d in (dets.detections if dets else ()) if d.cls == cls),
                    default=0.0,
                )
                for cls in self._classes[c]
            }
            for c in self.layout.containers
        }

    def _read_hands(self, t: float, rack: RackFrame | None, hands: HandSet | None) -> None:
        chosen: dict[str, object] = {}
        if hands is not None:
            for h in hands.hands:
                cur = chosen.get(h.side)
                if cur is None or h.conf > cur.conf:  # type: ignore[union-attr]
                    chosen[h.side] = h
        for side, slot in self._hands.items():
            h = chosen.get(side)
            slot.present = h is not None
            slot.conf = float(h.conf) if h is not None else 0.0
            slot.pts_px = np.asarray(h.landmarks_px, dtype=np.float64) if h is not None else None
            pts_rack = None
            if h is not None:
                if h.landmarks_rack is not None:
                    pts_rack = _finite(np.asarray(h.landmarks_rack)[:, :2])
                else:
                    pts_rack = self._to_rack(rack, slot.pts_px)
            if pts_rack is not None:
                pts_rack = np.stack(
                    [pts_rack[:, 0] / self.board_w, pts_rack[:, 1] / self.board_h], axis=1
                )
            slot.pts_rack = pts_rack
            slot.wrist = pts_rack[WRIST] if pts_rack is not None else None
            slot.palm_xy = (
                pts_rack[[WRIST, INDEX_MCP, PINKY_MCP]].mean(axis=0)
                if pts_rack is not None
                else None
            )
            slot.aperture, slot.curl, slot.spread = _grip(slot.pts_px)
            slot.palm.update(t, slot.palm_xy)

    def _pose_block(self, pose: PoseSet | None, rack: RackFrame | None) -> list[float]:
        p = None
        if pose is not None and pose.poses:
            p = max(pose.poses, key=lambda q: q.conf)
        if p is None:
            return [0.0] * (3 + 3 + 3 * len(POSE_JOINTS))
        world = _finite(np.asarray(p.world_landmarks, dtype=np.float64))
        torso = _finite(p.torso_vec_rack) if p.torso_vec_rack is not None else None
        out = [1.0, float(p.conf)]
        if world is None:
            return out + [0.0] + [0.0] * 3 + [0.0] * (3 * len(POSE_JOINTS))
        centre = (world[L_SHOULDER] + world[R_SHOULDER]) / 2
        width = float(np.linalg.norm(world[L_SHOULDER] - world[R_SHOULDER]))
        scale = width if width > EPS else 1.0
        out.append(width)
        out += list(torso) if torso is not None else [0.0, 0.0, 0.0]
        for j in POSE_JOINTS:
            out += list((world[j] - centre) / scale)
        return out

    # -- vectors -------------------------------------------------------------------------------

    def _frame_vector(
        self, t: float, rack: RackFrame | None, det_ran: bool, pose_block: Sequence[float]
    ) -> np.ndarray:
        v: list[float] = [
            1.0 if (rack is not None and rack.valid) else 0.0,
            1.0 if (rack is not None and rack.age_frames > 0) else 0.0,
            (len(rack.markers_seen) / 4.0) if rack is not None else 0.0,
            1.0 if det_ran else 0.0,
        ]
        for c in self.layout.containers:
            slot = self._conts[c]
            confs = self._class_conf.get(c, {})
            v += [
                1.0 if slot.present else 0.0,
                *(slot.pos if slot.pos is not None else (0.0, 0.0)),
                slot.area / (1280.0 * 720.0),
                _conf_matching(confs, "open"),
                _conf_matching(confs, "closed"),
                _conf_matching(confs, "lid"),
            ]
        for o in self.layout.objects:
            slot = self._objs[o]
            speed = slot.track.speed(t)
            v += [
                1.0 if slot.present else 0.0,
                slot.conf,
                *(slot.pos if slot.pos is not None else (0.0, 0.0)),
                speed if speed is not None else 0.0,
                slot.area / (1280.0 * 720.0),
                self._footprint_membership(slot.pos),
                *[self._zone_membership(z, slot.pos) for z in self.layout.zones],
            ]
        for h in self.layout.hands:
            slot = self._hands[h]
            speed = slot.palm.speed(t)
            v += [
                1.0 if slot.present else 0.0,
                slot.conf,
                *(slot.wrist if slot.wrist is not None else (0.0, 0.0)),
                *(slot.palm_xy if slot.palm_xy is not None else (0.0, 0.0)),
                speed if speed is not None else 0.0,
                slot.aperture,
                slot.curl,
                slot.spread,
                self._footprint_membership(slot.palm_xy),
                self._footprint_distance(slot.palm_xy),
                *[self._zone_distance(z, slot.palm_xy) for z in self.layout.zones],
            ]
        for h in self.layout.hands:
            for o in self.layout.objects:
                d, _ = self._pair_distance(h, o)
                rate = self._dist[(h, o)].update(t, d)
                co = _cosine(self._hands[h].palm.velocity(t), self._objs[o].track.velocity(t))
                v += [
                    d if d is not None else 0.0,
                    rate if rate is not None else 0.0,
                    co if co is not None else 0.0,
                ]
        a, b = (self._hands[s].palm_xy for s in self.layout.hands)
        v.append(float(np.linalg.norm(a - b)) if a is not None and b is not None else 0.0)
        v += list(pose_block)
        arr = np.asarray(v, dtype=np.float32)
        if arr.size != self.layout.n_frame:  # pragma: no cover - layout bug guard
            raise AssertionError(f"frame vector is {arr.size}, layout says {self.layout.n_frame}")
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def _pair_matrix(self, t: float, rack_ok: bool) -> tuple[np.ndarray, np.ndarray]:
        n_h, n_o = len(self.layout.hands), len(self.layout.objects)
        pair = np.zeros((n_h, n_o, self.layout.n_pair), dtype=np.float32)
        valid = np.zeros((n_h, n_o), dtype=bool)
        for hi, h in enumerate(self.layout.hands):
            for oi, o in enumerate(self.layout.objects):
                row, ok = self._pair_row(t, h, o, rack_ok)
                pair[hi, oi] = row
                valid[hi, oi] = ok
        return pair, valid

    def _pair_row(self, t: float, h: str, o: str, rack_ok: bool) -> tuple[np.ndarray, bool]:
        hs, os_ = self._hands[h], self._objs[o]
        ok = bool(hs.present and os_.present)
        v: list[float] = [
            1.0 if hs.present else 0.0,
            1.0 if os_.present else 0.0,
            1.0 if rack_ok else 0.0,
        ]
        dists = self._keypoint_distances(h, o)
        v += list(dists) if dists is not None else [0.0] * len(HAND_KEYPOINTS)
        min_d, mean_d = (
            (float(np.min(dists)), float(np.mean(dists))) if dists is not None else (0.0, 0.0)
        )
        v += [min_d, mean_d, self._frac_inside(h, o), self._dist_over_diag(h, o)]
        delta = os_.pos - hs.palm_xy if (os_.pos is not None and hs.palm_xy is not None) else None
        v += list(delta) if delta is not None else [0.0, 0.0]

        d, _ = self._pair_distance(h, o)
        v.append(self._last_rate(h, o, t, d))
        hv, ov = hs.palm.velocity(t), os_.track.velocity(t)
        hspeed = float(np.linalg.norm(hv)) if hv is not None else 0.0
        ospeed = float(np.linalg.norm(ov)) if ov is not None else 0.0
        rel = float(np.linalg.norm(hv - ov)) if (hv is not None and ov is not None) else 0.0
        co = _cosine(hv, ov)
        v += [hspeed, ospeed, rel, co if co is not None else 0.0]
        v += [hs.aperture, hs.curl, hs.spread, os_.conf, os_.area / (1280.0 * 720.0)]
        v.append(self._co_displacement(h, o, t))
        arr = np.asarray(v, dtype=np.float32)
        if arr.size != self.layout.n_pair:  # pragma: no cover - layout bug guard
            raise AssertionError(f"pair row is {arr.size}, layout says {self.layout.n_pair}")
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), ok

    # -- pair pieces ---------------------------------------------------------------------------

    def _keypoint_distances(self, h: str, o: str) -> np.ndarray | None:
        hs, os_ = self._hands[h], self._objs[o]
        if hs.pts_rack is None or os_.pos is None:
            return None
        pts = hs.pts_rack[list(HAND_KEYPOINTS)]
        return np.linalg.norm(pts - os_.pos[None, :], axis=1)

    def _pair_distance(self, h: str, o: str) -> tuple[float | None, bool]:
        d = self._keypoint_distances(h, o)
        return (None, False) if d is None else (float(np.min(d)), True)

    def _last_rate(self, h: str, o: str, t: float, d: float | None) -> float:
        """d(min distance)/dt as already computed for the frame vector this tick."""
        sc = self._dist[(h, o)]
        if d is None or len(sc.hist) < 2:
            return 0.0
        w = self.params.vel_window_s
        ref = next(((tt, p) for tt, p in reversed(sc.hist) if t - tt >= w), None)
        if ref is None or t <= ref[0]:
            return 0.0
        return float((d - ref[1]) / (t - ref[0]))

    def _frac_inside(self, h: str, o: str) -> float:
        hs, os_ = self._hands[h], self._objs[o]
        if hs.pts_px is None or os_.bbox is None:
            return 0.0
        x1, y1, x2, y2 = _grow(os_.bbox, self.params.bbox_margin)
        pts = hs.pts_px[list(HAND_KEYPOINTS)]
        inside = (pts[:, 0] >= x1) & (pts[:, 0] <= x2) & (pts[:, 1] >= y1) & (pts[:, 1] <= y2)
        return float(np.mean(inside))

    def _dist_over_diag(self, h: str, o: str) -> float:
        """Closest hand keypoint to the box centre, in units of the box's own diagonal.

        Scale-relative, so it survives the camera moving closer or further without a rack frame.
        """
        hs, os_ = self._hands[h], self._objs[o]
        if hs.pts_px is None or os_.bbox is None or os_.pos_px is None:
            return 0.0
        x1, y1, x2, y2 = os_.bbox
        diag = math.hypot(x2 - x1, y2 - y1)
        if diag < EPS:
            return 0.0
        pts = hs.pts_px[list(HAND_KEYPOINTS)]
        return float(np.min(np.linalg.norm(pts - os_.pos_px[None, :], axis=1)) / diag)

    def _co_displacement(self, h: str, o: str, t: float) -> float:
        """Cosine between the hand's and the object's displacement over the velocity window.

        A held object moves *with* the hand; one resting next to it does not.
        """
        hand_d = self._hands[h].palm.displacement(t)
        obj_d = self._objs[o].track.displacement(t)
        return _cosine(hand_d, obj_d) or 0.0

    # -- zones ---------------------------------------------------------------------------------

    def _zone_membership(self, name: str, pos: np.ndarray | None) -> float:
        z = self.zones.get(name)
        if z is None or pos is None:
            return 0.0
        return float(z.membership(self._denorm(pos), self.params.zone_sigma_m))

    def _zone_distance(self, name: str, pos: np.ndarray | None) -> float:
        z = self.zones.get(name)
        if z is None or pos is None:
            return 0.0
        cx, cy = z.centroid
        p = self._denorm(pos)
        return float(math.hypot(p[0] - cx, p[1] - cy) / max(self.board_w, self.board_h))

    def _footprint_membership(self, pos: np.ndarray | None) -> float:
        if pos is None or not self._footprints:
            return 0.0
        p = self._denorm(pos)
        return max(
            float(z.membership(p, self.params.zone_sigma_m)) for z in self._footprints.values()
        )

    def _footprint_distance(self, pos: np.ndarray | None) -> float:
        if pos is None or not self._footprints:
            return 0.0
        p = self._denorm(pos)
        return min(
            float(math.hypot(p[0] - z.centroid[0], p[1] - z.centroid[1]))
            for z in self._footprints.values()
        ) / max(self.board_w, self.board_h)

    def _denorm(self, pos: np.ndarray) -> np.ndarray:
        return np.array([pos[0] * self.board_w, pos[1] * self.board_h], dtype=np.float64)


def _conf_matching(confs: Mapping[str, float], token: str) -> float:
    """Best confidence among the container's classes whose name contains ``token``.

    Read from the protocol's declared class names (``outer_box_open``, ``lid``, ...) rather than
    hard-coded, so a second protocol naming its container differently still works.
    """
    return max((v for k, v in confs.items() if token in k), default=0.0)


def _grip(pts_px: np.ndarray | None) -> tuple[float, float, float]:
    """Grip shape from hand landmarks, normalised by palm size so it is scale free.

    ``aperture`` thumb tip to index tip, ``curl`` mean fingertip-to-wrist distance (small when
    the fingers are closed around something), ``spread`` index tip to pinky tip.
    """
    if pts_px is None:
        return 0.0, 0.0, 0.0
    pts = np.asarray(pts_px, dtype=np.float64)
    palm = float(np.linalg.norm(pts[INDEX_MCP] - pts[WRIST]))
    if palm < EPS:
        return 0.0, 0.0, 0.0
    aperture = float(np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP])) / palm
    tips = pts[[THUMB_TIP, INDEX_TIP, MIDDLE_TIP, PINKY_TIP]]
    curl = float(np.mean(np.linalg.norm(tips - pts[WRIST], axis=1))) / palm
    spread = float(np.linalg.norm(pts[INDEX_TIP] - pts[PINKY_TIP])) / palm
    return aperture, curl, spread


def feature_names(protocol) -> tuple[str, ...]:
    return FeatureLayout.from_protocol(protocol).frame_names


def pair_feature_names(protocol) -> tuple[str, ...]:
    return FeatureLayout.from_protocol(protocol).pair_names
