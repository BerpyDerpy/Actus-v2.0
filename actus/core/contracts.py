"""Message contracts shared by every lane. FROZEN after P0.

A lane that needs a change proposes it in its PR description; the integrator applies it.

Conventions
-----------
* ``frame_idx`` is the clock. It is the monotonic index assigned by the capture thread (or the
  replay source) and is carried by every per-frame observation.
* ``t`` / ``t_capture`` are seconds on the *pipeline clock*: ``time.monotonic()`` for a live
  source, the container timestamp for a file, synthetic seconds for replay. Never derive it
  from ``frame_idx / fps`` with a guessed fps. Wall-clock (ISO) time is added only by the event
  log.
* Probabilities are floats in [0, 1]. Constructors validate them and raise ``ValueError``.
* Rack coordinates are metres in the payload-rack frame defined in ``configs/rack.yaml``:
  origin at the board's top-left corner (facing the rack), +x right, +y down, +z into the rack.
* All classes are frozen dataclasses. ``__post_init__`` normalises inputs (lists → tuples or
  read-only numpy arrays, str → enum, dict → a private copy), so a producer can pass plain
  Python values and a consumer can rely on the stated types. Treat dict fields as read-only.
* ``to_jsonable`` / ``from_jsonable`` convert any contract (except ``FramePacket``) to and from
  JSON-safe dicts tagged with ``"type": <ClassName>``. Used by the event log and the UI socket.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SCHEMA_VERSION = 1

IDLE = "idle"  # ActionScores key and annotation label for "no step in progress"
OTHER = "other"  # annotation label for a non-step action (T-REC key `x`)

ROTATIONS = (0, 90, 180, 270)


# --------------------------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------------------------


class StepStatus(enum.StrEnum):
    PENDING = "pending"
    VERIFIED = "verified"
    OPERATOR_CONFIRMED = "operator_confirmed"
    SKIPPED = "skipped"
    OUT_OF_ORDER = "out_of_order"
    TIMED_OUT = "timed_out"


class AlertKind(enum.StrEnum):
    SKIP = "skip"
    OUT_OF_ORDER = "out_of_order"
    UNEXPECTED = "unexpected"
    TIMEOUT = "timeout"
    TOO_FAST = "too_fast"


class AnswerValue(enum.StrEnum):
    YES = "yes"
    NO = "no"


class InputVia(enum.StrEnum):
    VOICE = "voice"
    GUI = "gui"


class CommandName(enum.StrEnum):
    NEXT = "next"
    REPEAT = "repeat"
    WHATS_NEXT = "whats_next"
    CONFIRM = "confirm"
    CANCEL = "cancel"


class Topic(enum.StrEnum):
    FRAMES = "frames"  # FramePacket (raw)
    ANNOTATED = "annotated"  # FramePacket with overlays drawn (for recorder / stream / UI)
    DETECTIONS = "detections"  # DetectionSet
    HANDS = "hands"  # HandSet
    POSE = "pose"  # PoseSet
    RACK = "rack"  # RackFrame
    PREDICATES = "predicates"  # Predicates
    EVIDENCE = "evidence"  # StepEvidence
    ACTIONS = "actions"  # ActionScores
    EVENTS = "events"  # StepEvent
    ALERTS = "alerts"  # Alert
    PROMPTS = "prompts"  # Prompt
    ASKS = "asks"  # Ask
    ANSWERS = "answers"  # Answer
    COMMANDS = "commands"  # Command (spoken / GUI "next", "repeat", ...)
    STATS = "stats"  # Stats


# --------------------------------------------------------------------------------------------
# Validation / normalisation helpers
# --------------------------------------------------------------------------------------------


def _set(obj: object, name: str, value: Any) -> None:
    object.__setattr__(obj, name, value)


def _prob(name: str, p: Any) -> float:
    p = float(p)
    if not (0.0 <= p <= 1.0):  # also rejects NaN
        raise ValueError(f"{name} must be a probability in [0, 1], got {p!r}")
    return p


def _frame_idx(v: Any, name: str = "frame_idx") -> int:
    if isinstance(v, bool) or int(v) != v:
        raise ValueError(f"{name} must be an int, got {v!r}")
    v = int(v)
    if v < 0:
        raise ValueError(f"{name} must be >= 0, got {v}")
    return v


def _time(v: Any, name: str = "t") -> float:
    v = float(v)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v!r}")
    return v


def _opt_time(v: Any, name: str) -> float | None:
    return None if v is None else _time(v, name)


def _rotation(v: Any, name: str) -> int:
    v = int(v)
    if v not in ROTATIONS:
        raise ValueError(f"{name} must be one of {ROTATIONS}, got {v}")
    return v


def _array(v: Any, name: str, shape: tuple[int | None, ...], dtype: Any = np.float32) -> np.ndarray:
    arr = np.array(v, dtype=dtype, copy=True)
    ok = arr.ndim == len(shape) and all(
        s is None or a == s for a, s in zip(arr.shape, shape, strict=True)
    )
    if not ok:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    arr.setflags(write=False)
    return arr


def _opt_array(v: Any, name: str, shape: tuple[int | None, ...], dtype: Any = np.float32):
    return None if v is None else _array(v, name, shape, dtype)


def _prob_map(name: str, m: Mapping[str, Any]) -> dict[str, float]:
    return {str(k): _prob(f"{name}[{k!r}]", v) for k, v in m.items()}


# --------------------------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {}


def _register(cls: type) -> type:
    _REGISTRY[cls.__name__] = cls
    return cls


@dataclass(frozen=True, slots=True, eq=False)
class FramePacket:
    """One captured frame. ``image`` is BGR uint8 (H, W, 3) or grey (H, W); it is not copied."""

    frame_idx: int
    t_capture: float
    image: np.ndarray
    source_id: str
    rotation_applied: int = 0  # degrees CCW already applied to `image` (0/90/180/270)

    def __post_init__(self) -> None:
        _set(self, "frame_idx", _frame_idx(self.frame_idx))
        _set(self, "t_capture", _time(self.t_capture, "t_capture"))
        if not isinstance(self.image, np.ndarray) or self.image.ndim not in (2, 3):
            raise ValueError("image must be a 2-D or 3-D numpy array")
        _set(self, "source_id", str(self.source_id))
        _set(self, "rotation_applied", _rotation(self.rotation_applied, "rotation_applied"))


# --------------------------------------------------------------------------------------------
# Perception
# --------------------------------------------------------------------------------------------


@_register
@dataclass(frozen=True, slots=True)
class Detection:
    """One detected object. ``bbox_rack_xy`` is the box centre projected onto the rack plane."""

    track_id: int | None
    cls: str
    conf: float
    bbox_xyxy: tuple[float, float, float, float]  # image pixels
    bbox_rack_xy: tuple[float, float] | None = None  # rack metres, None if no valid rack frame

    def __post_init__(self) -> None:
        if self.track_id is not None:
            _set(self, "track_id", int(self.track_id))
        _set(self, "cls", str(self.cls))
        _set(self, "conf", _prob("conf", self.conf))
        x1, y1, x2, y2 = (float(v) for v in self.bbox_xyxy)
        if not (x1 <= x2 and y1 <= y2):
            raise ValueError(f"bbox_xyxy must satisfy x1<=x2, y1<=y2, got {self.bbox_xyxy}")
        _set(self, "bbox_xyxy", (x1, y1, x2, y2))
        if self.bbox_rack_xy is not None:
            rx, ry = (float(v) for v in self.bbox_rack_xy)
            _set(self, "bbox_rack_xy", (rx, ry))


@_register
@dataclass(frozen=True, slots=True)
class DetectionSet:
    """All detections for one frame (possibly empty). Topic ``detections``."""

    frame_idx: int
    t: float
    detections: tuple[Detection, ...]
    backend: str  # "custom" | "open_vocab" | "replay" | ...

    def __post_init__(self) -> None:
        _set(self, "frame_idx", _frame_idx(self.frame_idx))
        _set(self, "t", _time(self.t))
        dets = tuple(self.detections)
        if not all(isinstance(d, Detection) for d in dets):
            raise ValueError("detections must be Detection instances")
        _set(self, "detections", dets)
        _set(self, "backend", str(self.backend))


@_register
@dataclass(frozen=True, slots=True, eq=False)
class HandObs:
    side: str  # "left" | "right" (the operator's hand, as reported by MediaPipe)
    landmarks_px: np.ndarray  # (21, 2) image pixels, in the ORIGINAL (un-rotated) image
    landmarks_rack: np.ndarray | None  # (21, 3) rack metres
    conf: float

    def __post_init__(self) -> None:
        if self.side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {self.side!r}")
        _set(self, "landmarks_px", _array(self.landmarks_px, "landmarks_px", (21, 2)))
        _set(self, "landmarks_rack", _opt_array(self.landmarks_rack, "landmarks_rack", (21, 3)))
        _set(self, "conf", _prob("conf", self.conf))


@_register
@dataclass(frozen=True, slots=True)
class HandSet:
    """All hands for one frame (0–2). Topic ``hands``."""

    frame_idx: int
    t: float
    hands: tuple[HandObs, ...]
    rotation_used: int = 0  # rotation TTA winner, degrees CCW

    def __post_init__(self) -> None:
        _set(self, "frame_idx", _frame_idx(self.frame_idx))
        _set(self, "t", _time(self.t))
        hands = tuple(self.hands)
        if not all(isinstance(h, HandObs) for h in hands):
            raise ValueError("hands must be HandObs instances")
        _set(self, "hands", hands)
        _set(self, "rotation_used", _rotation(self.rotation_used, "rotation_used"))


@_register
@dataclass(frozen=True, slots=True, eq=False)
class PoseObs:
    landmarks_px: np.ndarray  # (33, 2) image pixels, in the ORIGINAL (un-rotated) image
    # (33, 3) metres, MediaPipe hip-centred world coordinates, as MediaPipe returned them: in the
    # axes of the image it was run on, i.e. rotated by `rotation_used` (not rotated back).
    world_landmarks: np.ndarray
    torso_vec_rack: np.ndarray | None  # (3,) unit vector hips→shoulders in rack coordinates
    conf: float
    rotation_used: int  # rotation TTA winner, degrees CCW
    visibility: np.ndarray | None = None  # (33,) per-landmark visibility in [0, 1]

    def __post_init__(self) -> None:
        _set(self, "landmarks_px", _array(self.landmarks_px, "landmarks_px", (33, 2)))
        _set(self, "world_landmarks", _array(self.world_landmarks, "world_landmarks", (33, 3)))
        _set(self, "torso_vec_rack", _opt_array(self.torso_vec_rack, "torso_vec_rack", (3,)))
        _set(self, "conf", _prob("conf", self.conf))
        _set(self, "rotation_used", _rotation(self.rotation_used, "rotation_used"))
        _set(self, "visibility", _opt_array(self.visibility, "visibility", (33,)))


@_register
@dataclass(frozen=True, slots=True)
class PoseSet:
    """All poses for one frame (0 or 1 today). Topic ``pose``."""

    frame_idx: int
    t: float
    poses: tuple[PoseObs, ...]

    def __post_init__(self) -> None:
        _set(self, "frame_idx", _frame_idx(self.frame_idx))
        _set(self, "t", _time(self.t))
        poses = tuple(self.poses)
        if not all(isinstance(p, PoseObs) for p in poses):
            raise ValueError("poses must be PoseObs instances")
        _set(self, "poses", poses)


@_register
@dataclass(frozen=True, slots=True, eq=False)
class RackFrame:
    """Camera ↔ rack registration for one frame. Topic ``rack``.

    When ``valid`` is False the geometric fields may be None. ``age_frames`` is 0 for a fresh
    marker solve and counts up while the last good pose is being held through an occlusion.
    """

    valid: bool
    H_img_to_rack: np.ndarray | None  # (3, 3) image px → rack-plane metres (z = 0)
    rvec: np.ndarray | None  # (3,) Rodrigues, rack → camera
    tvec: np.ndarray | None  # (3,) metres, rack → camera
    markers_seen: tuple[int, ...]
    reproj_err: float | None  # pixels, RMS of the image->rack plane fit (what gates `valid`)
    frame_idx: int = field(kw_only=True)
    t: float = field(kw_only=True)
    age_frames: int = field(default=0, kw_only=True)

    def __post_init__(self) -> None:
        _set(self, "valid", bool(self.valid))
        _set(self, "H_img_to_rack", _opt_array(self.H_img_to_rack, "H", (3, 3), np.float64))
        _set(self, "rvec", _opt_array(self.rvec, "rvec", (3,), np.float64))
        _set(self, "tvec", _opt_array(self.tvec, "tvec", (3,), np.float64))
        _set(self, "markers_seen", tuple(int(m) for m in self.markers_seen))
        if self.reproj_err is not None:
            _set(self, "reproj_err", float(self.reproj_err))
        if self.valid and self.H_img_to_rack is None:
            raise ValueError("a valid RackFrame needs H_img_to_rack")
        _set(self, "frame_idx", _frame_idx(self.frame_idx))
        _set(self, "t", _time(self.t))
        _set(self, "age_frames", _frame_idx(self.age_frames, "age_frames"))


# --------------------------------------------------------------------------------------------
# Spatial reasoning and models → evidence
# --------------------------------------------------------------------------------------------


@_register
@dataclass(frozen=True, slots=True)
class Predicates:
    """Per-frame probabilities of world-state predicates, e.g. ``red.in_zone_A: 0.93``.

    Names are ``<object>.<predicate>``. ``visibility`` in [0, 1] says how much of the scene the
    sensors could actually see this frame (1 = fully visible, 0 = blind).
    """

    frame_idx: int
    t: float
    values: dict[str, float]
    visibility: float

    def __post_init__(self) -> None:
        _set(self, "frame_idx", _frame_idx(self.frame_idx))
        _set(self, "t", _time(self.t))
        _set(self, "values", _prob_map("values", self.values))
        _set(self, "visibility", _prob("visibility", self.visibility))


@_register
@dataclass(frozen=True, slots=True)
class StepEvidence:
    """Prior-free evidence that ``step_id``'s transition has completed, as of this frame.

    ``p_complete`` must be computed from observations only — never from the protocol order or
    from which step is expected next. ``sources`` breaks it down per evidence source
    (e.g. ``{"rule": 0.91, "hoi": 0.88, "temporal": 0.74}``), each a probability.
    """

    frame_idx: int
    t: float
    step_id: str
    p_complete: float
    sources: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _set(self, "frame_idx", _frame_idx(self.frame_idx))
        _set(self, "t", _time(self.t))
        _set(self, "step_id", str(self.step_id))
        _set(self, "p_complete", _prob("p_complete", self.p_complete))
        _set(self, "sources", _prob_map("sources", self.sources))


@_register
@dataclass(frozen=True, slots=True)
class ActionScores:
    """Temporal-model scores over ``{step_id, ..., "idle"}`` for the step in progress."""

    frame_idx: int
    t: float
    scores: dict[str, float]

    def __post_init__(self) -> None:
        _set(self, "frame_idx", _frame_idx(self.frame_idx))
        _set(self, "t", _time(self.t))
        _set(self, "scores", _prob_map("scores", self.scores))

    def top(self) -> tuple[str, float]:
        return max(self.scores.items(), key=lambda kv: kv[1])


# --------------------------------------------------------------------------------------------
# Protocol engine outputs and operator inputs
# --------------------------------------------------------------------------------------------


@_register
@dataclass(frozen=True, slots=True)
class StepEvent:
    """A step's outcome changed. ``frame_idx``/``t`` are when the engine decided it.

    ``retroactive`` marks a status assigned after the fact (e.g. ``skipped`` once a successor
    is verified). ``t_start``/``t_end`` are None when unknown (e.g. a skipped step).
    """

    step_id: str
    status: StepStatus
    t_start: float | None
    t_end: float | None
    confidence: float
    evidence: dict[str, float] = field(default_factory=dict)
    frame_idx: int = field(kw_only=True)
    t: float = field(kw_only=True)
    retroactive: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        _set(self, "step_id", str(self.step_id))
        _set(self, "status", StepStatus(self.status))
        _set(self, "t_start", _opt_time(self.t_start, "t_start"))
        _set(self, "t_end", _opt_time(self.t_end, "t_end"))
        _set(self, "confidence", _prob("confidence", self.confidence))
        _set(self, "evidence", _prob_map("evidence", self.evidence))
        _set(self, "frame_idx", _frame_idx(self.frame_idx))
        _set(self, "t", _time(self.t))
        _set(self, "retroactive", bool(self.retroactive))


@_register
@dataclass(frozen=True, slots=True)
class Alert:
    kind: AlertKind
    message: str  # spoken and shown as-is
    step_ids: tuple[str, ...]
    frame_idx: int = field(kw_only=True)
    t: float = field(kw_only=True)

    def __post_init__(self) -> None:
        _set(self, "kind", AlertKind(self.kind))
        _set(self, "message", str(self.message))
        _set(self, "step_ids", tuple(str(s) for s in self.step_ids))
        _set(self, "frame_idx", _frame_idx(self.frame_idx))
        _set(self, "t", _time(self.t))


@_register
@dataclass(frozen=True, slots=True)
class Prompt:
    """Next-step suggestion. ``next_step_id`` is None once the procedure is complete."""

    next_step_id: str | None
    text: str
    frame_idx: int = field(kw_only=True)
    t: float = field(kw_only=True)

    def __post_init__(self) -> None:
        if self.next_step_id is not None:
            _set(self, "next_step_id", str(self.next_step_id))
        _set(self, "text", str(self.text))
        _set(self, "frame_idx", _frame_idx(self.frame_idx))
        _set(self, "t", _time(self.t))


@_register
@dataclass(frozen=True, slots=True)
class Ask:
    """A yes/no question to the operator, answered by an ``Answer`` with the same id."""

    question_id: str
    step_id: str
    text: str
    frame_idx: int = field(kw_only=True)
    t: float = field(kw_only=True)

    def __post_init__(self) -> None:
        _set(self, "question_id", str(self.question_id))
        _set(self, "step_id", str(self.step_id))
        _set(self, "text", str(self.text))
        _set(self, "frame_idx", _frame_idx(self.frame_idx))
        _set(self, "t", _time(self.t))


@_register
@dataclass(frozen=True, slots=True)
class Answer:
    question_id: str
    value: AnswerValue
    via: InputVia
    t: float = field(kw_only=True)
    frame_idx: int | None = field(default=None, kw_only=True)  # stamped by the receiver if known

    def __post_init__(self) -> None:
        _set(self, "question_id", str(self.question_id))
        _set(self, "value", AnswerValue(self.value))
        _set(self, "via", InputVia(self.via))
        _set(self, "t", _time(self.t))
        if self.frame_idx is not None:
            _set(self, "frame_idx", _frame_idx(self.frame_idx))


@_register
@dataclass(frozen=True, slots=True)
class Command:
    """A spoken or GUI command outside an Ask ("next", "repeat", "what's next", ...)."""

    name: CommandName
    via: InputVia
    t: float = field(kw_only=True)
    frame_idx: int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _set(self, "name", CommandName(self.name))
        _set(self, "via", InputVia(self.via))
        _set(self, "t", _time(self.t))
        if self.frame_idx is not None:
            _set(self, "frame_idx", _frame_idx(self.frame_idx))


@_register
@dataclass(frozen=True, slots=True)
class Stats:
    """Health / throughput numbers from one component, e.g. ``source="detector"``,
    ``values={"fps": 17.2, "dropped": 3}``. Topic ``stats``."""

    source: str
    values: dict[str, float | int | str | bool | None]
    t: float = field(kw_only=True)
    frame_idx: int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _set(self, "source", str(self.source))
        _set(self, "values", dict(self.values))
        _set(self, "t", _time(self.t))
        if self.frame_idx is not None:
            _set(self, "frame_idx", _frame_idx(self.frame_idx))


ENGINE_OUTPUTS = (StepEvent, Alert, Prompt, Ask)


# --------------------------------------------------------------------------------------------
# JSON codec
# --------------------------------------------------------------------------------------------


def to_jsonable(obj: Any) -> Any:
    """Convert a contract (or a container of them) to JSON-safe Python values.

    Dataclasses become dicts tagged ``"type": ClassName``; enums become their values; numpy
    arrays become nested lists; non-finite floats become None.
    """
    if isinstance(obj, FramePacket):
        raise TypeError("FramePacket carries an image and is not JSON-encoded")
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {"type": type(obj).__name__}
        for f in dataclasses.fields(obj):
            out[f.name] = to_jsonable(getattr(obj, f.name))
        return out
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def from_jsonable(data: Any) -> Any:
    """Inverse of ``to_jsonable``: rebuilds contracts from ``"type"``-tagged dicts."""
    if isinstance(data, Mapping):
        cls = _REGISTRY.get(data.get("type")) if isinstance(data.get("type"), str) else None
        decoded = {k: from_jsonable(v) for k, v in data.items() if not (cls and k == "type")}
        return cls(**decoded) if cls else decoded
    if isinstance(data, list):
        return [from_jsonable(v) for v in data]
    return data


def contract_types() -> Iterable[type]:
    """All JSON-encodable contract classes."""
    return tuple(_REGISTRY.values())
