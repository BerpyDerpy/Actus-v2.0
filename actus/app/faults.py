"""T-FAULT: fault injection for the ACTUS pipeline, and the safety net that answers it.

Five faults, each switched by a console hotkey, a GUI button, a schedule (``occlude@9-12``) or
code:

    occlude        a black box over the frame (default: all of it)               hotkey o
    dim            a gamma-darkened frame                                         hotkey d
    freeze         the camera keeps delivering its last frame                     hotkey f
    drop_detector  the detector stage stops ("did not run")                       hotkey x
    kill_network   the RTSP target goes away; recording carries on                hotkey n

The expected behaviour is always the same: **abstain or ask, or "stream down, still recording"
-- never a false ``verified``**. The faults are injected where a real failure would arrive, so
the real chain is what gets tested:

* image faults change the pixels the pipeline perceives (``PipelineCore.source_hooks``); the raw
  stream and recording keep the true frames, the annotated one shows the fault;
* on a source that supplies its own detections (the SIMULATED replay's stand-in for the detector)
  the same physical effect is applied to those detections too -- a real detector would see the
  black box or the dark frame in the pixels itself;
* ``drop_detector`` and the stale-frame guard switch the stage off (``core.disabled``) and cap
  the visibility the pipeline reports (``core.visibility_cap``). The cap pulls step evidence to
  0.5 and the engine's visibility to 0, exactly as the replay models an occlusion, so the engine
  cannot verify from a sensor that is not there and asks about the expected step instead.

``StaleFrameGuard`` is the defence against ``freeze``: it notices frames that repeat bit for bit
and suspends perception. It runs only while a ``FaultController`` is attached, because a video
file with duplicated frames is not a failure.

Everything a fault does is logged in ``runs/<id>/faults.jsonl`` -- the same SHA-256 chain as the
event log, plus the event log's chain tip at that moment, so the two files order against each
other and ``python -m actus.io.eventlog verify runs/<id>/faults.jsonl`` works on it.

Demo on a replayed take (offline, SIMULATED)::

    python -m actus.app.faults --demo all                  # every fault on the "correct" take
    python -m actus.app.faults --demo occlude --window 9-12
    python -m actus.app.pipeline --replay correct --fault occlude@9-12   # live app, with the GUI

The demo runs the take twice (fault-free baseline, then faulted) and checks the faulted run
against the replay's ground truth: no step verified that never happened, and no step verified
*while blind* for something that happened while blind.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import math
import os
import sys
import threading
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from typing import Protocol as TypingProtocol

import numpy as np
import yaml

from actus.core.bus import Bus, BusClosed
from actus.core.config import Config, load_config
from actus.core.contracts import (
    SCHEMA_VERSION,
    Alert,
    Ask,
    DetectionSet,
    FramePacket,
    Stats,
    StepEvent,
    StepStatus,
    Topic,
    to_jsonable,
)
from actus.io.eventlog import GENESIS_HASH, _hash_record, iso_at

logger = logging.getLogger("actus.app.faults")

FAULT_LOG = "faults.jsonl"
STALE = "stale_frames"  # the guard's "fault name" in the log and in stats
MANAGED_STAGES = frozenset({"detector", "handpose"})  # the stages a fault may switch off
DIM_BLIND_GAMMA = 3.0  # gamma at or above this counts as blinding the sensors
BLIND_COVER = 0.8  # an occlusion covering this fraction of the frame or more counts as blinding
DEMO_WINDOW = (9.0, 12.0)  # seconds; covers S3's placement (t = 10.0) in the "correct" replay
STALE_EXPECTED = (
    "Detector and hands/pose are suspended, visibility is forced to 0 and step evidence is "
    "uninformative, so nothing is verified from the repeated frame."
)


# --------------------------------------------------------------------------------------------
# The fault catalogue
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FaultInfo:
    name: str
    label: str  # shown as "FAULT INJECTED: <label>"
    key: str  # console / GUI hotkey
    summary: str
    expected: str  # the safe behaviour this fault must produce
    defaults: Mapping[str, Any] = field(default_factory=dict)


FAULTS: dict[str, FaultInfo] = {
    f.name: f
    for f in (
        FaultInfo(
            "occlude",
            "camera occluded",
            "o",
            "A black box over the frame (default: all of it).",
            "The rack and the objects go unseen, so scene visibility collapses. No step is "
            "verified from frames that could not see it; while a step is expected the engine "
            "asks the operator instead.",
            {"region": [0.0, 0.0, 1.0, 1.0], "max_covered": 0.5},
        ),
        FaultInfo(
            "dim",
            "lighting dimmed",
            "d",
            "A gamma-darkened frame (gamma > 1 is darker).",
            "Detection confidence falls and detections drop out, so evidence weakens and "
            "visibility falls. No step is verified from what the sensors could not see.",
            {"gamma": DIM_BLIND_GAMMA, "min_conf": None},
        ),
        FaultInfo(
            "freeze",
            "frame frozen",
            "f",
            "The camera keeps delivering its last frame.",
            "The stale-frame guard trips after a few identical frames: detector and hands/pose "
            "are suspended, visibility is forced to 0 and nothing is verified from the frozen "
            "scene.",
        ),
        FaultInfo(
            "drop_detector",
            "detector dropped",
            "x",
            "The detector stage stops and reports 'did not run'.",
            "Visibility is forced to 0 and step evidence is uninformative, so no step is "
            "verified; while a step is expected the engine asks the operator about it.",
        ),
        FaultInfo(
            "kill_network",
            "network target down",
            "n",
            "The RTSP target goes away.",
            "Stream down, still recording: the RTSP push stops and comes back when the target "
            "does; the recording, the event log and every engine output are unaffected.",
        ),
    )
}


def is_blinding(name: str, params: Mapping[str, Any]) -> bool:
    """Does this fault (with these params) take the sensors' view of the scene away?

    Only blinding faults are held to "nothing may be verified from it" by the oracle: a mild
    dim, or a box over a corner nobody is working in, leaves steps genuinely observable.
    """
    if name == "occlude":
        x0, y0, x1, y1 = _region(params.get("region", FAULTS["occlude"].defaults["region"]))
        return (x1 - x0) * (y1 - y0) >= BLIND_COVER
    if name == "dim":
        return float(params.get("gamma", DIM_BLIND_GAMMA)) >= DIM_BLIND_GAMMA
    return name in ("freeze", "drop_detector")


def _region(v: Any) -> tuple[float, float, float, float]:
    """``x0, y0, x1, y1`` as fractions of the frame."""
    if isinstance(v, str):
        v = v.split(",")
    try:
        x0, y0, x1, y1 = (float(a) for a in v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"region must be x0,y0,x1,y1 as fractions of the frame, got {v!r}") from e
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"region must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, got {v!r}")
    return x0, y0, x1, y1


def _covered(bbox: Sequence[float], box: tuple[int, int, int, int]) -> float:
    """Fraction of ``bbox``'s area lying inside ``box`` (pixels)."""
    x1, y1, x2, y2 = bbox
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area <= 0.0:
        return 0.0
    iw = max(0.0, min(x2, box[2]) - max(x1, box[0]))
    ih = max(0.0, min(y2, box[3]) - max(y1, box[1]))
    return iw * ih / area


@dataclass
class _Work:
    """What the faults transform for one frame."""

    img: np.ndarray
    dets: DetectionSet | None
    frame_idx: int
    t: float
    prev_img: np.ndarray | None  # what the pipeline was last given
    prev_dets: DetectionSet | None
    had_dets: bool  # the source supplied detections (a stand-in for the detector)


class _Fault:
    """One active fault. A subclass changes what the pipeline sees; the controller does the rest."""

    name = ""
    stages: frozenset[str] = frozenset()  # stages switched off while active
    cap: float | None = None  # visibility cap while active

    def __init__(self, params: Mapping[str, Any] | None = None, *, detector_conf: float = 0.25):
        allowed = FAULTS[self.name].defaults
        unknown = set(params or {}) - set(allowed)
        if unknown:
            raise ValueError(
                f"fault {self.name!r} has no parameter(s) {sorted(unknown)}; "
                f"it takes {sorted(allowed) or 'none'}"
            )
        self.params: dict[str, Any] = {**allowed, **(params or {})}
        self.detector_conf = float(detector_conf)

    def apply(self, w: _Work) -> None:
        """Transform ``w`` in place."""


class _Occlude(_Fault):
    name = "occlude"

    def __init__(self, params=None, *, detector_conf: float = 0.25):
        super().__init__(params, detector_conf=detector_conf)
        self.region = _region(self.params["region"])
        self.params["region"] = list(self.region)
        self.max_covered = float(self.params["max_covered"])
        if not 0.0 < self.max_covered <= 1.0:
            raise ValueError(f"max_covered must be in (0, 1], got {self.max_covered!r}")

    def apply(self, w: _Work) -> None:
        h, wd = w.img.shape[:2]
        x0, y0, x1, y1 = self.region
        box = (round(x0 * wd), round(y0 * h), round(x1 * wd), round(y1 * h))
        img = w.img.copy()
        patch = img[box[1] : box[3], box[0] : box[2]]  # a view into img
        # A covered lens is dark, not bit-identical from frame to frame: seeded sensor noise keeps
        # a blackout from looking like a frozen camera to the stale-frame guard, and keeps a
        # replay deterministic.
        patch[...] = np.random.default_rng(w.frame_idx).integers(
            0, 4, size=patch.shape, dtype=np.uint8
        )
        w.img = img
        if w.dets is not None and w.had_dets:  # what a real detector could no longer see
            kept = [d for d in w.dets.detections if _covered(d.bbox_xyxy, box) < self.max_covered]
            w.dets = DetectionSet(w.dets.frame_idx, w.dets.t, kept, w.dets.backend)


class _Dim(_Fault):
    name = "dim"

    def __init__(self, params=None, *, detector_conf: float = 0.25):
        super().__init__(params, detector_conf=detector_conf)
        self.gamma = float(self.params["gamma"])
        if not self.gamma > 0.0:
            raise ValueError(f"gamma must be > 0, got {self.gamma!r}")
        floor = self.params.get("min_conf")
        self.min_conf = self.detector_conf if floor is None else float(floor)
        self.params["min_conf"] = self.min_conf
        self._lut = np.round(255.0 * (np.arange(256) / 255.0) ** self.gamma).astype(np.uint8)

    def apply(self, w: _Work) -> None:
        import cv2

        w.img = cv2.LUT(w.img, self._lut)
        if w.dets is not None and w.had_dets:
            # Emulation for a stand-in detector: confidence falls with the light, and what falls
            # under the detector's own threshold is not reported at all.
            k = 0.5 ** (self.gamma - 1.0)
            kept = []
            for d in w.dets.detections:
                conf = min(1.0, d.conf * k)
                if conf >= self.min_conf:
                    kept.append(dataclasses.replace(d, conf=conf))
            w.dets = DetectionSet(w.dets.frame_idx, w.dets.t, kept, w.dets.backend)


class _Freeze(_Fault):
    name = "freeze"

    def apply(self, w: _Work) -> None:
        if w.prev_img is None:
            return  # nothing delivered yet to hold on to
        w.img = w.prev_img
        if w.had_dets:
            p = w.prev_dets
            w.dets = None if p is None else DetectionSet(w.frame_idx, w.t, p.detections, p.backend)


class _DropDetector(_Fault):
    name = "drop_detector"
    stages = frozenset({"detector"})
    cap = 0.0

    def apply(self, w: _Work) -> None:
        w.dets = None  # "did not run", also for a source that supplies its own detections


class _KillNetwork(_Fault):
    name = "kill_network"  # acts on the RTSP target, not on the frames


_FAULT_CLASSES: dict[str, type[_Fault]] = {
    c.name: c for c in (_Occlude, _Dim, _Freeze, _DropDetector, _KillNetwork)
}


def build_fault(
    name: str, params: Mapping[str, Any] | None = None, *, detector_conf: float = 0.25
) -> _Fault:
    """Construct (and thereby validate) a fault. Raises ``KeyError`` / ``ValueError``."""
    if name not in _FAULT_CLASSES:
        raise KeyError(f"unknown fault {name!r}; choose from {list(FAULTS)}")
    return _FAULT_CLASSES[name](params, detector_conf=detector_conf)


# --------------------------------------------------------------------------------------------
# The stale-frame guard: the system's own defence against `freeze`
# --------------------------------------------------------------------------------------------


class StaleFrameGuard:
    """Notices frames that repeat bit for bit.

    A live sensor never delivers the same pixels twice (noise), so ``stale_after`` repeats in a
    row mean the source is stuck. The first ``stale_after`` repeats still reach the pipeline;
    that latency (``stale_after`` frames) is the price of not mistaking a still scene for a fault.
    """

    def __init__(self, stale_after: int = 3) -> None:
        if stale_after < 1:
            raise ValueError("stale_after must be >= 1")
        self.stale_after = int(stale_after)
        self._prev: np.ndarray | None = None
        self.repeats = 0

    def update(self, img: np.ndarray) -> bool:
        """Feed the frame the pipeline is about to perceive; True while the source is stale."""
        prev = self._prev
        same = prev is not None and (
            img is prev or (img.shape == prev.shape and bool(np.array_equal(img, prev)))
        )
        self.repeats = self.repeats + 1 if same else 0
        self._prev = img
        return self.repeats >= self.stale_after

    def reset(self) -> None:
        self._prev, self.repeats = None, 0


# --------------------------------------------------------------------------------------------
# The RTSP target
# --------------------------------------------------------------------------------------------


class StreamTarget(TypingProtocol):
    """What ``kill_network`` acts on."""

    instant: bool  # cut/restore return at once (no worker thread needed)

    def cut(self) -> None: ...
    def restore(self) -> None: ...
    def status(self) -> dict[str, Any]: ...


class StreamerTarget:
    """The live RTSP push (T-IO's ``Streamer``). Cutting stops MediaMTX and both FFmpeg pushers,
    which is what a dead target looks like from here; restoring starts them again. The recorder,
    the event log and the GUI's MJPEG feed do not go through the streamer and are unaffected."""

    instant = False

    def __init__(self, streamer, fps: float) -> None:
        self.streamer = streamer
        self.fps = float(fps)
        self.down = False

    def cut(self) -> None:
        self.streamer.stop()
        self.down = True

    def restore(self) -> None:
        self.streamer.start(fps=self.fps)
        self.down = False

    def status(self) -> dict[str, Any]:
        return {"kind": "rtsp", "down": self.down}


class SimulatedTarget:
    """A stand-in RTSP target for offline demos, counting what it would have delivered.
    SIMULATED: no bytes go anywhere."""

    instant = True

    def __init__(self) -> None:
        self.down = False
        self.cuts = 0
        self.offered = 0
        self.delivered = 0
        self.dropped = 0

    def cut(self) -> None:
        self.down = True
        self.cuts += 1

    def restore(self) -> None:
        self.down = False

    def offer(self, frame_idx: int) -> None:
        self.offered += 1
        if self.down:
            self.dropped += 1
        else:
            self.delivered += 1

    def status(self) -> dict[str, Any]:
        return {
            "kind": "simulated",
            "down": self.down,
            "cuts": self.cuts,
            "offered": self.offered,
            "delivered": self.delivered,
            "dropped": self.dropped,
        }


# --------------------------------------------------------------------------------------------
# Schedules
# --------------------------------------------------------------------------------------------


def _parse_value(text: str) -> Any:
    text = text.strip()
    if "," in text:
        return [float(x) for x in text.split(",")]
    return yaml.safe_load(text)


@dataclass(frozen=True)
class ScheduledFault:
    """``name`` is on while ``start <= x < end``, ``x`` being seconds of pipeline time since the
    first frame (``unit="s"``) or the frame index (``unit="f"``)."""

    name: str
    start: float
    end: float = math.inf
    unit: str = "s"
    params: Mapping[str, Any] = field(default_factory=dict)

    def contains(self, x: float) -> bool:
        return self.start <= x < self.end

    def spec(self) -> str:
        end = "end" if math.isinf(self.end) else f"{self.end:g}"
        prm = ";".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}@{'f' if self.unit == 'f' else ''}{self.start:g}-{end}" + (
            f":{prm}" if prm else ""
        )


def parse_schedule_item(spec: str, *, detector_conf: float = 0.25) -> ScheduledFault:
    """``name@start-end[:key=value[;key=value...]]``; ``start``/``end`` in seconds of pipeline
    time, or frame indices with an ``f`` prefix (``dim@f90-135``). ``end`` may be empty or
    ``end``. A region is ``region=0,0,0.5,1``."""
    name, sep, rest = spec.strip().partition("@")
    if not sep or not name or not rest:
        raise ValueError(f"fault schedule must look like name@start-end, got {spec!r}")
    when, _, prm = rest.partition(":")
    unit = "s"
    if when.startswith("f"):
        unit, when = "f", when[1:]
    a, dash, b = when.partition("-")
    try:
        start = float(a)
        end = math.inf if b.strip() in ("", "end") else float(b)
    except ValueError as e:
        raise ValueError(f"bad time range in fault schedule {spec!r}") from e
    if not dash or not end > start >= 0.0:
        raise ValueError(f"fault schedule {spec!r}: need 0 <= start < end")
    params: dict[str, Any] = {}
    for item in filter(None, (p.strip() for p in prm.split(";"))):
        key, eq, raw = item.partition("=")
        if not eq or not key.strip():
            raise ValueError(f"fault schedule {spec!r}: parameter must be key=value, got {item!r}")
        params[key.strip()] = _parse_value(raw)
    build_fault(name, params, detector_conf=detector_conf)  # validates name and params
    return ScheduledFault(name, start, end, unit, params)


# --------------------------------------------------------------------------------------------
# The fault log
# --------------------------------------------------------------------------------------------


class _FaultLog:
    """Hash-chained JSONL of fault events. Same chain as ``events.jsonl`` (so ``verify_log``
    checks it), plus ``events_tip``: the event log's chain tip when the record was written.
    A run starts a fresh file; it never resumes an old one."""

    def __init__(self, run_dir: str | Path, *, clock=None, tip=None, filename: str = FAULT_LOG):
        self.path = Path(run_dir) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)
        self._clock, self._tip = clock, tip
        self._seq = 0
        self._prev = GENESIS_HASH

    def _now(self) -> str:
        if self._clock is not None:
            return iso_at(self._clock())
        return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def write(self, body: Mapping[str, Any]) -> dict[str, Any]:
        record = {
            "schema_version": SCHEMA_VERSION,
            "seq": self._seq,
            "ts": self._now(),
            **body,
            "events_tip": self._tip() if self._tip is not None else None,
            "prev_hash": self._prev,
        }
        record["hash"] = _hash_record(record)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._prev = record["hash"]
        self._seq += 1
        return record

    @property
    def tip(self) -> str:
        return self._prev


# --------------------------------------------------------------------------------------------
# The controller
# --------------------------------------------------------------------------------------------


class FaultController:
    """Switches faults on and off and applies them on the pipeline thread.

    ``attach(core)`` installs ``process`` as a ``PipelineCore.source_hooks`` entry. Requests from
    other threads (hotkeys, the GUI) are queued and take effect at the next frame, stamped with
    that frame's index and time, so a schedule and a replay are deterministic. A fault is active
    while it is switched on manually, or a scheduled window covers the frame.

        ctrl = FaultController(schedule=["occlude@9-12"]).attach(core)
        ctrl.open_log(run_dir, clock=clock, tip=lambda: eventlog.final_hash)
        ...                                   # run
        ctrl.close()
    """

    def __init__(
        self,
        *,
        schedule: Iterable[ScheduledFault | str] = (),
        defaults: Mapping[str, Mapping[str, Any]] | None = None,
        stale_after: int = 3,
        detector_conf: float = 0.25,
    ) -> None:
        self.detector_conf = float(detector_conf)
        self.schedule: tuple[ScheduledFault, ...] = tuple(
            s
            if isinstance(s, ScheduledFault)
            else parse_schedule_item(s, detector_conf=detector_conf)
            for s in schedule
        )
        self.defaults: dict[str, dict[str, Any]] = {n: {} for n in FAULTS}
        for name, params in (defaults or {}).items():
            if name in FAULTS:
                self.defaults[name] = dict(params or {})
                build_fault(name, self.defaults[name], detector_conf=detector_conf)
        self._guard = StaleFrameGuard(stale_after) if stale_after > 0 else None
        self._lock = threading.RLock()
        self._pending: list[tuple[str, bool | None, dict[str, Any]]] = []
        self._manual: dict[str, bool] = {}
        self._manual_params: dict[str, dict[str, Any]] = {}
        self._active: dict[str, _Fault] = {}
        self._window_of: dict[str, dict[str, Any]] = {}
        self._windows: list[dict[str, Any]] = []
        self._records: list[dict[str, Any]] = []
        self._refused: dict[str, str] = {}
        self._stale = False
        self._added: set[str] = set()
        self._my_tags: set[str] = set()
        self._prev_img: np.ndarray | None = None
        self._prev_dets: DetectionSet | None = None
        self._t0: float | None = None
        self._fi0 = 0
        self._last_fi: int | None = None
        self._last_t: float | None = None
        self._core: Any = None
        self._bus: Bus | None = None
        self._log: _FaultLog | None = None
        self._simulated = False
        self._target: StreamTarget | None = None
        self._net_state = "n/a"
        self._net_error: str | None = None
        self._pool: ThreadPoolExecutor | None = None

    @classmethod
    def from_config(cls, cfg: Config, *, schedule: Iterable[ScheduledFault | str] = ()):
        """Build from the ``faults:`` section (and ``detector.conf``), plus extra schedule items."""
        fc = cfg.get("faults") or {}
        conf = float(cfg.get("detector.conf", 0.25))
        defaults = {n: dict(fc.get(n) or {}) for n in FAULTS}
        return cls(
            schedule=[*(fc.get("schedule") or []), *schedule],
            defaults=defaults,
            stale_after=int(fc.get("stale_frames", 3)),
            detector_conf=conf,
        )

    # -- wiring -------------------------------------------------------------------------------

    def attach(self, core) -> FaultController:
        """Install as a source hook on a ``PipelineCore`` (idempotent)."""
        self._core = core
        self._simulated = bool(getattr(core, "simulated", False))
        if self.process not in core.source_hooks:
            core.source_hooks.append(self.process)
        if getattr(core, "evidence_mode", "spatial") == "replay":
            logger.warning(
                "faults: evidence_mode=replay feeds the stand-in evidence straight to the engine, "
                "so only drop_detector and freeze (visibility cap) reach it"
            )
        return self

    def bind_bus(self, bus: Bus) -> FaultController:
        self._bus = bus
        return self

    def open_log(self, run_dir: str | Path, *, clock=None, tip=None) -> Path:
        """Start ``<run_dir>/faults.jsonl``. ``clock`` (POSIX seconds) replaces the wall clock
        for ``ts``; ``tip`` returns the event log's current chain tip."""
        with self._lock:
            self._log = _FaultLog(run_dir, clock=clock, tip=tip)
            for rec in self._records:  # anything that happened before the log existed
                self._log.write(rec)
            return self._log.path

    def set_network_target(self, target: StreamTarget | None) -> None:
        with self._lock:
            self._target = target
            self._net_state = "n/a" if target is None else "up"

    # -- requests (any thread) ----------------------------------------------------------------

    def request(self, name: str, on: bool | None = None, params: Mapping[str, Any] | None = None):
        """Switch ``name`` on/off (``on=None`` toggles) at the next frame. Raises ``KeyError`` for
        an unknown fault and ``ValueError`` for bad params, in the caller's thread."""
        merged = {**self.defaults.get(name, {}), **(params or {})}
        build_fault(name, merged, detector_conf=self.detector_conf)
        with self._lock:
            self._pending.append((name, on, dict(params or {})))

    # -- per frame (pipeline thread) ----------------------------------------------------------

    def process(self, sf):
        """The source hook: apply the active faults to one ``SourceFrame``."""
        pkt = sf.packet
        fi, t = pkt.frame_idx, pkt.t_capture
        if self._t0 is None:
            self._t0, self._fi0 = t, fi
        self._last_fi, self._last_t = fi, t
        changed = self._advance(fi, t)

        w = _Work(
            pkt.image,
            sf.detections,
            fi,
            t,
            self._prev_img,
            self._prev_dets,
            had_dets=sf.detections is not None,
        )
        for name in FAULTS:  # fixed order: occlude, dim, freeze, drop_detector
            fault = self._active.get(name)
            if fault is not None:
                fault.apply(w)
        if self._guard is not None:
            stale = self._guard.update(w.img)
            if stale != self._stale:
                self._set_stale(stale, fi, t)
                changed = True
            if stale and w.had_dets:
                w.dets = None
        self._prev_img, self._prev_dets = w.img, w.dets

        self._sync(*self._down())
        if changed:
            self._publish_stats()
        offer = getattr(self._target, "offer", None)
        if offer is not None:
            offer(fi)
        if w.img is pkt.image and w.dets is sf.detections:
            return sf
        packet = FramePacket(fi, t, w.img, pkt.source_id, pkt.rotation_applied)
        return dataclasses.replace(sf, packet=packet, detections=w.dets)

    def _advance(self, fi: int, t: float) -> bool:
        """Apply queued requests and the schedule; True if a fault started or stopped."""
        with self._lock:
            pending, self._pending = self._pending, []
        for name, on, params in pending:
            new = (not self._manual.get(name, False)) if on is None else bool(on)
            self._manual[name] = new
            if new:
                self._manual_params[name] = params
        assert self._t0 is not None
        desired: dict[str, tuple[dict[str, Any], str]] = {}
        for name in FAULTS:
            if self._manual.get(name):
                desired[name] = (self._manual_params.get(name, {}), "manual")
                continue
            for sch in self.schedule:
                x = t - self._t0 if sch.unit == "s" else fi - self._fi0
                if sch.name == name and sch.contains(x):
                    desired[name] = (dict(sch.params), "schedule")
                    break
        for name in [n for n in self._refused if n not in desired]:
            del self._refused[name]  # a refusal lasts only while it is still being asked for
        changed = False
        for name in FAULTS:
            if name in self._active and name not in desired:
                self._stop(name, fi, t)
                changed = True
            elif name in desired and name not in self._active and name not in self._refused:
                changed |= self._start(name, *desired[name], fi, t)
        return changed

    def _start(self, name: str, params: dict[str, Any], source: str, fi: int, t: float) -> bool:
        merged = {**self.defaults.get(name, {}), **params}
        fault = build_fault(name, merged, detector_conf=self.detector_conf)
        blinding = is_blinding(name, fault.params)
        if name == "kill_network":
            ok, why = self._net_cut()
            if not ok:
                self._refused[name] = why
                self._record("unavailable", name, fi, t, source, fault.params, blinding, why)
                return True
        with self._lock:
            self._active[name] = fault
            window = {
                "fault": name,
                "label": FAULTS[name].label,
                "params": dict(fault.params),
                "source": source,
                "blinding": blinding,
                "frame_start": fi,
                "t_start": t,
                "frame_end": None,
                "t_end": None,
            }
            self._windows.append(window)
            self._window_of[name] = window
        self._record("start", name, fi, t, source, fault.params, blinding)
        return True

    def _stop(
        self, name: str, fi: int, t: float, *, reason: str | None = None, restore: bool = True
    ) -> None:
        if name == "kill_network" and restore:
            self._net_restore()
        with self._lock:
            fault = self._active.pop(name)
            window = self._window_of.pop(name)
            window["frame_end"], window["t_end"] = fi, t
        self._record(
            "stop", name, fi, t, window["source"], fault.params, window["blinding"], reason
        )

    def _set_stale(self, stale: bool, fi: int, t: float, reason: str | None = None) -> None:
        self._stale = stale
        self._record("detected" if stale else "cleared", STALE, fi, t, "guard", {}, True, reason)

    def _down(self) -> tuple[frozenset[str], float | None]:
        """Stages that must be off and the visibility cap, from everything active."""
        stages: set[str] = set()
        caps: list[float] = []
        for fault in self._active.values():
            stages |= fault.stages
            if fault.cap is not None:
                caps.append(fault.cap)
        if self._stale:
            stages |= MANAGED_STAGES
            caps.append(0.0)
        return frozenset(stages), (min(caps) if caps else None)

    def _tags(self) -> set[str]:
        tags = {f"FAULT INJECTED: {FAULTS[n].label}" for n in self._active}
        if self._stale:
            tags.add("FAULT DETECTED: stale frames")
        return tags

    def _sync(self, stages: frozenset[str], cap: float | None) -> None:
        """Push the outage state onto the core, releasing only what this controller added."""
        core = self._core
        if core is None:
            return
        add = stages - core.disabled
        core.disabled |= add
        self._added |= add
        release = self._added - stages
        core.disabled -= release
        self._added -= release
        core.visibility_cap = cap
        tags = self._tags()
        if tags != self._my_tags:
            core.tags.difference_update(self._my_tags)
            core.tags.update(tags)
            self._my_tags = tags

    # -- the network fault --------------------------------------------------------------------

    def _net_cut(self) -> tuple[bool, str]:
        target = self._target
        if target is None:
            return False, "no RTSP streamer is running (use --live with the stream sink)"
        self._net_state = "cutting"
        self._net_task("cut", target)
        return True, ""

    def _net_restore(self) -> None:
        target = self._target
        if target is not None:
            self._net_state = "restoring"
            self._net_task("restore", target)

    def _net_task(self, action: str, target: StreamTarget) -> None:
        def run() -> None:
            try:
                getattr(target, action)()
                self._net_state = "down" if action == "cut" else "up"
                self._net_error = None
            except Exception as exc:  # the fault must never take the pipeline down
                self._net_state = "error"
                self._net_error = f"{type(exc).__name__}: {exc}"
                logger.exception("network fault: %s failed", action)
            self._publish_stats()

        if getattr(target, "instant", False):
            run()
            return
        if self._pool is None:  # one worker: cut and restore run in the order they were asked
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fault-net")
        self._pool.submit(run)

    # -- logging and stats --------------------------------------------------------------------

    def _record(
        self,
        action: str,
        name: str,
        fi: int,
        t: float,
        source: str,
        params: Mapping[str, Any],
        blinding: bool,
        reason: str | None = None,
    ) -> None:
        info = FAULTS.get(name)
        rec: dict[str, Any] = {
            "type": "FaultEvent",
            "action": action,
            "fault": name,
            "label": info.label if info else name.replace("_", " "),
            "source": source,
            "params": to_jsonable(dict(params)),
            "blinding": blinding,
            "expected": info.expected if info else STALE_EXPECTED,
            "simulated": self._simulated,
            "frame_idx": fi,
            "t": t,
        }
        if reason:
            rec["reason"] = reason
        with self._lock:
            self._records.append(rec)
            if self._log is not None:
                try:
                    self._log.write(rec)
                except OSError:  # a full disk must not take the pipeline down with it
                    logger.exception("could not write %s", FAULT_LOG)
        logger.warning(
            "FAULT %s: %s at frame %d (t=%.2f s)%s",
            action.upper(),
            name,
            fi,
            t,
            f" - {reason}" if reason else "",
        )

    def stats(self) -> Stats:
        """Shown by the GUI's stats strip; the first key says whether a fault is injected."""
        with self._lock:
            active = [n for n in FAULTS if n in self._active]
            values: dict[str, float | int | str | bool | None] = {
                "injected": bool(active),
                "active": ",".join(active) or "none",
                "detected": STALE if self._stale else "none",
                "network": self._net_state,
                "windows": len(self._windows),
                "pending": len(self._pending),
            }
        return Stats("faults", values, t=self._last_t or 0.0, frame_idx=self._last_fi)

    def _publish_stats(self) -> None:
        if self._bus is not None:
            with contextlib.suppress(BusClosed):
                self._bus.publish(Topic.STATS, self.stats())

    def state(self) -> dict[str, Any]:
        """Everything the GUI panel needs (``GET /api/faults``)."""
        with self._lock:
            return {
                "available": True,
                "simulated": self._simulated,
                "faults": [
                    {
                        "name": i.name,
                        "label": i.label,
                        "key": i.key,
                        "summary": i.summary,
                        "expected": i.expected,
                        "defaults": {**i.defaults, **self.defaults.get(i.name, {})},
                        "active": i.name in self._active,
                        "refused": self._refused.get(i.name),
                    }
                    for i in FAULTS.values()
                ],
                "active": [n for n in FAULTS if n in self._active],
                "detected": [STALE] if self._stale else [],
                "network": {
                    "state": self._net_state,
                    "error": self._net_error,
                    "target": None if self._target is None else self._target.status(),
                },
                "guard": {
                    "enabled": self._guard is not None,
                    "stale_after": None if self._guard is None else self._guard.stale_after,
                },
                "windows": [dict(w) for w in self._windows[-20:]],
            }

    def windows(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(w) for w in self._windows]

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._records]

    def summary(self) -> dict[str, Any]:
        """For ``run.json``."""
        with self._lock:
            return {
                "simulated": self._simulated,
                "windows": [dict(w) for w in self._windows],
                "guard_stale_after": None if self._guard is None else self._guard.stale_after,
                "network": self._net_state,
                "log": FAULT_LOG if self._log is not None else None,
                "log_hash": self._log.tip if self._log is not None else None,
            }

    def close(self, *, restore_network: bool = True) -> dict[str, Any]:
        """End of run: stop what is still active (logged as ``end_of_run``), hand the core back
        clean and let any network worker finish. Safe to call twice."""
        fi, t = self._last_fi, self._last_t
        if fi is not None and t is not None:
            for name in [n for n in FAULTS if n in self._active]:
                self._stop(name, fi, t, reason="end_of_run", restore=restore_network)
            if self._stale:
                self._set_stale(False, fi, t, reason="end_of_run")
        self._sync(*self._down())
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        return self.summary()


# --------------------------------------------------------------------------------------------
# Console hotkeys and GUI routes
# --------------------------------------------------------------------------------------------


class HotkeyListener:
    """Single-key toggles on the console running the pipeline: o d f x n (see ``FAULTS``).
    Does nothing when stdin is not a terminal."""

    def __init__(self, controller: FaultController) -> None:
        self.controller = controller
        self._keys = {info.key: name for name, info in FAULTS.items()}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def usable() -> bool:
        return sys.stdin is not None and sys.stdin.isatty()

    def start(self) -> bool:
        if not self.usable():
            logger.info("fault hotkeys off: stdin is not a terminal")
            return False
        keys = " ".join(f"{i.key}={n}" for n, i in FAULTS.items())
        logger.info("fault hotkeys: %s", keys)
        self._thread = threading.Thread(target=self._run, name="fault-hotkeys", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _handle(self, ch: str) -> None:
        name = self._keys.get(ch.lower())
        if name is not None:
            logger.warning("fault hotkey %r: toggling %s", ch, name)
            self.controller.request(name)

    def _run(self) -> None:
        try:
            if os.name == "nt":
                self._run_windows()
            else:
                self._run_posix()
        except Exception:
            logger.exception("fault hotkeys stopped")

    def _run_windows(self) -> None:
        import msvcrt

        while not self._stop.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):  # the first half of an arrow / function key
                    msvcrt.getwch()
                    continue
                self._handle(ch)
            else:
                self._stop.wait(0.05)

    def _run_posix(self) -> None:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)  # keeps Ctrl-C working
            while not self._stop.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready:
                    ch = sys.stdin.read(1)
                    if ch:
                        self._handle(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def mount_fault_routes(app, controller: FaultController) -> None:
    """``GET /api/faults`` (state) and ``POST /api/faults/{name}?on=true|false`` (omit ``on`` to
    toggle). Takes effect at the next frame."""
    from fastapi import HTTPException

    @app.get("/api/faults")
    def get_faults() -> Any:
        return controller.state()

    @app.post("/api/faults/{name}")
    def set_fault(name: str, on: bool | None = None) -> Any:
        try:
            controller.request(name, on)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        return {"ok": True, "fault": name, "on": on, "queued": True}


# --------------------------------------------------------------------------------------------
# The safety oracle, on a replay with known ground truth
# --------------------------------------------------------------------------------------------


def truth_completion(scenario, protocol) -> dict[str, float | None]:
    """When each step's transition really completed in a replay scenario (its last occurrence),
    in the replay's own seconds; ``None`` for a step that never happened."""
    out: dict[str, float | None] = dict.fromkeys(protocol.step_ids)
    for action, s0, s1 in scenario.spans():
        if action.label in out and action.changes:
            out[action.label] = s0 + max(frac for frac, _ in action.changes) * (s1 - s0)
    return out


def evaluate_safety(
    outputs: Sequence[Any],
    *,
    truth: Mapping[str, float | None],
    windows: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Judge an engine output stream against ground truth and the fault windows.

    * ``false_verified``: steps ``verified`` that never happened;
    * ``fabricated``: a step ``verified`` *inside* a blinding window although its transition
      also completed inside that window -- the system claimed to have seen what it could not.
      (A step that completed just before the window, or is verified after it, is fine.)
    """
    events = [o for o in outputs if isinstance(o, StepEvent)]
    verified = [o for o in events if o.status is StepStatus.VERIFIED]
    false_verified = sorted({o.step_id for o in verified if truth.get(o.step_id) is None})
    final = {r["step_id"]: r["status"] for r in outcomes}
    asks = [o for o in outputs if isinstance(o, Ask)]
    fabricated: list[dict[str, Any]] = []
    in_window: list[dict[str, Any]] = []
    for w in windows:
        t0 = float(w["t_start"])
        t1 = math.inf if w["t_end"] is None else float(w["t_end"])
        for sid, c in truth.items():
            if c is None or not t0 <= c <= t1:
                continue
            decided = [
                e
                for e in events
                if e.step_id == sid
                and e.status in (StepStatus.VERIFIED, StepStatus.OPERATOR_CONFIRMED)
            ]
            first = decided[0] if decided else None
            in_window.append(
                {
                    "fault": w["fault"],
                    "step_id": sid,
                    "truth_t": round(c, 3),
                    "final_status": final.get(sid),
                    "asked": any(a.step_id == sid for a in asks),
                    "decided_as": None if first is None else first.status.value,
                    "decided_t": None if first is None else round(first.t, 3),
                }
            )
            if w["blinding"]:
                fabricated += [
                    {
                        "fault": w["fault"],
                        "step_id": sid,
                        "truth_t": round(c, 3),
                        "verified_t": round(e.t, 3),
                        "window": [t0, None if math.isinf(t1) else t1],
                    }
                    for e in verified
                    if e.step_id == sid and t0 <= e.t <= t1
                ]
    return {
        "false_verified": false_verified,
        "fabricated": fabricated,
        "steps_in_window": in_window,
    }


def _statuses(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {r["step_id"]: r["status"] for r in outcomes}


def _frames_recorded(run_dir: Path, stream: str = "annotated") -> int:
    n = 0
    for p in sorted((run_dir / "video").glob(f"{stream}_*.frames.csv")):
        with open(p, encoding="utf-8") as f:
            n += max(0, sum(1 for _ in f) - 1)  # minus the header
    return n


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def run_fault_demo(
    fault: str,
    *,
    scenario: str = "correct",
    window: tuple[float, float] = DEMO_WINDOW,
    params: Mapping[str, Any] | None = None,
    cfg: Config | None = None,
    run_id: str | None = None,
    answer: str | None = "yes",
    force: bool = False,
) -> dict[str, Any]:
    """Run a replayed take twice -- fault-free, then with ``fault`` injected over ``window``
    (seconds) -- and judge the faulted run against the replay's ground truth.

    SIMULATED throughout: replayed frames, the replay's stand-in detector (degraded by the same
    physical effect as the pixels) and, for ``kill_network``, a ``SimulatedTarget``. Writes
    ``faults_report.json`` / ``.md`` and ``faults.jsonl`` in the faulted run's directory.
    ``answer`` scripts the operator's reply to any Ask ("yes" = they did do the step).
    """
    from actus.app.pipeline import new_run_id, run_offline
    from actus.capture.replay import get_scenario
    from actus.io.eventlog import verify_log
    from actus.protocol.engine import describe
    from actus.protocol.schema import load_protocol

    if fault not in FAULTS:
        raise KeyError(f"unknown fault {fault!r}; choose from {list(FAULTS)}")
    info = FAULTS[fault]
    cfg = cfg if cfg is not None else load_config(overrides=["models.enabled=false"])
    rid = run_id or new_run_id(f"fault-{fault}-")
    truth = truth_completion(get_scenario(scenario), load_protocol(cfg.protocol.path))
    common: dict[str, Any] = {
        "models": None,
        "evidence": "spatial",
        "answer": answer,
        "force": force,
    }

    baseline = run_offline(f"replay:{scenario}", cfg, run_id=f"{rid}-baseline", **common)
    ctrl = FaultController.from_config(
        cfg, schedule=[ScheduledFault(fault, window[0], window[1], "s", dict(params or {}))]
    )
    target = SimulatedTarget()
    ctrl.set_network_target(target)
    faulted = run_offline(f"replay:{scenario}", cfg, run_id=rid, faults=ctrl, **common)

    windows = ctrl.windows()
    records = ctrl.records()
    safety = evaluate_safety(
        faulted.summary.outputs,
        truth=truth,
        windows=windows,
        outcomes=faulted.summary.outcomes,
    )
    blinding = any(w["blinding"] for w in windows)
    checks = [
        _check(
            "no_false_verified",
            not safety["false_verified"],
            f"steps verified that never happened: {safety['false_verified'] or 'none'}",
        ),
        _check(
            "no_fabricated_verification",
            not safety["fabricated"],
            "nothing verified inside the blind window for a transition that happened inside it"
            if blinding
            else "not applicable: this fault does not blind the sensors",
        ),
        _check(
            "fault_window_logged",
            any(r["action"] == "start" for r in records)
            and any(r["action"] == "stop" for r in records),
            f"{len(records)} fault records in {FAULT_LOG}",
        ),
        _check(
            "logs_verify",
            verify_log(faulted.events_path).ok
            and verify_log(faulted.run_dir / FAULT_LOG).ok,
            "events.jsonl and faults.jsonl hash chains intact",
        ),
    ]
    extra: dict[str, Any] = {}
    if fault == "freeze":
        tripped = [r for r in records if r["action"] == "detected"]
        checks.append(
            _check(
                "guard_tripped",
                bool(tripped),
                f"stale-frame guard detected the freeze at frame "
                f"{tripped[0]['frame_idx'] if tripped else '-'}",
            )
        )
    if fault == "kill_network":
        recorded = _frames_recorded(faulted.run_dir)
        same = [to_jsonable(o) for o in faulted.summary.outputs] == [
            to_jsonable(o) for o in baseline.summary.outputs
        ]
        checks += [
            _check("engine_unaffected", same, "every engine output equals the fault-free run's"),
            _check(
                "recording_continued",
                recorded == faulted.summary.frames,
                f"{recorded} annotated frames recorded of {faulted.summary.frames} processed",
            ),
            _check(
                "target_saw_the_outage",
                target.dropped > 0,
                f"SIMULATED target dropped {target.dropped} of {target.offered} frames",
            ),
        ]
        extra["target"] = target.status()
        extra["recording"] = {
            "frames_recorded": recorded,
            "frames_processed": faulted.summary.frames,
        }

    base_alerts = [describe(o) for o in baseline.summary.outputs if isinstance(o, Alert)]
    alerts = [describe(o) for o in faulted.summary.outputs if isinstance(o, Alert)]
    report: dict[str, Any] = {
        "fault": fault,
        "label": info.label,
        "scenario": scenario,
        "simulated": True,
        "window_s": list(window),
        "params": windows[0]["params"] if windows else dict(params or {}),
        "answer_policy": answer,
        "expected": info.expected,
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "baseline": {"statuses": _statuses(baseline.summary.outcomes), "alerts": base_alerts},
        "observed": {
            "statuses": _statuses(faulted.summary.outcomes),
            "alerts": alerts,
            "alerts_added_by_the_fault": [a for a in alerts if a not in base_alerts],
            "asks": [o.text for o in faulted.summary.outputs if isinstance(o, Ask)],
            "steps_in_window": safety["steps_in_window"],
            "fabricated": safety["fabricated"],
            **extra,
        },
        "run_dir": str(faulted.run_dir),
        "baseline_run_dir": str(baseline.run_dir),
        "note": (
            "SIMULATED: replayed frames; the replay's detector stand-in is degraded by the same "
            "effect as the pixels. Judged against the replay's ground truth."
        ),
    }
    (faulted.run_dir / "faults_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (faulted.run_dir / "faults_report.md").write_text(render_markdown([report]), encoding="utf-8")
    return report


DEMO_PLAN: tuple[tuple[str, dict[str, Any]], ...] = (
    ("occlude", {}),
    ("dim", {"gamma": 4.0}),
    ("freeze", {}),
    ("drop_detector", {}),
    ("kill_network", {}),
)


def run_fault_montage(
    faults: Sequence[str] | None = None,
    *,
    scenario: str = "correct",
    window: tuple[float, float] = DEMO_WINDOW,
    cfg: Config | None = None,
    run_id: str | None = None,
    answer: str | None = "yes",
    force: bool = False,
) -> dict[str, Any]:
    """``run_fault_demo`` for each fault (default: all five) plus one combined report in
    ``runs/<run_id>/faults_report.{json,md}``."""
    from actus.app.pipeline import new_run_id, run_dir_for

    cfg = cfg if cfg is not None else load_config(overrides=["models.enabled=false"])
    rid = run_id or new_run_id("faults-")
    plan = [(n, p) for n, p in DEMO_PLAN if faults is None or n in faults]
    reports = [
        run_fault_demo(
            name,
            scenario=scenario,
            window=window,
            params=params,
            cfg=cfg,
            run_id=f"{rid}-{name}",
            answer=answer,
            force=force,
        )
        for name, params in plan
    ]
    combined = {
        "scenario": scenario,
        "simulated": True,
        "passed": all(r["passed"] for r in reports),
        "reports": reports,
    }
    out = run_dir_for(cfg, rid)
    out.mkdir(parents=True, exist_ok=True)
    (out / "faults_report.json").write_text(
        json.dumps(combined, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (out / "faults_report.md").write_text(render_markdown(reports), encoding="utf-8")
    combined["report_dir"] = str(out)
    return combined


def render_markdown(reports: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# ACTUS fault injection report",
        "",
        "SIMULATED: replayed takes, judged against the replay's ground truth. The expected "
        "behaviour for every fault is: abstain or ask, or 'stream down, still recording' -- "
        "never a false `verified`.",
        "",
        "| fault | window (s) | result | steps that completed in the window | asks |",
        "|---|---|---|---|---|",
    ]
    for r in reports:
        steps = "; ".join(
            f"{s['step_id']}: {s['final_status']}" + (" (asked)" if s["asked"] else "")
            for s in r["observed"]["steps_in_window"]
        )
        window = f"{r['window_s'][0]:g}-{r['window_s'][1]:g}"
        lines.append(
            f"| {r['fault']} | {window} | {'PASS' if r['passed'] else 'FAIL'} | "
            f"{steps or '-'} | {len(r['observed']['asks'])} |"
        )
    for r in reports:
        lines += ["", f"## {r['fault']} ({r['label']})", "", f"Expected: {r['expected']}", ""]
        lines += [
            f"- {'PASS' if c['passed'] else 'FAIL'} `{c['name']}`: {c['detail']}"
            for c in r["checks"]
        ]
        obs = r["observed"]
        lines += [
            "",
            f"- baseline statuses: {r['baseline']['statuses']}",
            f"- faulted statuses: {obs['statuses']}",
            f"- asks: {obs['asks'] or 'none'}",
            f"- alerts added by the fault: {obs['alerts_added_by_the_fault'] or 'none'}",
            f"- run: `{r['run_dir']}`",
        ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ACTUS fault injection: list faults, run the demo.")
    ap.add_argument("--list", action="store_true", help="list the faults and their safe behaviour")
    ap.add_argument("--demo", metavar="FAULT|all", help="run a fault demo on a replayed take")
    ap.add_argument("--scenario", default="correct", help="replay scenario (default: correct)")
    ap.add_argument("--window", default="9-12", help="fault window in seconds, e.g. 9-12")
    ap.add_argument("--param", action="append", default=[], help="fault parameter key=value")
    ap.add_argument("--answer", choices=["yes", "no", "none"], default="yes")
    ap.add_argument("--run-id")
    ap.add_argument("--force", action="store_true", help="overwrite existing run directories")
    ap.add_argument("--override", action="append", default=[], help="config key.path=value")
    args = ap.parse_args(argv)

    if args.list or not args.demo:
        for name, info in FAULTS.items():
            print(f"{name:14s} key={info.key}  {info.summary}")
            print(f"{'':14s} expected: {info.expected}")
        return 0

    cfg = load_config(overrides=["models.enabled=false", *args.override])
    a, _, b = args.window.partition("-")
    window = (float(a), float(b))
    answer = None if args.answer == "none" else args.answer
    params = {}
    for item in args.param:
        key, _, raw = item.partition("=")
        params[key.strip()] = _parse_value(raw)

    if args.demo == "all":
        combined = run_fault_montage(
            scenario=args.scenario,
            window=window,
            cfg=cfg,
            run_id=args.run_id,
            answer=answer,
            force=args.force,
        )
        reports = combined["reports"]
        print(f"# report: {combined['report_dir']}")
    else:
        reports = [
            run_fault_demo(
                args.demo,
                scenario=args.scenario,
                window=window,
                params=params,
                cfg=cfg,
                run_id=args.run_id,
                answer=answer,
                force=args.force,
            )
        ]
    for r in reports:
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['fault']:14s} [SIMULATED]  {r['run_dir']}")
        for c in r["checks"]:
            print(f"      {'ok  ' if c['passed'] else 'FAIL'} {c['name']}: {c['detail']}")
    return 0 if all(r["passed"] for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
