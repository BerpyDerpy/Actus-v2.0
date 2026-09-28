"""End to end: capture -> perception -> models -> spatial -> engine -> voice/log/UI/RTSP/MP4.

Three layers, so the same code runs live, offline and in tests:

* ``PipelineCore.step(frame)`` is one synchronous, deterministic call per frame. It runs the
  perception stages, the model stack, the spatial estimator and the protocol engine, and returns
  a ``FrameResult`` holding every message produced and an annotated frame. It does no I/O and
  owns no threads. The engine is only ever driven from here, so it stays single-threaded.
* A **source** yields ``SourceFrame``s: ``CameraSource`` (webcam / RTSP; threaded, drops the
  oldest frame when processing falls behind and counts it), ``FileSource`` (sequential decode,
  container clock, never drops) and ``ReplayFrameSource`` (the P0 synthetic world, SIMULATED).
* ``Runner`` drives a source through the core: it applies operator input (bus ``answers`` /
  ``commands`` or a scripted policy), publishes to the bus, writes the event log and publishes
  ``stats``. ``LiveApp`` adds the sinks (voice, recorder, RTSP streamer, GUI server) around a
  Runner and writes the downlink packet at the end. ``tools/run_offline.py`` is the same Runner
  with no pacing, no sinks and a log clock derived from the frames, so it is reproducible.

Plugging in a better model
--------------------------
Every perception stage is a plain object, chosen in the ``pipeline:`` config section:

* ``pipeline.detector``: ``config`` (the T-PERC ``Detector`` per ``detector.backend``), ``none``,
  or ``"package.module:factory"``. A factory takes the ``Config`` and returns an object with
  ``detect(FramePacket) -> DetectionSet`` (optionally ``reset()`` and ``backend_name``).
* ``pipeline.handpose``: ``config`` (T-PERC ``RotationTTA``), ``none``, or a factory returning an
  object with ``process(FramePacket) -> (HandSet | None, PoseSet | None)``.
* ``pipeline.rack``: ``aruco`` (measured ``configs/rack.yaml``) or ``selfcal`` (T-HOI-TEMP's
  self-calibrating frame, for rigs that were never measured; protocol zones won't line up).
* HOI and the temporal model come from ``ModelStack`` (``models:`` section): drop new weights at
  the configured path stems and they load; missing files degrade to rule evidence.

The engine stays the judge whatever is plugged in: models only reach it as evidence.

    python -m actus.app.pipeline --replay skip_S2          # SIMULATED, GUI on :8000
    python -m actus.app.pipeline --live                    # camera per capture.source
    python -m actus.app.pipeline --file data/raw/op01_correct_01.mp4 --no-stream
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import importlib
import json
import logging
import signal
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from typing import Protocol as TypingProtocol

import numpy as np

from actus.core.bus import Bus, BusClosed, Subscription
from actus.core.config import Config, load_config, resolve_path
from actus.core.contracts import (
    ActionScores,
    Alert,
    Answer,
    AnswerValue,
    Ask,
    Command,
    DetectionSet,
    FramePacket,
    HandSet,
    InputVia,
    PoseSet,
    Predicates,
    Prompt,
    RackFrame,
    Stats,
    StepEvent,
    StepEvidence,
    Topic,
)
from actus.protocol.decision import DecisionParams
from actus.protocol.engine import OUTPUT_TOPICS, Output, ProtocolEngine
from actus.protocol.schema import Protocol, load_protocol

logger = logging.getLogger("actus.app.pipeline")

STAGES = ("rack", "detector", "handpose", "models", "spatial", "engine", "overlay")
ALERT_SHOW_S = 6.0  # how long the overlay keeps the latest alert on screen (pipeline seconds)
_FROM_CONFIG: Any = object()  # sentinel: build this stage from the config


# --------------------------------------------------------------------------------------------
# Stage interfaces and plug-ins
# --------------------------------------------------------------------------------------------


class DetectorStage(TypingProtocol):
    def detect(self, frame: FramePacket) -> DetectionSet: ...


class HandPoseStage(TypingProtocol):
    def process(self, frame: FramePacket) -> tuple[HandSet | None, PoseSet | None]: ...


class RackStage(TypingProtocol):
    def update(self, frame: FramePacket) -> RackFrame: ...


def load_plugin(spec: str, cfg: Config) -> Any:
    """Build a stage from ``"package.module:factory"``; ``factory(cfg)`` returns the stage."""
    module_name, sep, attr = str(spec).partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"plug-in must look like 'package.module:factory', got {spec!r}")
    factory = importlib.import_module(module_name)
    for part in attr.split("."):
        factory = getattr(factory, part)
    if not callable(factory):
        raise TypeError(f"plug-in {spec!r} is not callable")
    return factory(cfg)


def _is_none(choice: Any) -> bool:
    return choice is None or choice is False or str(choice).lower() in ("none", "off", "false")


def build_detector(cfg: Config) -> Any | None:
    choice = cfg.get("pipeline.detector", "config")
    if _is_none(choice):
        return None
    if choice == "config":
        from actus.perception.detector import Detector

        return Detector.from_config(cfg.detector)
    return load_plugin(choice, cfg)


def build_handpose(cfg: Config) -> Any | None:
    choice = cfg.get("pipeline.handpose", "config")
    if _is_none(choice):
        return None
    if choice == "config":
        p = cfg.perception
        if not (p.get("hands", True) or p.get("pose", True)):
            return None
        from actus.perception.rotation_tta import RotationTTA

        return RotationTTA.from_config(p)
    return load_plugin(choice, cfg)


def build_rack_stage(cfg: Config, spatial) -> Any:
    choice = cfg.get("pipeline.rack", "aruco")
    if choice == "aruco":
        return spatial.rack_estimator
    if choice == "selfcal":
        from actus.models.rackfit import MarkerFrameFitter

        return MarkerFrameFitter(
            marker_size=spatial.geometry.marker_size,
            hold_frames=int(cfg.get("rack.hold_frames", 45)),
        )
    return load_plugin(choice, cfg)


def stage_name(stage: Any) -> str | None:
    if stage is None:
        return None
    name = getattr(stage, "backend_name", None)
    return str(name) if name else type(stage).__name__


def cap_visibility(
    evidence: tuple[StepEvidence, ...], predicates: Predicates, cap: float
) -> tuple[tuple[StepEvidence, ...], Predicates]:
    """A known sensor outage: at most ``cap`` of the scene can be seen.

    Every probability is pulled toward 0.5 (uninformative) by ``1 - cap`` -- the same mixing the
    replay applies under an occlusion -- and the visibility handed to the engine is clipped to
    ``cap``. With ``cap = 0`` no step can reach ``p_enter``, and the engine sees a blind sensor
    and asks about the expected step.
    """
    v = min(max(float(cap), 0.0), 1.0)

    def mix(p: float) -> float:
        return v * p + (1.0 - v) * 0.5

    capped = tuple(
        dataclasses.replace(
            e,
            p_complete=mix(e.p_complete),
            sources={k: mix(p) for k, p in e.sources.items()},
        )
        for e in evidence
    )
    predicates = dataclasses.replace(
        predicates,
        values={k: mix(p) for k, p in predicates.values.items()},
        visibility=min(predicates.visibility, v),
    )
    return capped, predicates


# --------------------------------------------------------------------------------------------
# Per-frame data
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceFrame:
    """One frame from a source, plus anything the source already knows about it.

    ``detections`` supplied by a source (the replay) replace the detector. ``evidence``,
    ``predicates`` and ``actions`` are the replay's stand-ins, used only when the core runs with
    ``evidence_mode="replay"`` (and ``actions`` when no temporal model is loaded). ``label`` is
    ground truth, drawn on SIMULATED overlays only.
    """

    packet: FramePacket
    detections: DetectionSet | None = None
    evidence: tuple[StepEvidence, ...] | None = None
    predicates: Predicates | None = None
    actions: ActionScores | None = None
    label: str | None = None


@dataclass(frozen=True)
class FrameResult:
    frame_idx: int
    t: float
    rack: RackFrame | None
    detections: DetectionSet | None
    hands: HandSet | None
    pose: PoseSet | None
    predicates: Predicates
    evidence: tuple[StepEvidence, ...]
    actions: ActionScores | None
    outputs: list[Output]
    stats: tuple[Stats, ...]
    annotated: FramePacket | None
    timings_ms: dict[str, float]

    def data_messages(self) -> list[tuple[Topic, Any]]:
        """Everything except engine outputs and frames, in publish order."""
        out: list[tuple[Topic, Any]] = []
        if self.detections is not None:
            out.append((Topic.DETECTIONS, self.detections))
        if self.hands is not None:
            out.append((Topic.HANDS, self.hands))
        if self.pose is not None:
            out.append((Topic.POSE, self.pose))
        if self.rack is not None:
            out.append((Topic.RACK, self.rack))
        out.append((Topic.PREDICATES, self.predicates))
        out += [(Topic.EVIDENCE, e) for e in self.evidence]
        if self.actions is not None:
            out.append((Topic.ACTIONS, self.actions))
        out += [(Topic.STATS, s) for s in self.stats]
        return out


# --------------------------------------------------------------------------------------------
# Core
# --------------------------------------------------------------------------------------------


class PipelineCore:
    """Perception -> models -> spatial -> engine for one frame at a time. Single-threaded.

    Runtime hooks (for T-FAULT and for the GUI):

    * ``source_hooks``: callables ``SourceFrame -> SourceFrame`` applied first, in order, to the
      whole frame including any detections the source supplies (``FaultController.process``).
      They run after the Runner has published the raw frame, so the raw stream and recording keep
      the true image;
    * ``frame_hooks``: callables ``FramePacket -> FramePacket`` applied, in order, to the image
      before any perception runs (occlusion, dimming, a frozen frame, ...);
    * ``disabled``: stage names (``"detector"``, ``"handpose"``, ``"models"``) switched off; a
      disabled stage reports "did not run", exactly like a stage that is absent;
    * ``visibility_cap``: an upper bound in [0, 1] on how much the sensors can currently see
      (None = no bound), for a *known* outage. The engine's visibility is clipped to it and the
      step evidence is pulled toward 0.5 in proportion, as the replay models an occlusion, so
      nothing can be verified from a sensor that is not there and the engine asks instead;
    * ``tags``: short strings drawn on the annotated frame (e.g. ``"FAULT INJECTED: ..."``).
    """

    def __init__(
        self,
        protocol: Protocol,
        *,
        spatial,
        engine: ProtocolEngine,
        models=None,
        detector: Any | None = None,
        handpose: Any | None = None,
        rack_stage: Any | None = None,
        evidence_mode: str = "spatial",
        detector_every_n: int = 1,
        handpose_every_n: int = 1,
        overlay: bool = True,
        simulated: bool = False,
    ) -> None:
        if evidence_mode not in ("spatial", "replay"):
            raise ValueError(f"evidence_mode must be 'spatial' or 'replay', got {evidence_mode!r}")
        self.protocol = protocol
        self.spatial = spatial
        self.engine = engine
        self.models = models
        self.detector = detector
        self.handpose = handpose
        self.rack_stage = rack_stage if rack_stage is not None else spatial.rack_estimator
        self.evidence_mode = evidence_mode
        self.detector_every_n = max(1, int(detector_every_n))
        self.handpose_every_n = max(1, int(handpose_every_n))
        self.overlay = overlay
        self.simulated = simulated
        self.source_hooks: list[Callable[[SourceFrame], SourceFrame]] = []
        self.frame_hooks: list[Callable[[FramePacket], FramePacket]] = []
        self.disabled: set[str] = set()
        self.visibility_cap: float | None = None
        self.tags: set[str] = set()

        from actus.spatial.zones import as_polygon

        self._zones = {n: as_polygon(pts) for n, pts in protocol.zones.items()}
        self._n = 0  # frames processed
        self.last_frame_idx: int | None = None
        self.last_t: float | None = None
        self._prompt: Prompt | None = None
        self._alert: Alert | None = None
        self.fps: float | None = None  # set by the Runner, drawn on the overlay

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        *,
        protocol: Protocol | None = None,
        detector: Any = _FROM_CONFIG,
        handpose: Any = _FROM_CONFIG,
        models: Any = _FROM_CONFIG,
        evidence_mode: str | None = None,
        simulated: bool = False,
    ) -> PipelineCore:
        """Build every stage from ``cfg``; pass a stage (or None) to override it."""
        from actus.spatial.estimator import SpatialEstimator

        protocol = protocol or load_protocol(cfg.protocol.path)
        spatial = SpatialEstimator.from_config(cfg, protocol=protocol)
        params = DecisionParams.from_protocol(protocol, **dict(cfg.get("engine") or {}))
        if models is _FROM_CONFIG:
            from actus.models.runtime import ModelStack

            models = ModelStack.from_config(cfg, protocol)
        pcfg = cfg.get("pipeline") or {}
        return cls(
            protocol,
            spatial=spatial,
            engine=ProtocolEngine(protocol, params),
            models=models,
            detector=build_detector(cfg) if detector is _FROM_CONFIG else detector,
            handpose=build_handpose(cfg) if handpose is _FROM_CONFIG else handpose,
            rack_stage=build_rack_stage(cfg, spatial),
            evidence_mode=evidence_mode or pcfg.get("evidence", "spatial"),
            detector_every_n=pcfg.get("detector_every_n", 1),
            handpose_every_n=pcfg.get("handpose_every_n", 1),
            overlay=bool(pcfg.get("overlay", True)),
            simulated=simulated,
        )

    # -- queries ------------------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        models = self.models.status() if self.models is not None else None
        return {
            "detector": stage_name(self.detector),
            "handpose": stage_name(self.handpose),
            "rack": stage_name(self.rack_stage),
            "models": models,
            "models_enabled": bool(self.models is not None and self.models.enabled),
            "evidence": self.evidence_mode,
            "simulated": self.simulated,
        }

    # -- operator input ------------------------------------------------------------------------

    def answer(self, ans: Answer) -> list[Output]:
        return self._track(self.engine.answer(ans))

    def command(self, cmd: Command) -> list[Output]:
        return self._track(self.engine.command(cmd))

    def finish(self) -> list[Output]:
        return self._track(self.engine.finish())

    def _track(self, outputs: list[Output]) -> list[Output]:
        """Remember what the overlay shows: the prompt and the latest alert."""
        for o in outputs:
            if isinstance(o, Prompt):
                self._prompt = o
            elif isinstance(o, Alert):
                self._alert = o
        return outputs

    # -- per frame ----------------------------------------------------------------------------

    def step(self, sf: SourceFrame) -> FrameResult:
        for source_hook in self.source_hooks:
            sf = source_hook(sf)
        packet = sf.packet
        for hook in self.frame_hooks:
            packet = hook(packet)
        fi, t = packet.frame_idx, packet.t_capture
        timings: dict[str, float] = {}

        def timed(name: str, fn: Callable[[], Any]) -> Any:
            t0 = time.perf_counter()
            try:
                return fn()
            finally:
                timings[name] = (time.perf_counter() - t0) * 1000.0

        rack = timed("rack", lambda: self.rack_stage.update(packet))

        dets = sf.detections
        if dets is None and self._due("detector", self.detector, self.detector_every_n):
            dets = timed("detector", lambda: self.detector.detect(packet))
        hands = pose = None
        if self._due("handpose", self.handpose, self.handpose_every_n):
            hands, pose = timed("handpose", lambda: self.handpose.process(packet))

        mf = None
        if self.models is not None and "models" not in self.disabled:
            mf = timed(
                "models",
                lambda: self.models.update(
                    fi, t, rack=rack, detections=dets, hands=hands, pose=pose
                ),
            )
        sp = timed(
            "spatial",
            lambda: self.spatial.update(
                fi,
                t,
                rack=rack,
                detections=dets,
                hands=hands,
                pose=pose,
                hoi=(mf.hoi or None) if mf is not None else None,
            ),
        )
        evidence = tuple(sp.evidence)
        if mf is not None:
            evidence = tuple(self.models.fuse(evidence))
        predicates = sp.predicates
        if self.evidence_mode == "replay" and sf.evidence is not None:
            evidence = tuple(sf.evidence)
            predicates = sf.predicates or predicates
        actions = mf.actions if mf is not None and mf.actions is not None else sf.actions
        if self.visibility_cap is not None:
            evidence, predicates = cap_visibility(evidence, predicates, self.visibility_cap)

        outputs = timed(
            "engine",
            lambda: self.engine.update(fi, t, evidence, visibility=predicates, actions=actions),
        )
        self._track(outputs)
        self._n += 1
        self.last_frame_idx, self.last_t = fi, t

        stats = [sp.stats()]
        if mf is not None:
            stats.append(self.models.stats(mf))

        annotated = None
        if self.overlay:
            annotated = timed("overlay", lambda: self._annotate(packet, sp, sf.label, t))
        return FrameResult(
            fi,
            t,
            sp.rack,
            sp.detections,
            sp.hands,
            sp.pose,
            predicates,
            evidence,
            actions,
            outputs,
            tuple(stats),
            annotated,
            timings,
        )

    def _due(self, name: str, stage: Any, every_n: int) -> bool:
        return stage is not None and name not in self.disabled and self._n % every_n == 0

    def _annotate(self, packet: FramePacket, sp, label: str | None, t: float) -> FramePacket:
        from actus.app import overlay

        world = sp.world
        footprints = {
            n: np.asarray(o.footprint) for n, o in world.objects.items() if o.footprint is not None
        }
        points = {
            n: o.pos for n, o in world.objects.items() if o.kind == "movable" and o.pos is not None
        }
        alert = self._alert
        alert_text = alert.message if alert is not None and t - alert.t <= ALERT_SHOW_S else None
        ask = self.engine.open_ask
        img = overlay.render(
            packet.image,
            rack=sp.rack,
            geometry=self.spatial.geometry,
            zones=self._zones,
            footprints=footprints,
            points=points,
            detections=sp.detections,
            hands=sp.hands,
            pose=sp.pose,
            step_ids=self.protocol.step_ids,
            statuses={k: v.value for k, v in self.engine.statuses.items()},
            prompt=self._prompt.text if self._prompt is not None else None,
            ask=ask.text if ask is not None else None,
            alert=alert_text,
            fps=self.fps,
            simulated=self.simulated,
            label=label if self.simulated else None,
            tags=sorted(self.tags),
        )
        return FramePacket(
            packet.frame_idx, packet.t_capture, img, packet.source_id, packet.rotation_applied
        )


# --------------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------------


class FrameSource(TypingProtocol):
    kind: str
    live: bool  # frames arrive in real time on their own (a camera); no pacing needed
    publishes_frames: bool  # already on the bus's `frames` topic
    simulated: bool
    fps_hint: float | None  # nominal rate for video containers (recorder, RTSP)

    def __iter__(self) -> Iterator[SourceFrame | None]: ...  # None = idle tick
    @property
    def dropped(self) -> int: ...
    def close(self) -> None: ...


def parse_source(value: Any) -> int | str:
    """``"0"`` -> 0 (a webcam index); anything else unchanged."""
    s = str(value).strip()
    return int(s) if s.lstrip("-").isdigit() else s


def is_file_source(value: Any) -> bool:
    return not isinstance(value, int) and "://" not in str(value)


class FileSource:
    """Sequential decode of a video file. Never drops a frame, never loops.

    ``t`` is the container's own presentation time (``CAP_PROP_POS_MSEC``). If the container
    reports no usable timestamps, ``frame_idx / container_fps`` is used -- the container's declared
    rate, never a guessed default -- and a file with neither is refused. ``t`` is forced to be
    non-decreasing, which the engine requires.
    """

    kind = "file"
    live = False
    publishes_frames = False
    simulated = False

    def __init__(
        self,
        path: str | Path,
        *,
        source_id: str | None = None,
        rotation: int = 0,
        max_frames: int | None = None,
    ) -> None:
        import cv2

        self.path = resolve_path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"no such video file: {self.path}")
        self.source_id = source_id or self.path.stem
        self.rotation = int(rotation)
        self.max_frames = max_frames
        cap = cv2.VideoCapture(str(self.path))
        try:
            if not cap.isOpened():
                raise ValueError(f"could not open video {self.path}")
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            cap.release()
        self.fps_hint = fps if np.isfinite(fps) and fps > 0 else None
        self.frames_read = 0

    @property
    def dropped(self) -> int:
        return 0

    def close(self) -> None:
        pass

    def __iter__(self) -> Iterator[SourceFrame]:
        import cv2

        from actus.perception.rotation_tta import rotate_image

        cap = cv2.VideoCapture(str(self.path))
        use_pts: bool | None = None
        last_t: float | None = None
        idx = 0
        try:
            while self.max_frames is None or idx < self.max_frames:
                ok, img = cap.read()
                if not ok or img is None:
                    return
                msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                if use_pts is None and idx >= 1:
                    use_pts = np.isfinite(msec) and msec > 0
                    if not use_pts and self.fps_hint is None:
                        raise ValueError(
                            f"{self.path}: the container reports neither timestamps nor a frame "
                            "rate, so frames cannot be timed"
                        )
                if idx == 0:
                    t = max(0.0, msec / 1000.0) if np.isfinite(msec) else 0.0
                elif use_pts:
                    t = msec / 1000.0
                else:
                    t = idx / self.fps_hint  # type: ignore[operator]
                if last_t is not None:
                    t = max(t, last_t)
                last_t = t
                if self.rotation:
                    img = rotate_image(img, self.rotation)
                self.frames_read += 1
                yield SourceFrame(FramePacket(idx, t, img, self.source_id, self.rotation))
                idx += 1
        finally:
            cap.release()


class CameraSource:
    """A live camera or RTSP input via T-PERC's threaded ``Camera``.

    The camera thread publishes every frame on ``frames`` (so the recorder and the raw RTSP push
    see all of them); the pipeline reads through its own small ``drop_oldest`` subscription, so
    when processing falls behind it works on the freshest frame and the skipped ones are counted
    in ``dropped``. Yields ``None`` on idle ticks so the Runner can still serve operator input.
    """

    kind = "camera"
    live = True
    publishes_frames = True
    simulated = False

    def __init__(self, camera, bus: Bus, *, maxsize: int = 2, idle_s: float = 0.1) -> None:
        self.camera = camera
        self.bus = bus
        self.maxsize = maxsize
        self.idle_s = idle_s
        self.fps_hint = float(camera.requested_fps) if camera.requested_fps else None
        self._sub: Subscription | None = None

    @classmethod
    def from_config(cls, capture_cfg, bus: Bus, *, source: Any = None, **kw) -> CameraSource:
        from actus.capture.camera import Camera

        cam = Camera(
            parse_source(capture_cfg.source if source is None else source),
            source_id=capture_cfg.get("source_id", "cam0"),
            width=capture_cfg.get("width"),
            height=capture_cfg.get("height"),
            fps=capture_cfg.get("fps"),
            rotation=capture_cfg.get("rotation", 0),
        )
        return cls(cam, bus, **kw)

    @property
    def dropped(self) -> int:
        return self._sub.dropped if self._sub is not None else 0

    def __iter__(self) -> Iterator[SourceFrame | None]:
        self._sub = self.bus.subscribe(
            Topic.FRAMES, maxsize=self.maxsize, policy="drop_oldest", name="pipeline-frames"
        )
        self.camera.start(self.bus)
        while True:
            try:
                env = self._sub.get(timeout=self.idle_s)
            except BusClosed:
                return
            if env is None:
                if self.camera.ended.is_set() and self._sub.depth == 0:
                    if self.camera.error is not None:
                        raise RuntimeError(f"camera failed: {self.camera.error}")
                    return
                yield None
                continue
            yield SourceFrame(env.payload)

    def close(self) -> None:
        self.camera.stop()
        if self._sub is not None:
            self._sub.unsubscribe()


class ReplayFrameSource:
    """The P0 synthetic box world, rendered with real ArUco markers. Always SIMULATED.

    Replay detections are passed on with their rack positions stripped, so the spatial lane
    solves the rack from the rendered markers and projects them itself -- the real chain, minus
    the neural perception.
    """

    kind = "replay"
    live = False
    publishes_frames = False
    simulated = True

    def __init__(
        self,
        scenario: str,
        *,
        protocol: Protocol | None = None,
        fps: float = 15.0,
        seed: int = 0,
        noise: float = 0.04,
    ) -> None:
        from actus.capture.replay import ReplaySource

        self.scenario = scenario
        self.replay = ReplaySource(
            scenario, protocol=protocol, fps=fps, seed=seed, noise=noise, render=True
        )
        self.fps_hint = float(fps)

    @classmethod
    def from_config(cls, cfg: Config, scenario: str | None = None, protocol=None):
        r = cfg.replay
        return cls(
            scenario or r.scenario,
            protocol=protocol,
            fps=r.get("fps", 15.0),
            seed=r.get("seed", 0),
            noise=r.get("noise", 0.04),
        )

    @property
    def dropped(self) -> int:
        return 0

    def close(self) -> None:
        pass

    def __iter__(self) -> Iterator[SourceFrame]:
        for tick in self.replay:
            d = tick.detections
            dets = DetectionSet(
                d.frame_idx,
                d.t,
                [dataclasses.replace(x, bbox_rack_xy=None) for x in d.detections],
                d.backend,
            )
            yield SourceFrame(
                tick.frame,
                detections=dets,
                evidence=tick.evidence,
                predicates=tick.predicates,
                actions=tick.actions,
                label=tick.label,
            )


# --------------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------------

AnswerPolicy = Callable[[Ask], tuple[float, AnswerValue | str] | None]


def make_answer_policy(
    answer: AnswerPolicy | AnswerValue | str | None, delay_s: float = 1.0
) -> AnswerPolicy | None:
    """``"yes"``/``"no"`` answers every Ask ``delay_s`` after it is asked; a callable decides."""
    if answer is None or callable(answer):
        return answer
    fixed = AnswerValue(answer)
    return lambda _ask: (delay_s, fixed)


class PipelineClock:
    """POSIX seconds = ``base`` + the pipeline ``t`` of the frame being processed. Given to the
    event log in offline runs so its timestamps depend only on the input."""

    def __init__(self, base_posix_s: float = 0.0) -> None:
        self.base = float(base_posix_s)
        self.t = 0.0

    def __call__(self) -> float:
        return self.base + self.t


@dataclass
class RunSummary:
    frames: int
    dropped: int
    outputs: list[Output]
    outcomes: list[dict[str, Any]]
    wall_s: float
    reason: str = "end_of_stream"
    step_events: list[StepEvent] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)


class Runner:
    """Drive a ``FrameSource`` through a ``PipelineCore``.

    Per frame: apply operator input that is due, ``core.step``, then log engine outputs, publish
    everything, write the annotated frame and publish ``stats``. At the end of the stream the
    engine's ``finish()`` outputs are handled the same way. ``bus``, ``eventlog`` and
    ``annotated_writer`` are all optional; offline runs pass no bus.
    """

    def __init__(
        self,
        core: PipelineCore,
        source: FrameSource,
        *,
        bus: Bus | None = None,
        eventlog=None,
        answer: AnswerPolicy | AnswerValue | str | None = None,
        answer_delay_s: float = 1.0,
        realtime: bool = False,
        speed: float = 1.0,
        clock: PipelineClock | None = None,
        annotated_writer=None,
        stats_interval_s: float = 1.0,
        extra_stats: Callable[[], list[Stats]] | None = None,
        on_result: Callable[[FrameResult], None] | None = None,
        max_seconds: float | None = None,
        stop: threading.Event | None = None,
    ) -> None:
        self.core = core
        self.source = source
        self.bus = bus
        self.eventlog = eventlog
        self.policy = make_answer_policy(answer, answer_delay_s)
        self.realtime = realtime
        self.speed = max(1e-3, float(speed))
        self.clock = clock
        self.annotated_writer = annotated_writer
        self.stats_interval_s = stats_interval_s
        self.extra_stats = extra_stats
        self.on_result = on_result
        self.max_seconds = max_seconds
        self.stop_event = stop or threading.Event()
        self.outputs: list[Output] = []
        self.frames = 0
        self._inputs: Subscription | None = None
        self._due: list[tuple[float, Answer]] = []
        self._own: set[int] = set()  # ids of answers this runner published itself
        self._timings: dict[str, float] = {}
        self._fps_frames = 0
        self._fps_t0 = 0.0
        self._last_stats = 0.0

    # -- main loop ----------------------------------------------------------------------------

    def run(self) -> RunSummary:
        wall0 = time.monotonic()
        self._fps_t0 = self._last_stats = wall0
        if self.bus is not None:
            self._inputs = self.bus.subscribe(
                [Topic.ANSWERS, Topic.COMMANDS],
                maxsize=256,
                policy="drop_newest",
                name="pipeline-inputs",
            )
        reason = "end_of_stream"
        t_first: float | None = None
        try:
            for sf in self.source:
                if self.stop_event.is_set():
                    reason = "stopped"
                    break
                if sf is None:
                    self._service_inputs(None)
                    self._maybe_stats()
                    continue
                t = sf.packet.t_capture
                if t_first is None:
                    t_first = t
                if self.max_seconds is not None and t - t_first > self.max_seconds:
                    reason = "max_seconds"
                    break
                if self.realtime and not self.source.live:
                    self._pace(wall0, t - t_first)
                    if self.stop_event.is_set():
                        reason = "stopped"
                        break
                if self.clock is not None:
                    self.clock.t = t
                if self.bus is not None and not self.source.publishes_frames:
                    self._publish(Topic.FRAMES, sf.packet)
                self._service_inputs(t)
                self._process(sf)
        except KeyboardInterrupt:  # Ctrl-C (or SIGTERM, see LiveApp): still finish and log
            reason = "interrupted"
        finally:
            self.source.close()
        try:
            self._service_inputs(None)
            self._emit(self.core.finish())
        except Exception:
            if reason != "interrupted":
                raise
            logger.exception("engine finish after an interrupt")
        if self._inputs is not None:
            self._inputs.unsubscribe()
        self._maybe_stats(force=True)
        return RunSummary(
            frames=self.frames,
            dropped=self.source.dropped,
            outputs=self.outputs,
            outcomes=self.core.engine.outcomes(),
            wall_s=time.monotonic() - wall0,
            reason=reason,
            step_events=[o for o in self.outputs if isinstance(o, StepEvent)],
            alerts=[o for o in self.outputs if isinstance(o, Alert)],
        )

    def _pace(self, wall0: float, t_rel: float) -> None:
        delay = wall0 + t_rel / self.speed - time.monotonic()
        while delay > 0 and not self.stop_event.is_set():
            self._service_inputs(None)
            self.stop_event.wait(min(delay, 0.05))
            delay = wall0 + t_rel / self.speed - time.monotonic()

    def _process(self, sf: SourceFrame) -> None:
        res = self.core.step(sf)
        self.frames += 1
        for k, v in res.timings_ms.items():
            prev = self._timings.get(k)
            self._timings[k] = v if prev is None else 0.9 * prev + 0.1 * v
        if self.bus is not None:
            for topic, msg in res.data_messages():
                self._publish(topic, msg)
        self._emit(res.outputs)
        if res.annotated is not None:
            if self.bus is not None:
                self._publish(Topic.ANNOTATED, res.annotated)
            if self.annotated_writer is not None:
                self.annotated_writer.write(res.annotated)
        if self.on_result is not None:
            self.on_result(res)
        self._fps_frames += 1
        self._maybe_stats()

    # -- operator input -----------------------------------------------------------------------

    def _service_inputs(self, t: float | None) -> None:
        """Apply bus answers/commands, then scripted answers due at or before ``t``."""
        if self._inputs is not None:
            for env in self._inputs.drain():
                msg = env.payload
                if id(msg) in self._own:
                    self._own.discard(id(msg))
                    continue
                if isinstance(msg, Answer):
                    self._apply_answer(msg)
                elif isinstance(msg, Command):
                    self._emit(self.core.command(msg))
        if not self._due:
            return
        if t is None and self.core.last_t is not None:
            t = self.core.last_t
        if t is None:
            return
        ready = sorted((d for d in self._due if d[0] <= t + 1e-9), key=lambda d: d[0])
        self._due = [d for d in self._due if d[0] > t + 1e-9]
        for _, ans in ready:
            if self.bus is not None:
                self._own.add(id(ans))
                self._publish(Topic.ANSWERS, ans)
            self._apply_answer(ans)

    def _apply_answer(self, ans: Answer) -> None:
        if ans.frame_idx is None and self.core.last_frame_idx is not None:
            ans = dataclasses.replace(ans, frame_idx=self.core.last_frame_idx)
        self._log(ans)
        self._emit(self.core.answer(ans))

    # -- output -------------------------------------------------------------------------------

    def _emit(self, outputs: list[Output]) -> None:
        for o in outputs:
            self.outputs.append(o)
            self._log(o)
            if self.bus is not None:
                self._publish(OUTPUT_TOPICS[type(o)], o)
            if self.policy is not None and isinstance(o, Ask):
                decision = self.policy(o)
                if decision is not None:
                    delay, value = decision
                    ans = Answer(o.question_id, value, InputVia.VOICE, t=o.t + delay)
                    self._due.append((o.t + delay, ans))

    def _log(self, payload: Any) -> None:
        if self.eventlog is not None and not self.eventlog.closed:
            self.eventlog.append(payload)

    def _publish(self, topic: Topic, msg: Any) -> None:
        with contextlib.suppress(BusClosed):
            self.bus.publish(topic, msg)  # type: ignore[union-attr]

    # -- stats --------------------------------------------------------------------------------

    def _maybe_stats(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_stats < self.stats_interval_s:
            return
        span = now - self._fps_t0
        if span > 0 and self._fps_frames:
            self.core.fps = self._fps_frames / span
        self._fps_frames, self._fps_t0, self._last_stats = 0, now, now
        if self.bus is None:
            return
        values: dict[str, Any] = {
            "fps": round(self.core.fps, 2) if self.core.fps is not None else None,
            "dropped": self.source.dropped,
            "simulated": self.core.simulated,
            "frames": self.frames,
            "source": self.source.kind,
            "evidence": self.core.evidence_mode,
        }
        status = self.core.status()
        for k in ("detector", "handpose", "rack"):
            values[f"stage.{k}"] = status[k]
        values["models.enabled"] = status["models_enabled"]
        for k, v in self._timings.items():
            values[f"ms.{k}"] = round(v, 2)
        values["ms.total"] = round(sum(self._timings.values()), 2)
        values["bus.dropped"] = self.bus.stats()["dropped_total"]
        if self.core.tags:
            values["tags"] = ",".join(sorted(self.core.tags))
        t = self.core.last_t if self.core.last_t is not None else 0.0
        stats = Stats("pipeline", values, t=t, frame_idx=self.core.last_frame_idx)
        self._publish(Topic.STATS, stats)
        if self.extra_stats is not None:
            for s in self.extra_stats():
                self._publish(Topic.STATS, s)


# --------------------------------------------------------------------------------------------
# Sinks (live only)
# --------------------------------------------------------------------------------------------


class VoiceSink:
    """Speaks engine outputs (alert > ask > prompt) and runs the gated offline ASR.

    The ASR gate opens on an ``Ask`` and closes on its ``Answer``, on a newer ``Ask``, or on a
    later ``StepEvent`` for the asked step (the engine's convention for "resolved by evidence").
    Any piece that cannot start (no audio device, no weights, packages not installed) is logged
    and left out; GUI Yes/No keeps working without it.
    """

    def __init__(self, bus: Bus, voice_cfg, *, backend=None, asr: bool | None = None) -> None:
        from actus.voice.voice_queue import VoicePlayer, VoiceQueue

        q = voice_cfg.get("queue") or {}
        self.bus = bus
        self.queue = VoiceQueue(
            maxsize=int(q.get("maxsize", 16)),
            dedup_s=float(q.get("dedup_s", 5.0)),
            ttl_s=dict(q.get("ttl_s") or {}),
        )
        self.backend_name = "injected"
        if backend is None:
            backend, self.backend_name = _make_tts(voice_cfg)
        self.backend = backend
        self.player = VoicePlayer(backend, self.queue)
        self.asr = None
        self.gate = None
        self._ask: Ask | None = None
        want_asr = bool(voice_cfg.get("asr.enabled", True)) if asr is None else asr
        if want_asr:
            self._start_asr(voice_cfg)
        self._sub = bus.subscribe(
            [Topic.ALERTS, Topic.ASKS, Topic.PROMPTS, Topic.EVENTS, Topic.ANSWERS, Topic.COMMANDS],
            maxsize=128,
            policy="drop_newest",
            name="voice",
        )
        self._thread: threading.Thread | None = None

    def _start_asr(self, voice_cfg) -> None:
        try:
            from actus.voice.asr import AskGate, AsrService, VoskRecognizer

            rec = VoskRecognizer(voice_cfg.get("asr.model"))
            # Muted while TTS plays, so the mic never answers Actus's own question.
            self.gate = AskGate(mute_while=lambda: self.player.speaking)
            self.asr = AsrService(rec, self.gate, bus=self.bus)
            self.asr.start_mic()
        except Exception as exc:
            logger.warning(
                "voice input unavailable (%s: %s); GUI Yes/No still works", type(exc).__name__, exc
            )
            self.asr = None

    def start(self) -> None:
        self.player.start()
        self._thread = threading.Thread(target=self._run, name="voice-sink", daemon=True)
        self._thread.start()

    def handle(self, msg: Any) -> None:
        if isinstance(msg, (Command, Answer)):
            # The operator asked to hear something again ("repeat", "what's next", or a "no"
            # that re-prompts): the engine's re-issued text is not a duplicate.
            self.queue.forget_recent()
        if isinstance(msg, (Alert, Ask, Prompt)):
            self.queue.put_contract(msg)
        if isinstance(msg, Ask):
            self._ask = msg
            if self.gate is not None:
                self.gate.on_ask(msg)
        elif isinstance(msg, Answer):
            if self.gate is not None:
                self.gate.on_answer(msg)
            if self._ask is not None and msg.question_id == self._ask.question_id:
                self._ask = None
        elif isinstance(msg, StepEvent) and self._ask is not None:
            if msg.step_id == self._ask.step_id:
                self._ask = None
                if self.gate is not None:
                    self.gate.cancel()

    def _run(self) -> None:
        for env in self._sub:
            try:
                self.handle(env.payload)
            except Exception:
                logger.exception("voice sink")

    def stats(self) -> Stats:
        v: dict[str, Any] = {"tts": self.backend_name, "asr": self.asr is not None}
        v.update(self.queue.stats())
        if self.gate is not None:
            v["listening"] = self.gate.listening
        return Stats("voice", v, t=time.monotonic())

    def stop(self) -> None:
        self._sub.unsubscribe()
        if self.asr is not None:
            with contextlib.suppress(Exception):
                self.asr.stop()
        self.player.stop()
        with contextlib.suppress(Exception):
            self.backend.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def _make_tts(voice_cfg) -> tuple[Any, str]:
    from actus.voice.tts import NullTTS, create_backend

    try:
        backend = create_backend(voice_cfg.tts)
        return backend, type(backend).__name__
    except Exception as exc:
        logger.warning("no TTS engine (%s: %s); voice output is text-only", type(exc).__name__, exc)
        return NullTTS(), "none"


class UIServer:
    """The FastAPI dashboard (T-UI) on the pipeline's bus, served from a background thread."""

    def __init__(self, bus: Bus, cfg: Config, protocol: Protocol, faults: Any = None) -> None:
        import uvicorn

        from actus.app.server import create_app

        app = create_app(bus, config=cfg, protocol=protocol, enable_demo=False, faults=faults)
        self.host = cfg.get("ui.host", "127.0.0.1")
        self.port = int(cfg.get("ui.port", 8000))
        self.server = uvicorn.Server(
            uvicorn.Config(app, host=self.host, port=self.port, log_level="warning")
        )
        self._thread = threading.Thread(target=self.server.run, name="ui-server", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.server.should_exit = True
        self._thread.join(timeout=5.0)


# --------------------------------------------------------------------------------------------
# Runs: directories, metadata, downlink
# --------------------------------------------------------------------------------------------


def new_run_id(prefix: str = "") -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}{stamp}"


def run_dir_for(cfg: Config, run_id: str) -> Path:
    return resolve_path(cfg.get("run.runs_dir", "runs")) / run_id


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def write_downlink_for(
    run_dir: Path,
    *,
    run_id: str,
    protocol: Protocol,
    summary: RunSummary,
    eventlog,
    started_at: str,
    ended_at: str,
    max_bytes: int = 4096,
) -> int:
    from actus.io.downlink import build_downlink, write_downlink

    packet = build_downlink(
        run_id=run_id,
        protocol=protocol,
        step_events=summary.step_events,
        alerts=summary.alerts,
        started_at=started_at,
        ended_at=ended_at,
        log_hash=eventlog.final_hash,
    )
    return write_downlink(run_dir / "downlink.json", packet, max_bytes=max_bytes)


def outcome_line(outcomes: list[dict[str, Any]]) -> str:
    return ", ".join(f"{r['step_id']}={r['status']}" for r in outcomes)


# --------------------------------------------------------------------------------------------
# Live application
# --------------------------------------------------------------------------------------------


class LiveApp:
    """A Runner plus every live sink, with setup and teardown in one place.

    Each sink is optional and independently failure-tolerant: a missing ffmpeg/MediaMTX, audio
    device, Vosk model or ``ui`` extra is logged and that sink is left out; the run goes on and
    ``run.json`` records what actually ran. Recording and the event log never depend on the
    network.
    """

    def __init__(
        self,
        cfg: Config,
        *,
        source: str = "live",
        source_arg: Any = None,
        run_id: str | None = None,
        ui: bool = True,
        voice: bool = True,
        stream: bool = True,
        record: bool = True,
        realtime: bool = True,
        answer: str | None = None,
        max_seconds: float | None = None,
        evidence: str | None = None,
        core: PipelineCore | None = None,
        faults: Any = None,
    ) -> None:
        self.cfg = cfg
        self.bus = Bus()
        self.protocol = core.protocol if core is not None else load_protocol(cfg.protocol.path)
        simulated = source == "replay"
        if source == "replay":
            self.source: Any = ReplayFrameSource.from_config(cfg, source_arg, self.protocol)
        else:
            arg = parse_source(cfg.capture.source if source_arg is None else source_arg)
            if source == "file" or is_file_source(arg):
                self.source = FileSource(arg, rotation=cfg.get("capture.rotation", 0))
            else:
                self.source = CameraSource.from_config(cfg.capture, self.bus, source=arg)
        self.core = core or PipelineCore.from_config(
            cfg,
            protocol=self.protocol,
            detector=None if simulated else _FROM_CONFIG,
            handpose=None if simulated else _FROM_CONFIG,
            evidence_mode=evidence,
            simulated=simulated,
        )
        self.faults = self._make_faults(faults)
        self.run_id = run_id or cfg.get("run.run_id") or new_run_id("sim-" if simulated else "")
        self.run_dir = run_dir_for(cfg, str(self.run_id))
        self.want = {
            "ui": ui,
            "voice": voice and bool(cfg.get("voice.enabled", True)),
            "stream": stream and bool(cfg.get("stream.enabled", True)),
            "record": record and bool(cfg.get("record.enabled", True)),
        }
        self.realtime = realtime
        self.answer = answer
        self.max_seconds = max_seconds
        self.sinks: dict[str, Any] = {}
        self.stop_event = threading.Event()

    def _make_faults(self, faults: Any) -> Any:
        """``None`` follows ``faults.enabled``; ``True`` builds the controller from the config;
        a ``FaultController`` is used as given; ``False`` is off."""
        if faults is None:
            faults = bool(self.cfg.get("faults.enabled", False))
        if faults is False:
            return None
        from actus.app.faults import FaultController

        ctrl = FaultController.from_config(self.cfg) if faults is True else faults
        return ctrl.attach(self.core).bind_bus(self.bus)

    @property
    def fps_hint(self) -> float:
        return self.source.fps_hint or float(self.cfg.get("capture.fps", 30))

    def _start_record(self) -> Any:
        from actus.io.recorder import Recorder

        rec = Recorder(
            self.bus,
            self.run_dir,
            fps=self.fps_hint,
            segment_s=float(self.cfg.get("record.segment_s", 60)),
            raw=bool(self.cfg.get("record.raw", True)),
            annotated=bool(self.cfg.get("record.annotated", True)),
        )
        rec.start()
        return rec

    def _start_stream(self) -> Any:
        from actus.io.streamer import Streamer

        streamer = Streamer(self.bus, self.cfg.stream)
        streamer.start(fps=self.fps_hint)
        return streamer

    def _start_voice(self) -> Any:
        sink = VoiceSink(self.bus, self.cfg.voice)
        sink.start()
        return sink

    def _start_ui(self) -> Any:
        server = UIServer(self.bus, self.cfg, self.protocol, faults=self.faults)
        server.start()
        return server

    def _start_sinks(self) -> None:
        starters = {
            "record": self._start_record,
            "stream": self._start_stream,
            "voice": self._start_voice,
            "ui": self._start_ui,
        }
        for name, start in starters.items():
            if not self.want[name]:
                continue
            try:
                self.sinks[name] = start()
            except Exception as exc:
                logger.warning("%s disabled (%s: %s)", name, type(exc).__name__, exc)

    def _start_faults(self) -> Any:
        """Point ``kill_network`` at the running streamer and start the console hotkeys."""
        if self.faults is None:
            return None
        from actus.app.faults import HotkeyListener, StreamerTarget

        if "stream" in self.sinks:
            self.faults.set_network_target(StreamerTarget(self.sinks["stream"], self.fps_hint))
        if not self.cfg.get("faults.hotkeys", True):
            return None
        hotkeys = HotkeyListener(self.faults)
        return hotkeys if hotkeys.start() else None

    def _sink_stats(self) -> list[Stats]:
        out: list[Stats] = []
        now = time.monotonic()
        if "voice" in self.sinks:
            out.append(self.sinks["voice"].stats())
        if "record" in self.sinks:
            rec = self.sinks["record"].stats()
            out.append(Stats("recorder", _flatten(rec), t=now))
        if "stream" in self.sinks:
            st = self.sinks["stream"].stats()
            v = {"target": f"{self.cfg.stream.target_ip}:{self.cfg.stream.port}"}
            v.update(_flatten(st))
            out.append(Stats("streamer", v, t=now))
        if self.faults is not None:
            out.append(self.faults.stats())
        return out

    def run(self) -> tuple[RunSummary, Path]:
        from actus.io.eventlog import EventLog, iso_at

        self.run_dir.mkdir(parents=True, exist_ok=True)
        log = EventLog(
            self.run_dir,
            events_file=self.cfg.get("log.events_file", "events.jsonl"),
            catch_signals=False,  # handled here, so the engine's finish() still reaches the log
        )
        started_at = iso_at(time.time())
        meta = {
            "run_id": self.run_id,
            "mode": "live",
            "source": self.source.kind,
            "source_arg": str(getattr(self.source, "path", getattr(self.source, "scenario", ""))),
            "simulated": self.source.simulated,
            "protocol": self.protocol.id,
            "started_at": started_at,
            "stages": self.core.status(),
        }
        write_json(self.run_dir / "run.json", meta)

        if self.faults is not None:
            self.faults.open_log(self.run_dir, tip=lambda: log.final_hash)
        self._start_sinks()
        meta["sinks"] = sorted(self.sinks)
        hotkeys = self._start_faults()
        runner = Runner(
            self.core,
            self.source,
            bus=self.bus,
            eventlog=log,
            answer=self.answer,
            realtime=self.realtime,
            stats_interval_s=float(self.cfg.get("pipeline.stats_interval_s", 1.0)),
            extra_stats=self._sink_stats,
            max_seconds=self.max_seconds,
            stop=self.stop_event,
        )
        prev_term = _install_sigterm()
        try:
            summary = runner.run()
        finally:
            _restore_sigterm(prev_term)
            if hotkeys is not None:
                hotkeys.stop()
            if self.faults is not None:  # before the sinks stop: no point restarting the stream
                with contextlib.suppress(Exception):
                    meta["faults"] = self.faults.close(restore_network=False)
            with contextlib.suppress(Exception):  # what each sink actually did, for run.json
                meta["sink_status"] = {s.source: s.values for s in self._sink_stats()}
            for name in ("ui", "voice", "stream", "record"):
                sink = self.sinks.get(name)
                if sink is not None:
                    with contextlib.suppress(Exception):
                        sink.stop()
            with contextlib.suppress(Exception):
                self.bus.close()
        log.close(reason=summary.reason)
        ended_at = iso_at(time.time())
        size = write_downlink_for(
            self.run_dir,
            run_id=str(self.run_id),
            protocol=self.protocol,
            summary=summary,
            eventlog=log,
            started_at=started_at,
            ended_at=ended_at,
            max_bytes=int(self.cfg.get("log.downlink_max_bytes", 4096)),
        )
        meta.update(
            ended_at=ended_at,
            frames=summary.frames,
            dropped=summary.dropped,
            reason=summary.reason,
            outcomes=summary.outcomes,
            log_hash=log.final_hash,
            downlink_bytes=size,
        )
        write_json(self.run_dir / "run.json", meta)
        return summary, self.run_dir


@dataclass
class OfflineResult:
    run_id: str
    run_dir: Path
    summary: RunSummary
    events_path: Path
    events_sha256: str
    downlink_bytes: int
    simulated: bool


def take_start_posix(video: Path) -> tuple[float, str]:
    """Start time of a take from its T-REC metadata (``<stem>.json: started_at``), as POSIX
    seconds, and where it came from. Without metadata the log clock starts at the epoch, which
    keeps it a pure function of the input."""
    meta = video.with_suffix(".json")
    if meta.is_file():
        try:
            started = json.loads(meta.read_text(encoding="utf-8")).get("started_at")
            if started:
                when = dt.datetime.fromisoformat(str(started))
                if when.tzinfo is None:
                    when = when.replace(tzinfo=dt.UTC)
                return when.timestamp(), f"{meta.name}:started_at + container t"
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    return 0.0, "1970-01-01T00:00:00Z + container t (no take metadata)"


def run_offline(
    input_spec: str | Path,
    cfg: Config | None = None,
    *,
    run_id: str | None = None,
    answer: AnswerPolicy | AnswerValue | str | None = None,
    answer_delay_s: float = 1.0,
    video: bool = True,
    evidence: str | None = None,
    detector: Any = _FROM_CONFIG,
    handpose: Any = _FROM_CONFIG,
    models: Any = _FROM_CONFIG,
    max_frames: int | None = None,
    force: bool = False,
    on_result: Callable[[FrameResult], None] | None = None,
    faults: Any = None,
) -> OfflineResult:
    """Process a recorded take, or ``"replay:<scenario>"``, deterministically.

    Same core as the live run; no drops, no pacing, no voice/RTSP/GUI. The event log's clock is
    the take's start time plus the frames' own ``t``, so the same input and config give a
    byte-identical ``events.jsonl``. The annotated video goes to ``<run_dir>/video/``.
    ``on_result`` is called with every frame's ``FrameResult`` (T-EVAL uses it to map annotation
    frame indices onto the pipeline's own clock); it must not change anything it is given.
    ``faults`` is a ``FaultController`` (``actus.app.faults``): it is attached to the core and
    writes ``faults.jsonl`` beside the event log. A scheduled controller keeps the run
    deterministic.
    """
    import hashlib
    import shutil

    from actus.io.eventlog import EventLog, iso_at
    from actus.io.recorder import SegmentWriter

    cfg = cfg if cfg is not None else load_config()
    protocol = load_protocol(cfg.protocol.path)
    spec = str(input_spec)
    source: Any
    if spec.startswith("replay:"):
        scenario = spec.split(":", 1)[1]
        source = ReplayFrameSource.from_config(cfg, scenario, protocol)
        simulated, base, clock_src, stem = True, 0.0, "replay synthetic t", f"replay-{scenario}"
        detector = None if detector is _FROM_CONFIG else detector
        handpose = None if handpose is _FROM_CONFIG else handpose
    else:
        source = FileSource(spec, rotation=cfg.get("capture.rotation", 0), max_frames=max_frames)
        simulated, stem = False, source.path.stem
        base, clock_src = take_start_posix(source.path)
    core = PipelineCore.from_config(
        cfg,
        protocol=protocol,
        detector=detector,
        handpose=handpose,
        models=models,
        evidence_mode=evidence,
        simulated=simulated,
    )
    if faults is not None:
        faults.attach(core)
    run_id = str(run_id or f"offline-{stem}-{new_run_id()}")
    run_dir = run_dir_for(cfg, run_id)
    events_name = cfg.get("log.events_file", "events.jsonl")
    if run_dir.exists() and any(run_dir.iterdir()):
        if not force:
            raise FileExistsError(f"{run_dir} is not empty (pass force=True / --force)")
        for name in (events_name, "downlink.json", "run.json"):
            (run_dir / name).unlink(missing_ok=True)
        shutil.rmtree(run_dir / "video", ignore_errors=True)

    clock = PipelineClock(base)
    log = EventLog(run_dir, events_file=events_name, catch_signals=False, clock=clock)
    if faults is not None:
        faults.open_log(run_dir, clock=clock, tip=lambda: log.final_hash)
    writer = None
    if video and core.overlay:
        writer = SegmentWriter(
            run_dir / "video",
            "annotated",
            fps=source.fps_hint or 15.0,
            segment_s=float(cfg.get("record.segment_s", 60)),
        )
    runner = Runner(
        core,
        source,
        eventlog=log,
        answer=answer,
        answer_delay_s=answer_delay_s,
        clock=clock,
        annotated_writer=writer,
        on_result=on_result,
    )
    try:
        summary = runner.run()
    finally:
        if writer is not None:
            writer.close()
        if faults is not None:  # before run_end, so the fault log's chain tips stay in step
            faults.close()
    log.close(reason=summary.reason)
    size = write_downlink_for(
        run_dir,
        run_id=run_id,
        protocol=protocol,
        summary=summary,
        eventlog=log,
        started_at=iso_at(base),
        ended_at=iso_at(clock()),
        max_bytes=int(cfg.get("log.downlink_max_bytes", 4096)),
    )
    sha = hashlib.sha256(log.path.read_bytes()).hexdigest()
    meta: dict[str, Any] = {
        "run_id": run_id,
        "mode": "offline",
        "input": spec,
        "simulated": simulated,
        "protocol": protocol.id,
        "log_clock": clock_src,
        "stages": core.status(),
        "answer_policy": answer if isinstance(answer, str) else None,
        "frames": summary.frames,
        "reason": summary.reason,
        "outcomes": summary.outcomes,
        "events_sha256": sha,
        "log_hash": log.final_hash,
        "downlink_bytes": size,
        "wall_s": round(summary.wall_s, 3),
        "video": [str(s.path.name) for s in writer.segments] if writer is not None else [],
    }
    if faults is not None:
        meta["faults"] = faults.summary()
    write_json(run_dir / "run.json", meta)
    return OfflineResult(run_id, run_dir, summary, log.path, sha, size, simulated)


def _flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(_flatten(v, key + "."))
        elif v is None or isinstance(v, (bool, int, float, str)):
            out[key] = v
        else:
            out[key] = str(v)
    return out


def _install_sigterm() -> Any:
    """Turn SIGTERM into KeyboardInterrupt so teardown and the run_end record still happen."""
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "SIGTERM"):
        return None

    def _raise(signum, frame):
        raise KeyboardInterrupt

    try:
        return signal.signal(signal.SIGTERM, _raise)
    except (ValueError, OSError):
        return None


def _restore_sigterm(prev: Any) -> None:
    if prev is not None:
        with contextlib.suppress(ValueError, OSError):
            signal.signal(signal.SIGTERM, prev)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the ACTUS pipeline end to end.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--live", action="store_true", help="capture.source (webcam / rtsp / file)")
    src.add_argument("--file", help="a recorded take, played at its own rate")
    src.add_argument("--replay", metavar="SCENARIO", help="SIMULATED P0 replay scenario")
    ap.add_argument("--source", help="live: override capture.source (webcam index / rtsp:// URL)")
    ap.add_argument("--config", help="override YAML on top of configs/default.yaml")
    ap.add_argument("--override", action="append", default=[], help="key.path=value")
    ap.add_argument("--run-id")
    ap.add_argument("--evidence", choices=["spatial", "replay"], help="replay only: stand-in")
    ap.add_argument("--answer", choices=["yes", "no"], help="auto-answer every Ask (unattended)")
    ap.add_argument("--seconds", type=float, help="stop after this much source time")
    ap.add_argument("--no-realtime", action="store_true", help="files/replay: run flat out")
    ap.add_argument(
        "--faults",
        action="store_true",
        help="enable fault injection: console hotkeys o d f x n and the GUI panel",
    )
    ap.add_argument(
        "--fault",
        action="append",
        default=[],
        metavar="SPEC",
        help="scheduled fault, e.g. occlude@9-12 or dim@f90-135:gamma=4 (implies --faults)",
    )
    for name in ("ui", "voice", "stream", "record"):
        ap.add_argument(f"--no-{name}", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, overrides=args.override)
    faults = None
    if args.faults or args.fault:
        from actus.app.faults import FaultController

        faults = FaultController.from_config(cfg, schedule=args.fault)
    if args.replay:
        source, source_arg = "replay", args.replay
    elif args.file:
        source, source_arg = "file", args.file
    else:
        source, source_arg = "live", args.source

    app = LiveApp(
        cfg,
        source=source,
        source_arg=source_arg,
        run_id=args.run_id,
        ui=not args.no_ui,
        voice=not args.no_voice,
        stream=not args.no_stream,
        record=not args.no_record,
        realtime=not args.no_realtime,
        answer=args.answer,
        max_seconds=args.seconds,
        evidence=args.evidence,
        faults=faults,
    )
    print(f"# run {app.run_id} -> {app.run_dir}")
    if app.faults is not None:
        print("# FAULT INJECTION ON: hotkeys o d f x n; see actus/app/faults.py")
    if not args.no_ui:
        print(f"# GUI: http://{cfg.get('ui.host')}:{cfg.get('ui.port')}  (needs the ui extra)")
    summary, run_dir = app.run()
    tag = " [SIMULATED]" if app.source.simulated else ""
    print(f"# {summary.frames} frames, {summary.dropped} dropped, {summary.reason}{tag}")
    print(f"# outcomes: {outcome_line(summary.outcomes)}")
    print(f"# run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
