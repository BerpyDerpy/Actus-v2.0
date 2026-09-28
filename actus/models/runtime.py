"""The model lane's façade: one call per frame, for T-INT to wire into the pipeline.

    stack = ModelStack.from_config(cfg, protocol)
    out = stack.update(frame_idx, t, rack=rack, detections=dets, hands=hands, pose=pose)
    sf = spatial.update(frame_idx, t, rack=rack, detections=dets, hands=hands, pose=pose,
                        hoi=out.hoi)                       # HOI sharpens <object>.held
    evidence = stack.fuse(sf.evidence)                     # temporal joins as one more source
    for topic, msg in out.messages():
        bus.publish(topic, msg)

Order matters: the models run *before* the spatial estimator, because the world model takes
``hoi`` as an input. Both read the same perception output, so nothing is computed twice except
the rack projection, which is cheap.

Degrading, on purpose
---------------------
Nothing here is required for the pipeline to run. If the model files are missing -- which is the
state of the repo until the detector can see the objects well enough to train HOI -- the stack
reports ``enabled = False``, returns an empty ``hoi`` mapping and no ``ActionScores``, and
``fuse`` passes the spatial evidence through untouched. The engine then runs on rule evidence
alone, exactly as it does today. ``status()`` says which models actually loaded, so the UI and
the run log can show it rather than implying a model is contributing when none is.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from actus.core.contracts import (
    ActionScores,
    DetectionSet,
    FramePacket,
    HandSet,
    PoseSet,
    RackFrame,
    Stats,
    StepEvidence,
    Topic,
)
from actus.models.features import FeatureLayout, FeatureParams, FrameFeatures, FrameFeaturizer
from actus.models.fusion import DEFAULT_WEIGHT, EvidenceFusion
from actus.models.hoi import HOIModel, HOIResult
from actus.models.temporal import DEFAULT_FEATURE_FPS, TemporalModel


@dataclass(frozen=True)
class ModelFrame:
    """One frame of model output."""

    frame_idx: int
    t: float
    features: FrameFeatures
    hoi: dict[str, float] = field(default_factory=dict)  # {object: p(holding)}; absent = unknown
    hoi_result: HOIResult | None = None
    actions: ActionScores | None = None

    def messages(self) -> list[tuple[Topic, Any]]:
        return [] if self.actions is None else [(Topic.ACTIONS, self.actions)]


class ModelStack:
    """Featurizer + HOI + temporal + evidence fusion, as one per-frame call."""

    def __init__(
        self,
        protocol,
        *,
        hoi: HOIModel | None = None,
        temporal: TemporalModel | None = None,
        params: FeatureParams | None = None,
        board_size: tuple[float, float] = (1.0, 1.0),
        outer_footprint: Mapping[str, Any] | None = None,
        fusion_weight: float = DEFAULT_WEIGHT,
        fusion_enabled: bool = True,
    ) -> None:
        self.protocol = protocol
        self.layout = FeatureLayout.from_protocol(protocol)
        self.featurizer = FrameFeaturizer(
            protocol,
            params=params,
            layout=self.layout,
            board_size=board_size,
            outer_footprint=outer_footprint,
        )
        self.hoi = hoi
        self.temporal = temporal
        self.fusion = (
            EvidenceFusion(protocol.step_ids, weight=fusion_weight)
            if (fusion_enabled and temporal is not None)
            else None
        )
        self._last_actions: ActionScores | None = None

    # -- construction --------------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None = None, protocol=None) -> ModelStack:
        from actus.core.config import load_config, resolve_path

        cfg = cfg if cfg is not None else load_config()
        section = cfg["models"]
        if protocol is None:
            from actus.protocol.schema import load_protocol

            protocol = load_protocol(cfg["protocol"]["path"])

        enabled = bool(section.get("enabled", True))
        prefer_onnx = bool(section.get("prefer_onnx", True))
        hoi = temporal = None
        if enabled:
            hoi = _load_quietly(HOIModel, resolve_path(section["hoi"]), prefer_onnx)
            temporal = _load_quietly(TemporalModel, resolve_path(section["temporal"]), prefer_onnx)

        geometry = _geometry(cfg)
        fusion_cfg = dict(section.get("fusion", {}) or {})
        return cls(
            protocol,
            hoi=hoi,
            temporal=temporal,
            board_size=geometry.board_size if geometry else (1.0, 1.0),
            outer_footprint=(
                {"outer": geometry.outer_box_footprint}
                if geometry and geometry.outer_box_footprint
                else None
            ),
            fusion_weight=float(fusion_cfg.get("temporal_weight", DEFAULT_WEIGHT)),
            fusion_enabled=bool(fusion_cfg.get("enabled", True)),
        )

    @property
    def enabled(self) -> bool:
        return self.hoi is not None or self.temporal is not None

    def status(self) -> dict[str, Any]:
        return {
            "hoi": self.hoi.backend if self.hoi else None,
            "temporal": self.temporal.backend if self.temporal else None,
            "feature_fps": self.temporal.feature_fps if self.temporal else DEFAULT_FEATURE_FPS,
            "fusion": self.fusion.weight if self.fusion else 0.0,
            "hoi_coverage": round(self.hoi.coverage(), 4) if self.hoi else None,
        }

    def reset(self) -> None:
        self.featurizer.reset()
        if self.temporal is not None:
            self.temporal.reset()
        if self.hoi is not None:
            self.hoi.reset()
        if self.fusion is not None:
            self.fusion.reset()
        self._last_actions = None

    # -- per frame -----------------------------------------------------------------------------

    def update(
        self,
        frame_idx: int,
        t: float,
        *,
        frame: FramePacket | None = None,
        rack: RackFrame | None = None,
        detections: DetectionSet | None = None,
        hands: HandSet | None = None,
        pose: PoseSet | None = None,
    ) -> ModelFrame:
        del frame  # accepted for call-site symmetry with SpatialEstimator.update
        features = self.featurizer.update(
            frame_idx, t, rack=rack, detections=detections, hands=hands, pose=pose
        )
        hoi_result = self.hoi.predict(features) if self.hoi is not None else None
        actions = self.temporal.update(features) if self.temporal is not None else None
        if actions is not None:
            self._last_actions = actions
        return ModelFrame(
            frame_idx,
            t,
            features,
            hoi_result.holding() if hoi_result else {},
            hoi_result,
            actions,
        )

    def fuse(self, evidence: Iterable[StepEvidence]) -> tuple[StepEvidence, ...]:
        """Fold the latest action scores into the spatial lane's evidence (see fusion.py)."""
        evidence = tuple(evidence)
        if self.fusion is None:
            return evidence
        return self.fusion.fuse_all(evidence, self._last_actions)

    def stats(self, frame: ModelFrame) -> Stats:
        values: dict[str, Any] = {f"hoi.{k}": round(v, 4) for k, v in frame.hoi.items()}
        if frame.actions is not None:
            label, score = frame.actions.top()
            values["action.top"] = label
            values["action.score"] = round(score, 4)
        values["models.enabled"] = self.enabled
        return Stats("models", values, t=frame.t, frame_idx=frame.frame_idx)


def _geometry(cfg):
    try:
        from actus.spatial.rack_frame import RackGeometry

        return RackGeometry.load(cfg["rack"].get("config", "configs/rack.yaml"))
    except Exception:  # pragma: no cover - a missing rack.yaml must not break the pipeline
        return None


def _load_quietly(cls, path: Path, prefer_onnx: bool):
    """Load a model if it is there; a missing or stale file disables it rather than crashing."""
    if not Path(path).with_suffix(".npz").exists():
        return None
    try:
        return cls.load(path, prefer_onnx=prefer_onnx)
    except Exception as exc:
        print(f"models: {path} not loaded ({type(exc).__name__}: {exc})")
        return None
