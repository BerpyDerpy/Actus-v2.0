"""Spatial lane façade: one call per frame turns perception into rack-frame world state.

    est = SpatialEstimator.from_config(load_config())
    sf = est.update(frame_idx, t, frame=packet, detections=dets, hands=hands, pose=pose)
    for topic, msg in sf.messages():
        bus.publish(topic, msg)

``update`` solves the rack frame (from ``frame``, unless a ``RackFrame`` is passed), fills in
``Detection.bbox_rack_xy``, ``HandObs.landmarks_rack`` (the ray/rack-plane intersection, z = 0:
not true hand depth) and ``PoseObs.torso_vec_rack``, advances the ``WorldModel`` and emits
``Predicates`` and one ``StepEvidence`` per protocol step. ``SpatialFrame.stats()`` flattens the
rack status, visibility, object locations and torso vector into a ``Stats`` message
(``source="spatial"``) for the UI mini-map and the log.

Check it on the synthetic replay (rack solved from the rendered ArUco markers, the replay's own
rack positions discarded):  ``python -m actus.spatial.estimator --scenario skip_S2``
"""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from actus.core.contracts import (
    DetectionSet,
    FramePacket,
    HandObs,
    HandSet,
    PoseSet,
    Predicates,
    RackFrame,
    Stats,
    StepEvidence,
    Topic,
)
from actus.protocol.schema import Protocol, load_protocol
from actus.spatial.predicates import EvidenceModel
from actus.spatial.rack_frame import (
    RackFrameEstimator,
    RackGeometry,
    apply_homography,
    img_to_rack,
    torso_vec_rack,
)
from actus.spatial.world_state import (
    HAND_KEYPOINTS,
    HandInput,
    SpatialParams,
    WorldModel,
    WorldSnapshot,
)


@dataclass(frozen=True)
class SpatialFrame:
    frame_idx: int
    t: float
    rack: RackFrame | None
    detections: DetectionSet | None  # with bbox_rack_xy filled where possible
    hands: HandSet | None  # with landmarks_rack filled where possible
    pose: PoseSet | None  # with torso_vec_rack filled where possible
    world: WorldSnapshot
    predicates: Predicates
    evidence: tuple[StepEvidence, ...]

    def stats(self) -> Stats:
        v: dict[str, Any] = {"visibility": round(self.world.visibility, 4)}
        rk = self.rack
        v["rack.state"] = self.world.rack_state
        if rk is not None:
            v["rack.markers"] = len(rk.markers_seen)
            v["rack.age_frames"] = rk.age_frames
            v["rack.reproj_px"] = None if rk.reproj_err is None else round(rk.reproj_err, 3)
        for name, o in self.world.objects.items():
            v[f"{name}.loc"] = o.location
            v[f"{name}.x"] = None if o.pos is None else round(o.pos[0], 4)
            v[f"{name}.y"] = None if o.pos is None else round(o.pos[1], 4)
            if o.p_open is not None:
                v[f"{name}.p_open"] = round(o.p_open, 4)
        torso = None
        if self.pose is not None:
            torso = next(
                (p.torso_vec_rack for p in self.pose.poses if p.torso_vec_rack is not None), None
            )
        for i, ax in enumerate("xyz"):
            v[f"torso.{ax}"] = None if torso is None else round(float(torso[i]), 4)
        return Stats("spatial", v, t=self.t, frame_idx=self.frame_idx)

    def messages(self) -> list[tuple[Topic, Any]]:
        """What the spatial lane publishes, in order."""
        out: list[tuple[Topic, Any]] = []
        if self.rack is not None:
            out.append((Topic.RACK, self.rack))
        out.append((Topic.PREDICATES, self.predicates))
        out += [(Topic.EVIDENCE, e) for e in self.evidence]
        out.append((Topic.STATS, self.stats()))
        return out


class SpatialEstimator:
    def __init__(
        self,
        protocol: Protocol | str,
        *,
        geometry: RackGeometry | None = None,
        params: SpatialParams | None = None,
        rack_estimator: RackFrameEstimator | None = None,
    ) -> None:
        self.protocol = protocol if isinstance(protocol, Protocol) else load_protocol(protocol)
        self.params = params or SpatialParams()
        if rack_estimator is None:
            rack_estimator = RackFrameEstimator(geometry)
        self.rack_estimator = rack_estimator
        self.geometry = rack_estimator.geometry
        footprints: dict[str, Any] = {}
        world_probe = WorldModel(self.protocol, self.params)
        if self.geometry.outer_box_footprint and len(world_probe.container_names) == 1:
            footprints[world_probe.container_names[0]] = self.geometry.outer_box_footprint
        self.world = WorldModel(self.protocol, self.params, footprints=footprints)
        self.evidence_model = EvidenceModel(
            self.protocol, self.params, movables=self.world.movable_names
        )

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None = None, protocol=None) -> SpatialEstimator:
        from actus.core.config import load_config

        cfg = cfg if cfg is not None else load_config()
        rack_cfg = cfg["rack"]
        spatial_cfg = cfg.get("spatial") if hasattr(cfg, "get") else None
        params = SpatialParams.from_mapping(dict(spatial_cfg or {}))
        proto = protocol if protocol is not None else cfg["protocol"]["path"]
        return cls(proto, params=params, rack_estimator=RackFrameEstimator.from_config(rack_cfg))

    def reset(self) -> None:
        self.rack_estimator.reset()
        self.world.reset()
        self.evidence_model.reset()

    # -- per frame ------------------------------------------------------------------------------

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
        hoi: Mapping[str, float] | None = None,
    ) -> SpatialFrame:
        """Fold one frame's perception into the world state.

        Pass ``frame`` to solve the rack frame here, or ``rack`` if it was solved elsewhere;
        with neither, detections must already carry ``bbox_rack_xy`` (e.g. the replay source).
        ``hands=None`` means no hand model ran; an empty ``HandSet`` means it saw no hand.
        """
        if rack is None and frame is not None:
            rack = self.rack_estimator.update(frame)
        if rack is None:
            rack_state = "none"
        elif not rack.valid:
            rack_state = "invalid"
        else:
            rack_state = "held" if rack.age_frames else "fresh"

        detections = _project_detections(detections, rack)
        hands, hand_input = _project_hands(hands, rack)
        pose = self._project_pose(pose, rack)
        snap = self.world.update(
            frame_idx, t, detections=detections, hands=hand_input, hoi=hoi, rack_state=rack_state
        )
        return SpatialFrame(
            frame_idx,
            t,
            rack,
            detections,
            hands,
            pose,
            snap,
            self.evidence_model.predicates(snap),
            self.evidence_model.evidence(snap),
        )

    def _project_pose(self, pose: PoseSet | None, rack: RackFrame | None) -> PoseSet | None:
        if pose is None:
            return None
        poses = []
        for p in pose.poses:
            v = torso_vec_rack(p, rack, world_rotated_back=self.params.world_landmarks_rotated_back)
            poses.append(p if v is None else dataclasses.replace(p, torso_vec_rack=v))
        return PoseSet(pose.frame_idx, pose.t, poses)


def _project_detections(dets: DetectionSet | None, rack: RackFrame | None):
    if dets is None or rack is None or not rack.valid:
        return dets
    todo = [d for d in dets.detections if d.bbox_rack_xy is None]
    if not todo:
        return dets
    centres = [
        ((d.bbox_xyxy[0] + d.bbox_xyxy[2]) / 2, (d.bbox_xyxy[1] + d.bbox_xyxy[3]) / 2) for d in todo
    ]
    xy = apply_homography(rack.H_img_to_rack, centres)  # type: ignore[arg-type]
    it = iter(xy)
    out = []
    for d in dets.detections:
        if d.bbox_rack_xy is None:
            p = next(it)
            d = dataclasses.replace(d, bbox_rack_xy=None if np.isnan(p).any() else tuple(p))
        out.append(d)
    return DetectionSet(dets.frame_idx, dets.t, out, dets.backend)


def _project_hands(hands: HandSet | None, rack: RackFrame | None):
    if hands is None:
        return None, HandInput(observed=False)
    out: list[HandObs] = []
    px_keys, rack_keys = [], []
    for h in hands.hands:
        lm_rack = img_to_rack(rack, h.landmarks_px)
        if lm_rack is not None and np.isfinite(lm_rack).all():
            h = dataclasses.replace(h, landmarks_rack=np.hstack([lm_rack, np.zeros((21, 1))]))
            rack_keys.append(lm_rack[list(HAND_KEYPOINTS)])
        px_keys.append(np.asarray(h.landmarks_px, dtype=np.float64)[list(HAND_KEYPOINTS)])
        out.append(h)
    points_rack = tuple(rack_keys) if len(rack_keys) == len(px_keys) else None
    projected = HandSet(hands.frame_idx, hands.t, out, hands.rotation_used)
    return projected, HandInput(True, tuple(px_keys), points_rack)


# --------------------------------------------------------------------------------------------
# CLI: run on the synthetic replay
# --------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from actus.capture.replay import SCENARIOS, ReplaySource

    ap = argparse.ArgumentParser(description="Run the spatial lane on a replay scenario.")
    ap.add_argument("--scenario", default="correct", choices=sorted(SCENARIOS))
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=15, help="print every Nth frame")
    ap.add_argument(
        "--replay-xy",
        action="store_true",
        help="use the replay's rack positions instead of solving the rack from rendered frames",
    )
    args = ap.parse_args(argv)

    src = ReplaySource(args.scenario, fps=args.fps, seed=args.seed, render=not args.replay_xy)
    est = SpatialEstimator(src.protocol)
    steps = src.protocol.step_ids
    movables = est.world.movable_names
    containers = est.world.container_names
    print(f"# SIMULATED replay:{args.scenario} ({len(src)} frames @ {src.fps:g} fps)")
    print(
        f"{'frame':>5} {'t':>6} {'rack':>7} {'vis':>4} | "
        + " ".join(f"{c:>7}" for c in containers)
        + " "
        + " ".join(f"{m:>10}" for m in movables)
        + " | "
        + " ".join(f"{s:>4}" for s in steps)
        + "  truth"
    )
    p_enter = src.protocol.decision.p_enter
    first: dict[str, float] = {}
    for tick in src:
        dets = tick.detections
        if not args.replay_xy:  # make the lane project detections itself
            dets = DetectionSet(
                dets.frame_idx,
                dets.t,
                [dataclasses.replace(d, bbox_rack_xy=None) for d in dets.detections],
                dets.backend,
            )
        sf = est.update(tick.frame_idx, tick.t, frame=tick.frame, detections=dets)
        for e in sf.evidence:
            if e.p_complete >= p_enter and e.step_id not in first:
                first[e.step_id] = e.t
        if tick.frame_idx % args.stride == 0:
            w = sf.world
            print(
                f"{tick.frame_idx:5d} {tick.t:6.2f} {w.rack_state:>7} {w.visibility:4.2f} | "
                + " ".join(f"{w.objects[c].location:>7}" for c in containers)
                + " "
                + " ".join(f"{w.objects[m].location:>10}" for m in movables)
                + " | "
                + " ".join(f"{e.p_complete:4.2f}" for e in sf.evidence)
                + f"  {tick.label}"
            )
    order = sorted(first, key=first.get)  # type: ignore[arg-type]
    print(
        "# evidence >= p_enter first reached: " + ", ".join(f"{s}@{first[s]:.1f}s" for s in order)
    )
    print("# never reached: " + (", ".join(s for s in steps if s not in first) or "-"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
