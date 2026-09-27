"""Hands + pose from a single frame: MediaPipe Tasks Hand Landmarker and Pose Landmarker,
both in VIDEO mode, with world landmarks.

``HandPoseEstimator`` is the low-level wrapper: ``estimate(image, timestamp_ms)`` runs both
landmarkers on whatever image it is given (BGR, as ``FramePacket.image``) and returns the raw
MediaPipe output as plain arrays -- in the coordinate space of *that* image. ``process(frame)``
is the convenience path straight to the contracts (``HandSet`` / ``PoseSet``), with no rotation
search.

``rotation_tta.py`` builds on the raw ``estimate`` interface to search rotations and map the
result back to the frame's own coordinates; it is the path a real pipeline uses. This module's
``process`` exists for the simple case (rotation TTA disabled, or a known-upright camera) and as
the primitive rotation_tta calls into.

VIDEO-mode landmarkers require strictly increasing timestamps per instance. ``estimate`` clamps
every timestamp to be greater than the last one it was given, so a caller that probes the same
frame more than once (rotation TTA) never has to track this itself.

    est = HandPoseEstimator.from_config(cfg.perception)
    hand_set, pose_set = est.process(frame_packet)

Sanity check against a live camera (FPS only; add the detector separately for the full
pipeline check)::

    python -m actus.perception.handpose --source 0 --seconds 10
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from actus.core.config import resolve_path
from actus.core.contracts import FramePacket, HandObs, HandSet, PoseObs, PoseSet

DEFAULT_HAND_MODEL = "models/perception/hand_landmarker.task"
DEFAULT_POSE_MODEL_TMPL = "models/perception/pose_landmarker_{variant}.task"


# --------------------------------------------------------------------------------------------
# Raw (pre-contract) results -- internal to perception, in the coordinate space of whatever
# image `estimate` was given.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RawHand:
    handedness: str  # MediaPipe's "Left" / "Right", as reported (not yet lower-cased)
    landmarks_px: np.ndarray  # (21, 2) pixel coords in the image passed to `estimate`
    score: float  # handedness confidence


@dataclass(frozen=True)
class RawHands:
    hands: tuple[RawHand, ...]

    @property
    def mean_conf(self) -> float:
        return float(np.mean([h.score for h in self.hands])) if self.hands else 0.0


@dataclass(frozen=True)
class RawPose:
    landmarks_px: np.ndarray | None  # (33, 2) pixel coords in the image passed to `estimate`
    world_landmarks: np.ndarray | None  # (33, 3) metres, hip-centred
    visibility: np.ndarray | None  # (33,)
    conf: float  # mean visibility, 0 if no pose found


@dataclass(frozen=True)
class RawEstimate:
    hands: RawHands
    pose: RawPose

    @property
    def confidence(self) -> float:
        """Combined score rotation TTA uses to rank candidate rotations."""
        signals = [self.pose.conf]
        if self.hands.hands:
            signals.append(self.hands.mean_conf)
        return float(np.mean(signals))


# --------------------------------------------------------------------------------------------
# Estimator
# --------------------------------------------------------------------------------------------


class HandPoseEstimator:
    """Owns the MediaPipe landmarkers. Not thread-safe; use one per capture thread."""

    def __init__(
        self,
        *,
        enable_hands: bool = True,
        enable_pose: bool = True,
        num_hands: int = 2,
        pose_variant: str = "lite",
        hand_model_path: str | Path | None = None,
        pose_model_path: str | Path | None = None,
        min_hand_conf: float = 0.5,
        min_pose_conf: float = 0.5,
    ) -> None:
        self.enable_hands = enable_hands
        self.enable_pose = enable_pose
        self.num_hands = num_hands
        self.pose_variant = pose_variant
        self.hand_model_path = hand_model_path
        self.pose_model_path = pose_model_path
        self.min_hand_conf = min_hand_conf
        self.min_pose_conf = min_pose_conf
        self._hand_landmarker: Any = None
        self._pose_landmarker: Any = None
        self._loaded = False
        self._last_ts_ms = -1

    @classmethod
    def from_config(cls, cfg: Any) -> HandPoseEstimator:
        return cls(
            enable_hands=bool(cfg.get("hands", True)),
            enable_pose=bool(cfg.get("pose", True)),
            num_hands=int(cfg.get("num_hands", 2)),
            pose_variant=cfg.get("pose_model", "lite"),
            hand_model_path=cfg.get("hand_model_path"),
            pose_model_path=cfg.get("pose_model_path"),
            min_hand_conf=float(cfg.get("min_hand_conf", 0.5)),
            min_pose_conf=float(cfg.get("min_pose_conf", 0.5)),
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        from mediapipe.tasks.python import BaseOptions, vision

        if self.enable_hands:
            hand_path = resolve_path(self.hand_model_path or DEFAULT_HAND_MODEL)
            self._hand_landmarker = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(hand_path)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=self.num_hands,
                    min_hand_detection_confidence=self.min_hand_conf,
                )
            )
        if self.enable_pose:
            variant = self.pose_variant
            pose_path = self.pose_model_path or DEFAULT_POSE_MODEL_TMPL.format(variant=variant)
            self._pose_landmarker = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(resolve_path(pose_path))),
                    running_mode=vision.RunningMode.VIDEO,
                    min_pose_detection_confidence=self.min_pose_conf,
                )
            )
        self._loaded = True

    def estimate(self, image: np.ndarray, timestamp_ms: int) -> RawEstimate:
        """Run both landmarkers on ``image`` as given (no rotation, no coordinate remapping)."""
        self._ensure_loaded()
        import mediapipe as mp

        ts = max(int(timestamp_ms), self._last_ts_ms + 1)
        self._last_ts_ms = ts
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=_to_rgb(image))
        h, w = image.shape[:2]

        hands = RawHands(())
        if self._hand_landmarker is not None:
            hands = _raw_hands(self._hand_landmarker.detect_for_video(mp_image, ts), w, h)
        pose = RawPose(None, None, None, 0.0)
        if self._pose_landmarker is not None:
            pose = _raw_pose(self._pose_landmarker.detect_for_video(mp_image, ts), w, h)
        return RawEstimate(hands, pose)

    def process(
        self, frame: FramePacket, timestamp_ms: int | None = None
    ) -> tuple[HandSet, PoseSet]:
        """Estimate directly on ``frame.image``; ``rotation_used`` is always 0. For the
        adaptive path, use ``rotation_tta.RotationTTA`` instead."""
        ts = timestamp_ms if timestamp_ms is not None else round(frame.t_capture * 1000)
        raw = self.estimate(frame.image, ts)
        fi, t = frame.frame_idx, frame.t_capture
        hand_set = build_hand_set(fi, t, raw.hands.hands, rotation_used=0)
        pose_set = build_pose_set(fi, t, raw.pose, rotation_used=0)
        return hand_set, pose_set

    def close(self) -> None:
        if self._hand_landmarker is not None:
            self._hand_landmarker.close()
            self._hand_landmarker = None
        if self._pose_landmarker is not None:
            self._pose_landmarker.close()
            self._pose_landmarker = None
        self._loaded = False


# --------------------------------------------------------------------------------------------
# Contract builders -- shared by the plain path here and by rotation_tta after it un-rotates.
# --------------------------------------------------------------------------------------------


def build_hand_set(
    frame_idx: int, t: float, hands: tuple[RawHand, ...], *, rotation_used: int = 0
) -> HandSet:
    obs = tuple(
        HandObs(
            side=_side(h.handedness), landmarks_px=h.landmarks_px, landmarks_rack=None, conf=h.score
        )
        for h in hands
    )
    return HandSet(frame_idx, t, obs, rotation_used)


def build_pose_set(frame_idx: int, t: float, pose: RawPose, *, rotation_used: int = 0) -> PoseSet:
    if pose.landmarks_px is None:
        return PoseSet(frame_idx, t, ())
    obs = PoseObs(
        landmarks_px=pose.landmarks_px,
        world_landmarks=pose.world_landmarks,
        torso_vec_rack=None,
        conf=pose.conf,
        rotation_used=rotation_used,
        visibility=pose.visibility,
    )
    return PoseSet(frame_idx, t, (obs,))


def _side(handedness: str) -> str:
    s = handedness.strip().lower()
    if s not in ("left", "right"):
        raise ValueError(f"unexpected MediaPipe handedness {handedness!r}")
    return s


def _to_rgb(image: np.ndarray) -> np.ndarray:
    import cv2

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _raw_hands(result: Any, w: int, h: int) -> RawHands:
    hands = []
    for lm_list, handed_list in zip(result.hand_landmarks, result.handedness, strict=True):
        pts = np.array([[lm.x * w, lm.y * h] for lm in lm_list], dtype=np.float32)
        top = max(handed_list, key=lambda c: c.score)
        hands.append(RawHand(top.category_name, pts, float(top.score)))
    return RawHands(tuple(hands))


def _raw_pose(result: Any, w: int, h: int) -> RawPose:
    if not result.pose_landmarks:
        return RawPose(None, None, None, 0.0)
    lm_list = result.pose_landmarks[0]
    world_list = result.pose_world_landmarks[0]
    pts = np.array([[lm.x * w, lm.y * h] for lm in lm_list], dtype=np.float32)
    world = np.array([[p.x, p.y, p.z] for p in world_list], dtype=np.float32)
    vis = np.array([lm.visibility for lm in lm_list], dtype=np.float32)
    conf = float(np.mean(vis)) if vis.size else 0.0
    return RawPose(pts, world, vis, conf)


# --------------------------------------------------------------------------------------------
# CLI: live FPS sanity check
# --------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Live sanity check: hands + pose FPS from a camera.")
    ap.add_argument("--source", default="0", help="camera index, file path, or rtsp:// URL")
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args(argv)
    source: int | str = int(args.source) if args.source.lstrip("-").isdigit() else args.source

    from actus.capture.camera import Camera
    from actus.core.bus import Bus
    from actus.core.contracts import Topic

    bus = Bus()
    sub = bus.subscribe(Topic.FRAMES, maxsize=4, policy="drop_oldest", name="sanity")
    cam = Camera(source, source_id="sanity")
    est = HandPoseEstimator()
    cam.start(bus)
    n, t0 = 0, time.monotonic()
    try:
        while time.monotonic() - t0 < args.seconds:
            env = sub.get(timeout=1.0)
            if env is None:
                continue
            hand_set, pose_set = est.process(env.payload)
            n += 1
            print(
                f"frame {env.payload.frame_idx}: "
                f"hands={len(hand_set.hands)} pose={len(pose_set.poses)}"
            )
    finally:
        cam.stop()
        est.close()
        bus.close()
    dt = time.monotonic() - t0
    print(f"{n} frames in {dt:.1f}s = {n / dt:.1f} FPS" if dt > 0 else "no frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
