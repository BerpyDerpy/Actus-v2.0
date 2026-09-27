"""Rotation test-time augmentation for hands + pose.

MediaPipe's landmarkers assume a person standing upright in the frame. When the camera itself
is rolled -- the demo's 90 deg / 180 deg roll test, or a lack of fixed "up" in orbit -- running
at a single fixed orientation silently degrades. This module periodically probes the frame at
0/90/180/270 degrees, keeps whichever rotation gives the most confident landmarks, and maps the
winning result back into the frame's own pixel coordinates before it becomes a ``HandSet`` /
``PoseSet``. The winning rotation is cached on the instance, so most frames cost exactly one
``estimate`` call, not four: a full search only runs every ``every_n_frames`` frames, or sooner
if confidence at the cached rotation drops below ``conf_threshold``.

    tta = RotationTTA.from_config(cfg.perception)
    hand_set, pose_set = tta.process(frame_packet)
    tta.rotation_used   # the winning rotation, degrees CCW
"""

from __future__ import annotations

from typing import Any

import numpy as np

from actus.core.contracts import FramePacket, HandSet, PoseSet
from actus.perception.handpose import (
    HandPoseEstimator,
    RawEstimate,
    RawHand,
    RawHands,
    RawPose,
    build_hand_set,
    build_pose_set,
)

ROTATIONS = (0, 90, 180, 270)


# --------------------------------------------------------------------------------------------
# Point / image rotation geometry
# --------------------------------------------------------------------------------------------


def rotate_point(x: float, y: float, w: float, h: float, degrees_ccw: int) -> tuple[float, float]:
    """Where ``(x, y)`` in a ``w`` x ``h`` image lands after rotating that image
    ``degrees_ccw`` degrees counter-clockwise (0/90/180/270)."""
    if degrees_ccw == 0:
        return x, y
    if degrees_ccw == 90:
        return y, (w - 1) - x
    if degrees_ccw == 180:
        return (w - 1) - x, (h - 1) - y
    if degrees_ccw == 270:
        return (h - 1) - y, x
    raise ValueError(f"degrees_ccw must be one of {ROTATIONS}, got {degrees_ccw}")


def unrotate_point(x: float, y: float, w: float, h: float, degrees_ccw: int) -> tuple[float, float]:
    """Inverse of ``rotate_point``: map a point measured on the image *after* rotating a
    ``w`` x ``h`` image by ``degrees_ccw`` back to the pre-rotation coordinates."""
    rw, rh = (h, w) if degrees_ccw in (90, 270) else (w, h)
    return rotate_point(x, y, rw, rh, (360 - degrees_ccw) % 360)


def rotate_image(image: np.ndarray, degrees_ccw: int) -> np.ndarray:
    if degrees_ccw == 0:
        return image
    import cv2

    mapping = {
        90: cv2.ROTATE_90_COUNTERCLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_CLOCKWISE,
    }
    return cv2.rotate(image, mapping[degrees_ccw])


def _unrotate_points(pts: np.ndarray, w: int, h: int, degrees_ccw: int) -> np.ndarray:
    if degrees_ccw == 0 or pts.size == 0:
        return pts
    xy = np.array(
        [unrotate_point(float(x), float(y), w, h, degrees_ccw) for x, y in pts[:, :2]],
        dtype=pts.dtype,
    )
    return xy if pts.shape[1] == 2 else np.concatenate([xy, pts[:, 2:]], axis=1)


def _unrotate_raw(raw: RawEstimate, orig_w: int, orig_h: int, degrees_ccw: int) -> RawEstimate:
    """Map every pixel-space landmark in ``raw`` (measured on a rotated frame) back to the
    original frame's coordinates. World-space pose landmarks and confidences are unaffected."""
    if degrees_ccw == 0:
        return raw
    hands = tuple(
        RawHand(
            h.handedness,
            _unrotate_points(h.landmarks_px, orig_w, orig_h, degrees_ccw),
            h.score,
        )
        for h in raw.hands.hands
    )
    pose = raw.pose
    if pose.landmarks_px is not None:
        pose = RawPose(
            _unrotate_points(pose.landmarks_px, orig_w, orig_h, degrees_ccw),
            pose.world_landmarks,
            pose.visibility,
            pose.conf,
        )
    return RawEstimate(RawHands(hands), pose)


# --------------------------------------------------------------------------------------------
# Search + cache
# --------------------------------------------------------------------------------------------


class RotationTTA:
    """Wraps a ``HandPoseEstimator`` with periodic multi-rotation search and caching."""

    def __init__(
        self,
        estimator: HandPoseEstimator,
        *,
        candidates: tuple[int, ...] = ROTATIONS,
        conf_threshold: float = 0.5,
        every_n_frames: int = 30,
    ) -> None:
        if not candidates:
            raise ValueError("candidates must be non-empty")
        if any(c not in ROTATIONS for c in candidates):
            raise ValueError(f"candidates must be a subset of {ROTATIONS}, got {candidates}")
        self.estimator = estimator
        self.candidates = candidates
        self.conf_threshold = conf_threshold
        self.every_n_frames = max(1, every_n_frames)
        self.rotation_used = 0 if 0 in candidates else candidates[0]
        self._frames_since_search = 0

    @classmethod
    def from_config(cls, cfg: Any) -> RotationTTA:
        """``cfg`` is the ``perception`` config node (the same one passed to
        ``HandPoseEstimator.from_config``)."""
        estimator = HandPoseEstimator.from_config(cfg)
        tta_cfg = cfg.rotation_tta
        if not bool(tta_cfg.get("enabled", True)):
            return cls(estimator, candidates=(0,))
        return cls(
            estimator,
            conf_threshold=float(tta_cfg.get("conf_threshold", 0.5)),
            every_n_frames=int(tta_cfg.get("every_n_frames", 30)),
        )

    def process(
        self, frame: FramePacket, timestamp_ms: int | None = None
    ) -> tuple[HandSet, PoseSet]:
        ts = timestamp_ms if timestamp_ms is not None else round(frame.t_capture * 1000)
        h, w = frame.image.shape[:2]

        due = self._frames_since_search >= self.every_n_frames
        raw = self._estimate_at(frame.image, ts, self.rotation_used)
        if due or raw.confidence < self.conf_threshold:
            raw, self.rotation_used = self._search(
                frame.image, ts, current_rotation=self.rotation_used, current_raw=raw
            )
            self._frames_since_search = 0
        else:
            self._frames_since_search += 1

        unrotated = _unrotate_raw(raw, w, h, self.rotation_used)
        fi, t, rot = frame.frame_idx, frame.t_capture, self.rotation_used
        hand_set = build_hand_set(fi, t, unrotated.hands.hands, rotation_used=rot)
        pose_set = build_pose_set(fi, t, unrotated.pose, rotation_used=rot)
        return hand_set, pose_set

    def _estimate_at(self, image: np.ndarray, ts: int, degrees_ccw: int) -> RawEstimate:
        return self.estimator.estimate(rotate_image(image, degrees_ccw), ts)

    def _search(
        self, image: np.ndarray, ts: int, *, current_rotation: int, current_raw: RawEstimate
    ) -> tuple[RawEstimate, int]:
        best_raw, best_rot, best_conf = current_raw, current_rotation, current_raw.confidence
        for cand in self.candidates:
            if cand == current_rotation:
                continue
            raw = self._estimate_at(image, ts, cand)
            if raw.confidence > best_conf:
                best_raw, best_rot, best_conf = raw, cand, raw.confidence
        return best_raw, best_rot
