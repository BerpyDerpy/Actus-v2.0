import math
from pathlib import Path

import numpy as np
import pytest

from actus.core.contracts import FramePacket
from actus.perception.handpose import RawEstimate, RawHand, RawHands, RawPose
from actus.perception.rotation_tta import (
    ROTATIONS,
    RotationTTA,
    rotate_image,
    rotate_point,
    unrotate_point,
)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models" / "perception"
HAND_MODEL = MODELS_DIR / "hand_landmarker.task"
POSE_MODEL = MODELS_DIR / "pose_landmarker_lite.task"
SAMPLE_HAND_IMG = MODELS_DIR / "_sample_hand.jpg"

# --------------------------------------------------------------------------------------------
# Pure geometry: no MediaPipe needed.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("degrees_ccw", ROTATIONS)
def test_rotate_then_unrotate_is_identity(degrees_ccw):
    w, h = 640.0, 480.0
    rng = np.random.default_rng(0)
    for x, y in rng.uniform([0, 0], [w, h], size=(20, 2)):
        rx, ry = rotate_point(x, y, w, h, degrees_ccw)
        bx, by = unrotate_point(rx, ry, w, h, degrees_ccw)
        assert bx == pytest.approx(x, abs=1e-6)
        assert by == pytest.approx(y, abs=1e-6)


def test_rotate_point_matches_cv2_rotate_on_pixel_grid():
    """The point-math must agree with cv2.rotate on where a marked pixel actually ends up."""
    w, h = 12, 8
    img = np.zeros((h, w, 3), dtype=np.uint8)
    px, py = 3, 2  # a marked pixel, away from any symmetry axis
    img[py, px] = (255, 255, 255)

    for degrees_ccw in ROTATIONS:
        rotated = rotate_image(img, degrees_ccw)
        ys, xs = np.nonzero(rotated.sum(axis=-1))
        assert len(xs) == 1
        actual = (float(xs[0]), float(ys[0]))
        predicted = rotate_point(px, py, w, h, degrees_ccw)
        assert predicted == pytest.approx(actual, abs=1e-6)


def test_rotate_point_rejects_bad_angle():
    with pytest.raises(ValueError):
        rotate_point(0, 0, 10, 10, 45)


def test_rotate_image_swaps_dims_for_90_and_270():
    img = np.zeros((8, 12, 3), dtype=np.uint8)  # h=8, w=12
    assert rotate_image(img, 0).shape == (8, 12, 3)
    assert rotate_image(img, 90).shape == (12, 8, 3)
    assert rotate_image(img, 180).shape == (8, 12, 3)
    assert rotate_image(img, 270).shape == (12, 8, 3)


# --------------------------------------------------------------------------------------------
# Search / cache logic, with `_estimate_at` overridden so it is independent of MediaPipe and of
# what image content actually maps to which rotation.
# --------------------------------------------------------------------------------------------


class ScriptedTTA(RotationTTA):
    """Replaces the real per-rotation MediaPipe call with a scripted confidence table, so the
    search/cache state machine can be tested deterministically and without MediaPipe."""

    def __init__(self, script: dict[int, float], **kwargs) -> None:
        super().__init__(estimator=None, **kwargs)  # type: ignore[arg-type]
        self.script = script
        self.calls: list[int] = []

    def _estimate_at(self, image: np.ndarray, ts: int, degrees_ccw: int) -> RawEstimate:
        self.calls.append(degrees_ccw)
        conf = self.script[degrees_ccw]
        pose = RawPose(np.zeros((33, 2), np.float32), np.zeros((33, 3), np.float32), None, conf)
        return RawEstimate(RawHands(()), pose)


def _blank_frame(frame_idx=0, t=0.0, w=64, h=48) -> FramePacket:
    return FramePacket(frame_idx, t, np.zeros((h, w, 3), dtype=np.uint8), "cam0")


def test_no_search_while_confident_and_not_due():
    tta = ScriptedTTA({0: 0.9, 90: 0.1, 180: 0.1, 270: 0.1}, conf_threshold=0.5, every_n_frames=10)
    for i in range(5):
        tta.process(_blank_frame(i, float(i)))
    assert tta.calls == [0] * 5  # one call per frame, always at the cached rotation
    assert tta.rotation_used == 0


def test_periodic_search_every_n_frames():
    tta = ScriptedTTA({0: 0.9, 90: 0.9, 180: 0.9, 270: 0.9}, conf_threshold=0.5, every_n_frames=3)
    for i in range(4):
        tta.process(_blank_frame(i, float(i)))
    # frames 0,1,2 cost one call each (never due, never below threshold); frame 3 is due
    # (frames_since_search reaches 3) and searches the other 3 candidates.
    assert tta.calls == [0, 0, 0, 0, 90, 180, 270]
    assert tta.rotation_used == 0  # a tie never displaces the cached rotation


def test_low_confidence_triggers_search_and_switches_rotation():
    tta = ScriptedTTA({0: 0.2, 90: 0.3, 180: 0.9, 270: 0.1}, conf_threshold=0.5, every_n_frames=100)
    hand_set, pose_set = tta.process(_blank_frame())
    assert tta.rotation_used == 180
    assert tta.calls == [0, 90, 180, 270]
    assert pose_set.poses[0].rotation_used == 180
    assert hand_set.rotation_used == 180


def test_cached_rotation_persists_once_switched():
    tta = ScriptedTTA({0: 0.2, 90: 0.1, 180: 0.9, 270: 0.1}, conf_threshold=0.5, every_n_frames=100)
    tta.process(_blank_frame(0, 0.0))
    assert tta.rotation_used == 180
    tta.calls.clear()
    tta.process(_blank_frame(1, 1.0))  # cached rotation (180) now scores 0.9: no re-search
    assert tta.calls == [180]
    assert tta.rotation_used == 180


def test_restricting_candidates_to_zero_disables_search():
    tta = ScriptedTTA({0: 0.01}, candidates=(0,), conf_threshold=0.5, every_n_frames=1)
    tta.process(_blank_frame())
    assert tta.calls == [0]  # nothing else to try
    assert tta.rotation_used == 0


def test_landmarks_are_mapped_back_to_the_original_frame():
    """A hand found at pixel (5, 2) while searching at 90 degrees must land back at the point
    `unrotate_point` predicts, in the *original* (un-rotated) frame's coordinates."""
    w, h = 64, 48

    class OneShot(RotationTTA):
        def __init__(self):
            super().__init__(  # type: ignore[arg-type]
                estimator=None, candidates=(0, 90), conf_threshold=0.5, every_n_frames=100
            )

        def _estimate_at(self, image, ts, degrees_ccw):
            if degrees_ccw == 0:
                return RawEstimate(RawHands(()), RawPose(None, None, None, 0.0))
            pts = np.array([[5.0, 2.0]] * 21, dtype=np.float32)
            hand = RawHand("Right", pts, 0.9)
            return RawEstimate(RawHands((hand,)), RawPose(None, None, None, 0.0))

    tta = OneShot()
    hand_set, _ = tta.process(_blank_frame(w=w, h=h))
    assert tta.rotation_used == 90
    expected = unrotate_point(5.0, 2.0, w, h, 90)
    got = hand_set.hands[0].landmarks_px[0]
    assert (float(got[0]), float(got[1])) == pytest.approx(expected, abs=1e-4)


def test_rejects_empty_or_invalid_candidates():
    with pytest.raises(ValueError):
        RotationTTA(estimator=None, candidates=())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RotationTTA(estimator=None, candidates=(45,))  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------------
# Real MediaPipe integration -- the build plan's literal "done when" check. Skipped when the
# task-bundle files aren't present (they are a setup-time download, not part of the repo).
# --------------------------------------------------------------------------------------------

_missing_assets = not (HAND_MODEL.exists() and SAMPLE_HAND_IMG.exists())


@pytest.mark.skipif(_missing_assets, reason="hand_landmarker.task / sample image not downloaded")
def test_rotation_tta_recovers_180_degree_roll_within_3pct_of_upright():
    import cv2

    pytest.importorskip("mediapipe")
    from actus.perception.handpose import HandPoseEstimator

    img = cv2.imread(str(SAMPLE_HAND_IMG))
    h, w = img.shape[:2]

    baseline_est = HandPoseEstimator(
        enable_pose=False, hand_model_path=HAND_MODEL, min_hand_conf=0.3
    )
    baseline_hands, _ = baseline_est.process(_frame_with(img))
    baseline = {ho.side: ho.landmarks_px for ho in baseline_hands.hands}
    assert baseline, "sample image must have detectable hands for this check to mean anything"

    rolled = rotate_image(img, 180)
    tta_est = HandPoseEstimator(enable_pose=False, hand_model_path=HAND_MODEL, min_hand_conf=0.3)
    tta = RotationTTA(tta_est, conf_threshold=0.99, every_n_frames=1)  # force the search
    rolled_hands, _ = tta.process(_frame_with(rolled, frame_idx=1, t=1.0))

    assert tta.rotation_used == 180
    assert {ho.side for ho in rolled_hands.hands} == set(baseline)

    diag = math.hypot(w, h)
    for ho in rolled_hands.hands:
        # `ho.landmarks_px` is in the *rolled* frame's own coordinates (the contract). Rotating
        # 180 degrees is self-inverse, so applying it again maps back to the upright frame's
        # coordinates for comparison against the plain (no-TTA) baseline.
        mapped_back = np.array([rotate_point(x, y, w, h, 180) for x, y in ho.landmarks_px])
        diffs = np.linalg.norm(mapped_back - baseline[ho.side], axis=1)
        assert diffs.max() < 0.03 * diag, f"{ho.side} hand off by {diffs.max() / diag:.1%} of diag"


def _frame_with(image, frame_idx=0, t=0.0) -> FramePacket:
    return FramePacket(frame_idx, t, image, "cam0")
