from pathlib import Path

import numpy as np
import pytest

from actus.core.config import load_config
from actus.perception.handpose import (
    HandPoseEstimator,
    RawEstimate,
    RawHand,
    RawHands,
    RawPose,
    build_hand_set,
    build_pose_set,
)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models" / "perception"
HAND_MODEL = MODELS_DIR / "hand_landmarker.task"
POSE_MODEL = MODELS_DIR / "pose_landmarker_lite.task"
SAMPLE_HAND_IMG = MODELS_DIR / "_sample_hand.jpg"
SAMPLE_POSE_IMG = MODELS_DIR / "_sample_pose.jpg"

# --------------------------------------------------------------------------------------------
# Pure logic: raw dataclasses and contract builders, no MediaPipe needed.
# --------------------------------------------------------------------------------------------


def _hand(side="Left", score=0.9) -> RawHand:
    return RawHand(side, np.zeros((21, 2), dtype=np.float32), score)


def test_raw_hands_mean_conf_empty_is_zero():
    assert RawHands(()).mean_conf == 0.0


def test_raw_hands_mean_conf_averages_scores():
    hands = RawHands((_hand(score=0.8), _hand(score=0.4)))
    assert hands.mean_conf == pytest.approx(0.6)


def test_raw_estimate_confidence_uses_pose_only_when_no_hands():
    est = RawEstimate(RawHands(()), RawPose(np.zeros((33, 2)), np.zeros((33, 3)), None, 0.7))
    assert est.confidence == pytest.approx(0.7)


def test_raw_estimate_confidence_averages_pose_and_hands_when_both_present():
    hands = RawHands((_hand(score=0.9),))
    pose = RawPose(np.zeros((33, 2)), np.zeros((33, 3)), None, 0.5)
    est = RawEstimate(hands, pose)
    assert est.confidence == pytest.approx((0.5 + 0.9) / 2)


def test_raw_estimate_confidence_zero_when_nothing_found():
    est = RawEstimate(RawHands(()), RawPose(None, None, None, 0.0))
    assert est.confidence == 0.0


def test_build_hand_set_lowercases_side_and_carries_rotation():
    hands = (_hand("Left", 0.9), _hand("Right", 0.8))
    hs = build_hand_set(3, 1.5, hands, rotation_used=90)
    assert hs.frame_idx == 3 and hs.t == 1.5 and hs.rotation_used == 90
    assert [h.side for h in hs.hands] == ["left", "right"]
    assert all(h.landmarks_rack is None for h in hs.hands)


def test_build_hand_set_swaps_selfie_labels_for_an_unmirrored_camera():
    hands = (_hand("Left", 0.9), _hand("Right", 0.8))
    hs = build_hand_set(0, 0.0, hands, mirrored_input=False)
    assert [h.side for h in hs.hands] == ["right", "left"]


def test_raw_estimate_confidence_ignores_a_disabled_pose_landmarker():
    hands = RawHands((_hand("Left", 0.9), _hand("Right", 0.9)))
    est = RawEstimate(hands, RawPose(None, None, None, 0.0), pose_enabled=False)
    assert est.confidence == pytest.approx(0.9)


def test_clone_copies_settings_but_not_state():
    est = HandPoseEstimator(enable_pose=False, num_hands=1, mirrored_input=True)
    est._last_ts_ms = 1234
    c = est.clone()
    assert c is not est
    assert (c.enable_pose, c.num_hands, c.mirrored_input) == (False, 1, True)
    assert c._last_ts_ms == -1


def test_build_hand_set_rejects_unexpected_handedness():
    with pytest.raises(ValueError):
        build_hand_set(0, 0.0, (_hand("Sideways", 0.9),))


def test_build_pose_set_empty_when_no_landmarks():
    ps = build_pose_set(0, 0.0, RawPose(None, None, None, 0.0))
    assert ps.poses == ()


def test_build_pose_set_carries_world_landmarks_and_visibility():
    pose = RawPose(
        np.ones((33, 2), np.float32),
        np.ones((33, 3), np.float32),
        np.full(33, 0.8, np.float32),
        0.8,
    )
    ps = build_pose_set(2, 0.4, pose, rotation_used=180)
    assert len(ps.poses) == 1
    obs = ps.poses[0]
    assert obs.rotation_used == 180
    assert obs.conf == pytest.approx(0.8)
    assert obs.torso_vec_rack is None
    assert obs.world_landmarks.shape == (33, 3)
    assert obs.visibility.shape == (33,)


def test_handpose_estimator_from_config_reads_perception_section():
    cfg = load_config()
    est = HandPoseEstimator.from_config(cfg.perception)
    assert est.enable_hands is True
    assert est.enable_pose is True
    assert est.num_hands == 2
    assert est.pose_variant == "lite"
    assert est.mirrored_input is False


# --------------------------------------------------------------------------------------------
# Real MediaPipe integration -- skipped when the task-bundle files aren't present (a setup-time
# download, not part of the repo).
# --------------------------------------------------------------------------------------------

_missing_hand = not (HAND_MODEL.exists() and SAMPLE_HAND_IMG.exists())
_missing_pose = not (POSE_MODEL.exists() and SAMPLE_POSE_IMG.exists())


@pytest.mark.skipif(_missing_hand, reason="hand_landmarker.task / sample image not downloaded")
def test_process_detects_hands_on_a_real_image():
    import cv2

    pytest.importorskip("mediapipe")
    from actus.core.contracts import FramePacket

    img = cv2.imread(str(SAMPLE_HAND_IMG))
    est = HandPoseEstimator(enable_pose=False, hand_model_path=HAND_MODEL, min_hand_conf=0.3)
    hand_set, pose_set = est.process(FramePacket(0, 0.0, img, "cam0"))

    assert pose_set.poses == ()  # pose disabled
    assert {h.side for h in hand_set.hands} == {"left", "right"}
    for h in hand_set.hands:
        assert h.landmarks_px.shape == (21, 2)
        assert 0.0 <= h.conf <= 1.0
        assert h.landmarks_rack is None


@pytest.mark.skipif(_missing_pose, reason="pose_landmarker_lite.task / sample image not downloaded")
def test_process_detects_pose_on_a_real_image():
    import cv2

    pytest.importorskip("mediapipe")
    from actus.core.contracts import FramePacket

    img = cv2.imread(str(SAMPLE_POSE_IMG))
    est = HandPoseEstimator(enable_hands=False, pose_model_path=POSE_MODEL, min_pose_conf=0.3)
    hand_set, pose_set = est.process(FramePacket(0, 0.0, img, "cam0"))

    assert hand_set.hands == ()  # hands disabled
    assert len(pose_set.poses) == 1
    obs = pose_set.poses[0]
    assert obs.landmarks_px.shape == (33, 2)
    assert obs.world_landmarks.shape == (33, 3)
    assert obs.visibility.shape == (33,)
    assert obs.conf == pytest.approx(float(np.mean(obs.visibility)))


@pytest.mark.skipif(_missing_hand, reason="hand_landmarker.task / sample image not downloaded")
def test_estimate_tolerates_non_increasing_timestamps():
    """VIDEO-mode landmarkers require strictly increasing timestamps per instance; `estimate`
    must self-correct rather than crash when a caller (rotation TTA) probes the same frame more
    than once with the same or an out-of-order timestamp."""
    import cv2

    pytest.importorskip("mediapipe")

    img = cv2.imread(str(SAMPLE_HAND_IMG))
    est = HandPoseEstimator(enable_pose=False, hand_model_path=HAND_MODEL, min_hand_conf=0.3)

    est.estimate(img, 100)
    last_after_first = est._last_ts_ms
    est.estimate(img, 100)  # same timestamp again -- must not raise
    est.estimate(img, 50)  # an even earlier one -- must not raise
    assert est._last_ts_ms > last_after_first
