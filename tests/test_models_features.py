"""The feature spec and the self-calibrating rack frame.

The properties worth pinning down are the ones the models silently depend on: rack coordinates
really are rotation invariant, velocities really are per second (so a 30 fps take and a 15 fps
camera agree), and a missing input produces a cleared block with its flag down rather than a
zero that reads like a measurement.
"""

from __future__ import annotations

import numpy as np
import pytest

from actus.core.contracts import (
    Detection,
    DetectionSet,
    HandObs,
    HandSet,
    PoseObs,
    PoseSet,
    RackFrame,
)
from actus.models.features import FeatureLayout, FeatureParams, FrameFeaturizer
from actus.models.rackfit import MarkerFrameFitter, board_size_from
from actus.protocol.schema import load_protocol

MARKER_SIZE = 0.06


@pytest.fixture(scope="module")
def protocol():
    return load_protocol("protocols/bas_boxes.yaml")


@pytest.fixture
def featurizer(protocol):
    return FrameFeaturizer(
        protocol,
        board_size=(0.9, 0.8),
        outer_footprint={"outer": [[0.15, 0.30], [0.40, 0.55]]},
    )


def rack_frame(H=None, *, frame_idx=0, t=0.0, age=0):
    if H is None:
        H = np.diag([0.001, 0.001, 1.0])
    return RackFrame(
        True, H, None, None, (0, 1, 2, 3), 0.4, frame_idx=frame_idx, t=t, age_frames=age
    )


def hand_set(frame_idx, t, pts, side="right", conf=0.9):
    return HandSet(frame_idx, t, (HandObs(side, pts, None, conf),))


def detections(frame_idx, t, red_xy=(0.3, 0.2), box=(100.0, 100.0, 140.0, 140.0)):
    return DetectionSet(
        frame_idx,
        t,
        (
            Detection(1, "red_box", 0.8, box, red_xy),
            Detection(2, "outer_box_open", 0.6, (300.0, 300.0, 500.0, 480.0), (0.27, 0.42)),
        ),
        "test",
    )


# -- layout -------------------------------------------------------------------------------------


def test_layout_names_are_unique_and_sized(protocol):
    layout = FeatureLayout.from_protocol(protocol)
    assert layout.objects == ("red", "yellow")
    assert layout.containers == ("outer",)
    assert layout.zones == ("A", "B")
    assert len(set(layout.frame_names)) == len(layout.frame_names) == layout.n_frame
    assert len(set(layout.pair_names)) == len(layout.pair_names) == layout.n_pair


def test_vector_length_matches_the_layout(featurizer):
    ff = featurizer.update(0, 0.0, rack=rack_frame(), detections=detections(0, 0.0))
    assert ff.frame.shape == (featurizer.layout.n_frame,)
    assert ff.pair.shape == (2, 2, featurizer.layout.n_pair)
    assert ff.pair_valid.shape == (2, 2)


# -- missing inputs -----------------------------------------------------------------------------


def test_nothing_observed_is_finite_and_flagged_absent(featurizer):
    ff = featurizer.update(0, 0.0)
    names = featurizer.layout.frame_names
    assert np.all(np.isfinite(ff.frame))
    assert not ff.rack_valid
    assert not ff.pair_valid.any()
    for flag in ("rack.valid", "det.ran", "red.present", "hand_left.present", "pose.present"):
        assert ff.frame[names.index(flag)] == 0.0


def test_a_pair_is_invalid_unless_both_were_seen(featurizer):
    pts = np.random.default_rng(0).random((21, 2)) * 50 + 100
    only_hand = featurizer.update(0, 0.0, rack=rack_frame(), hands=hand_set(0, 0.0, pts))
    assert not only_hand.pair_valid.any()

    featurizer.reset()
    only_obj = featurizer.update(0, 0.0, rack=rack_frame(), detections=detections(0, 0.0))
    assert not only_obj.pair_valid.any()

    featurizer.reset()
    both = featurizer.update(
        0, 0.0, rack=rack_frame(), detections=detections(0, 0.0), hands=hand_set(0, 0.0, pts)
    )
    assert both.pair_valid[1, 0]  # right hand, red box
    assert not both.pair_valid[0, 0]  # the left hand was never seen


def test_detector_not_run_differs_from_detector_saw_nothing(featurizer):
    names = featurizer.layout.frame_names
    not_run = featurizer.update(0, 0.0, rack=rack_frame())
    featurizer.reset()
    saw_nothing = featurizer.update(
        1, 0.0, rack=rack_frame(), detections=DetectionSet(1, 0.0, (), "t")
    )
    assert not_run.frame[names.index("det.ran")] == 0.0
    assert saw_nothing.frame[names.index("det.ran")] == 1.0


# -- invariances --------------------------------------------------------------------------------


def rotate_points_90(pts, width):
    """Rotate image points 90 degrees CCW about the image centre convention used by the TTA."""
    pts = np.asarray(pts, dtype=np.float64)
    return np.stack([pts[:, 1], width - pts[:, 0]], axis=1)


def test_rack_features_survive_a_rotated_camera(protocol):
    """Roll the camera and the rack-frame numbers must not move.

    This is the orientation-agnostic claim, at the level the models actually consume it.
    """
    width = 1280.0
    rng = np.random.default_rng(1)
    pts = rng.random((21, 2)) * 200 + 200
    box = (300.0, 260.0, 360.0, 320.0)
    H = np.array([[0.0009, 0.0001, -0.05], [-0.0002, 0.0011, -0.08], [0.0, 0.0, 1.0]])

    # The same scene seen rotated: points move, and the homography absorbs the rotation so the
    # rack frame is unchanged.
    R = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, width], [0.0, 0.0, 1.0]])
    H_rot = H @ np.linalg.inv(R)
    rot_pts = rotate_points_90(pts, width)
    corners = rotate_points_90(np.array([[box[0], box[1]], [box[2], box[3]]]), width)
    rot_box = (
        float(min(corners[:, 0])),
        float(min(corners[:, 1])),
        float(max(corners[:, 0])),
        float(max(corners[:, 1])),
    )
    centre = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]])
    rack_xy = tuple(
        (np.hstack([centre, [[1.0]]]) @ H.T)[0][:2] / (np.hstack([centre, [[1.0]]]) @ H.T)[0][2]
    )

    def run(points, bbox, homography):
        fz = FrameFeaturizer(protocol, board_size=(0.9, 0.8))
        out = None
        for i in range(4):
            t = i / 30
            out = fz.update(
                i,
                t,
                rack=rack_frame(homography, frame_idx=i, t=t),
                detections=DetectionSet(
                    i, t, (Detection(1, "red_box", 0.8, bbox, rack_xy),), "test"
                ),
                hands=hand_set(i, t, points),
            )
        return fz.layout, out

    layout, upright = run(pts, box, H)
    _, rolled = run(rot_pts, rot_box, H_rot)

    rack_feature = [
        i
        for i, n in enumerate(layout.frame_names)
        if n.endswith((".x", ".y", "_x", "_y")) or "dist" in n or "in_zone" in n
    ]
    np.testing.assert_allclose(
        upright.frame[rack_feature], rolled.frame[rack_feature], rtol=1e-5, atol=1e-6
    )
    pair_cols = [
        i for i, n in enumerate(layout.pair_names) if n.endswith("_dist") or n in ("dx", "dy")
    ]
    np.testing.assert_allclose(
        upright.pair[1, 0][pair_cols], rolled.pair[1, 0][pair_cols], rtol=1e-5, atol=1e-6
    )


def test_speed_is_per_second_not_per_frame(protocol):
    """The same motion sampled at two frame rates must give the same speed."""

    def speed_at(fps):
        fz = FrameFeaturizer(protocol, board_size=(1.0, 1.0))
        idx = fz.layout.frame_names.index("red.speed")
        out = None
        for i in range(int(fps)):
            t = i / fps
            x = 0.1 + 0.2 * t  # 0.2 m/s
            out = fz.update(
                i,
                t,
                rack=rack_frame(frame_idx=i, t=t),
                detections=DetectionSet(
                    i, t, (Detection(1, "red_box", 0.9, (10.0, 10.0, 20.0, 20.0), (x, 0.5)),), "t"
                ),
            )
        return float(out.frame[idx])

    fast, slow = speed_at(30), speed_at(15)
    assert fast == pytest.approx(0.2, abs=0.02)
    assert fast == pytest.approx(slow, abs=0.02)


def test_grip_features_are_scale_free(protocol):
    """Zooming in must not change the grip shape features."""

    def grip(scale):
        fz = FrameFeaturizer(protocol, board_size=(0.9, 0.8))
        rng = np.random.default_rng(3)
        pts = (rng.random((21, 2)) * 100 + 50) * scale
        ff = fz.update(0, 0.0, rack=rack_frame(), hands=hand_set(0, 0.0, pts))
        names = fz.layout.frame_names
        return [ff.frame[names.index(f"hand_right.{k}")] for k in ("aperture", "curl", "spread")]

    np.testing.assert_allclose(grip(1.0), grip(2.5), rtol=1e-5)


def test_pose_block_reads_world_landmarks(protocol):
    fz = FrameFeaturizer(protocol, board_size=(0.9, 0.8))
    world = np.zeros((33, 3), dtype=np.float32)
    world[11] = [-0.2, 0.0, 0.0]  # left shoulder
    world[12] = [0.2, 0.0, 0.0]  # right shoulder
    pose = PoseSet(0, 0.0, (PoseObs(np.zeros((33, 2)), world, np.array([0.0, -1.0, 0.0]), 0.9, 0),))
    ff = fz.update(0, 0.0, rack=rack_frame(), pose=pose)
    names = fz.layout.frame_names
    assert ff.frame[names.index("pose.present")] == 1.0
    assert ff.frame[names.index("pose.shoulder_width")] == pytest.approx(0.4, abs=1e-5)
    assert ff.frame[names.index("pose.torso_y")] == pytest.approx(-1.0, abs=1e-5)


# -- rackfit ------------------------------------------------------------------------------------


def marker_corners(cx, cy, size=MARKER_SIZE):
    h = size / 2
    return np.array([[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]])


def project(H, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ np.asarray(H).T
    return ph[:, :2] / ph[:, 2:3]


TRUE_LAYOUT = {0: (0.0, 0.0), 1: (0.5, 0.02), 2: (0.52, 0.4), 3: (-0.01, 0.38)}


def synthetic_view(scale=1400.0, shift=(640.0, 360.0), skew=0.0):
    """A plausible camera looking at the marker plane."""
    return np.array([[scale, skew, shift[0]], [0.0, scale * 0.95, shift[1]], [0.0002, 0.0001, 1.0]])


def test_rackfit_recovers_the_marker_layout():
    H = synthetic_view()
    fitter = MarkerFrameFitter(marker_size=MARKER_SIZE, min_sightings=1)
    ids = sorted(TRUE_LAYOUT)
    corners = [project(H, marker_corners(*TRUE_LAYOUT[i])) for i in ids]
    for f in range(5):
        out = fitter.solve(ids, corners, frame_idx=f, t=f / 30)
    assert out.valid
    cal = fitter.calibration()
    for i, (x, y) in TRUE_LAYOUT.items():
        assert cal[i][0] == pytest.approx(x, abs=2e-3)
        assert cal[i][1] == pytest.approx(y, abs=2e-3)
    assert board_size_from(cal)[0] == pytest.approx(0.53, abs=5e-3)


def test_rackfit_is_rotation_invariant():
    """A rolled camera must produce the same rack coordinates for the same physical point."""
    ids = sorted(TRUE_LAYOUT)
    probe = np.array([[0.25, 0.2]])

    def rack_of_probe(view):
        fitter = MarkerFrameFitter(marker_size=MARKER_SIZE, min_sightings=1)
        corners = [project(view, marker_corners(*TRUE_LAYOUT[i])) for i in ids]
        out = fitter.solve(ids, corners, frame_idx=0, t=0.0)
        return project(out.H_img_to_rack, project(view, probe))[0]

    upright = rack_of_probe(synthetic_view())
    roll = np.array([[0.0, -1.0, 720.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rolled = rack_of_probe(roll @ synthetic_view())
    np.testing.assert_allclose(upright, rolled, atol=1e-4)
    np.testing.assert_allclose(upright, probe[0], atol=2e-3)


def test_rackfit_holds_through_an_occlusion_then_expires():
    H = synthetic_view()
    ids = sorted(TRUE_LAYOUT)
    corners = [project(H, marker_corners(*TRUE_LAYOUT[i])) for i in ids]
    fitter = MarkerFrameFitter(marker_size=MARKER_SIZE, hold_frames=10, min_sightings=1)
    fitter.solve(ids, corners, frame_idx=0, t=0.0)

    held = fitter.solve([], [], frame_idx=5, t=5 / 30)
    assert held.valid and held.age_frames == 5
    expired = fitter.solve([], [], frame_idx=40, t=40 / 30)
    assert not expired.valid


def test_rackfit_ignores_a_one_off_spurious_marker():
    """A stray pattern in the room must not be adopted into the rig."""
    H = synthetic_view()
    ids = sorted(TRUE_LAYOUT)
    corners = [project(H, marker_corners(*TRUE_LAYOUT[i])) for i in ids]
    fitter = MarkerFrameFitter(marker_size=MARKER_SIZE, min_sightings=3)
    for f in range(6):
        fitter.solve(ids, corners, frame_idx=f, t=f / 30)
    ghost_corners = [*corners, project(H, marker_corners(2.0, 2.0))]
    out = fitter.solve([*ids, 17], ghost_corners, frame_idx=6, t=0.2)
    assert out.valid
    probe = project(out.H_img_to_rack, project(H, np.array([[0.25, 0.2]])))[0]
    np.testing.assert_allclose(probe, [0.25, 0.2], atol=3e-3)


def test_featurizer_accepts_a_rackfit_frame(protocol):
    """rackfit emits the frozen RackFrame contract, so the featurizer needs no special case."""
    H = synthetic_view()
    ids = sorted(TRUE_LAYOUT)
    corners = [project(H, marker_corners(*TRUE_LAYOUT[i])) for i in ids]
    fitter = MarkerFrameFitter(marker_size=MARKER_SIZE, min_sightings=1)
    rack = fitter.solve(ids, corners, frame_idx=0, t=0.0)
    fz = FrameFeaturizer(
        protocol, params=FeatureParams(), board_size=board_size_from(fitter.calibration())
    )
    ff = fz.update(0, 0.0, rack=rack, detections=detections(0, 0.0))
    assert ff.rack_valid
    assert np.all(np.isfinite(ff.frame))
