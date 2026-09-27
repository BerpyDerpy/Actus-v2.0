"""T-SPATIAL rack frame: synthetic camera geometry, real ArUco rendering, holds, rotations."""

import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from actus.core.contracts import FramePacket, PoseObs  # noqa: E402
from actus.spatial.rack_frame import (  # noqa: E402
    CameraSpec,
    RackFrameEstimator,
    RackGeometry,
    img_to_rack,
    rack_to_img,
    rotate_camera_matrix,
    torso_vec_rack,
    unroll_image_vector,
)

W, H = 960, 720
HFOV = 70.0
GEO = RackGeometry.load()  # configs/rack.yaml (placeholder geometry is fine for tests)
# rack points to localise: zone centres, the outer box, board corners-ish
PROBE = np.array([[0.675, 0.225], [0.675, 0.575], [0.275, 0.425], [0.1, 0.7], [0.8, 0.1]])


def camera_K(w=W, h=H, hfov=HFOV):
    return CameraSpec(approx_hfov_deg=hfov, image_size=(w, h)).matrix(w, h)[0]


def rot_x(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def roll(deg):
    """Camera roll about its optical axis (image content turns by `deg`)."""
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def camera_pose(roll_deg=0.0, tilt=(8.0, -10.0), dist_m=1.15):
    """rack → camera (R, t) for a camera ~1.15 m in front of the board centre, looking at it."""
    R = roll(roll_deg) @ rot_x(tilt[0]) @ rot_y(tilt[1])
    centre = np.array([0.45, 0.40, 0.0])
    C = centre - R.T @ np.array([0, 0, dist_m])  # camera centre in rack coordinates
    return R, -R @ C


def project(K, R, t, pts_rack):
    p = np.asarray(pts_rack, float)
    if p.shape[1] == 2:
        p = np.hstack([p, np.zeros((len(p), 1))])
    cam = p @ R.T + t
    uv = cam @ K.T
    return uv[:, :2] / uv[:, 2:3]


def marker_obs(K, R, t, ids=(0, 1, 2, 3), noise=0.0, rng=None):
    corners = []
    for i in ids:
        c = project(K, R, t, GEO.marker_corners(i))
        if noise:
            c = c + rng.normal(0, noise, c.shape)
        corners.append(c)
    return list(ids), corners


def plane_G(K, R, t):
    """rack-plane metres → image pixels homography."""
    return K @ np.column_stack([R[:, 0], R[:, 1], t])


def render_board(K, R, t, w=W, h=H):
    """Perspective render of the board with real DICT_4X4_50 markers (1 px/mm texture)."""
    bw, bh = (round(v * 1000) for v in GEO.board_size)
    tex = np.full((bh, bw), 235, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    size = round(GEO.marker_size * 1000)
    for mid, (mx, my) in GEO.markers.items():
        m = cv2.aruco.generateImageMarker(d, mid, size)
        x0, y0 = round(mx * 1000 - size / 2), round(my * 1000 - size / 2)
        tex[y0 - 10 : y0 + size + 10, x0 - 10 : x0 + size + 10] = 255  # quiet zone
        tex[y0 : y0 + size, x0 : x0 + size] = m
    S = np.diag([1e-3, 1e-3, 1.0])  # texture px (mm) → rack metres (pixel centres ~ fine)
    M = plane_G(K, R, t) @ S
    img = cv2.warpPerspective(tex, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=60)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def rotate_px(pts, rotation, w, h):
    """Where pixels of a w×h image land after cv2.rotate by `rotation` degrees CCW."""
    x, y = pts[:, 0], pts[:, 1]
    if rotation == 0:
        return pts.copy()
    if rotation == 90:
        return np.column_stack([y, w - 1 - x])
    if rotation == 180:
        return np.column_stack([w - 1 - x, h - 1 - y])
    return np.column_stack([h - 1 - y, x])


CV_ROT = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}


# ---- pure geometry ------------------------------------------------------------------------------


@pytest.mark.parametrize("roll_deg", [0, 90, 180, 270, 37])
def test_rolled_camera_gives_same_rack_positions(roll_deg):
    K = camera_K()
    R, t = camera_pose(roll_deg)
    est = RackFrameEstimator(GEO)
    ids, corners = marker_obs(K, R, t)
    rf = est.solve(ids, corners, image_size=(W, H), frame_idx=0, t=0.0)
    assert rf.valid and rf.markers_seen == (0, 1, 2, 3) and rf.age_frames == 0
    assert rf.reproj_err < 1e-3
    got = img_to_rack(rf, project(K, R, t, PROBE))
    np.testing.assert_allclose(got, PROBE, atol=1e-4)
    # the recovered 3D pose matches (K is exact here)
    R_est, _ = cv2.Rodrigues(rf.rvec)
    np.testing.assert_allclose(R_est, R, atol=1e-3)
    np.testing.assert_allclose(rf.tvec, t, atol=1e-3)


@pytest.mark.parametrize("roll_deg", [90, 180])
def test_rolled_camera_with_corner_noise_within_1cm(roll_deg):
    rng = np.random.default_rng(roll_deg)
    K = camera_K()
    results = {}
    for r in (0, roll_deg):
        R, t = camera_pose(r)
        est = RackFrameEstimator(GEO)
        ids, corners = marker_obs(K, R, t, noise=0.4, rng=rng)
        rf = est.solve(ids, corners, image_size=(W, H), frame_idx=0, t=0.0)
        assert rf.valid
        results[r] = img_to_rack(rf, project(K, R, t, PROBE))
        assert np.linalg.norm(results[r] - PROBE, axis=1).max() < 0.01
    assert np.linalg.norm(results[0] - results[roll_deg], axis=1).max() < 0.01


def test_too_few_markers_is_invalid_without_history():
    K = camera_K()
    R, t = camera_pose()
    ids, corners = marker_obs(K, R, t, ids=(0, 1))
    rf = RackFrameEstimator(GEO).solve(ids, corners, image_size=(W, H), frame_idx=0, t=0.0)
    assert not rf.valid and rf.H_img_to_rack is None and rf.markers_seen == (0, 1)


def test_unknown_ids_are_ignored_and_outlier_marker_rejected():
    K = camera_K()
    R, t = camera_pose()
    ids, corners = marker_obs(K, R, t)
    corners[2] = corners[2] + 40.0  # a mis-placed / misdetected marker
    ids.append(42)
    corners.append(np.zeros((4, 2)))
    est = RackFrameEstimator(GEO)
    rf = est.solve(ids, corners, image_size=(W, H), frame_idx=0, t=0.0)
    assert rf.valid and 42 not in rf.markers_seen
    assert est.last_info["inlier_markers"] == 3
    np.testing.assert_allclose(img_to_rack(rf, project(K, R, t, PROBE)), PROBE, atol=1e-3)


# ---- hold through occlusion ---------------------------------------------------------------------


def test_hold_through_occlusion_then_expire():
    K = camera_K()
    R, t = camera_pose()
    est = RackFrameEstimator(GEO, hold_frames=10)
    ids, corners = marker_obs(K, R, t)
    for f in range(5):
        assert est.solve(ids, corners, image_size=(W, H), frame_idx=f, t=f / 15).age_frames == 0
    H_good = est.solve(ids, corners, image_size=(W, H), frame_idx=5, t=5 / 15).H_img_to_rack
    for f in range(6, 16):  # all markers occluded for 10 frames: held
        rf = est.solve([], [], image_size=(W, H), frame_idx=f, t=f / 15)
        assert rf.valid and rf.age_frames == f - 5 and rf.markers_seen == ()
        np.testing.assert_allclose(rf.H_img_to_rack, H_good)
    rf = est.solve([], [], image_size=(W, H), frame_idx=16, t=16 / 15)
    assert not rf.valid and est.last_info["status"] == "expired"
    # the next fresh solve starts over
    assert est.solve(ids, corners, image_size=(W, H), frame_idx=17, t=17 / 15).age_frames == 0


def test_agreeing_partial_markers_extend_the_hold():
    K = camera_K()
    R, t = camera_pose()
    est = RackFrameEstimator(GEO, hold_frames=5)
    ids, corners = marker_obs(K, R, t)
    est.solve(ids, corners, image_size=(W, H), frame_idx=0, t=0.0)
    two_ids, two = marker_obs(K, R, t, ids=(0, 3))  # an arm covers the right-hand markers
    for f in range(1, 40):
        rf = est.solve(two_ids, two, image_size=(W, H), frame_idx=f, t=f / 15)
        assert rf.valid and rf.age_frames == f and rf.markers_seen == (0, 3)


def test_contradicting_markers_drop_the_hold():
    K = camera_K()
    R, t = camera_pose()
    est = RackFrameEstimator(GEO, hold_frames=45)
    ids, corners = marker_obs(K, R, t)
    est.solve(ids, corners, image_size=(W, H), frame_idx=0, t=0.0)
    R2, t2 = camera_pose(roll_deg=25)  # the camera was bumped; only 2 markers are visible
    two_ids, two = marker_obs(K, R2, t2, ids=(0, 1))
    rf = est.solve(two_ids, two, image_size=(W, H), frame_idx=1, t=1 / 15)
    assert not rf.valid and est.last_info["status"] == "contradicted"


# ---- smoothing ----------------------------------------------------------------------------------


def test_smoothing_reduces_jitter_and_snaps_on_roll():
    K = camera_K()
    R, t = camera_pose()
    pts_px = project(K, R, t, PROBE)

    def jitter(smooth_tau_s):
        rng = np.random.default_rng(1)
        est = RackFrameEstimator(GEO, smooth_tau_s=smooth_tau_s)
        out = []
        for f in range(60):
            ids, corners = marker_obs(K, R, t, noise=0.5, rng=rng)
            rf = est.solve(ids, corners, image_size=(W, H), frame_idx=f, t=f / 30)
            out.append(img_to_rack(rf, pts_px))
        return np.std(np.array(out)[10:], axis=0).max()

    assert jitter(0.15) < 0.6 * jitter(0.0)

    # camera rolled 180° mid-run: the very first frame after the roll is already right
    est = RackFrameEstimator(GEO, smooth_tau_s=0.3)
    for f in range(10):
        est.solve(*marker_obs(K, R, t), image_size=(W, H), frame_idx=f, t=f / 30)
    R2, t2 = camera_pose(roll_deg=180)
    rf = est.solve(*marker_obs(K, R2, t2), image_size=(W, H), frame_idx=10, t=10 / 30)
    np.testing.assert_allclose(img_to_rack(rf, project(K, R2, t2, PROBE)), PROBE, atol=1e-3)


# ---- real ArUco detection on rendered images ----------------------------------------------------


def _solve_image(img, rotation=0):
    est = RackFrameEstimator(GEO)
    rf = est.update(FramePacket(0, 0.0, img, "test", rotation_applied=rotation))
    return rf, est


@pytest.mark.parametrize("roll_deg", [0, 90, 180])
def test_rendered_board_physical_roll_within_1cm(roll_deg):
    K = camera_K()
    R, t = camera_pose(roll_deg)
    rf, est = _solve_image(render_board(K, R, t))
    assert rf.valid and rf.markers_seen == (0, 1, 2, 3), est.last_info
    assert rf.reproj_err < 2.0
    got = img_to_rack(rf, project(K, R, t, PROBE))
    assert np.linalg.norm(got - PROBE, axis=1).max() < 0.01


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_rendered_board_software_rotation_matches_upright(rotation):
    K = camera_K()
    R, t = camera_pose()
    img = render_board(K, R, t)
    px = project(K, R, t, PROBE)
    rf0, _ = _solve_image(img)
    rf, est = _solve_image(cv2.rotate(img, CV_ROT[rotation]), rotation)
    assert rf.valid, est.last_info
    upright = img_to_rack(rf0, px)
    rotated = img_to_rack(rf, rotate_px(px, rotation, W, H))
    assert np.linalg.norm(rotated - upright, axis=1).max() < 0.01
    assert np.linalg.norm(rotated - PROBE, axis=1).max() < 0.01
    # the PnP pose sees the same camera distance whichever way the image was turned
    assert abs(np.linalg.norm(rf.tvec) - np.linalg.norm(rf0.tvec)) < 0.02


def test_rack_to_img_inverts_img_to_rack():
    K = camera_K()
    R, t = camera_pose(37)
    rf = RackFrameEstimator(GEO).solve(*marker_obs(K, R, t), image_size=(W, H), frame_idx=0, t=0.0)
    np.testing.assert_allclose(img_to_rack(rf, rack_to_img(rf, PROBE)), PROBE, atol=1e-9)
    assert img_to_rack(None, PROBE) is None


# ---- intrinsics under software rotation ---------------------------------------------------------


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_rotate_camera_matrix_matches_cv2_rotate(rotation):
    K = np.array([[700.0, 0, 500.0], [0, 690.0, 340.0], [0, 0, 1]])
    dist = np.array([-0.2, 0.05, 0.003, -0.002, 0.0])
    rng = np.random.default_rng(0)
    P = np.column_stack([rng.uniform(-0.4, 0.4, 20), rng.uniform(-0.3, 0.3, 20), np.full(20, 1.2)])
    px, _ = cv2.projectPoints(P, np.zeros(3), np.zeros(3), K, dist)
    expected = rotate_px(px.reshape(-1, 2), rotation, W, H)
    K2, d2 = rotate_camera_matrix(K, dist, rotation, W, H)
    rz = {90: np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])}
    rz[180] = rz[90] @ rz[90]
    rz[270] = rz[180] @ rz[90]
    got, _ = cv2.projectPoints(P @ rz[rotation].T, np.zeros(3), np.zeros(3), K2, d2)
    np.testing.assert_allclose(got.reshape(-1, 2), expected, atol=1e-6)


def test_uncalibrated_matrix_is_rotation_invariant():
    cam = CameraSpec(approx_hfov_deg=70, image_size=(1280, 720))
    K0, _ = cam.matrix(1280, 720, 0)
    K90, _ = cam.matrix(720, 1280, 90)
    assert K0[0, 0] == pytest.approx(K90[0, 0])
    assert K90[0, 2] == pytest.approx((720 - 1) / 2)


# ---- body relative to the rack ------------------------------------------------------------------


def _pose_with_torso(v_cam, rotation_used=0, visibility=None):
    wl = np.zeros((33, 3))
    wl[[11, 12]] = 0.5 * np.asarray(v_cam)  # shoulders
    wl[[11, 12], 0] += [-0.15, 0.15]
    wl[[23, 24], 0] = [-0.1, 0.1]  # hips around the origin
    return PoseObs(np.zeros((33, 2)), wl, None, 0.9, rotation_used, visibility)


def roll_image_vector(v, rotation):
    th = math.radians(rotation)
    c, s = math.cos(th), math.sin(th)
    return np.array([v[0] * c + v[1] * s, -v[0] * s + v[1] * c, v[2]])


@pytest.mark.parametrize("roll_deg", [0, 90, 180])
@pytest.mark.parametrize("tta", [0, 90, 270])
def test_torso_vector_is_expressed_in_the_rack_frame(roll_deg, tta):
    K = camera_K()
    R, t = camera_pose(roll_deg)
    rf = RackFrameEstimator(GEO).solve(*marker_obs(K, R, t), image_size=(W, H), frame_idx=0, t=0.0)
    up_rack = np.array([0.0, -1.0, 0.0])  # operator upright in front of the board
    v_cam = R @ up_rack
    pose = _pose_with_torso(roll_image_vector(v_cam, tta), rotation_used=tta)
    np.testing.assert_allclose(torso_vec_rack(pose, rf), up_rack, atol=1e-3)
    # if T-PERC already rotated the world landmarks back, say so
    pose_back = _pose_with_torso(v_cam, rotation_used=tta)
    np.testing.assert_allclose(
        torso_vec_rack(pose_back, rf, world_rotated_back=True), up_rack, atol=1e-3
    )


def test_torso_vector_needs_pose_and_visible_torso():
    pose = _pose_with_torso([0, -1, 0], visibility=np.zeros(33))
    K = camera_K()
    R, t = camera_pose()
    rf = RackFrameEstimator(GEO).solve(*marker_obs(K, R, t), image_size=(W, H), frame_idx=0, t=0.0)
    assert torso_vec_rack(pose, rf) is None
    assert torso_vec_rack(_pose_with_torso([0, -1, 0]), None) is None


def test_unroll_inverts_roll():
    v = np.array([0.3, -0.8, 0.2])
    for r in (0, 90, 180, 270):
        np.testing.assert_allclose(unroll_image_vector(roll_image_vector(v, r), r), v, atol=1e-12)
