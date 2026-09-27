"""Rack coordinate frame from ArUco markers.

Every spatial quantity in ACTUS is expressed in the payload-rack frame (``configs/rack.yaml``):
metres, origin at the board's top-left corner as seen facing the rack, +x right, +y down,
+z into the rack. Because the frame is re-solved from the markers on every frame, positions do
not move when the camera is rolled 90° or 180° (the orientation-agnostic requirement).

Per frame, ``RackFrameEstimator``:

1. detects the configured ArUco markers (unknown ids are ignored);
2. fits the image ↔ rack-plane homography to all marker corners (RANSAC, then least squares
   on the inliers). The homography does not depend on the camera intrinsics, so it is exact
   for an uncalibrated pinhole camera. ``reproj_err`` is the RMS pixel error of this fit;
3. is valid only with ≥ ``min_markers`` inlier markers and ``reproj_err`` ≤
   ``max_reproj_err_px``;
4. smooths the solution by filtering where the four board corners land in the image
   (time constant ``smooth_tau_s``), snapping instead when they jump by more than ``snap_px``
   (a moved or rolled camera re-locks at once rather than sliding);
5. recovers the 3D pose (rack → camera ``rvec``/``tvec``) with IPPE PnP from those smoothed
   corners and the camera matrix (calibrated ``K`` or one approximated from ``approx_hfov_deg``);
6. holds the last good solution through a marker occlusion, with ``age_frames`` counting
   frames since the last fresh solve. The hold lasts ``hold_frames`` after the last frame on
   which it was *confirmed*: a fresh solve, or any still-visible marker agreeing with it. A
   visible marker that disagrees (the camera moved) drops the hold immediately.

``H_img_to_rack`` maps raw (distorted) image pixels to the rack plane; lens distortion is only
used for PnP. With a typical webcam the plane error this leaves is covered by the reprojection
check.

Preview the board live (for H-PROPS): ``python -m actus.spatial.rack_frame --source 0``
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from actus.core.contracts import FramePacket, PoseObs, RackFrame

DEFAULT_RACK_CONFIG = "configs/rack.yaml"

# MediaPipe Pose landmark indices
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24


# --------------------------------------------------------------------------------------------
# Geometry config
# --------------------------------------------------------------------------------------------


def rotate_camera_matrix(
    K: np.ndarray, dist: np.ndarray, rotation: int, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Intrinsics for an image rotated CCW by ``rotation`` degrees (as ``cv2.rotate`` does).

    ``width``/``height`` are the *un-rotated* image size. Radial distortion is unchanged; the
    tangential terms rotate with the image.
    """
    K = np.asarray(K, dtype=np.float64).copy()
    dist = np.asarray(dist, dtype=np.float64).ravel().copy()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rotation %= 360
    if rotation == 0:
        return K, dist
    if rotation == 90:  # (x, y) -> (y, W-1-x)
        K = np.array([[fy, 0, cy], [0, fx, width - 1 - cx], [0, 0, 1]], dtype=np.float64)
    elif rotation == 180:  # (x, y) -> (W-1-x, H-1-y)
        K = np.array([[fx, 0, width - 1 - cx], [0, fy, height - 1 - cy], [0, 0, 1]])
    elif rotation == 270:  # (x, y) -> (H-1-y, x)
        K = np.array([[fy, 0, height - 1 - cy], [0, fx, cx], [0, 0, 1]], dtype=np.float64)
    else:
        raise ValueError(f"rotation must be 0/90/180/270, got {rotation}")
    if len(dist) >= 4:
        p1, p2 = dist[2], dist[3]
        dist[2], dist[3] = {90: (-p2, p1), 180: (-p1, -p2), 270: (p2, -p1)}[rotation]
    return K, dist


@dataclass(frozen=True)
class CameraSpec:
    """Intrinsics. Uncalibrated: square pixels, centred principal point, focal length from
    ``approx_hfov_deg`` across the long side of ``image_size``."""

    calibrated: bool = False
    approx_hfov_deg: float = 70.0
    image_size: tuple[int, int] = (1280, 720)  # (width, height) the K below belongs to
    K: np.ndarray | None = None
    dist: np.ndarray | None = None

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any] | None) -> CameraSpec:
        m = dict(m or {})
        K = m.get("K")
        dist = m.get("dist")
        return cls(
            calibrated=bool(m.get("calibrated", False)) and K is not None,
            approx_hfov_deg=float(m.get("approx_hfov_deg", 70.0)),
            image_size=tuple(int(v) for v in m.get("image_size", (1280, 720))),  # type: ignore[arg-type]
            K=None if K is None else np.asarray(K, dtype=np.float64).reshape(3, 3),
            dist=None if dist is None else np.asarray(dist, dtype=np.float64).ravel(),
        )

    def matrix(self, width: int, height: int, rotation: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """(K, dist) for an image of ``width``×``height`` that was rotated CCW by ``rotation``
        after capture (``FramePacket.rotation_applied``)."""
        base_w, base_h = (height, width) if rotation % 180 else (width, height)
        if self.calibrated and self.K is not None:
            cw, ch = self.image_size
            K = self.K.copy()
            K[0, :] *= base_w / cw
            K[1, :] *= base_h / ch
            dist = self.dist if self.dist is not None else np.zeros(5)
            return rotate_camera_matrix(K, dist, rotation, base_w, base_h)
        f = (max(base_w, base_h) / 2.0) / math.tan(math.radians(self.approx_hfov_deg) / 2.0)
        K = np.array([[f, 0, (width - 1) / 2.0], [0, f, (height - 1) / 2.0], [0, 0, 1]])
        return K, np.zeros(5)


@dataclass(frozen=True)
class RackGeometry:
    """The printed board: marker centres on the rack plane and their size."""

    markers: Mapping[int, tuple[float, float]]
    marker_size: float
    dictionary: str = "DICT_4X4_50"
    board_size: tuple[float, float] = (0.90, 0.80)  # (width, height), metres
    outer_box_footprint: tuple[tuple[float, float], ...] | None = None
    camera: CameraSpec = field(default_factory=CameraSpec)
    measured: bool = False

    def __post_init__(self) -> None:
        if len(self.markers) < 1:
            raise ValueError("rack geometry needs at least one marker")
        if self.marker_size <= 0:
            raise ValueError("marker_size must be > 0")

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> RackGeometry:
        board = m.get("board", {}) or {}
        aruco = m.get("aruco", {}) or {}
        fp = m.get("outer_box_footprint")
        return cls(
            markers={int(k): (float(v[0]), float(v[1])) for k, v in m["markers"].items()},
            marker_size=float(aruco.get("marker_size", 0.06)),
            dictionary=str(aruco.get("dictionary", "DICT_4X4_50")),
            board_size=(float(board.get("width", 0.9)), float(board.get("height", 0.8))),
            outer_box_footprint=None if fp is None else tuple(tuple(map(float, p)) for p in fp),  # type: ignore[arg-type]
            camera=CameraSpec.from_mapping(m.get("camera")),
            measured=bool(m.get("measured", False)),
        )

    @classmethod
    def load(cls, path: str | Path = DEFAULT_RACK_CONFIG) -> RackGeometry:
        from actus.core.config import load_yaml

        return cls.from_mapping(load_yaml(path))

    def marker_corners(self, marker_id: int) -> np.ndarray:
        """(4, 3) rack coordinates of a marker's corners in OpenCV order (TL, TR, BR, BL),
        the marker printed upright on the board."""
        cx, cy = self.markers[marker_id]
        h = self.marker_size / 2.0
        return np.array(
            [[cx - h, cy - h, 0], [cx + h, cy - h, 0], [cx + h, cy + h, 0], [cx - h, cy + h, 0]],
            dtype=np.float64,
        )

    @property
    def board_corners(self) -> np.ndarray:
        w, h = self.board_size
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)


# --------------------------------------------------------------------------------------------
# Homography helpers
# --------------------------------------------------------------------------------------------


def apply_homography(H: np.ndarray, points: Iterable[Iterable[float]] | np.ndarray) -> np.ndarray:
    """Map (N, 2) points through a 3×3 homography. Points at infinity become NaN."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    ph = np.hstack([p, np.ones((len(p), 1))]) @ np.asarray(H, dtype=np.float64).T
    with np.errstate(divide="ignore", invalid="ignore"):
        out = ph[:, :2] / ph[:, 2:3]
    out[~np.isfinite(out).all(axis=1)] = np.nan
    return out


def img_to_rack(rack: RackFrame | None, points_px) -> np.ndarray | None:
    """Image pixels → rack-plane metres (N, 2); None without a valid rack frame."""
    if rack is None or not rack.valid or rack.H_img_to_rack is None:
        return None
    return apply_homography(rack.H_img_to_rack, points_px)


def rack_to_img(rack: RackFrame | None, points_rack) -> np.ndarray | None:
    """Rack-plane metres → image pixels (N, 2); None without a valid rack frame."""
    if rack is None or not rack.valid or rack.H_img_to_rack is None:
        return None
    return apply_homography(np.linalg.inv(rack.H_img_to_rack), points_rack)


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


# --------------------------------------------------------------------------------------------
# Estimator
# --------------------------------------------------------------------------------------------


@dataclass
class _Solution:
    G: np.ndarray  # rack plane → image (3×3)
    ref_img: np.ndarray  # (4, 2) smoothed image positions of the board corners
    rvec: np.ndarray | None
    tvec: np.ndarray | None
    t: float


class RackFrameEstimator:
    """Stateful ArUco → ``RackFrame`` solver. One instance per camera; feed frames in order."""

    def __init__(
        self,
        geometry: RackGeometry | None = None,
        *,
        min_markers: int = 3,
        max_reproj_err_px: float = 4.0,
        hold_frames: int = 45,
        smooth_tau_s: float = 0.08,
        snap_px: float = 12.0,
    ) -> None:
        self.geometry = geometry if geometry is not None else RackGeometry.load()
        if min_markers < 1:
            raise ValueError("min_markers must be >= 1")
        self.min_markers = int(min_markers)
        self.max_reproj_err_px = float(max_reproj_err_px)
        self.hold_frames = int(hold_frames)
        self.smooth_tau_s = float(smooth_tau_s)
        self.snap_px = float(snap_px)
        self._detector = None
        self._sol: _Solution | None = None
        self._last_fresh_idx: int | None = None
        self._last_confirmed_idx: int | None = None
        self.last_info: dict[str, Any] = {}

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> RackFrameEstimator:
        """Build from the ``rack`` section of ``configs/default.yaml``."""
        from actus.core.config import load_yaml

        g = RackGeometry.from_mapping(load_yaml(cfg.get("config", DEFAULT_RACK_CONFIG)))
        return cls(
            g,
            min_markers=int(cfg.get("min_markers", 3)),
            max_reproj_err_px=float(cfg.get("max_reproj_err_px", 4.0)),
            hold_frames=int(cfg.get("hold_frames", 45)),
            smooth_tau_s=float(cfg.get("smooth_tau_s", 0.08)),
            snap_px=float(cfg.get("snap_px", 12.0)),
        )

    def reset(self) -> None:
        self._sol = None
        self._last_fresh_idx = self._last_confirmed_idx = None

    # -- detection ------------------------------------------------------------------------------

    def _aruco(self):
        if self._detector is None:
            import cv2

            d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.geometry.dictionary))
            params = cv2.aruco.DetectorParameters()
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self._detector = cv2.aruco.ArucoDetector(d, params)
        return self._detector

    def detect(self, image: np.ndarray) -> tuple[list[int], list[np.ndarray]]:
        """Known marker ids and their (4, 2) image corners. A duplicated id keeps its largest
        instance."""
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners, ids, _ = self._aruco().detectMarkers(gray)
        best: dict[int, np.ndarray] = {}
        if ids is not None:
            for c, i in zip(corners, ids.ravel(), strict=True):
                mid = int(i)
                if mid not in self.geometry.markers:
                    continue
                c = np.asarray(c, dtype=np.float64).reshape(4, 2)
                if mid not in best or cv2.contourArea(c.astype(np.float32)) > cv2.contourArea(
                    best[mid].astype(np.float32)
                ):
                    best[mid] = c
        ids_out = sorted(best)
        return ids_out, [best[i] for i in ids_out]

    def update(self, frame: FramePacket) -> RackFrame:
        """Detect markers in ``frame`` and solve."""
        ids, corners = self.detect(frame.image)
        h, w = frame.image.shape[:2]
        return self.solve(
            ids,
            corners,
            image_size=(w, h),
            frame_idx=frame.frame_idx,
            t=frame.t_capture,
            rotation=frame.rotation_applied,
        )

    # -- solve ----------------------------------------------------------------------------------

    def solve(
        self,
        ids: Sequence[int],
        corners: Sequence[np.ndarray],
        *,
        image_size: tuple[int, int],
        frame_idx: int,
        t: float,
        rotation: int = 0,
    ) -> RackFrame:
        """Solve from already-detected markers (``corners[i]`` is (4, 2) pixels for ``ids[i]``,
        OpenCV corner order). Unknown ids are ignored."""
        pairs = [
            (int(i), np.asarray(c, dtype=np.float64).reshape(4, 2))
            for i, c in zip(ids, corners, strict=True)
            if int(i) in self.geometry.markers
        ]
        seen = tuple(sorted({i for i, _ in pairs}))
        info: dict[str, Any] = {"markers_detected": len(seen), "status": "invalid"}
        fit = self._fit(pairs) if len(seen) >= self.min_markers else None

        if fit is not None:
            G, err, inlier_ids = fit
            info.update(reproj_err_px=err, inlier_markers=len(inlier_ids))
            if len(inlier_ids) >= self.min_markers and err <= self.max_reproj_err_px:
                sol = self._smooth(G, t)
                K, dist = self.geometry.camera.matrix(*image_size, rotation)
                sol.rvec, sol.tvec = self._pose(sol.ref_img, K, dist)
                if sol.rvec is not None:
                    info["pnp_err_px"] = self._pnp_err(pairs, sol.rvec, sol.tvec, K, dist)
                self._sol = sol
                self._last_fresh_idx = self._last_confirmed_idx = frame_idx
                info["status"] = "fresh"
                self.last_info = info
                return self._frame(sol, seen, err, frame_idx, t, age=0)

        # No fresh solve: hold the last good one if it is recent and not contradicted.
        sol = self._sol
        if sol is not None and self._last_confirmed_idx is not None:
            err_seen = None
            if pairs:
                img = np.vstack([c for _, c in pairs])
                rk = np.vstack([self.geometry.marker_corners(i)[:, :2] for i, _ in pairs])
                err_seen = _rms(apply_homography(sol.G, rk), img)
                if err_seen <= self.max_reproj_err_px * 2:
                    self._last_confirmed_idx = frame_idx
            contradicted = err_seen is not None and err_seen > self.max_reproj_err_px * 3
            fresh_idx = self._last_fresh_idx if self._last_fresh_idx is not None else frame_idx
            age = max(0, frame_idx - fresh_idx)
            if not contradicted and frame_idx - self._last_confirmed_idx <= self.hold_frames:
                info.update(status="held", age_frames=age, held_err_px=err_seen)
                self.last_info = info
                return self._frame(sol, seen, err_seen, frame_idx, t, age=age)
            info["status"] = "contradicted" if contradicted else "expired"
            self.reset()
        self.last_info = info
        return RackFrame(False, None, None, None, seen, None, frame_idx=frame_idx, t=t)

    def _fit(self, pairs: list[tuple[int, np.ndarray]]):
        """Rack plane → image homography over all corners; returns (G, rms_px, inlier_ids)."""
        import cv2

        img = np.vstack([c for _, c in pairs])
        rk = np.vstack([self.geometry.marker_corners(i)[:, :2] for i, _ in pairs])
        if len(pairs) >= 3:
            G, mask = cv2.findHomography(rk, img, cv2.RANSAC, self.max_reproj_err_px)
        else:
            G, mask = cv2.findHomography(rk, img, 0)
        if G is None or not np.isfinite(G).all() or abs(np.linalg.det(G)) < 1e-12:
            return None
        mask = np.ones(len(img), bool) if mask is None else mask.ravel().astype(bool)
        # a marker is an inlier only if all four of its corners are
        per_marker = mask.reshape(-1, 4).all(axis=1)
        inlier_ids = [i for (i, _), ok in zip(pairs, per_marker, strict=True) if ok]
        if not inlier_ids:
            return None
        keep = np.repeat(per_marker, 4)
        if keep.sum() >= 4 and not keep.all():
            G2, _ = cv2.findHomography(rk[keep], img[keep], 0)
            if G2 is not None and np.isfinite(G2).all():
                G = G2
        G = G / G[2, 2]
        return G, _rms(apply_homography(G, rk[keep]), img[keep]), inlier_ids

    def _smooth(self, G_raw: np.ndarray, t: float) -> _Solution:
        import cv2

        board = self.geometry.board_corners
        ref_raw = apply_homography(G_raw, board)
        prev = self._sol
        if prev is None or not np.isfinite(ref_raw).all():
            ref = ref_raw
        else:
            jump = float(np.max(np.linalg.norm(ref_raw - prev.ref_img, axis=1)))
            dt = max(0.0, t - prev.t)
            alpha = 1.0 if self.smooth_tau_s <= 0 else 1.0 - math.exp(-dt / self.smooth_tau_s)
            # move faster the larger the jump; snap beyond snap_px (camera moved or rolled)
            alpha = 1.0 if jump >= self.snap_px else alpha + (1 - alpha) * jump / self.snap_px
            ref = prev.ref_img + alpha * (ref_raw - prev.ref_img)
        G = cv2.getPerspectiveTransform(board.astype(np.float32), ref.astype(np.float32))
        G = np.asarray(G, dtype=np.float64)
        return _Solution(G / G[2, 2], ref, None, None, t)

    def _pose(self, ref_img: np.ndarray, K: np.ndarray, dist: np.ndarray):
        import cv2

        obj = np.hstack([self.geometry.board_corners, np.zeros((4, 1))])
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj, ref_img.reshape(-1, 1, 2), K, dist, flags=cv2.SOLVEPNP_IPPE
            )
        except cv2.error:
            return None, None
        if not ok or float(tvec.ravel()[2]) <= 0:  # the board must be in front of the camera
            return None, None
        return rvec.ravel().astype(np.float64), tvec.ravel().astype(np.float64)

    def _pnp_err(self, pairs, rvec, tvec, K, dist) -> float:
        import cv2

        img = np.vstack([c for _, c in pairs])
        obj = np.vstack([self.geometry.marker_corners(i) for i, _ in pairs])
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        return _rms(proj.reshape(-1, 2), img)

    def _frame(self, sol: _Solution, seen, err, frame_idx: int, t: float, age: int) -> RackFrame:
        return RackFrame(
            True,
            np.linalg.inv(sol.G),
            sol.rvec,
            sol.tvec,
            seen,
            err,
            frame_idx=frame_idx,
            t=t,
            age_frames=age,
        )


# --------------------------------------------------------------------------------------------
# Body relative to the rack
# --------------------------------------------------------------------------------------------


def unroll_image_vector(v: np.ndarray, rotation: int) -> np.ndarray:
    """Undo a CCW image rotation on a camera-axis vector (x right, y down, z forward)."""
    th = math.radians(rotation % 360)
    c, s = math.cos(th), math.sin(th)
    x, y, z = (float(a) for a in v)
    return np.array([x * c - y * s, x * s + y * c, z])


def torso_vec_rack(
    pose: PoseObs,
    rack: RackFrame | None,
    *,
    world_rotated_back: bool = False,
    min_visibility: float = 0.3,
) -> np.ndarray | None:
    """Unit hips→shoulders vector in rack coordinates, or None.

    MediaPipe world landmarks are hip-centred with axes that follow the camera image (x right,
    y down, z away from the camera). Unless ``world_rotated_back`` says T-PERC already undid the
    rotation-TTA roll, ``pose.rotation_used`` is undone here first. The camera → rack rotation
    comes from the rack frame's PnP pose.
    """
    if rack is None or not rack.valid or rack.rvec is None:
        return None
    if pose.visibility is not None:
        vis = pose.visibility[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]]
        if float(vis.min()) < min_visibility:
            return None
    wl = np.asarray(pose.world_landmarks, dtype=np.float64)
    v = wl[[L_SHOULDER, R_SHOULDER]].mean(axis=0) - wl[[L_HIP, R_HIP]].mean(axis=0)
    if not world_rotated_back and pose.rotation_used:
        v = unroll_image_vector(v, pose.rotation_used)
    import cv2

    R, _ = cv2.Rodrigues(np.asarray(rack.rvec, dtype=np.float64))
    v_rack = R.T @ v
    n = float(np.linalg.norm(v_rack))
    if not math.isfinite(n) or n < 1e-6:
        return None
    return v_rack / n


# --------------------------------------------------------------------------------------------
# Overlay
# --------------------------------------------------------------------------------------------


def draw_overlay(
    image: np.ndarray,
    rack: RackFrame | None,
    *,
    zones: Mapping[str, np.ndarray] | None = None,
    footprints: Mapping[str, np.ndarray] | None = None,
    geometry: RackGeometry | None = None,
    points: Mapping[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    """Draw the rack board outline, zones, footprints and labelled rack points onto a copy of
    ``image``. ``zones``/``footprints`` are (N, 2) rack-metre polygons."""
    import cv2

    out = image.copy()
    if rack is None or not rack.valid:
        cv2.putText(out, "RACK: no frame", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return out

    def poly(pts, color, label=None, thick=2):
        px = rack_to_img(rack, pts)
        if px is None or not np.isfinite(px).all():
            return
        cv2.polylines(out, [px.round().astype(np.int32)], True, color, thick, cv2.LINE_AA)
        if label:
            x, y = px.min(axis=0)
            cv2.putText(
                out, label, (int(x) + 4, int(y) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
            )

    if geometry is not None:
        poly(geometry.board_corners, (200, 200, 200), None, 1)
        # rack axes from the origin: +x red, +y green
        o = rack_to_img(rack, [[0, 0], [0.15, 0], [0, 0.15]])
        if o is not None and np.isfinite(o).all():
            o = o.round().astype(int)
            cv2.arrowedLine(out, tuple(o[0]), tuple(o[1]), (0, 0, 255), 2, tipLength=0.2)
            cv2.arrowedLine(out, tuple(o[0]), tuple(o[2]), (0, 200, 0), 2, tipLength=0.2)
    for name, pts in (zones or {}).items():
        poly(pts, (60, 200, 60), f"Zone {name}")
    for name, pts in (footprints or {}).items():
        poly(pts, (0, 165, 255), name)
    for name, xy in (points or {}).items():
        px = rack_to_img(rack, [xy])
        if px is not None and np.isfinite(px).all():
            x, y = (int(v) for v in px[0].round())
            cv2.circle(out, (x, y), 5, (255, 0, 255), -1)
            cv2.putText(out, name, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    state = "held" if rack.age_frames else "fresh"
    err = "-" if rack.reproj_err is None else f"{rack.reproj_err:.2f}px"
    cv2.putText(
        out,
        f"RACK {state} markers={list(rack.markers_seen)} err={err} age={rack.age_frames}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0) if not rack.age_frames else (0, 200, 255),
        2,
    )
    return out


# --------------------------------------------------------------------------------------------
# CLI: live board check
# --------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Solve the rack frame from a camera or an image.")
    ap.add_argument("--source", default="0", help="webcam index, video file or rtsp:// URL")
    ap.add_argument("--image", type=Path, help="solve a single image and print the result")
    ap.add_argument("--rack", default=DEFAULT_RACK_CONFIG)
    ap.add_argument("--protocol", default="protocols/bas_boxes.yaml", help="zones to overlay")
    ap.add_argument("--min-markers", type=int, default=3)
    args = ap.parse_args(argv)

    import time

    import cv2

    from actus.protocol.schema import load_protocol
    from actus.spatial.zones import as_polygon

    geo = RackGeometry.load(args.rack)
    est = RackFrameEstimator(geo, min_markers=args.min_markers)
    zones = {k: as_polygon(v) for k, v in load_protocol(args.protocol).zones.items()}
    fps = {}
    if geo.outer_box_footprint:
        fps["outer box"] = as_polygon(geo.outer_box_footprint)
    if not geo.measured:
        print("WARNING: configs/rack.yaml has measured: false (placeholder geometry)")

    if args.image:
        img = cv2.imread(str(args.image))
        if img is None:
            print(f"cannot read {args.image}", file=sys.stderr)
            return 1
        rf = est.update(FramePacket(0, 0.0, img, "image"))
        print(f"valid={rf.valid} markers={list(rf.markers_seen)} reproj_err={rf.reproj_err}")
        print(f"info={est.last_info}")
        if rf.valid:
            print(f"tvec (m, rack origin in camera)={np.round(rf.tvec, 3).tolist()}")
            out = args.image.with_name(args.image.stem + "_rack.png")
            cv2.imwrite(str(out), draw_overlay(img, rf, zones=zones, footprints=fps, geometry=geo))
            print(f"overlay written to {out}")
        return 0 if rf.valid else 2

    src: Any = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"cannot open source {args.source}", file=sys.stderr)
        return 1
    idx, t0 = 0, time.monotonic()
    print("q / Esc to quit")
    while True:
        ok, img = cap.read()
        if not ok:
            break
        rf = est.update(FramePacket(idx, time.monotonic() - t0, img, "preview"))
        cv2.imshow("rack frame", draw_overlay(img, rf, zones=zones, footprints=fps, geometry=geo))
        idx += 1
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
