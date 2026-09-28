"""A self-calibrating rack frame, for footage whose rig does not match ``configs/rack.yaml``.

Why this exists
---------------
``actus.spatial.rack_frame`` is the real rack frame and stays the one the live pipeline uses. It
solves against the *measured* marker positions in ``configs/rack.yaml``, with each marker assumed
printed upright on a common board.

The H-FILM takes do not satisfy that. The markers are printed on loose sheets laid on the floor
at arbitrary individual rotations, and the rig was re-laid between operators at different sizes
(roughly 0.50 x 0.50 m for op02, 0.58 x 0.35 m for op03, 0.60 x 0.31 m for op04, 0.95 x 0.65 m
for op05). Fitting the config's model to those corners gives tens of pixels of reprojection
error against a 4 px gate, so ``RackFrame.valid`` is False on every recorded frame and every
rack-relative quantity is unavailable.

What this does instead
----------------------
Each marker is a square of known physical size (``aruco.marker_size``, the one number in
``rack.yaml`` that is right by construction -- it was printed that way). So *one* visible marker
is enough to define the floor plane in metres: the homography from its four corners to a
centred square of that size maps the whole plane.

``MarkerFrameFitter`` picks a reference marker, builds that homography, and simultaneously
learns where the other markers sit in the reference's frame. Once a marker has been seen
alongside the reference it can act as a stand-in, so the frame survives the reference being
occluded -- which is what happens constantly when someone reaches over the rig.

Properties, and the ones it does not have:

* **Metric.** Units are metres, from the printed marker size.
* **Rotation invariant.** Rolling the camera rotates the image, not the frame. This is the
  property the orientation-agnostic claim rests on, and it holds here.
* **Consistent within a session**, because the sheets don't move. Not consistent *across*
  operators: the origin is wherever the reference marker was laid that day, so absolute
  positions are not comparable between operators. Relative features (hand-to-object distance,
  co-motion, velocities) are unaffected, which is why the models lean on those.
* **Not a substitute for measuring the rig.** It recovers the frame, not the protocol's zones:
  zone polygons in the protocol YAML are in a measured frame this one does not reproduce.

``calibration()`` returns what it learned, which is exactly the ``markers:`` block a measured
``configs/rack.yaml`` would carry for that session.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from actus.core.contracts import FramePacket, RackFrame

DEFAULT_MARKER_SIZE = 0.060


def _unit_square(size: float) -> np.ndarray:
    """A marker's own corners in its local frame, OpenCV order (TL, TR, BR, BL), centred."""
    h = size / 2.0
    return np.array([[-h, -h], [h, -h], [h, h], [-h, h]], dtype=np.float64)


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    import cv2

    if len(src) < 4:
        return None
    H, _ = cv2.findHomography(np.asarray(src, np.float64), np.asarray(dst, np.float64), 0)
    if H is None or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        return None
    return H / H[2, 2]


def _apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ph = np.hstack([p, np.ones((len(p), 1))]) @ np.asarray(H, dtype=np.float64).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return ph[:, :2] / ph[:, 2:3]


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


@dataclass
class _Known:
    """A marker's corners in the reference marker's plane frame, averaged over sightings."""

    corners: np.ndarray  # (4, 2) metres
    n: int = 1

    def blend(self, obs: np.ndarray, max_n: int = 60) -> None:
        w = 1.0 / min(self.n + 1, max_n)
        self.corners = (1 - w) * self.corners + w * np.asarray(obs, np.float64)
        self.n += 1


@dataclass
class MarkerFrameFitter:
    """Builds a metric plane frame from whichever markers are visible.

    Feed frames in order. ``update`` returns the frozen ``RackFrame`` contract, so every
    consumer (``FrameFeaturizer`` included) treats it exactly like the real rack frame.
    """

    marker_size: float = DEFAULT_MARKER_SIZE
    hold_frames: int = 45  # keep the last good frame this long through an occlusion
    max_reproj_err_px: float = 6.0
    min_sightings: int = 3  # a learned marker must recur before it may drive the solve
    frozen: bool = False  # stop learning (set after a calibration pass)
    known: dict[int, _Known] = field(default_factory=dict)
    reference: int | None = None
    _detector: object | None = field(default=None, repr=False)
    _H: np.ndarray | None = field(default=None, repr=False)
    _last_fresh: int | None = field(default=None, repr=False)

    def reset(self) -> None:
        self.known.clear()
        self.reference = None
        self._H = None
        self._last_fresh = None

    # -- calibration ---------------------------------------------------------------------------

    def calibration(self) -> dict[int, tuple[float, float]]:
        """Marker centres in the reference marker's frame: the ``markers:`` block for this rig."""
        return {i: tuple(k.corners.mean(axis=0)) for i, k in sorted(self.known.items())}

    def _seed(self, ids: Sequence[int]) -> None:
        """Adopt the lowest visible id as the reference and put it at the origin."""
        ref = min(int(i) for i in ids)
        self.reference = ref
        self.known = {ref: _Known(_unit_square(self.marker_size))}

    def _learn(self, seen: dict[int, np.ndarray], H_img_to_rack: np.ndarray) -> None:
        for i, img_corners in seen.items():
            rack = _apply(H_img_to_rack, img_corners)
            if not np.all(np.isfinite(rack)):
                continue
            if i in self.known:
                self.known[i].blend(rack)
            else:
                self.known[i] = _Known(rack)

    # -- per frame -----------------------------------------------------------------------------

    def update(self, frame: FramePacket) -> RackFrame:
        from actus.spatial.rack_frame import RackFrameEstimator

        est = getattr(self, "_detector", None)
        if est is None:
            est = RackFrameEstimator(_dummy_geometry(self.marker_size))
            self._detector = est
        ids, corners = est.detect(frame.image)
        return self.solve(ids, corners, frame_idx=frame.frame_idx, t=frame.t_capture)

    def solve(
        self,
        ids: Sequence[int],
        corners: Sequence[np.ndarray],
        *,
        frame_idx: int,
        t: float,
    ) -> RackFrame:
        seen = {
            int(i): np.asarray(c, dtype=np.float64).reshape(4, 2)
            for i, c in zip(ids, corners, strict=True)
        }
        seen_ids = tuple(sorted(seen))
        if seen and self.reference is None:
            self._seed(list(seen))

        src, dst = [], []
        for i, img_corners in seen.items():
            k = self.known.get(i)
            # A spurious ArUco hit (a stray pattern in the room) would otherwise drag the frame,
            # so a learned marker only counts once it has been seen repeatedly. The reference
            # marker defines the frame and is always trusted.
            if k is not None and (i == self.reference or k.n >= self.min_sightings):
                src.append(img_corners)
                dst.append(k.corners)
        H = None
        err: float | None = None
        if src:
            src_a, dst_a = np.vstack(src), np.vstack(dst)
            H = _homography(src_a, dst_a)
            if H is not None:
                # Reprojection error in pixels: map the known rack corners back to the image.
                back = _homography(dst_a, src_a)
                err = None if back is None else _rms(_apply(back, dst_a), src_a)
                if err is not None and err > self.max_reproj_err_px:
                    H = None

        if H is not None:
            if not self.frozen:
                self._learn(seen, H)
            self._H = H
            self._last_fresh = frame_idx
            return RackFrame(
                True, H, None, None, seen_ids, err, frame_idx=frame_idx, t=t, age_frames=0
            )

        if self._H is not None and self._last_fresh is not None:
            age = frame_idx - self._last_fresh
            if age <= self.hold_frames:
                return RackFrame(
                    True,
                    self._H,
                    None,
                    None,
                    seen_ids,
                    None,
                    frame_idx=frame_idx,
                    t=t,
                    age_frames=max(0, age),
                )
        return RackFrame(False, None, None, None, seen_ids, None, frame_idx=frame_idx, t=t)


def _dummy_geometry(marker_size: float, n_ids: int = 50):
    """A geometry object good only for running the ArUco detector.

    ``RackFrameEstimator.detect`` drops ids that are not in ``geometry.markers``, so every id the
    dictionary can produce is declared here. The positions are placeholders and never read: this
    module derives the plane from the markers' own corners, not from a configured layout.
    """
    from actus.spatial.rack_frame import RackGeometry

    return RackGeometry(markers={i: (0.0, 0.0) for i in range(n_ids)}, marker_size=marker_size)


def board_size_from(calibration: dict[int, tuple[float, float]]) -> tuple[float, float]:
    """Extent of the calibrated marker layout, for normalising positions. (1, 1) if unknown."""
    if len(calibration) < 2:
        return (1.0, 1.0)
    pts = np.array(list(calibration.values()), dtype=np.float64)
    span = pts.max(axis=0) - pts.min(axis=0)
    return (float(max(span[0], 1e-3)), float(max(span[1], 1e-3)))
