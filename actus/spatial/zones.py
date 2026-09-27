"""Zones on the rack plane: hard and soft point-in-polygon tests, in rack metres.

A protocol gives a zone as 2 points (the rectangle ``[[xmin, ymin], [xmax, ymax]]``) or as 3+
polygon vertices. ``Zone.membership(p, sigma)`` is the probability that a point measured with
isotropic Gaussian position noise ``sigma`` lies inside, approximated as ``Φ(d / sigma)`` with
``d`` the signed distance to the boundary (positive inside). A box resting on a zone edge
therefore reads about 0.5, which lands in the engine's ask band instead of flickering 0/1.

    zones = zones_from_protocol(proto)
    zones["A"].contains((0.6, 0.2))            # True
    zones["A"].membership((0.55, 0.2), 0.02)   # 0.5 (on the edge)
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np


def as_polygon(points: Iterable[Iterable[float]]) -> np.ndarray:
    """(N, 2) float64 polygon. Two points are read as a rectangle's opposite corners."""
    pts = np.asarray([tuple(map(float, p)) for p in points], dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"zone points must be (x, y) pairs, got shape {pts.shape}")
    if len(pts) == 2:
        (x1, y1), (x2, y2) = pts.min(axis=0), pts.max(axis=0)
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)
    if len(pts) < 3:
        raise ValueError("a zone needs 2 points (rectangle) or at least 3 (polygon)")
    return pts


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass(frozen=True, eq=False)
class Zone:
    name: str
    polygon: np.ndarray  # (N, 2) rack metres

    def __post_init__(self) -> None:
        poly = as_polygon(self.polygon)
        poly.setflags(write=False)
        object.__setattr__(self, "polygon", poly)

    # -- geometry -------------------------------------------------------------------------------

    def signed_distance(self, points: Iterable[float] | np.ndarray) -> np.ndarray | float:
        """Distance to the boundary, positive inside. Accepts one point or an (M, 2) array."""
        p = np.asarray(points, dtype=np.float64)
        single = p.ndim == 1
        p = p.reshape(-1, 2)
        a = self.polygon
        b = np.roll(a, -1, axis=0)
        ab = b - a  # (N, 2)
        ap = p[:, None, :] - a[None, :, :]  # (M, N, 2)
        denom = np.maximum((ab * ab).sum(axis=1), 1e-18)
        s = np.clip((ap * ab[None]).sum(axis=2) / denom, 0.0, 1.0)
        closest = a[None] + s[..., None] * ab[None]
        dist = np.linalg.norm(p[:, None, :] - closest, axis=2).min(axis=1)
        # even-odd ray casting along +x
        ay, by = a[:, 1][None], b[:, 1][None]
        py, px = p[:, 1:2], p[:, 0:1]
        crosses = (ay > py) != (by > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_int = a[:, 0][None] + (py - ay) * (b[:, 0] - a[:, 0])[None] / (by - ay)
        inside = (crosses & (px < x_int)).sum(axis=1) % 2 == 1
        out = np.where(inside, dist, -dist)
        return float(out[0]) if single else out

    def contains(self, point: Iterable[float]) -> bool:
        return bool(self.signed_distance(np.asarray(point, dtype=np.float64)) >= 0.0)

    def membership(self, point: Iterable[float] | None, sigma: float) -> float:
        """P(inside) for a point with Gaussian noise ``sigma`` (metres); 0 for ``None``."""
        if point is None:
            return 0.0
        d = float(self.signed_distance(np.asarray(point, dtype=np.float64)))
        if sigma <= 0:
            return 1.0 if d >= 0 else 0.0
        return _normal_cdf(d / sigma)

    def distance_outside(self, point: Iterable[float]) -> float:
        """0 inside, else the distance to the zone (metres)."""
        return max(0.0, -float(self.signed_distance(np.asarray(point, dtype=np.float64))))

    @property
    def centroid(self) -> tuple[float, float]:
        """Area centroid (vertex mean for a degenerate polygon)."""
        x, y = self.polygon[:, 0], self.polygon[:, 1]
        xs, ys = np.roll(x, -1), np.roll(y, -1)
        cross = x * ys - xs * y
        area = cross.sum() / 2.0
        if abs(area) < 1e-12:
            cx, cy = self.polygon.mean(axis=0)
            return float(cx), float(cy)
        return (
            float(((x + xs) * cross).sum() / (6 * area)),
            float(((y + ys) * cross).sum() / (6 * area)),
        )

    def translated(self, dx: float, dy: float) -> Zone:
        return Zone(self.name, self.polygon + np.array([dx, dy]))

    def to_list(self) -> list[list[float]]:
        return [[float(x), float(y)] for x, y in self.polygon]


def zones_from_mapping(zones: Mapping[str, Iterable[Iterable[float]]]) -> dict[str, Zone]:
    return {str(name): Zone(str(name), as_polygon(pts)) for name, pts in zones.items()}


def zones_from_protocol(protocol) -> dict[str, Zone]:
    """The protocol's named zones (``protocol.zones``) as ``Zone`` objects."""
    return zones_from_mapping(protocol.zones)
