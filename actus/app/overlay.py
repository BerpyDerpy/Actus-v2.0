"""Server-side overlays for the ``annotated`` stream (recorder, RTSP push, GUI MJPEG).

Everything the operator and a reviewer need is burned into the frame, so the annotated video is
self-explanatory without the dashboard: the rack board, zones and container footprints (via
``actus.spatial.rack_frame.draw_overlay``), detection boxes with track ids, hand and upper-body
skeletons, a step checklist, the next-step prompt, the open question, the latest alert, the
pipeline rate, and a SIMULATED tag whenever the input is synthetic.

OpenCV's Hershey fonts are ASCII only, so every string is reduced to ASCII before drawing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from actus.core.contracts import DetectionSet, HandSet, PoseSet, RackFrame, StepStatus

# MediaPipe hand topology (21 landmarks) and the upper-body subset of the 33-point pose.
HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)  # fmt: skip
POSE_EDGES = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24))

# BGR
STATUS_COLOURS: dict[str, tuple[int, int, int]] = {
    StepStatus.PENDING.value: (90, 90, 90),
    StepStatus.VERIFIED.value: (60, 170, 60),
    StepStatus.OPERATOR_CONFIRMED.value: (200, 150, 40),
    StepStatus.OUT_OF_ORDER.value: (0, 140, 255),
    StepStatus.SKIPPED.value: (40, 40, 210),
    StepStatus.TIMED_OUT.value: (0, 200, 230),
}
_CLASS_COLOURS = (
    (0, 200, 255),
    (255, 120, 0),
    (0, 0, 230),
    (230, 0, 230),
    (0, 230, 0),
    (255, 255, 0),
)
_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX


def ascii_text(s: str) -> str:
    return s.encode("ascii", "replace").decode("ascii")


def _class_colour(cls: str) -> tuple[int, int, int]:
    return _CLASS_COLOURS[sum(map(ord, cls)) % len(_CLASS_COLOURS)]


def _text(img: np.ndarray, s: str, org: tuple[int, int], scale: float, colour, thick: int = 1):
    import cv2

    s = ascii_text(s)
    cv2.putText(img, s, org, _FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, _FONT, scale, colour, thick, cv2.LINE_AA)


def _fit(s: str, width_px: int, scale: float) -> str:
    """Truncate ``s`` so it fits ``width_px`` at ``scale`` (Hershey simplex ~ 20 px/char at 1.0)."""
    max_chars = max(4, int(width_px / (20 * scale)))
    return s if len(s) <= max_chars else s[: max_chars - 3] + "..."


def _band(img: np.ndarray, y0: int, y1: int, alpha: float = 0.55) -> None:
    y0, y1 = max(0, y0), min(img.shape[0], y1)
    if y1 > y0:
        img[y0:y1] = (img[y0:y1].astype(np.float32) * (1 - alpha)).astype(img.dtype)


def draw_detections(img: np.ndarray, dets: DetectionSet | None) -> None:
    import cv2

    if dets is None:
        return
    for d in dets.detections:
        x1, y1, x2, y2 = (round(v) for v in d.bbox_xyxy)
        c = _class_colour(d.cls)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2, cv2.LINE_AA)
        tid = "" if d.track_id is None else f"#{d.track_id} "
        _text(img, f"{tid}{d.cls} {d.conf:.2f}", (x1, max(12, y1 - 5)), 0.45, c)


def draw_hands(img: np.ndarray, hands: HandSet | None) -> None:
    import cv2

    if hands is None:
        return
    for h in hands.hands:
        pts = np.asarray(h.landmarks_px, dtype=np.float64)
        if not np.isfinite(pts).all():
            continue
        p = pts.round().astype(int)
        c = (255, 200, 0) if h.side == "left" else (0, 220, 255)
        for a, b in HAND_EDGES:
            cv2.line(img, tuple(p[a]), tuple(p[b]), c, 2, cv2.LINE_AA)
        for q in p:
            cv2.circle(img, tuple(q), 2, (255, 255, 255), -1, cv2.LINE_AA)


def draw_pose(img: np.ndarray, pose: PoseSet | None, min_vis: float = 0.3) -> None:
    import cv2

    if pose is None:
        return
    for obs in pose.poses:
        pts = np.asarray(obs.landmarks_px, dtype=np.float64)
        vis = obs.visibility if obs.visibility is not None else np.ones(len(pts))
        for a, b in POSE_EDGES:
            if vis[a] < min_vis or vis[b] < min_vis:
                continue
            if not (np.isfinite(pts[a]).all() and np.isfinite(pts[b]).all()):
                continue
            pa, pb = tuple(pts[a].round().astype(int)), tuple(pts[b].round().astype(int))
            cv2.line(img, pa, pb, (180, 105, 255), 2, cv2.LINE_AA)


def draw_checklist(
    img: np.ndarray, step_ids: Sequence[str], statuses: Mapping[str, str], y: int
) -> None:
    import cv2

    x = 8
    for sid in step_ids:
        status = str(statuses.get(sid, StepStatus.PENDING.value))
        label = ascii_text(sid)
        (w, h), _ = cv2.getTextSize(label, _FONT, 0.5, 1)
        cv2.rectangle(img, (x, y - h - 6), (x + w + 10, y + 4), STATUS_COLOURS.get(status), -1)
        cv2.putText(img, label, (x + 5, y), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        x += w + 16


def render(
    image: np.ndarray,
    *,
    rack: RackFrame | None = None,
    geometry=None,
    zones: Mapping[str, np.ndarray] | None = None,
    footprints: Mapping[str, np.ndarray] | None = None,
    points: Mapping[str, tuple[float, float]] | None = None,
    detections: DetectionSet | None = None,
    hands: HandSet | None = None,
    pose: PoseSet | None = None,
    step_ids: Sequence[str] = (),
    statuses: Mapping[str, str] | None = None,
    prompt: str | None = None,
    ask: str | None = None,
    alert: str | None = None,
    fps: float | None = None,
    simulated: bool = False,
    label: str | None = None,
    tags: Iterable[str] = (),
) -> np.ndarray:
    """Return an annotated copy of ``image`` (BGR; grey input is converted)."""
    import cv2

    from actus.spatial.rack_frame import draw_overlay

    img = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    out = draw_overlay(
        img, rack, zones=zones, footprints=footprints, geometry=geometry, points=points
    )
    draw_pose(out, pose)
    draw_detections(out, detections)
    draw_hands(out, hands)

    h, w = out.shape[:2]
    right = []
    if fps is not None:
        right.append(f"{fps:4.1f} fps")
    right += [str(t) for t in tags]
    if simulated:
        right.append("SIMULATED")
    for i, s in enumerate(right):
        colour = (0, 0, 255) if s in ("SIMULATED",) or s.startswith("FAULT") else (255, 255, 255)
        (tw, _), _ = cv2.getTextSize(ascii_text(s), _FONT, 0.55, 1)
        _text(out, s, (w - tw - 10, 25 + 22 * i), 0.55, colour)

    if alert:
        _band(out, 34, 64, alpha=0.3)
        out[34:64, :, 2] = np.maximum(out[34:64, :, 2], 140)
        _text(out, _fit("ALERT: " + alert, w - 16, 0.55), (8, 56), 0.55, (255, 255, 255))

    band_h = 58 if (prompt or ask) else 30
    _band(out, h - band_h, h)
    if step_ids:
        draw_checklist(out, step_ids, statuses or {}, h - band_h + 20)
    if ask:
        _text(out, _fit("ASK: " + ask, w - 16, 0.55), (8, h - 10), 0.55, (0, 220, 255))
    elif prompt:
        _text(out, _fit("NEXT: " + prompt, w - 16, 0.55), (8, h - 10), 0.55, (255, 255, 255))
    if label:
        _text(out, f"truth: {label}", (w - 170, h - band_h + 20), 0.45, (200, 200, 200))
    return out
