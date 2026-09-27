"""Live ArUco marker check for the recording rig. Run it before ``record_takes.py``.

    uv run python tools/check_markers.py --source http://192.168.1.23:8080/video   # phone
    uv run python tools/check_markers.py --source 0                                  # webcam
    uv run python tools/check_markers.py --source replay:correct                     # self-test

A small always-on-top window shows the live feed, so you can keep moving the markers or the
camera and see the result at once. Every detected marker is outlined and labelled with its id:
green is a rack marker, red a rack marker too small to rely on, orange a rack marker that isn't
where ``configs/rack.yaml`` puts it, and grey an id that isn't in rack.yaml (ignored). The
banner at the top says:

* ``READY``      every rack marker is visible and at least ``--min-side-px`` pixels on a side;
* ``PARTIAL``    enough markers for the rack frame to solve (``--min-markers``), but a hand
                 covering one more would lose it;
* ``NOT READY``  fewer than that.

Press ``q`` (or close the window) when it reads READY. The exit status is 0 when the last check
was READY and 1 otherwise, so a script can gate on it.

``fit`` is how well the visible markers match the layout in ``configs/rack.yaml`` (RMS pixels of
the board homography). It is not part of READY: a board built before rack.yaml was measured
(``measured: false``) can fit badly until the file is updated.

The recorder runs the same check live with ``--markers`` (``record_takes.py ... --markers``): a
background thread checks the newest frame a few times a second, the preview shows a status
strip (never burned into the saved video), and each take's JSON gets a ``markers`` summary.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from actus.core.contracts import RackFrame  # noqa: E402
from actus.spatial.rack_frame import (  # noqa: E402
    DEFAULT_RACK_CONFIG,
    RackFrameEstimator,
    RackGeometry,
    rack_to_img,
)

WINDOW = "ACTUS marker check"
READY, PARTIAL, NOT_READY = "ready", "partial", "not_ready"
STALE_S = 1.0  # a live status older than this is shown as unknown (the camera stalled)

GREEN, RED, ORANGE, AMBER = (60, 200, 60), (40, 40, 230), (0, 140, 255), (0, 190, 255)
GREY = (160, 160, 160)
STATE_COLOR = {READY: (40, 150, 40), PARTIAL: (0, 150, 220), NOT_READY: (40, 40, 200)}
STATE_TEXT = {READY: "READY", PARTIAL: "PARTIAL", NOT_READY: "NOT READY"}


# --------------------------------------------------------------------------------------------
# Checking one frame
# --------------------------------------------------------------------------------------------


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


@dataclass(frozen=True)
class MarkerStatus:
    """What one check found. ``corners`` / ``unknown`` map an id to its (4, 2) image corners."""

    expected: tuple[int, ...]
    corners: dict[int, np.ndarray]
    unknown: dict[int, np.ndarray]
    side_px: dict[int, float]
    off_layout: tuple[int, ...]
    rack: RackFrame
    rack_err: float | None  # RMS px of the board fit, None when too few markers to fit
    min_side_px: float
    min_markers: int
    image_size: tuple[int, int]  # (w, h)
    board_size: tuple[float, float]  # rack board (w, h), metres
    detect_ms: float
    t: float  # time.monotonic() of the check

    @property
    def seen(self) -> tuple[int, ...]:
        return tuple(sorted(self.corners))

    @property
    def missing(self) -> tuple[int, ...]:
        return tuple(i for i in self.expected if i not in self.corners)

    @property
    def small(self) -> tuple[int, ...]:
        return tuple(i for i in self.seen if self.side_px[i] < self.min_side_px)

    @property
    def usable(self) -> tuple[int, ...]:
        return tuple(i for i in self.seen if self.side_px[i] >= self.min_side_px)

    @property
    def state(self) -> str:
        n = len(self.usable)
        if n == len(self.expected):
            return READY
        return PARTIAL if n >= self.min_markers else NOT_READY

    def problems(self) -> list[str]:
        """Short hints, most important first (ASCII: they are printed to the console too)."""
        out = []
        if self.missing:
            ids = ", ".join(map(str, self.missing))
            out.append(f"missing {ids}: out of view, covered, glare or wrong dictionary")
        for i in self.small:
            out.append(
                f"{i} is {self.side_px[i]:.0f}px (< {self.min_side_px:g}): move camera closer"
            )
        if self.off_layout:
            ids = ", ".join(map(str, self.off_layout))
            out.append(f"{ids} not where rack.yaml puts it: swapped or rotated?")
        if not self.rack.valid and len(self.seen) >= self.min_markers:
            fit = "" if self.rack_err is None else f" (fit {self.rack_err:.1f}px)"
            out.append(f"markers don't fit the rack.yaml layout{fit}")
        return out

    def summary_line(self) -> str:
        n, total = len(self.usable), len(self.expected)
        line = f"{STATE_TEXT[self.state]}: {n}/{total} markers"
        if self.missing:
            line += f", missing {','.join(map(str, self.missing))}"
        if self.small:
            line += f", too small {','.join(map(str, self.small))}"
        return line


class MarkerChecker:
    """Detects the rack's ArUco markers in one image and judges whether the rig is ready."""

    def __init__(
        self,
        geometry: RackGeometry | None = None,
        *,
        min_side_px: float = 20.0,
        min_markers: int = 3,
        max_reproj_err_px: float = 4.0,
    ) -> None:
        self.geometry = geometry if geometry is not None else RackGeometry.load()
        self.expected = tuple(sorted(self.geometry.markers))
        self.min_side_px = float(min_side_px)
        self.min_markers = min(int(min_markers), len(self.expected))
        self.max_reproj_err_px = float(max_reproj_err_px)
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.geometry.dictionary))
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # as the rack solver does
        self._detector = cv2.aruco.ArucoDetector(d, params)
        # no hold and no smoothing: every check judges its own frame only
        self._est = RackFrameEstimator(
            self.geometry,
            min_markers=self.min_markers,
            max_reproj_err_px=self.max_reproj_err_px,
            hold_frames=0,
            smooth_tau_s=0.0,
        )
        self._n = 0

    @classmethod
    def from_config(cls, rack_config: str | None = None, **overrides) -> MarkerChecker:
        """Thresholds from the ``rack`` section of ``configs/default.yaml``."""
        from actus.core.config import load_config

        cfg = load_config()["rack"]
        kw = {
            "min_markers": int(cfg.get("min_markers", 3)),
            "max_reproj_err_px": float(cfg.get("max_reproj_err_px", 4.0)),
        }
        kw.update({k: v for k, v in overrides.items() if v is not None})
        return cls(RackGeometry.load(rack_config or cfg.get("config", DEFAULT_RACK_CONFIG)), **kw)

    def check(self, image: np.ndarray) -> MarkerStatus:
        t0 = time.perf_counter()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        raw_corners, ids, _ = self._detector.detectMarkers(gray)
        found: dict[int, np.ndarray] = {}
        unknown: dict[int, np.ndarray] = {}
        if ids is not None:
            for c, i in zip(raw_corners, ids.ravel(), strict=True):
                mid, c = int(i), np.asarray(c, dtype=np.float64).reshape(4, 2)
                bucket = found if mid in self.geometry.markers else unknown
                if mid not in bucket or _area(c) > _area(bucket[mid]):  # a duplicate: keep largest
                    bucket[mid] = c
        side = {i: _side(c) for i, c in found.items()}

        h, w = image.shape[:2]
        self._n += 1
        self._est.reset()
        seen = sorted(found)
        rack = self._est.solve(
            seen,
            [found[i] for i in seen],
            image_size=(w, h),
            frame_idx=self._n,
            t=time.monotonic(),
        )
        off = []
        if rack.valid:
            for i in seen:
                px = rack_to_img(rack, self.geometry.marker_corners(i)[:, :2])
                if px is None or _rms(px, found[i]) > 2 * self.max_reproj_err_px:
                    off.append(i)
        info = self._est.last_info
        return MarkerStatus(
            expected=self.expected,
            corners=found,
            unknown=unknown,
            side_px=side,
            off_layout=tuple(off),
            rack=rack,
            rack_err=info.get("reproj_err_px"),
            min_side_px=self.min_side_px,
            min_markers=self.min_markers,
            image_size=(w, h),
            board_size=self.geometry.board_size,
            detect_ms=(time.perf_counter() - t0) * 1000.0,
            t=time.monotonic(),
        )


def _area(c: np.ndarray) -> float:
    return float(cv2.contourArea(c.astype(np.float32)))


def _side(c: np.ndarray) -> float:
    """Mean edge length of a (4, 2) marker quad, pixels."""
    return float(np.linalg.norm(np.roll(c, -1, axis=0) - c, axis=1).mean())


# --------------------------------------------------------------------------------------------
# Live monitor (used by record_takes --markers)
# --------------------------------------------------------------------------------------------


@dataclass
class MarkerStats:
    """Counts over the checks of one take."""

    expected: tuple[int, ...]
    checks: int = 0
    states: Counter = field(default_factory=Counter)
    seen: Counter = field(default_factory=Counter)

    def add(self, st: MarkerStatus) -> None:
        self.checks += 1
        self.states[st.state] += 1
        self.seen.update(st.usable)

    def summary(self) -> dict:
        n = max(self.checks, 1)
        return {
            "checks": self.checks,
            "ready_frac": round(self.states[READY] / n, 3),
            "partial_frac": round(self.states[PARTIAL] / n, 3),
            "not_ready_frac": round(self.states[NOT_READY] / n, 3),
            "seen_frac": {str(i): round(self.seen[i] / n, 3) for i in self.expected},
        }


class MarkerMonitor(threading.Thread):
    """Checks the newest frame of a running source ``hz`` times a second, off the recording
    loop. ``source`` is anything with ``latest() -> (seq, image, t) | None`` and ``eof``."""

    def __init__(self, source, checker: MarkerChecker, hz: float = 8.0):
        super().__init__(daemon=True)
        self.source, self.checker, self.hz = source, checker, float(hz)
        self._lock = threading.Lock()
        self._status: MarkerStatus | None = None
        self._stats = MarkerStats(checker.expected)
        self._stop = threading.Event()

    def run(self) -> None:
        last_seq = None
        while not self._stop.is_set() and not self.source.eof:
            t_next = time.monotonic() + 1.0 / self.hz
            latest = self.source.latest()
            if latest is not None and latest[0] != last_seq:
                last_seq = latest[0]
                st = self.checker.check(latest[1])
                with self._lock:
                    self._status = st
                    self._stats.add(st)
            self._stop.wait(max(0.0, t_next - time.monotonic()))

    def status(self) -> MarkerStatus | None:
        with self._lock:
            return self._status

    def reset_stats(self) -> None:
        with self._lock:
            self._stats = MarkerStats(self.checker.expected)

    def summary(self) -> dict:
        with self._lock:
            out = self._stats.summary()
        g = self.checker.geometry
        return {
            **out,
            "hz": self.hz,
            "dictionary": g.dictionary,
            "min_side_px": self.checker.min_side_px,
            "min_markers": self.checker.min_markers,
            "rack_measured": g.measured,
        }

    def stop(self) -> None:
        self._stop.set()


# --------------------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------------------


def _put(img, text, org, scale=0.5, color=(255, 255, 255), thick=1, outline=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    if outline:  # dark halo, legible over any camera image
        cv2.putText(img, text, org, font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thick, cv2.LINE_AA)


def _marker_color(st: MarkerStatus, mid: int) -> tuple[int, int, int]:
    if mid in st.small:
        return RED
    return ORANGE if mid in st.off_layout else GREEN


def draw_markers(view: np.ndarray, st: MarkerStatus, scale: float, labels: bool = True) -> None:
    """Outline every detected marker on ``view`` (the camera image resized by ``scale``)."""
    thick = 2 if labels else 1
    for group, is_rack in ((st.unknown, False), (st.corners, True)):
        for mid, c in group.items():
            color = _marker_color(st, mid) if is_rack else GREY
            pts = (c * scale).round().astype(np.int32)
            cv2.polylines(view, [pts], True, color, thick, cv2.LINE_AA)
            cv2.circle(view, tuple(pts[0]), 3, color, -1)  # corner 0 = the marker's top-left
            if labels:  # beside the marker, so the label never hides it
                x, y = pts[:, 0].max() + 5, int(pts[:, 1].mean()) + 8
                _put(view, str(mid), (int(x), y), 0.7 if is_rack else 0.5, color, 2)


def draw_board(view: np.ndarray, st: MarkerStatus, scale: float, zones=None) -> None:
    """Board outline and protocol zones where the rack solve puts them."""
    if not st.rack.valid:
        return
    bw, bh = st.board_size
    polys = [([[0, 0], [bw, 0], [bw, bh], [0, bh]], GREY)]
    polys += [(z, GREEN) for z in (zones or {}).values()]
    for pts, color in polys:
        px = rack_to_img(st.rack, pts)
        if px is not None and np.isfinite(px).all():
            pts_px = (px * scale).round().astype(np.int32)
            cv2.polylines(view, [pts_px], True, color, 1, cv2.LINE_AA)


def render_check(
    image: np.ndarray,
    st: MarkerStatus,
    *,
    view_width: int = 640,
    zones=None,
    simulated: bool = False,
    rate_hz: float | None = None,
) -> np.ndarray:
    """The standalone checker's window: a state banner above the feed (with marker outlines,
    board and zones drawn on it) and a panel below with the per-marker list and hints. Nothing
    is written over the camera image except the outlines."""
    scale = min(1.0, view_width / image.shape[1])
    feed = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image.copy()
    draw_board(feed, st, scale, zones)
    draw_markers(feed, st, scale)
    w = feed.shape[1]
    hints = st.problems()[:3]

    banner = np.empty((36, w, 3), np.uint8)
    banner[:] = STATE_COLOR[st.state]
    text = f"{STATE_TEXT[st.state]}  {len(st.usable)}/{len(st.expected)} markers"
    if simulated:
        text += "  SIMULATED"
    _put(banner, text, (10, 25), 0.7, (255, 255, 255), 2, outline=False)

    panel = np.full((58 + 20 * len(hints), w, 3), 32, np.uint8)
    x = 10
    for mid in st.expected:
        if mid in st.missing:
            txt, color = f"{mid}: --", RED
        elif mid in st.small:
            txt, color = f"{mid}: {st.side_px[mid]:.0f}px small", RED
        else:
            txt, color = f"{mid}: {st.side_px[mid]:.0f}px", _marker_color(st, mid)
        _put(panel, txt, (x, 20), 0.5, color, outline=False)
        x += 18 + cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
    fit = "-" if st.rack_err is None else f"{st.rack_err:.1f}px"
    _put(panel, f"fit {fit}", (x + 4, 20), 0.5, GREEN if st.rack.valid else AMBER, outline=False)
    for i, hint in enumerate(hints):
        _put(panel, hint, (10, 42 + 20 * i), 0.45, AMBER, outline=False)
    foot = "q quit"
    if rate_hz is not None:
        foot += f"   {rate_hz:4.1f} checks/s   {st.detect_ms:4.1f} ms"
    if st.unknown:
        foot += f"   ignored ids {','.join(map(str, sorted(st.unknown)))}"
    _put(panel, foot, (10, panel.shape[0] - 10), 0.45, (200, 200, 200), outline=False)
    return np.vstack([banner, feed, panel])


def draw_marker_strip(view: np.ndarray, st: MarkerStatus | None, scale: float) -> None:
    """The recorder's compact version: thin outlines and one status line above the key help."""
    h = view.shape[0]
    y = h - 36
    if st is None or time.monotonic() - st.t > STALE_S:
        _put(view, "MARKERS ?", (12, y), 0.5, GREY)
        return
    draw_markers(view, st, scale, labels=False)
    _put(view, "MARKERS", (12, y), 0.5, (220, 220, 220))
    x = 90
    for mid in st.expected:
        color = RED if mid not in st.usable else _marker_color(st, mid)
        _put(view, str(mid), (x, y), 0.5, color, 2)
        x += 18
    _put(view, STATE_TEXT[st.state], (x + 8, y), 0.5, STATE_COLOR[st.state], 2)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Live ArUco marker check for the recording rig (run before record_takes).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Keys: q / Esc quit. Exit status 0 when the last check was READY.",
    )
    ap.add_argument(
        "--source", default="0", help="webcam index, http/rtsp URL, file, replay:<name>"
    )
    ap.add_argument("--rack", help=f"rack geometry YAML (default {DEFAULT_RACK_CONFIG})")
    ap.add_argument("--protocol", default="protocols/bas_boxes.yaml", help="zones to overlay")
    ap.add_argument(
        "--min-side-px", type=float, default=20.0, help="smallest usable marker side (default 20)"
    )
    ap.add_argument("--min-markers", type=int, help="markers for PARTIAL (default: config, 3)")
    ap.add_argument("--view-width", type=int, default=640, help="window width (default 640)")
    ap.add_argument("--width", type=int, default=1280, help="webcam resolution request")
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=30.0, help="webcam fps request / replay rate")
    ap.add_argument(
        "--api", default="any", choices=["any", "dshow", "msmf", "avfoundation", "ffmpeg"]
    )
    ap.add_argument("--duration", type=float, help="stop after this many seconds")
    ap.add_argument("--no-preview", action="store_true", help="no window: print status changes")
    ap.add_argument("--warmup-timeout", type=float, default=15.0)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from actus.protocol.schema import load_protocol
    from actus.spatial.zones import as_polygon
    from tools.record_takes import open_source

    checker = MarkerChecker.from_config(
        args.rack, min_side_px=args.min_side_px, min_markers=args.min_markers
    )
    geo = checker.geometry
    zones = {k: as_polygon(v) for k, v in load_protocol(args.protocol).zones.items()}
    print(f"expecting markers {', '.join(map(str, checker.expected))} ({geo.dictionary}, "
          f"{geo.marker_size * 1000:g} mm) from {args.source}")  # fmt: skip
    if not geo.measured:
        print(
            "NOTE: configs/rack.yaml is placeholder geometry (measured: false); 'fit' may be poor"
        )

    src = open_source(args.source, args)
    src.start()
    simulated = bool(src.describe().get("simulated"))
    if simulated:
        print("SIMULATED source: rendered replay frames, not a camera")
    deadline = time.monotonic() + args.warmup_timeout
    while src.latest() is None:
        if time.monotonic() > deadline or src.eof:
            src.stop()
            raise SystemExit(f"no frames from {args.source}; check --source")
        time.sleep(0.01)

    preview = not args.no_preview
    if preview:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        with contextlib.suppress(cv2.error):  # not every GUI backend has always-on-top
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
        print("q / Esc to quit")
    t_start = time.monotonic()
    last_seq, st, prev_line, shown = None, None, None, False
    stamps: list[float] = []
    try:
        while True:
            if args.duration and time.monotonic() - t_start >= args.duration:
                break
            if src.eof:
                print("video source ended")
                break
            latest = src.latest()
            if latest is not None and latest[0] != last_seq:
                last_seq = latest[0]
                st = checker.check(latest[1])
                now = time.monotonic()
                stamps = [s for s in stamps if now - s < 2.0] + [now]
                line = st.summary_line()
                if line != prev_line:
                    hints = st.problems()
                    print(line + (f"  ({hints[0]})" if hints else ""))
                    prev_line = line
                if preview:
                    span = stamps[-1] - stamps[0]
                    rate = (len(stamps) - 1) / span if span > 0 else None
                    view = render_check(
                        latest[1],
                        st,
                        view_width=args.view_width,
                        zones=zones,
                        simulated=simulated,
                        rate_hz=rate,
                    )
                    cv2.imshow(WINDOW, view)
                    shown = True
            if preview:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if shown and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break  # window closed with its X button
            else:
                time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        src.stop()
        if preview:
            cv2.destroyAllWindows()
    if st is None:
        print("no frame was checked")
        return 1
    print(f"last check: {st.summary_line()}")
    return 0 if st.state == READY else 1


if __name__ == "__main__":
    raise SystemExit(main())
