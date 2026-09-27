"""T-REC tools/check_markers.py: the live ArUco rig check, standalone and inside record_takes."""

import itertools
import json
import sys
import time

import numpy as np
import pytest

from actus.core.config import REPO_ROOT
from actus.spatial.rack_frame import RackGeometry

cv2 = pytest.importorskip("cv2")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tools import check_markers as cm  # noqa: E402
from tools import record_takes as rt  # noqa: E402

PX_PER_M, MARGIN = 1000, 50


def _replay_frame(i=20):
    from actus.capture.replay import ReplaySource

    for tick in ReplaySource("correct", fps=15, render=True):
        if tick.frame_idx == i:
            return tick.frame.image
    raise AssertionError("replay ended early")


def _board(geo, moved=None):
    """A fronto-parallel board image with the markers at their rack.yaml centres
    (``moved``: {id: (dx_px, dy_px)} puts a marker somewhere else)."""
    w, h = (int(v * PX_PER_M) + 2 * MARGIN for v in geo.board_size)
    img = np.full((h, w, 3), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, geo.dictionary))
    side = round(geo.marker_size * PX_PER_M)
    for mid, (x, y) in geo.markers.items():
        dx, dy = (moved or {}).get(mid, (0, 0))
        x0 = round(x * PX_PER_M) + MARGIN - side // 2 + dx
        y0 = round(y * PX_PER_M) + MARGIN - side // 2 + dy
        m = cv2.aruco.generateImageMarker(d, mid, side)
        img[y0 : y0 + side, x0 : x0 + side] = m[..., None]
    return img


def _cover(img, corners, pad=6):
    x0, y0 = (corners.min(axis=0) - pad).astype(int)
    x1, y1 = (corners.max(axis=0) + pad).astype(int)
    out = img.copy()
    out[y0:y1, x0:x1] = 128
    return out


def test_replay_frame_is_ready():
    chk = cm.MarkerChecker.from_config()
    st = chk.check(_replay_frame())
    assert st.seen == (0, 1, 2, 3) and st.missing == () and st.small == ()
    assert st.state == cm.READY
    assert st.rack.valid and st.rack_err < 2.0 and st.off_layout == ()
    assert st.problems() == []


def test_covered_markers_are_partial_then_not_ready():
    chk = cm.MarkerChecker.from_config()
    img = _replay_frame()
    full = chk.check(img)
    one = chk.check(_cover(img, full.corners[2]))
    assert one.missing == (2,) and one.state == cm.PARTIAL  # 3 left: the rack still solves
    assert one.rack.valid
    assert one.problems()[0].startswith("missing 2")
    two = chk.check(_cover(_cover(img, full.corners[2]), full.corners[3]))
    assert two.missing == (2, 3) and two.state == cm.NOT_READY
    assert two.summary_line() == "NOT READY: 2/4 markers, missing 2,3"


def test_small_markers_do_not_count():
    chk = cm.MarkerChecker(RackGeometry.load(), min_side_px=1000)
    st = chk.check(_replay_frame())
    assert st.seen == (0, 1, 2, 3) and st.small == (0, 1, 2, 3) and st.usable == ()
    assert st.state == cm.NOT_READY
    assert any("move camera closer" in p for p in st.problems())


def test_marker_off_layout_is_flagged_but_still_ready():
    geo = RackGeometry.load()
    chk = cm.MarkerChecker(geo)
    good = chk.check(_board(geo))
    assert good.state == cm.READY and good.off_layout == () and good.rack_err < 1.0
    moved = chk.check(_board(geo, moved={3: (150, -120)}))
    assert moved.seen == (0, 1, 2, 3) and moved.state == cm.READY  # layout isn't part of READY
    assert moved.rack.valid and moved.off_layout == (3,)  # RANSAC fits 0-2, 3 is the outlier
    assert any("not where rack.yaml" in p for p in moved.problems())


def test_unknown_ids_are_ignored():
    geo = RackGeometry.load()
    chk = cm.MarkerChecker(RackGeometry(markers={0: geo.markers[0]}, marker_size=0.06))
    st = chk.check(_board(geo))
    assert st.seen == (0,) and sorted(st.unknown) == [1, 2, 3]
    assert st.state == cm.READY  # min_markers is capped at the markers the rack has


def test_render_views():
    chk = cm.MarkerChecker.from_config()
    img = _replay_frame()
    st = chk.check(img)
    view = cm.render_check(img, st, view_width=320, simulated=True, rate_hz=10.0)
    assert view.shape[1] == 320 and view.shape[0] > 240  # banner and panel around the feed
    strip = img.copy()
    cm.draw_marker_strip(strip, st, 1.0)
    cm.draw_marker_strip(strip, None, 1.0)  # no check yet: "MARKERS ?"
    assert not np.array_equal(strip, img)


class _StaticSource:
    """A camera that keeps delivering the same image as new frames."""

    def __init__(self, img):
        self.img, self.eof = img, False
        self._seq = itertools.count()

    def latest(self):
        return next(self._seq), self.img, time.monotonic()


def test_monitor_collects_take_stats():
    mon = cm.MarkerMonitor(_StaticSource(_replay_frame()), cm.MarkerChecker.from_config(), hz=50)
    mon.start()
    try:
        deadline = time.monotonic() + 5
        while mon.summary()["checks"] < 3 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert mon.status().state == cm.READY
        s = mon.summary()
        assert s["checks"] >= 3 and s["ready_frac"] == 1.0
        assert s["seen_frac"] == {"0": 1.0, "1": 1.0, "2": 1.0, "3": 1.0}
        mon.reset_stats()
        assert mon.summary()["checks"] <= 1
    finally:
        mon.stop()


def test_cli_headless_replay_exits_ready(capsys):
    rc = cm.main(["--source", "replay:correct", "--no-preview", "--duration", "1.0"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SIMULATED" in out and "last check: READY: 4/4 markers" in out


def test_record_takes_markers_flag_writes_summary(tmp_path):
    rt.main([
        "--operator", "op00", "--scenario", "selftest", "--source", "replay:correct",
        "--fps", "15", "--duration", "2", "--no-preview", "--out", str(tmp_path), "--markers",
    ])  # fmt: skip
    meta = json.loads((tmp_path / "op00_selftest_01.json").read_text(encoding="utf-8"))
    mk = meta["markers"]
    assert mk["checks"] >= 5  # ~8 Hz for 2 s
    assert mk["ready_frac"] > 0.8 and set(mk["seen_frac"]) == {"0", "1", "2", "3"}
    assert mk["dictionary"] == "DICT_4X4_50"
    assert meta["cameras"][0]["frame_count_ok"]


def test_record_takes_without_flag_has_no_marker_summary(tmp_path):
    rt.main([
        "--operator", "op00", "--scenario", "selftest", "--source", "replay:correct",
        "--fps", "15", "--duration", "1", "--no-preview", "--out", str(tmp_path),
    ])  # fmt: skip
    meta = json.loads((tmp_path / "op00_selftest_01.json").read_text(encoding="utf-8"))
    assert "markers" not in meta
