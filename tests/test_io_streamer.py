import subprocess
import sys
import time

import numpy as np

from actus.core.bus import Bus
from actus.core.config import load_config
from actus.core.contracts import FramePacket, Topic
from actus.io.streamer import Streamer, build_ffmpeg_push_cmd, build_mediamtx_cmd


def test_build_ffmpeg_push_cmd_has_the_expected_shape():
    cmd = build_ffmpeg_push_cmd(
        ffmpeg_bin="ffmpeg",
        width=640,
        height=480,
        fps=15.0,
        rtsp_url="rtsp://127.0.0.1:8554/actus/raw",
        bitrate_kbps=1500,
    )
    assert cmd[0] == "ffmpeg"
    assert cmd[cmd.index("-s") + 1] == "640x480"
    assert cmd[cmd.index("-r") + 1] == "15"
    assert cmd[cmd.index("-b:v") + 1] == "1500k"
    assert "libx264" in cmd
    assert cmd[-1] == "rtsp://127.0.0.1:8554/actus/raw"


def test_build_mediamtx_cmd_with_and_without_config():
    assert build_mediamtx_cmd(mediamtx_bin="bin/mediamtx") == ["bin/mediamtx"]
    assert build_mediamtx_cmd(mediamtx_bin="bin/mediamtx", config_path="bin/mediamtx.yml") == [
        "bin/mediamtx",
        "bin/mediamtx.yml",
    ]


def _make_fake_popen(out_dir):
    """A ``subprocess.Popen`` stand-in for ffmpeg: a Python process that counts stdin bytes and
    writes the running total to ``<out_dir>/<raw|annotated>.count``, keyed off the real
    ffmpeg-style argv's RTSP path (its last element)."""

    def fake_popen(cmd, **kwargs):
        url = cmd[-1]
        name = "raw" if "raw" in url else "annotated"
        out_path = str(out_dir / f"{name}.count")
        script = (
            "import sys\n"
            "n = 0\n"
            "while True:\n"
            "    chunk = sys.stdin.buffer.read(65536)\n"
            "    if not chunk:\n"
            "        break\n"
            "    n += len(chunk)\n"
            f"    open({out_path!r}, 'w').write(str(n))\n"
        )
        return subprocess.Popen([sys.executable, "-c", script], **kwargs)

    return fake_popen


def test_streamer_pipes_bus_frames_to_the_ffmpeg_pusher(tmp_path):
    cfg = load_config(overrides=["stream.target_ip=127.0.0.1", "stream.port=8554"]).stream
    bus = Bus()
    popen = _make_fake_popen(tmp_path)
    streamer = Streamer(bus, cfg, popen=popen, manage_mediamtx=False)
    streamer.start(fps=10.0)

    img = np.zeros((24, 32, 3), dtype=np.uint8)
    n = 5
    try:
        for i in range(n):
            bus.publish(Topic.FRAMES, FramePacket(i, i / 10.0, img, "cam0"))
            time.sleep(0.05)  # the pusher's queue is bounded (drop_oldest); pace like a live feed
        raw_count_file = tmp_path / "raw.count"
        deadline = time.monotonic() + 5.0
        while not raw_count_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.2)  # let the last write land
    finally:
        streamer.stop()

    assert raw_count_file.exists()
    assert int(raw_count_file.read_text()) == img.nbytes * n
    assert not (tmp_path / "annotated.count").exists()  # no annotated frames were published


def test_streamer_stats_report_pusher_health(tmp_path):
    cfg = load_config(overrides=["stream.target_ip=127.0.0.1", "stream.port=8554"]).stream
    bus = Bus()
    popen = _make_fake_popen(tmp_path)
    streamer = Streamer(bus, cfg, popen=popen, manage_mediamtx=False)
    streamer.start(fps=10.0)
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    try:
        bus.publish(Topic.FRAMES, FramePacket(0, 0.0, img, "cam0"))
        deadline = time.monotonic() + 5.0
        while "raw" not in streamer.stats() and time.monotonic() < deadline:
            time.sleep(0.05)
        stats = streamer.stats()
        assert stats["raw"]["running"] is True
    finally:
        streamer.stop()
