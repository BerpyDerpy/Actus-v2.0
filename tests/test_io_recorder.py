import time

import cv2
import numpy as np

from actus.core.bus import Bus
from actus.core.contracts import FramePacket, Topic
from actus.io.recorder import Recorder, SegmentWriter


def _decoded_frame_count(path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def test_segment_writer_rolls_over_on_capture_time(tmp_path):
    w = SegmentWriter(tmp_path, "raw", fps=10.0, segment_s=1.0)
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    for i in range(30):  # 3 s at 10 fps -> 3 one-second segments
        w.write(FramePacket(i, i * 0.1, img, "cam0"))
    w.close()

    assert len(w.segments) == 3
    for seg in w.segments:
        assert seg.frames_written == 10
        assert seg.path.exists()
        assert seg.sidecar.exists()
        rows = seg.sidecar.read_text(encoding="utf-8").strip().splitlines()
        assert rows[0] == "frame_idx,t_capture,source_id"
        assert len(rows) - 1 == seg.frames_written
        assert _decoded_frame_count(seg.path) == seg.frames_written


def test_segment_writer_accepts_grey_frames(tmp_path):
    w = SegmentWriter(tmp_path, "raw", fps=10.0, segment_s=60.0)
    grey = np.zeros((48, 64), dtype=np.uint8)
    w.write(FramePacket(0, 0.0, grey, "cam0"))
    w.close()
    assert len(w.segments) == 1
    assert _decoded_frame_count(w.segments[0].path) == 1


def test_segment_writer_rolls_over_on_size_change(tmp_path):
    w = SegmentWriter(tmp_path, "raw", fps=10.0, segment_s=60.0)
    w.write(FramePacket(0, 0.0, np.zeros((48, 64, 3), dtype=np.uint8), "cam0"))
    w.write(FramePacket(1, 0.1, np.zeros((32, 32, 3), dtype=np.uint8), "cam0"))
    w.close()
    assert len(w.segments) == 2
    assert w.segments[0].frames_written == 1
    assert w.segments[1].frames_written == 1


def test_recorder_end_to_end_via_bus(tmp_path):
    bus = Bus()
    rec = Recorder(bus, tmp_path, fps=10.0, segment_s=100.0, raw=True, annotated=True)
    rec.start()
    img = np.zeros((24, 32, 3), dtype=np.uint8)
    n = 15
    try:
        for i in range(n):
            frame = FramePacket(i, i / 10.0, img, "cam0")
            bus.publish(Topic.FRAMES, frame)
            bus.publish(Topic.ANNOTATED, frame)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            stats = rec.stats()
            if stats["raw"]["frames_written"] == n and stats["annotated"]["frames_written"] == n:
                break
            time.sleep(0.05)
    finally:
        rec.stop()

    stats = rec.stats()
    assert stats["raw"]["frames_written"] == n
    assert stats["annotated"]["frames_written"] == n
    assert (rec.video_dir / "raw_000.mp4").exists()
    assert (rec.video_dir / "annotated_000.mp4").exists()
    assert _decoded_frame_count(rec.video_dir / "raw_000.mp4") == n


def test_segment_writer_backs_off_after_a_failed_open(tmp_path, monkeypatch):
    import actus.io.recorder as rec

    class _Closed:
        def isOpened(self):
            return False

        def release(self):
            pass

    opens = []

    def _fake_writer(*args, **kwargs):
        opens.append(args[0])
        return _Closed()

    monkeypatch.setattr(rec.cv2, "VideoWriter", _fake_writer)
    w = SegmentWriter(tmp_path, "raw", fps=10.0, segment_s=60.0)
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    errors = 0
    for i in range(20):
        try:
            w.write(FramePacket(i, i * 0.1, img, "cam0"))
        except RuntimeError:
            errors += 1
    assert errors == 1  # one failed open, then the writer backs off instead of retrying 20x
    assert len(opens) == 1
    assert w.frames_dropped == 19
    assert opens[0].endswith("raw_000.mp4")  # the segment number was not burned
