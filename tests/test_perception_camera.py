import time

import numpy as np

from actus.capture.camera import Camera, CameraError
from actus.core.bus import Bus
from actus.core.contracts import Topic


class FakeCapture:
    """A ``cv2.VideoCapture``-like stand-in: replays a fixed list of frames (optionally paired
    with explicit ``CAP_PROP_POS_MSEC`` values), with scriptable transient read failures."""

    def __init__(self, frames, *, msec=None, fail_pattern=(), opens=True):
        self._frames = list(frames)
        self._msec = list(msec) if msec is not None else None
        self._fail_pattern = list(fail_pattern)
        self._call = 0
        self._delivered = 0
        self._opened = opens
        self.set_calls: list[tuple[int, float]] = []

    def isOpened(self):
        return self._opened

    def read(self):
        fails_here = self._call < len(self._fail_pattern) and self._fail_pattern[self._call]
        self._call += 1
        if fails_here:
            return False, None
        if self._delivered >= len(self._frames):
            return False, None
        frame = self._frames[self._delivered]
        self._delivered += 1
        return True, frame

    def get(self, prop):
        if self._msec is not None:
            idx = self._delivered - 1
            return self._msec[idx] if 0 <= idx < len(self._msec) else 0.0
        return 0.0

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return True

    def release(self):
        self._opened = False


def _frame(value: int = 0) -> np.ndarray:
    return np.full((48, 64, 3), value, dtype=np.uint8)


def _drain(sub, n_expected, timeout=2.0):
    out = []
    deadline = time.monotonic() + timeout
    while len(out) < n_expected and time.monotonic() < deadline:
        env = sub.get(timeout=0.1)
        if env is not None:
            out.append(env.payload)
    return out


def test_file_source_frame_idx_is_monotonic_from_zero_and_never_loops():
    frames = [_frame(i) for i in range(5)]
    cap = FakeCapture(frames, msec=[100.0 * (i + 1) for i in range(5)])
    cam = Camera("clip.mp4", source_id="test", capture_factory=lambda: cap)
    bus = Bus()
    sub = bus.subscribe(Topic.FRAMES, maxsize=100, name="sub")

    cam.start(bus)
    assert cam.ended.wait(timeout=2.0)
    packets = _drain(sub, 5)

    assert [p.frame_idx for p in packets] == [0, 1, 2, 3, 4]
    assert [p.t_capture for p in packets] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert all(p.source_id == "test" for p in packets)
    assert cam.frames_captured == 5

    # EOF must not loop or reopen: nothing more arrives even after waiting.
    time.sleep(0.2)
    assert sub.get(timeout=0.1) is None


def test_rotation_is_applied_and_recorded():
    import cv2

    img = _frame(0)
    img[0, 0] = (1, 2, 3)  # mark a corner so rotation direction is checkable
    cap = FakeCapture([img], msec=[10.0])
    cam = Camera("clip.mp4", source_id="test", rotation=90, capture_factory=lambda: cap)
    bus = Bus()
    sub = bus.subscribe(Topic.FRAMES, maxsize=10)

    cam.start(bus)
    assert cam.ended.wait(timeout=2.0)
    packets = _drain(sub, 1)

    assert len(packets) == 1
    packet = packets[0]
    assert packet.rotation_applied == 90
    expected = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    assert packet.image.shape == expected.shape
    assert np.array_equal(packet.image, expected)


def test_live_source_retries_transient_failures_then_ends():
    frames = [_frame(i) for i in range(3)]
    # fail, fail, ok, ok, ok, then read() naturally exhausts -> further reads also fail
    cap = FakeCapture(frames, fail_pattern=[True, True])
    cam = Camera(0, source_id="test", max_live_read_failures=5, capture_factory=lambda: cap)
    bus = Bus()
    sub = bus.subscribe(Topic.FRAMES, maxsize=10)

    cam.start(bus)
    assert cam.ended.wait(timeout=2.0)
    packets = _drain(sub, 3)

    assert [p.frame_idx for p in packets] == [0, 1, 2]
    assert cam.read_failures >= 2


def test_live_source_gives_up_after_max_consecutive_failures():
    cap = FakeCapture([], fail_pattern=[True] * 100)
    cam = Camera(0, source_id="test", max_live_read_failures=3, capture_factory=lambda: cap)
    bus = Bus()
    bus.subscribe(Topic.FRAMES, maxsize=10)

    cam.start(bus)
    assert cam.ended.wait(timeout=2.0)
    assert cam.frames_captured == 0
    assert cam.read_failures == 3
    # A disconnect is surfaced, not mistaken for a clean end of stream.
    assert isinstance(cam.error, CameraError)


def test_open_failure_raises_camera_error():
    cap = FakeCapture([], opens=False)
    cam = Camera("clip.mp4", capture_factory=lambda: cap)
    bus = Bus()
    try:
        cam.start(bus)
        raise AssertionError("expected CameraError")
    except CameraError:
        pass
    # Waiters are woken and the half-open capture is released.
    assert cam.ended.is_set()
    assert isinstance(cam.error, CameraError)


def test_file_frame_zero_at_zero_ms_keeps_the_container_clock():
    frames = [_frame(i) for i in range(3)]
    cap = FakeCapture(frames, msec=[0.0, 33.0, 66.0])
    cam = Camera("clip.mp4", source_id="test", capture_factory=lambda: cap)
    bus = Bus()
    sub = bus.subscribe(Topic.FRAMES, maxsize=10)
    cam.start(bus)
    assert cam.ended.wait(timeout=2.0)
    packets = _drain(sub, 3)
    assert [p.t_capture for p in packets] == [0.0, 0.033, 0.066]
    assert cam.clock == "container"


def test_live_device_paths_and_pipelines_are_not_files():
    from actus.capture.camera import _is_file_source

    assert _is_file_source("data/raw/op01_correct_01.mp4")
    assert _is_file_source("C:/takes/op01_correct_01.mp4")
    assert not _is_file_source(0)
    assert not _is_file_source("1")
    assert not _is_file_source("rtsp://10.0.0.5:8554/live")
    assert not _is_file_source("/dev/video0")
    assert not _is_file_source("v4l2src ! videoconvert ! appsink")


def test_stop_ends_the_capture_thread_promptly():
    def infinite_frames():
        while True:
            yield _frame(0)

    class Infinite(FakeCapture):
        def __init__(self):
            super().__init__([])
            self._gen = infinite_frames()

        def read(self):
            return True, next(self._gen)

    cap = Infinite()
    cam = Camera(0, capture_factory=lambda: cap)
    bus = Bus()
    bus.subscribe(Topic.FRAMES, maxsize=4, policy="drop_oldest")

    cam.start(bus)
    time.sleep(0.05)
    cam.stop(timeout=2.0)
    assert cam.ended.is_set()
    assert not cap.isOpened()


def test_invalid_rotation_rejected():
    import pytest

    with pytest.raises(ValueError):
        Camera(0, rotation=45)
