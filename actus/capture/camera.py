"""Threaded frame capture: a webcam, a video file, or an RTSP stream -> ``FramePacket``.

``frame_idx`` is assigned by the capture thread and is monotonic starting at 0. A file source
reaches EOF once and stops; it is never rewound or reopened. ``t_capture`` is
``time.monotonic()`` for a live source (webcam / RTSP) and the container's own timestamp
(seconds) for a file -- never ``frame_idx / fps`` with a guessed fps.

    cam = Camera.from_config(cfg.capture)
    cam.start(bus)
    ...
    cam.stop()

``rotation`` (0/90/180/270, degrees CCW) is applied to every frame before it is published, and
recorded on the packet as ``rotation_applied`` so downstream lanes never need to guess it.
``Camera.rotate()`` changes it live -- e.g. for a "rotate camera" button in a standby preview
before recording starts -- and takes effect on the next captured frame.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from typing import Protocol as TypingProtocol

import numpy as np

from actus.core.bus import Bus
from actus.core.contracts import FramePacket, Topic

ROTATIONS = (0, 90, 180, 270)


class CameraError(RuntimeError):
    """Raised when the capture source cannot be opened."""


class VideoCapture(TypingProtocol):
    """The subset of ``cv2.VideoCapture`` this module relies on (for test doubles)."""

    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def get(self, prop: int) -> float: ...
    def set(self, prop: int, value: float) -> bool: ...
    def release(self) -> None: ...


def _is_file_source(source: int | str) -> bool:
    """A file path is a "file" source; an int index or a ``scheme://`` URL is "live"."""
    if isinstance(source, int):
        return False
    return "://" not in str(source)


class Camera:
    """Threaded capture of one source. Not reentrant: call ``start`` at most once."""

    def __init__(
        self,
        source: int | str,
        *,
        source_id: str = "cam0",
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        rotation: int = 0,
        max_live_read_failures: int = 30,
        capture_factory: Any = None,
    ) -> None:
        if rotation not in ROTATIONS:
            raise ValueError(f"rotation must be one of {ROTATIONS}, got {rotation}")
        self.source = source
        self.source_id = str(source_id)
        self.width = width
        self.height = height
        self.requested_fps = fps
        self.rotation = rotation
        self.max_live_read_failures = max_live_read_failures
        self._is_file = _is_file_source(source)
        self._rotation_lock = threading.Lock()
        # Overridable for tests: a zero-arg callable returning a VideoCapture-like object,
        # in place of the real ``cv2.VideoCapture(source)``.
        self._capture_factory = capture_factory

        self._cap: VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.ended = threading.Event()
        self.opened = False
        self.frames_captured = 0
        self.read_failures = 0
        self.error: Exception | None = None
        self._started_monotonic = 0.0

    @classmethod
    def from_config(cls, cfg: Any) -> Camera:
        return cls(
            cfg.source,
            source_id=cfg.get("source_id", "cam0"),
            width=cfg.get("width"),
            height=cfg.get("height"),
            fps=cfg.get("fps"),
            rotation=cfg.get("rotation", 0),
        )

    # -- lifecycle ------------------------------------------------------------------------------

    def start(self, bus: Bus, *, topic: Topic = Topic.FRAMES) -> None:
        if self._thread is not None:
            raise RuntimeError("Camera.start() called more than once")
        self._open()
        self._thread = threading.Thread(
            target=self._run, args=(bus, topic), name=f"camera-{self.source_id}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float | None = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def rotate(self, *, step: int = 1) -> int:
        """Cycle the live rotation by 90deg x ``step`` (mod 360) while capturing -- e.g. a
        "rotate camera" button in a standby preview before recording starts. Thread-safe;
        takes effect on the next captured frame. Returns the new rotation."""
        with self._rotation_lock:
            idx = ROTATIONS.index(self.rotation)
            self.rotation = ROTATIONS[(idx + step) % len(ROTATIONS)]
            return self.rotation

    def _open(self) -> None:
        import cv2

        if self._capture_factory is not None:
            cap = self._capture_factory()
        else:
            cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise CameraError(f"could not open capture source {self.source!r}")
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.requested_fps:
            cap.set(cv2.CAP_PROP_FPS, self.requested_fps)
        self._cap = cap
        self.opened = True

    def _run(self, bus: Bus, topic: Topic) -> None:
        assert self._cap is not None
        self._started_monotonic = time.monotonic()
        consecutive_failures = 0
        try:
            while not self._stop.is_set():
                ok, image = self._cap.read()
                if not ok or image is None:
                    if self._is_file:
                        break  # EOF: end the stream, never loop or reopen
                    consecutive_failures += 1
                    self.read_failures += 1
                    if consecutive_failures >= self.max_live_read_failures:
                        break
                    continue
                consecutive_failures = 0
                t_capture = self._timestamp()
                with self._rotation_lock:
                    rotation = self.rotation
                if rotation:
                    image = self._rotate(image, rotation)
                packet = FramePacket(
                    self.frames_captured, t_capture, image, self.source_id, rotation
                )
                self.frames_captured += 1
                bus.publish(topic, packet)
        except Exception as e:  # surfaced via `.error`, never raised across the thread boundary
            self.error = e
        finally:
            self.ended.set()
            self._release()

    def _timestamp(self) -> float:
        if self._is_file:
            import cv2

            assert self._cap is not None
            msec = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            if msec and msec > 0:
                return msec / 1000.0
            return time.monotonic() - self._started_monotonic
        return time.monotonic()

    def _rotate(self, image: np.ndarray, rotation: int) -> np.ndarray:
        import cv2

        mapping = {
            90: cv2.ROTATE_90_COUNTERCLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_CLOCKWISE,
        }
        return cv2.rotate(image, mapping[rotation])

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
