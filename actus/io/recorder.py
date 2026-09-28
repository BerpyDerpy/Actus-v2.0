"""Segmented MP4 recording of the ``frames`` and ``annotated`` bus topics.

Each stream (raw, annotated) gets its own numbered segment files under
``runs/<run_id>/video/<stream>_<NNN>.mp4``, rolled over every ``segment_s`` seconds of the
frames' own ``t_capture`` (never wall-clock, and never guessed from ``frame_idx``/fps — frame
index is the clock). Each segment gets a sidecar ``<stream>_<NNN>.frames.csv`` with one row per
written frame, so a frame in the video can be matched back to its ``frame_idx``.

    rec = Recorder(bus, run_dir, fps=30.0, segment_s=60.0)
    rec.start()
    ...
    rec.stop()
"""

from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from actus.core.bus import Bus, BusClosed, Subscription
from actus.core.contracts import FramePacket, Topic

DEFAULT_CODEC = "mp4v"


@dataclass(slots=True)
class SegmentInfo:
    path: Path
    sidecar: Path
    first_frame_idx: int
    frames_written: int = 0


class SegmentWriter:
    """Writes one topic's ``FramePacket``s to segmented MP4 files with frame-index sidecars."""

    def __init__(
        self,
        out_dir: str | Path,
        stream_name: str,
        *,
        fps: float,
        segment_s: float = 60.0,
        codec: str = DEFAULT_CODEC,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        if segment_s <= 0:
            raise ValueError("segment_s must be > 0")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stream_name = stream_name
        self.fps = float(fps)
        self.segment_s = float(segment_s)
        self.codec = codec
        self._writer: cv2.VideoWriter | None = None
        self._sidecar_f = None
        self._sidecar_w = None
        self._seg_idx = -1
        self._seg_start_t: float | None = None
        self._size: tuple[int, int] | None = None
        self.segments: list[SegmentInfo] = []
        self.errors: list[str] = []  # the most recent errors (capped); error_count has them all
        self.error_count = 0
        self.frames_dropped = 0  # frames skipped while a failed writer is backing off
        self.retry_s = 5.0
        self._retry_after: float | None = None

    def note_error(self, msg: str) -> None:
        self.error_count += 1
        self.errors.append(msg)
        if len(self.errors) > 50:
            del self.errors[:-50]

    def write(self, frame: FramePacket) -> None:
        if (
            self._writer is None
            and self._retry_after is not None
            and time.monotonic() < self._retry_after
        ):
            # The last open failed (codec missing, disk full): don't retry on every frame.
            self.frames_dropped += 1
            return
        img = frame.image
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        rolled_over = (
            self._writer is None
            or self._seg_start_t is None
            or frame.t_capture - self._seg_start_t >= self.segment_s
            or self._size != (w, h)
        )
        if rolled_over:
            self._roll(w, h, frame)
        assert self._writer is not None and self._sidecar_w is not None
        self._writer.write(np.ascontiguousarray(img))
        self._sidecar_w.writerow([frame.frame_idx, f"{frame.t_capture:.6f}", frame.source_id])
        self.segments[-1].frames_written += 1

    def _roll(self, w: int, h: int, frame: FramePacket) -> None:
        self._close_segment()
        seg_idx = self._seg_idx + 1  # committed only once the writer opens
        stem = f"{self.stream_name}_{seg_idx:03d}"
        path = self.out_dir / f"{stem}.mp4"
        sidecar = self.out_dir / f"{stem}.frames.csv"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*self.codec), self.fps, (w, h))
        if not writer.isOpened():
            writer.release()
            self._retry_after = time.monotonic() + self.retry_s
            raise RuntimeError(f"could not open a {self.codec!r} writer for {path}")
        self._retry_after = None
        self._seg_idx = seg_idx
        self._size = (w, h)
        self._seg_start_t = frame.t_capture
        self._writer = writer
        # kept open across many write() calls, so a `with` block doesn't fit here.
        self._sidecar_f = open(sidecar, "w", newline="", encoding="utf-8")  # noqa: SIM115
        self._sidecar_w = csv.writer(self._sidecar_f)
        self._sidecar_w.writerow(["frame_idx", "t_capture", "source_id"])
        self.segments.append(SegmentInfo(path, sidecar, frame.frame_idx))

    def _close_segment(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._sidecar_f is not None:
            self._sidecar_f.close()
            self._sidecar_f = None

    def close(self) -> None:
        self._close_segment()


@dataclass(slots=True)
class _Stream:
    topic: Topic
    writer: SegmentWriter
    sub: Subscription | None = None
    thread: threading.Thread | None = None


class Recorder:
    """Subscribes to ``frames`` / ``annotated`` and records each to segmented MP4s."""

    def __init__(
        self,
        bus: Bus,
        run_dir: str | Path,
        *,
        fps: float,
        segment_s: float = 60.0,
        raw: bool = True,
        annotated: bool = True,
        codec: str = DEFAULT_CODEC,
        queue_size: int = 128,
    ) -> None:
        self._bus = bus
        self.video_dir = Path(run_dir) / "video"
        self._queue_size = queue_size
        self._streams: dict[str, _Stream] = {}
        if raw:
            self._streams["raw"] = _Stream(
                Topic.FRAMES,
                SegmentWriter(self.video_dir, "raw", fps=fps, segment_s=segment_s, codec=codec),
            )
        if annotated:
            self._streams["annotated"] = _Stream(
                Topic.ANNOTATED,
                SegmentWriter(
                    self.video_dir, "annotated", fps=fps, segment_s=segment_s, codec=codec
                ),
            )
        self._stop = threading.Event()

    def start(self) -> None:
        for name, stream in self._streams.items():
            stream.sub = self._bus.subscribe(
                stream.topic,
                maxsize=self._queue_size,
                policy="drop_oldest",
                name=f"recorder-{name}",
            )
            stream.thread = threading.Thread(
                target=self._run, args=(stream,), daemon=True, name=f"recorder-{name}"
            )
            stream.thread.start()

    def _run(self, stream: _Stream) -> None:
        sub = stream.sub
        while not self._stop.is_set():
            try:
                env = sub.get(timeout=0.2)
            except BusClosed:
                return
            if env is None:
                continue
            try:
                stream.writer.write(env.payload)
            except (RuntimeError, ValueError, OSError, cv2.error) as e:
                stream.writer.note_error(str(e))

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        for stream in self._streams.values():
            if stream.sub is not None:
                stream.sub.unsubscribe()
        for stream in self._streams.values():
            if stream.thread is not None:
                stream.thread.join(timeout=timeout)
            stream.writer.close()

    def stats(self) -> dict:
        return {
            name: {
                "segments": len(s.writer.segments),
                "frames_written": sum(seg.frames_written for seg in s.writer.segments),
                "frames_dropped": s.writer.frames_dropped,
                "errors": s.writer.error_count,
                "last_error": s.writer.errors[-1] if s.writer.errors else None,
            }
            for name, s in self._streams.items()
        }
