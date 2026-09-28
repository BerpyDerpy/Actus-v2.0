"""RTSP push (via a bundled MediaMTX + FFmpeg) and WebRTC serving for the GUI.

MediaMTX is the local RTSP/WebRTC server; two FFmpeg processes pipe the ``frames`` and
``annotated`` bus topics into it as raw BGR video, each publishing H.264 to
``rtsp://<target_ip>:<port>/<path>`` (``configs/default.yaml``'s ``stream.paths``). The same
MediaMTX instance serves WebRTC to the GUI on ``stream.webrtc_port``. Every process (MediaMTX
and each pusher) is a ``ManagedProcess``: a crash is a restart with exponential backoff, not a
dead stream. Stream health is published to the ``stats`` topic.

    streamer = Streamer(bus, cfg.stream)
    streamer.start(fps=30.0)
    ...
    streamer.stop()
"""

from __future__ import annotations

import contextlib
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from actus.core.bus import Bus, BusClosed, Subscription
from actus.core.contracts import FramePacket, Stats, Topic
from actus.io.process import BackoffPolicy, ManagedProcess, resolve_binary

STATS_INTERVAL_S = 1.0
_DEFAULT_BACKOFF = BackoffPolicy()


def build_ffmpeg_push_cmd(
    *,
    ffmpeg_bin: str,
    width: int,
    height: int,
    fps: float,
    rtsp_url: str,
    bitrate_kbps: int = 2500,
) -> list[str]:
    """FFmpeg reading raw BGR frames on stdin, encoding H.264 and pushing RTSP."""
    return [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:g}",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-tune",
        "zerolatency",
        "-pix_fmt",
        "yuv420p",
        "-b:v",
        f"{bitrate_kbps}k",
        "-f",
        "rtsp",
        "-rtsp_transport",
        "tcp",
        rtsp_url,
    ]


def build_mediamtx_cmd(*, mediamtx_bin: str, config_path: str | Path | None = None) -> list[str]:
    """MediaMTX takes its config path (if any) as a single positional argument. With none it
    looks for ``mediamtx.yml`` in the *working directory* (then /etc paths), not next to the
    binary, so ``Streamer`` passes ``bin/mediamtx.yml`` explicitly when it exists."""
    cmd = [mediamtx_bin]
    if config_path is not None:
        cmd.append(str(config_path))
    return cmd


@dataclass(frozen=True, slots=True)
class StreamSpec:
    name: str  # "raw" | "annotated"
    topic: Topic
    path: str  # e.g. "actus/raw"


class Streamer:
    """Wires the ``frames``/``annotated`` bus topics into FFmpeg RTSP pushers behind MediaMTX."""

    def __init__(
        self,
        bus: Bus,
        cfg,
        *,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        manage_mediamtx: bool = True,
        backoff: BackoffPolicy = _DEFAULT_BACKOFF,
    ) -> None:
        self._bus = bus
        self._cfg = cfg
        self._popen = popen
        self._manage_mediamtx = manage_mediamtx
        self._backoff = backoff
        self._mediamtx: ManagedProcess | None = None
        self._pushers: dict[str, ManagedProcess] = {}
        self._frame_size: dict[str, tuple[int, int]] = {}
        self._last_stats_t: dict[str, float] = {}
        self._specs = [
            StreamSpec("raw", Topic.FRAMES, cfg.paths["raw"]),
            StreamSpec("annotated", Topic.ANNOTATED, cfg.paths["annotated"]),
        ]
        self._subs: list[Subscription] = []
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._fps = 30.0
        self.errors = 0
        self.last_error: str | None = None

    def start(self, *, fps: float = 30.0) -> None:
        self._fps = float(fps)
        self._stop.clear()
        if self._manage_mediamtx:
            mediamtx_bin = resolve_binary(self._cfg.mediamtx_bin)
            # The setup scripts put MediaMTX's default config beside the binary; use it if there.
            config = Path(mediamtx_bin).with_name("mediamtx.yml")
            self._mediamtx = ManagedProcess(
                build_mediamtx_cmd(
                    mediamtx_bin=mediamtx_bin, config_path=config if config.is_file() else None
                ),
                name="mediamtx",
                backoff=self._backoff,
                popen=self._popen,
            )
            self._mediamtx.start()
        for spec in self._specs:
            sub = self._bus.subscribe(
                spec.topic, maxsize=4, policy="drop_oldest", name=f"stream-{spec.name}"
            )
            self._subs.append(sub)
            t = threading.Thread(
                target=self._run, args=(spec, sub), daemon=True, name=f"stream-{spec.name}"
            )
            self._threads.append(t)
            t.start()

    def _ensure_pusher(self, spec: StreamSpec, w: int, h: int) -> ManagedProcess:
        pusher = self._pushers.get(spec.name)
        if pusher is not None and self._frame_size.get(spec.name) == (w, h):
            return pusher
        if pusher is not None:
            pusher.stop()
        ffmpeg_bin = resolve_binary(self._cfg.ffmpeg_bin)
        url = f"rtsp://{self._cfg.target_ip}:{self._cfg.port}/{spec.path}"
        cmd = build_ffmpeg_push_cmd(
            ffmpeg_bin=ffmpeg_bin,
            width=w,
            height=h,
            fps=self._fps,
            rtsp_url=url,
            bitrate_kbps=self._cfg.bitrate_kbps,
        )
        pusher = ManagedProcess(
            cmd,
            name=f"ffmpeg-{spec.name}",
            backoff=self._backoff,
            want_stdin=True,
            popen=self._popen,
        )
        pusher.start()
        self._pushers[spec.name] = pusher
        self._frame_size[spec.name] = (w, h)
        return pusher

    def _run(self, spec: StreamSpec, sub: Subscription) -> None:
        while not self._stop.is_set():
            try:
                env = sub.get(timeout=0.2)
            except BusClosed:
                return
            if env is None:
                continue
            try:
                frame: FramePacket = env.payload
                img = frame.image
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                h, w = img.shape[:2]
                pusher = self._ensure_pusher(spec, w, h)
                ok = pusher.write_stdin(np.ascontiguousarray(img).tobytes())
                self._maybe_publish_stats(spec, pusher, ok)
            except Exception as exc:  # never let one bad frame kill the stream thread
                self.errors += 1
                self.last_error = f"{spec.name}: {type(exc).__name__}: {exc}"

    def _maybe_publish_stats(
        self, spec: StreamSpec, pusher: ManagedProcess, write_ok: bool
    ) -> None:
        now = time.monotonic()
        if now - self._last_stats_t.get(spec.name, 0.0) < STATS_INTERVAL_S:
            return
        self._last_stats_t[spec.name] = now
        values = {"running": pusher.running, "restarts": pusher.restarts, "write_ok": write_ok}
        stats = Stats(source=f"streamer.{spec.name}", values=values, t=now)
        with contextlib.suppress(BusClosed):
            self._bus.publish(Topic.STATS, stats)

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        for sub in self._subs:
            sub.unsubscribe()
        for t in self._threads:
            t.join(timeout=timeout)
        self._subs.clear()
        self._threads.clear()
        for pusher in self._pushers.values():
            pusher.stop(timeout=timeout)
        self._pushers.clear()
        if self._mediamtx is not None:
            self._mediamtx.stop(timeout=timeout)
            self._mediamtx = None

    def stats(self) -> dict:
        out: dict[str, dict] = {}
        if self.errors:
            out["streamer"] = {"errors": self.errors, "last_error": self.last_error}
        if self._mediamtx is not None:
            out["mediamtx"] = self._mediamtx.stats()
        for name, p in self._pushers.items():
            out[name] = p.stats()
        return out
