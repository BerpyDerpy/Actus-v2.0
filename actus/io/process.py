"""A subprocess kept alive with exponential-backoff restarts.

Not RTSP-specific: ``streamer.py`` uses one ``ManagedProcess`` for the bundled MediaMTX server
and one per FFmpeg pusher, so both "the RTSP push reconnects" and "MediaMTX itself comes back
after a crash" are the same code path.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


def resolve_binary(path: str | Path) -> str:
    """The given path, a ``.exe`` sibling on Windows, or whatever ``PATH`` resolves it to.

    Falls back to the input unchanged so a missing binary fails with a clear
    ``FileNotFoundError`` from the subprocess call, rather than a confusing lookup error here.
    """
    import shutil

    p = Path(path)
    if p.is_file():
        return str(p)
    if sys.platform == "win32" and p.suffix == "":
        exe = p.with_name(p.name + ".exe")
        if exe.is_file():
            return str(exe)
    found = shutil.which(str(path))
    return found if found else str(path)


@dataclass(frozen=True, slots=True)
class BackoffPolicy:
    initial_s: float = 1.0
    max_s: float = 30.0
    factor: float = 2.0

    def delay(self, attempt: int) -> float:
        """Delay before restart attempt ``attempt`` (1-based). 0 for the first attempt."""
        if attempt <= 0:
            return 0.0
        return min(self.max_s, self.initial_s * (self.factor ** (attempt - 1)))


_DEFAULT_BACKOFF = BackoffPolicy()


class ManagedProcess:
    """Starts ``cmd`` and restarts it with backoff whenever it exits unexpectedly.

    ``stop()`` is the only way to end it cleanly; any other exit is treated as a crash and
    triggers a restart. ``popen`` is injectable so tests can point ``cmd`` at a small Python
    script instead of a real ffmpeg/MediaMTX binary.
    """

    def __init__(
        self,
        cmd: Sequence[str],
        *,
        name: str,
        backoff: BackoffPolicy = _DEFAULT_BACKOFF,
        want_stdin: bool = False,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        self.cmd = list(cmd)
        self.name = name
        self.backoff = backoff
        self._want_stdin = want_stdin
        self._popen = popen
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._watchdog: threading.Thread | None = None
        self.restarts = 0
        self.last_exit_code: int | None = None
        self.last_error: str | None = None
        self._started_at: float | None = None

    def start(self) -> None:
        self._stop.clear()
        self._spawn()
        self._watchdog = threading.Thread(
            target=self._watch, name=f"{self.name}-watchdog", daemon=True
        )
        self._watchdog.start()

    def _spawn(self) -> None:
        with self._lock:
            try:
                self._proc = self._popen(
                    self.cmd,
                    stdin=subprocess.PIPE if self._want_stdin else subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._started_at = time.monotonic()
                self.last_error = None
            except OSError as e:
                self._proc = None
                self.last_error = str(e)

    def _watch(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            proc = self._proc
            if proc is None:
                # the last spawn failed outright (e.g. binary not found): back off and retry
                attempt += 1
                if self._stop.wait(self.backoff.delay(attempt)):
                    return
                self._spawn()
                if self._proc is not None:
                    attempt = 0
                continue
            code = proc.wait()
            if self._stop.is_set():
                return
            self.last_exit_code = code
            self.restarts += 1
            attempt += 1
            if self._stop.wait(self.backoff.delay(attempt)):
                return
            self._spawn()
            if self._proc is not None:
                attempt = 0

    def write_stdin(self, data: bytes) -> bool:
        """Best-effort write; False (never raises) if the process is down or the pipe broke."""
        proc = self._proc
        if proc is None or proc.stdin is None:
            return False
        try:
            proc.stdin.write(data)
            proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError) as e:
            self.last_error = str(e)
            return False

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        proc = self._proc
        if proc is not None:
            if proc.stdin is not None:
                with contextlib.suppress(OSError):
                    proc.stdin.close()
            try:
                proc.terminate()
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=timeout)
            except OSError:
                pass
        if self._watchdog is not None:
            self._watchdog.join(timeout=timeout)

    @property
    def running(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    @property
    def uptime_s(self) -> float | None:
        if self._started_at is None or not self.running:
            return None
        return time.monotonic() - self._started_at

    def stats(self) -> dict:
        return {
            "name": self.name,
            "running": self.running,
            "restarts": self.restarts,
            "last_exit_code": self.last_exit_code,
            "last_error": self.last_error,
            "uptime_s": self.uptime_s,
        }
