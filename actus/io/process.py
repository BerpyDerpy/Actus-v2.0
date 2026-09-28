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

    The backoff only resets once a process has stayed up for ``stable_s``: a child that dies
    straight after every spawn (ffmpeg with the target down) backs off to ``max_s`` rather than
    respawning every ``initial_s`` for ever. ``stop()`` and the watchdog share a lock, so a
    restart can never slip in after ``stop()`` and leave an orphan holding a port.
    """

    def __init__(
        self,
        cmd: Sequence[str],
        *,
        name: str,
        backoff: BackoffPolicy = _DEFAULT_BACKOFF,
        want_stdin: bool = False,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        stable_s: float = 10.0,
    ) -> None:
        self.cmd = list(cmd)
        self.name = name
        self.backoff = backoff
        self.stable_s = float(stable_s)
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
            if self._stop.is_set():
                return
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
                continue
            code = proc.wait()
            if self._stop.is_set():
                return
            # Only a process that stayed up counts as "recovered"; a crash loop keeps backing off.
            started = self._started_at
            if started is not None and time.monotonic() - started >= self.stable_s:
                attempt = 0
            self.last_exit_code = code
            self.restarts += 1
            attempt += 1
            if self._stop.wait(self.backoff.delay(attempt)):
                return
            self._spawn()

    def write_stdin(self, data: bytes) -> bool:
        """Best-effort write; False (never raises) if the process is down or the pipe broke."""
        proc = self._proc
        if proc is None or proc.stdin is None:
            return False
        try:
            proc.stdin.write(data)
            proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError, ValueError) as e:  # ValueError: pipe closed by stop()
            self.last_error = str(e)
            return False

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        with self._lock:  # no spawn can start after this point; see _spawn
            proc = self._proc
        if proc is not None:
            self._end(proc, timeout)
        if self._watchdog is not None:
            self._watchdog.join(timeout=timeout)
        with self._lock:
            late = self._proc
        if late is not None and late is not proc:  # belt and braces
            self._end(late, timeout)

    @staticmethod
    def _end(proc: subprocess.Popen, timeout: float) -> None:
        if proc.stdin is not None:
            # Closing can block if another thread is stuck writing to a stalled child, so do it
            # off-thread and move on to terminate() (which unblocks that write) if it hangs.
            def _close() -> None:
                with contextlib.suppress(OSError, ValueError):
                    proc.stdin.close()

            closer = threading.Thread(target=_close, daemon=True)
            closer.start()
            closer.join(0.5)
            # Give a child reading stdin (ffmpeg) a moment to exit cleanly on EOF.
            with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                proc.wait(timeout=min(1.0, timeout))
        try:
            if proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                proc.wait(timeout=timeout)
        except OSError:
            pass

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
