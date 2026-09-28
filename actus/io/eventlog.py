"""Tamper-evident JSONL event log at ``runs/<run_id>/events.jsonl``.

Each line is one engine output (``StepEvent``, ``Alert``, ``Prompt``, ``Ask``) or operator
``Answer``, tagged with a schema version, an ISO timestamp and the frame index, plus the
SHA-256 hash chain: every record's ``hash`` covers its own fields (including ``prev_hash``), so
changing or deleting any line breaks the chain from that point on. ``verify_log`` checks it.
Deleting lines *from the end* leaves a valid prefix, so a complete log must also end with its
``run_end`` record (``require_end``), and a verifier holding the downlink packet can pin the tip
(``expect_hash``). The CLI does both.

Crash recovery: reopening a run dir resumes the chain. A last line cut off mid-write (no
trailing newline, or not valid JSON) is moved to ``<events>.partial`` and the chain resumes from
the last complete record.

    log = EventLog(run_dir)
    log.append(step_event)          # -> the record written, with its "hash"
    ...
    summary = log.close()           # run_end summary; also runs on Ctrl-C / SIGTERM / atexit

Validate a log from the shell: ``python -m actus.io.eventlog verify runs/<run_id>/events.jsonl``
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import hashlib
import json
import signal
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from actus.core.contracts import SCHEMA_VERSION, Alert, Answer, Ask, Prompt, StepEvent, to_jsonable

LOGGABLE = (StepEvent, Alert, Prompt, Ask, Answer)
GENESIS_HASH = "0" * 64


def _iso_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def iso_at(posix_s: float) -> str:
    """ISO-8601 UTC timestamp (millisecond precision) for POSIX seconds."""
    dt = datetime.fromtimestamp(posix_s, UTC)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _canonical(record: dict[str, Any]) -> str:
    body = {k: v for k, v in record.items() if k != "hash"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def _hash_record(record: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(record).encode("utf-8")).hexdigest()


@dataclass(slots=True)
class VerifyResult:
    ok: bool
    lines_checked: int
    error: str | None = None
    bad_line: int | None = None
    ended: bool = False  # the last record is a run_end
    tip: str | None = None  # the last record's hash


def verify_log(
    path: str | Path, *, require_end: bool = False, expect_hash: str | None = None
) -> VerifyResult:
    """Recompute the hash chain and report the first line that doesn't match.

    ``require_end`` also fails a chain whose last record is not ``run_end`` (trailing records
    deleted, or a run that never closed). ``expect_hash`` fails unless the chain ends exactly at
    that hash (e.g. ``log_hash`` from the run's downlink packet)."""
    prev = GENESIS_HASH
    checked = 0
    last_type: str | None = None
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as e:
                return VerifyResult(False, checked, f"line {lineno}: invalid JSON ({e})", lineno)
            if rec.get("prev_hash") != prev:
                return VerifyResult(
                    False, checked, f"line {lineno}: prev_hash does not match", lineno
                )
            recomputed = _hash_record(rec)
            if recomputed != rec.get("hash"):
                return VerifyResult(
                    False, checked, f"line {lineno}: hash mismatch (tampered)", lineno
                )
            prev = rec["hash"]
            last_type = rec.get("type")
            checked += 1
    ended = last_type == "run_end"
    tip = prev if checked else None
    if require_end and not ended:
        return VerifyResult(
            False,
            checked,
            "chain intact but the last record is not run_end (truncated, or never closed)",
            None,
            ended,
            tip,
        )
    if expect_hash is not None and tip != expect_hash:
        return VerifyResult(
            False, checked, f"chain ends at {tip}, expected {expect_hash}", None, ended, tip
        )
    return VerifyResult(True, checked, ended=ended, tip=tip)


class EventLog:
    """Appends contracts to the hash-chained JSONL log for one run.

    ``catch_signals`` installs SIGINT/SIGTERM handlers that close the log (writing the
    ``run_end`` summary) before re-raising, so a Ctrl-C during a live demo still leaves a valid
    log. Signal handlers can only be installed from the main thread; construct with
    ``catch_signals=False`` from a worker thread or in tests and rely on ``atexit`` /an explicit
    ``close()``. A signal that lands while this same thread is mid-write is deferred until the
    write finishes (the lock is not re-entrant, and a half-written record must not be followed
    by ``run_end``). ``close()`` removes the atexit hook and restores the previous handlers.

    ``append`` after ``close`` raises: nothing may follow ``run_end``.

    ``clock`` (optional) returns POSIX seconds and replaces the wall clock for ``ts``,
    ``started_at`` and ``duration_s``. ``tools/run_offline.py`` passes one derived from the
    pipeline's frame clock, so re-running the same input yields a byte-identical log.
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        events_file: str = "events.jsonl",
        schema_version: int = SCHEMA_VERSION,
        catch_signals: bool = True,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / events_file
        self.schema_version = schema_version
        self._clock = clock
        self._lock = threading.Lock()
        self._owner: int | None = None  # thread id holding _lock
        self._deferred_signal: tuple[int, Any] | None = None
        self._seq, self._prev_hash = self._resume()
        self._counts: Counter[str] = Counter()
        self._closed = False
        self._final_hash: str | None = None
        self._started_at = self._now_iso()
        self._start_perf = self._elapsed_base()
        self._prev_handlers: dict[int, Any] = {}
        if catch_signals:
            self._install_signal_handlers()
        atexit.register(self._atexit_close)

    def _now_iso(self) -> str:
        return _iso_now() if self._clock is None else iso_at(self._clock())

    def _elapsed_base(self) -> float:
        return time.monotonic() if self._clock is None else self._clock()

    def _resume(self) -> tuple[int, str]:
        """Continue an existing log's chain (e.g. a run resumed after a crash) instead of
        starting a fresh genesis, so re-opening the same run never breaks verification."""
        if not self.path.exists():
            return 0, GENESIS_HASH
        data = self.path.read_bytes()
        good_end = 0  # byte offset just past the last complete, parseable record
        last_rec: dict[str, Any] | None = None
        pos = 0
        for line in data.splitlines(keepends=True):
            end = pos + len(line)
            body = line.strip()
            if body:
                if not line.endswith(b"\n"):
                    break  # the process died mid-write
                try:
                    rec = json.loads(body)
                except ValueError:
                    break
                last_rec = rec
            good_end = end
            pos = end
        tail = data[good_end:]
        if tail.strip():
            if len([ln for ln in tail.splitlines() if ln.strip()]) > 1:
                # Only the very last record can be an interrupted write; anything else is damage.
                raise ValueError(
                    f"{self.path}: unreadable record before the end of the log; not resuming"
                )
            partial = self.path.with_name(self.path.name + ".partial")
            with open(partial, "ab") as f:
                f.write(tail if tail.endswith(b"\n") else tail + b"\n")
            with open(self.path, "r+b") as f:
                f.truncate(good_end)
        if last_rec is None:
            return 0, GENESIS_HASH
        rec = last_rec
        return int(rec["seq"]) + 1, str(rec["hash"])

    def _install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
            if sig is None:
                continue
            try:
                self._prev_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._on_signal)
            except (ValueError, OSError):
                pass  # not the main thread, or unsupported on this platform

    def _restore_signal_handlers(self) -> None:
        for sig, prev in list(self._prev_handlers.items()):
            with contextlib.suppress(ValueError, OSError, TypeError):
                if signal.getsignal(sig) == self._on_signal:
                    signal.signal(sig, prev)
        self._prev_handlers.clear()

    def _on_signal(self, signum: int, frame: Any) -> None:
        if self._owner == threading.get_ident():
            # This thread is inside append/close holding the lock: finish that first.
            self._deferred_signal = (signum, frame)
            return
        self._handle_signal(signum, frame)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        prev = self._prev_handlers.get(signum)
        self.close(reason=f"signal:{signal.Signals(signum).name}")
        if prev == signal.SIG_IGN:
            return
        if callable(prev):
            prev(signum, frame)
        else:
            raise SystemExit(128 + signum)

    @contextlib.contextmanager
    def _locked(self):
        with self._lock:
            self._owner = threading.get_ident()
            try:
                yield
            finally:
                self._owner = None
        deferred, self._deferred_signal = self._deferred_signal, None
        if deferred is not None:
            self._handle_signal(*deferred)

    def append(self, payload: StepEvent | Alert | Prompt | Ask | Answer) -> dict[str, Any]:
        if not isinstance(payload, LOGGABLE):
            raise TypeError(f"EventLog only logs {[c.__name__ for c in LOGGABLE]}, got {payload!r}")
        body = to_jsonable(payload)
        with self._locked():
            if self._closed:
                raise RuntimeError(f"event log {self.path} is closed; nothing may follow run_end")
            record = {
                "schema_version": self.schema_version,
                "seq": self._seq,
                "ts": self._now_iso(),
                **body,
                "prev_hash": self._prev_hash,
            }
            record["hash"] = _hash_record(record)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._prev_hash = record["hash"]
            self._seq += 1
            self._counts[body["type"]] += 1
        return record

    def close(self, reason: str = "normal") -> dict[str, Any] | None:
        """Write the ``run_end`` summary. Idempotent: only the first call writes anything."""
        summary = self._close(reason)
        if summary is not None:
            with contextlib.suppress(Exception):
                atexit.unregister(self._atexit_close)
            self._restore_signal_handlers()
        return summary

    def _close(self, reason: str) -> dict[str, Any] | None:
        with self._locked():
            if self._closed:
                return None
            self._closed = True
            summary = {
                "schema_version": self.schema_version,
                "seq": self._seq,
                "ts": self._now_iso(),
                "type": "run_end",
                "reason": reason,
                "started_at": self._started_at,
                "duration_s": self._elapsed_base() - self._start_perf,
                "event_counts": dict(self._counts),
                "prev_hash": self._prev_hash,
            }
            summary["hash"] = _hash_record(summary)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary, separators=(",", ":")) + "\n")
            self._prev_hash = summary["hash"]
            self._final_hash = summary["hash"]
        return summary

    def _atexit_close(self) -> None:
        with contextlib.suppress(Exception):
            self.close(reason="atexit")

    @property
    def final_hash(self) -> str:
        """The current chain tip (the last written record's hash), for the downlink packet."""
        return self._prev_hash

    @property
    def closed(self) -> bool:
        return self._closed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify an ACTUS event log's tamper-evident chain.")
    ap.add_argument("command", choices=["verify"])
    ap.add_argument("path", type=Path)
    ap.add_argument(
        "--allow-open",
        action="store_true",
        help="accept a log without a final run_end record (a run still in progress)",
    )
    ap.add_argument(
        "--expect-hash", default=None, help="the chain tip to require, e.g. downlink log_hash"
    )
    ap.add_argument(
        "--downlink",
        type=Path,
        default=None,
        help="read --expect-hash from this downlink.json's log_hash",
    )
    args = ap.parse_args(argv)
    expect = args.expect_hash
    if expect is None and args.downlink is not None:
        expect = json.loads(args.downlink.read_text(encoding="utf-8")).get("log_hash")
    result = verify_log(args.path, require_end=not args.allow_open, expect_hash=expect)
    if result.ok:
        pinned = ", tip matches" if expect else ""
        print(f"OK {args.path}: {result.lines_checked} records, chain intact{pinned}")
        return 0
    print(f"TAMPERED {args.path}: {result.error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
