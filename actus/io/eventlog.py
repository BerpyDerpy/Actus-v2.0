"""Tamper-evident JSONL event log at ``runs/<run_id>/events.jsonl``.

Each line is one engine output (``StepEvent``, ``Alert``, ``Prompt``, ``Ask``) or operator
``Answer``, tagged with a schema version, an ISO timestamp and the frame index, plus the
SHA-256 hash chain: every record's ``hash`` covers its own fields (including ``prev_hash``), so
changing or deleting any line breaks the chain from that point on. ``verify_log`` checks it.

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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from actus.core.contracts import SCHEMA_VERSION, Alert, Answer, Ask, Prompt, StepEvent, to_jsonable

LOGGABLE = (StepEvent, Alert, Prompt, Ask, Answer)
GENESIS_HASH = "0" * 64


def _iso_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


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


def verify_log(path: str | Path) -> VerifyResult:
    """Recompute the hash chain and report the first line that doesn't match."""
    prev = GENESIS_HASH
    checked = 0
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
            checked += 1
    return VerifyResult(True, checked)


class EventLog:
    """Appends contracts to the hash-chained JSONL log for one run.

    ``catch_signals`` installs SIGINT/SIGTERM handlers that close the log (writing the
    ``run_end`` summary) before re-raising, so a Ctrl-C during a live demo still leaves a valid
    log. Signal handlers can only be installed from the main thread; construct with
    ``catch_signals=False`` from a worker thread or in tests and rely on ``atexit`` /an explicit
    ``close()``.
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        events_file: str = "events.jsonl",
        schema_version: int = SCHEMA_VERSION,
        catch_signals: bool = True,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / events_file
        self.schema_version = schema_version
        self._lock = threading.Lock()
        self._seq, self._prev_hash = self._resume()
        self._counts: Counter[str] = Counter()
        self._closed = False
        self._final_hash: str | None = None
        self._started_at = _iso_now()
        self._start_perf = time.monotonic()
        self._prev_handlers: dict[int, Any] = {}
        if catch_signals:
            self._install_signal_handlers()
        atexit.register(self._atexit_close)

    def _resume(self) -> tuple[int, str]:
        """Continue an existing log's chain (e.g. a run resumed after a crash) instead of
        starting a fresh genesis, so re-opening the same run never breaks verification."""
        if not self.path.exists():
            return 0, GENESIS_HASH
        last: str | None = None
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last = line
        if last is None:
            return 0, GENESIS_HASH
        rec = json.loads(last)
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

    def _on_signal(self, signum: int, frame: Any) -> None:
        self.close(reason=f"signal:{signal.Signals(signum).name}")
        prev = self._prev_handlers.get(signum)
        if callable(prev):
            prev(signum, frame)
        else:
            raise SystemExit(128 + signum)

    def append(self, payload: StepEvent | Alert | Prompt | Ask | Answer) -> dict[str, Any]:
        if not isinstance(payload, LOGGABLE):
            raise TypeError(f"EventLog only logs {[c.__name__ for c in LOGGABLE]}, got {payload!r}")
        body = to_jsonable(payload)
        with self._lock:
            record = {
                "schema_version": self.schema_version,
                "seq": self._seq,
                "ts": _iso_now(),
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
        with self._lock:
            if self._closed:
                return None
            self._closed = True
            summary = {
                "schema_version": self.schema_version,
                "seq": self._seq,
                "ts": _iso_now(),
                "type": "run_end",
                "reason": reason,
                "started_at": self._started_at,
                "duration_s": time.monotonic() - self._start_perf,
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
        return self._final_hash if self._final_hash is not None else self._prev_hash

    @property
    def closed(self) -> bool:
        return self._closed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify an ACTUS event log's tamper-evident chain.")
    ap.add_argument("command", choices=["verify"])
    ap.add_argument("path", type=Path)
    args = ap.parse_args(argv)
    result = verify_log(args.path)
    if result.ok:
        print(f"OK {args.path}: {result.lines_checked} records, chain intact")
        return 0
    print(f"TAMPERED {args.path}: {result.error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
