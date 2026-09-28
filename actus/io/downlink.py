"""The bandwidth-restricted "downlink packet": one JSON file per run, budgeted under 4 KB.

It is the artefact a real downlink (bounded, occasional) could actually carry: the step outcome
table, the alerts raised, run timings and the event log's final hash (so a ground station can
demand the full log and check it matches). Build it from the ``StepEvent``/``Alert`` history
collected during a run, then write it with a hard size budget.

    packet = build_downlink(
        run_id=run_id, protocol=protocol, step_events=events, alerts=alerts,
        started_at=started_at, ended_at=ended_at, log_hash=log.final_hash,
    )
    write_downlink("runs/<run_id>/downlink.json", packet)
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from actus.core.config import resolve_path
from actus.core.contracts import SCHEMA_VERSION, Alert, StepEvent, StepStatus
from actus.protocol.schema import Protocol

DEFAULT_MAX_BYTES = 4096


def _step_row(step_id: str, ev: StepEvent | None) -> dict[str, Any]:
    if ev is None:
        return {"id": step_id, "status": StepStatus.PENDING.value}
    row: dict[str, Any] = {
        "id": step_id,
        "status": ev.status.value,
        "confidence": round(ev.confidence, 3),
    }
    if ev.t_start is not None:
        row["t_start"] = round(ev.t_start, 3)
    if ev.t_end is not None:
        row["t_end"] = round(ev.t_end, 3)
    if ev.retroactive:
        row["retroactive"] = True
    return row


def _alert_row(a: Alert) -> dict[str, Any]:
    return {
        "kind": a.kind.value,
        "step_ids": list(a.step_ids),
        "t": round(a.t, 3),
        "message": a.message,
    }


def build_downlink(
    *,
    run_id: str,
    protocol: Protocol,
    step_events: Iterable[StepEvent],
    alerts: Iterable[Alert] = (),
    started_at: str,
    ended_at: str | None = None,
    log_hash: str,
) -> dict[str, Any]:
    """The step outcome table (one row per protocol step, latest status wins), the alerts
    raised, run timings and the log hash. ``step_events``/``alerts`` should be in chronological
    order; only the latest event per step is kept in the table."""
    latest: dict[str, StepEvent] = {}
    for ev in step_events:
        latest[ev.step_id] = ev
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "protocol_id": protocol.id,
        "started_at": started_at,
        "ended_at": ended_at,
        "steps": [_step_row(sid, latest.get(sid)) for sid in protocol.step_ids],
        "alerts": [_alert_row(a) for a in alerts],
        "log_hash": log_hash,
    }


def _dumps(packet: dict[str, Any]) -> bytes:
    return json.dumps(packet, separators=(",", ":")).encode("utf-8")


def encode_downlink(packet: dict[str, Any], *, max_bytes: int = DEFAULT_MAX_BYTES) -> bytes:
    """Compact JSON encoding, enforced under ``max_bytes``.

    If the packet doesn't fit, it is trimmed in order of least essential first:

    1. alert messages are dropped;
    2. the oldest alerts are dropped, keeping the latest ones. The packet then says so:
       ``alerts_total``, ``alerts_dropped`` and ``alert_counts`` (per kind, over every alert),
       so a long run with many alerts still produces a packet instead of failing at run end.

    Raises ``ValueError`` only if the packet is still over budget with no alerts at all (the
    step table and header alone are too big)."""
    data = _dumps(packet)
    if len(data) <= max_bytes:
        return data
    alerts = list(packet.get("alerts") or [])
    trimmed = dict(packet)
    trimmed["alerts"] = [{k: v for k, v in a.items() if k != "message"} for a in alerts]
    data = _dumps(trimmed)
    if len(data) <= max_bytes:
        return data
    counts: dict[str, int] = {}
    for a in alerts:
        kind = str(a.get("kind"))
        counts[kind] = counts.get(kind, 0) + 1
    rows = trimmed["alerts"]
    # Binary search for the largest number of most recent alerts that fits.
    lo, hi = 0, len(rows)
    best: bytes | None = None
    while lo <= hi:
        keep = (lo + hi) // 2
        candidate = dict(trimmed)
        candidate["alerts"] = rows[len(rows) - keep :] if keep else []
        candidate["alerts_total"] = len(rows)
        candidate["alerts_dropped"] = len(rows) - keep
        candidate["alert_counts"] = counts
        enc = _dumps(candidate)
        if len(enc) <= max_bytes:
            best = enc
            lo = keep + 1
        else:
            hi = keep - 1
    if best is None:
        raise ValueError(
            f"downlink packet is over the {max_bytes}-byte budget even with no alerts "
            f"({len(_dumps(dict(trimmed, alerts=[])))} bytes)"
        )
    return best


def write_downlink(
    path: str | Path, packet: dict[str, Any], *, max_bytes: int = DEFAULT_MAX_BYTES
) -> int:
    """Writes the encoded packet and returns its size in bytes."""
    data = encode_downlink(packet, max_bytes=max_bytes)
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return len(data)
