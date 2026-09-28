import json

import pytest

from actus.core.contracts import Alert, AlertKind, StepEvent, StepStatus
from actus.io.downlink import build_downlink, encode_downlink, write_downlink
from actus.protocol.schema import load_protocol


def _six_step_run():
    proto = load_protocol("protocols/bas_boxes.yaml")
    events = [
        StepEvent(
            sid,
            StepStatus.VERIFIED,
            i * 2.0,
            i * 2.0 + 1.0,
            0.9 + 0.01 * i,
            {"rule": 0.9, "hoi": 0.85},
            frame_idx=i * 30,
            t=i * 2.0 + 1.0,
        )
        for i, sid in enumerate(proto.step_ids)
    ]
    alerts = [
        Alert(
            AlertKind.OUT_OF_ORDER,
            "Out of sequence: yellow before red",
            (proto.step_ids[1],),
            frame_idx=15,
            t=1.5,
        ),
    ]
    return proto, events, alerts


def test_downlink_packet_is_under_4kb_for_a_six_step_run(tmp_path):
    proto, events, alerts = _six_step_run()
    packet = build_downlink(
        run_id="20260927-000000",
        protocol=proto,
        step_events=events,
        alerts=alerts,
        started_at="2026-09-27T00:00:00.000Z",
        ended_at="2026-09-27T00:05:00.000Z",
        log_hash="a" * 64,
    )
    assert len(packet["steps"]) == len(proto.step_ids)

    path = tmp_path / "downlink.json"
    size = write_downlink(path, packet)
    assert size < 4096
    assert path.stat().st_size == size

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["protocol_id"] == proto.id
    assert data["log_hash"] == "a" * 64
    assert [s["id"] for s in data["steps"]] == proto.step_ids
    assert all(s["status"] == "verified" for s in data["steps"])


def test_downlink_marks_unreached_steps_pending():
    proto, events, _alerts = _six_step_run()
    packet = build_downlink(
        run_id="r",
        protocol=proto,
        step_events=events[:2],  # only the first two steps ever happened
        alerts=(),
        started_at="t0",
        ended_at=None,
        log_hash="b" * 64,
    )
    statuses = {s["id"]: s["status"] for s in packet["steps"]}
    assert statuses[proto.step_ids[0]] == "verified"
    assert statuses[proto.step_ids[-1]] == "pending"


def test_downlink_keeps_the_latest_event_per_step():
    proto, events, _alerts = _six_step_run()
    sid = proto.step_ids[0]
    corrected = StepEvent(
        sid, StepStatus.SKIPPED, None, None, 0.0, frame_idx=999, t=99.0, retroactive=True
    )
    packet = build_downlink(
        run_id="r",
        protocol=proto,
        step_events=[*events, corrected],
        alerts=(),
        started_at="t0",
        ended_at="t1",
        log_hash="c" * 64,
    )
    row = next(s for s in packet["steps"] if s["id"] == sid)
    assert row["status"] == "skipped"
    assert row["retroactive"] is True


def test_encode_downlink_trims_messages_then_raises_if_still_over_budget():
    packet = {
        "steps": [{"id": f"S{i}"} for i in range(20)],
        "alerts": [
            {"kind": "skip", "step_ids": ["S1"], "t": 0.0, "message": "x" * 200} for _ in range(10)
        ],
    }
    # comfortably fits after messages are trimmed
    data = encode_downlink(packet, max_bytes=2000)
    assert len(data) <= 2000
    assert b"message" not in data

    # still too big even with no alerts at all: the step table alone is over budget
    with pytest.raises(ValueError):
        encode_downlink(packet, max_bytes=50)


def test_encode_downlink_keeps_the_latest_alerts_when_there_are_too_many():
    import json

    packet = {
        "steps": [{"id": f"S{i}"} for i in range(6)],
        "alerts": [
            {"kind": "timeout" if i % 2 else "unexpected", "step_ids": ["S1"], "t": float(i)}
            for i in range(200)
        ],
    }
    data = encode_downlink(packet, max_bytes=1500)
    assert len(data) <= 1500
    out = json.loads(data)
    assert out["alerts_total"] == 200
    assert out["alerts_dropped"] == 200 - len(out["alerts"])
    assert out["alert_counts"] == {"unexpected": 100, "timeout": 100}
    assert out["alerts"][-1]["t"] == 199.0  # the most recent alerts are the ones kept
