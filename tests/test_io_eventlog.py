import json

import pytest

from actus.core.contracts import (
    Alert,
    AlertKind,
    Answer,
    AnswerValue,
    Ask,
    InputVia,
    Prompt,
    StepEvent,
    StepStatus,
)
from actus.io.eventlog import GENESIS_HASH, EventLog, verify_log


def _sample_payloads():
    return [
        StepEvent("S1", StepStatus.VERIFIED, 0.0, 1.2, 0.95, {"rule": 0.9}, frame_idx=10, t=1.2),
        Alert(
            AlertKind.OUT_OF_ORDER,
            "Out of sequence: yellow before red",
            ("S2",),
            frame_idx=20,
            t=2.0,
        ),
        Prompt("S3", "Place the red box in Zone A", frame_idx=21, t=2.1),
        Ask("q1", "S3", "Did you place it?", frame_idx=22, t=2.2),
        Answer("q1", AnswerValue.YES, InputVia.VOICE, t=2.3, frame_idx=23),
    ]


def test_append_writes_a_valid_chained_line_per_event(tmp_path):
    log = EventLog(tmp_path, catch_signals=False)
    for payload in _sample_payloads():
        record = log.append(payload)
        assert record["schema_version"] == 1
        assert "ts" in record and "hash" in record
    log.close()

    lines = log.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(_sample_payloads()) + 1  # + run_end

    first = json.loads(lines[0])
    assert first["prev_hash"] == GENESIS_HASH
    assert first["type"] == "StepEvent"
    assert first["step_id"] == "S1"
    assert first["status"] == "verified"
    assert first["frame_idx"] == 10

    last = json.loads(lines[-1])
    assert last["type"] == "run_end"
    assert last["event_counts"] == {
        "StepEvent": 1,
        "Alert": 1,
        "Prompt": 1,
        "Ask": 1,
        "Answer": 1,
    }

    result = verify_log(log.path)
    assert result.ok
    assert result.lines_checked == len(lines)


def test_append_rejects_non_loggable_types(tmp_path):
    log = EventLog(tmp_path, catch_signals=False)
    with pytest.raises(TypeError):
        log.append("not a contract")  # type: ignore[arg-type]
    log.close()


def test_close_is_idempotent(tmp_path):
    log = EventLog(tmp_path, catch_signals=False)
    log.append(_sample_payloads()[0])
    first = log.close()
    second = log.close()
    assert first is not None
    assert second is None  # already closed; no duplicate run_end line
    lines = log.path.read_text(encoding="utf-8").strip().splitlines()
    assert sum(1 for line in lines if json.loads(line)["type"] == "run_end") == 1


def test_reopening_the_same_run_continues_the_chain(tmp_path):
    log1 = EventLog(tmp_path, catch_signals=False)
    log1.append(_sample_payloads()[0])
    log1.close()

    log2 = EventLog(tmp_path, catch_signals=False)
    log2.append(_sample_payloads()[1])
    log2.close()

    result = verify_log(log2.path)
    assert result.ok
    lines = log2.path.read_text(encoding="utf-8").strip().splitlines()
    seqs = [json.loads(line)["seq"] for line in lines]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)  # no seq reused across the reopen


def test_verify_log_detects_a_one_byte_edit(tmp_path):
    log = EventLog(tmp_path, catch_signals=False)
    for payload in _sample_payloads():
        log.append(payload)
    log.close()

    assert verify_log(log.path).ok

    original = log.path.read_text(encoding="utf-8")
    tampered = original.replace(
        "Out of sequence: yellow before red", "Out of sequence: yellow Before red", 1
    )
    assert tampered != original
    log.path.write_text(tampered, encoding="utf-8")

    result = verify_log(log.path)
    assert not result.ok
    assert result.bad_line is not None


def test_verify_log_detects_a_deleted_line(tmp_path):
    log = EventLog(tmp_path, catch_signals=False)
    for payload in _sample_payloads():
        log.append(payload)
    log.close()

    lines = log.path.read_text(encoding="utf-8").splitlines()
    del lines[1]  # remove one record from the middle of the chain
    log.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = verify_log(log.path)
    assert not result.ok


def test_verify_log_require_end_detects_deleted_trailing_records(tmp_path):
    log = EventLog(tmp_path, catch_signals=False)
    for payload in _sample_payloads():
        log.append(payload)
    log.close()
    tip = log.final_hash

    assert verify_log(log.path, require_end=True, expect_hash=tip).ok

    lines = log.path.read_text(encoding="utf-8").splitlines()
    log.path.write_text("\n".join(lines[:-3]) + "\n", encoding="utf-8")  # drop the tail
    assert verify_log(log.path).ok  # the prefix still chains...
    assert not verify_log(log.path, require_end=True).ok  # ...but it has no run_end
    assert not verify_log(log.path, expect_hash=tip).ok  # ...and does not end at the tip


def test_resume_moves_a_half_written_last_line_aside(tmp_path):
    log1 = EventLog(tmp_path, catch_signals=False)
    log1.append(_sample_payloads()[0])
    log1.append(_sample_payloads()[1])
    # Simulate a crash mid-append: a truncated record with no newline, and no run_end.
    with open(log1.path, "a", encoding="utf-8") as f:
        f.write('{"schema_version":1,"seq":2,"ts":"2026-')

    log2 = EventLog(tmp_path, catch_signals=False)
    log2.append(_sample_payloads()[2])
    log2.close()

    assert verify_log(log2.path, require_end=True).ok
    partial = log2.path.with_name(log2.path.name + ".partial")
    assert partial.read_text(encoding="utf-8").startswith('{"schema_version":1,"seq":2')


def test_resume_refuses_a_log_damaged_before_its_end(tmp_path):
    log1 = EventLog(tmp_path, catch_signals=False)
    for payload in _sample_payloads()[:3]:
        log1.append(payload)
    log1.close()
    lines = log1.path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1][:20]  # corrupt a middle record
    log1.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        EventLog(tmp_path, catch_signals=False)


def test_append_after_close_raises_and_final_hash_is_the_tip(tmp_path):
    log = EventLog(tmp_path, catch_signals=False)
    log.append(_sample_payloads()[0])
    summary = log.close()
    assert log.final_hash == summary["hash"]
    with pytest.raises(RuntimeError):
        log.append(_sample_payloads()[1])
    assert verify_log(log.path, require_end=True, expect_hash=log.final_hash).ok
