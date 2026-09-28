"""T-EVAL: the evaluation harness. Ground truth from annotations, scoring against engine outputs,
the report's shape, and an end-to-end run on the SIMULATED replay (no cameras, no models)."""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from actus.core.contracts import (  # noqa: E402
    Alert,
    AlertKind,
    Answer,
    Ask,
    StepEvent,
    StepStatus,
)
from actus.io.eventlog import EventLog  # noqa: E402
from actus.protocol.schema import load_protocol  # noqa: E402
from tools.evaluate import (  # noqa: E402
    DEFINITIONS,
    METRICS,
    FrameClock,
    aggregate,
    build_report,
    derive_truth,
    load_spec,
    read_events,
    read_label_rows,
    replay_takes,
    rescore,
    run_evaluation,
    score_take,
    write_label_rows,
)

PROTO = load_protocol("protocols/bas_boxes.yaml")
SPEC = load_spec()
FPS = 10.0
TIMEOUT_S = 60.0


def _rows(*segs):
    """(label, first frame) pairs -> T-REC label-change rows at FPS."""
    return [(f, f / FPS, lab) for lab, f in segs]


def _truth(rows, n_frames):
    clock = FrameClock(range(n_frames), [i / FPS for i in range(n_frames)])
    return derive_truth(rows, PROTO, clock=clock, n_frames=n_frames)


def _seq(outputs):
    return list(enumerate(outputs))


def _done(step_id, t_end, status=StepStatus.VERIFIED, *, t=None, retro=False):
    t = t_end + 0.8 if t is None else t
    fi = round(t * FPS)
    return StepEvent(step_id, status, t_end - 1.0, t_end, 0.9, frame_idx=fi, t=t, retroactive=retro)


def _alert(kind, step_ids, t):
    return Alert(kind, "msg", step_ids, frame_idx=round(t * FPS), t=t)


def _score(truth, outputs, meta=None):
    meta = {"take_id": "t", "operator": "op99", "scenario": "x", **(meta or {})}
    return score_take(meta, truth, _seq(outputs), PROTO, SPEC, timeout_s=TIMEOUT_S)


CORRECT = _rows(
    ("idle", 0), ("S1", 10), ("idle", 30), ("S2", 40), ("S3", 60), ("idle", 80),
    ("S4", 90), ("S5", 110), ("idle", 130), ("S6", 140), ("idle", 160),
)  # fmt: skip
SKIP_S2 = _rows(
    ("idle", 0), ("S1", 10), ("idle", 30), ("S4", 40), ("S5", 60), ("S6", 80), ("idle", 100),
)  # fmt: skip
S4_BEFORE_S2 = _rows(
    ("idle", 0), ("S1", 10), ("S4", 30), ("S5", 50), ("S2", 70), ("S3", 90), ("S6", 110),
    ("idle", 130),
)  # fmt: skip


# --------------------------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------------------------


def test_truth_of_a_correct_take():
    tr = _truth(CORRECT, 180)
    assert [o.step_id for o in tr.occurrences] == ["S1", "S2", "S3", "S4", "S5", "S6"]
    assert tr.never_performed == () and tr.incidents == ()
    assert tr.is_correct(TIMEOUT_S)
    s3 = tr.occurrences[2]
    assert (s3.f_start, s3.f_end, s3.t_end) == (60, 80, pytest.approx(8.0))
    assert tr.duration_s == pytest.approx(17.9)


def test_truth_tells_a_skip_from_an_out_of_order_step():
    skip = _truth(SKIP_S2, 120)
    assert skip.never_performed == ("S2", "S3")
    (inc,) = skip.incidents
    assert (inc.kind, inc.trigger, inc.missing) == ("skip", "S4", ("S2", "S3"))
    assert inc.t == pytest.approx(6.0) and not skip.is_correct(TIMEOUT_S)

    ooo = _truth(S4_BEFORE_S2, 140)
    assert ooo.never_performed == ()
    (inc,) = ooo.incidents  # S5 jumps the same steps: no second incident
    assert (inc.kind, inc.trigger, inc.missing) == ("out_of_order", "S4", ("S2", "S3"))


def test_truth_merges_fumbles_folds_unknown_labels_and_ignores_rows_past_the_end():
    rows = _rows(
        ("idle", 0), ("S1", 10), ("idle", 20), ("S1", 25), ("idle", 30), ("S2", 40),
        ("other", 50), ("S2", 60), ("S7", 70), ("S3", 80), ("idle", 90), ("S4", 500),
    )  # fmt: skip
    tr = _truth(rows, 100)
    assert [(o.step_id, o.f_start, o.f_end) for o in tr.occurrences] == [
        ("S1", 10, 30),  # idle between: one fumbled performance
        ("S2", 40, 50),
        ("S2", 60, 70),  # a non-step action between: a real repeat
        ("S3", 80, 90),
    ]
    assert tr.other_segments == 2  # "other" and the out-of-protocol "S7"
    assert "S4" in tr.never_performed  # its row lies past the last frame


def test_a_long_pause_makes_a_take_not_correct():
    rows = _rows(("idle", 0), ("S1", 10), ("S2", 20), ("S3", 30), ("idle", 40), ("S4", 700),
                 ("S5", 710), ("S6", 720), ("idle", 730))  # fmt: skip
    tr = _truth(rows, 740)
    assert tr.max_gap_s == pytest.approx(67.0) and not tr.is_correct(TIMEOUT_S)


def test_label_rows_round_trip_and_default_to_idle(tmp_path):
    path = tmp_path / "a.csv"
    write_label_rows(path, [(12, 1.2, "S1"), (30, 3.0, "idle")])
    assert read_label_rows(path) == [(0, 0.0, "idle"), (12, 1.2, "S1"), (30, 3.0, "idle")]


def test_frame_clock_is_exact_where_it_was_told_and_linear_elsewhere():
    c = FrameClock([0, 10, 20], [0.0, 1.0, 2.5])
    assert c.t_at(10) == 1.0 and c.t_at(15) == pytest.approx(1.75)
    assert c.t_at(30) == pytest.approx(2.5 + 10 * 0.125)
    assert FrameClock.from_points(c.points([0, 20])).t_at(20) == 2.5


# --------------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------------


def test_completions_match_one_to_one_within_the_tolerance():
    tr = _truth(CORRECT, 180)  # S1 ends at 3.0, S3 at 8.0
    outs = [
        _done("S1", 2.4),  # within ±2 s, but the next claim is closer
        _done("S1", 2.9),  # a duplicate claim: only one of the two can match, and this one does
        _done("S3", 10.5),  # 2.5 s late: outside ±2 s
        _done("S2", 5.0, retro=True),  # retroactive events are not completion claims
    ]
    s = _score(tr, outs)
    assert (s.gt_steps, s.tp, s.fp, s.fn) == (6, 1, 2, 5)
    assert s.latencies == [pytest.approx(2.9 + 0.8 - 3.0)]  # the closer claim won


def test_an_operator_confirmation_matches_a_step_done_before_the_ask():
    tr = _truth(CORRECT, 180)  # S3 ends at 8.0
    late = _done("S3", 11.0, StepStatus.OPERATOR_CONFIRMED, t=12.0)  # asked 3 s after
    early = _done("S5", 9.0, StepStatus.OPERATOR_CONFIRMED, t=10.0)  # S5 ends at 13.0
    s = _score(tr, [late, early])
    assert (s.tp, s.fp) == (1, 1)
    assert s.latencies == []  # confirmations carry the Ask's timing, not the detection's


def test_a_live_out_of_order_alert_before_a_skip_is_credited_to_the_skip():
    tr = _truth(SKIP_S2, 120)  # S4 ends at 6.0
    outs = [
        _alert(AlertKind.OUT_OF_ORDER, ("S4", "S2", "S3"), 6.5),
        _alert(AlertKind.SKIP, ("S2", "S3"), 10.0),
    ]
    s = _score(tr, outs)
    assert (s.skip_gt, s.skip_tp, s.skip_fp, s.skip_fn) == (2, 2, 0, 0)
    assert (s.ooo_gt, s.ooo_tp, s.ooo_fp, s.ooo_fn, s.ooo_skip_precursors) == (0, 0, 0, 0, 1)


def test_out_of_order_alerts_are_matched_per_incident_and_in_time():
    tr = _truth(S4_BEFORE_S2, 140)  # S4 ends at 5.0
    hit = _score(tr, [_alert(AlertKind.OUT_OF_ORDER, ("S4", "S2", "S3"), 5.9)])
    assert (hit.ooo_gt, hit.ooo_tp, hit.ooo_fp, hit.ooo_fn) == (1, 1, 0, 0)
    late = _score(tr, [_alert(AlertKind.OUT_OF_ORDER, ("S4", "S2", "S3"), 11.0)])
    assert (late.ooo_tp, late.ooo_fp, late.ooo_fn) == (0, 1, 1)
    wrong_skip = _score(tr, [_alert(AlertKind.SKIP, ("S2",), 12.0)])
    assert (wrong_skip.skip_gt, wrong_skip.skip_fp) == (0, 1)


def test_every_alert_on_a_correct_take_is_false():
    tr = _truth(CORRECT, 180)
    outs = [
        _alert(AlertKind.OUT_OF_ORDER, ("S4", "S2"), 9.0),
        _alert(AlertKind.TOO_FAST, ("S3",), 8.0),
    ]
    s = _score(tr, outs)
    assert s.correct_take and s.false_alerts == 2 and s.ooo_fp == 1
    m = aggregate([s])["false_alerts_per_min_correct_takes"]
    assert m.value == pytest.approx(2 / (17.9 / 60))


def test_ask_then_correct_compares_the_resolution_with_the_annotations():
    tr = _truth(SKIP_S2, 120)  # S2 never performed; S5 ends at 8.0
    ask_s5 = Ask("q1-S5", "S5", "Did you?", frame_idx=90, t=9.0)
    ask_s2 = Ask("q2-S2", "S2", "Did you?", frame_idx=95, t=9.5)
    outs = [
        ask_s5,
        Answer("q1-S5", "yes", "voice", t=10.0, frame_idx=100),
        _done("S5", 9.0, StepStatus.OPERATOR_CONFIRMED, t=10.0),  # right: S5 was done
        ask_s2,
        _done("S2", 9.0, StepStatus.SKIPPED, t=11.0, retro=True),  # right: S2 never was
        Ask("q1-S5", "S5", "Did you?", frame_idx=120, t=12.0),  # a `repeat`: same question
    ]
    s = _score(tr, outs)
    assert (s.asks, s.asks_correct) == (2, 2)
    assert s.answers == {"yes": 1}

    unresolved = _score(_truth(CORRECT, 180), [Ask("q1-S3", "S3", "?", frame_idx=90, t=9.0)])
    assert (unresolved.asks, unresolved.asks_correct) == (1, 0)  # S3 was done, never ticked


def test_no_procedure_clip_counts_only_false_completions():
    tr = _truth(_rows(("idle", 0), ("other", 20), ("idle", 60)), 100)
    s = _score(tr, [_done("S1", 4.0), _alert(AlertKind.TIMEOUT, ("S1",), 6.0)])
    assert not tr.has_procedure and not s.correct_take
    assert (s.gt_steps, s.tp, s.fp, s.skip_gt, s.false_alerts) == (0, 0, 1, 0, 0)


def test_metrics_with_no_cases_are_none_not_zero():
    s = _score(_truth(CORRECT, 180), [])
    m = aggregate([s])
    assert m["skip_alert_precision"].value is None  # nothing alerted, nothing to be precise about
    assert m["skip_alert_recall"].value is None  # nothing was skipped
    assert m["step_completion_recall"].value == 0.0
    assert m["step_completion_recall"].show() == "0.000 (0/6)"


# --------------------------------------------------------------------------------------------
# Spec, log, report
# --------------------------------------------------------------------------------------------


def test_the_spec_is_pre_registered_and_complete():
    assert SPEC.version >= 1 and SPEC.frozen_on
    assert SPEC.completion_tol_s == 2.0  # the build plan's ±2 s
    assert {c.metric for c in SPEC.criteria} == set(METRICS) == set(DEFINITIONS)
    assert len(SPEC.sha256) == 64
    c = SPEC.criteria[0]
    assert c.judge(None) == "n/a"


def test_a_tampered_log_is_refused(tmp_path):
    log = EventLog(tmp_path, catch_signals=False)
    log.append(_done("S1", 2.0))
    log.append(_alert(AlertKind.SKIP, ("S2",), 3.0))
    log.close()
    recs, run_end = read_events(log.path)
    assert [type(o).__name__ for _, o in recs] == ["StepEvent", "Alert"]
    assert [q for q, _ in recs] == [0, 1] and run_end["type"] == "run_end"
    assert recs[0][1].status is StepStatus.VERIFIED  # decoded back into the contract

    text = log.path.read_text(encoding="utf-8").replace('"S2"', '"S3"', 1)
    log.path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="fails verification"):
        read_events(log.path)


def _assert_ui_shape(report):
    """The Validation tab renders top-level scalars as cards and every top-level array as a
    table whose columns come from the first row."""
    for key, value in report.items():
        if key == "detail":
            assert isinstance(value, dict)
        elif isinstance(value, list):
            assert value, key
            cols = list(value[0])
            for row in value:
                assert list(row) == cols, key
                assert all(v is None or isinstance(v, (str, int, float)) for v in row.values())
        else:
            assert value is None or isinstance(value, (str, int, float, bool)), key


def test_the_report_is_flat_where_the_ui_reads_it():
    ends = [3, 6, 8, 11, 13, 16]  # CORRECT's annotated ends, claimed exactly
    claims = [_done(s, t) for s, t in zip(PROTO.step_ids, ends, strict=True)]
    scores = [
        _score(_truth(CORRECT, 180), claims),
        _score(_truth(SKIP_S2, 120), [_alert(AlertKind.SKIP, ("S2", "S3"), 10.0)]),
    ]
    report = build_report(scores, PROTO, SPEC, {"eval_id": "x", "answer_policy": "oracle"})
    _assert_ui_shape(report)
    assert report["data"].startswith("REAL") and report["simulated"] is False
    assert report["step_completion_recall"] == pytest.approx(6 / 10)
    assert [r["metric"] for r in report["definitions"]] == list(METRICS)
    json.dumps(report)  # serialisable as-is


# --------------------------------------------------------------------------------------------
# End to end on the SIMULATED replay
# --------------------------------------------------------------------------------------------


def _replay_eval(tmp_path, scenarios, **kw):
    from actus.core.config import load_config

    cfg = load_config()
    inputs = replay_takes(scenarios, [0], PROTO, cfg)
    report = run_evaluation(
        inputs,
        tmp_path / "ev",
        spec=SPEC,
        overrides=["models.enabled=false", "pipeline.overlay=false"],
        evidence="replay",
        stages={"models": None},
        **kw,
    )
    return report, tmp_path / "ev"


def test_replay_evaluation_end_to_end(tmp_path):
    report, ev = _replay_eval(tmp_path, ["correct", "skip_S2", "S4_before_S2"])
    _assert_ui_shape(report)
    assert report["simulated"] is True and report["data"].startswith("SIMULATED")
    assert "SIMULATED" in (ev / "report.md").read_text(encoding="utf-8")

    assert report["step_completion_precision"] == 1.0
    assert report["step_completion_recall"] == 1.0
    assert report["skip_alert_precision"] == report["skip_alert_recall"] == 1.0
    assert report["out_of_order_alert_precision"] == report["out_of_order_alert_recall"] == 1.0
    assert report["false_alerts_per_min_correct_takes"] == 0.0
    assert abs(report["median_detection_latency_s"]) <= 2.0

    rows = {r["take"]: r for r in report["takes"]}
    assert rows["replay-correct-s0"]["take_type"] == "correct"
    assert rows["replay-skip_S2-s0"]["outcome"].split()[1:3] == ["S2s", "S3s"]

    # every number traces to a take: the per-take files are all there, and rescoring from them
    # alone reproduces the report
    for tid in rows:
        take_dir = ev / "takes" / tid
        for name in ("events.jsonl", "annotations.csv", "clock.json", "take.json", "score.json"):
            assert (take_dir / name).is_file(), (tid, name)
    again = rescore(ev, spec=SPEC)
    for m in METRICS:
        assert again[m] == report[m], m
    assert again["takes"] == report["takes"]


def test_replay_ask_path_with_the_oracle_operator(tmp_path):
    report, ev = _replay_eval(tmp_path, ["occluded"], answer_policy="oracle")
    (row,) = report["takes"]
    assert row["outcome"].split()[2] == "S3c"  # operator_confirmed after the blindness ask
    assert report["ask_then_correct_rate"] == 1.0
    assert report["step_completion_recall"] == 1.0
    score = json.loads((ev / "takes" / row["take"] / "score.json").read_text(encoding="utf-8"))
    assert score["asks"][0]["answer"] == "yes" and score["asks"][0]["correct"] is True


def test_an_existing_evaluation_is_not_overwritten(tmp_path):
    (tmp_path / "ev").mkdir()
    (tmp_path / "ev" / "eval.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _replay_eval(tmp_path, ["correct"])
