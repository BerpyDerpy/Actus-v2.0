"""T-EVAL: replay held-out takes through the offline pipeline and score them against the human
step annotations. Writes the report card the GUI's Validation tab and the pitch both use.

    uv run python tools/evaluate.py                                   # the held-out split
    uv run python tools/evaluate.py --operators op05                  # explicit operators
    uv run python tools/evaluate.py --takes op05_correct_01,op05_skip_S2_01
    uv run python tools/evaluate.py --replay all --seeds 0,1,2 --evidence replay   # SIMULATED
    uv run python tools/evaluate.py --rescore runs/eval/heldout-test-20261001-120000

How it works
------------
1. **Select takes.** By default, every labelled take whose operator is in the ``test`` split of
   ``data/splits/report.json`` (``tools/make_splits.py``), minus the scenarios the spec excludes
   (Task B follows a different protocol). Unlabelled takes and takes with no label CSV are listed
   as excluded, with the reason, rather than silently dropped.
2. **Run** each take through ``actus.app.pipeline.run_offline``: the same core as the live system,
   deterministic, no drops. Whatever the config plugs in (detector, hands/pose, HOI, temporal
   model) is what gets evaluated, so a better model is evaluated by changing the config, not
   this file. Asks are answered by a *simulated* operator (``oracle`` by default: it answers
   truthfully from the annotations), and the report says so.
3. **Score from the artefacts.** Each take is scored from its hash-chained ``events.jsonl``
   (verified first; a tampered log is refused) and a copy of its annotation CSV, both kept in
   ``<eval_dir>/takes/<take>/`` with ``clock.json`` (the pipeline ``t`` of every annotated frame)
   and ``score.json`` (every match, with the log ``seq`` of each event it used). So every number
   in the report traces to a take, and ``--rescore`` recomputes the report from those files
   alone without re-running anything.
4. **Judge** the headline metrics against the pre-registered targets in
   ``runs/eval/criteria.yaml`` (written before any real number existed; its SHA-256 goes into
   every report) and write ``report.json`` (for the UI) and ``report.md`` (for the pitch) into
   the evaluation directory. A real evaluation is also published to ``runs/eval/report.json`` /
   ``report.md``, which is what the Validation tab shows; a SIMULATED one only with ``--publish``.

Ground truth comes from the annotations alone, never from the scenario name: a take's annotated
step sequence says which steps were performed, in what order, which were never performed, and
whether the take counts as "correct". See ``DEFINITIONS`` for every metric.

Clocks: annotations are keyed by ``frame_idx``. Each annotated frame is mapped onto the
pipeline's own ``t`` for that frame (recorded during the run), so annotations and engine
outputs are compared on one clock -- never ``frame_idx / fps``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import shutil
import statistics
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from actus.core.config import Config, load_config, resolve_path  # noqa: E402
from actus.core.contracts import (  # noqa: E402
    IDLE,
    OTHER,
    Alert,
    AlertKind,
    Answer,
    AnswerValue,
    Ask,
    StepEvent,
    StepStatus,
    from_jsonable,
)
from actus.io.eventlog import verify_log  # noqa: E402
from actus.protocol.decision import DecisionParams  # noqa: E402
from actus.protocol.engine import DONE  # noqa: E402
from actus.protocol.schema import Protocol, load_protocol  # noqa: E402

SPEC_PATH = "runs/eval/criteria.yaml"
EVAL_ROOT = "runs/eval"
ANSWER_POLICIES = ("oracle", "yes", "no", "none")
LOG_META_KEYS = frozenset({"schema_version", "seq", "ts", "prev_hash", "hash"})
STATUS_CODE = {
    "verified": "v",
    "operator_confirmed": "c",
    "out_of_order": "o",
    "skipped": "s",
    "timed_out": "t",
    "pending": "p",
}
EPS = 1e-9

# The headline metrics, in report order, and what each one means. The definitions are copied
# into every report.md and report.json, so a number is never shown without its definition.
DEFINITIONS: dict[str, str] = {
    "step_completion_precision": (
        "Of the completions the engine claimed (non-retroactive StepEvents with status "
        "verified, out_of_order or operator_confirmed), the fraction that match an annotated "
        "performance of the same step. An evidence-based completion matches if its t_end is "
        "within ±completion_tol_s of the annotated end. An operator-confirmed completion's "
        "t_end is the time of the Ask, so it matches a performance that ended no later than "
        "completion_tol_s after it. Matching is one-to-one, closest first. Completions on clips "
        "with no procedure are all false."
    ),
    "step_completion_recall": (
        "Of the annotated step performances, the fraction matched by a claimed completion as "
        "above. A performance is one label segment of a protocol step; segments of the same "
        "step separated only by idle are merged into one (a fumble, not a repeat)."
    ),
    "skip_alert_precision": (
        "Counted per step. Of the steps named by skip alerts, the fraction the annotations show "
        "were never performed in that take."
    ),
    "skip_alert_recall": (
        "Of the protocol steps never performed in a take that performed at least one step, the "
        "fraction named by a skip alert."
    ),
    "out_of_order_alert_precision": (
        "Of the out_of_order alerts, the fraction that match an annotated out-of-order "
        "incident: a step performed while a step it requires was not yet performed, where that "
        "required step was performed later. An alert matches if it names one of the incident's "
        "missing steps and fires within ±alert_tol_s of the annotated end of the step that "
        "jumped ahead. An alert that matches an incident whose missing steps were never "
        "performed is the live precursor of a skip (at that moment a skip and an out-of-order "
        "step look the same); it is credited through the skip alert and counts in neither the "
        "numerator nor the denominator."
    ),
    "out_of_order_alert_recall": (
        "Of the annotated out-of-order incidents (as above), the fraction matched by an "
        "out_of_order alert."
    ),
    "false_alerts_per_min_correct_takes": (
        "Alerts of every kind on correct takes, divided by those takes' total duration in "
        "minutes. A take is correct from its annotations alone: every protocol step performed "
        "exactly once, in an order the protocol allows, no non-step ('other') action, and no "
        "stretch without progress as long as the protocol timeout."
    ),
    "median_detection_latency_s": (
        "Median, over matched evidence-based completions, of the engine's decision time "
        "(StepEvent.t, which includes the dwell) minus the annotated end of the step. Negative "
        "when the engine decides before the annotator marked the end, which is when the hand "
        "withdrew. Operator-confirmed completions are left out: their timing is the Ask's."
    ),
    "ask_rate_per_min": (
        "Distinct Asks (by question_id) divided by the total duration of every evaluated take, "
        "in minutes."
    ),
    "ask_then_correct_rate": (
        "Of the distinct Asks, the fraction whose step ended with the right outcome: the first "
        "completion or skip of the asked step after the Ask agrees with whether the annotations "
        "show the step performed by then (within completion_tol_s). An Ask still unresolved at "
        "the end is right only if the step was never performed. The answers come from the "
        "simulated operator named in the report."
    ),
}
METRICS: tuple[str, ...] = tuple(DEFINITIONS)

TERMS: dict[str, str] = {
    "annotated end": (
        "The frame at which the next label begins in the take's annotation CSV (the recording "
        "protocol's step-boundary rule: `0` is pressed once the end state is reached and the "
        "hand withdraws), on the pipeline clock."
    ),
    "pipeline clock": (
        "The `t` the pipeline gave each frame: the container timestamp for a take, synthetic "
        "seconds for the replay. Annotation frame indices are mapped onto it frame by frame."
    ),
    "oracle operator": (
        "The simulated operator used to answer Asks: 'yes' if the annotations show the asked step "
        "performed by the time it answers (answer_delay_s after the Ask, plus completion_tol_s), "
        "else 'no'. It is truthful by construction, so ask-then-correct measures what the engine "
        "does with a truthful answer, not how real operators answer."
    ),
}


# --------------------------------------------------------------------------------------------
# The pre-registered spec (runs/eval/criteria.yaml)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Criterion:
    metric: str
    op: str  # ">=" | "<="
    target: float
    why: str = ""

    def judge(self, value: float | None) -> str:
        if value is None:
            return "n/a"
        ok = value >= self.target - EPS if self.op == ">=" else value <= self.target + EPS
        return "PASS" if ok else "FAIL"


@dataclass(frozen=True)
class EvalSpec:
    version: int
    frozen_on: str
    completion_tol_s: float
    alert_tol_s: float
    answer_delay_s: float
    answer_policy: str
    split: str
    exclude_scenarios: tuple[str, ...]
    criteria: tuple[Criterion, ...]
    path: str
    sha256: str


def load_spec(path: str | Path = SPEC_PATH) -> EvalSpec:
    p = resolve_path(path)
    raw = p.read_bytes()
    data = yaml.safe_load(raw) or {}
    params = data.get("parameters") or {}
    takes = data.get("takes") or {}
    criteria = []
    for c in data.get("criteria") or []:
        metric, op = str(c["metric"]), str(c["op"])
        if metric not in DEFINITIONS:
            raise ValueError(f"{p}: criterion on unknown metric {metric!r}; know {METRICS}")
        if op not in (">=", "<="):
            raise ValueError(f"{p}: criterion {metric}: op must be '>=' or '<=', got {op!r}")
        criteria.append(Criterion(metric, op, float(c["target"]), str(c.get("why", ""))))
    policy = str(params.get("answer_policy", "oracle"))
    if policy not in ANSWER_POLICIES:
        raise ValueError(f"{p}: answer_policy must be one of {ANSWER_POLICIES}, got {policy!r}")
    spec = EvalSpec(
        version=int(data.get("version", 0)),
        frozen_on=str(data.get("frozen_on", "")),
        completion_tol_s=float(params.get("completion_tol_s", 2.0)),
        alert_tol_s=float(params.get("alert_tol_s", 5.0)),
        answer_delay_s=float(params.get("answer_delay_s", 1.0)),
        answer_policy=policy,
        split=str(takes.get("split", "test")),
        exclude_scenarios=tuple(str(s) for s in takes.get("exclude_scenarios") or ()),
        criteria=tuple(criteria),
        path=_rel(p),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    if spec.completion_tol_s <= 0 or spec.alert_tol_s <= 0 or spec.answer_delay_s < 0:
        raise ValueError(f"{p}: tolerances must be > 0 and answer_delay_s >= 0")
    return spec


# --------------------------------------------------------------------------------------------
# Ground truth from annotations
# --------------------------------------------------------------------------------------------

LabelRow = tuple[int, float, str]  # (frame_idx, t, label): a label change, T-REC convention


def read_label_rows(path: str | Path) -> list[LabelRow]:
    """Label changes from a T-REC sidecar CSV, sorted by frame. Like ``LabelTrack``, the label
    before the first row is ``idle``."""
    rows: dict[int, tuple[float, str]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows[int(r["frame_idx"])] = (float(r["t"]), str(r["label"]).strip())
    if 0 not in rows:
        rows[0] = (0.0, IDLE)
    return [(fi, t, lab) for fi, (t, lab) in sorted(rows.items())]


def write_label_rows(path: str | Path, rows: Iterable[LabelRow]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "t", "label"])
        for fi, t, lab in rows:
            w.writerow([int(fi), f"{float(t):.4f}", lab])


class FrameClock:
    """``frame_idx`` -> pipeline ``t``. Exact for the frames it was given, linear between them
    and beyond them (only ever needed for the annotation-derived clock the oracle uses)."""

    def __init__(self, frames: Iterable[int], times: Iterable[float]) -> None:
        pairs = sorted(dict(zip((int(f) for f in frames), times, strict=True)).items())
        if not pairs:
            raise ValueError("a FrameClock needs at least one (frame_idx, t) point")
        self._exact = {f: float(t) for f, t in pairs}
        self.frames = np.array([f for f, _ in pairs], dtype=float)
        self.times = np.array([t for _, t in pairs], dtype=float)

    @classmethod
    def from_rows(cls, rows: Sequence[LabelRow]) -> FrameClock:
        return cls((r[0] for r in rows), (r[1] for r in rows))

    @classmethod
    def from_points(cls, points: Sequence[Sequence[float]]) -> FrameClock:
        return cls((int(p[0]) for p in points), (float(p[1]) for p in points))

    def t_at(self, frame: int) -> float:
        hit = self._exact.get(int(frame))
        if hit is not None:
            return hit
        f, t = self.frames, self.times
        if len(f) == 1:
            return float(t[0])
        if f[0] <= frame <= f[-1]:
            return float(np.interp(frame, f, t))
        rate = (t[-1] - t[0]) / (f[-1] - f[0])
        ref = 0 if frame < f[0] else -1
        return float(t[ref] + (frame - f[ref]) * rate)

    def points(self, frames: Iterable[int]) -> list[list[float]]:
        return [[int(fr), self.t_at(fr)] for fr in sorted({int(x) for x in frames})]


@dataclass(frozen=True)
class Occurrence:
    """One annotated performance of a step. ``f_end`` is exclusive (the frame where the next
    label begins); None means the step ran to the end of the take and the take's length was not
    known yet (only in the pre-run truth the oracle uses)."""

    step_id: str
    f_start: int
    f_end: int | None
    t_start: float
    t_end: float  # the annotated end, pipeline clock (inf if f_end is None)


@dataclass(frozen=True)
class Incident:
    """A step performed while a step it requires had not been. ``kind`` is ``out_of_order`` if
    any missing step was performed later, else ``skip`` (they were all never performed)."""

    kind: str
    trigger: str
    missing: tuple[str, ...]
    t: float  # annotated end of the trigger step


@dataclass(frozen=True)
class GroundTruth:
    occurrences: tuple[Occurrence, ...]
    performed: tuple[str, ...]  # first performances, in completion order
    never_performed: tuple[str, ...]  # protocol order; empty for a clip with no procedure
    incidents: tuple[Incident, ...]
    other_segments: int
    t_first: float
    t_last: float
    max_gap_s: float  # longest stretch with no completion while the procedure was unfinished

    @property
    def duration_s(self) -> float:
        return max(0.0, self.t_last - self.t_first)

    @property
    def has_procedure(self) -> bool:
        return bool(self.occurrences)

    def is_correct(self, timeout_s: float) -> bool:
        return (
            self.has_procedure
            and not self.never_performed
            and not self.incidents
            and self.other_segments == 0
            and len(self.occurrences) == len(self.performed)
            and self.max_gap_s < timeout_s
        )

    def performed_by(self, step_id: str, t: float) -> bool:
        return any(o.step_id == step_id and o.t_end <= t + EPS for o in self.occurrences)

    def to_json(self) -> dict[str, Any]:
        return {
            "occurrences": [
                {
                    "step_id": o.step_id,
                    "f_start": o.f_start,
                    "f_end": o.f_end,
                    "t_start": _r(o.t_start),
                    "t_end": _r(o.t_end),
                }
                for o in self.occurrences
            ],
            "performed": list(self.performed),
            "never_performed": list(self.never_performed),
            "incidents": [
                {"kind": i.kind, "trigger": i.trigger, "missing": list(i.missing), "t": _r(i.t)}
                for i in self.incidents
            ],
            "other_segments": self.other_segments,
            "t_first": _r(self.t_first),
            "t_last": _r(self.t_last),
            "max_gap_s": _r(self.max_gap_s),
        }


def _segments(
    rows: Sequence[LabelRow], step_ids: set[str], n_frames: int | None
) -> list[tuple[str, int, int | None]]:
    """(label, f_start, f_end) runs. Labels outside the protocol (a stray ``S7``) fold into
    ``other``; rows at or past the end of the take are dropped."""
    norm: list[tuple[str, int]] = []
    for fi, _t, lab in rows:
        if n_frames is not None and fi >= n_frames:
            break
        lab = lab if lab in step_ids or lab in (IDLE, OTHER) else OTHER
        if norm and norm[-1][0] == lab:
            continue
        norm.append((lab, fi))
    return [
        (lab, f0, norm[i + 1][1] if i + 1 < len(norm) else n_frames)
        for i, (lab, f0) in enumerate(norm)
    ]


def derive_truth(
    rows: Sequence[LabelRow],
    protocol: Protocol,
    *,
    clock: FrameClock,
    n_frames: int | None,
) -> GroundTruth:
    """What the annotations say happened, on ``clock``'s time axis."""
    step_ids = set(protocol.step_ids)

    def end_t(f1: int | None) -> float:
        if f1 is None:
            return math.inf
        assert n_frames is None or f1 <= n_frames
        return clock.t_at(n_frames - 1 if n_frames is not None and f1 >= n_frames else f1)

    occs: list[Occurrence] = []
    others = 0
    last_real: str | None = None  # the last non-idle label seen
    for lab, f0, f1 in _segments(rows, step_ids, n_frames):
        if lab == IDLE:
            continue
        if lab == OTHER:
            others += 1
            last_real = OTHER
            continue
        if last_real == lab and occs and occs[-1].step_id == lab:  # only idle between: merge
            prev = occs[-1]
            occs[-1] = Occurrence(lab, prev.f_start, f1, prev.t_start, end_t(f1))
        else:
            occs.append(Occurrence(lab, f0, f1, clock.t_at(f0), end_t(f1)))
        last_real = lab

    done: set[str] = set()
    flagged: set[str] = set()
    raw: list[tuple[str, tuple[str, ...], float]] = []
    performed: list[str] = []
    for o in sorted(occs, key=lambda o: (o.t_end, o.f_start)):
        if o.step_id in done:
            continue
        need = protocol.required_before(o.step_id)
        missing = tuple(s for s in protocol.step_ids if s in need and s not in done | flagged)
        if missing:
            raw.append((o.step_id, missing, o.t_end))
            flagged.update(missing)
        done.add(o.step_id)
        flagged.discard(o.step_id)
        performed.append(o.step_id)
    never = tuple(s for s in protocol.step_ids if s not in done) if occs else ()
    incidents = tuple(
        Incident("skip" if set(m) <= set(never) else "out_of_order", trig, m, t)
        for trig, m, t in raw
    )

    t_first = clock.t_at(0)
    t_last = clock.t_at(n_frames - 1) if n_frames else clock.t_at(rows[-1][0] if rows else 0)
    marks = [t_first, *sorted(o.t_end for o in occs if math.isfinite(o.t_end))]
    gaps = [b - a for a, b in itertools.pairwise(marks)]
    if never or not occs:
        gaps.append(t_last - marks[-1])
    return GroundTruth(
        occurrences=tuple(occs),
        performed=tuple(performed),
        never_performed=never,
        incidents=incidents,
        other_segments=others,
        t_first=t_first,
        t_last=t_last,
        max_gap_s=max(gaps, default=0.0),
    )


# --------------------------------------------------------------------------------------------
# Engine outputs, read back from the hash-chained log
# --------------------------------------------------------------------------------------------

Rec = tuple[int, Any]  # (log seq, decoded record)


def read_events(path: str | Path) -> tuple[list[Rec], dict[str, Any] | None]:
    """Every logged output with its ``seq``, and the ``run_end`` record. The chain is verified
    first: a log that fails verification is refused, never scored."""
    path = Path(path)
    check = verify_log(path)
    if not check.ok:
        raise ValueError(f"{path}: event log fails verification ({check.error}); not scoring it")
    recs: list[Rec] = []
    run_end: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("type") == "run_end":
            run_end = rec
            continue
        body = {k: v for k, v in rec.items() if k not in LOG_META_KEYS}
        recs.append((int(rec["seq"]), from_jsonable(body)))
    return recs, run_end


# --------------------------------------------------------------------------------------------
# Scoring one take
# --------------------------------------------------------------------------------------------


def _greedy(cands: list[tuple[float, int, int]]) -> list[tuple[int, int]]:
    """One-to-one matching, closest first; ties broken by index, so it is deterministic."""
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs = []
    for _, a, b in sorted(cands):
        if a in used_a or b in used_b:
            continue
        used_a.add(a)
        used_b.add(b)
        pairs.append((a, b))
    return pairs


def completion_offset(occ: Occurrence, ev: StepEvent, tol: float) -> float | None:
    """Signed offset (claimed end - annotated end) if ``ev`` may match ``occ``, else None."""
    if ev.step_id != occ.step_id or not math.isfinite(occ.t_end):
        return None
    end = ev.t_end if ev.t_end is not None else ev.t
    d = end - occ.t_end
    if ev.status is StepStatus.OPERATOR_CONFIRMED:
        return d if d >= -tol - EPS else None  # confirmed after (or just before) it was done
    return d if abs(d) <= tol + EPS else None


def match_completions(
    occs: Sequence[Occurrence], comps: Sequence[Rec], tol: float
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """(occurrence index, completion index, offset) pairs, unmatched occurrences, unmatched
    completions."""
    cands, offsets = [], {}
    for i, o in enumerate(occs):
        for j, (_seq, ev) in enumerate(comps):
            d = completion_offset(o, ev, tol)
            if d is not None:
                cands.append((abs(d), i, j))
                offsets[i, j] = d
    pairs = _greedy(cands)
    mi, mj = {i for i, _ in pairs}, {j for _, j in pairs}
    return (
        [(i, j, offsets[i, j]) for i, j in pairs],
        [i for i in range(len(occs)) if i not in mi],
        [j for j in range(len(comps)) if j not in mj],
    )


@dataclass
class TakeScore:
    take_id: str
    operator: str
    scenario: str
    simulated: bool
    has_procedure: bool
    correct_take: bool
    duration_s: float
    events_sha256: str
    # step completions
    gt_steps: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    latencies: list[float] = field(default_factory=list)
    per_step: dict[str, dict[str, Any]] = field(default_factory=dict)
    # alerts
    skip_gt: int = 0
    skip_tp: int = 0
    skip_fp: int = 0
    skip_fn: int = 0
    ooo_gt: int = 0
    ooo_tp: int = 0
    ooo_fp: int = 0
    ooo_fn: int = 0
    ooo_skip_precursors: int = 0
    alerts_by_kind: Counter = field(default_factory=Counter)
    false_alerts: int = 0
    # asks
    asks: int = 0
    asks_correct: int = 0
    answers: Counter = field(default_factory=Counter)
    outcome: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def alerts(self) -> int:
        return sum(self.alerts_by_kind.values())

    def row(self) -> dict[str, Any]:
        """One flat row of the take-level table (the UI renders it as-is)."""
        kind = "correct" if self.correct_take else "procedure" if self.has_procedure else "none"
        lat = statistics.median(self.latencies) if self.latencies else None
        return {
            "take": self.take_id,
            "operator": self.operator,
            "scenario": self.scenario,
            "take_type": kind,
            "minutes": round(self.duration_s / 60.0, 2),
            "gt_steps": self.gt_steps,
            "completions_tp_fp_fn": f"{self.tp}/{self.fp}/{self.fn}",
            "skips_gt_tp_fp_fn": f"{self.skip_gt}/{self.skip_tp}/{self.skip_fp}/{self.skip_fn}",
            "ooo_gt_tp_fp_fn": f"{self.ooo_gt}/{self.ooo_tp}/{self.ooo_fp}/{self.ooo_fn}",
            "alerts": self.alerts,
            "false_alerts": self.false_alerts if self.correct_take else "-",
            "asks_correct": f"{self.asks_correct}/{self.asks}",
            "median_latency_s": "-" if lat is None else round(lat, 2),
            "outcome": self.outcome,
            "events_sha256": self.events_sha256[:12],
        }


def score_take(
    meta: Mapping[str, Any],
    truth: GroundTruth,
    recs: Sequence[Rec],
    protocol: Protocol,
    spec: EvalSpec,
    *,
    timeout_s: float,
    events_sha256: str = "",
) -> TakeScore:
    tol = spec.completion_tol_s
    step_events = [(q, o) for q, o in recs if isinstance(o, StepEvent)]
    alerts = [(q, o) for q, o in recs if isinstance(o, Alert)]
    comps = [(q, o) for q, o in step_events if not o.retroactive and o.status in DONE]
    occs = truth.occurrences

    s = TakeScore(
        take_id=str(meta["take_id"]),
        operator=str(meta.get("operator", "")),
        scenario=str(meta.get("scenario", "")),
        simulated=bool(meta.get("simulated", False)),
        has_procedure=truth.has_procedure,
        correct_take=truth.is_correct(timeout_s),
        duration_s=truth.duration_s,
        events_sha256=events_sha256,
    )

    # -- step completions --------------------------------------------------------------------------
    pairs, missed, false = match_completions(occs, comps, tol)
    s.gt_steps, s.tp, s.fp, s.fn = len(occs), len(pairs), len(false), len(missed)
    per_step: dict[str, dict[str, Any]] = {
        sid: {"gt": 0, "tp": 0, "fp": 0, "fn": 0, "latencies": []} for sid in protocol.step_ids
    }
    for o in occs:
        per_step[o.step_id]["gt"] += 1
    matches = []
    for i, j, d in pairs:
        seq, ev = comps[j]
        per_step[ev.step_id]["tp"] += 1
        latency = None
        if ev.status is not StepStatus.OPERATOR_CONFIRMED:
            latency = ev.t - occs[i].t_end
            s.latencies.append(latency)
            per_step[ev.step_id]["latencies"].append(latency)
        matches.append(
            {
                "step_id": ev.step_id,
                "seq": seq,
                "status": ev.status.value,
                "annotated_end": _r(occs[i].t_end),
                "claimed_end": _r(ev.t_end),
                "decided_at": _r(ev.t),
                "offset_s": _r(d),
                "latency_s": _r(latency),
            }
        )
    for i in missed:
        per_step[occs[i].step_id]["fn"] += 1
    for j in false:
        per_step[comps[j][1].step_id]["fp"] += 1
    s.per_step = per_step

    # -- skip alerts (per step) --------------------------------------------------------------------
    named: list[str] = []
    for _, a in alerts:
        if a.kind is AlertKind.SKIP:
            named += [x for x in a.step_ids if x not in named]
    gt_skip = truth.never_performed
    s.skip_gt = len(gt_skip)
    s.skip_tp = sum(1 for x in named if x in gt_skip)
    s.skip_fp = sum(1 for x in named if x not in gt_skip)
    s.skip_fn = sum(1 for x in gt_skip if x not in named)

    # -- out-of-order alerts (per incident) --------------------------------------------------------
    ooo = [(q, a) for q, a in alerts if a.kind is AlertKind.OUT_OF_ORDER]
    cands = []
    for i, inc in enumerate(truth.incidents):
        for j, (_, a) in enumerate(ooo):
            if set(a.step_ids[1:]) & set(inc.missing) and abs(a.t - inc.t) <= spec.alert_tol_s:
                cands.append((abs(a.t - inc.t), i, j))
    ooo_pairs = _greedy(cands)
    matched_inc = {i for i, _ in ooo_pairs}
    matched_alert = {j for _, j in ooo_pairs}
    s.ooo_gt = sum(1 for inc in truth.incidents if inc.kind == "out_of_order")
    s.ooo_tp = sum(1 for i, _ in ooo_pairs if truth.incidents[i].kind == "out_of_order")
    s.ooo_skip_precursors = sum(1 for i, _ in ooo_pairs if truth.incidents[i].kind == "skip")
    s.ooo_fp = len(ooo) - len(matched_alert)
    s.ooo_fn = sum(
        1
        for i, inc in enumerate(truth.incidents)
        if inc.kind == "out_of_order" and i not in matched_inc
    )

    # -- every alert, and false alerts on correct takes --------------------------------------------
    s.alerts_by_kind = Counter(a.kind.value for _, a in alerts)
    s.false_alerts = s.alerts if s.correct_take else 0

    # -- asks --------------------------------------------------------------------------------------
    ask_rows = score_asks(truth, recs, tol)
    s.asks = len(ask_rows)
    s.asks_correct = sum(1 for r in ask_rows if r["correct"])
    s.answers = Counter(str(a.value.value) for _, a in recs if isinstance(a, Answer))

    # -- final outcome table -----------------------------------------------------------------------
    final: dict[str, str] = {}
    for _, ev in step_events:
        final[ev.step_id] = ev.status.value
    s.outcome = " ".join(
        f"{sid}{STATUS_CODE.get(final.get(sid, 'pending'), '?')}" for sid in protocol.step_ids
    )

    s.detail = {
        "take": dict(meta),
        "truth": truth.to_json(),
        "correct_take": s.correct_take,
        "completions": {
            "matched": matches,
            "missed": [
                {"step_id": occs[i].step_id, "annotated_end": _r(occs[i].t_end)} for i in missed
            ],
            "false": [
                {
                    "step_id": comps[j][1].step_id,
                    "seq": comps[j][0],
                    "status": comps[j][1].status.value,
                    "claimed_end": _r(comps[j][1].t_end),
                }
                for j in false
            ],
        },
        "skips": {
            "annotated": list(gt_skip),
            "alerted": named,
            "alert_seqs": [q for q, a in alerts if a.kind is AlertKind.SKIP],
        },
        "out_of_order": {
            "incidents": truth.to_json()["incidents"],
            "alerts": [
                {
                    "seq": q,
                    "step_ids": list(a.step_ids),
                    "t": _r(a.t),
                    "matched_incident": next((i for i, j in ooo_pairs if j == k), None),
                }
                for k, (q, a) in enumerate(ooo)
            ],
        },
        "alerts": [
            {"seq": q, "kind": a.kind.value, "step_ids": list(a.step_ids), "t": _r(a.t)}
            for q, a in alerts
        ],
        "asks": ask_rows,
        "final_status": final,
    }
    return s


def score_asks(truth: GroundTruth, recs: Sequence[Rec], tol: float) -> list[dict[str, Any]]:
    """One row per distinct Ask: how it resolved, and whether that agrees with the annotations."""
    rows = []
    seen: set[str] = set()
    for idx, (seq, o) in enumerate(recs):
        if not isinstance(o, Ask) or o.question_id in seen:
            continue
        seen.add(o.question_id)
        res = next(
            (
                (q, e)
                for q, e in recs[idx + 1 :]
                if isinstance(e, StepEvent)
                and e.step_id == o.step_id
                and (e.status in DONE or e.status is StepStatus.SKIPPED)
            ),
            None,
        )
        ans = next(
            (a for _, a in recs if isinstance(a, Answer) and a.question_id == o.question_id), None
        )
        ever = any(x.step_id == o.step_id for x in truth.occurrences)
        if res is None:
            resolution, correct = "unresolved", not ever
        else:
            done_by = truth.performed_by(o.step_id, res[1].t + tol)
            resolution = res[1].status.value
            correct = done_by if res[1].status in DONE else not done_by
        rows.append(
            {
                "question_id": o.question_id,
                "step_id": o.step_id,
                "ask_seq": seq,
                "t": _r(o.t),
                "answer": ans.value.value if ans is not None else None,
                "resolution": resolution,
                "resolution_seq": res[0] if res is not None else None,
                "correct": bool(correct),
            }
        )
    return rows


# --------------------------------------------------------------------------------------------
# Aggregation, criteria, report
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Metric:
    value: float | None
    unit: str  # "ratio" | "per_min" | "s"
    num: float = 0.0
    den: float = 0.0

    @classmethod
    def ratio(cls, num: float, den: float) -> Metric:
        return cls(num / den if den else None, "ratio", num, den)

    @classmethod
    def rate(cls, count: float, minutes: float) -> Metric:
        return cls(count / minutes if minutes > 0 else None, "per_min", count, minutes)

    @classmethod
    def median(cls, xs: Sequence[float]) -> Metric:
        return cls(float(statistics.median(xs)) if xs else None, "s", 0.0, float(len(xs)))

    def show(self) -> str:
        if self.value is None:
            return "n/a (no cases)"
        if self.unit == "ratio":
            return f"{self.value:.3f} ({self.num:g}/{self.den:g})"
        if self.unit == "per_min":
            return f"{self.value:.3f}/min ({self.num:g} in {self.den:.1f} min)"
        return f"{self.value:+.2f} s (n={self.den:g})"

    def to_json(self) -> dict[str, Any]:
        return {"value": _r(self.value), "unit": self.unit, "num": self.num, "den": _r(self.den)}


def aggregate(scores: Sequence[TakeScore]) -> dict[str, Metric]:
    def total(attr: str) -> int:
        return sum(getattr(s, attr) for s in scores)

    tp, fp, fn = total("tp"), total("fp"), total("fn")
    stp, sfp, sfn = total("skip_tp"), total("skip_fp"), total("skip_fn")
    otp, ofp, ofn = total("ooo_tp"), total("ooo_fp"), total("ooo_fn")
    minutes = sum(s.duration_s for s in scores) / 60.0
    correct_minutes = sum(s.duration_s for s in scores if s.correct_take) / 60.0
    metrics = {
        "step_completion_precision": Metric.ratio(tp, tp + fp),
        "step_completion_recall": Metric.ratio(tp, tp + fn),
        "skip_alert_precision": Metric.ratio(stp, stp + sfp),
        "skip_alert_recall": Metric.ratio(stp, stp + sfn),
        "out_of_order_alert_precision": Metric.ratio(otp, otp + ofp),
        "out_of_order_alert_recall": Metric.ratio(otp, otp + ofn),
        "false_alerts_per_min_correct_takes": Metric.rate(total("false_alerts"), correct_minutes),
        "median_detection_latency_s": Metric.median([x for s in scores for x in s.latencies]),
        "ask_rate_per_min": Metric.rate(total("asks"), minutes),
        "ask_then_correct_rate": Metric.ratio(total("asks_correct"), total("asks")),
    }
    assert tuple(metrics) == METRICS
    return metrics


def judge(metrics: Mapping[str, Metric], spec: EvalSpec) -> list[dict[str, Any]]:
    return [
        {
            "metric": c.metric,
            "target": f"{c.op} {c.target:g}",
            "value": metrics[c.metric].show(),
            "result": c.judge(metrics[c.metric].value),
        }
        for c in spec.criteria
    ]


def per_step_rows(scores: Sequence[TakeScore], protocol: Protocol) -> list[dict[str, Any]]:
    rows = []
    for st in protocol.steps:
        agg = {"gt": 0, "tp": 0, "fp": 0, "fn": 0}
        lat: list[float] = []
        for s in scores:
            d = s.per_step.get(st.id)
            if d is None:
                continue
            for k in agg:
                agg[k] += d[k]
            lat += d["latencies"]
        prec = Metric.ratio(agg["tp"], agg["tp"] + agg["fp"]).value
        rec = Metric.ratio(agg["tp"], agg["tp"] + agg["fn"]).value
        rows.append(
            {
                "step": st.id,
                "text": st.text,
                "performed": agg["gt"],
                "matched": agg["tp"],
                "missed": agg["fn"],
                "false": agg["fp"],
                "precision": "-" if prec is None else round(prec, 3),
                "recall": "-" if rec is None else round(rec, 3),
                "median_latency_s": round(statistics.median(lat), 2) if lat else "-",
            }
        )
    return rows


def alert_rows(scores: Sequence[TakeScore]) -> list[dict[str, Any]]:
    kinds = [k.value for k in AlertKind]
    return [
        {
            "kind": k,
            "all_takes": sum(s.alerts_by_kind.get(k, 0) for s in scores),
            "correct_takes": sum(s.alerts_by_kind.get(k, 0) for s in scores if s.correct_take),
        }
        for k in kinds
    ]


def build_report(
    scores: Sequence[TakeScore],
    protocol: Protocol,
    spec: EvalSpec,
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    """The report card. Top-level scalars and flat row arrays only (plus ``detail``), because the
    Validation tab renders scalars as cards and every top-level array as a table."""
    metrics = aggregate(scores)
    crit = judge(metrics, spec)
    passed = sum(1 for c in crit if c["result"] == "PASS")
    na = sum(1 for c in crit if c["result"] == "n/a")
    simulated = bool(meta.get("simulated"))
    operators = sorted({s.operator for s in scores})
    excluded = list(meta.get("excluded") or [])
    report: dict[str, Any] = {
        "data": (
            "SIMULATED replay: synthetic scenes, not real footage"
            if simulated
            else f"REAL held-out takes ({meta.get('selection', 'selected takes')})"
        ),
        "criteria_met": f"{passed}/{len(crit)}" + (f" ({na} n/a)" if na else ""),
        **{name: _r(m.value) for name, m in metrics.items()},
        "takes_evaluated": str(len(scores)),
        "takes_excluded": str(len(excluded)),
        "operators": ", ".join(operators) or "-",
        "protocol": protocol.id,
        "answers": f"simulated operator: {meta.get('answer_policy')}",
        "criteria_version": f"v{spec.version} (frozen {spec.frozen_on})",
        "eval_id": str(meta.get("eval_id", "")),
        "generated_at": str(meta.get("generated_at", "")),
        "simulated": simulated,
        "criteria": crit,
        "takes": [s.row() for s in scores],
        "per_step": per_step_rows(scores, protocol),
        "alerts_by_kind": alert_rows(scores),
        "definitions": [{"metric": k, "definition": v} for k, v in DEFINITIONS.items()],
        "detail": {
            "metrics": {k: m.to_json() for k, m in metrics.items()},
            "spec": {
                "path": spec.path,
                "sha256": spec.sha256,
                "version": spec.version,
                "frozen_on": spec.frozen_on,
                "completion_tol_s": spec.completion_tol_s,
                "alert_tol_s": spec.alert_tol_s,
                "answer_delay_s": spec.answer_delay_s,
            },
            "counts": {
                "correct_takes": sum(1 for s in scores if s.correct_take),
                "no_procedure_clips": sum(1 for s in scores if not s.has_procedure),
                "minutes": _r(sum(s.duration_s for s in scores) / 60.0),
                "ooo_alerts_crediting_a_skip": sum(s.ooo_skip_precursors for s in scores),
                "answers": dict(sum((s.answers for s in scores), Counter())),
            },
            "excluded_takes": excluded,
            "terms": TERMS,
            "provenance": dict(meta.get("provenance") or {}),
            "takes_dir": str(meta.get("takes_dir", "")),
        },
    }
    return report


def _md_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> str:
    if not rows:
        return "_none_\n"
    cols = list(columns or rows[0].keys())

    def cell(v: Any) -> str:
        return str("-" if v is None else v).replace("|", "\\|").replace("\n", " ")

    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(cell(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(out) + "\n"


def render_markdown(report: Mapping[str, Any]) -> str:
    d = report["detail"]
    spec, prov = d["spec"], d["provenance"]
    lines = ["# ACTUS evaluation report card", ""]
    if report["simulated"]:
        lines += [
            "> **SIMULATED.** These numbers come from the synthetic replay, not from real "
            "footage. They exercise the harness and the protocol engine, not perception. Do not "
            "quote them as results.",
            "",
        ]
    commit = prov.get("git_commit") or "unknown"
    dirty = " (with uncommitted changes)" if prov.get("git_dirty") else ""
    lines += [
        f"- **Data:** {report['data']}",
        f"- **Criteria met:** {report['criteria_met']}, against pre-registered criteria "
        f"{report['criteria_version']} (`{spec['path']}`, sha256 `{spec['sha256'][:16]}`)",
        f"- **Protocol:** `{report['protocol']}`; operators: {report['operators']}",
        f"- **Takes:** {report['takes_evaluated']} evaluated, {report['takes_excluded']} "
        f"excluded ({d['counts']['correct_takes']} correct, "
        f"{d['counts']['no_procedure_clips']} with no procedure; "
        f"{d['counts']['minutes']} min in total)",
        f"- **Asks answered by:** {report['answers']} (see *oracle operator* below)",
        f"- **Pipeline stages:** {prov.get('stages', 'unknown')}",
        f"- **Code:** git `{commit}`{dirty}",
        f"- **Evaluation:** `{report['eval_id']}`, generated {report['generated_at']}; per-take "
        f"logs, annotations and matches in `{d['takes_dir']}/<take>/`",
        "",
        "## Report card",
        "",
        "Targets were fixed before the first held-out number existed.",
        "",
        _md_table(report["criteria"]),
        "## Takes",
        "",
        "`outcome` codes: v verified, c operator_confirmed, o out_of_order, s skipped, "
        "t timed_out, p pending. `events_sha256` is the first 12 hex digits of the take's "
        "`events.jsonl`, whose hash chain was verified before scoring.",
        "",
        _md_table(report["takes"]),
        "## Per step",
        "",
        _md_table(report["per_step"]),
        "## Alerts by kind",
        "",
        _md_table(report["alerts_by_kind"]),
        "## Excluded takes",
        "",
        _md_table(d["excluded_takes"]),
        "## Definitions",
        "",
        f"Tolerances: completion ±{spec['completion_tol_s']:g} s, alert ±{spec['alert_tol_s']:g} "
        f"s; the simulated operator answers {spec['answer_delay_s']:g} s after an Ask.",
        "",
    ]
    lines += [f"- **{k}**: {v}" for k, v in DEFINITIONS.items()]
    lines += [f"- *{k}*: {v}" for k, v in d["terms"].items()]
    lines += ["", "## Caveats", ""]
    lines += [f"- {c}" for c in _caveats(report)]
    if prov.get("command"):
        lines += ["", "## Reproduce", "", "```sh", str(prov["command"]), "```"]
    return "\n".join(lines) + "\n"


def _caveats(report: Mapping[str, Any]) -> list[str]:
    out = []
    if report["simulated"]:
        out.append(
            "SIMULATED: the replay's ground truth is its own script, and in `--evidence replay` "
            "mode the engine is fed the replay's stand-in evidence rather than perception."
        )
    out.append(
        "Asks are answered by a simulated operator, so the ask metrics describe the engine's "
        "handling of an answer, not how real operators answer."
    )
    na = [c["metric"] for c in report["criteria"] if c["result"] == "n/a"]
    if na:
        out.append(f"No cases for: {', '.join(na)}. n/a is not a pass.")
    n = int(report["takes_evaluated"])
    if n < 10:
        out.append(f"Only {n} take(s): each take moves these numbers a lot.")
    out.append(
        "Ground truth is human annotation with step boundaries marked by hand; the tolerances "
        "above absorb that, and they were fixed in advance."
    )
    return out


# --------------------------------------------------------------------------------------------
# Selecting and running takes
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TakeInput:
    take_id: str
    operator: str
    scenario: str
    input_spec: str  # a video path, or "replay:<scenario>"
    rows: tuple[LabelRow, ...]
    annotations_source: str
    simulated: bool
    seed: int | None = None


def split_operators(report_path: str | Path, split: str) -> list[str]:
    p = resolve_path(report_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"{p} not found: run tools/make_splits.py first, or pass --operators / --takes"
        )
    ops = json.loads(p.read_text(encoding="utf-8")).get("operators") or {}
    chosen = sorted(o for o, s in ops.items() if s == split)
    if not chosen:
        raise ValueError(f"{p}: no operator is in split {split!r} ({ops})")
    return chosen


def collect_takes(
    raw_dir: str | Path,
    *,
    operators: Sequence[str] | None = None,
    stems: Sequence[str] | None = None,
    exclude_scenarios: Sequence[str] = (),
) -> tuple[list[TakeInput], list[dict[str, str]]]:
    """Labelled camera-1 takes for ``operators`` (or exactly ``stems``), and the ones left out
    with the reason. Explicit ``stems`` are never excluded by scenario."""
    from tools._datalib import discover_takes
    from tools.record_takes import is_pending, parse_take_stem

    raw_dir = resolve_path(raw_dir)
    wanted_ops = set(operators) if operators is not None else None
    wanted = set(stems) if stems is not None else None

    def selected(stem: str, operator: str) -> bool:
        if wanted is not None:
            return stem in wanted
        return wanted_ops is None or operator in wanted_ops

    inputs: list[TakeInput] = []
    excluded: list[dict[str, str]] = []
    found: set[str] = set()
    for take in discover_takes(raw_dir):
        if not selected(take.stem, take.operator):
            continue
        found.add(take.stem)
        if wanted is None and take.scenario in exclude_scenarios:
            excluded.append({"take": take.stem, "reason": f"scenario {take.scenario!r} excluded"})
            continue
        if is_pending(take.video_path):
            excluded.append({"take": take.stem, "reason": "unlabelled (only the initial idle)"})
            continue
        inputs.append(
            TakeInput(
                take.stem,
                take.operator,
                take.scenario,
                str(take.video_path),
                tuple(read_label_rows(take.labels_path)),
                _rel(take.labels_path),
                False,
            )
        )
    for video in sorted(raw_dir.glob("*.mp4")):  # labelled takes were found above
        try:
            info = parse_take_stem(video.stem)
        except ValueError:
            continue
        if info["cam"] == 1 and video.stem not in found and selected(video.stem, info["operator"]):
            found.add(video.stem)
            excluded.append({"take": video.stem, "reason": "no label CSV"})
    if wanted is not None and wanted - found:
        raise FileNotFoundError(f"no such take(s) in {raw_dir}: {sorted(wanted - found)}")
    return inputs, excluded


def replay_takes(
    scenarios: Sequence[str], seeds: Sequence[int], protocol: Protocol, cfg: Config
) -> list[TakeInput]:
    from actus.capture.replay import SCENARIOS, ReplaySource

    names = sorted(SCENARIOS) if list(scenarios) == ["all"] else list(scenarios)
    out = []
    for name in names:
        for seed in seeds:
            src = ReplaySource(
                name,
                protocol=protocol,
                fps=float(cfg.get("replay.fps", 15.0)),
                seed=int(seed),
                noise=float(cfg.get("replay.noise", 0.04)),
            )
            out.append(
                TakeInput(
                    f"replay-{name}-s{seed}",
                    "SIMULATED",
                    name,
                    f"replay:{name}",
                    tuple(src.annotation_rows()),
                    "actus.capture.replay (the scenario script)",
                    True,
                    int(seed),
                )
            )
    return out


def make_answer(
    policy: str, truth: GroundTruth, spec: EvalSpec
) -> Callable[[Ask], tuple[float, AnswerValue]] | str | None:
    """The simulated operator. ``oracle`` answers truthfully from the annotations."""
    if policy == "none":
        return None
    if policy in ("yes", "no"):
        return policy

    def oracle(ask: Ask) -> tuple[float, AnswerValue]:
        by = ask.t + spec.answer_delay_s + spec.completion_tol_s
        done = truth.performed_by(ask.step_id, by)
        return spec.answer_delay_s, AnswerValue.YES if done else AnswerValue.NO

    return oracle


def run_take(
    take: TakeInput,
    cfg: Config,
    protocol: Protocol,
    spec: EvalSpec,
    *,
    answer_policy: str,
    evidence: str | None = None,
    video: bool = False,
    max_frames: int | None = None,
    force: bool = False,
    stages: Mapping[str, Any] | None = None,
) -> Path:
    """Run one take through ``run_offline`` and store what scoring needs next to its log."""
    from actus.app.pipeline import run_offline

    # The oracle answers during the run, so it uses the annotations' own clock (T-REC's `t`,
    # which is the take's timeline); scoring later uses the pipeline's clock.
    pre = derive_truth(take.rows, protocol, clock=FrameClock.from_rows(take.rows), n_frames=None)
    frames: list[int] = []
    times: list[float] = []

    def on_result(res) -> None:
        frames.append(res.frame_idx)
        times.append(res.t)

    kwargs: dict[str, Any] = dict(stages or {})
    if max_frames is not None and not take.simulated:
        kwargs["max_frames"] = max_frames
    res = run_offline(
        take.input_spec,
        cfg,
        run_id=take.take_id,
        answer=make_answer(answer_policy, pre, spec),
        answer_delay_s=spec.answer_delay_s,
        video=video,
        evidence=evidence,
        force=force,
        on_result=on_result,
        **kwargs,
    )
    if not frames:
        raise RuntimeError(f"{take.take_id}: the pipeline processed no frames")
    clock = FrameClock(frames, times)
    write_label_rows(res.run_dir / "annotations.csv", take.rows)
    needed = [*(r[0] for r in take.rows), frames[0], frames[-1], 0]
    _write_json(
        res.run_dir / "clock.json",
        {
            "note": "pipeline t of every annotated frame (and the first and last frame)",
            "frames_processed": len(frames),
            "n_frames": int(frames[-1]) + 1,
            "points": clock.points(f for f in needed if f <= frames[-1]),
        },
    )
    _write_json(
        res.run_dir / "take.json",
        {
            "take_id": take.take_id,
            "operator": take.operator,
            "scenario": take.scenario,
            "input": _rel(take.input_spec) if not take.simulated else take.input_spec,
            "annotations_source": take.annotations_source,
            "annotations_sha256": hashlib.sha256(
                (res.run_dir / "annotations.csv").read_bytes()
            ).hexdigest(),
            "simulated": take.simulated,
            "seed": take.seed,
            "answer_policy": answer_policy,
            "evidence": evidence,
            "events_file": res.events_path.name,
            "events_sha256": res.events_sha256,
        },
    )
    return res.run_dir


def score_take_dir(
    take_dir: str | Path, protocol: Protocol, spec: EvalSpec, *, timeout_s: float
) -> TakeScore:
    """Score one take from the files in its directory alone, and write ``score.json``."""
    take_dir = Path(take_dir)
    meta = json.loads((take_dir / "take.json").read_text(encoding="utf-8"))
    ck = json.loads((take_dir / "clock.json").read_text(encoding="utf-8"))
    rows = read_label_rows(take_dir / "annotations.csv")
    truth = derive_truth(
        rows, protocol, clock=FrameClock.from_points(ck["points"]), n_frames=int(ck["n_frames"])
    )
    events = take_dir / meta.get("events_file", "events.jsonl")
    recs, _ = read_events(events)
    sha = hashlib.sha256(events.read_bytes()).hexdigest()
    score = score_take(meta, truth, recs, protocol, spec, timeout_s=timeout_s, events_sha256=sha)
    _write_json(take_dir / "score.json", {"row": score.row(), **score.detail})
    return score


def run_evaluation(
    inputs: Sequence[TakeInput],
    eval_dir: str | Path,
    *,
    spec: EvalSpec,
    config: str | None = None,
    overrides: Sequence[str] = (),
    answer_policy: str | None = None,
    evidence: str | None = None,
    video: bool = False,
    max_frames: int | None = None,
    force: bool = False,
    stages: Mapping[str, Any] | None = None,
    excluded: Sequence[Mapping[str, str]] = (),
    selection: str = "",
    command: str = "",
) -> dict[str, Any]:
    """Run every take, score it from its log, write ``report.json`` / ``report.md`` into
    ``eval_dir`` and return the report."""
    if not inputs:
        raise ValueError("no takes to evaluate")
    eval_dir = Path(eval_dir)
    takes_dir = eval_dir / "takes"
    if eval_dir.exists() and any(eval_dir.iterdir()) and not force:
        raise FileExistsError(f"{eval_dir} is not empty (pass --force, or a new --eval-id)")
    policy = answer_policy or spec.answer_policy
    base = [*overrides, f"run.runs_dir={takes_dir.resolve().as_posix()}"]

    def cfg_for(take: TakeInput) -> Config:
        extra = [f"replay.seed={take.seed}"] if take.seed is not None else []
        return load_config(config, overrides=[*base, *extra])

    cfg0 = cfg_for(inputs[0])
    protocol = load_protocol(cfg0.protocol.path)
    timeout_s = _timeout_s(cfg0, protocol)
    for take in inputs:
        print(f"# {take.take_id}: running", flush=True)
        run_take(
            take,
            cfg_for(take),
            protocol,
            spec,
            answer_policy=policy,
            evidence=evidence,
            video=video,
            max_frames=max_frames,
            force=force,
            stages=stages,
        )
    meta = {
        "eval_id": eval_dir.name,
        "created_at": _now_iso(),
        "simulated": any(t.simulated for t in inputs),
        "protocol_path": _rel(resolve_path(cfg0.protocol.path)),
        "protocol_id": protocol.id,
        "timeout_s": timeout_s,
        "answer_policy": policy,
        "evidence": evidence,
        "selection": selection,
        "takes": [t.take_id for t in inputs],
        "excluded": list(excluded),
        "overrides": list(overrides),
        "config": config,
        "command": command,
    }
    _write_json(eval_dir / "eval.json", meta)
    return rescore(eval_dir, spec=spec)


def rescore(eval_dir: str | Path, *, spec: EvalSpec) -> dict[str, Any]:
    """(Re)build the report of an evaluation from the files it left behind."""
    eval_dir = Path(eval_dir)
    meta = json.loads((eval_dir / "eval.json").read_text(encoding="utf-8"))
    protocol = load_protocol(meta["protocol_path"])
    timeout_s = float(meta["timeout_s"])
    scores = [
        score_take_dir(eval_dir / "takes" / tid, protocol, spec, timeout_s=timeout_s)
        for tid in meta["takes"]
    ]
    report = build_report(
        scores,
        protocol,
        spec,
        {
            **meta,
            "generated_at": _now_iso(),
            "takes_dir": _rel(eval_dir / "takes"),
            "provenance": _provenance(eval_dir, meta),
        },
    )
    _write_json(eval_dir / "report.json", report)
    (eval_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def publish(eval_dir: str | Path, out_root: str | Path = EVAL_ROOT) -> Path:
    """Copy an evaluation's report to ``runs/eval/report.{json,md}`` (what the UI shows)."""
    eval_dir, out_root = Path(eval_dir), resolve_path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for name in ("report.json", "report.md"):
        shutil.copyfile(eval_dir / name, out_root / name)
    return out_root / "report.json"


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def _r(x: float | None, nd: int = 4) -> float | None:
    if x is None or not math.isfinite(x):
        return None
    return round(float(x), nd)


def _rel(path: str | Path) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def _now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _timeout_s(cfg: Config, protocol: Protocol) -> float:
    return DecisionParams.from_protocol(protocol, **dict(cfg.get("engine") or {})).timeout_s


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _provenance(eval_dir: Path, meta: Mapping[str, Any]) -> dict[str, Any]:
    stages = set()
    for tid in meta["takes"]:
        run_json = eval_dir / "takes" / tid / "run.json"
        if run_json.is_file():
            st = json.loads(run_json.read_text(encoding="utf-8")).get("stages") or {}
            stages.add(
                ", ".join(f"{k}={st.get(k)}" for k in ("detector", "handpose", "rack", "evidence"))
                + f", models={'on' if st.get('models_enabled') else 'off'}"
            )
    status = _git("status", "--porcelain")
    return {
        "stages": " | ".join(sorted(stages)) or "unknown",
        "stages_consistent": len(stages) <= 1,
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": sys.version.split()[0],
        "config": meta.get("config"),
        "overrides": meta.get("overrides"),
        "command": meta.get("command"),
    }


def _csv_arg(spec: str | None) -> list[str] | None:
    return None if spec is None else [x.strip() for x in spec.split(",") if x.strip()]


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    what = ap.add_mutually_exclusive_group()
    what.add_argument("--split", help="split in data/splits/report.json (default: the spec's)")
    what.add_argument("--operators", help="comma-separated operator ids")
    what.add_argument("--takes", help="comma-separated take stems")
    what.add_argument("--replay", help="SIMULATED: comma-separated replay scenarios, or 'all'")
    what.add_argument("--rescore", metavar="EVAL_DIR", help="rebuild a report from its logs")
    ap.add_argument("--seeds", default="0", help="replay only: comma-separated seeds")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--splits-report", default="data/splits/report.json")
    ap.add_argument("--spec", default=SPEC_PATH, help="pre-registered criteria YAML")
    ap.add_argument("--config", help="override YAML on top of configs/default.yaml")
    ap.add_argument("--override", action="append", default=[], help="key.path=value")
    ap.add_argument("--out", default=EVAL_ROOT, help="root of evaluation outputs")
    ap.add_argument("--eval-id", help="default: heldout-<split>-<time> / replay-<time>")
    ap.add_argument("--answer", choices=ANSWER_POLICIES, help="simulated operator policy")
    ap.add_argument("--evidence", choices=["spatial", "replay"], help="replay only: stand-in")
    ap.add_argument("--video", action="store_true", help="also write annotated MP4s (large)")
    ap.add_argument("--max-frames", type=int, help="takes only: stop after this many frames")
    ap.add_argument("--force", action="store_true", help="overwrite an existing evaluation")
    pub = ap.add_mutually_exclusive_group()
    pub.add_argument("--publish", action="store_true", help="also publish a SIMULATED report")
    pub.add_argument("--no-publish", action="store_true", help="don't update runs/eval/report.*")
    args = ap.parse_args(argv)

    spec = load_spec(args.spec)
    out_root = resolve_path(args.out)
    command = "uv run python tools/evaluate.py " + " ".join(
        sys.argv[1:] if argv is None else argv
    )

    if args.rescore:
        eval_dir = resolve_path(args.rescore)
        report = rescore(eval_dir, spec=spec)
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        excluded: list[dict[str, str]] = []
        if args.replay:
            cfg = load_config(args.config, overrides=args.override)
            protocol = load_protocol(cfg.protocol.path)
            seeds = [int(s) for s in _csv_arg(args.seeds) or ["0"]]
            inputs = replay_takes(_csv_arg(args.replay) or ["all"], seeds, protocol, cfg)
            selection = f"replay {args.replay}, seeds {','.join(map(str, seeds))}"
            eval_id = args.eval_id or f"replay-{stamp}"
        else:
            if args.takes:
                stems, ops = _csv_arg(args.takes), None
                selection = f"takes {args.takes}"
            elif args.operators:
                stems, ops = None, _csv_arg(args.operators)
                selection = f"operators {args.operators}"
            else:
                split = args.split or spec.split
                stems, ops = None, split_operators(args.splits_report, split)
                selection = f"split {split!r}: operators {', '.join(ops)}"
            inputs, excluded = collect_takes(
                args.raw_dir, operators=ops, stems=stems, exclude_scenarios=spec.exclude_scenarios
            )
            eval_id = args.eval_id or f"heldout-{args.split or spec.split}-{stamp}"
            for e in excluded:
                print(f"# excluded {e['take']}: {e['reason']}")
            if not inputs:
                print("no labelled takes to evaluate", file=sys.stderr)
                return 1
        eval_dir = out_root / eval_id
        report = run_evaluation(
            inputs,
            eval_dir,
            spec=spec,
            config=args.config,
            overrides=args.override,
            answer_policy=args.answer,
            evidence=args.evidence,
            video=args.video,
            max_frames=args.max_frames,
            force=args.force,
            excluded=excluded,
            selection=selection,
            command=command,
        )

    simulated = bool(report["simulated"])
    if args.publish or (not simulated and not args.no_publish):
        print(f"# published -> {publish(eval_dir, out_root)}")
    tag = " [SIMULATED]" if simulated else ""
    print(f"# report{tag}: {eval_dir / 'report.md'}")
    for c in report["criteria"]:
        print(f"  {c['result']:<5} {c['metric']:<36} {c['value']}  (target {c['target']})")
    print(f"# criteria met: {report['criteria_met']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
