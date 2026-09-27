"""T-ENGINE property tests over random evidence streams (hypothesis).

1. No step ever completes by evidence (``verified`` / ``out_of_order``) without its evidence
   having crossed ``p_enter`` at ``t_end`` and held >= ``p_exit`` until the decision, which comes
   at least ``dwell_s`` later.
2. The prior never changes a completion: the same evidence under a strict order, the reversed
   order, and no order at all gives the same completions on the same frames.
3. Each alert fires exactly once per incident.
"""

from __future__ import annotations

import itertools

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from actus.core.contracts import Alert, AlertKind, StepEvent, StepEvidence, StepStatus
from actus.protocol.decision import DecisionParams
from actus.protocol.engine import ProtocolEngine
from actus.protocol.schema import Protocol, load_protocol

PROTO = load_protocol("protocols/bas_boxes.yaml")
IDS = PROTO.step_ids
FPS = 10.0
N_FRAMES = 160
PARAMS = DecisionParams.from_protocol(PROTO, timeout_s=4.0)  # short, so timeouts happen

LEVELS = [0.02, 0.1, 0.3, 0.44, 0.5, 0.62, 0.7, 0.79, 0.81, 0.9, 0.97]
VIS_LEVELS = [0.1, 0.45, 0.6, 0.95]
COMPLETED = {StepStatus.VERIFIED, StepStatus.OUT_OF_ORDER}


def _with_order(order) -> Protocol:
    data = PROTO.model_dump()
    data["order"] = order
    return Protocol.model_validate(data)


PROTOS = {
    "strict": PROTO,
    "reversed": _with_order([[b, a] for a, b in itertools.pairwise(IDS)]),
    "none": _with_order([]),
}

segments = st.lists(st.tuples(st.integers(1, 40), st.sampled_from(LEVELS)), min_size=1, max_size=8)
vis_segments = st.lists(
    st.tuples(st.integers(1, 40), st.sampled_from(VIS_LEVELS)), min_size=1, max_size=5
)


def _expand(segs, n=N_FRAMES) -> list[float]:
    out: list[float] = []
    for dur, level in segs:
        out += [level] * dur
    out += [out[-1]] * n
    return out[:n]


@st.composite
def streams(draw):
    ev = {s: _expand(draw(segments)) for s in IDS}
    vis = _expand(draw(st.one_of(st.just([(1, 0.95)]), vis_segments)))
    return ev, vis


def _run(proto: Protocol, ev, vis):
    engine = ProtocolEngine(proto, PARAMS)
    out = []
    for f in range(N_FRAMES):
        t = f / FPS
        evidence = [StepEvidence(f, t, s, ev[s][f]) for s in IDS]
        out += engine.update(f, t, evidence, visibility=vis[f])
    out += engine.finish()
    return out


SETTINGS = settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@SETTINGS
@given(streams())
def test_no_completion_without_evidence_above_p_enter(data):
    ev, vis = data
    for e in _run(PROTO, ev, vis):
        if not (isinstance(e, StepEvent) and e.status in COMPLETED):
            continue
        p = ev[e.step_id]
        f_cross = round(e.t_end * FPS)
        assert p[f_cross] >= PARAMS.p_enter
        assert all(v >= PARAMS.p_exit for v in p[f_cross : e.frame_idx + 1])
        assert e.t - e.t_end >= PARAMS.dwell_s - 1e-9
        assert e.confidence >= PARAMS.p_exit


@SETTINGS
@given(streams())
def test_prior_never_changes_a_completion(data):
    ev, vis = data

    def completions(proto):
        return sorted(
            (e.frame_idx, e.step_id, e.t_end)
            for e in _run(proto, ev, vis)
            if isinstance(e, StepEvent) and e.status in COMPLETED
        )

    ref = completions(PROTOS["strict"])
    assert completions(PROTOS["reversed"]) == ref
    assert completions(PROTOS["none"]) == ref


@SETTINGS
@given(streams())
def test_each_alert_fires_once_per_incident(data):
    ev, vis = data
    out = _run(PROTO, ev, vis)
    alerted_missing: set[str] = set()  # listed as missing, incident still open
    skipped: set[str] = set()
    timeout_armed = True
    by_frame_events: dict[int, list[StepEvent]] = {}
    for o in out:
        if isinstance(o, StepEvent):
            by_frame_events.setdefault(o.frame_idx, []).append(o)
    completions_seen: set[int] = set()
    for o in out:
        if isinstance(o, StepEvent):
            if o.status in COMPLETED | {StepStatus.OPERATOR_CONFIRMED, StepStatus.PENDING}:
                alerted_missing.discard(o.step_id)  # incident resolved (done or reopened)
                skipped.discard(o.step_id)
                timeout_armed = True  # progress
            if o.status is StepStatus.SKIPPED:
                assert o.retroactive
        elif isinstance(o, Alert):
            same_frame = by_frame_events.get(o.frame_idx, [])
            if o.kind is AlertKind.OUT_OF_ORDER:
                missing = set(o.step_ids[1:])
                assert missing and not (missing & alerted_missing), "missing step re-alerted"
                alerted_missing |= missing
            elif o.kind is AlertKind.SKIP:
                ids = set(o.step_ids)
                assert not (ids & skipped), "step skipped twice in one incident"
                skipped |= ids
                alerted_missing -= ids
                assert ids <= {e.step_id for e in same_frame if e.status is StepStatus.SKIPPED}
            elif o.kind is AlertKind.TIMEOUT:
                assert timeout_armed, "second timeout without progress"
                timeout_armed = False
            elif o.kind is AlertKind.UNEXPECTED:
                (sid,) = o.step_ids
                assert any(e.step_id == sid and e.status is StepStatus.PENDING for e in same_frame)
            elif o.kind is AlertKind.TOO_FAST:
                key = hash((o.frame_idx, o.step_ids))
                assert key not in completions_seen
                completions_seen.add(key)
