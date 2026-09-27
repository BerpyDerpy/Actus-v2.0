"""T-ENGINE replay tests: exact event sequences for every scenario, across seeds.

The sequences are written in the compact form of ``engine.describe``. Each scenario must give
exactly this sequence for every seed at the replay's nominal noise (0.04): the engine's
decisions must not depend on the noise realisation.
"""

import pytest

from actus.capture.replay import (
    SCENARIOS,
    STEP_S,
    Action,
    ReplaySource,
    Scenario,
    idle,
    step,
)
from actus.core.bus import Bus
from actus.core.contracts import (
    Alert,
    AlertKind,
    Answer,
    Ask,
    Prompt,
    StepEvent,
    StepStatus,
    Topic,
)
from actus.protocol.engine import (
    EngineRunner,
    ProtocolEngine,
    describe,
    run_ticks,
)
from actus.protocol.schema import load_protocol

PROTO = load_protocol("protocols/bas_boxes.yaml")
SEEDS = range(5)

_CORRECT_TAIL = [
    "event S4 verified",
    "prompt S5",
    "event S5 verified",
    "prompt S6",
    "event S6 verified",
    "prompt -",
]
_UP_TO_S3 = [
    "prompt S1",
    "event S1 verified",
    "prompt S2",
    "event S2 verified",
    "prompt S3",
]

EXPECTED: dict[tuple[str, str | None], list[str]] = {
    ("correct", None): [*_UP_TO_S3, "event S3 verified", "prompt S4", *_CORRECT_TAIL],
    # A skip surfaces live as out_of_order (S4 done while S2, S3 clearly not done), and is
    # confirmed as a retroactive skip when the procedure can go no further (S6 closes the box).
    ("skip_S2", None): [
        "prompt S1",
        "event S1 verified",
        "prompt S2",
        "event S4 out_of_order",
        "alert out_of_order S4,S2,S3",
        "prompt S2",
        "event S5 out_of_order",
        "prompt S2",
        "event S6 verified",
        "event S2 skipped (retro)",
        "event S3 skipped (retro)",
        "alert skip S2,S3",
        "prompt -",
    ],
    ("S4_before_S2", None): [
        "prompt S1",
        "event S1 verified",
        "prompt S2",
        "event S4 out_of_order",
        "alert out_of_order S4,S2,S3",
        "prompt S2",
        "event S5 out_of_order",
        "prompt S2",
        "event S2 verified",
        "prompt S3",
        "event S3 verified",
        "prompt S6",
        "event S6 verified",
        "prompt -",
    ],
    # S3's evidence sits at ~0.62: ask after 3 s. Unanswered, S3 stays pending (never accused:
    # its evidence is unsure, not low) and the prompt keeps pointing at it.
    ("unsure_S3", None): [
        *_UP_TO_S3,
        "ask S3",
        "event S4 verified",
        "prompt S3",
        "event S5 verified",
        "prompt S3",
        "event S6 verified",
        "prompt S3",
    ],
    ("unsure_S3", "yes"): [
        *_UP_TO_S3,
        "ask S3",
        "event S4 verified",
        "prompt S3",
        "event S3 operator_confirmed",
        "prompt S5",
        "event S5 verified",
        "prompt S6",
        "event S6 verified",
        "prompt -",
    ],
    ("unsure_S3", "no"): [
        *_UP_TO_S3,
        "ask S3",
        "event S4 verified",
        "prompt S3",
        "prompt S3",  # the "no" re-prompt
        "event S5 verified",
        "prompt S3",
        "event S6 verified",
        "prompt S3",
    ],
    # Blind mid-S3: ask about the expected step; when the view returns, evidence verifies it
    # (which closes the ask).
    ("occluded", None): [
        *_UP_TO_S3,
        "ask S3",
        "event S3 verified",
        "prompt S4",
        *_CORRECT_TAIL,
    ],
    ("occluded", "yes"): [
        *_UP_TO_S3,
        "ask S3",
        "event S3 operator_confirmed",
        "prompt S4",
        *_CORRECT_TAIL,
    ],
    ("occluded", "no"): [
        *_UP_TO_S3,
        "ask S3",
        "prompt S3",
        "event S3 verified",
        "prompt S4",
        *_CORRECT_TAIL,
    ],
    # Red box returned to the outer box: S3 then S2 are undone -> one incident, one alert.
    ("unexpected", None): [
        *_UP_TO_S3,
        "event S3 verified",
        "prompt S4",
        "event S3 pending",
        "alert unexpected S3",
        "prompt S3",
        "event S2 pending",
        "prompt S2",
        "event S2 verified",
        "prompt S3",
        "event S3 verified",
        "prompt S4",
        *_CORRECT_TAIL,
    ],
    ("timeout", None): [
        *_UP_TO_S3,
        "event S3 verified",
        "prompt S4",
        "event S4 timed_out",
        "alert timeout S4",
        *_CORRECT_TAIL,
    ],
}


def _run(scenario, *, seed=0, noise=0.04, answer=None, **kw):
    src = scenario if isinstance(scenario, ReplaySource) else None
    if src is None:
        src = ReplaySource(scenario, protocol=PROTO, seed=seed, noise=noise)
    engine = ProtocolEngine(PROTO)
    return engine, run_ticks(engine, src, answer=answer, **kw)


def test_every_replay_scenario_has_an_expected_sequence():
    assert {name for name, _ in EXPECTED} == set(SCENARIOS)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(("scenario", "answer"), sorted(EXPECTED, key=str))
def test_exact_event_sequence(scenario, answer, seed):
    _, out = _run(scenario, seed=seed, answer=answer)
    assert [describe(o) for o in out] == EXPECTED[(scenario, answer)]


@pytest.mark.parametrize("seed", range(10))
def test_unsure_step_is_never_accused_under_heavy_noise(seed):
    """At 2x noise the unsure S3 may legitimately verify, but it is always asked about and never
    flagged as out of order or skipped."""
    _, out = _run("unsure_S3", seed=seed, noise=0.08)
    kinds = [o.kind for o in out if isinstance(o, Alert)]
    assert kinds == []
    assert any(isinstance(o, Ask) and o.step_id == "S3" for o in out)
    for o in out:
        if isinstance(o, StepEvent) and o.step_id != "S3":
            assert o.status is StepStatus.VERIFIED


def test_card_messages():
    _, out = _run("S4_before_S2")
    (alert,) = [o for o in out if isinstance(o, Alert)]
    assert alert.message == (
        "Out of sequence: yellow box removed before red box. Expected: take out the red box."
    )
    _, out = _run("unsure_S3")
    (ask,) = [o for o in out if isinstance(o, Ask)]
    assert ask.text == "I couldn't confirm the red box is in Zone A. Did you place it?"
    assert ask.question_id == "q1-S3"
    _, out = _run("skip_S2")
    skip = next(o for o in out if isinstance(o, Alert) and o.kind is AlertKind.SKIP)
    assert skip.message == "Skipped: take out the red box; place the red box in Zone A."
    assert out[-1].text == "Procedure finished with 2 skipped steps."
    _, out = _run("unexpected")
    unexp = next(o for o in out if isinstance(o, Alert))
    assert unexp.message == "Unexpected: red box moved out of Zone A."
    _, out = _run("correct")
    assert [o.text for o in out if isinstance(o, Prompt)][:2] == [
        "Open the outer box",
        "Take out the red box",
    ]
    assert out[-1].text == "Procedure complete." and out[-1].next_step_id is None


@pytest.mark.parametrize("seed", SEEDS)
def test_completion_timing_tracks_ground_truth(seed):
    """t_end is when the evidence crossed p_enter (shortly after the true post-state change);
    the decision is dwell_s later; t_start is the true step start (from ActionScores)."""
    src = ReplaySource("correct", protocol=PROTO, seed=seed)
    fps = src.fps
    change_t: dict[str, float] = {}
    start_t: dict[str, float] = {}
    for a, s0, s1 in src.scenario.spans():
        if a.label.startswith("S"):
            start_t[a.label] = s0
            change_t[a.label] = s0 + max(frac for frac, _ in a.changes) * (s1 - s0)
    _, out = _run(src)
    events = [o for o in out if isinstance(o, StepEvent)]
    assert len(events) == 6
    for e in events:
        assert change_t[e.step_id] <= e.t_end <= change_t[e.step_id] + 0.6
        assert e.t - e.t_end == pytest.approx(0.8, abs=1.5 / fps)
        assert e.t_start == pytest.approx(start_t[e.step_id], abs=1.5 / fps)
        assert 0.8 <= e.confidence <= 1.0
        assert e.evidence == {"replay": pytest.approx(e.evidence["replay"])}
        assert e.frame_idx == round(e.t * fps)


def test_outcome_table():
    engine, _ = _run("skip_S2")
    table = {r["step_id"]: r["status"] for r in engine.outcomes()}
    assert table == {
        "S1": "verified",
        "S2": "skipped",
        "S3": "skipped",
        "S4": "out_of_order",
        "S5": "out_of_order",
        "S6": "verified",
    }
    assert engine.finished


def test_operator_confirmed_event_fields():
    _, out = _run("occluded", answer="yes")
    (ask,) = [o for o in out if isinstance(o, Ask)]
    (ev,) = [o for o in out if isinstance(o, StepEvent) and o.step_id == "S3"]
    assert ev.status is StepStatus.OPERATOR_CONFIRMED
    assert ev.evidence["operator"] == 1.0
    assert ev.t_end == ask.t and ev.t_start <= ev.t_end
    assert ev.confidence < 0.8  # the machine's own confidence, honestly reported
    assert ev.t == pytest.approx(ask.t + 1.0, abs=0.1)


def test_too_fast_needs_an_observed_short_step():
    # S3 performed in 0.6 s (the protocol's minimum is 1.0 s)
    acts = (
        idle(2.0),
        step("S1"),
        idle(1.0),
        step("S2"),
        step("S3", 0.6),
        idle(1.0),
        step("S4"),
        step("S5"),
        idle(1.0),
        step("S6"),
        idle(2.0),
    )
    fast = Scenario("fast_S3", "S3 in 0.6 s", acts)
    _, out = _run(ReplaySource(fast, protocol=PROTO))
    alerts = [o for o in out if isinstance(o, Alert)]
    assert [(a.kind, a.step_ids) for a in alerts] == [(AlertKind.TOO_FAST, ("S3",))]
    assert alerts[0].message.startswith("Too fast: place the red box in Zone A took 0.")
    # the step still counts as verified: too_fast is a warning, not a verdict
    s3 = next(o for o in out if isinstance(o, StepEvent) and o.step_id == "S3")
    assert s3.status is StepStatus.VERIFIED
    # without action evidence t_start falls back to the previous completion: conservative
    _, out = _run(ReplaySource(fast, protocol=PROTO), use_actions=False)
    assert not [o for o in out if isinstance(o, Alert)]


def test_no_too_fast_on_normal_runs():
    for name in ("correct", "S4_before_S2", "unexpected"):
        _, out = _run(name)
        assert not any(isinstance(o, Alert) and o.kind is AlertKind.TOO_FAST for o in out)


def test_non_step_action_does_not_break_the_run():
    acts = (
        idle(2.0),
        step("S1"),
        Action("other", STEP_S, name="scratch_head"),
        step("S2"),
        step("S3"),
        step("S4"),
        step("S5"),
        step("S6"),
        idle(2.0),
    )
    _, out = _run(ReplaySource(Scenario("other", "", acts), protocol=PROTO))
    assert [describe(o) for o in out if isinstance(o, (StepEvent, Alert))] == [
        f"event S{i} verified" for i in range(1, 7)
    ]


# -- bus runner --------------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", ["correct", "skip_S2", "unexpected"])
def test_runner_matches_direct_drive(scenario):
    bus = Bus()
    runner = EngineRunner(ProtocolEngine(PROTO), bus)
    ReplaySource(scenario, protocol=PROTO).run(bus, realtime=False, close_bus=True)
    out = runner.run()
    assert [describe(o) for o in out] == EXPECTED[(scenario, None)]
    assert runner.late == 0 and runner.sub.dropped == 0


def test_runner_pump_publishes_outputs_and_applies_answers():
    bus = Bus()
    runner = EngineRunner(ProtocolEngine(PROTO), bus)
    outs = bus.subscribe(
        [Topic.EVENTS, Topic.ALERTS, Topic.PROMPTS, Topic.ASKS], maxsize=1000, name="test"
    )
    src = ReplaySource("occluded", protocol=PROTO)
    answered = False
    for tick in src:
        bus.publish(Topic.PREDICATES, tick.predicates)
        for ev in tick.evidence:
            bus.publish(Topic.EVIDENCE, ev)
        bus.publish(Topic.ACTIONS, tick.actions)
        new = runner.pump()
        for o in new:
            if isinstance(o, Ask) and not answered:
                bus.publish(Topic.ANSWERS, Answer(o.question_id, "yes", "gui", t=tick.t))
                answered = True
    runner.pump()
    envs = outs.drain()
    assert [describe(env.payload) for env in envs] == EXPECTED[("occluded", "yes")]
    topic_of = {
        StepEvent: Topic.EVENTS,
        Alert: Topic.ALERTS,
        Prompt: Topic.PROMPTS,
        Ask: Topic.ASKS,
    }
    assert all(env.topic is topic_of[type(env.payload)] for env in envs)


def test_runner_counts_late_evidence():
    bus = Bus()
    runner = EngineRunner(ProtocolEngine(PROTO), bus)
    ticks = list(ReplaySource("correct", protocol=PROTO))[:3]
    for tick in ticks[:2]:
        for ev in tick.evidence:
            bus.publish(Topic.EVIDENCE, ev)
    runner.pump()
    bus.publish(Topic.EVIDENCE, ticks[0].evidence[0])  # frame 0 again: late
    runner.pump()
    assert runner.late == 1
