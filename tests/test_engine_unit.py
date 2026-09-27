"""T-ENGINE unit tests: hand-written evidence streams, one rule at a time."""

import pytest

from actus.core.contracts import (
    Alert,
    AlertKind,
    Answer,
    Ask,
    Command,
    Predicates,
    Prompt,
    StepEvent,
    StepEvidence,
    StepStatus,
)
from actus.protocol.decision import DecisionParams
from actus.protocol.engine import ProtocolEngine, describe
from actus.protocol.schema import Protocol, load_protocol

PROTO = load_protocol("protocols/bas_boxes.yaml")
LOW = 0.05


class Drive:
    """Feeds a constant-per-segment evidence stream at 10 fps."""

    def __init__(self, proto: Protocol = PROTO, fps: float = 10.0, **params):
        pr = DecisionParams.from_protocol(proto, **params) if params else None
        self.engine = ProtocolEngine(proto, pr)
        self.fps = fps
        self.f = 0
        self.p = dict.fromkeys(proto.step_ids, LOW)
        self.vis = 0.95
        self.out: list = []

    @property
    def t(self) -> float:
        return self.f / self.fps

    def run(self, seconds: float, vis: float | None = None, **p: float) -> list:
        self.p.update(p)
        if vis is not None:
            self.vis = vis
        new: list = []
        for _ in range(round(seconds * self.fps)):
            ev = [StepEvidence(self.f, self.t, s, v, {"rule": v}) for s, v in self.p.items()]
            new += self.engine.update(self.f, self.t, ev, visibility=self.vis)
            self.f += 1
        self.out += new
        return new

    def do(self, step_id: str, seconds: float = 1.0) -> list:
        """Perform a step cleanly: evidence high long enough to verify."""
        return self.run(seconds, **{step_id: 0.93})

    def answer(self, qid: str, value: str) -> list:
        new = self.engine.answer(Answer(qid, value, "gui", t=self.t))
        self.out += new
        return new

    def command(self, name: str) -> list:
        new = self.engine.command(Command(name, "voice", t=self.t))
        self.out += new
        return new


def d(outputs) -> list[str]:
    return [describe(o) for o in outputs]


# -- completion ----------------------------------------------------------------------------------


def test_first_frame_prompts_the_first_step():
    dr = Drive()
    new = dr.run(0.1)
    assert d(new) == ["prompt S1"] and new[0].text == "Open the outer box"
    assert dr.engine.prompt_target == "S1"


def test_completion_needs_crossing_plus_dwell():
    dr = Drive()
    dr.run(0.5)
    assert d(dr.run(0.5, S1=0.95)) == []  # 0.5 s above p_enter: not yet
    assert d(dr.run(0.5, S1=0.5)) == []  # fell below p_exit: reset
    assert d(dr.run(0.7, S1=0.79)) == []  # never crossed p_enter again
    new = dr.run(0.3, S1=0.85) + dr.run(0.6, S1=0.65)  # crossed, then held above p_exit
    assert d(new) == ["event S1 verified", "prompt S2"]
    ev = new[0]
    assert ev.t_end == pytest.approx(2.2)  # the crossing frame
    assert ev.t == pytest.approx(3.0)  # 0.8 s dwell later
    assert ev.evidence == {"rule": 0.65}


def test_evidence_for_unknown_steps_and_alternative_inputs():
    e = ProtocolEngine(PROTO)
    ev = {"S1": StepEvidence(0, 0.0, "S1", 0.9), "zz": StepEvidence(0, 0.0, "zz", 0.9)}
    pr = Predicates(0, 0.0, {}, visibility=0.2)
    assert d(e.update(0, 0.0, ev, visibility=pr)) == ["prompt S1"]
    assert e._blind()
    with pytest.raises(ValueError):
        e.update(0, 0.1, [])
    with pytest.raises(ValueError):
        e.update(1, -1.0, [])


# -- asks and answers ----------------------------------------------------------------------------


def _s1_done() -> Drive:
    dr = Drive()
    dr.run(0.5)
    dr.do("S1")
    dr.run(0.5, S1=0.93)
    return dr


def test_band_ask_then_yes_confirms():
    dr = _s1_done()
    assert d(dr.run(2.9, S2=0.62)) == []
    new = dr.run(0.2)
    assert d(new) == ["ask S2"]
    ask = new[0]
    assert ask.question_id == "q1-S2"
    assert ask.text == "I couldn't confirm it. Did you take out the red box?"
    assert dr.engine.open_ask == ask
    new = dr.answer(ask.question_id, "yes")
    assert d(new) == ["event S2 operator_confirmed", "prompt S3"]
    assert new[0].evidence == {"rule": 0.62, "operator": 1.0}
    assert new[0].confidence == pytest.approx(0.62)
    assert dr.engine.open_ask is None
    assert dr.answer(ask.question_id, "yes") == []  # closed: ignored


def test_band_ask_then_no_reprompts_and_cools_down():
    dr = _s1_done()
    (ask,) = dr.run(3.1, S2=0.62)
    new = dr.answer(ask.question_id, "no")
    assert d(new) == ["prompt S2"] and new[0].text == "Take out the red box"
    assert dr.engine.statuses["S2"] is StepStatus.PENDING
    assert d(dr.run(12.9)) == []  # 10 s cooldown, then 3 s in band again
    assert d(dr.run(0.2)) == ["ask S2"]
    assert dr.out[-1].question_id == "q2-S2"


def test_ask_is_closed_by_evidence_and_late_answers_are_ignored():
    dr = _s1_done()
    (ask,) = dr.run(3.1, S2=0.62)
    assert d(dr.do("S2")) == ["event S2 verified", "prompt S3"]
    assert dr.engine.open_ask is None
    assert dr.answer(ask.question_id, "yes") == []
    assert dr.answer("nope", "yes") == []


def test_blindness_asks_about_the_expected_step_once_per_episode():
    dr = _s1_done()
    assert d(dr.run(1.4, vis=0.2)) == []
    new = dr.run(0.2)
    assert d(new) == ["ask S2"]
    assert dr.engine.statuses["S2"] is StepStatus.PENDING
    dr.answer(new[0].question_id, "no")
    assert [o for o in dr.run(5.0) if isinstance(o, Ask)] == []  # same blind episode
    dr.run(0.5, vis=0.95)
    assert d(dr.run(1.6, vis=0.2)) == ["ask S2"]  # new episode


def test_no_band_ask_or_accusation_while_blind():
    dr = _s1_done()
    # blind for a long time, S2 evidence in the band and S3 done: no band ask, no accusation
    dr.run(0.2, vis=0.1)
    new = dr.run(10.0, S2=0.6, S3=0.93)
    assert [describe(o) for o in new if not isinstance(o, Ask)] == [
        "event S3 verified",
        "prompt S2",
    ]
    assert all(o.step_id == "S2" for o in new if isinstance(o, Ask))  # blindness ask only


def test_unanswered_ask_is_superseded_after_15s():
    dr = _s1_done()
    (a1,) = dr.run(3.1, S2=0.62)  # asked on the last-but-one frame
    new = dr.run(13.0, S4=0.62)
    assert [o for o in new if isinstance(o, Ask)] == []  # S2's ask still fresh
    new = dr.run(2.2)
    asks = [o for o in new if isinstance(o, Ask)]
    assert [a.step_id for a in asks] == ["S4"]
    assert asks[0].t - a1.t == pytest.approx(15.0, abs=0.11)
    assert dr.answer(a1.question_id, "yes") == []  # superseded: closed


def test_commands():
    dr = _s1_done()
    assert d(dr.command("whats_next")) == ["prompt S2"]
    assert d(dr.command("next")) == ["prompt S2"]  # never marks a step done
    assert dr.engine.statuses["S2"] is StepStatus.PENDING
    assert d(dr.command("repeat")) == ["prompt S2"]
    assert dr.command("confirm") == [] and dr.command("cancel") == []  # no open ask
    (ask,) = dr.run(3.1, S2=0.62)
    rep = dr.command("repeat")
    assert d(rep) == ["ask S2"] and rep[0].question_id == ask.question_id
    assert d(dr.command("cancel")) == []
    assert dr.engine.open_ask is None
    assert dr.answer(ask.question_id, "yes") == []
    assert d(dr.run(12.9)) == []  # cancel cools down like "no"
    (ask2,) = dr.run(0.2)
    assert d(dr.command("confirm")) == ["event S2 operator_confirmed", "prompt S3"]
    assert ask2.question_id == "q2-S2"


def test_inputs_before_the_first_frame_are_ignored():
    e = ProtocolEngine(PROTO)
    assert e.answer(Answer("q1-S1", "yes", "gui", t=0.0)) == []
    assert e.command(Command("repeat", "gui", t=0.0)) == []
    assert e.finish() == []


# -- out of order, skip --------------------------------------------------------------------------


def test_unsure_predecessor_is_not_accused_until_clearly_low():
    dr = _s1_done()
    new = dr.run(2.0, S2=0.6, S3=0.93)  # S2 unsure (in band, < 3 s so no ask yet)
    assert d(new) == ["event S3 verified", "prompt S2"]
    assert d(dr.run(0.7, S2=LOW)) == []  # low, but not yet for dwell_s
    new = dr.run(0.2)
    assert d(new) == ["alert out_of_order S3,S2"]
    assert new[0].message == (
        "Out of sequence: red box placed in Zone A. Expected: take out the red box."
    )


def test_skip_is_finalised_by_finish_and_late_completion_is_out_of_order():
    dr = _s1_done()
    new = dr.do("S3", 1.0)
    assert d(new) == ["event S3 out_of_order", "alert out_of_order S3,S2", "prompt S2"]
    fin = dr.engine.finish()
    assert d(fin) == ["event S2 skipped (retro)", "alert skip S2"]
    assert fin[0].retroactive and fin[0].t_start is None and fin[0].t_end is None
    statuses = dr.engine.statuses
    assert statuses["S2"] is StepStatus.SKIPPED
    assert {statuses[s] for s in ("S4", "S5", "S6")} == {StepStatus.PENDING}
    assert dr.engine.finish() == []  # nothing new to finalise
    # the operator does S2 after all: late, no new alert
    dr.run(0.1)
    assert d(dr.do("S2")) == ["event S2 out_of_order", "prompt S4"]


def test_jumping_to_the_last_step_skips_the_rest_without_a_double_alert():
    dr = _s1_done()
    new = dr.run(1.0, S1=LOW, S6=0.93)  # closing undoes S1 legitimately
    assert d(new) == [
        "event S6 verified",
        "event S2 skipped (retro)",
        "event S3 skipped (retro)",
        "event S4 skipped (retro)",
        "event S5 skipped (retro)",
        "alert skip S2,S3,S4,S5",
        "prompt -",
    ]
    assert dr.engine.finished


def test_partial_order_allows_either_branch_first():
    data = PROTO.model_dump()
    data["order"] = [
        ["S1", "S2"],
        ["S2", "S3"],
        ["S1", "S4"],
        ["S4", "S5"],
        ["S3", "S6"],
        ["S5", "S6"],
    ]
    proto = Protocol.model_validate(data)
    dr = Drive(proto)
    dr.run(0.5)
    dr.do("S1")
    new = dr.do("S4") + dr.do("S5")
    assert d(new) == ["event S4 verified", "prompt S2", "event S5 verified", "prompt S2"]
    # S6 is the only sink: doing it now leaves nothing else to do, so S2/S3 are skipped
    assert d(dr.run(1.0, S1=LOW, S6=0.93)) == [
        "event S6 verified",
        "event S2 skipped (retro)",
        "event S3 skipped (retro)",
        "alert skip S2,S3",
        "prompt -",
    ]


# -- unexpected ----------------------------------------------------------------------------------


def test_undo_of_a_done_step_is_unexpected_and_reopens_it():
    dr = _s1_done()
    dr.do("S2")
    dr.do("S3")
    new = dr.run(1.0, S3=LOW)
    assert d(new) == ["event S3 pending", "alert unexpected S3", "prompt S3"]
    assert new[0].confidence == pytest.approx(1 - LOW)
    assert new[1].message == "Unexpected: red box moved out of Zone A."
    new = dr.run(1.0, S2=LOW)  # same object: same incident, no second alert
    assert d(new) == ["event S2 pending", "prompt S2"]
    new = dr.do("S2") + dr.do("S3")
    assert d(new) == ["event S2 verified", "prompt S3", "event S3 verified", "prompt S4"]
    # the incident is closed: a new undo alerts again
    assert d(dr.run(1.0, S3=LOW))[1] == "alert unexpected S3"


def test_undo_while_blind_is_not_unexpected():
    dr = _s1_done()
    dr.do("S2")
    assert d(dr.run(3.0, vis=0.1, S2=LOW)) == ["ask S3"]  # blindness ask only
    assert dr.engine.statuses["S2"] is StepStatus.VERIFIED


def test_reopening_the_box_after_closing_is_unexpected_but_closing_is_not():
    dr = _s1_done()
    for s in ("S2", "S3", "S4", "S5"):
        dr.do(s)
    new = dr.run(1.0, S1=LOW, S6=0.93)
    assert d(new) == ["event S6 verified", "prompt -"]
    assert dr.engine.finished
    new = dr.run(1.0, S1=0.93, S6=LOW)
    assert d(new) == ["event S6 pending", "alert unexpected S6", "prompt S6"]
    assert new[1].message == "Unexpected: outer box opened."
    assert not dr.engine.finished


def test_reopened_step_is_not_flagged_missing_against_earlier_completions():
    dr = _s1_done()
    for s in ("S2", "S3", "S4", "S5"):
        dr.do(s)
    new = dr.run(2.0, S2=LOW, S3=LOW)  # red box goes back in after S5
    assert [o.kind for o in new if isinstance(o, Alert)] == [AlertKind.UNEXPECTED]
    assert dr.engine.prompt_target == "S2"


# -- timing --------------------------------------------------------------------------------------


def test_timeout_fires_once_per_wait_and_resets_on_progress():
    dr = Drive(timeout_s=5.0)
    new = dr.run(5.1)
    assert d(new) == ["prompt S1", "event S1 timed_out", "alert timeout S1"]
    assert new[2].message == "Timeout: no progress for 5 seconds. Next step: open the outer box."
    assert d(dr.run(20.0)) == []
    assert d(dr.do("S1")) == ["event S1 verified", "prompt S2"]
    # S1 was decided 0.8 s into do(); 0.1 + 4.5 s later is still inside the 5 s budget
    assert d(dr.run(4.5, S1=0.93)) == []
    assert d(dr.run(0.5)) == ["event S2 timed_out", "alert timeout S2"]


def test_no_timeout_after_the_procedure_is_finished():
    dr = Drive(timeout_s=5.0)
    dr.run(0.2)
    for s in PROTO.step_ids:
        dr.do(s)
        dr.p[s] = 0.93
    dr.p["S1"] = LOW
    assert dr.engine.finished
    assert d(dr.run(20.0)) == []


# -- a second protocol from YAML alone -----------------------------------------------------------


def test_messages_for_another_protocol():
    proto = Protocol.model_validate(
        {
            "id": "mix",
            "title": "Sample mixing",
            "objects": {
                "cup": {"label": "cup", "classes": ["cup"]},
                "bottle": {"label": "bottle", "classes": ["bottle"]},
            },
            "zones": {"W": [[0.1, 0.1], [0.3, 0.3]]},
            "steps": [
                {
                    "id": "M1",
                    "text": "Place the cup in the work zone",
                    "post": {"cup.in_zone_W": True},
                },
                {"id": "M2", "text": "Open the bottle", "post": {"bottle.open": True}},
                {"id": "M3", "text": "Pour into the cup", "post": {"bottle.poured": True}},
            ],
        }
    )
    dr = Drive(proto)
    dr.run(0.5)
    new = dr.do("M2")
    alert = next(o for o in new if isinstance(o, Alert))
    assert alert.message == (
        "Out of sequence: bottle opened before cup. Expected: place the cup in the work zone."
    )
    dr.run(0.5, M2=0.93)
    new = dr.do("M1") + dr.do("M3")
    assert d(new) == ["event M1 verified", "prompt M3", "event M3 verified", "prompt -"]
    new = dr.run(1.0, M3=LOW)
    assert [o.message for o in new if isinstance(o, Alert)] == [
        "Unexpected: bottle no longer poured."
    ]
    assert isinstance(new[-1], Prompt) and new[-1].next_step_id == "M3"
    assert all(isinstance(o, (StepEvent, Alert, Prompt, Ask)) for o in dr.out)
