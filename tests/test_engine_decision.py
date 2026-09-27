import pytest

from actus.protocol.decision import (
    Candidate,
    DecisionParams,
    DwellGate,
    SustainTimer,
    order_simultaneous,
)
from actus.protocol.schema import load_protocol

FPS = 10.0


def _feed(gate: DwellGate, ps: list[float]) -> list[bool]:
    return [gate.update(i / FPS, p, i) for i, p in enumerate(ps)]


def test_dwell_gate_fires_after_dwell_at_frame_resolution():
    g = DwellGate(0.8, 0.6, 0.8)
    fired = _feed(g, [0.1, 0.9] + [0.85] * 8)
    # crossed at frame 1 (t=0.1); 0.8 s later is frame 9 (t=0.9). 9/10 - 1/10 is not exactly 0.8
    assert fired == [False] * 9 + [True]
    assert g.t_cross == pytest.approx(0.1) and g.frame_cross == 1
    assert g.mean == pytest.approx((0.9 + 0.85 * 8) / 9)


def test_dwell_gate_hysteresis_holds_between_exit_and_enter():
    g = DwellGate(0.8, 0.6, 0.8)
    fired = _feed(g, [0.85, 0.7, 0.65, 0.61, 0.7, 0.62, 0.7, 0.75, 0.7, 0.7])
    assert fired[-2] is True  # held >= exit for 0.8 s without re-crossing enter


def test_dwell_gate_resets_below_exit_and_needs_a_new_crossing():
    g = DwellGate(0.8, 0.6, 0.8)
    fired = _feed(g, [0.9] * 5 + [0.55] + [0.79] * 10)
    assert not any(fired) and not g.active
    fired = _feed(g, [0.9] * 9)
    assert fired[-1]


def test_dwell_gate_zero_dwell_fires_on_crossing():
    g = DwellGate(0.8, 0.6, 0.0)
    assert _feed(g, [0.5, 0.81]) == [False, True]


def test_dwell_gate_rejects_bad_thresholds():
    with pytest.raises(ValueError):
        DwellGate(0.6, 0.8, 1.0)


def test_sustain_timer_fires_once_per_episode():
    s = SustainTimer(0.5)
    held = [s.update(i / FPS, c) for i, c in enumerate([True] * 8)]
    assert held.index(True) == 5
    s.fired = True
    assert s.update(0.8, True) and s.fired
    s.update(0.9, False)  # episode breaks
    assert not s.fired and s.t0 is None


def test_sustain_timer_grace_bridges_short_gaps():
    s = SustainTimer(1.0, grace_s=0.3)
    conds = [True] * 5 + [False] * 2 + [True] * 5  # 0.2 s gap < 0.3 s grace
    held = [s.update(i / FPS, c) for i, c in enumerate(conds)]
    assert held[5] is False and held[6] is False  # never "held" during the gap
    assert held.index(True) == 10  # measured from the episode start at t=0
    s2 = SustainTimer(1.0, grace_s=0.3)
    conds = [True] * 5 + [False] * 4 + [True] * 12  # 0.4 s gap breaks the episode
    held = [s2.update(i / FPS, c) for i, c in enumerate(conds)]
    assert held.index(True) == 19


def test_order_simultaneous_crossing_time_then_evidence_then_prior():
    a = Candidate("S3", t_cross=1.0, p=0.95, rank=2)
    b = Candidate("S2", t_cross=0.9, p=0.85, rank=1)
    assert [c.key for c in order_simultaneous([a, b], 0.05)] == ["S2", "S3"]  # earlier crossing
    a = Candidate("S3", t_cross=1.0, p=0.95, rank=2)
    b = Candidate("S2", t_cross=1.0, p=0.85, rank=1)
    assert [c.key for c in order_simultaneous([b, a], 0.05)] == ["S3", "S2"]  # clear evidence
    a = Candidate("S3", t_cross=1.0, p=0.88, rank=2)
    b = Candidate("S2", t_cross=1.0, p=0.86, rank=1)
    assert [c.key for c in order_simultaneous([a, b], 0.05)] == ["S2", "S3"]  # tie -> prior
    # the prior only reorders; it never drops or adds a candidate
    assert sorted(c.key for c in order_simultaneous([a, b], 0.05)) == ["S2", "S3"]


def test_params_from_protocol_and_overrides():
    proto = load_protocol("protocols/bas_boxes.yaml")
    pr = DecisionParams.from_protocol(proto)
    assert (pr.p_enter, pr.p_exit, pr.dwell_s) == (0.8, 0.6, 0.8)
    assert (pr.ask_low, pr.ask_high, pr.ask_after_s) == (0.45, 0.8, 3.0)
    assert pr.timeout_s == 60 and pr.min_duration_s == {"S3": 1.0}
    assert pr.p_undo_enter == pytest.approx(0.2) and pr.p_undo_exit == pytest.approx(0.4)
    pr2 = DecisionParams.from_protocol(proto, vis_ask_after_s=2.0)
    assert pr2.vis_ask_after_s == 2.0 and pr2.p_enter == 0.8
    with pytest.raises(KeyError):
        DecisionParams.from_protocol(proto, not_a_param=1)
    with pytest.raises(ValueError):
        DecisionParams(p_enter=0.6, p_exit=0.7)
    with pytest.raises(ValueError):
        DecisionParams(ask_low=0.5, ask_high=0.9)
