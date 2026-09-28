"""T-FAULT: fault injection and the safety net that answers it.

Layers, cheapest first: each fault's effect on a frame and its detections, the stale-frame guard,
the controller against a fake core (requests, schedules, stage/cap/tag bookkeeping, the network
target, the hash-chained fault log), the safety oracle on synthetic outputs, then the real
pipeline on the SIMULATED replay -- the ``visibility_cap`` hook, a scheduled fault through
``run_offline`` and ``LiveApp``, and the demo that runs every fault against ground truth.
"""

import dataclasses
import json
import types

import numpy as np
import pytest

from actus.app.faults import (
    _FAULT_CLASSES,
    DEMO_PLAN,
    FAULT_LOG,
    FAULTS,
    FaultController,
    HotkeyListener,
    SimulatedTarget,
    StaleFrameGuard,
    _Work,
    build_fault,
    evaluate_safety,
    is_blinding,
    ScheduledFault,
    parse_schedule_item,
    run_fault_demo,
    truth_completion,
)
from actus.app.pipeline import (
    LiveApp,
    PipelineCore,
    ReplayFrameSource,
    SourceFrame,
    cap_visibility,
    run_offline,
)
from actus.capture.replay import get_scenario
from actus.core.config import load_config
from actus.core.contracts import (
    Ask,
    Detection,
    DetectionSet,
    FramePacket,
    Predicates,
    StepEvent,
    StepEvidence,
    StepStatus,
)
from actus.io.eventlog import verify_log
from actus.protocol.schema import load_protocol


def _cfg(tmp_path, *extra):
    return load_config(
        overrides=[
            f"run.runs_dir={tmp_path.as_posix()}",
            "models.enabled=false",
            "faults.hotkeys=false",
            *extra,
        ]
    )


def _img(value=120):
    return np.full((48, 64, 3), value, np.uint8)


def _dets(fi=0, t=0.0):
    boxes = [
        Detection(1, "red_box", 0.9, (2, 2, 12, 12)),  # left half
        Detection(2, "yellow_box", 0.9, (40, 30, 60, 46)),  # right half
    ]
    return DetectionSet(fi, t, boxes, "test")


def _sf(fi, *, dets=True, vary=True):
    image = _img(10 + fi % 200) if vary else _img()
    packet = FramePacket(fi, fi / 15.0, image, "test")
    return SourceFrame(packet, detections=_dets(fi, fi / 15.0) if dets else None)


def _core():
    return types.SimpleNamespace(
        source_hooks=[],
        disabled=set(),
        tags=set(),
        visibility_cap=None,
        simulated=False,
        evidence_mode="spatial",
    )


def _feed(ctrl, frames):
    return [ctrl.process(sf) for sf in frames]


# --------------------------------------------------------------------------------------------
# The catalogue, schedules, blinding
# --------------------------------------------------------------------------------------------


def test_the_catalogue_is_the_five_faults_on_the_card():
    assert list(FAULTS) == ["occlude", "dim", "freeze", "drop_detector", "kill_network"]
    assert set(_FAULT_CLASSES) == set(FAULTS)
    assert len({f.key for f in FAULTS.values()}) == len(FAULTS)
    assert all(f.expected and f.summary and f.label for f in FAULTS.values())
    assert [n for n, _ in DEMO_PLAN] == list(FAULTS)


def test_schedule_items_parse_and_round_trip():
    s = parse_schedule_item("occlude@9-12")
    assert (s.name, s.start, s.end, s.unit) == ("occlude", 9.0, 12.0, "s")
    assert s.contains(9.0) and s.contains(11.99) and not s.contains(12.0)

    d = parse_schedule_item("dim@f90-135:gamma=4")
    assert d.unit == "f" and (d.start, d.end) == (90.0, 135.0) and d.params == {"gamma": 4}

    o = parse_schedule_item("occlude@1-:region=0,0,0.5,1")
    assert o.end == float("inf") and o.params == {"region": [0.0, 0.0, 0.5, 1.0]}
    assert parse_schedule_item("occlude@1-end").end == float("inf")

    again = parse_schedule_item(s.spec())
    assert (again.name, again.start, again.end) == (s.name, s.start, s.end)

    # Lists and None params survive the round trip too.
    assert parse_schedule_item(o.spec()).params == o.params
    none_param = ScheduledFault("dim", 1.0, 2.0, "s", {"gamma": 4, "min_conf": None})
    assert parse_schedule_item(none_param.spec()).params == {"gamma": 4}


@pytest.mark.parametrize(
    ("spec", "error"),
    [
        ("occlude", ValueError),  # no window
        ("occlude@12-9", ValueError),  # backwards
        ("occlude@9", ValueError),  # no dash
        ("occlude@1-2:gamma", ValueError),  # not key=value
        ("occlude@1-2:bogus=1", ValueError),  # not a parameter of this fault
        ("nope@1-2", KeyError),  # not a fault
        ("dim@9-12:gamma=", ValueError),  # empty value (was a TypeError)
        ("dim@9-12:gamma=[1", ValueError),  # unparseable value (was a YAMLError)
    ],
)
def test_bad_schedule_items_are_refused_up_front(spec, error):
    with pytest.raises(error):
        parse_schedule_item(spec)


def test_only_blinding_faults_are_held_to_the_no_fabrication_rule():
    assert is_blinding("occlude", FAULTS["occlude"].defaults)
    assert not is_blinding("occlude", {"region": [0.0, 0.0, 0.3, 0.3]})
    assert is_blinding("dim", FAULTS["dim"].defaults)
    assert not is_blinding("dim", {"gamma": 1.5})
    assert is_blinding("freeze", {}) and is_blinding("drop_detector", {})
    assert not is_blinding("kill_network", {})


# --------------------------------------------------------------------------------------------
# What each fault does to a frame
# --------------------------------------------------------------------------------------------


def _work(img=None, dets=None, *, prev=None, had_dets=True, fi=5, t=0.33):
    prev_img, prev_dets = prev if prev is not None else (None, None)
    return _Work(_img() if img is None else img, dets, fi, t, prev_img, prev_dets, had_dets)


def test_occlusion_blacks_out_the_region_and_drops_what_it_covers():
    src = _img()
    w = _work(src, _dets())
    build_fault("occlude", {"region": [0, 0, 0.5, 1]}).apply(w)
    assert w.img[:, :32].max() <= 3  # a covered lens is dark, with sensor noise
    assert (w.img[:, 32:] == 120).all()
    assert (src == 120).all()  # the input frame is not modified
    assert [d.cls for d in w.dets.detections] == ["yellow_box"]


def test_a_full_frame_occlusion_is_not_bit_identical_frame_to_frame():
    """Otherwise the stale-frame guard would mistake a blackout for a frozen camera."""
    fault = build_fault("occlude")
    a, b = _work(fi=1), _work(fi=2)
    fault.apply(a)
    fault.apply(b)
    assert not np.array_equal(a.img, b.img)


def test_occlusion_leaves_a_real_detectors_output_alone():
    w = _work(dets=None, had_dets=False)
    build_fault("occlude").apply(w)
    assert w.dets is None  # the black box is in the pixels; the real detector sees it itself


def test_dimming_darkens_the_frame_and_thins_the_detections():
    w = _work(_img(128), _dets())
    build_fault("dim", {"gamma": 2.0}, detector_conf=0.25).apply(w)
    assert (w.img == 64).all()
    assert len(w.dets.detections) == 2 and w.dets.detections[0].conf == pytest.approx(0.45)

    w = _work(_img(128), _dets())
    build_fault("dim", {"gamma": 4.0}, detector_conf=0.25).apply(w)  # 0.9 * 0.125 < 0.25
    assert len(w.dets.detections) == 0


def test_freezing_repeats_the_last_frame_and_its_detections_restamped():
    prev_img, prev_dets = _img(50), _dets(4, 0.27)
    w = _work(_img(200), _dets(5, 0.33), prev=(prev_img, prev_dets), fi=5, t=0.33)
    build_fault("freeze").apply(w)
    assert w.img is prev_img
    assert w.dets.frame_idx == 5 and w.dets.t == 0.33
    assert w.dets.detections == prev_dets.detections

    first = _work(_img(200), _dets())  # nothing delivered yet: nothing to hold on to
    build_fault("freeze").apply(first)
    assert first.img.max() == 200


def test_dropping_the_detector_reports_did_not_run():
    fault = build_fault("drop_detector")
    w = _work(dets=_dets())
    fault.apply(w)
    assert w.dets is None
    assert fault.stages == {"detector"} and fault.cap == 0.0


def test_bad_fault_parameters_are_refused():
    with pytest.raises(ValueError):
        build_fault("occlude", {"region": [0.6, 0, 0.4, 1]})
    with pytest.raises(ValueError):
        build_fault("dim", {"gamma": 0})
    with pytest.raises(ValueError):
        build_fault("freeze", {"speed": 2})
    with pytest.raises(KeyError):
        build_fault("nope")


# --------------------------------------------------------------------------------------------
# The stale-frame guard
# --------------------------------------------------------------------------------------------


def test_guard_trips_after_n_repeats_and_clears_on_a_fresh_frame():
    g = StaleFrameGuard(stale_after=3)
    a = _img(1)
    assert [g.update(a) for _ in range(5)] == [False, False, False, True, True]
    assert g.update(_img(2)) is False and g.repeats == 0

    g = StaleFrameGuard(stale_after=3)
    assert [g.update(a.copy()) for _ in range(4)] == [False, False, False, True]  # equal pixels

    with pytest.raises(ValueError):
        StaleFrameGuard(0)


# --------------------------------------------------------------------------------------------
# The controller, against a fake core
# --------------------------------------------------------------------------------------------


def test_a_scheduled_window_opens_and_closes_on_the_frames_own_clock():
    core = _core()
    ctrl = FaultController(schedule=["occlude@1-2"]).attach(core)
    frames = [_sf(i) for i in range(45)]  # 15 fps: the window is frames 15..29
    out = _feed(ctrl, frames[:15])
    assert all(o is f for o, f in zip(out, frames[:15], strict=True))  # untouched, same object
    assert core.tags == set()

    out = _feed(ctrl, frames[15:30])
    assert all(o.packet.image.max() <= 3 and not o.detections.detections for o in out)
    assert core.tags == {"FAULT INJECTED: camera occluded"}
    assert ctrl.stats().values["injected"] is True

    _feed(ctrl, frames[30:])
    assert core.tags == set() and ctrl.stats().values["injected"] is False
    (window,) = ctrl.windows()
    assert (window["frame_start"], window["frame_end"]) == (15, 30)
    assert (window["t_start"], window["t_end"]) == (1.0, 2.0)
    assert window["blinding"] is True and window["source"] == "schedule"
    assert [r["action"] for r in ctrl.records()] == ["start", "stop"]


def test_a_frame_index_schedule_ignores_the_clock():
    ctrl = FaultController(schedule=["freeze@f3-6"]).attach(_core())
    _feed(ctrl, [_sf(i) for i in range(10)])
    (window,) = ctrl.windows()
    assert (window["frame_start"], window["frame_end"]) == (3, 6)


def test_ready_made_schedule_objects_are_validated_up_front():
    with pytest.raises(ValueError):
        FaultController(schedule=[ScheduledFault("freeze", 9.0, 12.0, "s", {"gamma": 4})])
    with pytest.raises(ValueError):
        FaultController(schedule=[ScheduledFault("dim", 12.0, 9.0)])
    with pytest.raises(KeyError):
        FaultController(schedule=[ScheduledFault("nope", 1.0, 2.0)])


def test_a_manual_off_overrides_a_scheduled_window():
    core = _core()
    ctrl = FaultController(schedule=["dim@f2-100"]).attach(core)
    _feed(ctrl, [_sf(i) for i in range(4)])
    assert ctrl.state()["active"] == ["dim"]
    ctrl.request("dim")  # a toggle of an active (scheduled) fault turns it off
    _feed(ctrl, [_sf(i) for i in range(4, 8)])
    assert ctrl.state()["active"] == []
    ctrl.request("dim", True)
    ctrl.request("dim", False)  # explicit states apply in order
    _feed(ctrl, [_sf(8)])
    assert ctrl.state()["active"] == []


def test_toggles_within_one_frame_count_once():
    ctrl = FaultController().attach(_core())
    for _ in range(4):  # a held-down key
        ctrl.request("dim")
    _feed(ctrl, [_sf(0)])
    assert ctrl.state()["active"] == ["dim"]


def test_new_params_restart_an_active_fault():
    ctrl = FaultController().attach(_core())
    ctrl.request("dim", True, {"gamma": 3.0})
    _feed(ctrl, [_sf(0)])
    ctrl.request("dim", True, {"gamma": 1.5})
    _feed(ctrl, [_sf(1)])
    windows = ctrl.windows()
    assert [w["params"]["gamma"] for w in windows] == [3.0, 1.5]
    assert windows[0]["frame_end"] == 1 and windows[1]["frame_end"] is None


def test_a_controller_attached_to_a_new_core_starts_a_new_run():
    ctrl = FaultController(schedule=["dim@f0-2"]).attach(_core())
    _feed(ctrl, [_sf(i) for i in range(4)])
    assert len(ctrl.windows()) == 1
    ctrl.close()
    ctrl.attach(_core())
    assert ctrl.windows() == [] and ctrl.records() == []
    _feed(ctrl, [_sf(i) for i in range(10, 14)])  # the schedule is relative to this run
    assert [w["frame_start"] for w in ctrl.windows()] == [10]


def test_a_failed_network_cut_ends_the_window_instead_of_claiming_an_outage():
    class Broken:
        instant = False

        def cut(self):
            raise OSError("streamer would not stop")

        def restore(self):
            pass

        def status(self):
            return {}

    ctrl = FaultController().attach(_core())
    ctrl.set_network_target(Broken())
    ctrl.request("kill_network", True)
    ctrl.process(_sf(0))
    ctrl._pool.shutdown(wait=True)  # let the worker fail
    ctrl._pool = None
    ctrl.process(_sf(1))
    (window,) = ctrl.windows()
    assert window["frame_end"] == 1
    entry = next(f for f in ctrl.state()["faults"] if f["name"] == "kill_network")
    assert entry["active"] is False and "cut failed" in entry["refused"]


def test_requests_take_effect_at_the_next_frame_and_toggle():
    core = _core()
    ctrl = FaultController().attach(core)
    ctrl.request("dim")
    bright = SourceFrame(FramePacket(0, 0.0, _img(200), "test"))
    out = ctrl.process(bright)
    assert out.packet.image.max() < 200  # gamma 3: 200 -> 123
    assert core.tags == {"FAULT INJECTED: lighting dimmed"}

    ctrl.request("dim")  # toggles off
    sf = _sf(1)
    assert ctrl.process(sf) is sf and core.tags == set()
    assert [w["source"] for w in ctrl.windows()] == ["manual"]


def test_requests_are_validated_in_the_callers_thread():
    ctrl = FaultController()
    with pytest.raises(KeyError):
        ctrl.request("nope")
    with pytest.raises(ValueError):
        ctrl.request("dim", params={"gamma": -1})
    assert ctrl.stats().values["pending"] == 0


def test_dropping_the_detector_switches_the_stage_off_and_caps_visibility():
    core = _core()
    core.disabled.add("models")  # someone else's switch must survive
    ctrl = FaultController().attach(core)
    ctrl.request("drop_detector", True)
    out = ctrl.process(_sf(0))
    assert out.detections is None  # a source-supplied stand-in is "did not run" too
    assert core.disabled == {"models", "detector"} and core.visibility_cap == 0.0

    ctrl.request("drop_detector", False)
    sf = _sf(1)
    assert ctrl.process(sf) is sf
    assert core.disabled == {"models"} and core.visibility_cap is None

    core.disabled.add("detector")  # already off for another reason: not ours to switch back on
    ctrl.request("drop_detector", True)
    ctrl.process(_sf(2))
    ctrl.request("drop_detector", False)
    ctrl.process(_sf(3))
    assert "detector" in core.disabled


def test_a_freeze_is_caught_by_the_guard_and_released_when_frames_move_again():
    core = _core()
    ctrl = FaultController(stale_after=3).attach(core)
    _feed(ctrl, [_sf(i) for i in range(5)])
    ctrl.request("freeze", True)
    out = _feed(ctrl, [_sf(i) for i in range(5, 13)])

    assert out[0].packet.image is out[1].packet.image  # the camera repeats itself
    assert out[0].detections.frame_idx == 5  # the held detections are restamped
    assert out[2].detections is None  # the guard tripped on the 3rd repeat: detector suspended
    assert core.disabled == {"detector", "handpose"} and core.visibility_cap == 0.0
    assert core.tags == {"FAULT INJECTED: frame frozen", "FAULT DETECTED: stale frames"}
    assert ctrl.stats().values["detected"] == "stale_frames"

    ctrl.request("freeze", False)
    out = ctrl.process(_sf(13))
    assert out.detections is not None
    assert core.disabled == set() and core.visibility_cap is None and core.tags == set()
    assert [r["action"] for r in ctrl.records()] == ["start", "detected", "stop", "cleared"]


def test_the_guard_can_be_switched_off():
    ctrl = FaultController(stale_after=0).attach(_core())
    ctrl.request("freeze", True)
    _feed(ctrl, [_sf(i) for i in range(10)])
    assert ctrl.stats().values["detected"] == "none"


def test_kill_network_is_refused_without_a_target_and_logged_once():
    core = _core()
    ctrl = FaultController().attach(core)
    ctrl.request("kill_network", True)
    _feed(ctrl, [_sf(i) for i in range(4)])
    assert core.tags == set() and ctrl.windows() == []
    (rec,) = ctrl.records()
    assert rec["action"] == "unavailable" and "streamer" in rec["reason"]
    entry = next(f for f in ctrl.state()["faults"] if f["name"] == "kill_network")
    assert entry["active"] is False and entry["refused"]

    ctrl.request("kill_network", False)  # asking again later is possible
    ctrl.process(_sf(4))
    entry = next(f for f in ctrl.state()["faults"] if f["name"] == "kill_network")
    assert entry["refused"] is None


def test_kill_network_cuts_and_restores_a_simulated_target_frame_exactly():
    ctrl = FaultController().attach(_core())
    target = SimulatedTarget()
    ctrl.set_network_target(target)
    _feed(ctrl, [_sf(i) for i in range(2)])
    ctrl.request("kill_network", True)
    _feed(ctrl, [_sf(i) for i in range(2, 5)])
    ctrl.request("kill_network", False)
    _feed(ctrl, [_sf(i) for i in range(5, 7)])
    assert (target.offered, target.delivered, target.dropped, target.cuts) == (7, 4, 3, 1)
    assert not target.down and ctrl.state()["network"]["state"] == "up"


def test_a_slow_target_is_cut_and_restored_in_order_off_the_pipeline_thread():
    class Slow:
        instant = False

        def __init__(self):
            self.calls = []

        def cut(self):
            self.calls.append("cut")

        def restore(self):
            self.calls.append("restore")

        def status(self):
            return {}

    target = Slow()
    ctrl = FaultController().attach(_core())
    ctrl.set_network_target(target)
    ctrl.request("kill_network", True)
    ctrl.process(_sf(0))
    ctrl.request("kill_network", False)
    ctrl.process(_sf(1))
    ctrl.close()  # waits for the worker
    assert target.calls == ["cut", "restore"]
    assert ctrl.state()["network"]["state"] == "up"


def test_close_ends_active_faults_and_hands_the_core_back_clean():
    core = _core()
    ctrl = FaultController().attach(core)
    ctrl.request("drop_detector", True)
    _feed(ctrl, [_sf(0), _sf(1)])
    summary = ctrl.close()
    assert core.disabled == set() and core.visibility_cap is None and core.tags == set()
    assert summary["windows"][0]["frame_end"] == 1
    assert ctrl.records()[-1]["reason"] == "end_of_run"
    ctrl.close()  # twice is fine


def test_the_hotkeys_toggle_by_the_keys_on_the_buttons():
    ctrl = FaultController()
    keys = HotkeyListener(ctrl)
    keys._handle("O")
    keys._handle("z")  # not a fault key
    assert ctrl.stats().values["pending"] == 1
    ctrl.process(_sf(0))
    assert ctrl.stats().values["active"] == "occlude"


# --------------------------------------------------------------------------------------------
# The fault log
# --------------------------------------------------------------------------------------------


def _scheduled_run(tmp_path, name):
    ctrl = FaultController(schedule=["occlude@1-2", "freeze@2.5-3"]).attach(_core())
    ctrl.open_log(tmp_path / name, clock=lambda: 1_790_000_000.0, tip=lambda: "f" * 64)
    _feed(ctrl, [_sf(i) for i in range(60)])
    ctrl.close()
    return tmp_path / name / FAULT_LOG


def test_the_fault_log_is_a_verifiable_hash_chain_and_deterministic(tmp_path):
    a = _scheduled_run(tmp_path, "a")
    assert verify_log(a).ok
    recs = [json.loads(line) for line in a.read_text(encoding="utf-8").splitlines()]
    assert recs[0]["type"] == "FaultEvent" and recs[0]["events_tip"] == "f" * 64
    assert recs[0]["ts"].startswith("2026-") and recs[0]["expected"]
    assert {r["fault"] for r in recs} >= {"occlude", "freeze"}
    assert a.read_bytes() == _scheduled_run(tmp_path, "b").read_bytes()

    a.write_text(
        a.read_text(encoding="utf-8").replace('"action":"start"', '"action":"stxrt"', 1),
        encoding="utf-8",
    )
    assert not verify_log(a).ok


# --------------------------------------------------------------------------------------------
# The safety oracle
# --------------------------------------------------------------------------------------------


def _event(step, status, t):
    return StepEvent(step, status, None, None, 0.9, frame_idx=int(t * 15), t=t)


BLIND = {"fault": "occlude", "blinding": True, "t_start": 9.0, "t_end": 12.0}
TRUTH = {"S2": 7.0, "S3": 10.0, "S4": None}


def test_a_verification_inside_a_blind_window_of_something_that_happened_in_it_is_fabricated():
    out = [_event("S3", StepStatus.VERIFIED, 10.9)]
    res = evaluate_safety(out, truth=TRUTH, windows=[BLIND])
    assert [f["step_id"] for f in res["fabricated"]] == ["S3"]


@pytest.mark.parametrize(
    ("event", "why"),
    [
        (_event("S3", StepStatus.VERIFIED, 12.5), "decided after the window"),
        (_event("S2", StepStatus.VERIFIED, 9.5), "completed before the window"),
        (_event("S3", StepStatus.OPERATOR_CONFIRMED, 11.0), "the operator, not the sensors"),
    ],
)
def test_legitimate_outcomes_are_not_flagged(event, why):
    assert not evaluate_safety([event], truth=TRUTH, windows=[BLIND])["fabricated"], why


def test_only_blinding_windows_count_and_never_performed_steps_are_always_false():
    mild = {**BLIND, "blinding": False}
    out = [_event("S3", StepStatus.VERIFIED, 10.9)]
    assert not evaluate_safety(out, truth=TRUTH, windows=[mild])["fabricated"]

    out = [_event("S4", StepStatus.VERIFIED, 3.0)]
    assert evaluate_safety(out, truth=TRUTH, windows=[])["false_verified"] == ["S4"]


def test_steps_in_the_window_are_reported_with_how_they_ended():
    out = [
        Ask("q1-S3", "S3", "Did you place it?", frame_idx=150, t=10.0),
        _event("S3", StepStatus.OPERATOR_CONFIRMED, 11.0),
    ]
    outcomes = [{"step_id": "S3", "status": "operator_confirmed"}]
    rows = evaluate_safety(out, truth=TRUTH, windows=[BLIND], outcomes=outcomes)["steps_in_window"]
    assert rows == [
        {
            "fault": "occlude",
            "step_id": "S3",
            "truth_t": 10.0,
            "final_status": "operator_confirmed",
            "asked": True,
            "decided_as": "operator_confirmed",
            "decided_t": 11.0,
        }
    ]


def test_truth_completion_reads_the_replay_scripts():
    proto = load_protocol("protocols/bas_boxes.yaml")
    correct = truth_completion(get_scenario("correct"), proto)
    assert correct["S1"] == pytest.approx(3.75) and correct["S3"] == pytest.approx(10.0)
    skipped = truth_completion(get_scenario("skip_S2"), proto)
    assert skipped["S2"] is None and skipped["S3"] is None and skipped["S4"] is not None


# --------------------------------------------------------------------------------------------
# The pipeline hooks the faults use
# --------------------------------------------------------------------------------------------


def test_cap_visibility_pulls_evidence_toward_uninformative():
    ev = (StepEvidence(3, 0.2, "S1", 0.9, {"rule": 0.9}),)
    pred = Predicates(3, 0.2, {"outer.open": 0.9}, 0.95)
    half, p_half = cap_visibility(ev, pred, 0.5)
    assert half[0].p_complete == pytest.approx(0.7)
    assert half[0].sources["rule"] == pytest.approx(0.7)
    assert p_half.visibility == 0.5 and p_half.values["outer.open"] == pytest.approx(0.7)
    blind, p_blind = cap_visibility(ev, pred, 0.0)
    assert blind[0].p_complete == 0.5 and p_blind.visibility == 0.0
    _, p_clear = cap_visibility(ev, pred, 1.0)
    assert p_clear.visibility == 0.95  # a cap never raises visibility


def _replay_core(tmp_path):
    return PipelineCore.from_config(
        _cfg(tmp_path), detector=None, handpose=None, models=None, simulated=True
    )


def test_source_hooks_run_before_perception(tmp_path):
    core = _replay_core(tmp_path)
    first = next(iter(ReplayFrameSource("correct")))
    assert core.step(first).detections is not None  # the replay supplies its stand-in
    core = _replay_core(tmp_path)
    core.source_hooks.append(lambda sf: dataclasses.replace(sf, detections=None))
    assert core.step(first).detections is None


def test_a_capped_pipeline_verifies_nothing_and_the_engine_asks(tmp_path):
    core = _replay_core(tmp_path)
    core.visibility_cap = 0.0
    results = [core.step(sf) for sf in ReplayFrameSource("correct")]
    outputs = [o for r in results for o in r.outputs]
    assert not [o for o in outputs if isinstance(o, StepEvent) and o.status is StepStatus.VERIFIED]
    assert all(r.predicates.visibility == 0.0 for r in results)
    assert all(e.p_complete == 0.5 for r in results for e in r.evidence)
    assert any(isinstance(o, Ask) for o in outputs)  # blind while a step is expected -> ask


# --------------------------------------------------------------------------------------------
# End to end on the SIMULATED replay (not run when this file was written -- see Vault/T-FAULT.md)
# --------------------------------------------------------------------------------------------


def test_scheduled_fault_runs_are_deterministic(tmp_path):
    cfg = _cfg(tmp_path)
    logs = []
    for name in ("a", "b"):
        ctrl = FaultController(schedule=["occlude@9-12"])
        res = run_offline("replay:correct", cfg, run_id=name, models=None, faults=ctrl, video=False)
        logs.append((res.events_path.read_bytes(), (res.run_dir / FAULT_LOG).read_bytes()))
    assert logs[0] == logs[1]
    assert verify_log(res.run_dir / FAULT_LOG).ok
    meta = json.loads((res.run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["faults"]["windows"][0]["fault"] == "occlude" and meta["simulated"] is True


def test_live_app_wires_the_controller_end_to_end(tmp_path):
    cfg = _cfg(tmp_path)
    ctrl = FaultController(schedule=["drop_detector@9-12"])
    app = LiveApp(
        cfg,
        source="replay",
        source_arg="correct",
        run_id="live",
        ui=False,
        voice=False,
        stream=False,
        record=False,
        realtime=False,
        faults=ctrl,
    )
    _, run_dir = app.run()
    assert verify_log(run_dir / FAULT_LOG).ok
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["faults"]["windows"][0]["fault"] == "drop_detector"
    assert "faults" in meta["sink_status"] and app.core.tags == set()


@pytest.mark.parametrize("fault", list(FAULTS))
def test_every_fault_demo_ends_safe_on_the_replay(fault, tmp_path):
    """The acceptance criterion: each fault, injected over S3's placement on a replayed take,
    ends with abstain-or-ask (or 'stream down, still recording') and never a false verified."""
    report = run_fault_demo(fault, cfg=_cfg(tmp_path), run_id=f"demo-{fault}")
    failed = [c for c in report["checks"] if not c["passed"]]
    assert report["passed"], failed
    assert report["simulated"] is True and report["expected"] == FAULTS[fault].expected
    run_dir = tmp_path / f"demo-{fault}"
    assert (run_dir / "faults_report.json").is_file() and (run_dir / "faults_report.md").is_file()


# --------------------------------------------------------------------------------------------
# The GUI routes
# --------------------------------------------------------------------------------------------


def test_fault_routes_list_toggle_and_reject():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from actus.app.server import create_app
    from actus.core.bus import Bus

    ctrl = FaultController()
    client = TestClient(create_app(Bus(), faults=ctrl, enable_demo=False))
    body = client.get("/api/faults").json()
    assert body["available"] is True and [f["name"] for f in body["faults"]] == list(FAULTS)
    assert client.post("/api/faults/dim?on=true").json()["queued"] is True
    assert ctrl.stats().values["pending"] == 1
    assert client.post("/api/faults/nope").status_code == 404
    assert client.post("/api/faults/dim?on=maybe").status_code == 422

    plain = TestClient(create_app(Bus(), enable_demo=False))
    assert plain.get("/api/faults").status_code == 404
