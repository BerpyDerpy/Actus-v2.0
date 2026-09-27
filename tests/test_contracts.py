import json
import math

import numpy as np
import pytest

from actus.core import contracts as c


def _roundtrip(obj):
    data = c.to_jsonable(obj)
    text = json.dumps(data, allow_nan=False)  # must be strict JSON
    return c.from_jsonable(json.loads(text))


def test_every_topic_is_named_by_the_plan():
    plan = {
        "frames",
        "detections",
        "hands",
        "pose",
        "rack",
        "predicates",
        "evidence",
        "actions",
        "events",
        "alerts",
        "prompts",
        "asks",
        "answers",
        "stats",
    }
    assert plan <= {t.value for t in c.Topic}


def test_step_status_values():
    assert {s.value for s in c.StepStatus} == {
        "verified",
        "operator_confirmed",
        "skipped",
        "out_of_order",
        "timed_out",
        "pending",
    }
    assert {k.value for k in c.AlertKind} == {
        "skip",
        "out_of_order",
        "unexpected",
        "timeout",
        "too_fast",
    }


def test_contracts_are_frozen():
    p = c.Predicates(0, 0.0, {"red.held": 0.5}, 1.0)
    with pytest.raises(AttributeError):
        p.visibility = 0.2  # type: ignore[misc]


def test_dicts_are_copied():
    values = {"red.held": 0.5}
    p = c.Predicates(0, 0.0, values, 1.0)
    values["red.held"] = 0.9
    assert p.values["red.held"] == 0.5


@pytest.mark.parametrize("bad", [-0.01, 1.01, math.nan])
def test_probabilities_are_validated(bad):
    with pytest.raises(ValueError):
        c.Predicates(0, 0.0, {"x.y": bad}, 1.0)
    with pytest.raises(ValueError):
        c.StepEvidence(0, 0.0, "S1", bad)
    with pytest.raises(ValueError):
        c.Detection(1, "red_box", bad, (0, 0, 1, 1))


@pytest.mark.parametrize("bad", [-1, 1.5, "3", True])
def test_frame_idx_is_validated(bad):
    with pytest.raises(ValueError):
        c.Predicates(bad, 0.0, {}, 1.0)


def test_time_must_be_finite():
    with pytest.raises(ValueError):
        c.StepEvidence(0, math.inf, "S1", 0.5)


def test_enum_coercion_from_strings():
    ev = c.StepEvent("S1", "verified", 1.0, 2.0, 0.9, {"rule": 0.9}, frame_idx=30, t=2.0)
    assert ev.status is c.StepStatus.VERIFIED
    with pytest.raises(ValueError):
        c.StepEvent("S1", "done", None, None, 0.9, frame_idx=0, t=0.0)
    ans = c.Answer("q1", "yes", "voice", t=3.0)
    assert ans.value is c.AnswerValue.YES and ans.via is c.InputVia.VOICE


def test_engine_outputs_require_clock_keywords():
    with pytest.raises(TypeError):
        c.Alert(c.AlertKind.SKIP, "Skipped", ("S2",))  # type: ignore[call-arg]
    a = c.Alert("skip", "Skipped: take out the red box", ["S2"], frame_idx=10, t=1.0)
    assert a.step_ids == ("S2",)


def test_arrays_are_shaped_readonly_copies():
    lm = np.zeros((21, 2))
    h = c.HandObs("left", lm, None, 0.9)
    assert h.landmarks_px.dtype == np.float32
    assert not h.landmarks_px.flags.writeable
    lm[0, 0] = 5
    assert h.landmarks_px[0, 0] == 0
    with pytest.raises(ValueError):
        c.HandObs("left", np.zeros((20, 2)), None, 0.9)
    with pytest.raises(ValueError):
        c.HandObs("middle", lm, None, 0.9)


def test_rotation_values():
    with pytest.raises(ValueError):
        c.PoseObs(np.zeros((33, 2)), np.zeros((33, 3)), None, 0.9, rotation_used=45)


def test_bbox_order_checked():
    with pytest.raises(ValueError):
        c.Detection(1, "red_box", 0.9, (10, 10, 5, 20))


def test_valid_rack_frame_needs_homography():
    with pytest.raises(ValueError):
        c.RackFrame(True, None, None, None, (0, 1, 2), 1.0, frame_idx=0, t=0.0)
    rf = c.RackFrame(False, None, None, None, (), None, frame_idx=0, t=0.0)
    assert rf.age_frames == 0


def test_frame_packet_is_not_json_encoded():
    pkt = c.FramePacket(0, 0.0, np.zeros((4, 4, 3), np.uint8), "cam0")
    with pytest.raises(TypeError):
        c.to_jsonable(pkt)
    with pytest.raises(ValueError):
        c.FramePacket(0, 0.0, np.zeros((4, 4, 3), np.uint8), "cam0", rotation_applied=30)


SAMPLES = [
    c.Detection(3, "red_box", 0.88, (1, 2, 3, 4), (0.6, 0.2)),
    c.DetectionSet(5, 0.33, [c.Detection(None, "lid", 0.5, (0, 0, 1, 1))], "open_vocab"),
    c.Predicates(7, 0.5, {"red.in_zone_A": 0.93, "outer.open": 0.1}, 0.97),
    c.StepEvidence(7, 0.5, "S3", 0.91, {"rule": 0.93, "hoi": 0.88}),
    c.ActionScores(7, 0.5, {"S1": 0.1, "S2": 0.7, "idle": 0.2}),
    c.StepEvent("S2", "skipped", None, None, 0.0, frame_idx=90, t=6.0, retroactive=True),
    c.Alert("out_of_order", "Out of sequence", ("S4", "S2"), frame_idx=100, t=6.6),
    c.Prompt(None, "All steps complete.", frame_idx=300, t=20.0),
    c.Ask("q7", "S3", "Did you place it?", frame_idx=150, t=10.0),
    c.Answer("q7", "no", "gui", t=11.0, frame_idx=165),
    c.Command("repeat", "voice", t=12.0),
    c.Stats("detector", {"fps": 17.5, "backend": "replay", "ok": True, "gpu": None}, t=1.0),
]


@pytest.mark.parametrize("obj", SAMPLES, ids=lambda o: type(o).__name__)
def test_json_roundtrip(obj):
    back = _roundtrip(obj)
    assert type(back) is type(obj)
    assert back == obj


def test_json_roundtrip_with_arrays():
    hand = c.HandObs("right", np.arange(42).reshape(21, 2), np.ones((21, 3)), 0.8)
    hs = c.HandSet(4, 0.2, [hand], rotation_used=180)
    back = _roundtrip(hs)
    assert back.rotation_used == 180
    np.testing.assert_allclose(back.hands[0].landmarks_px, hand.landmarks_px)
    np.testing.assert_allclose(back.hands[0].landmarks_rack, hand.landmarks_rack)

    pose = c.PoseObs(np.zeros((33, 2)), np.zeros((33, 3)), [0, -1, 0], 0.9, 90, np.ones(33))
    back = _roundtrip(c.PoseSet(4, 0.2, [pose]))
    np.testing.assert_allclose(back.poses[0].torso_vec_rack, [0, -1, 0])

    rf = c.RackFrame(
        True, np.eye(3), [0, 0, 0], [0, 0, 1], [0, 1, 3], math.nan, frame_idx=4, t=0.2, age_frames=2
    )
    back = _roundtrip(rf)
    assert back.valid and back.markers_seen == (0, 1, 3) and back.reproj_err is None
    np.testing.assert_allclose(back.H_img_to_rack, np.eye(3))


def test_every_registered_type_has_a_roundtrip_sample():
    covered = {type(o) for o in SAMPLES} | {c.HandObs, c.HandSet, c.PoseObs, c.PoseSet, c.RackFrame}
    assert set(c.contract_types()) <= covered
