"""T-SPATIAL predicates and step evidence, end to end on the P0 replay and on synthetic scenes."""

import dataclasses

import numpy as np
import pytest

from actus.capture.replay import ReplaySource
from actus.core.contracts import (
    Detection,
    DetectionSet,
    HandObs,
    HandSet,
    PoseObs,
    PoseSet,
    Topic,
)
from actus.protocol.schema import load_protocol
from actus.spatial.estimator import SpatialEstimator
from actus.spatial.predicates import calibrate, fit_temperature
from actus.spatial.rack_frame import RackFrameEstimator, RackGeometry
from actus.spatial.world_state import SpatialParams

PROTO = load_protocol("protocols/bas_boxes.yaml")
P_ENTER = PROTO.decision.p_enter
STEPS = PROTO.step_ids


def run_replay(scenario, *, render=False, protocol=PROTO, fps=15.0):
    src = ReplaySource(scenario, protocol=PROTO, fps=fps, render=render)
    est = SpatialEstimator(protocol)
    frames = []
    for tick in src:
        dets = tick.detections
        if render:  # make the lane solve the rack and project detections itself
            dets = DetectionSet(
                dets.frame_idx,
                dets.t,
                [dataclasses.replace(d, bbox_rack_xy=None) for d in dets.detections],
                dets.backend,
            )
        frames.append((tick, est.update(tick.frame_idx, tick.t, frame=tick.frame, detections=dets)))
    return frames


def evidence_matrix(frames):
    return np.array([[e.p_complete for e in sf.evidence] for _, sf in frames])


def first_crossings(frames, thr=P_ENTER):
    E = evidence_matrix(frames)
    t = np.array([sf.t for _, sf in frames])
    out = {}
    for j, s in enumerate(STEPS):
        hit = np.nonzero(E[:, j] >= thr)[0]
        if len(hit):
            out[s] = float(t[hit[0]])
    return out


def order(crossings):
    return sorted(crossings, key=crossings.get)


# ---- replay scenarios ---------------------------------------------------------------------------


def test_correct_run_from_rendered_frames():
    """Full lane: ArUco on the SIMULATED renders → projection → world → evidence."""
    frames = run_replay("correct", render=True)
    assert all(sf.world.rack_state == "fresh" for _, sf in frames)
    # the solved rack matches the replay's own rack coordinates
    for tick, sf in frames[::25]:
        truth = {d.cls: d.bbox_rack_xy for d in tick.detections.detections}
        for d in sf.detections.detections:
            assert np.hypot(*np.subtract(d.bbox_rack_xy, truth[d.cls])) < 0.01
    assert order(first_crossings(frames)) == STEPS
    last = frames[-1][1]
    assert last.world.location("outer") == "closed"
    assert last.world.location("red") == "zone_A"
    assert last.world.location("yellow") == "zone_B"
    # every evidence crossing happens after the truth step started (no early completion)
    starts = {}
    for tick, _ in frames:
        starts.setdefault(tick.label, tick.t)
    for s, t in first_crossings(frames).items():
        assert t >= starts[s], (s, t, starts[s])


def test_skip_S2_never_evidences_S2_or_S3():
    frames = run_replay("skip_S2")
    E = evidence_matrix(frames)
    assert E[:, STEPS.index("S2")].max() < 0.5
    assert E[:, STEPS.index("S3")].max() < 0.5
    assert order(first_crossings(frames)) == ["S1", "S4", "S5", "S6"]
    assert frames[-1][1].world.location("red") == "in_outer"


def test_S4_before_S2_order_is_what_happened():
    assert order(first_crossings(run_replay("S4_before_S2"))) == [
        "S1",
        "S4",
        "S5",
        "S2",
        "S3",
        "S6",
    ]


def test_unexpected_return_undoes_and_redoes_S2_S3():
    frames = run_replay("unexpected")
    E = evidence_matrix(frames)
    for s in ("S2", "S3"):
        col = E[:, STEPS.index(s)] >= P_ENTER
        rising = int(np.sum(col[1:] & ~col[:-1]))
        assert rising == 2, (s, rising)
    assert frames[-1][1].world.location("red") == "zone_A"


def test_occlusion_regresses_evidence_and_reports_low_visibility():
    frames = run_replay("occluded")
    src = ReplaySource("occluded")
    t0, t1 = src.scenario.modifiers[0].t0, src.scenario.modifiers[0].t1
    mid = [sf for tick, sf in frames if t0 + 1.0 <= tick.t < t1 - 0.5]
    assert max(sf.predicates.visibility for sf in mid) < 0.4
    s3 = [sf.evidence[STEPS.index("S3")].p_complete for sf in mid]
    assert all(0.3 < p < 0.7 for p in s3)  # "don't know", in or near the ask band
    assert frames[-1][1].evidence[STEPS.index("S3")].p_complete >= P_ENTER


def test_evidence_does_not_depend_on_protocol_order():
    """Prior-free: the same observations give the same evidence under any step order."""
    shuffled = PROTO.model_copy(update={"order": [("S6", "S1"), ("S5", "S2")]})
    a = evidence_matrix(run_replay("S4_before_S2"))
    b = evidence_matrix(run_replay("S4_before_S2", protocol=shuffled))
    np.testing.assert_array_equal(a, b)


def test_initial_state_alone_does_not_count_as_evidence():
    frames = run_replay("correct")
    E = evidence_matrix(frames)
    assert E[0].max() < 0.1
    assert E[: int(3.5 * 15), STEPS.index("S6")].max() < 0.2  # box closed from the start


# ---- synthetic scenes ---------------------------------------------------------------------------


def _det(cls, xy, conf=0.9):
    return Detection(None, cls, conf, (0.0, 0.0, 1.0, 1.0), tuple(xy))


@pytest.mark.parametrize("hand_model", [False, True])
def test_wrong_initial_prior_cannot_fabricate_a_completion(hand_model):
    """The protocol says the red box starts in the closed outer box; in fact it has been
    sitting in Zone A all along. Nobody did S2 or S3, so neither may be evidenced."""
    est = SpatialEstimator(PROTO)
    peak = dict.fromkeys(STEPS, 0.0)
    for i in range(6 * 15):
        t = i / 15
        dets = [_det("outer_box_closed", (0.275, 0.425)), _det("red_box", (0.675, 0.225))]
        # the hand model sees one hand, far from every detection
        hands = HandSet(i, t, [HandObs("left", np.full((21, 2), 500.0), None, 0.9)])
        sf = est.update(
            i, t, detections=DetectionSet(i, t, dets, "test"), hands=hands if hand_model else None
        )
        for e in sf.evidence:
            peak[e.step_id] = max(peak[e.step_id], e.p_complete)
    assert sf.world.location("red") == "zone_A"  # the world model follows the observations
    ask_low = PROTO.decision.ask_band[0]
    # never verified, never even asked about; with a hand model, "held" is ruled out too
    limit = 0.1 if hand_model else ask_low
    assert peak["S2"] < limit and peak["S3"] < limit, peak


def test_sources_breakdown_and_messages():
    frames = run_replay("correct")
    _, sf = frames[-1]
    ev = {e.step_id: e for e in sf.evidence}
    assert set(ev["S2"].sources) == {"armed", "post", "handled", "rule", "visibility"}
    assert set(ev["S1"].sources) == {"armed", "post", "rule", "visibility"}
    assert ev["S3"].sources["rule"] == ev["S3"].p_complete
    assert set(PROTO.predicates) <= set(sf.predicates.values)
    lo, hi = SpatialParams().p_clip
    assert all(lo <= p <= hi for p in sf.predicates.values.values())
    topics = [tp for tp, _ in sf.messages()]
    assert topics.count(Topic.EVIDENCE) == len(STEPS)
    assert topics[0] == Topic.PREDICATES and topics[-1] == Topic.STATS  # no rack frame given
    st = sf.stats()
    assert st.source == "spatial" and st.values["red.loc"] == "zone_A"
    assert st.values["outer.p_open"] < 0.5 and st.values["rack.state"] == "none"


def test_temperature_calibration():
    assert calibrate(0.8, 1.0) == 0.8
    assert calibrate(0.8, 2.0) < 0.8 and calibrate(0.8, 0.5) > 0.8
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.05, 0.95, 4000)
    labels = rng.uniform(size=4000) < true_p
    logit = np.log(true_p / (1 - true_p))
    overconfident = 1 / (1 + np.exp(-3 * logit))
    assert fit_temperature(overconfident, labels) == pytest.approx(3.0, rel=0.2)
    params = SpatialParams(temperatures={"red.held": 2.0})
    est = SpatialEstimator(PROTO, params=params)
    sf = est.update(0, 0.0, detections=DetectionSet(0, 0.0, [], "x"))
    raw = sf.world.predicates["red.held"]
    v = sf.world.visibility
    expected = v * calibrate(raw, 2.0) + (1 - v) * 0.5
    assert sf.predicates.values["red.held"] == pytest.approx(max(expected, 0.01))


# ---- hands and pose projected into the rack frame -----------------------------------------------


def test_hands_and_pose_are_projected_with_the_rack_frame():
    src = ReplaySource("correct", render=True)
    tick = next(iter(src))
    est = SpatialEstimator(PROTO)
    # a hand whose landmarks sit at the replay's pixel for rack point (0.6, 0.7)
    from actus.capture.replay import rack_to_px

    px = np.tile(rack_to_px(0.6, 0.7), (21, 1))
    hands = HandSet(tick.frame_idx, tick.t, [HandObs("right", px, None, 0.9)])
    wl = np.zeros((33, 3))
    wl[[11, 12], 1] = -0.5  # shoulders above the hips in camera axes
    pose = PoseSet(tick.frame_idx, tick.t, [PoseObs(np.zeros((33, 2)), wl, None, 0.9, 0)])
    sf = est.update(
        tick.frame_idx, tick.t, frame=tick.frame, detections=tick.detections, hands=hands, pose=pose
    )
    assert sf.rack is not None and sf.rack.valid
    np.testing.assert_allclose(
        sf.hands.hands[0].landmarks_rack[:, :2], [[0.6, 0.7]] * 21, atol=0.005
    )
    assert np.all(sf.hands.hands[0].landmarks_rack[:, 2] == 0)
    torso = sf.pose.poses[0].torso_vec_rack
    assert torso is not None and np.linalg.norm(torso) == pytest.approx(1.0)
    assert sf.stats().values["torso.y"] is not None
    assert next(tp for tp, _ in sf.messages()) == Topic.RACK


def test_from_config_uses_rack_and_spatial_sections():
    from actus.core.config import load_config

    cfg = load_config(overrides=["rack.hold_frames=7", "spatial.place_rate=0.5"])
    est = SpatialEstimator.from_config(cfg)
    assert est.rack_estimator.hold_frames == 7
    assert est.params.place_rate == 0.5
    assert est.world.container_names == ["outer"]
    # the single container gets the rack.yaml footprint
    fp = est.world.snapshot().objects["outer"].footprint
    assert fp is not None and fp[0] == pytest.approx((0.15, 0.30))
    assert isinstance(est.rack_estimator, RackFrameEstimator)
    assert isinstance(est.geometry, RackGeometry)
