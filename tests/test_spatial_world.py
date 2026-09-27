"""T-SPATIAL world state: occlusion logic, hand gating, fps independence, vocabulary."""

import numpy as np
import pytest

from actus.core.contracts import Detection, DetectionSet
from actus.protocol.schema import Protocol, load_protocol
from actus.spatial.world_state import (
    HandInput,
    SpatialParams,
    WorldModel,
    expm,
    parse_vocabulary,
)
from actus.spatial.zones import Zone

PROTO = load_protocol("protocols/bas_boxes.yaml")
FOOTPRINT = {"outer": [[0.15, 0.30], [0.40, 0.55]]}
OUTER_XY = (0.275, 0.425)
IN_BOX = np.array([0.24, 0.42])
ZONE_A = np.array([0.675, 0.225])
REST_HAND = np.array([0.60, 0.75])


def model(**params):
    return WorldModel(PROTO, SpatialParams(**params), footprints=FOOTPRINT)


def det(cls, xy, conf=0.9, tid=None):
    return Detection(tid, cls, conf, (0.0, 0.0, 1.0, 1.0), None if xy is None else tuple(xy))


def hands_at(*centres):
    """Hand model ran; one hand per centre (8 keypoints in a 3 cm cluster)."""
    offs = np.array([[0, 0], [0.02, 0], [0, 0.02], [0.02, 0.02], [0.01, 0.01], [-0.01, 0.0],
                     [0.0, -0.01], [0.01, -0.01]])  # fmt: skip
    rack = tuple(np.asarray(c, float) + offs for c in centres)
    px = tuple(np.full((8, 2), 5000.0) for _ in centres)  # far outside any bbox
    return HandInput(True, px, rack)


def lerp(a, b, u):
    return np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * min(max(u, 0), 1)


def run(wm, fps, script, until):
    """script(t) -> (dets list, HandInput) ; returns list of snapshots."""
    out = []
    for i in range(round(until * fps)):
        t = i / fps
        dets, hands = script(t)
        out.append(
            wm.update(
                i, t, detections=DetectionSet(i, t, dets, "test"), hands=hands, rack_state="fresh"
            )
        )
    return out


def carry_script(dropout=(2.5, 4.0), blackout=False, into_box=False):
    """Box opens; the red box is lifted out and carried, vanishes for `dropout`, then either
    reappears and is placed in Zone A (default) or, with into_box, is lowered back into the box
    before vanishing and the hand withdraws."""
    mid = np.array([0.50, 0.28])

    def script(t):
        yellow = det("yellow_box", (0.31, 0.43))
        outer = det("outer_box_open", OUTER_XY)
        d0, d1 = dropout
        if t < 1.0:
            hand, red = lerp(REST_HAND, IN_BOX, 0), IN_BOX
        elif t < 1.5:
            hand, red = lerp(REST_HAND, IN_BOX, (t - 1.0) / 0.5), IN_BOX
        elif into_box:
            # carry out to `mid`, bring it back over the box and lower it in (it vanishes);
            # the hand lingers briefly, then withdraws
            if t < 2.3:
                hand = red = lerp(IN_BOX, mid, (t - 1.5) / 0.8)
            elif t < d0:
                hand = red = lerp(mid, [0.28, 0.44], (t - 2.3) / (d0 - 2.3))
            else:
                red = None
                hand = lerp([0.28, 0.44], REST_HAND, (t - d0 - 0.3) / 0.7)
        elif t < d0:
            hand = red = lerp(IN_BOX, mid, (t - 1.5) / (d0 - 1.5))
        elif t < d1:
            hand, red = lerp(mid, mid + np.array([0.05, -0.02]), (t - d0) / (d1 - d0)), None
        elif t < d1 + 1.0:
            hand = red = lerp(mid + np.array([0.05, -0.02]), ZONE_A, (t - d1) / 1.0)
        else:
            red = ZONE_A
            hand = lerp(ZONE_A, REST_HAND, (t - d1 - 1.0) / 0.5)
        in_dropout = not into_box and d0 <= t < d1
        if blackout and in_dropout:
            return [], HandInput(True, (), ())  # nothing visible at all
        dets = [outer, yellow] + ([] if red is None else [det("red_box", red)])
        return dets, hands_at(hand)

    return script


# ---- occlusion logic ----------------------------------------------------------------------------


@pytest.mark.parametrize("blackout", [False, True])
def test_held_object_stays_held_through_1_5s_dropout(blackout):
    wm = model()
    fps = 30
    snaps = run(wm, fps, carry_script(blackout=blackout), until=6.5)
    at = {round(s.t, 3): s for s in snaps}
    assert at[1.0].location("red") == "in_outer"
    assert at[2.4].location("red") == "held"
    dropout = [s for s in snaps if 2.5 <= s.t < 4.0]
    assert len(dropout) == 45
    assert all(s.location("red") == "held" for s in dropout), [
        (round(s.t, 2), s.location("red")) for s in dropout if s.location("red") != "held"
    ]
    assert all(s.objects["red"].belief["held"] > 0.6 for s in dropout)
    assert snaps[-1].location("red") == "zone_A"
    assert snaps[-1].predicates["red.in_zone_A"] > 0.9
    assert snaps[-1].location("yellow") == "in_outer"
    if blackout:
        assert min(s.visibility for s in dropout) < 0.5  # the blind spell is reported


def test_object_vanishing_into_open_box_becomes_in_outer():
    wm = model()
    snaps = run(wm, 30, carry_script(dropout=(3.0, 99), into_box=True), until=5.0)
    at = {round(s.t, 3): s for s in snaps}
    assert at[2.2].location("red") == "held"
    assert snaps[-1].location("red") == "in_outer"
    assert snaps[-1].predicates["red.in_outer"] > 0.8


@pytest.mark.parametrize("fps", [10, 15, 60])
def test_result_does_not_depend_on_frame_rate(fps):
    ref = run(model(), 30, carry_script(), until=6.5)
    snaps = run(model(), fps, carry_script(), until=6.5)
    for t_check in (2.3, 3.5, 6.0):
        a = min(ref, key=lambda s: abs(s.t - t_check))
        b = min(snaps, key=lambda s: abs(s.t - t_check))
        assert a.location("red") == b.location("red")
        assert abs(a.predicates["red.held"] - b.predicates["red.held"]) < 0.2


def test_undetected_object_without_hands_is_not_invented_held():
    """A box resting in Zone A that the detector loses (hand far away) must not become held."""
    wm = model()

    def script(t):
        red = [det("red_box", ZONE_A)] if t < 2.0 else []
        return [det("outer_box_open", OUTER_XY), *red], hands_at(REST_HAND)

    snaps = run(wm, 30, script, until=3.5)
    assert snaps[59].location("red") == "zone_A"
    assert snaps[-1].objects["red"].belief["held"] < 0.3
    assert snaps[-1].location("red") in ("zone_A", "unknown")


def test_box_resting_on_zone_edge_is_uncertain():
    wm = model()
    edge = np.array([0.55, 0.225])  # exactly on Zone A's left edge

    def script(t):
        red = lerp([0.45, 0.30], edge, (t - 1.0) / 0.8) if t < 1.8 else edge
        hand = red if t < 2.0 else lerp(edge, REST_HAND, (t - 2.0) / 0.5)
        return [det("outer_box_open", OUTER_XY), det("red_box", red)], hands_at(hand)

    snaps = run(wm, 30, script, until=6.0)
    p = snaps[-1].predicates["red.in_zone_A"]
    assert 0.3 < p < 0.7, p


def test_container_opens_only_on_evidence_and_initial_prior():
    wm = model()
    s0 = wm.snapshot()
    assert s0.location("outer") == "closed"
    assert s0.location("red") == "in_outer" and s0.location("yellow") == "in_outer"

    def script(t):
        cls = "outer_box_closed" if t < 1.0 else "outer_box_open"
        hand = OUTER_XY if 0.8 < t < 1.5 else REST_HAND
        return [det(cls, OUTER_XY)], hands_at(hand)

    snaps = run(wm, 30, script, until=2.5)
    assert snaps[25].location("outer") == "closed"
    assert snaps[-1].location("outer") == "open" and snaps[-1].objects["outer"].p_open > 0.9


def test_footprint_follows_the_container():
    wm = model()
    moved = (0.40, 0.60)
    run(wm, 30, lambda t: ([det("outer_box_closed", moved)], hands_at(REST_HAND)), until=3.0)
    fp = Zone("outer", np.array(wm.snapshot().objects["outer"].footprint))
    assert fp.centroid == pytest.approx(moved, abs=0.01)


def test_hoi_input_drives_held():
    wm = model()

    def script(t):
        return [det("outer_box_open", OUTER_XY), det("red_box", (0.5, 0.3))], HandInput(False)

    for i in range(45):
        t = i / 30
        dets, hands = script(t)
        s = wm.update(
            i, t, detections=DetectionSet(i, t, dets, "x"), hands=hands, hoi={"red": 0.95}
        )
    assert s.location("red") == "held"


def test_visibility_drops_when_expected_objects_vanish():
    wm = model()
    seen = [
        det("outer_box_open", OUTER_XY),
        det("red_box", ZONE_A),
        det("yellow_box", (0.675, 0.575)),
    ]
    run(wm, 30, lambda t: (seen, hands_at(REST_HAND)), until=1.0)
    assert wm.visibility > 0.9
    run(wm, 30, lambda t: ([], HandInput(True, (), ())), until=1.5)
    assert wm.visibility < 0.4


# ---- vocabulary, maths --------------------------------------------------------------------------


def test_vocabulary_of_bas_protocol():
    containers, movables = parse_vocabulary(PROTO)
    assert containers == ["outer"] and movables == ["red", "yellow"]
    wm = model()
    assert set(PROTO.predicates) <= set(wm.predicate_names)
    assert "red.in_zone_B" in wm.predicate_names and "outer.open" in wm.predicate_names
    assert wm.states == ["in_outer", "held", "zone_A", "zone_B", "elsewhere"]


def _proto(**over):
    base = {
        "id": "t",
        "title": "t",
        "zones": {"A": [[0, 0], [1, 1]]},
        "steps": [{"id": "S1", "text": "x", "post": {"cup.in_zone_A": True}}],
    }
    base.update(over)
    return Protocol.model_validate(base)


def test_unsupported_predicate_is_rejected():
    p = _proto(steps=[{"id": "S1", "text": "x", "post": {"cup.on_table": True}}])
    with pytest.raises(ValueError, match=r"cup\.on_table"):
        WorldModel(p)


def test_object_cannot_be_container_and_movable():
    p = _proto(
        steps=[
            {"id": "S1", "text": "x", "post": {"cup.in_box": True}},
            {"id": "S2", "text": "y", "post": {"box.held": True}},
        ]
    )
    with pytest.raises(ValueError, match="both"):
        WorldModel(p)


def test_protocol_without_containers_or_objects_works():
    wm = WorldModel(_proto())
    assert wm.states == ["held", "zone_A", "elsewhere"]
    assert wm.snapshot().location("cup") == "unknown"  # no initial prior, never seen
    for i in range(30):
        s = wm.update(i, i / 15, detections=DetectionSet(i, i / 15, [det("cup", (0.5, 0.5))], "x"))
    assert s.location("cup") == "zone_A"


def test_unknown_spatial_parameter_is_rejected():
    with pytest.raises(KeyError, match="held_timeout_s"):
        SpatialParams.from_mapping({"held_timeout_s": 1.5})
    assert SpatialParams.from_mapping({"p_clip": [0.05, 0.95]}).p_clip == (0.05, 0.95)


def test_expm_gives_stochastic_matrix():
    Q = np.array([[-2.0, 1.5, 0.5], [0.1, -0.1, 0.0], [3.0, 0.0, -3.0]])
    P = expm(Q * 0.7)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-9)
    assert (P >= 0).all()
    np.testing.assert_allclose(expm(Q * 0.35) @ expm(Q * 0.35), P, atol=1e-9)
