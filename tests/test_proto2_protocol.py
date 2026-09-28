"""T-PROTO2: the second protocol (protocols/sample_mixing.yaml) needs no code.

It must validate, be executable from its own ``initial`` state, stay inside what the world model
can sense, respect the rule that keeps the engine from crying "unexpected" (post-states persist),
and drive the *real* protocol engine to sensible outcomes and messages from hand-built evidence.
The full spatial + pipeline run is in ``test_proto2_demo.py``.
"""

import types
from collections.abc import Mapping

import numpy as np
import pytest

from actus.core.config import load_config
from actus.core.contracts import Alert, AlertKind, Ask, Prompt, StepEvidence, StepEvent, StepStatus
from actus.protocol.engine import DONE, ProtocolEngine, run_ticks
from actus.protocol.schema import Protocol, load_protocol
from actus.spatial.rack_frame import RackGeometry
from actus.spatial.world_state import parse_vocabulary

PROTOCOL = "protocols/sample_mixing.yaml"
CONFIG = "configs/sample_mixing.yaml"
STEP_IDS = ["S1", "S2", "S3", "S4", "S5"]


@pytest.fixture(scope="module")
def proto() -> Protocol:
    return load_protocol(PROTOCOL)


# ---- the file itself -----------------------------------------------------------------------------


def test_loads_with_the_expected_shape(proto):
    assert proto.id == "sample_mixing_v1"
    assert proto.step_ids == STEP_IDS
    assert proto.order == "strict"
    assert {s.id: s.requires_hoi for s in proto.steps if s.requires_hoi} == {
        "S2": "bottle",
        "S4": "book",
    }
    assert sorted(proto.zones) == ["M", "R", "S"]
    assert proto.predicates  # nothing empty


def test_objects_are_plain_coco_classes(proto):
    """Detected by stock COCO weights: each object maps to the one class of its own name."""
    assert {o: s.classes for o, s in proto.objects.items()} == {
        "cup": ["cup"],
        "bottle": ["bottle"],
        "book": ["book"],
    }


def test_world_model_can_sense_every_predicate(proto):
    containers, movables = parse_vocabulary(proto)  # raises on an unsupported predicate form
    assert containers == []
    assert movables == ["book", "bottle", "cup"]


def test_zones_fit_the_board_and_avoid_markers_and_each_other(proto):
    geo = RackGeometry.load()
    width, height = geo.board_size
    rects = {}
    for name, pts in proto.zones.items():
        (x1, y1), (x2, y2) = np.min(pts, axis=0), np.max(pts, axis=0)
        assert len(pts) == 2, f"zone {name} is a rectangle"
        assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, f"zone {name} is on the board"
        rects[name] = (x1, y1, x2, y2)

    def overlap(a, b, margin=0.0):
        return not (
            a[2] + margin <= b[0]
            or b[2] + margin <= a[0]
            or a[3] + margin <= b[1]
            or b[3] + margin <= a[1]
        )

    for name, r in rects.items():
        for mid in geo.markers:
            corners = geo.marker_corners(mid)
            m = (*corners[:, :2].min(axis=0), *corners[:, :2].max(axis=0))
            assert not overlap(r, m, margin=0.01), f"zone {name} covers marker {mid}"
    names = sorted(rects)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not overlap(rects[a], rects[b]), f"zones {a} and {b} overlap"


def test_executable_from_its_own_initial_state(proto):
    """Walk the steps in order over the predicate state: an explicit precondition holds when its
    step is reached and every post-state is new. A protocol failing this could never complete."""
    state = dict(proto.initial)
    for s in proto.steps:
        for k, v in s.pre.items():
            assert state.get(k) == v, f"{s.id}: precondition {k}={v} does not hold when reached"
        assert any(state.get(k) != v for k, v in s.post.items()), f"{s.id}: nothing changes"
        state.update(s.post)


def test_post_states_persist_to_the_end(proto):
    """The design rule that keeps the engine quiet: no later step reverses an earlier post-state.
    (A reversal is 'undone', flagged `unexpected` unless the reversing step is completing at the
    same instant, which a carry between two places never is.)"""
    steps = proto.steps
    for i, a in enumerate(steps):
        for b in steps[i + 1 :]:
            for k, v in a.post.items():
                assert b.post.get(k, v) == v, f"{b.id} reverses {a.id}'s post-state {k}"


def test_every_step_names_its_object_and_a_question(proto):
    """Placements can be unsure, so they carry a tailored question; the wording says the object."""
    for s in proto.steps:
        obj = s.requires_hoi or next(iter(s.post)).split(".", 1)[0]
        assert proto.object_label(obj) in s.text.lower()
        if s.ask is not None:
            assert proto.object_label(obj) in s.ask and s.ask.endswith("?")


# ---- the config overlay -------------------------------------------------------------------------


def test_overlay_selects_the_protocol_and_coco_detector():
    cfg = load_config(CONFIG)  # strict: every key must already exist in default.yaml
    assert cfg.protocol.path == PROTOCOL
    assert load_protocol(cfg.protocol.path).id == "sample_mixing_v1"
    assert cfg.detector.backend == "custom"
    assert str(cfg.detector.custom_weights).endswith("yolo11n.pt")  # stock COCO weights
    assert cfg.models.enabled is False  # HOI/temporal belong to the box protocol
    # the default config is untouched: the box protocol is still the primary one
    assert load_config().protocol.path == "protocols/bas_boxes.yaml"


# ---- the engine on this protocol, from hand-built evidence ---------------------------------------

FPS = 15.0
LOW, HIGH = 0.02, 0.95


def _ticks(proto: Protocol, on: Mapping[str, float | tuple[float, float]], seconds: float):
    """Evidence for each step: LOW until ``on[step]`` (a time, or (time, level)), then it holds.
    Steps not in ``on`` stay LOW. Post-states persist, so evidence never falls again."""
    for i in range(round(seconds * FPS)):
        t = i / FPS
        ev = []
        for sid in proto.step_ids:
            spec = on.get(sid)
            p = LOW
            if spec is not None:
                when, level = spec if isinstance(spec, tuple) else (spec, HIGH)
                p = level if t >= when else LOW
            ev.append(StepEvidence(i, t, sid, p, {"rule": p}))
        yield types.SimpleNamespace(
            frame_idx=i,
            t=t,
            evidence=tuple(ev),
            predicates=types.SimpleNamespace(visibility=0.95),
            actions=None,
        )


def _run(proto, on, seconds, **kw):
    engine = ProtocolEngine(proto)
    out = run_ticks(engine, _ticks(proto, on, seconds), **kw)
    return out, {r["step_id"]: r["status"] for r in engine.outcomes()}


def _alerts(out) -> list[Alert]:
    return [o for o in out if isinstance(o, Alert)]


def test_correct_order_verifies_every_step_quietly(proto):
    on = {"S1": 2.0, "S2": 4.0, "S3": 6.0, "S4": 8.0, "S5": 10.0}
    out, status = _run(proto, on, 13.0)
    assert status == dict.fromkeys(STEP_IDS, "verified")
    assert _alerts(out) == [] and not [o for o in out if isinstance(o, Ask)]
    prompts = [o for o in out if isinstance(o, Prompt)]
    assert [p.next_step_id for p in prompts] == [*STEP_IDS, None]
    assert [p.text for p in prompts[:3]] == [
        "Place the cup on the mixing pad",
        "Take the bottle from the stowage area",
        "Place the bottle next to the cup",
    ]
    assert prompts[-1].text == "Procedure complete."


def test_bottle_before_cup_is_an_out_of_sequence_alert_worded_from_the_yaml(proto):
    on = {"S2": 3.0, "S3": 5.0, "S1": 8.0, "S4": 10.0, "S5": 12.0}
    out, status = _run(proto, on, 15.0)
    alerts = _alerts(out)
    assert [a.kind for a in alerts] == [AlertKind.OUT_OF_ORDER]
    assert alerts[0].message == (
        "Out of sequence: bottle moved out of Zone S before cup. "
        "Expected: place the cup on the mixing pad."
    )
    assert alerts[0].step_ids == ("S2", "S1")
    assert all(StepStatus(s) in DONE for s in status.values()), status  # recovered, none skipped


def test_never_placing_the_cup_is_skipped_retroactively(proto):
    on = {"S2": 3.0, "S3": 5.0, "S4": 7.0, "S5": 9.0}
    out, status = _run(proto, on, 12.0)
    assert [a.kind for a in _alerts(out)] == [AlertKind.OUT_OF_ORDER, AlertKind.SKIP]
    assert _alerts(out)[-1].message == "Skipped: place the cup on the mixing pad."
    assert status["S1"] == "skipped"
    assert all(StepStatus(status[s]) in DONE for s in STEP_IDS[1:]), status
    skipped = [o for o in out if isinstance(o, StepEvent) and o.status is StepStatus.SKIPPED]
    assert [(e.step_id, e.retroactive) for e in skipped] == [("S1", True)]


def test_unsure_placement_asks_with_the_yaml_question_and_a_yes_confirms_it(proto):
    """S5's evidence hovers at 0.62 (inside the ask band): no verification, one Ask in the words of
    the YAML, and the operator's "yes" is logged as operator_confirmed, never as verified."""
    on = {"S1": 2.0, "S2": 4.0, "S3": 6.0, "S4": 8.0, "S5": (10.0, 0.62)}
    out, status = _run(proto, on, 18.0)
    assert status["S5"] == "pending" and status["S4"] == "verified"
    asks = [o for o in out if isinstance(o, Ask)]
    assert [(a.step_id, a.text) for a in asks] == [
        ("S5", "I couldn't confirm the book is in the record area. Did you place it?")
    ]

    out, status = _run(proto, on, 18.0, answer="yes")
    assert status["S5"] == "operator_confirmed"
    events = [o for o in out if isinstance(o, StepEvent) and o.step_id == "S5"]
    assert [e.status for e in events] == [StepStatus.OPERATOR_CONFIRMED]


def test_answering_no_re_prompts_the_step_and_does_not_confirm_it(proto):
    on = {"S1": 2.0, "S2": 4.0, "S3": 6.0, "S4": 8.0, "S5": (10.0, 0.62)}
    out, status = _run(proto, on, 18.0, answer="no")
    assert status["S5"] == "pending"
    prompts = [o for o in out if isinstance(o, Prompt)]
    assert prompts[-1].next_step_id == "S5"  # asked again to do it


def test_the_prior_never_completes_a_step(proto):
    """No evidence at all for anything: nothing is verified, whatever the protocol expects."""
    out, status = _run(proto, {}, 20.0)
    assert set(status.values()) == {"pending"}
    assert [o for o in out if isinstance(o, StepEvent) and o.status is not StepStatus.PENDING] == []
