"""The runtime wrappers: streaming parity, abstention, and the limits on evidence fusion.

The fusion tests are the important ones. They are the arithmetic proof of build plan section 2's
evidence/expectation split: a model that has learned the usual order of the procedure must not
be able to complete a step whose precondition was never observed.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_models_nn import random_mlp, random_tcn

from actus.core.contracts import IDLE, ActionScores, StepEvidence
from actus.models.features import SPEC_VERSION, FrameFeatures
from actus.models.fusion import EvidenceFusion, noisy_or
from actus.models.hoi import HOI_CLASSES, HOIModel
from actus.models.nn import MLP, TCN, ModelSpec, mlp_arch, save_npz, tcn_arch
from actus.models.runtime import ModelStack
from actus.models.temporal import TemporalModel, resample_to_grid
from actus.protocol.schema import load_protocol

STEPS = ("S1", "S2", "S3", "S4", "S5", "S6")
CLASSES = (*STEPS, IDLE)


@pytest.fixture(scope="module")
def protocol():
    return load_protocol("protocols/bas_boxes.yaml")


def make_temporal(n_features=9, dilations=(1, 2), seed=0):
    rng = np.random.default_rng(seed)
    spec, weights = random_tcn(
        rng, n_features=n_features, channels=8, n_classes=len(CLASSES), dilations=dilations
    )
    spec = ModelSpec(
        "tcn",
        CLASSES,
        SPEC_VERSION,
        n_features,
        {**tcn_arch(8, dilations, 3), "feature_fps": 15.0},
        {},
    )
    net = TCN(spec, weights)
    return (
        TemporalModel(
            net,
            CLASSES,
            feature_fps=15.0,
            n_features=n_features,
            receptive_field=net.receptive_field,
        ),
        weights,
        spec,
    )


def features(i, t, vec, *, pair=None, valid=None):
    n_pair = 4 if pair is None else pair.shape[-1]
    return FrameFeatures(
        i,
        t,
        np.asarray(vec, dtype=np.float32),
        np.zeros((2, 2, n_pair), dtype=np.float32) if pair is None else pair,
        np.zeros((2, 2), dtype=bool) if valid is None else valid,
        True,
    )


# -- temporal -----------------------------------------------------------------------------------


def test_resample_to_grid_holds_the_most_recent_row():
    t = np.array([0.0, 0.05, 0.10, 0.30])
    frames = np.arange(4)[:, None].astype(np.float32)
    grid, out = resample_to_grid(t, frames, fps=10.0)
    np.testing.assert_allclose(grid, [0.0, 0.1, 0.2, 0.3], atol=1e-9)
    np.testing.assert_allclose(out.ravel(), [0, 2, 2, 3])


def test_streaming_matches_the_offline_pass():
    """update() frame by frame must reproduce predict_sequence() exactly.

    Without this the offline evaluation would not describe the live system.
    """
    model, _, _ = make_temporal()
    rng = np.random.default_rng(1)
    rows = rng.normal(size=(40, model.n_features)).astype(np.float32)
    offline = model.predict_sequence(rows)

    model.reset()
    streamed = []
    for i, row in enumerate(rows):
        out = model.update(features(i, i / 15.0, row))
        assert out is not None
        streamed.append([out.scores[c] for c in model.classes])
    np.testing.assert_allclose(np.array(streamed), offline, rtol=1e-8, atol=1e-9)


def test_a_faster_camera_does_not_change_the_grid():
    """At 30 fps only every second frame advances the 15 Hz grid."""
    model, _, _ = make_temporal()
    rng = np.random.default_rng(2)
    rows = rng.normal(size=(20, model.n_features)).astype(np.float32)
    emitted = sum(model.update(features(i, i / 30.0, rows[i])) is not None for i in range(20))
    assert emitted == 10


def test_action_scores_are_a_valid_contract():
    model, _, _ = make_temporal()
    out = model.update(features(0, 0.0, np.zeros(model.n_features, dtype=np.float32)))
    assert isinstance(out, ActionScores)
    assert set(out.scores) == set(CLASSES)
    assert sum(out.scores.values()) == pytest.approx(1.0)
    label, score = out.top()
    assert label in CLASSES and 0.0 <= score <= 1.0


def test_temporal_rejects_the_wrong_feature_width():
    model, _, _ = make_temporal()
    with pytest.raises(ValueError, match="features"):
        model.update(features(0, 0.0, np.zeros(model.n_features + 1, dtype=np.float32)))


def test_temporal_load_round_trip(tmp_path):
    model, weights, spec = make_temporal()
    save_npz(tmp_path / "temporal.npz", spec, weights)
    loaded = TemporalModel.load(tmp_path / "temporal.npz")
    assert loaded.classes == CLASSES
    assert loaded.feature_fps == 15.0
    assert loaded.backend == "numpy"
    rows = np.random.default_rng(3).normal(size=(20, model.n_features)).astype(np.float32)
    np.testing.assert_allclose(loaded.predict_sequence(rows), model.predict_sequence(rows))


# -- HOI ----------------------------------------------------------------------------------------


def make_hoi(n_features=6, seed=0):
    rng = np.random.default_rng(seed)
    spec, weights = random_mlp(rng, n_features=n_features, hidden=(8,), n_classes=3)
    spec = ModelSpec(
        "mlp",
        HOI_CLASSES,
        SPEC_VERSION,
        n_features,
        {**mlp_arch((8,)), "hands": ["left", "right"], "objects": ["red", "yellow"]},
        {},
    )
    return (
        HOIModel(
            MLP(spec, weights),
            HOI_CLASSES,
            hands=("left", "right"),
            objects=("red", "yellow"),
            n_features=n_features,
        ),
        weights,
        spec,
    )


def test_hoi_abstains_on_unseen_objects():
    """An object nobody saw must produce no entry at all, not a confident guess."""
    model, _, _ = make_hoi()
    rng = np.random.default_rng(4)
    pair = rng.normal(size=(2, 2, model.n_features)).astype(np.float32)
    valid = np.array([[False, False], [True, False]])  # right hand + red only
    result = model.predict(features(0, 0.0, np.zeros(3), pair=pair, valid=valid))
    holding = result.holding()
    assert set(holding) == {"red"}
    assert "yellow" not in holding
    assert result.label(0, 0) is None
    assert result.label(1, 0) in HOI_CLASSES
    np.testing.assert_allclose(result.probs[0, 0], 0.0)


def test_hoi_holding_takes_the_more_confident_hand():
    model, _, _ = make_hoi()
    pair = np.zeros((2, 2, model.n_features), dtype=np.float32)
    valid = np.array([[True, False], [True, False]])
    result = model.predict(features(0, 0.0, np.zeros(3), pair=pair, valid=valid))
    hold_idx = HOI_CLASSES.index("holding")
    assert result.holding()["red"] == pytest.approx(float(np.max(result.probs[:, 0, hold_idx])))


def test_hoi_coverage_counts_what_it_could_answer():
    model, _, _ = make_hoi()
    pair = np.zeros((2, 2, model.n_features), dtype=np.float32)
    valid = np.array([[True, False], [False, False]])
    model.predict(features(0, 0.0, np.zeros(3), pair=pair, valid=valid))
    assert model.coverage() == pytest.approx(0.25)
    model.reset()
    assert model.coverage() == 0.0


# -- fusion -------------------------------------------------------------------------------------


def evidence(step, p, armed, frame_idx=0, t=0.0, visibility=1.0, handled=None):
    sources = {"rule": p, "armed": armed, "post": p, "visibility": visibility}
    if handled is not None:
        sources["handled"] = handled
    return StepEvidence(frame_idx, t, step, p, sources)


def finished(fusion, step, *, t=0.4, frame_idx=6):
    """Drive ``fusion`` to the state where ``step``'s action has peaked and then ended."""
    for i in range(6):
        fusion.observe(actions({step: 1.0}, i, i / 15))
    act = actions({IDLE: 1.0}, frame_idx, t)
    fusion.observe(act)
    return act


def actions(scores, frame_idx=0, t=0.0):
    full = dict.fromkeys(CLASSES, 0.0)
    full.update(scores)
    total = sum(full.values()) or 1.0
    return ActionScores(frame_idx, t, {k: v / total for k, v in full.items()})


def test_fusion_cannot_exceed_the_armed_latch():
    """No temporal confidence completes a step whose precondition was never observed."""
    fusion = EvidenceFusion(STEPS, weight=1.0)
    act = finished(fusion, "S2")
    fused = fusion.fuse(evidence("S2", 0.0, armed=0.0, visibility=0.0), act)
    assert fused.p_complete == 0.0


def test_fusion_respects_its_bounds_for_any_weight():
    """Randomised: fused is capped by armed, never lowers the rule, and honours visibility."""
    rng = np.random.default_rng(5)
    for _ in range(200):
        armed = float(rng.random())
        rule = float(rng.random())
        weight = float(rng.random())
        visibility = float(rng.random())
        fusion = EvidenceFusion(STEPS, weight=weight)
        act = finished(fusion, "S3")
        ev = evidence("S3", rule, armed=armed, visibility=visibility)
        fused = fusion.fuse(ev, act).p_complete
        # Fusion never lowers the incoming rule, and its own contribution is capped by `armed`
        # and by how blind the sensors were. (Real spatial evidence always has rule <= armed,
        # since rule = min(armed, post, handled); the rng here does not respect that.)
        assert fused >= rule - 1e-9
        assert fused <= max(rule, armed) + 1e-9
        assert fused <= max(rule, weight * (1.0 - visibility)) + 1e-9


def test_a_clear_view_stops_the_model_raising_anything():
    """With visibility 1, a low post-condition is an observation the model may not overrule.

    This is the structural version of the guarantee: not "it usually does not", but "it cannot".
    """
    for weight in (0.25, 0.5, 1.0):
        fusion = EvidenceFusion(STEPS, weight=weight)
        act = finished(fusion, "S2")
        ev = evidence("S2", 0.05, armed=0.95, visibility=1.0)
        assert fusion.fuse(ev, act).p_complete == pytest.approx(0.05)


def test_fusion_can_stand_in_for_a_missing_handled_term():
    """The detector never saw the object in a hand, but the action model saw the interaction."""
    fusion = EvidenceFusion(STEPS, weight=1.0)
    act = finished(fusion, "S2")
    ev = evidence("S2", 0.0, armed=0.9, visibility=1.0, handled=0.0)
    ev = StepEvidence(6, 0.4, "S2", 0.0, {**ev.sources, "post": 0.85, "rule": 0.0})
    fused = fusion.fuse(ev, act)
    assert fused.p_complete > 0.5  # post is there, only `handled` was missing


def test_fusion_is_a_no_op_without_action_evidence():
    fusion = EvidenceFusion(STEPS, weight=0.5)
    ev = evidence("S2", 0.42, armed=0.9)
    assert fusion.fuse_all([ev], None) == (ev,)
    idle_only = actions({IDLE: 1.0})
    assert fusion.fuse_all([ev], idle_only)[0].p_complete == pytest.approx(0.42)


def test_fusion_passes_through_evidence_without_an_armed_source():
    """Evidence from a producer that does not latch preconditions is left alone."""
    fusion = EvidenceFusion(STEPS, weight=1.0)
    act = finished(fusion, "S2")
    bare = StepEvidence(6, 0.4, "S2", 0.3, {"rule": 0.3})
    assert fusion.fuse(bare, act) is bare


def test_fusion_raises_evidence_when_the_sensors_were_blind():
    """The case fusion exists for: the operator's arm hid the placement, the model saw it."""
    fusion = EvidenceFusion(STEPS, weight=0.8)
    act = finished(fusion, "S3")
    fused = fusion.fuse(evidence("S3", 0.45, armed=0.95, visibility=0.1), act)
    assert fused.p_complete > 0.45
    assert fused.sources["temporal"] > 0.0
    assert fused.sources["rule"] == pytest.approx(0.45)


def test_fusion_ignores_an_action_still_in_progress():
    """A step in progress is not a completed step."""
    fusion = EvidenceFusion(STEPS, weight=1.0)
    act = actions({"S3": 1.0})
    for i in range(6):
        fusion.observe(actions({"S3": 1.0}, i, i / 15))
    fused = fusion.fuse(evidence("S3", 0.4, armed=0.95, visibility=0.0), act)
    assert fused.p_complete == pytest.approx(0.4)


def test_noisy_or_is_monotone():
    assert noisy_or(0.0, 0.0) == 0.0
    assert noisy_or(0.5, 0.0) == pytest.approx(0.5)
    assert noisy_or(0.5, 0.5) == pytest.approx(0.75)


# -- ModelStack ---------------------------------------------------------------------------------


def test_stack_degrades_without_any_models(protocol):
    """The pipeline must run with no trained models present, which is today's repo state."""
    stack = ModelStack(protocol)
    assert not stack.enabled
    out = stack.update(0, 0.0)
    assert out.hoi == {}
    assert out.actions is None
    assert out.messages() == []
    ev = (StepEvidence(0, 0.0, "S1", 0.5, {"rule": 0.5, "armed": 0.9}),)
    assert stack.fuse(ev) == ev
    assert stack.status()["temporal"] is None


def test_stack_from_config_tolerates_missing_model_files(protocol):
    from actus.core.config import load_config

    cfg = load_config(overrides=["models.hoi=models/does-not-exist/hoi"])
    stack = ModelStack.from_config(cfg, protocol)
    assert stack.hoi is None


def test_stack_publishes_action_scores(protocol):
    from actus.models.features import FeatureLayout

    n_features = FeatureLayout.from_protocol(protocol).n_frame
    model, _, _ = make_temporal(n_features=n_features)
    stack = ModelStack(protocol, temporal=model)
    out = stack.update(0, 0.0)
    assert out.actions is not None
    topics = [t for t, _ in out.messages()]
    assert topics and topics[0].value == "actions"
    assert stack.stats(out).source == "models"
    assert stack.status()["temporal"] == "numpy"


# -- ONNX backend ---------------------------------------------------------------------------


def test_onnx_backend_matches_numpy(tmp_path):
    """The exported ONNX graph must agree with the numpy model of record."""
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from actus.models.nn import build_torch_tcn, export_onnx, save_npz, tcn_weights

    n_features, dilations = 12, (1, 2)
    module = build_torch_tcn(n_features, 8, len(CLASSES), dilations, 3).eval()
    mean = np.zeros(n_features, dtype=np.float32)
    std = np.ones(n_features, dtype=np.float32)
    spec = ModelSpec(
        "tcn",
        CLASSES,
        SPEC_VERSION,
        n_features,
        {**tcn_arch(8, dilations, 3), "feature_fps": 15.0},
        {},
    )
    save_npz(tmp_path / "temporal.npz", spec, tcn_weights(module, mean, std))
    assert export_onnx(
        module, tmp_path / "temporal.onnx", example_shape=(1, 32, n_features), dynamic_time=True
    )

    onnx_model = TemporalModel.load(tmp_path / "temporal.npz")
    assert onnx_model.backend == "onnx"
    numpy_model = TemporalModel.load(tmp_path / "temporal.npz", prefer_onnx=False)
    assert numpy_model.backend == "numpy"

    rows = np.random.default_rng(7).normal(size=(30, n_features)).astype(np.float32)
    np.testing.assert_allclose(
        onnx_model.predict_sequence(rows), numpy_model.predict_sequence(rows), atol=1e-5
    )


def test_temporal_recovers_from_a_long_stall():
    """A big timestamp jump must not spin the grid forward one step at a time."""
    model, _, _ = make_temporal()
    rows = np.zeros(model.n_features, dtype=np.float32)
    assert model.update(features(0, 0.0, rows)) is not None
    out = model.update(features(1, 600.0, rows))  # ten minutes later
    assert out is not None
    assert len(model._buf) <= model.receptive_field
    assert model.update(features(2, 600.0 + 1 / 15.0, rows)) is not None
