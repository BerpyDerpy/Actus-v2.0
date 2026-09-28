"""The numpy forward passes must agree with the torch modules they were exported from.

Two implementations of each architecture exist on purpose (training needs autograd, the runtime
must not pay for it). These tests are what stops them drifting apart, plus the format guards
that keep a stale model from being fed features it was never trained on.
"""

from __future__ import annotations

import numpy as np
import pytest

from actus.models.nn import (
    MLP,
    TCN,
    ModelFormatError,
    ModelSpec,
    load_model,
    load_npz,
    mlp_arch,
    receptive_field,
    save_npz,
    softmax,
    tcn_arch,
)


def random_mlp(rng, n_features=12, hidden=(16, 8), n_classes=3):
    arch = mlp_arch(hidden)
    sizes = [n_features, *hidden, n_classes]
    weights = {
        "mean": rng.normal(size=n_features).astype(np.float32),
        "std": (rng.random(n_features) + 0.5).astype(np.float32),
    }
    for i in range(len(sizes) - 1):
        weights[f"w{i}"] = rng.normal(size=(sizes[i + 1], sizes[i])).astype(np.float32) * 0.3
        weights[f"b{i}"] = rng.normal(size=sizes[i + 1]).astype(np.float32) * 0.1
    spec = ModelSpec("mlp", ("none", "touching", "holding"), 1, n_features, arch, {})
    return spec, weights


def random_tcn(rng, n_features=9, channels=8, n_classes=4, dilations=(1, 2), kernel=3):
    weights = {
        "mean": rng.normal(size=n_features).astype(np.float32),
        "std": (rng.random(n_features) + 0.5).astype(np.float32),
    }
    sizes = [n_features] + [channels] * len(dilations)
    for i in range(len(dilations)):
        weights[f"conv{i}_w"] = (
            rng.normal(size=(sizes[i + 1], sizes[i], kernel)).astype(np.float32) * 0.3
        )
        weights[f"conv{i}_b"] = rng.normal(size=sizes[i + 1]).astype(np.float32) * 0.1
        if sizes[i] != sizes[i + 1]:
            weights[f"proj{i}_w"] = (
                rng.normal(size=(sizes[i + 1], sizes[i], 1)).astype(np.float32) * 0.3
            )
            weights[f"proj{i}_b"] = rng.normal(size=sizes[i + 1]).astype(np.float32) * 0.1
    weights["head_w"] = rng.normal(size=(n_classes, channels, 1)).astype(np.float32) * 0.3
    weights["head_b"] = rng.normal(size=n_classes).astype(np.float32) * 0.1
    spec = ModelSpec(
        "tcn",
        tuple(f"c{i}" for i in range(n_classes)),
        1,
        n_features,
        tcn_arch(channels, dilations, kernel),
        {},
    )
    return spec, weights


# -- format -------------------------------------------------------------------------------------


def test_npz_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    spec, weights = random_mlp(rng)
    path = save_npz(tmp_path / "m.npz", spec, weights)
    back_spec, back_weights = load_npz(path)
    assert back_spec == spec
    assert set(back_weights) == set(weights)
    for k in weights:
        np.testing.assert_allclose(back_weights[k], weights[k])


def test_load_model_refuses_a_stale_feature_spec(tmp_path):
    rng = np.random.default_rng(0)
    spec, weights = random_mlp(rng)
    save_npz(tmp_path / "m.npz", spec, weights)
    with pytest.raises(ModelFormatError, match="feature spec"):
        load_model(tmp_path / "m.npz", feature_spec_version=spec.feature_spec_version + 1)
    assert isinstance(load_model(tmp_path / "m.npz", feature_spec_version=1), MLP)


def test_rejects_a_foreign_npz(tmp_path):
    path = tmp_path / "other.npz"
    np.savez(path, w0=np.zeros(3))
    with pytest.raises(ModelFormatError):
        load_npz(path)


# -- arithmetic ---------------------------------------------------------------------------------


def test_softmax_is_a_distribution():
    x = np.array([[1.0, 2.0, 3.0], [-50.0, 0.0, 50.0]])
    p = softmax(x, axis=-1)
    np.testing.assert_allclose(p.sum(axis=-1), 1.0)
    assert np.all(p >= 0)


def test_mlp_batch_matches_single_rows():
    rng = np.random.default_rng(1)
    spec, weights = random_mlp(rng)
    net = MLP(spec, weights)
    x = rng.normal(size=(5, spec.n_features)).astype(np.float32)
    batch = net.predict(x)
    for i in range(len(x)):
        np.testing.assert_allclose(net.predict(x[i]), batch[i], rtol=1e-10, atol=1e-12)


def test_tcn_is_causal():
    """Output t must not move when inputs after t change -- the property streaming relies on."""
    rng = np.random.default_rng(2)
    spec, weights = random_tcn(rng)
    net = TCN(spec, weights)
    x = rng.normal(size=(40, spec.n_features)).astype(np.float32)
    y = net.logits(x)
    x2 = x.copy()
    x2[25:] = rng.normal(size=(15, spec.n_features))
    np.testing.assert_allclose(net.logits(x2)[:25], y[:25], rtol=1e-9, atol=1e-10)


def test_tcn_prefix_equals_full_sequence():
    """Scoring a prefix gives the same last row as scoring the whole sequence.

    This is why a live stream warming up reproduces the offline evaluation exactly.
    """
    rng = np.random.default_rng(3)
    spec, weights = random_tcn(rng)
    net = TCN(spec, weights)
    x = rng.normal(size=(30, spec.n_features)).astype(np.float32)
    full = net.logits(x)
    for t in (0, 1, 5, 17, 29):
        np.testing.assert_allclose(net.logits(x[: t + 1])[-1], full[t], rtol=1e-9, atol=1e-10)


def test_receptive_field_matches_the_network():
    rng = np.random.default_rng(4)
    spec, weights = random_tcn(rng, dilations=(1, 2, 4), kernel=3)
    net = TCN(spec, weights)
    assert net.receptive_field == receptive_field((1, 2, 4), 3) == 15
    x = np.zeros((40, spec.n_features), dtype=np.float32)
    x[40 - net.receptive_field] = 5.0  # just inside the receptive field of the last output
    moved = net.logits(x)[-1]
    x2 = np.zeros_like(x)
    x2[40 - net.receptive_field - 1] = 5.0  # just outside it
    unmoved = net.logits(x2)[-1]
    base = net.logits(np.zeros_like(x))[-1]
    assert not np.allclose(moved, base)
    np.testing.assert_allclose(unmoved, base, rtol=1e-9, atol=1e-10)


# -- torch parity -------------------------------------------------------------------------------


def test_numpy_mlp_matches_torch():
    torch = pytest.importorskip("torch")
    from actus.models.nn import build_torch_mlp, mlp_weights

    rng = np.random.default_rng(5)
    hidden, n_features, n_classes = [16, 8], 12, 3
    module = build_torch_mlp(n_features, hidden, n_classes).eval()
    mean = rng.normal(size=n_features).astype(np.float32)
    std = (rng.random(n_features) + 0.5).astype(np.float32)
    spec = ModelSpec("mlp", ("a", "b", "c"), 1, n_features, mlp_arch(hidden), {})
    net = MLP(spec, mlp_weights(module, mean, std))

    x = rng.normal(size=(7, n_features)).astype(np.float32)
    with torch.no_grad():
        ref = module(torch.from_numpy((x - mean) / np.maximum(std, 1e-6))).numpy()
    np.testing.assert_allclose(net.logits(x), ref, rtol=1e-4, atol=1e-5)


def test_numpy_tcn_matches_torch():
    torch = pytest.importorskip("torch")
    from actus.models.nn import build_torch_tcn, tcn_weights

    rng = np.random.default_rng(6)
    n_features, channels, n_classes, dilations, kernel = 9, 12, 4, (1, 2, 4), 3
    module = build_torch_tcn(n_features, channels, n_classes, dilations, kernel).eval()
    mean = rng.normal(size=n_features).astype(np.float32)
    std = (rng.random(n_features) + 0.5).astype(np.float32)
    spec = ModelSpec("tcn", tuple("abcd"), 1, n_features, tcn_arch(channels, dilations, kernel), {})
    net = TCN(spec, tcn_weights(module, mean, std))

    x = rng.normal(size=(35, n_features)).astype(np.float32)
    with torch.no_grad():
        ref = module(torch.from_numpy(((x - mean) / np.maximum(std, 1e-6))[None]).float())
    np.testing.assert_allclose(net.logits(x), ref.numpy()[0], rtol=1e-4, atol=1e-4)
