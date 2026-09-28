"""The two tiny networks T-HOI-TEMP trains, plus a dependency-free numpy forward pass.

Why two implementations of the same arithmetic
----------------------------------------------
Training needs an autograd framework; the runtime must not. So each architecture is defined
twice in this one file, deliberately side by side:

* ``build_torch_mlp`` / ``build_torch_tcn`` -- used only by ``tools/train_*.py`` (torch is
  imported inside the function, so the base install never pays for it).
* ``MLP`` / ``TCN`` -- numpy forward passes over the exported weights, with no heavy dependency
  at all.

``tests/test_models_nn.py`` asserts the two agree to within float tolerance, so the pair can't
silently drift. The runtime prefers ONNX (``onnxruntime``, fast) and falls back to these numpy
classes, which means the models still run — and the tests still cover them — on the base install
where neither torch nor onnxruntime is present.

The on-disk format
------------------
One ``.npz`` per model: a JSON ``spec`` string plus the weight arrays. The spec records the
feature-spec version the weights were trained against, so ``load`` refuses a model whose
features no longer match the running code rather than quietly feeding it the wrong columns.
An ``.onnx`` file sits next to it with the same stem and the same spec in its metadata.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FORMAT = "actus-model-1"


class ModelFormatError(ValueError):
    """The file is not an ACTUS model, or was trained against a different feature spec."""


# ----------------------------------------------------------------------------------------------
# Spec
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    kind: str  # "mlp" | "tcn"
    classes: tuple[str, ...]
    feature_spec_version: int
    n_features: int
    arch: Mapping[str, Any]
    meta: Mapping[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {
                "format": FORMAT,
                "kind": self.kind,
                "classes": list(self.classes),
                "feature_spec_version": self.feature_spec_version,
                "n_features": self.n_features,
                "arch": dict(self.arch),
                "meta": dict(self.meta),
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> ModelSpec:
        d = json.loads(text)
        if d.get("format") != FORMAT:
            raise ModelFormatError(f"not an {FORMAT} file (format={d.get('format')!r})")
        return cls(
            kind=str(d["kind"]),
            classes=tuple(d["classes"]),
            feature_spec_version=int(d["feature_spec_version"]),
            n_features=int(d["n_features"]),
            arch=dict(d.get("arch", {})),
            meta=dict(d.get("meta", {})),
        )


def save_npz(path: str | Path, spec: ModelSpec, weights: Mapping[str, np.ndarray]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
    np.savez(path, spec=np.array(spec.to_json()), **arrays)
    return path


def load_npz(path: str | Path) -> tuple[ModelSpec, dict[str, np.ndarray]]:
    with np.load(Path(path), allow_pickle=False) as z:
        if "spec" not in z:
            raise ModelFormatError(f"{path} has no spec entry")
        spec = ModelSpec.from_json(str(z["spec"].item()))
        weights = {k: np.asarray(z[k], dtype=np.float32) for k in z.files if k != "spec"}
    return spec, weights


# ----------------------------------------------------------------------------------------------
# numpy forward passes
# ----------------------------------------------------------------------------------------------


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _standardise(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / np.maximum(std, 1e-6)


class MLP:
    """``Linear -> ReLU`` stack with a linear head. Input is standardised by stored statistics."""

    def __init__(self, spec: ModelSpec, weights: Mapping[str, np.ndarray]) -> None:
        self.spec = spec
        self.mean = weights["mean"].astype(np.float64)
        self.std = weights["std"].astype(np.float64)
        n = int(spec.arch["n_layers"])
        self.layers = [
            (weights[f"w{i}"].astype(np.float64), weights[f"b{i}"].astype(np.float64))
            for i in range(n)
        ]

    def logits(self, x: np.ndarray) -> np.ndarray:
        """``x`` is (F,) or (N, F); returns (C,) or (N, C)."""
        single = x.ndim == 1
        h = _standardise(np.atleast_2d(np.asarray(x, dtype=np.float64)), self.mean, self.std)
        for i, (w, b) in enumerate(self.layers):
            h = h @ w.T + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        return h[0] if single else h

    def predict(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.logits(x), axis=-1)


def causal_conv1d(x: np.ndarray, w: np.ndarray, b: np.ndarray, dilation: int) -> np.ndarray:
    """``x`` is (T, C_in), ``w`` is (C_out, C_in, K); returns (T, C_out).

    Causal: output ``t`` sees inputs ``t - (K-1)*d .. t`` and never the future, so the streaming
    runtime gets the same answer for frame ``t`` that the offline pass does.
    """
    t_len, _ = x.shape
    c_out, _, k = w.shape
    pad = (k - 1) * dilation
    xp = np.concatenate([np.zeros((pad, x.shape[1]), dtype=x.dtype), x], axis=0)
    out = np.broadcast_to(b, (t_len, c_out)).astype(np.float64).copy()
    for j in range(k):
        start = j * dilation
        out += xp[start : start + t_len] @ w[:, :, j].T
    return out


class TCN:
    """Dilated causal temporal convolution network with residual blocks and a 1x1 head."""

    def __init__(self, spec: ModelSpec, weights: Mapping[str, np.ndarray]) -> None:
        self.spec = spec
        self.mean = weights["mean"].astype(np.float64)
        self.std = weights["std"].astype(np.float64)
        self.dilations = tuple(int(d) for d in spec.arch["dilations"])
        self.kernel = int(spec.arch["kernel"])
        self.blocks = []
        for i in range(len(self.dilations)):
            w = weights[f"conv{i}_w"].astype(np.float64)
            b = weights[f"conv{i}_b"].astype(np.float64)
            proj = weights.get(f"proj{i}_w")
            proj_b = weights.get(f"proj{i}_b")
            self.blocks.append(
                (
                    w,
                    b,
                    None if proj is None else proj.astype(np.float64),
                    None if proj_b is None else proj_b.astype(np.float64),
                )
            )
        self.head_w = weights["head_w"].astype(np.float64)
        self.head_b = weights["head_b"].astype(np.float64)

    @property
    def receptive_field(self) -> int:
        return 1 + sum((self.kernel - 1) * d for d in self.dilations)

    def logits(self, x: np.ndarray) -> np.ndarray:
        """``x`` is (T, F) in time order; returns (T, C)."""
        h = _standardise(np.asarray(x, dtype=np.float64), self.mean, self.std)
        for (w, b, proj, proj_b), d in zip(self.blocks, self.dilations, strict=True):
            y = np.maximum(causal_conv1d(h, w, b, d), 0.0)
            skip = h if proj is None else causal_conv1d(h, proj, proj_b, 1)
            h = y + skip
        return causal_conv1d(h, self.head_w, self.head_b, 1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.logits(x), axis=-1)


def load_model(path: str | Path, *, feature_spec_version: int | None = None):
    """Load an ``.npz`` model and return an ``MLP`` or ``TCN``."""
    spec, weights = load_npz(path)
    if feature_spec_version is not None and spec.feature_spec_version != feature_spec_version:
        raise ModelFormatError(
            f"{path} was trained against feature spec v{spec.feature_spec_version}, "
            f"this code is v{feature_spec_version}; re-run tools/extract_features.py and retrain"
        )
    if spec.kind == "mlp":
        return MLP(spec, weights)
    if spec.kind == "tcn":
        return TCN(spec, weights)
    raise ModelFormatError(f"unknown model kind {spec.kind!r}")


# ----------------------------------------------------------------------------------------------
# torch definitions (training only)
# ----------------------------------------------------------------------------------------------


def build_torch_mlp(n_features: int, hidden: Sequence[int], n_classes: int):
    """``nn.Sequential`` matching ``MLP``'s numpy forward pass exactly."""
    import torch.nn as nn

    sizes = [n_features, *hidden]
    layers: list[Any] = []
    for a, b in itertools.pairwise(sizes):
        layers += [nn.Linear(a, b), nn.ReLU()]
    layers.append(nn.Linear(sizes[-1], n_classes))
    return nn.Sequential(*layers)


def mlp_weights(module, mean: np.ndarray, std: np.ndarray) -> dict[str, np.ndarray]:
    import torch.nn as nn

    out: dict[str, np.ndarray] = {"mean": np.asarray(mean), "std": np.asarray(std)}
    i = 0
    for layer in module:
        if isinstance(layer, nn.Linear):
            out[f"w{i}"] = layer.weight.detach().cpu().numpy()
            out[f"b{i}"] = layer.bias.detach().cpu().numpy()
            i += 1
    return out


def mlp_arch(hidden: Sequence[int]) -> dict[str, Any]:
    return {"hidden": list(hidden), "n_layers": len(hidden) + 1}


def build_torch_tcn(
    n_features: int, channels: int, n_classes: int, dilations: Sequence[int], kernel: int = 3
):
    """Residual dilated causal TCN matching ``TCN``'s numpy forward pass exactly."""
    import torch
    import torch.nn as nn

    class CausalBlock(nn.Module):
        def __init__(self, c_in: int, c_out: int, dilation: int) -> None:
            super().__init__()
            self.pad = (kernel - 1) * dilation
            self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)
            self.proj = None if c_in == c_out else nn.Conv1d(c_in, c_out, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, C, T)
            y = torch.relu(self.conv(nn.functional.pad(x, (self.pad, 0))))
            return y + (x if self.proj is None else self.proj(x))

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            sizes = [n_features] + [channels] * len(dilations)
            self.blocks = nn.ModuleList(
                CausalBlock(sizes[i], sizes[i + 1], d) for i, d in enumerate(dilations)
            )
            self.head = nn.Conv1d(sizes[-1], n_classes, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, T, F) -> (N, T, C)
            h = x.transpose(1, 2)
            for blk in self.blocks:
                h = blk(h)
            return self.head(h).transpose(1, 2)

    return Net()


def tcn_weights(module, mean: np.ndarray, std: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {"mean": np.asarray(mean), "std": np.asarray(std)}
    for i, blk in enumerate(module.blocks):
        out[f"conv{i}_w"] = blk.conv.weight.detach().cpu().numpy()
        out[f"conv{i}_b"] = blk.conv.bias.detach().cpu().numpy()
        if blk.proj is not None:
            out[f"proj{i}_w"] = blk.proj.weight.detach().cpu().numpy()
            out[f"proj{i}_b"] = blk.proj.bias.detach().cpu().numpy()
    out["head_w"] = module.head.weight.detach().cpu().numpy()
    out["head_b"] = module.head.bias.detach().cpu().numpy()
    return out


def tcn_arch(channels: int, dilations: Sequence[int], kernel: int) -> dict[str, Any]:
    return {"channels": channels, "dilations": list(dilations), "kernel": kernel}


def receptive_field(dilations: Sequence[int], kernel: int) -> int:
    return 1 + sum((kernel - 1) * int(d) for d in dilations)


def export_onnx(
    module, path: str | Path, *, example_shape: tuple[int, ...], dynamic_time: bool = False
) -> Path | None:
    """Best-effort ONNX export. Returns None (with a printed reason) when ``onnx`` is missing.

    The ``.npz`` is the model of record: the runtime falls back to it, so a failed ONNX export
    costs speed, not correctness.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(*example_shape)
    dynamic_axes = {"x": {0: "batch", 1: "time"}, "y": {0: "batch", 1: "time"}}
    try:
        torch.onnx.export(
            module.eval(),
            (example,),
            str(path),
            input_names=["x"],
            output_names=["y"],
            dynamic_axes=dynamic_axes if dynamic_time else {"x": {0: "batch"}, "y": {0: "batch"}},
            dynamo=False,
        )
    except Exception as exc:  # pragma: no cover - depends on the optional onnx package
        print(f"  ONNX export skipped: {type(exc).__name__}: {exc}")
        return None
    return path
