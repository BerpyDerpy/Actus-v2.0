"""Runtime wrapper for the hand-object interaction classifier.

Per (hand, object) pair it predicts ``none`` / ``touching`` / ``holding`` from the pair features
in ``actus.models.features``. Its output is folded into the world model as
``SpatialEstimator.update(hoi={object: p(holding)})``, where it sharpens the ``<object>.held``
predicate that the protocol's ``requires_hoi`` steps depend on.

Abstention is a feature, not an error
-------------------------------------
A pair is scored only when **both** the hand and the object were actually observed this frame.
An object nobody can see produces no entry in the returned mapping at all, so the world model
keeps its own belief instead of being handed a confident-looking guess. This matters more than
it sounds: the alternative -- predicting "held" from hand pose alone -- would let the model
infer the object's state from where we are in the procedure, which is precisely the protocol
prior leaking into evidence that is supposed to be prior-free (build plan section 2). The
classifier is allowed to say "I can't see it".

``coverage()`` reports how often it could answer, which is what the model card reports next to
the accuracy: a classifier that abstains on 90% of frames is not a working sensor, however good
its numbers on the other 10% look.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from actus.models.features import SPEC_VERSION, FrameFeatures
from actus.models.nn import MLP, ModelFormatError, load_npz, softmax

NONE, TOUCHING, HOLDING = "none", "touching", "holding"
HOI_CLASSES: tuple[str, ...] = (NONE, TOUCHING, HOLDING)


@dataclass(frozen=True)
class HOIResult:
    """One frame of pair predictions."""

    frame_idx: int
    t: float
    hands: tuple[str, ...]
    objects: tuple[str, ...]
    probs: np.ndarray  # (H, O, 3), rows for invalid pairs are zero
    valid: np.ndarray  # (H, O) bool

    def holding(self) -> dict[str, float]:
        """``{object: p(holding)}`` for the objects that were actually observed.

        Taken over hands as a max: either hand holding the object means it is held. Objects with
        no valid pair are omitted, which the world model reads as "no information".
        """
        out: dict[str, float] = {}
        hi = HOI_CLASSES.index(HOLDING)
        for o, name in enumerate(self.objects):
            rows = self.valid[:, o]
            if rows.any():
                out[name] = float(np.max(self.probs[rows, o, hi]))
        return out

    def touching(self) -> dict[str, float]:
        out: dict[str, float] = {}
        ti = HOI_CLASSES.index(TOUCHING)
        for o, name in enumerate(self.objects):
            rows = self.valid[:, o]
            if rows.any():
                out[name] = float(np.max(self.probs[rows, o, ti]))
        return out

    def label(self, hand: int, obj: int) -> str | None:
        if not self.valid[hand, obj]:
            return None
        return HOI_CLASSES[int(np.argmax(self.probs[hand, obj]))]


class HOIModel:
    """A trained per-pair classifier. Stateless: every frame is scored on its own features."""

    def __init__(
        self,
        net: MLP | Any,
        classes: Sequence[str],
        *,
        hands: Sequence[str],
        objects: Sequence[str],
        n_features: int,
        onnx_session: Any = None,
    ) -> None:
        if tuple(classes) != HOI_CLASSES:
            raise ModelFormatError(f"HOI classes must be {HOI_CLASSES}, got {tuple(classes)}")
        self.net = net
        self.classes = tuple(classes)
        self.hands = tuple(hands)
        self.objects = tuple(objects)
        self.n_features = int(n_features)
        self._onnx = onnx_session
        self._seen = 0
        self._answered = 0

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        prefer_onnx: bool = True,
        feature_spec_version: int | None = SPEC_VERSION,
    ) -> HOIModel:
        path = Path(path)
        npz = path.with_suffix(".npz")
        spec, weights = load_npz(npz)
        if spec.kind != "mlp":
            raise ModelFormatError(f"{npz} is a {spec.kind!r} model, expected 'mlp'")
        if feature_spec_version is not None and spec.feature_spec_version != feature_spec_version:
            raise ModelFormatError(
                f"{npz} was trained against feature spec v{spec.feature_spec_version}, "
                f"this code is v{feature_spec_version}; re-extract features and retrain"
            )
        from actus.models.temporal import _try_onnx

        return cls(
            MLP(spec, weights),
            spec.classes,
            hands=spec.arch.get("hands", ("left", "right")),
            objects=spec.arch.get("objects", ()),
            n_features=spec.n_features,
            onnx_session=_try_onnx(path.with_suffix(".onnx")) if prefer_onnx else None,
        )

    @property
    def backend(self) -> str:
        return "onnx" if self._onnx is not None else "numpy"

    def reset(self) -> None:
        self._seen = self._answered = 0

    def coverage(self) -> float:
        """Share of (hand, object) pair slots the model could actually score."""
        return self._answered / self._seen if self._seen else 0.0

    # -- inference -----------------------------------------------------------------------------

    def predict(self, features: FrameFeatures) -> HOIResult:
        pair, valid = features.pair, features.pair_valid
        if pair.shape[-1] != self.n_features:
            raise ValueError(f"model expects {self.n_features} features, got {pair.shape[-1]}")
        n_h, n_o = valid.shape
        probs = np.zeros((n_h, n_o, len(self.classes)), dtype=np.float64)
        rows = np.argwhere(valid)
        self._seen += valid.size
        self._answered += len(rows)
        if len(rows):
            batch = np.stack([pair[h, o] for h, o in rows]).astype(np.float32)
            out = self._forward(batch)
            for (h, o), p in zip(rows, out, strict=True):
                probs[h, o] = p
        return HOIResult(
            features.frame_idx,
            features.t,
            self.hands[:n_h],
            self.objects[:n_o],
            probs,
            np.asarray(valid, dtype=bool),
        )

    def predict_rows(self, rows: np.ndarray) -> np.ndarray:
        """Score a plain ``(N, F)`` batch of pair features. Used by the training report."""
        return self._forward(np.asarray(rows, dtype=np.float32))

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        if self._onnx is not None:
            out = self._onnx.run(None, {"x": batch})[0]
            return softmax(np.asarray(out, dtype=np.float64), axis=-1)
        return self.net.predict(batch)
