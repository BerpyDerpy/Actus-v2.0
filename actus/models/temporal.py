"""Runtime wrapper for the temporal action model: features in, ``ActionScores`` out.

This is the HAR model the problem statement asks for. It is a sensor, not the judge: it says
which step *appears to be in progress*, and the protocol engine uses that only to date a step's
onset (``StepEvent.t_start``) and, through ``actus.models.fusion``, as one bounded evidence
source among several. It never completes a step by itself.

Streaming without drifting from training
----------------------------------------
Two things keep the live answer identical to the offline one:

* **A fixed feature rate.** Features arrive at whatever rate the camera runs; the model consumes
  a grid at ``feature_fps`` (the rate it was trained on), filled with the most recent frame at
  each grid instant. A 30 fps take and a 15 fps live camera therefore produce the same sequence,
  which is why the model does not have to be retrained per camera.
* **Causality.** The network is a causal TCN: output ``t`` depends only on inputs up to ``t``.
  Running it over a growing buffer gives bit-for-bit the same value as running it over the whole
  take offline and reading position ``t`` -- including during warm-up, when the buffer is still
  shorter than the receptive field. That is what makes ``tools/evaluate.py`` numbers meaningful
  for the live system.

Backends: ONNX through onnxruntime when available (fast), else the numpy ``TCN`` in
``actus.models.nn``. Both are exercised by the tests, and the parity test pins them together.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from actus.core.contracts import IDLE, ActionScores
from actus.models.features import SPEC_VERSION, FrameFeatures
from actus.models.nn import TCN, ModelFormatError, load_npz, softmax

DEFAULT_FEATURE_FPS = 15.0


class TemporalModel:
    """A trained TCN over per-frame features, driven one frame at a time."""

    def __init__(
        self,
        net: TCN | Any,
        classes: Sequence[str],
        *,
        feature_fps: float = DEFAULT_FEATURE_FPS,
        n_features: int,
        receptive_field: int,
        onnx_session: Any = None,
    ) -> None:
        self.net = net
        self.classes = tuple(classes)
        self.feature_fps = float(feature_fps)
        self.n_features = int(n_features)
        self.receptive_field = int(receptive_field)
        self._onnx = onnx_session
        self.reset()

    # -- construction --------------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        prefer_onnx: bool = True,
        feature_spec_version: int | None = SPEC_VERSION,
    ) -> TemporalModel:
        """Load from ``<stem>.npz``; use the sibling ``<stem>.onnx`` when it is loadable.

        The ``.npz`` is always read because it carries the spec (classes, feature rate, the
        feature-spec version). ONNX only replaces the arithmetic.
        """
        path = Path(path)
        npz = path.with_suffix(".npz")
        spec, weights = load_npz(npz)
        if spec.kind != "tcn":
            raise ModelFormatError(f"{npz} is a {spec.kind!r} model, expected 'tcn'")
        if feature_spec_version is not None and spec.feature_spec_version != feature_spec_version:
            raise ModelFormatError(
                f"{npz} was trained against feature spec v{spec.feature_spec_version}, "
                f"this code is v{feature_spec_version}; re-extract features and retrain"
            )
        net = TCN(spec, weights)
        session = _try_onnx(path.with_suffix(".onnx")) if prefer_onnx else None
        return cls(
            net,
            spec.classes,
            feature_fps=float(spec.arch.get("feature_fps", DEFAULT_FEATURE_FPS)),
            n_features=spec.n_features,
            receptive_field=net.receptive_field,
            onnx_session=session,
        )

    @property
    def backend(self) -> str:
        return "onnx" if self._onnx is not None else "numpy"

    # -- streaming -----------------------------------------------------------------------------

    def reset(self) -> None:
        self._buf: deque[np.ndarray] = deque(maxlen=self.receptive_field)
        self._next_t: float | None = None
        self._last: np.ndarray | None = None

    def update(self, features: FrameFeatures) -> ActionScores | None:
        """Feed one frame. Returns scores when this frame advances the fixed-rate grid.

        Between grid instants it returns None rather than a repeat, so a consumer can tell a
        fresh estimate from a stale one. ``latest()`` re-reads the last scores at any time.
        """
        if features.frame.shape[-1] != self.n_features:
            raise ValueError(
                f"model expects {self.n_features} features, got {features.frame.shape[-1]}"
            )
        step = 1.0 / self.feature_fps
        if self._next_t is None:
            self._next_t = features.t
        if features.t + 1e-9 < self._next_t:
            return None
        # Catch up if frames were dropped, so the grid stays at feature_fps. Bounded by the
        # receptive field: after a long stall there is nothing useful to reconstruct, and the
        # buffer would discard the older samples anyway.
        for _ in range(self.receptive_field):
            self._buf.append(np.asarray(features.frame, dtype=np.float32))
            self._next_t += step
            if features.t + 1e-9 < self._next_t:
                break
        else:
            self._next_t = features.t + step
        scores = self._infer()
        self._last = scores
        return ActionScores(
            features.frame_idx, features.t, dict(zip(self.classes, scores, strict=True))
        )

    def latest(self) -> np.ndarray | None:
        return self._last

    def _infer(self) -> np.ndarray:
        window = np.stack(self._buf).astype(np.float32)
        if self._onnx is not None:
            out = self._onnx.run(None, {"x": window[None]})[0]
            return softmax(np.asarray(out[0][-1], dtype=np.float64))
        return softmax(self.net.logits(window)[-1])

    # -- offline -------------------------------------------------------------------------------

    def predict_sequence(self, frames: np.ndarray) -> np.ndarray:
        """Score a whole take at once: ``(T, F)`` features in, ``(T, C)`` probabilities out.

        Causality means row ``t`` here equals what ``update`` returns at frame ``t`` live, which
        is what lets the offline evaluation stand in for the live system.
        """
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim != 2 or frames.shape[1] != self.n_features:
            raise ValueError(f"expected (T, {self.n_features}), got {frames.shape}")
        if self._onnx is not None:
            out = self._onnx.run(None, {"x": frames[None]})[0][0]
            return softmax(np.asarray(out, dtype=np.float64), axis=-1)
        return self.net.predict(frames)

    def idle_index(self) -> int:
        return self.classes.index(IDLE) if IDLE in self.classes else -1


def resample_to_grid(
    t: np.ndarray, frames: np.ndarray, fps: float
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``frames`` onto a fixed ``fps`` grid, holding the most recent row (zero-order hold).

    Used at training time so the model sees exactly what ``TemporalModel.update`` will feed it
    live. Returns the grid times and the resampled rows.
    """
    t = np.asarray(t, dtype=np.float64)
    frames = np.asarray(frames)
    if len(t) == 0:
        return t, frames
    step = 1.0 / float(fps)
    grid = np.arange(t[0], t[-1] + 1e-9, step)
    idx = np.searchsorted(t, grid, side="right") - 1
    idx = np.clip(idx, 0, len(t) - 1)
    return grid, frames[idx]


def _try_onnx(path: Path):
    if not path.exists():
        return None
    try:
        import onnxruntime as ort

        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception:  # pragma: no cover - optional dependency / bad export
        return None
