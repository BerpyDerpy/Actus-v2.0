"""Reading the feature cache, splitting it by operator, and scoring predictions.

Shared by ``tools/train_temporal.py`` and ``tools/train_hoi.py`` so both report the same numbers
the same way. Nothing here imports torch: the metrics are also what ``tools/evaluate.py`` and
the tests can use on the base install.

Splitting is **by operator**, never by frame or by take. Frames from one take are near-duplicates
of each other, and one operator's takes share a rig layout, a shirt and a set of habits, so a
frame-level split would report a number that says nothing about a new person. op05 is the
held-out operator (build plan, H-FILM).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from actus.core.contracts import IDLE, OTHER


@dataclass(frozen=True)
class TakeFeatures:
    """One take's cached features and labels."""

    take: str
    operator: str
    scenario: str
    frame_idx: np.ndarray  # (N,)
    t: np.ndarray  # (N,)
    frame: np.ndarray  # (N, F)
    pair: np.ndarray  # (N, H, O, P)
    pair_valid: np.ndarray  # (N, H, O)
    label: np.ndarray  # (N,) index into label_names
    rack_valid: np.ndarray  # (N,)
    det_seen: np.ndarray  # (N, K)

    def __len__(self) -> int:
        return len(self.frame_idx)


@dataclass(frozen=True)
class FeatureCache:
    manifest: Mapping
    takes: tuple[TakeFeatures, ...]

    @property
    def label_names(self) -> tuple[str, ...]:
        return tuple(self.manifest["label_names"])

    @property
    def frame_names(self) -> tuple[str, ...]:
        return tuple(self.manifest["frame_names"])

    @property
    def pair_names(self) -> tuple[str, ...]:
        return tuple(self.manifest["pair_names"])

    @property
    def objects(self) -> tuple[str, ...]:
        return tuple(self.manifest["objects"])

    @property
    def hands(self) -> tuple[str, ...]:
        return tuple(self.manifest["hands"])

    @property
    def operators(self) -> tuple[str, ...]:
        return tuple(sorted({t.operator for t in self.takes}))

    def by_operators(self, operators: Iterable[str]) -> tuple[TakeFeatures, ...]:
        wanted = set(operators)
        return tuple(t for t in self.takes if t.operator in wanted)


def load_cache(path: str | Path, *, takes: Sequence[str] | None = None) -> FeatureCache:
    path = Path(path)
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no feature cache at {path}; run tools/extract_features.py first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    loaded: list[TakeFeatures] = []
    for entry in manifest["takes"]:
        stem = entry["take"]
        if takes and not any(f in stem for f in takes):
            continue
        npz = path / f"{stem}.npz"
        if not npz.exists():
            continue
        with np.load(npz) as z:
            loaded.append(
                TakeFeatures(
                    stem,
                    entry["operator"],
                    entry["scenario"],
                    z["frame_idx"],
                    z["t"],
                    z["frame"],
                    z["pair"],
                    z["pair_valid"],
                    z["label"],
                    z["rack_valid"],
                    z["det_seen"],
                )
            )
    if not loaded:
        raise FileNotFoundError(f"feature cache at {path} has no usable takes")
    return FeatureCache(manifest, tuple(loaded))


# ----------------------------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------------------------


DEFAULT_TEST = ("op05",)
DEFAULT_VAL = ("op04",)


@dataclass(frozen=True)
class Split:
    name: str
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def describe(self) -> str:
        return f"{self.name}: train={list(self.train)} val={list(self.val)} test={list(self.test)}"


def operator_split(
    operators: Sequence[str],
    *,
    test: Sequence[str] = DEFAULT_TEST,
    val: Sequence[str] = DEFAULT_VAL,
) -> Split:
    ops = list(operators)
    test_ops = tuple(o for o in ops if o in set(test))
    val_ops = tuple(o for o in ops if o in set(val))
    train_ops = tuple(o for o in ops if o not in set(test_ops) | set(val_ops))
    if not train_ops:
        raise ValueError(f"no training operators left from {ops}")
    return Split("fixed", train_ops, val_ops, test_ops)


def leave_one_out_splits(operators: Sequence[str]) -> list[Split]:
    """One split per operator held out, the next operator used for validation.

    With five operators and a rig that changed between them, a single fixed split is one sample
    of a very small distribution. Reporting every fold makes the spread visible instead of
    letting the choice of held-out operator flatter the result.
    """
    ops = list(operators)
    out = []
    for i, held in enumerate(ops):
        val = ops[(i + 1) % len(ops)]
        train = tuple(o for o in ops if o not in (held, val))
        if train:
            out.append(Split(f"loo_{held}", train, (val,), (held,)))
    return out


# ----------------------------------------------------------------------------------------------
# Label handling
# ----------------------------------------------------------------------------------------------


def action_classes(label_names: Sequence[str]) -> tuple[str, ...]:
    """The temporal model's output vocabulary: the protocol's steps plus ``idle``.

    ``ActionScores`` is defined over ``{step_id, ..., "idle"}``, so the annotation's ``other``
    (a deliberate non-step action) is folded into ``idle``: from the engine's point of view both
    mean "no step is in progress". They stay distinct in the annotations.
    """
    return tuple(n for n in label_names if n != OTHER)


def map_labels(labels: np.ndarray, label_names: Sequence[str]) -> np.ndarray:
    """Map cache label indices onto ``action_classes`` indices (``other`` -> ``idle``)."""
    classes = action_classes(label_names)
    lookup = {}
    for i, name in enumerate(label_names):
        target = IDLE if name == OTHER else name
        lookup[i] = classes.index(target)
    return np.array([lookup[int(v)] for v in labels], dtype=np.int64)


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------


def confusion(true: np.ndarray, pred: np.ndarray, n: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=np.int64)
    np.add.at(m, (true.astype(int), pred.astype(int)), 1)
    return m


def frame_accuracy(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(true == pred)) if len(true) else 0.0


def per_class_f1(true: np.ndarray, pred: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float64)
    for c in range(n):
        tp = float(np.sum((pred == c) & (true == c)))
        fp = float(np.sum((pred == c) & (true != c)))
        fn = float(np.sum((pred != c) & (true == c)))
        out[c] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return out


def macro_f1(true: np.ndarray, pred: np.ndarray, n: int, *, ignore: Sequence[int] = ()) -> float:
    f1 = per_class_f1(true, pred, n)
    keep = [c for c in range(n) if c not in set(ignore) and np.any(true == c)]
    return float(np.mean(f1[keep])) if keep else 0.0


def segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous runs as ``(class, start, end_exclusive)``."""
    if len(labels) == 0:
        return []
    bounds = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(labels)]])
    return [(int(labels[s]), int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def segmental_f1(
    true: np.ndarray, pred: np.ndarray, *, overlap: float, ignore: Sequence[int] = ()
) -> float:
    """Segmental F1 at an IoU threshold (Lea et al.), the standard action-segmentation metric.

    Frame accuracy rewards getting the long ``idle`` stretches right and barely notices a step
    boundary landing a second late. This scores whole segments: each predicted segment matches
    at most one ground-truth segment of the same class with IoU >= ``overlap``.
    """
    ignore = set(ignore)
    true_segs = [s for s in segments(true) if s[0] not in ignore]
    pred_segs = [s for s in segments(pred) if s[0] not in ignore]
    used = [False] * len(true_segs)
    tp = 0
    for cls, ps, pe in pred_segs:
        best, best_iou = -1, 0.0
        for i, (tcls, ts, te) in enumerate(true_segs):
            if tcls != cls or used[i]:
                continue
            inter = max(0, min(pe, te) - max(ps, ts))
            union = max(pe, te) - min(ps, ts)
            iou = inter / union if union else 0.0
            if iou > best_iou:
                best, best_iou = i, iou
        if best >= 0 and best_iou >= overlap:
            used[best] = True
            tp += 1
    fp = len(pred_segs) - tp
    fn = len(true_segs) - tp
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def smooth_predictions(pred: np.ndarray, width: int) -> np.ndarray:
    """Median filter over time, to remove single-frame label flicker before segmental scoring."""
    if width <= 1 or len(pred) == 0:
        return pred
    half = width // 2
    padded = np.pad(pred, (half, half), mode="edge")
    out = np.empty_like(pred)
    for i in range(len(pred)):
        window = padded[i : i + 2 * half + 1]
        vals, counts = np.unique(window, return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out
