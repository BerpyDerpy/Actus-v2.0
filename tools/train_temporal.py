#!/usr/bin/env python3
"""Train the temporal action model -- the HAR model the problem statement asks for (T-HOI-TEMP).

    uv run python tools/train_temporal.py                  # fixed operator split, 3 seeds
    uv run python tools/train_temporal.py --loo            # leave-one-operator-out, every fold
    uv run python tools/train_temporal.py --seeds 5 --epochs 60

A causal dilated TCN over 2 s windows of the cached features, predicting which protocol step is
in progress (or idle). Reads ``data/features`` (see ``tools/extract_features.py``), writes
``models/temporal/temporal.npz``, a matching ``.onnx``, a ``metrics.json`` and a ``card.md``.

How the numbers are kept honest
-------------------------------
* **Split by operator.** Never by frame or by take. op05 is held out (build plan, H-FILM).
* **Seed-averaged.** Every split is trained ``--seeds`` times and the report gives mean and
  spread, because with five operators a single run's number is mostly noise.
* **Baselines alongside.** Majority class and a duration-only prior are scored on the same test
  frames. A model that cannot beat "guess from how far through the take we are" has not learned
  the activity, and on a dataset where every take runs the same six steps in nearly the same
  order that baseline is strong. It is reported whatever it says.
* **Segmental F1, not just frame accuracy.** Frames are dominated by long idle stretches, so
  frame accuracy flatters everything.
* **Loss is masked over each window's warm-up.** The first ``receptive_field - 1`` positions of
  a training window see zero-padded context that never occurs mid-take, so they are excluded
  rather than teaching the model to trust a padded history.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from actus.core.contracts import IDLE  # noqa: E402
from actus.models.dataset import (  # noqa: E402
    Split,
    TakeFeatures,
    action_classes,
    confusion,
    frame_accuracy,
    leave_one_out_splits,
    load_cache,
    macro_f1,
    map_labels,
    operator_split,
    per_class_f1,
    segmental_f1,
    smooth_predictions,
)
from actus.models.nn import (  # noqa: E402
    ModelSpec,
    build_torch_tcn,
    export_onnx,
    receptive_field,
    save_npz,
    tcn_arch,
    tcn_weights,
)
from actus.models.temporal import DEFAULT_FEATURE_FPS, resample_to_grid  # noqa: E402

DEFAULT_DILATIONS = (1, 2, 4, 8)
OVERLAPS = (0.10, 0.25, 0.50)


# ----------------------------------------------------------------------------------------------
# Sequences
# ----------------------------------------------------------------------------------------------


class Sequences:
    """Per-take (features, labels) resampled onto the model's fixed feature-rate grid."""

    def __init__(self, takes: Sequence[TakeFeatures], label_names, fps: float) -> None:
        self.classes = action_classes(label_names)
        self.fps = float(fps)
        self.items: list[tuple[str, np.ndarray, np.ndarray]] = []
        for tk in takes:
            labels = map_labels(tk.label, label_names)
            grid, frames = resample_to_grid(tk.t, tk.frame, fps)
            _, lab = resample_to_grid(tk.t, labels, fps)
            if len(grid) > 1:
                self.items.append((tk.take, frames.astype(np.float32), lab.astype(np.int64)))

    def __len__(self) -> int:
        return len(self.items)

    @property
    def n_features(self) -> int:
        return self.items[0][1].shape[1] if self.items else 0

    def stacked(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.concatenate([f for _, f, _ in self.items]) if self.items else np.zeros((0, 0))
        y = np.concatenate([b for _, _, b in self.items]) if self.items else np.zeros(0, int)
        return x, y

    def statistics(self) -> tuple[np.ndarray, np.ndarray]:
        x, _ = self.stacked()
        return x.mean(axis=0), x.std(axis=0) + 1e-6

    def class_weights(self, n_classes: int) -> np.ndarray:
        _, y = self.stacked()
        counts = np.bincount(y, minlength=n_classes).astype(np.float64)
        w = np.where(counts > 0, 1.0 / np.maximum(counts, 1.0), 0.0)
        return (w / w.sum() * np.count_nonzero(counts)).astype(np.float32)


def sample_windows(seqs: Sequences, length: int, batch: int, rng: np.random.Generator):
    """Random fixed-length windows, one batch. Short takes are left-padded like a stream start."""
    xs, ys, ms = [], [], []
    for _ in range(batch):
        _, f, lab = seqs.items[rng.integers(len(seqs.items))]
        if len(f) <= length:
            pad = length - len(f)
            x = np.concatenate([np.zeros((pad, f.shape[1]), np.float32), f])
            y = np.concatenate([np.zeros(pad, np.int64), lab])
            mask = np.concatenate([np.zeros(pad, bool), np.ones(len(f), bool)])
        else:
            start = int(rng.integers(0, len(f) - length))
            x, y = f[start : start + length], lab[start : start + length]
            mask = np.ones(length, bool)
        xs.append(x)
        ys.append(y)
        ms.append(mask)
    return np.stack(xs), np.stack(ys), np.stack(ms)


# ----------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------


def train_one(
    train: Sequences,
    val: Sequences,
    *,
    classes: Sequence[str],
    seed: int,
    args,
    verbose: bool = False,
):
    """Train one TCN. Returns (numpy-ready weights, arch, history)."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")

    n_features = train.n_features
    n_classes = len(classes)
    dilations = tuple(args.dilations)
    rf = receptive_field(dilations, args.kernel)
    mean, std = train.statistics()
    weights = torch.tensor(train.class_weights(n_classes), device=device)

    net = build_torch_tcn(n_features, args.channels, n_classes, dilations, args.kernel).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights, reduction="none")

    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(std, dtype=torch.float32, device=device)
    window = max(args.window, rf * 2)

    best = {"score": -1.0, "state": None, "epoch": -1}
    history = []
    for epoch in range(args.epochs):
        net.train()
        total = 0.0
        for _ in range(args.steps_per_epoch):
            x, y, m = sample_windows(train, window, args.batch, rng)
            xb = (torch.tensor(x, device=device) - mean_t) / std_t
            yb = torch.tensor(y, device=device)
            mb = torch.tensor(m, device=device)
            mb[:, : rf - 1] = False  # warm-up positions see padding that never occurs mid-take
            logits = net(xb)
            loss = loss_fn(logits.reshape(-1, n_classes), yb.reshape(-1))
            loss = (loss * mb.reshape(-1)).sum() / mb.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
        score = _score(net, val or train, mean_t, std_t, device, n_classes, classes)
        history.append(
            {"epoch": epoch, "loss": total / args.steps_per_epoch, "val_macro_f1": score}
        )
        if verbose:
            print(
                f"    epoch {epoch:3d} loss {total / args.steps_per_epoch:.4f} val_f1 {score:.4f}"
            )
        if score > best["score"]:
            best = {
                "score": score,
                "state": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                "epoch": epoch,
            }
        elif args.patience and epoch - best["epoch"] >= args.patience:
            break

    if best["state"] is not None:
        net.load_state_dict(best["state"])
    net.eval().to("cpu")
    history.append({"best_epoch": best["epoch"], "best_val_macro_f1": best["score"]})
    return (
        net,
        tcn_weights(net, mean, std),
        tcn_arch(args.channels, dilations, args.kernel),
        history,
    )


def _score(net, seqs: Sequences, mean_t, std_t, device, n_classes, classes) -> float:
    import torch

    net.eval()
    trues, preds = [], []
    with torch.no_grad():
        for _, f, lab in seqs.items:
            xb = (torch.tensor(f[None], device=device) - mean_t) / std_t
            preds.append(net(xb)[0].argmax(dim=-1).cpu().numpy())
            trues.append(lab)
    net.train()
    if not trues:
        return 0.0
    true, pred = np.concatenate(trues), np.concatenate(preds)
    idle = classes.index(IDLE) if IDLE in classes else -1
    return macro_f1(true, pred, n_classes, ignore=(idle,) if idle >= 0 else ())


# ----------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------


def predict_sequences(
    model, seqs: Sequences, smooth: int
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    out = []
    for name, f, lab in seqs.items:
        probs = model.predict_sequence(f)
        pred = smooth_predictions(probs.argmax(axis=-1), smooth)
        out.append((name, lab, pred))
    return out


def evaluate(per_take, classes: Sequence[str]) -> dict:
    n = len(classes)
    idle = classes.index(IDLE) if IDLE in classes else -1
    ignore = (idle,) if idle >= 0 else ()
    true = np.concatenate([t for _, t, _ in per_take]) if per_take else np.zeros(0, int)
    pred = np.concatenate([p for _, _, p in per_take]) if per_take else np.zeros(0, int)
    seg = {
        f"segmental_f1@{int(o * 100)}": float(
            np.mean([segmental_f1(t, p, overlap=o, ignore=ignore) for _, t, p in per_take])
        )
        for o in OVERLAPS
    }
    return {
        "frames": len(true),
        "frame_accuracy": frame_accuracy(true, pred),
        "macro_f1_steps": macro_f1(true, pred, n, ignore=ignore),
        **seg,
        "per_class_f1": {
            c: float(v) for c, v in zip(classes, per_class_f1(true, pred, n), strict=True)
        },
        "confusion": confusion(true, pred, n).tolist(),
    }


def baseline_majority(train: Sequences, test: Sequences, classes) -> dict:
    _, y = train.stacked()
    top = int(np.bincount(y, minlength=len(classes)).argmax())
    per = [(name, lab, np.full_like(lab, top)) for name, _, lab in test.items]
    return evaluate(per, classes)


def baseline_duration(train: Sequences, test: Sequences, classes, bins: int = 20) -> dict:
    """Predict the step from how far through the take we are, and nothing else.

    On a dataset where every take runs the same procedure at a similar pace this is a genuinely
    hard baseline, which is the point of reporting it.
    """
    table = np.zeros((bins, len(classes)), dtype=np.float64)
    for _, _f, lab in train.items:
        pos = np.linspace(0, bins - 1e-9, len(lab)).astype(int)
        np.add.at(table, (pos, lab), 1)
    argmax = table.argmax(axis=1)
    per = []
    for name, _f, lab in test.items:
        pos = np.linspace(0, bins - 1e-9, len(lab)).astype(int)
        per.append((name, lab, argmax[pos]))
    return evaluate(per, classes)


# ----------------------------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------------------------


def diagnostic_within_operator(cache, args, classes) -> dict:
    """Train and test on *different takes of the same operators*. Not a headline number.

    Splitting this way leaks operator identity, the rig layout and the clothing, so it always
    flatters the model and must never be quoted as a result. It is here to separate two
    explanations of a weak held-out score: features that carry no usable signal, or features
    that carry signal which does not transfer between operators. A high number here with a low
    held-out one points at transfer -- which is what the rig changing between operators and the
    per-operator differences in hand visibility would predict.
    """
    from actus.models.nn import TCN
    from actus.models.temporal import TemporalModel

    takes = list(cache.takes)
    test_takes = [t for i, t in enumerate(takes) if i % 3 == 0]
    train_takes = [t for i, t in enumerate(takes) if i % 3 != 0]
    names = cache.label_names
    train = Sequences(train_takes, names, args.feature_fps)
    test = Sequences(test_takes, names, args.feature_fps)
    if not len(train) or not len(test):
        return {}
    _net, weights, arch, _ = train_one(train, test, classes=classes, seed=0, args=args)
    spec = ModelSpec(
        "tcn",
        tuple(classes),
        cache.manifest["feature_spec_version"],
        train.n_features,
        {**arch, "feature_fps": args.feature_fps},
        {"split": "within_operator"},
    )
    model = TemporalModel(
        TCN(spec, weights),
        classes,
        feature_fps=args.feature_fps,
        n_features=train.n_features,
        receptive_field=receptive_field(tuple(args.dilations), args.kernel),
    )
    metrics = evaluate(predict_sequences(model, test, args.smooth), classes)
    metrics["baseline_duration"] = baseline_duration(train, test, classes)["macro_f1_steps"]
    metrics["n_train_takes"] = len(train_takes)
    metrics["n_test_takes"] = len(test_takes)
    return metrics


def benchmark(path: Path, n_features: int, *, n: int = 300) -> dict:
    """Per-frame latency of the exported model, both backends. Answers "runs in real time".

    Timed the way the pipeline calls it: one ``update`` per frame over a warm streaming buffer,
    not a batched pass over a whole take.
    """
    from actus.models.features import FrameFeatures
    from actus.models.temporal import TemporalModel

    rng = np.random.default_rng(0)
    rows = rng.normal(size=(n, n_features)).astype(np.float32)
    out = {}
    for label, prefer in (("onnx", True), ("numpy", False)):
        try:
            model = TemporalModel.load(path, prefer_onnx=prefer)
        except Exception:
            continue
        if model.backend != label:
            continue
        for i in range(min(60, n)):  # warm up, and fill the receptive field
            model.update(_features(i, i / model.feature_fps, rows[i], FrameFeatures))
        t0 = time.perf_counter()
        for i in range(n):
            model.update(_features(i, i / model.feature_fps, rows[i], FrameFeatures))
        elapsed = time.perf_counter() - t0
        out[label] = {
            "ms_per_frame": round(1000 * elapsed / n, 3),
            "max_fps": round(n / elapsed, 1),
        }
    return out


def _features(i: int, t: float, row: np.ndarray, cls):
    return cls(i, t, row, np.zeros((2, 2, 1), np.float32), np.zeros((2, 2), bool), True)


def summarise(runs: list[dict], key: str) -> dict:
    vals = [r[key] for r in runs if key in r]
    if not vals:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}


HEADLINE = ("frame_accuracy", "macro_f1_steps", *[f"segmental_f1@{int(o * 100)}" for o in OVERLAPS])


def write_card(path: Path, report: dict) -> None:
    m = report["manifest"]
    lines = [
        "# Temporal action model (ACTUS T-HOI-TEMP)",
        "",
        f"Trained {report['trained_at']} from `{report['cache']}` "
        f"(feature spec v{m['feature_spec_version']}, detector `{m['detector_backend']}`).",
        "",
        "A causal dilated TCN over 2 s windows of rack-frame hand, pose and object features,",
        "predicting which protocol step is in progress. It is an evidence source: the protocol",
        "engine decides completions, and this model only dates a step's onset and contributes a",
        "bounded term to the fused evidence (see `actus/models/fusion.py`).",
        "",
        "## Data",
        "",
        f"- takes: {report['n_takes']}, frames at {report['feature_fps']} Hz: {report['n_frames']}",
        f"- operators: {', '.join(report['operators'])}",
        f"- classes: {', '.join(report['classes'])}",
        "",
        "## Held-out results",
        "",
        f"Split by operator. Mean +/- standard deviation over {report['seeds']} seeds.",
        "",
        "| split | test operator | frame acc | macro F1 (steps) "
        "| seg F1@10 | seg F1@25 | seg F1@50 |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in report["splits"]:
        s = entry["summary"]

        def cell(key: str, s=s) -> str:
            return f"{s[key]['mean']:.3f} ± {s[key]['std']:.3f}"

        lines.append(
            f"| {entry['split']} | {', '.join(entry['test_operators'])} | "
            + " | ".join(cell(k) for k in HEADLINE)
            + " |"
        )
    lines += [
        "",
        "## Baselines, same test frames",
        "",
        "| split | model | frame acc | macro F1 (steps) | seg F1@25 |",
        "|---|---|---|---|---|",
    ]
    for entry in report["splits"]:
        for name in ("majority", "duration"):
            b = entry["baselines"][name]
            lines.append(
                f"| {entry['split']} | {name} | {b['frame_accuracy']:.3f} | "
                f"{b['macro_f1_steps']:.3f} | {b['segmental_f1@25']:.3f} |"
            )
    lines += [
        "",
        "## Diagnostic: is it the features, or the transfer?",
        "",
        "Trained and tested on **different takes of the same operators**. This leaks operator",
        "identity, the rig layout and the clothing, so it is not a result and must not be",
        "quoted as one. It separates two explanations of the held-out scores above: features",
        "carrying no signal, or features carrying signal that does not transfer between people.",
        "",
        *(
            [
                "| | frame acc | macro F1 (steps) | seg F1@25 | duration baseline |",
                "|---|---|---|---|---|",
                f"| within-operator | {d['frame_accuracy']:.3f} | {d['macro_f1_steps']:.3f} | "
                f"{d['segmental_f1@25']:.3f} | {d['baseline_duration']:.3f} |",
            ]
            if (d := report.get("diagnostic_within_operator"))
            else ["Not run."]
        ),
        "",
        "## Throughput",
        "",
        "One `update()` per frame over a warm streaming buffer, on this machine's CPU.",
        "",
        "| backend | ms/frame | max fps |",
        "|---|---|---|",
        *[
            f"| {k} | {v['ms_per_frame']} | {v['max_fps']} |"
            for k, v in (report.get("benchmark") or {}).items()
        ],
        "",
        "## Reading these numbers",
        "",
        "**The model loses to the duration baseline on every fold.** Two things are worth",
        "saying about that rather than burying it.",
        "",
        "First, why the baseline is so strong here: every take runs the same six steps in the",
        "same order at a similar pace, so position-within-the-take almost determines the label.",
        "Its segmental F1 is especially flattering because it emits exactly six clean segments",
        "in the canonical order by construction.",
        "",
        "Second, why that does not make it a useful model: the duration prior predicts the",
        "canonical procedure whatever the operator does. It reports every step as completed, in",
        "order, always -- so it scores well on these metrics while being structurally incapable",
        "of detecting the skips and out-of-order steps the system exists to catch. It is the",
        "right baseline to report and the wrong thing to ship. Neither model is the judge: the",
        "protocol engine is, and it decides from evidence.",
        "",
        "The within-operator diagnostic above lands below the baseline too, which points at the",
        "features rather than at cross-operator transfer. That is consistent with what the cache",
        "manifest says about perception: `red_box` is detected on 0.0% of frames and",
        "`yellow_box` on 8%, so the object channels are largely dead, and hands are present on",
        "roughly half of frames. This model is bottlenecked on perception, not on architecture",
        "or seeds -- the same bottleneck that blocks the HOI classifier. Retrain once",
        "T-DET-TRAIN lands.",
        "",
        "## Limitations",
        "",
        f"- The dataset is {report['n_takes']} takes from {len(report['operators'])} operators,"
        " filmed on Earth in one room. These are held-out *operators*, not held-out environments.",
        "- The marker rig was re-laid between operators at different sizes, so rack coordinates"
        " are consistent within a session but not comparable across operators"
        " (`actus/models/rackfit.py`). Relative features are unaffected; absolute positions are.",
        f"- Object features come from the `{m['detector_backend']}` detector. Its per-class"
        " detection rates are in the cache manifest; where a class is rarely detected, the model"
        " is effectively working from hands and pose alone.",
        "- `other`-labelled frames are folded into `idle`, because `ActionScores` is defined over"
        " the protocol's steps plus `idle`.",
        "- Confusion matrices and per-class F1 are in `metrics.json`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cache", default="data/features")
    p.add_argument("--out", default="models/temporal")
    p.add_argument("--feature-fps", type=float, default=DEFAULT_FEATURE_FPS)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--steps-per-epoch", type=int, default=40)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--kernel", type=int, default=3)
    p.add_argument("--dilations", type=int, nargs="*", default=list(DEFAULT_DILATIONS))
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--smooth", type=int, default=9, help="median filter width, in grid samples")
    p.add_argument("--loo", action="store_true", help="also run leave-one-operator-out folds")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def run_split(cache, split: Split, args, classes) -> dict:
    from actus.models.nn import TCN
    from actus.models.temporal import TemporalModel

    fps = args.feature_fps
    names = cache.label_names
    train = Sequences(cache.by_operators(split.train), names, fps)
    val = Sequences(cache.by_operators(split.val), names, fps)
    test = Sequences(cache.by_operators(split.test), names, fps)
    if not len(train) or not len(test):
        raise RuntimeError(f"{split.name}: empty train or test split")

    runs, best = [], None
    for seed in range(args.seeds):
        t0 = time.time()
        net, weights, arch, history = train_one(
            train, val, classes=classes, seed=seed, args=args, verbose=args.verbose
        )
        arch = {**arch, "feature_fps": fps}
        spec = ModelSpec(
            "tcn",
            tuple(classes),
            cache.manifest["feature_spec_version"],
            train.n_features,
            arch,
            {"split": split.name, "seed": seed},
        )
        model = TemporalModel(
            TCN(spec, weights),
            classes,
            feature_fps=fps,
            n_features=train.n_features,
            receptive_field=receptive_field(tuple(args.dilations), args.kernel),
        )
        metrics = evaluate(predict_sequences(model, test, args.smooth), classes)
        metrics["seed"] = seed
        metrics["seconds"] = round(time.time() - t0, 1)
        runs.append(metrics)
        val_score = history[-1].get("best_val_macro_f1", 0.0) if history else 0.0
        if best is None or val_score > best["val"]:
            best = {"val": val_score, "net": net, "weights": weights, "spec": spec}
        print(
            f"  seed {seed}: frame_acc {metrics['frame_accuracy']:.3f}  "
            f"macro_f1 {metrics['macro_f1_steps']:.3f}  "
            f"segF1@25 {metrics['segmental_f1@25']:.3f}  ({metrics['seconds']}s)"
        )

    return {
        "split": split.name,
        "train_operators": list(split.train),
        "val_operators": list(split.val),
        "test_operators": list(split.test),
        "runs": runs,
        "summary": {k: summarise(runs, k) for k in HEADLINE},
        "baselines": {
            "majority": baseline_majority(train, test, classes),
            "duration": baseline_duration(train, test, classes),
        },
        "_best": best,
    }


def main(argv: Sequence[str] | None = None) -> int:
    from actus.core.config import resolve_path

    args = parse_args(argv)
    cache = load_cache(resolve_path(args.cache))
    classes = action_classes(cache.label_names)
    fps = args.feature_fps
    all_seqs = Sequences(cache.takes, cache.label_names, fps)
    print(f"cache: {len(cache.takes)} takes, {len(all_seqs.stacked()[1])} samples at {fps} Hz")
    print(f"classes: {list(classes)}   features: {all_seqs.n_features}")

    splits = [operator_split(cache.operators)]
    if args.loo:
        splits += leave_one_out_splits(cache.operators)

    results = []
    for split in splits:
        print(f"\n{split.describe()}")
        results.append(run_split(cache, split, args, classes))

    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = results[0].pop("_best")
    for r in results:
        r.pop("_best", None)
    save_npz(out_dir / "temporal.npz", best["spec"], best["weights"])
    export_onnx(
        best["net"],
        out_dir / "temporal.onnx",
        example_shape=(1, max(args.window, 64), all_seqs.n_features),
        dynamic_time=True,
    )

    bench = benchmark(out_dir / "temporal.npz", all_seqs.n_features)
    print("\ndiagnostic (leaks operator identity -- not a result):")
    diagnostic = diagnostic_within_operator(cache, args, classes)
    if diagnostic:
        print(
            f"  within-operator  frame_acc {diagnostic['frame_accuracy']:.3f}  "
            f"macro_f1 {diagnostic['macro_f1_steps']:.3f}  "
            f"segF1@25 {diagnostic['segmental_f1@25']:.3f}  "
            f"(duration baseline {diagnostic['baseline_duration']:.3f})"
        )
    report = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cache": args.cache,
        "benchmark": bench,
        "diagnostic_within_operator": diagnostic,
        "manifest": {
            k: cache.manifest[k]
            for k in ("feature_spec_version", "detector_backend", "rack_mode", "rack_measured")
            if k in cache.manifest
        },
        "feature_fps": fps,
        "seeds": args.seeds,
        "classes": list(classes),
        "operators": list(cache.operators),
        "n_takes": len(cache.takes),
        "n_frames": len(all_seqs.stacked()[1]),
        "architecture": {
            "channels": args.channels,
            "dilations": list(args.dilations),
            "kernel": args.kernel,
            "receptive_field_samples": receptive_field(tuple(args.dilations), args.kernel),
            "receptive_field_s": receptive_field(tuple(args.dilations), args.kernel) / fps,
        },
        "splits": results,
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    write_card(out_dir / "card.md", report)
    print(f"\nwrote {out_dir}/temporal.npz, metrics.json, card.md")
    for entry in results:
        s = entry["summary"]
        print(
            f"  {entry['split']:12s} frame_acc {s['frame_accuracy']['mean']:.3f}  "
            f"macro_f1 {s['macro_f1_steps']['mean']:.3f}  "
            f"segF1@25 {s['segmental_f1@25']['mean']:.3f}  "
            f"(duration baseline {entry['baselines']['duration']['macro_f1_steps']:.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
