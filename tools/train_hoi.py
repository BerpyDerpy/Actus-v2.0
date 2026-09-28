#!/usr/bin/env python3
"""Train the hand-object interaction classifier (T-HOI-TEMP).

    uv run python tools/train_hoi.py --coverage           # can this even be trained yet?
    uv run python tools/train_hoi.py                      # train, evaluate, export
    uv run python tools/train_hoi.py --write-verify-sheet # 200 pairs for a human to check

A small MLP over the per-(hand, object) features in ``data/features``, predicting
``none`` / ``touching`` / ``holding``. Its output feeds the world model as
``{object: p(holding)}``, sharpening the ``<object>.held`` predicate that the protocol's
``requires_hoi`` steps rest on.

Supervision is weak: the hold windows come from the human step annotations plus the protocol
(``actus/models/weaklabels.py``), and geometry decides which hand. So the headline number is
"agreement with weak labels on held-out operators", which is not the same as accuracy, and the
card says so. ``--write-verify-sheet`` draws a class-balanced sample for a human to label by
hand; once ``data/hoi_verify/verified.csv`` has a filled-in ``true_label`` column, the run also
reports accuracy against it, which is the number worth quoting.

The coverage gate
-----------------
A pair can only be scored when the hand *and* the object were both detected. If the detector
rarely sees an object there is nothing to learn, and a model trained on the few frames where it
does would be both useless and misleading. The run refuses to train below ``--min-holding``
holding examples and prints the per-object detection rates that caused it, rather than emitting
a model with impressive-looking numbers over a handful of frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from actus.models.dataset import (  # noqa: E402
    Split,
    confusion,
    leave_one_out_splits,
    load_cache,
    macro_f1,
    operator_split,
    per_class_f1,
)
from actus.models.hoi import HOI_CLASSES, HOLDING  # noqa: E402
from actus.models.nn import (  # noqa: E402
    ModelSpec,
    build_torch_mlp,
    export_onnx,
    mlp_arch,
    mlp_weights,
    save_npz,
)
from actus.models.weaklabels import (  # noqa: E402
    IGNORE,
    WeakLabelParams,
    apply_verified,
    label_counts,
    verification_sample,
    weak_labels,
)
from actus.protocol.schema import load_protocol  # noqa: E402

VERIFY_DIR = "data/hoi_verify"
VERIFY_FIELDS = ("take", "frame_idx", "hand", "object", "weak_label", "true_label")


# ----------------------------------------------------------------------------------------------
# Building the training table
# ----------------------------------------------------------------------------------------------


class PairTable:
    """Flattened (hand, object) rows with weak labels, kept alongside their provenance."""

    def __init__(self, cache, protocol, params: WeakLabelParams) -> None:
        self.cache = cache
        self.protocol = protocol
        self.params = params
        self.per_take: dict[str, np.ndarray] = {}
        self.rows: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
        self.operators: list[np.ndarray] = []
        self.objects_idx: list[np.ndarray] = []
        for take in cache.takes:
            weak = weak_labels(
                labels=take.label,
                t=take.t,
                pair=take.pair,
                pair_valid=take.pair_valid,
                label_names=cache.label_names,
                pair_names=cache.pair_names,
                objects=cache.objects,
                protocol=protocol,
                params=params,
            )
            self.per_take[take.take] = weak
            keep = np.argwhere(weak != IGNORE)
            if not len(keep):
                continue
            self.rows.append(np.stack([take.pair[f, h, o] for f, h, o in keep]))
            self.labels.append(np.array([weak[f, h, o] for f, h, o in keep], dtype=np.int64))
            self.operators.append(np.array([take.operator] * len(keep)))
            self.objects_idx.append(keep[:, 2])

    def stacked(self):
        if not self.rows:
            empty = np.zeros((0, len(self.cache.pair_names)), dtype=np.float32)
            return empty, np.zeros(0, np.int64), np.zeros(0, dtype="<U8"), np.zeros(0, np.int64)
        return (
            np.concatenate(self.rows).astype(np.float32),
            np.concatenate(self.labels),
            np.concatenate(self.operators),
            np.concatenate(self.objects_idx),
        )

    def counts(self) -> dict[str, int]:
        merged = {name: 0 for name in HOI_CLASSES}
        merged["ignored"] = 0
        for weak in self.per_take.values():
            for k, v in label_counts(weak).items():
                merged[k] += v
        return merged


def coverage_report(cache, table: PairTable) -> dict:
    det = {}
    for i, cls in enumerate(cache.manifest["detector_classes"]):
        rates = [float(t.det_seen[:, i].mean()) for t in cache.takes]
        det[cls] = round(float(np.mean(rates)), 4)
    pair = {}
    for o, name in enumerate(cache.objects):
        rates = [float(t.pair_valid[:, :, o].any(axis=1).mean()) for t in cache.takes]
        pair[name] = round(float(np.mean(rates)), 4)
    return {
        "detector_backend": cache.manifest["detector_backend"],
        "detection_rate": det,
        "pair_coverage": pair,
        "weak_label_counts": table.counts(),
    }


def print_coverage(report: dict) -> None:
    print(f"detector backend: {report['detector_backend']}")
    print("\nper-frame detection rate:")
    for k, v in report["detection_rate"].items():
        print(f"  {k:20s} {v:.3f}")
    print("\nHOI pair coverage (hand AND object visible):")
    for k, v in report["pair_coverage"].items():
        print(f"  {k:20s} {v:.3f}")
    print("\nweak labels:")
    for k, v in report["weak_label_counts"].items():
        print(f"  {k:20s} {v}")


# ----------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------


def train_one(x: np.ndarray, y: np.ndarray, *, seed: int, args, n_classes: int):
    import torch

    torch.manual_seed(seed)
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    mean, std = x.mean(axis=0), x.std(axis=0) + 1e-6
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = np.where(counts > 0, 1.0 / np.maximum(counts, 1.0), 0.0)
    w = (w / w.sum() * np.count_nonzero(counts)).astype(np.float32)

    net = build_torch_mlp(x.shape[1], args.hidden, n_classes).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))

    xb = torch.tensor((x - mean) / std, dtype=torch.float32, device=device)
    yb = torch.tensor(y, device=device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = len(xb)
    for _ in range(args.epochs):
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, args.batch):
            idx = perm[i : i + args.batch]
            loss = loss_fn(net(xb[idx]), yb[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval().to("cpu")
    return net, mlp_weights(net, mean, std)


def evaluate_rows(model, x: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    if not len(y):
        return {"rows": 0}
    pred = model.predict_rows(x).argmax(axis=1)
    hold = HOI_CLASSES.index(HOLDING)
    tp = float(np.sum((pred == hold) & (y == hold)))
    fp = float(np.sum((pred == hold) & (y != hold)))
    fn = float(np.sum((pred != hold) & (y == hold)))
    return {
        "rows": len(y),
        "accuracy": float(np.mean(pred == y)),
        "macro_f1": macro_f1(y, pred, n_classes),
        "holding_precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "holding_recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "per_class_f1": {
            c: float(v) for c, v in zip(HOI_CLASSES, per_class_f1(y, pred, n_classes), strict=True)
        },
        "confusion": confusion(y, pred, n_classes).tolist(),
    }


def run_split(table: PairTable, split: Split, args, cache) -> dict:
    from actus.models.hoi import HOIModel
    from actus.models.nn import MLP

    x, y, ops, _ = table.stacked()
    n_classes = len(HOI_CLASSES)
    tr = np.isin(ops, split.train)
    te = np.isin(ops, split.test)
    if not tr.any() or not te.any():
        raise RuntimeError(f"{split.name}: empty train or test split")

    runs, best = [], None
    for seed in range(args.seeds):
        net, weights = train_one(x[tr], y[tr], seed=seed, args=args, n_classes=n_classes)
        spec = ModelSpec(
            "mlp",
            HOI_CLASSES,
            cache.manifest["feature_spec_version"],
            x.shape[1],
            {**mlp_arch(args.hidden), "hands": list(cache.hands), "objects": list(cache.objects)},
            {"split": split.name, "seed": seed, "supervision": "weak"},
        )
        model = HOIModel(
            MLP(spec, weights),
            HOI_CLASSES,
            hands=cache.hands,
            objects=cache.objects,
            n_features=x.shape[1],
        )
        metrics = evaluate_rows(model, x[te], y[te], n_classes)
        metrics["seed"] = seed
        runs.append(metrics)
        if best is None or metrics["macro_f1"] > best["score"]:
            best = {
                "score": metrics["macro_f1"],
                "net": net,
                "weights": weights,
                "spec": spec,
                "model": model,
            }
        print(
            f"  seed {seed}: acc {metrics['accuracy']:.3f}  macro_f1 {metrics['macro_f1']:.3f}  "
            f"holding P/R {metrics['holding_precision']:.3f}/{metrics['holding_recall']:.3f}"
        )
    return {
        "split": split.name,
        "train_operators": list(split.train),
        "test_operators": list(split.test),
        "runs": runs,
        "summary": {
            k: {
                "mean": float(np.mean([r[k] for r in runs])),
                "std": float(np.std([r[k] for r in runs])),
            }
            for k in ("accuracy", "macro_f1", "holding_precision", "holding_recall")
        },
        "_best": best,
    }


# ----------------------------------------------------------------------------------------------
# Human verification
# ----------------------------------------------------------------------------------------------


def write_verify_sheet(cache, table: PairTable, split: Split, path: Path, n: int, seed: int) -> int:
    """Draw pairs for a human to check: held-out operators first, topped up from the rest.

    The held-out split is preferred because that is where the reported number matters, but with
    the current detector it holds very few scorable pairs, so the sheet falls back to the other
    operators rather than handing back a near-empty file.
    """
    rng = np.random.default_rng(seed)
    test_takes = [t for t in cache.takes if t.operator in set(split.test)]
    other_takes = [t for t in cache.takes if t not in test_takes]

    def draw(takes, want: int) -> list[dict]:
        if not takes or want <= 0:
            return []
        rows: list[dict] = []
        per_take = max(1, want // len(takes))
        for take in takes:
            weak = table.per_take.get(take.take)
            if weak is None:
                continue
            rows += verification_sample(
                weak,
                take=take.take,
                frame_idx=take.frame_idx,
                hands=cache.hands,
                objects=cache.objects,
                n=per_take,
                rng=rng,
            )
        rng.shuffle(rows)
        return rows

    rows = draw(test_takes, n)
    if len(rows) < n:
        rows += draw(other_takes, n - len(rows))
    rows = rows[:n]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=VERIFY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def read_verified(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("true_label")]


def score_verified(model, cache, table: PairTable, verified: Sequence[dict]) -> dict:
    xs, ys = [], []
    for take in cache.takes:
        weak = table.per_take.get(take.take)
        if weak is None:
            continue
        truth, mask = apply_verified(
            weak,
            take=take.take,
            frame_idx=take.frame_idx,
            hands=cache.hands,
            objects=cache.objects,
            verified=verified,
        )
        for f, h, o in np.argwhere(mask):
            xs.append(take.pair[f, h, o])
            ys.append(truth[f, h, o])
    if not xs:
        return {"rows": 0}
    out = evaluate_rows(model, np.stack(xs).astype(np.float32), np.array(ys), len(HOI_CLASSES))
    out["source"] = "human-verified"
    return out


# ----------------------------------------------------------------------------------------------
# Card
# ----------------------------------------------------------------------------------------------


def benchmark(path: Path, n_features: int, *, n: int = 500) -> dict:
    """Per-frame latency, timed as the pipeline calls it: all four (hand, object) pairs at once."""
    from actus.models.features import FrameFeatures
    from actus.models.hoi import HOIModel

    rng = np.random.default_rng(0)
    out = {}
    for label, prefer in (("onnx", True), ("numpy", False)):
        try:
            model = HOIModel.load(path, prefer_onnx=prefer)
        except Exception:
            continue
        if model.backend != label:
            continue
        pair = rng.normal(size=(2, 2, n_features)).astype(np.float32)
        valid = np.ones((2, 2), dtype=bool)
        frame = FrameFeatures(0, 0.0, np.zeros(1, np.float32), pair, valid, True)
        for _ in range(50):
            model.predict(frame)
        t0 = time.perf_counter()
        for _ in range(n):
            model.predict(frame)
        elapsed = time.perf_counter() - t0
        out[label] = {
            "ms_per_frame": round(1000 * elapsed / n, 3),
            "max_fps": round(n / elapsed, 1),
        }
    return out


def write_card(path: Path, report: dict) -> None:
    cov = report["coverage"]
    lines = [
        "# Hand-object interaction classifier (ACTUS T-HOI-TEMP)",
        "",
        f"Trained {report['trained_at']} from `{report['cache']}` "
        f"(feature spec v{report['feature_spec_version']}, detector `{cov['detector_backend']}`).",
        "",
        "An MLP over per-(hand, object) geometry: keypoint distances in rack coordinates, grip",
        "shape, relative velocity and co-motion. Output feeds the world model as",
        "`{object: p(holding)}`. It abstains whenever the hand or the object was not observed.",
        "",
        "## Supervision",
        "",
        "Weak. Hold windows come from the human step annotations plus the protocol's",
        "`requires_hoi` (see `actus/models/weaklabels.py`); geometry decides which hand. Frames",
        f"within {report['params']['boundary_margin_s']} s of a window boundary are dropped.",
        "",
        "| weak class | rows |",
        "|---|---|",
    ]
    for k, v in cov["weak_label_counts"].items():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## Coverage",
        "",
        "Share of frames each object was detected, and the share of frames a pair could be",
        "scored at all. A low number here caps everything below it.",
        "",
        "| object | pair coverage |",
        "|---|---|",
    ]
    for k, v in cov["pair_coverage"].items():
        lines.append(f"| {k} | {v:.3f} |")
    lines += ["", "| detector class | detection rate |", "|---|---|"]
    for k, v in cov["detection_rate"].items():
        lines.append(f"| {k} | {v:.3f} |")

    lines += [
        "",
        "## Held-out results",
        "",
        "Split by operator, mean +/- standard deviation over "
        f"{report['seeds']} seeds. Agreement with **weak** labels, not verified truth.",
        "",
        "| split | test | accuracy | macro F1 | holding precision | holding recall |",
        "|---|---|---|---|---|---|",
    ]
    for entry in report["splits"]:
        s = entry["summary"]
        lines.append(
            f"| {entry['split']} | {', '.join(entry['test_operators'])} | "
            f"{s['accuracy']['mean']:.3f} ± {s['accuracy']['std']:.3f} | "
            f"{s['macro_f1']['mean']:.3f} ± {s['macro_f1']['std']:.3f} | "
            f"{s['holding_precision']['mean']:.3f} | {s['holding_recall']['mean']:.3f} |"
        )
    ver = report.get("verified") or {}
    lines += ["", "## Human-verified subset", ""]
    if ver.get("rows"):
        lines += [
            f"{ver['rows']} pairs checked by hand.",
            "",
            f"- accuracy {ver['accuracy']:.3f}, macro F1 {ver['macro_f1']:.3f}",
            f"- holding precision {ver['holding_precision']:.3f}, "
            f"recall {ver['holding_recall']:.3f}",
        ]
    else:
        lines += [
            "**Not done yet.** Run `tools/train_hoi.py --write-verify-sheet`, fill in the",
            f"`true_label` column of `{VERIFY_DIR}/sheet.csv`, save it as",
            f"`{VERIFY_DIR}/verified.csv`, and re-run. Until then every number above is",
            "agreement with weak labels, which share the annotations' mistakes.",
        ]
    lines += [
        "",
        "## Throughput",
        "",
        "All four (hand, object) pairs scored per frame, on this machine's CPU.",
        "",
        "| backend | ms/frame | max fps |",
        "|---|---|---|",
        *[
            f"| {k} | {v['ms_per_frame']} | {v['max_fps']} |"
            for k, v in (report.get("benchmark") or {}).items()
        ],
        "",
        "## Limitations",
        "",
        "- Weak supervision: the labels are derived, not observed. The verified subset is the",
        "  only unbiased estimate here.",
        "- The model abstains when either the hand or the object is unseen, so its usefulness is",
        "  bounded by the detector's per-object rate above, not by the accuracy figures.",
        "- Rack coordinates are consistent within an operator's session but not across operators"
        " (`actus/models/rackfit.py`); the scale-relative features carry most of the signal.",
        '- Trained per (hand, object) pair independently: it cannot express "only one of these',
        '  two boxes is held".',
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
    p.add_argument("--out", default="models/hoi")
    p.add_argument("--protocol", default=None, help="default: the one recorded in the cache")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--hidden", type=int, nargs="*", default=[64, 32])
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--boundary-margin-s", type=float, default=0.4)
    p.add_argument("--min-holding", type=int, default=200, help="refuse to train below this")
    p.add_argument("--loo", action="store_true", help="also run leave-one-operator-out folds")
    p.add_argument("--coverage", action="store_true", help="report coverage and exit")
    p.add_argument("--write-verify-sheet", action="store_true")
    p.add_argument("--verify-n", type=int, default=200)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    from actus.core.config import resolve_path

    args = parse_args(argv)
    cache = load_cache(resolve_path(args.cache))
    protocol_path = args.protocol or cache.manifest.get("protocol", "protocols/bas_boxes.yaml")
    protocol = load_protocol(resolve_path(protocol_path))
    params = WeakLabelParams(boundary_margin_s=args.boundary_margin_s)
    table = PairTable(cache, protocol, params)
    coverage = coverage_report(cache, table)
    print(f"cache: {len(cache.takes)} takes, objects {list(cache.objects)}\n")
    print_coverage(coverage)

    split = operator_split(cache.operators)
    if args.write_verify_sheet:
        path = resolve_path(VERIFY_DIR) / "sheet.csv"
        n = write_verify_sheet(cache, table, split, path, args.verify_n, seed=0)
        print(f"\nwrote {n} rows to {path}")
        print("fill in true_label, save as verified.csv in the same folder, then re-run")
        return 0
    if args.coverage:
        return 0

    n_holding = coverage["weak_label_counts"].get(HOLDING, 0)
    if n_holding < args.min_holding:
        print(
            f"\nREFUSING TO TRAIN: only {n_holding} '{HOLDING}' examples "
            f"(need >= {args.min_holding}).\n"
            "The classifier scores a (hand, object) pair only when both were detected, so the\n"
            "per-object detection rates above are the binding constraint. Train the custom\n"
            "detector (T-DET-TRAIN) or improve the open-vocab prompts, re-run\n"
            "tools/extract_features.py, then train this.\n"
            "Nothing was written: a model fitted to this little data would report numbers that\n"
            "mean nothing."
        )
        out_dir = resolve_path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "coverage.json").write_text(json.dumps(coverage, indent=1), encoding="utf-8")
        print(f"wrote {out_dir}/coverage.json")
        return 2

    splits = [split] + (leave_one_out_splits(cache.operators) if args.loo else [])
    results = []
    for sp in splits:
        print(f"\n{sp.describe()}")
        results.append(run_split(table, sp, args, cache))

    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = results[0].pop("_best")
    for r in results:
        r.pop("_best", None)
    save_npz(out_dir / "hoi.npz", best["spec"], best["weights"])
    export_onnx(best["net"], out_dir / "hoi.onnx", example_shape=(1, len(cache.pair_names)))
    verified = read_verified(resolve_path(VERIFY_DIR) / "verified.csv")
    report = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cache": args.cache,
        "feature_spec_version": cache.manifest["feature_spec_version"],
        "seeds": args.seeds,
        "params": {"boundary_margin_s": args.boundary_margin_s},
        "coverage": coverage,
        "splits": results,
        "verified": score_verified(best["model"], cache, table, verified) if verified else None,
        "benchmark": benchmark(out_dir / "hoi.npz", len(cache.pair_names)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    write_card(out_dir / "card.md", report)
    print(f"\nwrote {out_dir}/hoi.npz, metrics.json, card.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
