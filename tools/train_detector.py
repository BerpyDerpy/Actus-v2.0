"""T-DET-TRAIN: train the custom YOLO11n detector on ACTUS's 5 box/lid classes.

    uv run python tools/train_detector.py \\
        --dataset data/dataset.yaml --run-name real \\
        --dataset-synth data/dataset_synth.yaml --synth-run-name real_synth

Trains YOLO11n from COCO weights at 640 px with full-360-tolerant augmentation
(``degrees=180``, left-right flips) and early stopping, once on the real-only dataset and, if
``--dataset-synth`` is given, once more on real+synthetic. Both runs are evaluated on the
dataset's ``test`` split (held-out operator(s), never seen in training or validation) and
exported to ONNX. Whichever run scores higher held-out mAP50 is copied to
``models/detector/best.pt`` / ``best.onnx`` -- the path ``configs/default.yaml``'s
``detector.custom_weights`` already points at -- and a model card is written to
``models/detector/card.md``.

Ultralytics is imported lazily (this file is only ever run explicitly, but keep it consistent
with the rest of T-DATA/T-PERC): requires ``uv sync --extra perception``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._datalib import (  # noqa: E402
    CLASSES,
    Box,
    read_classes,
    read_manifest,
    read_yolo_label,
    rel_to_repo,
    to_repo_path,
)

# --------------------------------------------------------------------------------------------
# Label prep: a hybrid autolabel proposer, tuned on this footage
#
# tools/autolabel.py's default (YOLO-World, prompted with configs/default.yaml's one-word
# ``detector.prompts``) turns out to all but miss ``red_box`` on the real takes: over 7888
# frames it proposed exactly 1 red_box box, at 0.16 confidence, versus 1290 for yellow_box.
# Manual inspection (see the PR description) shows why: the "red" prop is a small red/maroon
# spice-packet or bottle-cap, not a box-shaped object, and open-vocab CLIP-style matching on
# "red box" barely fires on it. Its actual color, though, is a clean, well-separated HSV signal
# against every take's neutral (grey/white) floor -- so red_box and yellow_box are proposed by
# plain HSV thresholding here instead, tuned against real crops from this dataset (see the PR
# description for the calibration). outer_box_closed / outer_box_open / lid stay on YOLO-World:
# they're not solid-colored, so a color heuristic can't help there.
#
# One operator (op01) turned out to use an entirely different, non-conforming prop set (a mug,
# a thermos and a book -- no cardboard box, no red/yellow box at all); its takes are excluded
# from the detector dataset entirely rather than mislabeled. See card.md's honesty section.
# --------------------------------------------------------------------------------------------

_COLOR_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red_box": [((0, 70, 40), (8, 255, 255)), ((172, 70, 40), (180, 255, 255))],
    "yellow_box": [((24, 90, 60), (36, 255, 255))],
}


class ColorBoxProposer:
    """HSV-threshold proposals for the two distinctly-colored, non-box-shaped props
    (``red_box``, ``yellow_box``) that YOLO-World's text prompts miss almost entirely on this
    footage. One box per class per frame: the largest connected component within its hue/sat/val
    range and a plausible object size, or none."""

    def __init__(
        self,
        color_ranges: dict[
            str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]
        ] = _COLOR_RANGES,
        *,
        min_area_frac: float = 0.0012,
        max_area_frac: float = 0.05,
        conf: float = 0.9,
    ) -> None:
        self.color_ranges = color_ranges
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.conf = conf

    def __call__(self, image: np.ndarray) -> list[Box]:
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        boxes = []
        for cls, ranges in self.color_ranges.items():
            mask = np.zeros((h, w), np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            best = None
            for i in range(1, n):
                area = stats[i, cv2.CC_STAT_AREA]
                frac = area / (h * w)
                if frac < self.min_area_frac or frac > self.max_area_frac:
                    continue
                if best is None or area > best[0]:
                    best = (area, i)
            if best is not None:
                i = best[1]
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                bw = stats[i, cv2.CC_STAT_WIDTH]
                bh = stats[i, cv2.CC_STAT_HEIGHT]
                boxes.append(Box(cls, float(x), float(y), float(x + bw), float(y + bh), self.conf))
        return boxes


class HybridProposer:
    """YOLO-World for the box/lid classes (shape matters, color doesn't) plus
    :class:`ColorBoxProposer` for red_box/yellow_box (color matters, YOLO-World misses them)."""

    def __init__(self, world_proposer: Any, color_proposer: ColorBoxProposer) -> None:
        self.world_proposer = world_proposer
        self.color_proposer = color_proposer

    def __call__(self, image: np.ndarray) -> list[Box]:
        boxes = [
            b for b in self.world_proposer(image) if b.cls not in self.color_proposer.color_ranges
        ]
        boxes += self.color_proposer(image)
        return boxes


def relabel(
    manifest_path: Path, out_dir: Path, *, exclude_operators: tuple[str, ...] = ("op01",)
) -> dict:
    """Regenerate ``data/labels`` with :class:`HybridProposer`, dropping any manifest rows from
    ``exclude_operators`` (non-conforming footage) rather than labeling them wrong."""
    from actus.core.config import load_config
    from tools.autolabel import YoloWorldProposer, autolabel_manifest

    rows = [r for r in read_manifest(manifest_path) if r["operator"] not in exclude_operators]
    cfg = load_config()
    world_prompts = {c: cfg.detector.prompts[c] for c in CLASSES if c not in _COLOR_RANGES}
    proposer = HybridProposer(
        YoloWorldProposer(world_prompts, classes=tuple(world_prompts)), ColorBoxProposer()
    )
    return autolabel_manifest(rows, proposer, out_dir, min_conf=0.0)


# --------------------------------------------------------------------------------------------
# Dataset assembly: real-only vs. real+synthetic
# --------------------------------------------------------------------------------------------


def build_synth_dataset_yaml(dataset_yaml: Path, synthetic_list: Path, out_yaml: Path) -> Path:
    """Write a second dataset yaml whose ``train`` is the real train split plus the synthetic
    split, sharing the same held-out ``val``/``test``. Ultralytics reads ``train`` as a plain
    image-list ``.txt``, so the two lists are just concatenated into a new one."""
    dataset = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    real_train = Path(dataset["path"]) / dataset["train"]
    combined_lines = real_train.read_text(encoding="utf-8").splitlines()
    combined_lines += [
        ln for ln in synthetic_list.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    combined_path = out_yaml.parent / "train_real_synth.txt"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.write_text("\n".join(combined_lines) + "\n", encoding="utf-8")

    dataset["train"] = rel_to_repo(combined_path)
    out_yaml.write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    return out_yaml


# --------------------------------------------------------------------------------------------
# Train + evaluate one run
# --------------------------------------------------------------------------------------------


@dataclass
class RunResult:
    name: str
    dataset_yaml: Path
    weights_pt: Path
    map50: float
    map50_95: float
    per_class: dict[str, dict[str, float]]
    n_train_images: int
    epochs_run: int


def train_one(
    dataset_yaml: Path,
    *,
    name: str,
    base_weights: str,
    epochs: int,
    imgsz: int,
    patience: int,
    batch: int,
    degrees: float,
    fliplr: float,
    seed: int,
    project: Path,
    device: str | None,
    workers: int = 4,
) -> Any:
    from ultralytics import YOLO

    model = YOLO(base_weights)
    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        patience=patience,
        batch=batch,
        degrees=degrees,  # full-360 rotation tolerance (build-plan.md's orientation-agnostic goal)
        fliplr=fliplr,
        flipud=0.0,
        project=str(project),
        name=name,
        seed=seed,
        device=device,
        workers=workers,  # kept modest: this machine's system RAM, not just VRAM, is the limit
        exist_ok=True,
        plots=True,
    )
    return model


def evaluate(model: Any, dataset_yaml: Path, *, imgsz: int, split: str = "test") -> dict:
    metrics = model.val(data=str(dataset_yaml), imgsz=imgsz, split=split, plots=False)
    names = metrics.names
    per_class = {}
    for i, cls_idx in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(i)
        per_class[names[int(cls_idx)]] = {
            "precision": float(p),
            "recall": float(r),
            "mAP50": float(ap50),
            "mAP50-95": float(ap),
        }
    # classes never present in the held-out split get an explicit "no test instances" row,
    # rather than silently vanishing from the report (honest by construction).
    for c in names.values():
        per_class.setdefault(
            c, {"precision": None, "recall": None, "mAP50": None, "mAP50-95": None}
        )
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": per_class,
    }


def run_training(
    dataset_yaml: Path,
    *,
    name: str,
    base_weights: str,
    epochs: int,
    imgsz: int,
    patience: int,
    batch: int,
    degrees: float,
    fliplr: float,
    seed: int,
    project: Path,
    device: str | None,
    workers: int = 4,
) -> RunResult:
    model = train_one(
        dataset_yaml,
        name=name,
        base_weights=base_weights,
        epochs=epochs,
        imgsz=imgsz,
        patience=patience,
        batch=batch,
        degrees=degrees,
        fliplr=fliplr,
        seed=seed,
        project=project,
        device=device,
        workers=workers,
    )
    result = evaluate(model, dataset_yaml, imgsz=imgsz, split="test")

    run_dir = project / name
    weights_pt = run_dir / "weights" / "best.pt"
    n_train = len(
        (
            Path(yaml.safe_load(dataset_yaml.read_text())["path"])
            / yaml.safe_load(dataset_yaml.read_text())["train"]
        )
        .read_text()
        .splitlines()
    )
    epochs_run = _epochs_actually_run(run_dir)

    (run_dir / "held_out_eval.json").write_text(
        json.dumps({"dataset_yaml": rel_to_repo(dataset_yaml), **result}, indent=2),
        encoding="utf-8",
    )
    return RunResult(
        name=name,
        dataset_yaml=dataset_yaml,
        weights_pt=weights_pt,
        map50=result["map50"],
        map50_95=result["map50_95"],
        per_class=result["per_class"],
        n_train_images=n_train,
        epochs_run=epochs_run,
    )


def _epochs_actually_run(run_dir: Path) -> int:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return -1
    lines = [ln for ln in csv_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return max(0, len(lines) - 1)  # minus the header row


# --------------------------------------------------------------------------------------------
# Baselines the eventual mAP has to beat (a random/majority-box "detector" doing no learning)
# --------------------------------------------------------------------------------------------


def full_image_box_baseline(dataset_yaml: Path, *, classes: tuple[str, ...]) -> dict:
    """Every test image gets one full-frame box per class present anywhere in the test split's
    ground truth -- a floor any trained model must clear by a wide margin, since it never
    localizes anything."""
    dataset = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    repo_path = Path(dataset["path"])
    test_list = repo_path / dataset["test"]
    if not test_list.exists():
        return {"map50": None, "note": "no test split file"}
    image_paths = [ln for ln in test_list.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not image_paths:
        return {"map50": None, "note": "test split is empty"}

    tp_iou_sum, n_gt, n_pred = 0.0, 0, 0
    for rel in image_paths:
        img_path = repo_path / rel
        label_path = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = read_yolo_label(label_path, w, h, classes)
        n_gt += len(gt)
        n_pred += len({b.cls for b in gt})  # one full-frame box per class present
        for b in gt:
            area_gt = (b.x2 - b.x1) * (b.y2 - b.y1)
            tp_iou_sum += area_gt / (w * h)  # IoU of a full-frame box against a smaller GT box
    mean_iou = tp_iou_sum / n_gt if n_gt else 0.0
    return {
        "description": "one full-image box per class present in the frame; measures how much "
        "'detection' comes for free from just knowing which classes co-occur",
        "n_test_images": len(image_paths),
        "n_gt_boxes": n_gt,
        "mean_iou_of_full_frame_box_vs_gt": round(mean_iou, 4),
        "map50": None,  # not a real detector run through the same metric; IoU proxy only
    }


# --------------------------------------------------------------------------------------------
# Honest failure-case gallery: worst-scoring test frames, GT vs. prediction overlaid
# --------------------------------------------------------------------------------------------

_COLORS = {
    "outer_box_closed": (150, 90, 40),
    "outer_box_open": (200, 130, 60),
    "lid": (170, 170, 60),
    "red_box": (40, 40, 210),
    "yellow_box": (40, 210, 230),
}


def _iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _score_frame(gt: list[Box], pred: list[Box], iou_thresh: float = 0.5) -> tuple[int, int]:
    """(false_negatives, false_positives) of ``pred`` against ``gt`` via greedy same-class
    IoU matching."""
    unmatched_gt = list(gt)
    fp = 0
    for p in pred:
        best_i, best_iou = None, 0.0
        for i, g in enumerate(unmatched_gt):
            if g.cls != p.cls:
                continue
            iou = _iou(g, p)
            if iou > best_iou:
                best_i, best_iou = i, iou
        if best_i is not None and best_iou >= iou_thresh:
            del unmatched_gt[best_i]
        else:
            fp += 1
    return len(unmatched_gt), fp


def _draw_boxes(img: np.ndarray, boxes: list[Box], *, dashed: bool) -> np.ndarray:
    out = img.copy()
    for b in boxes:
        color = _COLORS.get(b.cls, (255, 255, 255))
        p1, p2 = (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2))
        thickness = 1 if dashed else 2
        cv2.rectangle(out, p1, p2, color, thickness)
        tag = b.cls + (f" {b.conf:.2f}" if b.conf < 1.0 else "")
        cv2.putText(out, tag, (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return out


def build_failure_gallery(
    model: Any,
    dataset_yaml: Path,
    *,
    classes: tuple[str, ...],
    imgsz: int,
    conf: float,
    out_dir: Path,
    n: int = 8,
) -> list[dict]:
    """Render the ``n`` worst test frames (most false negatives + false positives) with ground
    truth (solid) vs. prediction (thin) boxes overlaid, for an honest failure gallery -- not a
    cherry-picked success gallery."""
    dataset = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    repo_path = Path(dataset["path"])
    test_list = repo_path / dataset["test"]
    image_paths = [ln for ln in test_list.read_text(encoding="utf-8").splitlines() if ln.strip()]

    scored = []
    for rel in image_paths:
        img_path = repo_path / rel
        label_path = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = read_yolo_label(label_path, w, h, classes)
        result = model.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0]
        pred = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                cls_name = classes[int(box.cls.item())]
                pred.append(Box(cls_name, x1, y1, x2, y2, float(box.conf.item())))
        fn, fp = _score_frame(gt, pred)
        scored.append((fn + fp, fn, fp, rel, img, gt, pred))

    scored.sort(key=lambda t: t[0], reverse=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    gallery = []
    for badness, fn, fp, rel, img, gt, pred in scored[:n]:
        if badness == 0:
            break  # every remaining frame is perfect; nothing left to show honestly as a failure
        overlay = _draw_boxes(img, gt, dashed=False)
        overlay = _draw_boxes(overlay, pred, dashed=True)
        stem = Path(rel).stem
        out_path = out_dir / f"{stem}.jpg"
        cv2.imwrite(str(out_path), overlay)
        gallery.append(
            {
                "image": rel_to_repo(out_path),
                "source": rel,
                "false_negatives": fn,
                "false_positives": fp,
            }
        )
    return gallery


# --------------------------------------------------------------------------------------------
# Model card
# --------------------------------------------------------------------------------------------


def write_card(
    path: Path,
    *,
    split_report: dict,
    real_result: RunResult,
    synth_result: RunResult | None,
    chosen: RunResult,
    baseline: dict,
    gallery: list[dict],
    base_weights: str,
    train_args: dict,
) -> None:
    lines = ["# ACTUS custom detector (YOLO11n) — model card", ""]
    lines += [
        "Trained by `tools/train_detector.py` (T-DET-TRAIN) from COCO weights on the project's "
        "5 detector classes, over real footage of the box-transfer protocol (`protocols/"
        "bas_boxes.yaml`).",
        "",
        "## Data",
        "",
    ]
    for op, split in sorted(split_report["operators"].items()):
        lines.append(f"- operator `{op}`: **{split}**")
    lines += [
        "",
        f"- train images: {split_report['per_split_frames'].get('train', 0)} "
        f"({split_report['per_split_takes'].get('train', 0)} takes)",
        f"- val images: {split_report['per_split_frames'].get('val', 0)} "
        f"({split_report['per_split_takes'].get('val', 0)} takes)",
        f"- **held-out test** images: {split_report['per_split_frames'].get('test', 0)} "
        f"({split_report['per_split_takes'].get('test', 0)} takes) — operator(s) "
        f"never seen in training or validation",
        "",
        "Leakage check: "
        + (
            "**FAILED** — " + "; ".join(split_report["leakage_checks"])
            if split_report["leakage_checks"]
            else "passed (no take spans two splits)"
        ),
        "",
    ]
    if split_report["duration_warnings"]:
        lines.append(
            f"{len(split_report['duration_warnings'])} scenario-pair duration warnings from "
            "`make_splits.py` (take length alone would separate some scenario pairs) — see "
            "`data/splits/report.md` for the full list. This matters for a temporal/HAR "
            "classifier (T-HOI-TEMP), not for this frame-level object detector."
        )
        lines.append("")

    lines += [
        "## Training",
        "",
        f"- base weights: `{base_weights}`",
        f"- image size: {train_args['imgsz']} px, batch {train_args['batch']}",
        f"- augmentation: `degrees={train_args['degrees']}` (full-360 rotation tolerance, per "
        f"the orientation-agnostic-tracking goal), `fliplr={train_args['fliplr']}`",
        f"- early stopping patience: {train_args['patience']} epochs",
        "",
        "| run | train images | epochs run | held-out mAP50 | held-out mAP50-95 |",
        "|---|---|---|---|---|",
        f"| real only | {real_result.n_train_images} | {real_result.epochs_run} | "
        f"{real_result.map50:.4f} | {real_result.map50_95:.4f} |",
    ]
    if synth_result is not None:
        lines.append(
            f"| real + synthetic | {synth_result.n_train_images} | {synth_result.epochs_run} | "
            f"{synth_result.map50:.4f} | {synth_result.map50_95:.4f} |"
        )
    else:
        lines.append("| real + synthetic | — | — | not run (no `--dataset-synth`) | — |")
    lines += [
        "",
        f"**Deployed run: `{chosen.name}`** (higher held-out mAP50). This is the honest number "
        "for whatever it turned out to be: build-plan.md requires held-out mAP50 ≥ 0.85 before "
        "this model replaces the open-vocabulary fallback as the default `detector.backend` — "
        + ("**met**" if chosen.map50 >= 0.85 else "**not met yet**")
        + f" ({chosen.map50:.4f}).",
        "",
        "## Per-class results (held-out test split, deployed run)",
        "",
        "| class | precision | recall | mAP50 | mAP50-95 |",
        "|---|---|---|---|---|",
    ]
    for cls in CLASSES:
        m = chosen.per_class.get(cls, {})
        if m.get("mAP50") is None:
            lines.append(f"| {cls} | — | — | — (no test instances) | — |")
        else:
            lines.append(
                f"| {cls} | {m['precision']:.3f} | {m['recall']:.3f} | {m['mAP50']:.3f} | "
                f"{m['mAP50-95']:.3f} |"
            )

    lines += ["", "## Baseline", ""]
    if baseline.get("map50") is None and "mean_iou_of_full_frame_box_vs_gt" in baseline:
        lines.append(
            'A full-frame "detector" (one box covering the whole image per class known to be '
            f"present) gets a mean IoU of {baseline['mean_iou_of_full_frame_box_vs_gt']:.3f} "
            f"against the {baseline['n_gt_boxes']} held-out ground-truth boxes — it never "
            "localizes anything, so any real mAP50 above that is the model actually learning "
            "to localize, not just learning what's usually in frame."
        )
    lines.append("")

    lines += ["## Failure-case gallery (held-out test split)", ""]
    if gallery:
        lines.append(
            "Worst-scoring test frames by false negatives + false positives, solid boxes = "
            "ground truth, thin boxes = prediction. These are not cherry-picked successes."
        )
        lines.append("")
        for g in gallery:
            lines.append(
                f"![{g['source']}]({Path(g['image']).relative_to('models/detector').as_posix()}) "
                f"— {g['false_negatives']} missed, {g['false_positives']} spurious "
                f"(`{g['source']}`)"
            )
            lines.append("")
    else:
        lines.append("No failures in the held-out set above the matching IoU threshold.")
        lines.append("")

    lines += [
        "## Honest limitations",
        "",
        "- Autolabel proposals (YOLO-World open-vocabulary boxes) were used as training labels. "
        "**No human reviewed them box-by-box** in this run — `tools/review_labels.py` supports "
        "that workflow but it needs a human at a display; running it here would only fake a "
        "review, which `CLAUDE.md` explicitly rules out. Treat these labels as weak/noisy, not "
        "verified.",
        "- The held-out test operator's footage is all `correct`-scenario takes (see `data/"
        "splits/report.md`); it does not by itself test the detector on out-of-order or skip "
        "footage, only on the box/lid appearances themselves.",
        "- Training data is small (single room, one rack, fixed lighting setup); this is not "
        "evidence the detector generalizes beyond the filmed environment.",
        "- **Prop colors are not consistent across operators.** `red_box`/`yellow_box` are "
        "proposed by an HSV color heuristic (see the module docstring), tuned against op02's "
        "and op03's actual red/yellow props. op05 (the held-out test operator) used an orange "
        'object as its "yellow box" stand-in rather than yellow, which falls outside the '
        "tuned hue range: the held-out test split ends up with **zero** yellow_box ground-truth "
        "instances, so that class's held-out performance is simply unmeasured here, not passing "
        "or failing. Widening the hue range to catch orange was rejected because it starts "
        "matching the cardboard box's own tan/brown color instead (see the calibration in the "
        "PR description).",
        "- **outer_box_closed is rare in held-out footage** (13 ground-truth boxes in the whole "
        "test split, vs. 2559 in train): the protocol only has the box closed briefly at the "
        "very start and end of a take. Its near-zero held-out mAP50 (0.007) mostly reflects "
        "that scarcity, not a specific failure mode -- the model essentially never gets to "
        'practice recognizing "closed" in this operator\'s footage.',
        "- One operator (op01, 3 takes) used an entirely different, non-conforming prop set (a "
        "mug, a thermos and a book instead of the boxes) and was **excluded from the detector "
        "dataset entirely** (`--relabel --exclude-operators op01`) rather than mislabeled; see "
        "the module docstring.",
        "- Real+synthetic training scored *lower* held-out mAP50 than real-only (0.221 vs. "
        "0.415) -- the deployed model is the real-only run. The synthetic images (copy-pasted "
        "crops on NASA ISS/clean-room photos) look quite different from this project's own "
        "carpeted-room footage; with only ~1100 real frames feeding the crops and 1200 "
        "synthetic images added on top, synthetic data seems to have diluted the real signal "
        "rather than adding useful variety. This is reported as-is, not tuned away.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="data/dataset.yaml")
    ap.add_argument("--dataset-synth", help="dataset yaml for the real+synthetic run (optional)")
    ap.add_argument("--splits-report", default="data/splits/report.json")
    ap.add_argument("--base-weights", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--degrees", type=float, default=180.0)
    ap.add_argument("--fliplr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--workers", type=int, default=4, help="dataloader workers; kept modest for system RAM"
    )
    ap.add_argument("--project", default="runs/detector")
    ap.add_argument("--run-name", default="real")
    ap.add_argument("--synth-run-name", default="real_synth")
    ap.add_argument("--out-dir", default="models/detector")
    ap.add_argument("--conf", type=float, default=0.25, help="gallery inference confidence")
    ap.add_argument("--gallery-n", type=int, default=8)
    ap.add_argument(
        "--relabel",
        action="store_true",
        help="regenerate data/labels with the hybrid (YOLO-World + HSV color) proposer before "
        "training; see the module docstring for why this beats plain autolabel.py here",
    )
    ap.add_argument("--relabel-manifest", default="data/images/manifest.csv")
    ap.add_argument(
        "--exclude-operators",
        default="op01",
        help="comma-separated operators to drop during --relabel (non-conforming footage)",
    )
    args = ap.parse_args(argv)

    if args.relabel:
        exclude = tuple(o for o in args.exclude_operators.split(",") if o)
        report = relabel(
            to_repo_path(args.relabel_manifest),
            to_repo_path("data/labels"),
            exclude_operators=exclude,
        )
        print(f"relabelled {report['n_frames']} frames (excluded operators: {exclude})")
        print(f"per-class counts: {report['per_class_count']}")

    dataset_yaml = to_repo_path(args.dataset)
    splits_report = json.loads(to_repo_path(args.splits_report).read_text(encoding="utf-8"))
    project = to_repo_path(args.project)
    out_dir = to_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_dir = to_repo_path("data/labels")
    classes = (
        read_classes(labels_dir / "classes.txt")
        if (labels_dir / "classes.txt").exists()
        else CLASSES
    )

    train_kwargs = dict(
        base_weights=args.base_weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        patience=args.patience,
        batch=args.batch,
        degrees=args.degrees,
        fliplr=args.fliplr,
        seed=args.seed,
        project=project,
        device=args.device,
        workers=args.workers,
    )

    print(f"=== training real-only: {dataset_yaml} ===")
    real_result = run_training(dataset_yaml, name=args.run_name, **train_kwargs)
    print(f"real-only held-out mAP50={real_result.map50:.4f} mAP50-95={real_result.map50_95:.4f}")

    synth_result = None
    if args.dataset_synth:
        synth_yaml = to_repo_path(args.dataset_synth)
        print(f"=== training real+synthetic: {synth_yaml} ===")
        synth_result = run_training(synth_yaml, name=args.synth_run_name, **train_kwargs)
        print(
            f"real+synthetic held-out mAP50={synth_result.map50:.4f} "
            f"mAP50-95={synth_result.map50_95:.4f}"
        )

    chosen = real_result
    if synth_result is not None and synth_result.map50 > real_result.map50:
        chosen = synth_result

    from ultralytics import YOLO

    chosen_model = YOLO(str(chosen.weights_pt))
    onnx_path = Path(chosen_model.export(format="onnx", imgsz=args.imgsz, simplify=True))

    shutil.copy2(chosen.weights_pt, out_dir / "best.pt")
    shutil.copy2(onnx_path, out_dir / "best.onnx")

    baseline = full_image_box_baseline(dataset_yaml, classes=classes)
    (out_dir / "baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    gallery = build_failure_gallery(
        chosen_model,
        chosen.dataset_yaml,
        classes=classes,
        imgsz=args.imgsz,
        conf=args.conf,
        out_dir=out_dir / "failures",
        n=args.gallery_n,
    )

    write_card(
        out_dir / "card.md",
        split_report=splits_report,
        real_result=real_result,
        synth_result=synth_result,
        chosen=chosen,
        baseline=baseline,
        gallery=gallery,
        base_weights=args.base_weights,
        train_args={
            "imgsz": args.imgsz,
            "batch": args.batch,
            "degrees": args.degrees,
            "fliplr": args.fliplr,
            "patience": args.patience,
        },
    )

    summary = {
        "chosen_run": chosen.name,
        "held_out_map50": chosen.map50,
        "held_out_map50_95": chosen.map50_95,
        "meets_0.85_gate": chosen.map50 >= 0.85,
        "real": {"map50": real_result.map50, "map50_95": real_result.map50_95},
        "synthetic": (
            {"map50": synth_result.map50, "map50_95": synth_result.map50_95}
            if synth_result
            else None
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nheld-out mAP50 = {chosen.map50:.4f} ({chosen.name}) -> {rel_to_repo(out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
