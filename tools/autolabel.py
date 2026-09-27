"""T-DATA: propose YOLO-format box labels for extract_frames' output, for human review.

The default backend is YOLO-World (open-vocabulary), matching the runtime fallback detector in
``configs/default.yaml``'s ``detector.prompts`` — one prompt per class in ``_datalib.CLASSES``.
Proposals are *unreviewed*: run tools/review_labels.py on them before training.

    uv run python tools/autolabel.py --manifest data/images/manifest.csv --out-dir data/labels

The default backend requires ``uv sync --extra perception``; ultralytics is imported lazily so
the base install and CI never need it. Any ``propose(image) -> list[Box]`` callable can be used
instead of ``YoloWorldProposer`` (see ``autolabel_manifest``) — this is how tests exercise the
file I/O without ultralytics installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._datalib import (  # noqa: E402
    CLASSES,
    Box,
    read_manifest,
    rel_to_repo,
    to_repo_path,
    write_classes,
    write_yolo_label,
)

Proposer = Callable[[np.ndarray], list[Box]]


class YoloWorldProposer:
    """Open-vocabulary proposals via ultralytics YOLO-World."""

    def __init__(
        self,
        prompts: dict[str, str],
        *,
        weights: str = "yolov8s-worldv2.pt",
        conf: float = 0.15,
        classes: tuple[str, ...] = CLASSES,
    ) -> None:
        from ultralytics import YOLO  # heavy; only imported when this backend is actually used

        self.classes = classes
        self.conf = conf
        self.model = YOLO(weights)
        self.model.set_classes([prompts[c] for c in classes])

    def __call__(self, image: np.ndarray) -> list[Box]:
        result = self.model.predict(image, conf=self.conf, verbose=False)[0]
        out = []
        for box in result.boxes:
            idx = int(box.cls.item())
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            out.append(Box(self.classes[idx], x1, y1, x2, y2, float(box.conf.item())))
        return out


def autolabel_manifest(
    rows: list[dict],
    propose: Proposer,
    out_dir: Path,
    *,
    classes: tuple[str, ...] = CLASSES,
    min_conf: float = 0.0,
) -> dict:
    """Run ``propose`` over every manifest row and write YOLO labels under ``out_dir``. Returns a
    summary report (per-class counts and mean confidence)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_classes(out_dir / "classes.txt", classes)

    per_class_n = dict.fromkeys(classes, 0)
    per_class_conf: dict[str, list[float]] = {c: [] for c in classes}
    n_empty = 0

    for row in rows:
        image_path = REPO_ROOT / row["image_path"]
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"manifest references a missing image: {image_path}")
        h, w = img.shape[:2]
        boxes = [b for b in propose(img) if b.conf >= min_conf]
        if not boxes:
            n_empty += 1
        for b in boxes:
            per_class_n[b.cls] += 1
            per_class_conf[b.cls].append(b.conf)

        label_path = out_dir / row["take"] / f"frame_{int(row['frame_idx']):06d}.txt"
        write_yolo_label(label_path, boxes, w, h, classes)

    report = {
        "n_frames": len(rows),
        "n_frames_with_no_proposals": n_empty,
        "min_conf": min_conf,
        "classes": list(classes),
        "per_class_count": per_class_n,
        "per_class_mean_conf": {
            c: (sum(v) / len(v) if v else None) for c, v in per_class_conf.items()
        },
    }
    (out_dir / "autolabel_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default="data/images/manifest.csv")
    ap.add_argument("--out-dir", default="data/labels")
    ap.add_argument("--weights", default="yolov8s-worldv2.pt")
    ap.add_argument("--conf", type=float, default=0.15)
    args = ap.parse_args(argv)

    manifest_path = to_repo_path(args.manifest)
    rows = read_manifest(manifest_path)
    if not rows:
        print(f"{manifest_path} is empty", file=sys.stderr)
        return 1

    from actus.core.config import load_config

    cfg = load_config()
    prompts = {c: cfg.detector.prompts[c] for c in CLASSES}
    proposer = YoloWorldProposer(prompts, weights=args.weights, conf=args.conf)

    out_dir = to_repo_path(args.out_dir)
    report = autolabel_manifest(rows, proposer, out_dir, min_conf=args.conf)
    print(
        f"labelled {report['n_frames']} frames "
        f"({report['n_frames_with_no_proposals']} with no proposals) -> {rel_to_repo(out_dir)}\n"
        f"counts: {report['per_class_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
