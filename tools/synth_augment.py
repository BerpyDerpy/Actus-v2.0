"""T-DATA: synthetic data generation — copy-paste augmentation for detector training.

Crops boxes out of reviewed real frames (``data/labels``), then pastes randomly rotated
(full 360 degrees), scaled, brightness-jittered and blurred copies onto background images (e.g.
public-domain ISS-interior photos placed in a local directory — this tool never downloads
anything itself; "a download during setup is fine" happens elsewhere, not at run time). Output
is a separate ``synthetic`` split so it can be ablated against real-only training.

    uv run python tools/synth_augment.py --manifest data/images/manifest.csv \\
        --labels-dir data/labels --backgrounds-dir data/backgrounds/iss --n-images 200

Crops are bounding-box rectangles with a feathered alpha edge, not full instance-segmentation
masks (the dataset only has boxes to draw from); pass ``--grabcut`` for a slower, tighter
per-crop foreground refinement.

Writes ``data/synthetic/images/synth_{i:06d}.jpg``, ``data/synthetic/labels/synth_{i:06d}.txt``
(YOLO format), ``data/splits/synthetic.txt`` and ``data/synthetic/report.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

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
    write_yolo_label,
)


@dataclass(frozen=True)
class Crop:
    cls: str
    bgr: np.ndarray  # (h, w, 3) uint8
    alpha: np.ndarray  # (h, w) float32 in [0, 1]


def _feathered_alpha(h: int, w: int, feather: int = 4) -> np.ndarray:
    alpha = np.ones((h, w), np.float32)
    if feather > 0 and h > 2 * feather and w > 2 * feather:
        ramp = np.linspace(0, 1, feather, dtype=np.float32)
        alpha[:feather, :] *= ramp[:, None]
        alpha[-feather:, :] *= ramp[::-1, None]
        alpha[:, :feather] *= ramp[None, :]
        alpha[:, -feather:] *= ramp[None, ::-1]
    return alpha


def _grabcut_alpha(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    margin = max(1, min(h, w) // 10)
    if w - 2 * margin < 1 or h - 2 * margin < 1:
        return _feathered_alpha(h, w)
    mask = np.zeros((h, w), np.uint8)
    rect = (margin, margin, w - 2 * margin, h - 2 * margin)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return _feathered_alpha(h, w)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    return cv2.GaussianBlur(fg, (5, 5), 0) if fg.max() > 0 else _feathered_alpha(h, w)


def extract_crops(
    rows: list[dict],
    labels_dir: Path,
    classes: tuple[str, ...] = CLASSES,
    *,
    grabcut: bool = False,
    margin_px: int = 3,
) -> list[Crop]:
    crops: list[Crop] = []
    for row in rows:
        image_path = REPO_ROOT / row["image_path"]
        label_path = labels_dir / row["take"] / f"frame_{int(row['frame_idx']):06d}.txt"
        img = cv2.imread(str(image_path))
        if img is None or not label_path.exists():
            continue
        h, w = img.shape[:2]
        for b in read_yolo_label(label_path, w, h, classes):
            x1, y1 = max(0, int(b.x1) - margin_px), max(0, int(b.y1) - margin_px)
            x2, y2 = min(w, int(b.x2) + margin_px), min(h, int(b.y2) + margin_px)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            patch = img[y1:y2, x1:x2]
            alpha = _grabcut_alpha(patch) if grabcut else _feathered_alpha(*patch.shape[:2])
            crops.append(Crop(b.cls, patch, alpha))
    return crops


def _rotate_rgba(
    bgr: np.ndarray, alpha: np.ndarray, degrees: float
) -> tuple[np.ndarray, np.ndarray]:
    h, w = bgr.shape[:2]
    diag = math.ceil(math.hypot(h, w))
    cx, cy = w / 2, h / 2
    m = cv2.getRotationMatrix2D((cx, cy), degrees, 1.0)
    m[0, 2] += diag / 2 - cx
    m[1, 2] += diag / 2 - cy
    rot_bgr = cv2.warpAffine(bgr, m, (diag, diag), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    rot_a = cv2.warpAffine(alpha, m, (diag, diag), flags=cv2.INTER_LINEAR, borderValue=0)
    return rot_bgr, rot_a


def _alpha_bbox(alpha: np.ndarray, thresh: float = 0.15) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(alpha >= thresh)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def compose_image(
    background: np.ndarray,
    crops: list[Crop],
    rng: np.random.Generator,
    *,
    scale_range: tuple[float, float] = (0.6, 1.6),
    brightness_range: tuple[float, float] = (0.75, 1.3),
    blur_prob: float = 0.35,
) -> tuple[np.ndarray, list[Box]]:
    canvas = background.copy()
    ch, cw = canvas.shape[:2]
    boxes: list[Box] = []
    for crop in crops:
        scale = float(rng.uniform(*scale_range))
        h0, w0 = crop.bgr.shape[:2]
        size = (max(1, round(w0 * scale)), max(1, round(h0 * scale)))
        bgr = cv2.resize(crop.bgr, size)
        alpha = cv2.resize(crop.alpha, size)
        bgr, alpha = _rotate_rgba(bgr, alpha, float(rng.uniform(0, 360)))  # full 360 deg rotation

        gain = float(rng.uniform(*brightness_range))
        bgr = np.clip(bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        if rng.random() < blur_prob:
            k = int(rng.choice([3, 5]))
            bgr = cv2.GaussianBlur(bgr, (k, k), 0)

        crop_box = _alpha_bbox(alpha)
        if crop_box is None:
            continue
        bx1, by1, bx2, by2 = crop_box
        bh, bw = by2 - by1, bx2 - bx1
        if bh >= ch or bw >= cw:
            continue
        x0 = int(rng.integers(0, cw - bw))
        y0 = int(rng.integers(0, ch - bh))

        patch = bgr[by1:by2, bx1:bx2].astype(np.float32)
        a = alpha[by1:by2, bx1:bx2][:, :, None]
        region = canvas[y0 : y0 + bh, x0 : x0 + bw].astype(np.float32)
        canvas[y0 : y0 + bh, x0 : x0 + bw] = (a * patch + (1 - a) * region).astype(np.uint8)
        boxes.append(Box(crop.cls, float(x0), float(y0), float(x0 + bw), float(y0 + bh)))
    return canvas, boxes


def _load_background(path: Path, size: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    w, h = size
    ih, iw = img.shape[:2]
    scale = max(w / iw, h / ih)
    img = cv2.resize(img, (max(1, round(iw * scale)), max(1, round(ih * scale))))
    ih, iw = img.shape[:2]
    x0 = int(rng.integers(0, max(1, iw - w + 1)))
    y0 = int(rng.integers(0, max(1, ih - h + 1)))
    return img[y0 : y0 + h, x0 : x0 + w].copy()


def generate(
    crops: list[Crop],
    backgrounds: list[Path],
    out_dir: Path,
    *,
    n_images: int,
    objects_per_image: tuple[int, int],
    canvas_size: tuple[int, int] = (640, 480),
    seed: int = 0,
    classes: tuple[str, ...] = CLASSES,
) -> dict:
    rng = np.random.default_rng(seed)
    images_dir, labels_dir = out_dir / "images", out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    per_class = dict.fromkeys(classes, 0)
    for i in range(n_images):
        bg_path = backgrounds[int(rng.integers(0, len(backgrounds)))]
        bg = _load_background(bg_path, canvas_size, rng)
        k = int(rng.integers(objects_per_image[0], objects_per_image[1] + 1))
        chosen = [crops[int(j)] for j in rng.integers(0, len(crops), size=min(k, len(crops)))]
        img, boxes = compose_image(bg, chosen, rng)

        stem = f"synth_{i:06d}"
        cv2.imwrite(str(images_dir / f"{stem}.jpg"), img)
        write_yolo_label(labels_dir / f"{stem}.txt", boxes, canvas_size[0], canvas_size[1], classes)
        for b in boxes:
            per_class[b.cls] += 1

    return {
        "n_images": n_images,
        "n_boxes": sum(per_class.values()),
        "per_class_count": per_class,
        "backgrounds": [rel_to_repo(p) for p in backgrounds],
        "canvas_size": list(canvas_size),
        "seed": seed,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default="data/images/manifest.csv")
    ap.add_argument("--labels-dir", default="data/labels")
    ap.add_argument("--backgrounds-dir", required=True)
    ap.add_argument("--out-dir", default="data/synthetic")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--n-images", type=int, default=200)
    ap.add_argument("--objects-per-image", default="1-3")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grabcut", action="store_true")
    args = ap.parse_args(argv)

    manifest_path = to_repo_path(args.manifest)
    labels_dir = to_repo_path(args.labels_dir)
    backgrounds_dir = to_repo_path(args.backgrounds_dir)
    out_dir = to_repo_path(args.out_dir)
    splits_dir = to_repo_path(args.splits_dir)

    backgrounds = sorted(
        p for p in backgrounds_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not backgrounds:
        print(f"no background images in {backgrounds_dir}; supply real ISS-interior imagery first")
        return 1

    rows = read_manifest(manifest_path)
    if not rows:
        print(f"{manifest_path} is empty", file=sys.stderr)
        return 1
    classes_path = labels_dir / "classes.txt"
    classes = read_classes(classes_path) if classes_path.exists() else CLASSES
    crops = extract_crops(rows, labels_dir, classes, grabcut=args.grabcut)
    if not crops:
        print(f"no reviewed boxes under {labels_dir}; run autolabel.py / review_labels.py first")
        return 1

    lo, hi = (int(v) for v in args.objects_per_image.split("-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    report = generate(
        crops,
        backgrounds,
        out_dir,
        n_images=args.n_images,
        objects_per_image=(lo, hi),
        canvas_size=(args.width, args.height),
        seed=args.seed,
        classes=classes,
    )
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    splits_dir.mkdir(parents=True, exist_ok=True)
    img_paths = sorted((out_dir / "images").glob("*.jpg"))
    (splits_dir / "synthetic.txt").write_text(
        "\n".join(rel_to_repo(p) for p in img_paths) + ("\n" if img_paths else ""), encoding="utf-8"
    )

    print(f"generated {report['n_images']} images, {report['n_boxes']} boxes -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
