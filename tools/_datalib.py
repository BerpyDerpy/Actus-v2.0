"""Shared on-disk conventions for the T-DATA tools: extract_frames, autolabel, review_labels,
make_splits and synth_augment. Not a task of its own — each tool above owns its CLI and
behaviour — but their outputs compose into one pipeline through the conventions below:

    data/raw/{op}_{scenario}_{n}.mp4 + .csv          T-REC's take + label CSV
        -> extract_frames.py -> data/images/{stem}/frame_{idx:06d}.jpg + data/images/manifest.csv
        -> autolabel.py      -> data/labels/{stem}/frame_{idx:06d}.txt (YOLO) + classes.txt
        -> review_labels.py  -> edits the same .txt files in place + data/labels/review_status.csv
        -> make_splits.py    -> data/splits/{train,val,test}.txt + report.json/.md + dataset.yaml
        -> synth_augment.py  -> data/synthetic/images|labels + data/splits/synthetic.txt

Label files are exactly the Ultralytics YOLO format (5 columns, class index into ``CLASSES``), so
``data/dataset.yaml`` trains as-is. The extracted-frames directory is named ``images``, not
``frames``: Ultralytics finds an image's label by replacing the literal path segment
``/images/`` with ``/labels/``, so the sibling ``data/images/`` / ``data/labels/`` names are load
bearing, not cosmetic.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.record_takes import LabelTrack, parse_take_stem  # noqa: E402

# The 5 detector classes named in build-plan.md's T-DATA card, in class-index order. This matches
# the key order of configs/default.yaml's detector.prompts.
CLASSES: tuple[str, ...] = (
    "outer_box_closed",
    "outer_box_open",
    "lid",
    "red_box",
    "yellow_box",
)

MANIFEST_FIELDS = (
    "image_path",
    "take",
    "operator",
    "scenario",
    "n",
    "frame_idx",
    "t",
    "label",
    "sample",
)


def to_repo_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def rel_to_repo(path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


# --------------------------------------------------------------------------------------------
# Takes (T-REC's output)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Take:
    """One take as T-REC wrote it: the primary-camera video, its label CSV, and (if present)
    the session metadata JSON."""

    stem: str
    operator: str
    scenario: str
    n: int
    video_path: Path
    labels_path: Path
    meta_path: Path

    def label_track(self) -> LabelTrack:
        return LabelTrack.load(self.labels_path)

    def meta(self) -> dict:
        if self.meta_path.exists():
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        return {}

    def duration_s(self) -> float | None:
        meta = self.meta()
        return float(meta["duration_s"]) if "duration_s" in meta else None


def discover_takes(raw_dir: str | Path) -> list[Take]:
    """Every primary-camera take in ``raw_dir`` that has a label CSV. Skips ``*_cam2`` files:
    they share the primary camera's frame indices and labels."""
    raw_dir = Path(raw_dir)
    out = []
    for video in sorted(raw_dir.glob("*.mp4")):
        try:
            info = parse_take_stem(video.stem)
        except ValueError:
            continue
        if info["cam"] != 1:
            continue
        labels = video.with_suffix(".csv")
        if not labels.exists():
            continue
        out.append(
            Take(
                video.stem,
                info["operator"],
                info["scenario"],
                info["n"],
                video,
                labels,
                video.with_suffix(".json"),
            )
        )
    return out


# --------------------------------------------------------------------------------------------
# Frame manifest (extract_frames -> autolabel / make_splits / synth_augment)
# --------------------------------------------------------------------------------------------


def write_manifest(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in MANIFEST_FIELDS})


def read_manifest(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------------------------
# YOLO label files (autolabel <-> review_labels <-> synth_augment)
# --------------------------------------------------------------------------------------------


def write_classes(path: str | Path, classes: tuple[str, ...] = CLASSES) -> None:
    Path(path).write_text("\n".join(classes) + "\n", encoding="utf-8")


def read_classes(path: str | Path) -> tuple[str, ...]:
    text = Path(path).read_text(encoding="utf-8")
    return tuple(line.strip() for line in text.splitlines() if line.strip())


@dataclass(frozen=True)
class Box:
    """One box, in image pixel coordinates. ``conf`` is 1.0 for an accepted / synthetic label."""

    cls: str
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0


def write_yolo_label(
    path: str | Path, boxes: list[Box], img_w: int, img_h: int, classes: tuple[str, ...] = CLASSES
) -> None:
    lines = []
    for b in boxes:
        cid = classes.index(b.cls)
        cx, cy = (b.x1 + b.x2) / 2 / img_w, (b.y1 + b.y2) / 2 / img_h
        w, h = (b.x2 - b.x1) / img_w, (b.y2 - b.y1) / img_h
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_yolo_label(
    path: str | Path, img_w: int, img_h: int, classes: tuple[str, ...] = CLASSES
) -> list[Box]:
    path = Path(path)
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cid_s, cx_s, cy_s, w_s, h_s = line.split()[:5]
        cid, cx, cy, w, h = int(cid_s), float(cx_s), float(cy_s), float(w_s), float(h_s)
        x1, y1 = (cx - w / 2) * img_w, (cy - h / 2) * img_h
        x2, y2 = (cx + w / 2) * img_w, (cy + h / 2) * img_h
        boxes.append(Box(classes[cid], x1, y1, x2, y2))
    return boxes
