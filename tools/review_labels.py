"""T-DATA: fast OpenCV review of autolabel's proposed boxes.

    uv run python tools/review_labels.py --manifest data/images/manifest.csv \\
        --labels-dir data/labels

Boxes are numbered in the overlay next to their class and confidence (if any). Keys:

    1-9       select that box                    d          delete the selected box
    c then 1-5  relabel the selected box          n then drag then 1-5   draw a new box
    space/Enter accept the frame and advance      b          back to the previous frame
    u         undo the last edit on this frame    q / Esc    save and quit

Target throughput: 400 frames/hour. Edits are written straight back to the YOLO ``.txt`` files;
``data/labels/review_status.csv`` records which frames have been accepted.

Headless / scripted mode (tests, batch review of frames known to already be correct):

    uv run python tools/review_labels.py --no-preview --script "space;d,space;r4,space" ...

``--script`` is a ``;``-separated list of per-frame op groups, each a ``,``-separated list of
ops: a bare digit selects a box, ``draw:x1,y1,x2,y2,C`` adds a box of class ``C`` (1-based),
``rC`` relabels the selected box to class ``C``, ``d`` deletes it, ``u`` undoes, and
``space``/``accept`` accepts the frame (frames with no op group are accepted as proposed).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2

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

STATUS_FIELDS = ("image_path", "take", "frame_idx", "reviewed", "n_boxes", "reviewed_at")

COLORS = {
    "outer_box_closed": (150, 90, 40),
    "outer_box_open": (200, 130, 60),
    "lid": (170, 170, 60),
    "red_box": (40, 40, 210),
    "yellow_box": (40, 210, 230),
}


@dataclass
class FrameReview:
    """One frame's editable box list, with undo."""

    image_path: Path
    label_path: Path
    img_w: int
    img_h: int
    boxes: list[Box]
    classes: tuple[str, ...] = CLASSES
    selected: int | None = None
    _history: list[list[Box]] = field(default_factory=list)

    def _push(self) -> None:
        self._history.append(list(self.boxes))

    def select(self, i: int) -> None:
        self.selected = i if 0 <= i < len(self.boxes) else None

    def delete_selected(self) -> None:
        if self.selected is None:
            return
        self._push()
        del self.boxes[self.selected]
        self.selected = None

    def relabel_selected(self, cls_name: str) -> None:
        if self.selected is None:
            return
        self._push()
        b = self.boxes[self.selected]
        self.boxes[self.selected] = Box(cls_name, b.x1, b.y1, b.x2, b.y2, b.conf)

    def add_box(self, x1: float, y1: float, x2: float, y2: float, cls_name: str) -> None:
        if x2 <= x1 or y2 <= y1:
            return
        self._push()
        self.boxes.append(Box(cls_name, x1, y1, x2, y2, 1.0))
        self.selected = len(self.boxes) - 1

    def undo(self) -> None:
        if self._history:
            self.boxes = self._history.pop()
            self.selected = None

    def save(self) -> None:
        write_yolo_label(self.label_path, self.boxes, self.img_w, self.img_h, self.classes)


def load_review(row: dict, labels_dir: Path, classes: tuple[str, ...] = CLASSES) -> FrameReview:
    image_path = REPO_ROOT / row["image_path"]
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]
    label_path = labels_dir / row["take"] / f"frame_{int(row['frame_idx']):06d}.txt"
    boxes = read_yolo_label(label_path, w, h, classes)
    return FrameReview(image_path, label_path, w, h, boxes, classes)


class ReviewStatus:
    """``data/labels/review_status.csv``: which frames have been accepted."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}

    @classmethod
    def load(cls, path: Path) -> ReviewStatus:
        status = cls()
        if path.exists():
            with open(path, encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    status.rows[row["image_path"]] = row
        return status

    def mark(self, review: FrameReview, image_path_rel: str, take: str, frame_idx: int) -> None:
        self.rows[image_path_rel] = {
            "image_path": image_path_rel,
            "take": take,
            "frame_idx": frame_idx,
            "reviewed": "1",
            "n_boxes": len(review.boxes),
            "reviewed_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=STATUS_FIELDS)
            w.writeheader()
            for row in self.rows.values():
                w.writerow(row)
        tmp.replace(path)


# --------------------------------------------------------------------------------------------
# Headless / scripted driver (also the core of the interactive loop below)
# --------------------------------------------------------------------------------------------


def parse_script(spec: str) -> list[list[str]]:
    """``"space;d,space;r4,space"`` -> per-frame op lists, applied in manifest row order."""
    if not spec:
        return []
    return [group.split(",") for group in spec.split(";")]


def apply_op(review: FrameReview, op: str, classes: tuple[str, ...]) -> bool:
    """Apply one scripted op to ``review``. Returns True if the op accepts (finishes) the frame."""
    op = op.strip()
    if op in ("space", "accept", "a"):
        return True
    if op in ("u", "undo"):
        review.undo()
    elif op in ("d", "delete"):
        review.delete_selected()
    elif op.startswith("r") and op[1:].isdigit():
        idx = int(op[1:]) - 1
        if 0 <= idx < len(classes):
            review.relabel_selected(classes[idx])
    elif op.startswith("draw:"):
        x1, y1, x2, y2, c = (float(v) for v in op[len("draw:") :].split(","))
        idx = int(c) - 1
        if 0 <= idx < len(classes):
            review.add_box(x1, y1, x2, y2, classes[idx])
    elif op.isdigit():
        review.select(int(op) - 1)
    return False


def run_review(
    rows: list[dict],
    labels_dir: Path,
    ops_by_row: list[list[str]],
    *,
    classes: tuple[str, ...] = CLASSES,
    status: ReviewStatus | None = None,
) -> ReviewStatus:
    status = status if status is not None else ReviewStatus()
    for i, row in enumerate(rows):
        ops = ops_by_row[i] if i < len(ops_by_row) else []
        review = load_review(row, labels_dir, classes)
        accepted = not ops  # a frame with no scripted ops is accepted as proposed
        for op in ops:
            if apply_op(review, op, classes):
                accepted = True
        if accepted:
            review.save()
            status.mark(review, row["image_path"], row["take"], int(row["frame_idx"]))
    return status


# --------------------------------------------------------------------------------------------
# Interactive loop (not unit-tested: it needs a real display; the logic above is)
# --------------------------------------------------------------------------------------------


def render(review: FrameReview, drag: tuple[tuple[int, int], tuple[int, int]] | None):
    img = cv2.imread(str(review.image_path)).copy()
    for i, b in enumerate(review.boxes):
        color = COLORS.get(b.cls, (255, 255, 255))
        thick = 3 if i == review.selected else 1
        cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, thick)
        tag = f"{i + 1}:{b.cls}" + (f" {b.conf:.2f}" if b.conf < 1.0 else "")
        cv2.putText(
            img, tag, (int(b.x1), max(12, int(b.y1) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )  # fmt: skip
    if drag:
        cv2.rectangle(img, drag[0], drag[1], (255, 255, 255), 1)
    lines = [
        "1-9 select  c+1-5 relabel  d delete  n+drag+1-5 new box",
        "space accept  b back  u undo  q quit  classes: "
        + " ".join(f"{i + 1}={c}" for i, c in enumerate(review.classes)),
    ]
    y = img.shape[0] - 30
    for ln in lines:
        cv2.putText(img, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(
            img, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )
        y += 16
    return img


def _interactive_review(
    rows: list[dict], labels_dir: Path, classes: tuple[str, ...], status: ReviewStatus
) -> ReviewStatus:
    window = "ACTUS label reviewer"
    drag_state: dict[str, tuple[int, int] | None] = {"start": None, "cur": None}

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: None) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            drag_state["start"], drag_state["cur"] = (x, y), (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drag_state["start"]:
            drag_state["cur"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drag_state["start"]:
            drag_state["done"] = (drag_state["start"], (x, y))
            drag_state["start"] = None

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    mode = "normal"  # normal | await_relabel | drawing | await_draw_class
    pending_draw: tuple[tuple[int, int], tuple[int, int]] | None = None

    i = 0
    review = load_review(rows[i], labels_dir, classes)
    while 0 <= i < len(rows):
        drag = None
        if mode == "drawing" and drag_state["start"]:
            drag = (drag_state["start"], drag_state["cur"])
        cv2.imshow(window, render(review, drag))
        key = cv2.waitKey(20) & 0xFF
        done = drag_state.pop("done", None)
        if mode == "drawing" and done:
            pending_draw, mode = done, "await_draw_class"

        if key in (27, ord("q")):
            break
        elif key in (ord(" "), 13):
            review.save()
            status.mark(review, rows[i]["image_path"], rows[i]["take"], int(rows[i]["frame_idx"]))
            i += 1
            if i < len(rows):
                review = load_review(rows[i], labels_dir, classes)
            mode = "normal"
        elif key == ord("b") and i > 0:
            i -= 1
            review = load_review(rows[i], labels_dir, classes)
            mode = "normal"
        elif key == ord("u"):
            review.undo()
        elif key == ord("d"):
            review.delete_selected()
        elif key == ord("c"):
            mode = "await_relabel"
        elif key == ord("n"):
            mode = "drawing"
        elif mode == "await_relabel" and chr(key).isdigit() and key != ord("0"):
            idx = int(chr(key)) - 1
            if 0 <= idx < len(classes):
                review.relabel_selected(classes[idx])
            mode = "normal"
        elif mode == "await_draw_class" and chr(key).isdigit() and key != ord("0"):
            idx = int(chr(key)) - 1
            if 0 <= idx < len(classes) and pending_draw:
                (x1, y1), (x2, y2) = pending_draw
                review.add_box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), classes[idx])
            mode, pending_draw = "normal", None
        elif mode == "normal" and chr(key).isdigit():
            review.select(int(chr(key)) - 1)
    cv2.destroyAllWindows()
    return status


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Review autolabel's proposed boxes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--manifest", default="data/images/manifest.csv")
    ap.add_argument("--labels-dir", default="data/labels")
    ap.add_argument("--status", help="review status CSV path (default: under --labels-dir)")
    ap.add_argument("--only-unreviewed", action="store_true")
    ap.add_argument("--no-preview", action="store_true", help="headless: use --script, no window")
    ap.add_argument("--script", default="", help='per-frame ops, e.g. "space;d,space;r4,space"')
    args = ap.parse_args(argv)

    manifest_path = to_repo_path(args.manifest)
    labels_dir = to_repo_path(args.labels_dir)
    classes_path = labels_dir / "classes.txt"
    classes = read_classes(classes_path) if classes_path.exists() else CLASSES
    status_path = to_repo_path(args.status) if args.status else labels_dir / "review_status.csv"
    status = ReviewStatus.load(status_path)

    rows = read_manifest(manifest_path)
    if args.only_unreviewed:
        rows = [r for r in rows if r["image_path"] not in status.rows]
    if not rows:
        print("nothing to review")
        return 0

    if args.no_preview or args.script:
        ops = parse_script(args.script)
        status = run_review(rows, labels_dir, ops, classes=classes, status=status)
    else:
        status = _interactive_review(rows, labels_dir, classes, status)

    status.save(status_path)
    n_reviewed = sum(1 for r in status.rows.values() if r["reviewed"] == "1")
    print(f"reviewed {n_reviewed}/{len(rows)} frames -> {rel_to_repo(status_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
