import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._datalib import Box, read_yolo_label, write_manifest, write_yolo_label  # noqa: E402
from tools.review_labels import (  # noqa: E402
    FrameReview,
    ReviewStatus,
    apply_op,
    load_review,
    main,
    parse_script,
    run_review,
)


def test_frame_review_select_delete_undo():
    boxes = [Box("red_box", 1, 1, 5, 5), Box("yellow_box", 10, 10, 20, 20)]
    review = FrameReview(Path("x.jpg"), Path("x.txt"), 100, 100, list(boxes))
    review.select(1)
    assert review.selected == 1
    review.delete_selected()
    assert review.boxes == [boxes[0]]
    assert review.selected is None
    review.undo()
    assert review.boxes == boxes


def test_frame_review_relabel_and_add_box():
    review = FrameReview(Path("x.jpg"), Path("x.txt"), 100, 100, [Box("red_box", 1, 1, 5, 5)])
    review.select(0)
    review.relabel_selected("lid")
    assert review.boxes[0].cls == "lid"

    review.add_box(10, 10, 20, 20, "outer_box_open")
    assert len(review.boxes) == 2
    assert review.selected == 1  # new box auto-selected

    review.add_box(5, 5, 5, 5, "lid")  # degenerate box: ignored
    assert len(review.boxes) == 2


def test_frame_review_save_round_trips_yolo(tmp_path):
    label_path = tmp_path / "frame_000000.txt"
    review = FrameReview(
        tmp_path / "img.jpg", label_path, 200, 100, [Box("red_box", 20, 10, 80, 40)]
    )
    review.save()
    boxes = read_yolo_label(label_path, 200, 100)
    assert len(boxes) == 1
    assert boxes[0].cls == "red_box"


def test_review_status_load_save_round_trip(tmp_path):
    review = FrameReview(Path("x.jpg"), Path("x.txt"), 10, 10, [Box("red_box", 0, 0, 5, 5)])
    status = ReviewStatus()
    status.mark(review, "data/frames/a/frame_000000.jpg", "a", 0)
    path = tmp_path / "review_status.csv"
    status.save(path)

    loaded = ReviewStatus.load(path)
    assert "data/frames/a/frame_000000.jpg" in loaded.rows
    assert loaded.rows["data/frames/a/frame_000000.jpg"]["n_boxes"] == "1"


def test_parse_script_and_apply_op():
    groups = parse_script("space;d,space;r4,space")
    assert groups == [["space"], ["d", "space"], ["r4", "space"]]

    review = FrameReview(Path("x.jpg"), Path("x.txt"), 10, 10, [Box("red_box", 0, 0, 5, 5)])
    assert apply_op(review, "1", ("red_box", "yellow_box")) is False  # select box #1 (1-based)
    assert review.selected == 0
    assert apply_op(review, "d", ("red_box", "yellow_box")) is False
    assert review.boxes == []
    assert apply_op(review, "space", ("red_box", "yellow_box")) is True


def test_apply_op_draw_adds_a_box():
    review = FrameReview(Path("x.jpg"), Path("x.txt"), 100, 100, [])
    apply_op(review, "draw:10,10,50,50,4", ("outer_box_closed", "red_box", "yellow_box", "lid"))
    assert review.boxes == [Box("lid", 10, 10, 50, 50)]


def _make_manifest_and_labels(tmp_path: Path, n: int, boxes_per_frame: list[Box]):
    frames_dir = tmp_path / "frames" / "t01"
    labels_dir = tmp_path / "labels"
    frames_dir.mkdir(parents=True)
    rows = []
    for i in range(n):
        img_path = frames_dir / f"frame_{i:06d}.jpg"
        cv2.imwrite(str(img_path), np.full((100, 200, 3), 30, np.uint8))
        write_yolo_label(labels_dir / "t01" / f"frame_{i:06d}.txt", boxes_per_frame, 200, 100)
        rows.append(
            {
                "image_path": str(img_path),
                "take": "t01",
                "operator": "op01",
                "scenario": "correct",
                "n": 1,
                "frame_idx": i,
                "t": 0.0,
                "label": "idle",
                "sample": "sparse",
            }
        )
    manifest_path = tmp_path / "frames" / "manifest.csv"
    write_manifest(manifest_path, rows)
    return manifest_path, labels_dir, rows


def test_run_review_headless_edits_and_accepts(tmp_path):
    boxes = [Box("red_box", 20, 10, 80, 40), Box("yellow_box", 5, 5, 15, 15)]
    _manifest_path, labels_dir, rows = _make_manifest_and_labels(tmp_path, 2, boxes)

    # frame 0: delete box 2, accept. frame 1: relabel box 1 to lid, accept.
    ops = [["2", "d", "space"], ["1", "r3", "space"]]
    status = run_review(rows, labels_dir, ops)

    f0 = read_yolo_label(labels_dir / "t01" / "frame_000000.txt", 200, 100)
    assert [b.cls for b in f0] == ["red_box"]
    f1 = read_yolo_label(labels_dir / "t01" / "frame_000001.txt", 200, 100)
    assert [b.cls for b in f1] == ["lid", "yellow_box"]

    assert len(status.rows) == 2
    assert all(r["reviewed"] == "1" for r in status.rows.values())


def test_run_review_accepts_unscripted_frames_as_proposed(tmp_path):
    boxes = [Box("red_box", 20, 10, 80, 40)]
    _manifest_path, labels_dir, rows = _make_manifest_and_labels(tmp_path, 1, boxes)
    status = run_review(rows, labels_dir, [])
    assert len(status.rows) == 1
    unchanged = read_yolo_label(labels_dir / "t01" / "frame_000000.txt", 200, 100)
    assert [b.cls for b in unchanged] == ["red_box"]


def test_load_review_missing_image_raises(tmp_path):
    row = {"image_path": str(tmp_path / "nope.jpg"), "take": "t01", "frame_idx": 0}
    with pytest.raises(FileNotFoundError):
        load_review(row, tmp_path / "labels")


def test_cli_headless_script_writes_status(tmp_path):
    boxes = [Box("red_box", 20, 10, 80, 40)]
    manifest_path, labels_dir, _rows = _make_manifest_and_labels(tmp_path, 2, boxes)
    rc = main(
        [
            "--manifest", str(manifest_path), "--labels-dir", str(labels_dir),
            "--no-preview", "--script", "space;d,space",
        ]
    )  # fmt: skip
    assert rc == 0
    status_path = labels_dir / "review_status.csv"
    assert status_path.exists()
    loaded = ReviewStatus.load(status_path)
    assert len(loaded.rows) == 2


def test_cli_only_unreviewed_skips_already_marked(tmp_path):
    boxes = [Box("red_box", 20, 10, 80, 40)]
    manifest_path, labels_dir, _rows = _make_manifest_and_labels(tmp_path, 2, boxes)
    main(["--manifest", str(manifest_path), "--labels-dir", str(labels_dir), "--no-preview"])
    first = ReviewStatus.load(labels_dir / "review_status.csv").rows
    assert len(first) == 2

    rc = main(
        [
            "--manifest", str(manifest_path), "--labels-dir", str(labels_dir),
            "--no-preview", "--only-unreviewed",
        ]
    )  # fmt: skip
    assert rc == 0  # "nothing to review" path, since both frames are already reviewed
