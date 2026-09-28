import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._datalib import Box, read_yolo_label, write_manifest, write_yolo_label  # noqa: E402
from tools.synth_augment import (  # noqa: E402
    Crop,
    _alpha_bbox,
    _feathered_alpha,
    _rotate_rgba,
    compose_image,
    extract_crops,
    generate,
    main,
)


def test_feathered_alpha_is_one_in_the_middle_and_ramps_at_the_edges():
    alpha = _feathered_alpha(20, 20, feather=4)
    assert alpha[10, 10] == pytest.approx(1.0)
    assert alpha[0, 10] == pytest.approx(0.0)  # the very edge is fully transparent
    assert 0.0 < alpha[1, 10] < 1.0  # ramps up moving inward from the edge
    assert alpha[1, 1] < alpha[1, 10]  # a corner is feathered on both axes, so darker


def test_rotate_rgba_preserves_total_alpha_mass_at_0_degrees():
    bgr = np.full((10, 20, 3), 200, np.uint8)
    alpha = np.ones((10, 20), np.float32)
    rot_bgr, rot_alpha = _rotate_rgba(bgr, alpha, 0.0)
    assert rot_alpha.sum() == pytest.approx(alpha.sum(), rel=0.05)
    assert rot_bgr.shape[:2] == rot_alpha.shape


def test_alpha_bbox_finds_the_nonzero_region():
    alpha = np.zeros((50, 50), np.float32)
    alpha[10:20, 5:15] = 1.0
    box = _alpha_bbox(alpha)
    assert box == (5, 10, 15, 20)


def test_alpha_bbox_none_when_empty():
    assert _alpha_bbox(np.zeros((10, 10), np.float32)) is None


def _make_crop(cls: str, w: int = 30, h: int = 20) -> Crop:
    bgr = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    return Crop(cls, bgr, _feathered_alpha(h, w))


def test_compose_image_pastes_boxes_within_canvas():
    canvas = np.full((200, 300, 3), 50, np.uint8)
    crops = [_make_crop("red_box"), _make_crop("yellow_box")]
    rng = np.random.default_rng(1)
    out, boxes = compose_image(canvas, crops, rng)
    assert out.shape == canvas.shape
    assert {b.cls for b in boxes} == {"red_box", "yellow_box"}
    for b in boxes:
        assert 0 <= b.x1 < b.x2 <= 300
        assert 0 <= b.y1 < b.y2 <= 200


def test_extract_crops_reads_reviewed_boxes(tmp_path):
    frames_dir = tmp_path / "frames" / "t01"
    labels_dir = tmp_path / "labels"
    frames_dir.mkdir(parents=True)
    img_path = frames_dir / "frame_000000.jpg"
    cv2.imwrite(str(img_path), np.full((100, 200, 3), 10, np.uint8))
    write_yolo_label(
        labels_dir / "t01" / "frame_000000.txt", [Box("red_box", 20, 10, 80, 40)], 200, 100
    )
    rows = [{"image_path": str(img_path), "take": "t01", "frame_idx": 0}]

    crops = extract_crops(rows, labels_dir)
    assert len(crops) == 1
    assert crops[0].cls == "red_box"
    assert crops[0].bgr.shape[0] > 0 and crops[0].bgr.shape[1] > 0


def test_extract_crops_skips_frames_with_no_label_file(tmp_path):
    frames_dir = tmp_path / "frames" / "t01"
    frames_dir.mkdir(parents=True)
    img_path = frames_dir / "frame_000000.jpg"
    cv2.imwrite(str(img_path), np.full((50, 50, 3), 10, np.uint8))
    rows = [{"image_path": str(img_path), "take": "t01", "frame_idx": 0}]
    assert extract_crops(rows, tmp_path / "labels") == []


def test_generate_writes_images_labels_and_report(tmp_path):
    crops = [_make_crop("red_box"), _make_crop("yellow_box")]
    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()
    bg_path = bg_dir / "bg.jpg"
    cv2.imwrite(str(bg_path), np.full((480, 640, 3), 40, np.uint8))

    out_dir = tmp_path / "synthetic"
    report = generate(crops, [bg_path], out_dir, n_images=4, objects_per_image=(1, 2), seed=0)
    assert report["n_images"] == 4
    images = sorted((out_dir / "images").glob("*.jpg"))
    labels = sorted((out_dir / "labels").glob("*.txt"))
    assert len(images) == 4 and len(labels) == 4
    for lbl in labels:
        boxes = read_yolo_label(lbl, 640, 480)
        assert all(b.cls in ("red_box", "yellow_box") for b in boxes)


def test_generate_is_deterministic_for_a_given_seed(tmp_path):
    crops = [_make_crop("red_box")]
    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()
    bg_path = bg_dir / "bg.jpg"
    cv2.imwrite(str(bg_path), np.full((480, 640, 3), 40, np.uint8))

    r1 = generate(crops, [bg_path], tmp_path / "a", n_images=2, objects_per_image=(1, 1), seed=7)
    r2 = generate(crops, [bg_path], tmp_path / "b", n_images=2, objects_per_image=(1, 1), seed=7)
    img_a = cv2.imread(str(tmp_path / "a" / "images" / "synth_000000.jpg"))
    img_b = cv2.imread(str(tmp_path / "b" / "images" / "synth_000000.jpg"))
    assert np.array_equal(img_a, img_b)
    assert r1["per_class_count"] == r2["per_class_count"]


def _setup_pipeline_dir(tmp_path: Path) -> tuple[Path, Path, Path]:
    frames_dir = tmp_path / "frames" / "t01"
    labels_dir = tmp_path / "labels"
    frames_dir.mkdir(parents=True)
    img_path = frames_dir / "frame_000000.jpg"
    cv2.imwrite(str(img_path), np.full((100, 200, 3), 10, np.uint8))
    write_yolo_label(
        labels_dir / "t01" / "frame_000000.txt", [Box("red_box", 20, 10, 80, 40)], 200, 100
    )
    manifest_path = tmp_path / "frames" / "manifest.csv"
    write_manifest(
        manifest_path,
        [
            {
                "image_path": str(img_path),
                "take": "t01",
                "operator": "op01",
                "scenario": "correct",
                "n": 1,
                "frame_idx": 0,
                "t": 0.0,
                "label": "idle",
                "sample": "sparse",
            }
        ],
    )
    return manifest_path, labels_dir, tmp_path


def test_cli_end_to_end(tmp_path):
    manifest_path, labels_dir, root = _setup_pipeline_dir(tmp_path)
    bg_dir = root / "backgrounds"
    bg_dir.mkdir()
    cv2.imwrite(str(bg_dir / "bg.jpg"), np.full((480, 640, 3), 40, np.uint8))

    out_dir = root / "synthetic"
    splits_dir = root / "splits"
    _mark_reviewed_and_split(manifest_path, labels_dir, splits_dir, {"op01": "train"})
    rc = main(
        [
            "--manifest", str(manifest_path), "--labels-dir", str(labels_dir),
            "--backgrounds-dir", str(bg_dir), "--out-dir", str(out_dir),
            "--splits-dir", str(splits_dir), "--n-images", "3", "--seed", "0",
        ]
    )  # fmt: skip
    assert rc == 0
    assert len(list((out_dir / "images").glob("*.jpg"))) == 3
    synth_list = (splits_dir / "synthetic.txt").read_text(encoding="utf-8").splitlines()
    assert len(synth_list) == 3
    for line in synth_list:
        assert Path(line).exists()

    # A second, smaller run leaves no stale images behind and lists only its own.
    rc = main(
        [
            "--manifest", str(manifest_path), "--labels-dir", str(labels_dir),
            "--backgrounds-dir", str(bg_dir), "--out-dir", str(out_dir),
            "--splits-dir", str(splits_dir), "--n-images", "1", "--seed", "0",
        ]
    )  # fmt: skip
    assert rc == 0
    assert len(list((out_dir / "images").glob("*.jpg"))) == 1
    assert len((splits_dir / "synthetic.txt").read_text(encoding="utf-8").splitlines()) == 1


def _mark_reviewed_and_split(manifest_path, labels_dir, splits_dir, operators, *, reviewed=True):
    import csv
    import json

    from tools._datalib import read_manifest

    labels_dir.mkdir(parents=True, exist_ok=True)
    with open(labels_dir / "review_status.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "take", "frame_idx", "reviewed", "n_boxes", "reviewed_at"])
        for r in read_manifest(manifest_path):
            flag = "1" if reviewed else "0"
            w.writerow([r["image_path"], r["take"], r["frame_idx"], flag, 1, ""])
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "report.json").write_text(json.dumps({"operators": operators}), encoding="utf-8")


@pytest.mark.parametrize(
    ("operators", "reviewed"),
    [({"op01": "test"}, True), ({"op01": "train"}, False)],
    ids=["held-out-operator", "unreviewed"],
)
def test_cli_never_crops_held_out_or_unreviewed_frames(tmp_path, operators, reviewed):
    manifest_path, labels_dir, root = _setup_pipeline_dir(tmp_path)
    bg_dir = root / "backgrounds"
    bg_dir.mkdir()
    cv2.imwrite(str(bg_dir / "bg.jpg"), np.full((480, 640, 3), 40, np.uint8))
    splits_dir = root / "splits"
    _mark_reviewed_and_split(manifest_path, labels_dir, splits_dir, operators, reviewed=reviewed)
    rc = main(
        [
            "--manifest", str(manifest_path), "--labels-dir", str(labels_dir),
            "--backgrounds-dir", str(bg_dir), "--out-dir", str(root / "synthetic"),
            "--splits-dir", str(splits_dir), "--n-images", "1",
        ]
    )  # fmt: skip
    assert rc == 1  # the only frame is excluded, so there is nothing to crop


def test_cli_errors_without_backgrounds(tmp_path):
    manifest_path, labels_dir, root = _setup_pipeline_dir(tmp_path)
    empty_bg = root / "empty_backgrounds"
    empty_bg.mkdir()
    rc = main(
        [
            "--manifest", str(manifest_path), "--labels-dir", str(labels_dir),
            "--backgrounds-dir", str(empty_bg), "--out-dir", str(root / "synthetic"),
        ]
    )  # fmt: skip
    assert rc == 1


def test_cli_errors_without_reviewed_boxes(tmp_path):
    frames_dir = tmp_path / "frames" / "t01"
    frames_dir.mkdir(parents=True)
    img_path = frames_dir / "frame_000000.jpg"
    cv2.imwrite(str(img_path), np.full((50, 50, 3), 10, np.uint8))
    manifest_path = tmp_path / "frames" / "manifest.csv"
    write_manifest(
        manifest_path,
        [
            {
                "image_path": str(img_path), "take": "t01", "operator": "op01",
                "scenario": "correct", "n": 1, "frame_idx": 0, "t": 0.0,
                "label": "idle", "sample": "sparse",
            }
        ],
    )  # fmt: skip
    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()
    cv2.imwrite(str(bg_dir / "bg.jpg"), np.full((480, 640, 3), 40, np.uint8))
    rc = main(
        [
            "--manifest", str(manifest_path), "--labels-dir", str(tmp_path / "labels"),
            "--backgrounds-dir", str(bg_dir), "--out-dir", str(tmp_path / "synthetic"),
        ]
    )  # fmt: skip
    assert rc == 1
