"""Tests for the pure/deterministic logic in tools/train_detector.py (T-DET-TRAIN).

Training itself (Ultralytics/YOLO) isn't unit-tested here -- it needs real weights and a GPU
and is exercised by actually running the tool (see the PR description for real held-out
numbers). What's tested is everything around it: the color-heuristic proposer, the box-matching
used for the failure gallery, the real+synthetic dataset assembly, the full-frame baseline, and
the model card's formatting (a regression guard for an f-string bug this file once had).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._datalib import Box, write_yolo_label  # noqa: E402
from tools.train_detector import (  # noqa: E402
    ColorBoxProposer,
    HybridProposer,
    RunResult,
    _iou,
    _score_frame,
    build_synth_dataset_yaml,
    full_image_box_baseline,
    write_card,
)

# --------------------------------------------------------------------------------------------
# ColorBoxProposer / HybridProposer
# --------------------------------------------------------------------------------------------


def _synthetic_scene() -> np.ndarray:
    """A grey floor with a red patch and a yellow patch, roughly matching the real footage's
    HSV separation (see the module docstring's calibration notes)."""
    img = np.full((200, 300, 3), (180, 180, 180), np.uint8)  # neutral grey, BGR
    img[40:80, 30:90] = (30, 30, 200)  # red patch (BGR: high R)
    img[120:150, 150:230] = (20, 210, 230)  # yellow patch (BGR: high G+R)
    return img


def test_color_box_proposer_finds_red_and_yellow():
    proposer = ColorBoxProposer()
    boxes = {b.cls: b for b in proposer(_synthetic_scene())}
    assert set(boxes) == {"red_box", "yellow_box"}
    red = boxes["red_box"]
    assert 20 < (red.x2 - red.x1) < 70
    assert 20 < (red.y2 - red.y1) < 50


def test_color_box_proposer_ignores_grey_background():
    grey = np.full((100, 100, 3), (150, 150, 150), np.uint8)
    assert ColorBoxProposer()(grey) == []


def test_color_box_proposer_area_bounds_reject_huge_or_tiny_blobs():
    img = np.full((200, 300, 3), (180, 180, 180), np.uint8)
    img[:, :] = (30, 30, 200)  # the whole frame is "red": above max_area_frac
    assert ColorBoxProposer()(img) == []

    tiny = np.full((200, 300, 3), (180, 180, 180), np.uint8)
    tiny[10:12, 10:12] = (30, 30, 200)  # a couple of pixels: below min_area_frac
    assert ColorBoxProposer()(tiny) == []


def test_hybrid_proposer_combines_world_and_color_boxes():
    def fake_world(image: np.ndarray) -> list[Box]:
        return [Box("lid", 5, 5, 15, 15, 0.9)]

    boxes = HybridProposer(fake_world, ColorBoxProposer())(_synthetic_scene())
    classes = {b.cls for b in boxes}
    assert classes == {"lid", "red_box", "yellow_box"}


# --------------------------------------------------------------------------------------------
# IoU / failure-gallery scoring
# --------------------------------------------------------------------------------------------


def test_iou_identical_boxes_is_one():
    b = Box("red_box", 0, 0, 10, 10)
    assert _iou(b, b) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    a = Box("red_box", 0, 0, 10, 10)
    b = Box("red_box", 20, 20, 30, 30)
    assert _iou(a, b) == 0.0


def test_score_frame_perfect_match_has_no_errors():
    gt = [Box("red_box", 0, 0, 10, 10)]
    pred = [Box("red_box", 0, 0, 10, 10, 0.9)]
    assert _score_frame(gt, pred) == (0, 0)


def test_score_frame_counts_missed_and_spurious():
    gt = [Box("red_box", 0, 0, 10, 10), Box("yellow_box", 50, 50, 60, 60)]
    pred = [Box("red_box", 0, 0, 10, 10, 0.9), Box("lid", 80, 80, 90, 90, 0.5)]
    fn, fp = _score_frame(gt, pred)
    assert fn == 1  # yellow_box never predicted
    assert fp == 1  # lid predicted but not in gt


def test_score_frame_lets_the_most_confident_prediction_match_first():
    gt = [Box("red_box", 0, 0, 10, 10), Box("red_box", 4, 0, 14, 10)]
    weak = Box("red_box", 2, 0, 12, 10, 0.3)  # IoU 0.67 with both; listed first by the model
    strong = Box("red_box", 0, 0, 10, 10, 0.9)  # IoU 1.0 with the first, 0.43 with the second
    # The strong box takes the first GT and the weak one the second: a perfect frame. Matching
    # in list order would give the first GT to the weak box and leave (1 missed, 1 spurious).
    assert _score_frame(gt, [weak, strong]) == (0, 0)


def test_label_path_for_swaps_the_images_directory_on_any_os():
    from tools.train_detector import label_path_for

    img = Path("D:/repo") / "data" / "images" / "op02_correct_01" / "frame_000120.jpg"
    assert label_path_for(img) == (
        Path("D:/repo") / "data" / "labels" / "op02_correct_01" / "frame_000120.txt"
    )


def test_drop_operators_from_splits_removes_excluded_frames(tmp_path):
    from tools.train_detector import drop_operators_from_splits

    (tmp_path / "splits").mkdir()
    (tmp_path / "splits" / "train.txt").write_text(
        "images/op01_correct_01/frame_000000.jpg\nimages/op02_correct_01/frame_000000.jpg\n",
        encoding="utf-8",
    )
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump({"path": str(tmp_path), "train": "splits/train.txt"}), encoding="utf-8"
    )
    assert drop_operators_from_splits(dataset_yaml, ("op01",)) == 1
    lines = (tmp_path / "splits" / "train.txt").read_text(encoding="utf-8").splitlines()
    assert lines == ["images/op02_correct_01/frame_000000.jpg"]


def test_score_frame_wrong_class_at_same_location_is_both_fn_and_fp():
    gt = [Box("red_box", 0, 0, 10, 10)]
    pred = [Box("yellow_box", 0, 0, 10, 10, 0.9)]
    assert _score_frame(gt, pred) == (1, 1)


# --------------------------------------------------------------------------------------------
# Real+synthetic dataset assembly
# --------------------------------------------------------------------------------------------


def test_build_synth_dataset_yaml_concatenates_train_lists(tmp_path):
    (tmp_path / "images").mkdir()
    train_txt = tmp_path / "train.txt"
    train_txt.write_text("images/a.jpg\nimages/b.jpg\n")
    synth_txt = tmp_path / "synthetic.txt"
    synth_txt.write_text("images/synth_a.jpg\nimages/synth_b.jpg\n")

    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "train.txt",
                "val": "val.txt",
                "test": "test.txt",
                "names": {0: "red_box"},
            }
        )
    )

    out_yaml = tmp_path / "dataset_synth.yaml"
    build_synth_dataset_yaml(dataset_yaml, synth_txt, out_yaml)

    combined = yaml.safe_load(out_yaml.read_text())
    assert combined["val"] == "val.txt"  # val/test untouched
    assert combined["test"] == "test.txt"
    combined_train = (tmp_path / combined["train"]).read_text().splitlines()
    assert combined_train == [
        "images/a.jpg",
        "images/b.jpg",
        "images/synth_a.jpg",
        "images/synth_b.jpg",
    ]


# --------------------------------------------------------------------------------------------
# Full-frame baseline
# --------------------------------------------------------------------------------------------


def _write_fake_split(tmp_path: Path, classes=("red_box",)) -> Path:
    images_dir = tmp_path / "images" / "take1"
    labels_dir = tmp_path / "labels" / "take1"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    img_path = images_dir / "frame_000000.jpg"
    cv2.imwrite(str(img_path), np.zeros((100, 200, 3), np.uint8))
    write_yolo_label(
        labels_dir / "frame_000000.txt", [Box("red_box", 50, 25, 100, 75)], 200, 100, classes
    )

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    rel_img = f"images/take1/{img_path.name}"
    (splits_dir / "test.txt").write_text(rel_img + "\n")

    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "splits/train.txt",
                "val": "splits/val.txt",
                "test": "splits/test.txt",
                "names": {0: "red_box"},
            }
        )
    )
    return dataset_yaml


def test_full_image_box_baseline_reports_mean_iou(tmp_path):
    dataset_yaml = _write_fake_split(tmp_path)
    result = full_image_box_baseline(dataset_yaml, classes=("red_box",))
    assert result["n_test_images"] == 1
    assert result["n_gt_boxes"] == 1
    # gt box is 50x50=2500px of a 200x100=20000px frame -> full-frame box's IoU is 0.125
    assert result["mean_iou_of_full_frame_box_vs_gt"] == pytest.approx(0.125)
    assert result["map50"] is None


# --------------------------------------------------------------------------------------------
# Model card formatting (regression guard: this file once had an un-interpolated f-string)
# --------------------------------------------------------------------------------------------


def _fake_run(name: str, map50: float) -> RunResult:
    return RunResult(
        name=name,
        dataset_yaml=Path("data/dataset.yaml"),
        weights_pt=Path("weights/best.pt"),
        map50=map50,
        map50_95=map50 * 0.8,
        per_class={
            "red_box": {"precision": 0.7, "recall": 0.6, "mAP50": map50, "mAP50-95": map50 * 0.8}
        },
        n_train_images=100,
        epochs_run=30,
    )


def test_write_card_interpolates_every_field(tmp_path):
    split_report = {
        "operators": {"op02": "train", "op05": "test"},
        "per_split_frames": {"train": 100, "val": 0, "test": 20},
        "per_split_takes": {"train": 5, "val": 0, "test": 2},
        "leakage_checks": [],
        "duration_warnings": [],
    }
    real = _fake_run("real", 0.5)
    card_path = tmp_path / "card.md"
    write_card(
        card_path,
        split_report=split_report,
        real_result=real,
        synth_result=None,
        chosen=real,
        baseline={"map50": None},
        gallery=[],
        base_weights="yolo11n.pt",
        train_args={"imgsz": 640, "batch": 16, "degrees": 180.0, "fliplr": 0.5, "patience": 15},
    )
    text = card_path.read_text(encoding="utf-8")
    assert "{train_args" not in text  # every f-string field actually interpolated
    assert "fliplr=0.5" in text
    assert "degrees=180.0" in text
    assert "0.5000" in text  # the real run's mAP50


def test_write_card_has_no_numbers_from_an_earlier_run_and_links_relative(tmp_path):
    split_report = {
        "operators": {"op02": "train", "op05": "test"},
        "per_split_frames": {"train": 100, "val": 0, "test": 20},
        "per_split_takes": {"train": 5, "val": 0, "test": 2},
        "leakage_checks": [],
        "duration_warnings": [],
    }
    real = _fake_run("real", 0.5)
    real.per_class["lid"] = {"precision": None, "recall": None, "mAP50": None, "mAP50-95": None}
    out_dir = tmp_path / "detector_v2"  # not models/detector
    (out_dir / "failures").mkdir(parents=True)
    card_path = out_dir / "card.md"
    write_card(
        card_path,
        split_report=split_report,
        real_result=real,
        synth_result=None,
        chosen=real,
        baseline={"map50": None},
        gallery=[
            {
                "image": str(out_dir / "failures" / "01_op05_correct_01_frame_000001.jpg"),
                "source": "images/op05_correct_01/frame_000001.jpg",
                "false_negatives": 1,
                "false_positives": 0,
            }
        ],
        base_weights="yolo11n.pt",
        train_args={"imgsz": 640, "batch": 16, "degrees": 180.0, "fliplr": 0.5, "patience": 15},
        review={"reviewed": 40, "total": 120},
    )
    text = card_path.read_text(encoding="utf-8")
    assert "(failures/01_op05_correct_01_frame_000001.jpg)" in text
    for stale in ("0.221", "0.415", "2559", "0.007", "op01"):
        assert stale not in text
    assert "40 of 120" in text and "`lid`" in text
