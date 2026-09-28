import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._datalib import CLASSES, Box, read_yolo_label, write_manifest  # noqa: E402
from tools.autolabel import autolabel_manifest, main  # noqa: E402


def _make_frames(tmp_path: Path, n: int) -> list[dict]:
    frames_dir = tmp_path / "frames" / "op01_correct_01"
    frames_dir.mkdir(parents=True)
    rows = []
    for i in range(n):
        img_path = frames_dir / f"frame_{i:06d}.jpg"
        cv2.imwrite(str(img_path), np.full((100, 200, 3), 30, np.uint8))
        rows.append(
            {
                "image_path": str(img_path),
                "take": "op01_correct_01",
                "operator": "op01",
                "scenario": "correct",
                "n": 1,
                "frame_idx": i,
                "t": round(i / 15, 4),
                "label": "idle",
                "sample": "sparse",
            }
        )
    manifest_path = tmp_path / "frames" / "manifest.csv"
    write_manifest(manifest_path, rows)
    return rows


def _fake_propose(image: np.ndarray) -> list[Box]:
    h, w = image.shape[:2]
    return [Box("red_box", w * 0.1, h * 0.1, w * 0.4, h * 0.4, 0.7)]


def test_autolabel_manifest_writes_yolo_labels_and_report(tmp_path):
    rows = _make_frames(tmp_path, 3)
    out_dir = tmp_path / "labels"
    report = autolabel_manifest(rows, _fake_propose, out_dir)

    assert (out_dir / "classes.txt").read_text(encoding="utf-8").splitlines() == list(CLASSES)
    assert report["n_frames"] == 3
    assert report["n_frames_with_no_proposals"] == 0
    assert report["per_class_count"]["red_box"] == 3
    assert report["per_class_count"]["yellow_box"] == 0
    assert report["per_class_mean_conf"]["red_box"] == pytest.approx(0.7)
    assert report["per_class_mean_conf"]["yellow_box"] is None

    label_path = out_dir / "op01_correct_01" / "frame_000000.txt"
    boxes = read_yolo_label(label_path, 200, 100)
    assert len(boxes) == 1
    b = boxes[0]
    assert b.cls == "red_box"
    assert b.x1 == pytest.approx(20, abs=1) and b.y1 == pytest.approx(10, abs=1)
    assert b.x2 == pytest.approx(80, abs=1) and b.y2 == pytest.approx(40, abs=1)


def test_autolabel_manifest_min_conf_filters_proposals(tmp_path):
    rows = _make_frames(tmp_path, 1)

    def low_conf_propose(image):
        return [Box("red_box", 1, 1, 5, 5, 0.05)]

    out_dir = tmp_path / "labels"
    report = autolabel_manifest(rows, low_conf_propose, out_dir, min_conf=0.1)
    assert report["n_frames_with_no_proposals"] == 1
    assert report["per_class_count"]["red_box"] == 0
    label_path = out_dir / "op01_correct_01" / "frame_000000.txt"
    assert label_path.read_text(encoding="utf-8") == ""


def test_autolabel_manifest_never_overwrites_reviewed_frames(tmp_path):
    rows = _make_frames(tmp_path, 2)
    out_dir = tmp_path / "labels"
    reviewed = rows[0]
    label_path = out_dir / reviewed["take"] / f"frame_{int(reviewed['frame_idx']):06d}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("4 0.5 0.5 0.1 0.1\n", encoding="utf-8")  # a human's correction

    report = autolabel_manifest(rows, _fake_propose, out_dir, skip={reviewed["image_path"]})
    assert label_path.read_text(encoding="utf-8") == "4 0.5 0.5 0.1 0.1\n"
    assert report["n_frames_skipped_reviewed"] == 1


def test_autolabel_manifest_missing_image_raises(tmp_path):
    rows = _make_frames(tmp_path, 1)
    rows[0]["image_path"] = str(tmp_path / "nope.jpg")
    with pytest.raises(FileNotFoundError):
        autolabel_manifest(rows, _fake_propose, tmp_path / "labels")


def test_cli_reports_error_on_empty_manifest(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, [])
    assert main(["--manifest", str(manifest_path), "--out-dir", str(tmp_path / "labels")]) == 1


def test_yolo_world_proposer_needs_perception_extra():
    pytest.importorskip("ultralytics")
    from tools.autolabel import YoloWorldProposer

    # Only asserts the class is importable and constructible when the extra is installed; a real
    # forward pass needs network access to fetch weights, which CI does not have.
    assert YoloWorldProposer is not None
