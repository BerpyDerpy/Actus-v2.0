import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._datalib import discover_takes, read_manifest  # noqa: E402
from tools.extract_frames import _thin_to_budget, main, sample_plan  # noqa: E402
from tools.record_takes import main as record_main  # noqa: E402


def _record(raw_dir, operator, scenario, *, duration=2.0, fps=15.0, script=""):
    args = [
        "--operator", operator, "--scenario", scenario,
        "--source", f"replay:{scenario}", "--fps", str(fps), "--duration", str(duration),
        "--no-preview", "--out", str(raw_dir),
    ]  # fmt: skip
    if script:
        args += ["--script", script]
    record_main(args)


def test_sample_plan_dense_around_boundaries_sparse_elsewhere():
    plan = sample_plan([50], 100, fps=10, dense_radius_s=1.0, dense_stride=2, sparse_stride=20)
    assert plan[50] == "dense"
    assert all(plan[fi] == "dense" for fi in range(40, 61, 2))
    assert plan[0] == "sparse"
    assert 60 in plan and 61 not in plan  # radius is exactly 10 frames either side of 50


def test_sample_plan_dense_wins_on_overlap():
    plan = sample_plan([10], 20, fps=10, dense_radius_s=0.5, dense_stride=1, sparse_stride=1)
    # every frame is both a dense candidate (near 10) and a sparse candidate (stride 1)
    for fi in range(5, 16):
        assert plan[fi] == "dense"


def test_thin_to_budget_keeps_all_dense_first():
    rows = [{"sample": "dense"} for _ in range(5)] + [{"sample": "sparse"} for _ in range(20)]
    thinned = _thin_to_budget(rows, 10)
    assert sum(1 for r in thinned if r["sample"] == "dense") == 5
    assert len(thinned) == 10


def test_thin_to_budget_thins_dense_when_it_alone_exceeds_budget():
    rows = [{"sample": "dense"} for _ in range(20)]
    assert len(_thin_to_budget(rows, 5)) == 5


def test_thin_to_budget_spreads_sparse_frames_over_every_take():
    rows = [{"sample": "sparse", "take": f"t{i // 190}"} for i in range(1900)]  # 10 takes
    thinned = _thin_to_budget(rows, 1000)
    assert len(thinned) == 1000
    per_take = {}
    for r in thinned:
        per_take[r["take"]] = per_take.get(r["take"], 0) + 1
    assert set(per_take) == {f"t{i}" for i in range(10)}  # the last takes are not starved
    assert max(per_take.values()) - min(per_take.values()) <= 1


def test_write_yolo_label_clips_boxes_to_the_image(tmp_path):
    from tools._datalib import Box, read_yolo_label, write_yolo_label

    path = tmp_path / "l.txt"
    write_yolo_label(
        path, [Box("red_box", 600, -10, 700, 50), Box("lid", 700, 0, 800, 10)], 640, 480
    )
    (box,) = read_yolo_label(path, 640, 480)  # the fully-outside box is dropped
    assert (round(box.x1), round(box.y1), round(box.x2), round(box.y2)) == (600, 0, 640, 50)
    for tok in path.read_text(encoding="utf-8").split()[1:]:
        assert 0.0 <= float(tok) <= 1.0


def test_thin_to_budget_noop_under_budget():
    rows = [{"sample": "sparse"} for _ in range(3)]
    assert _thin_to_budget(rows, 10) == rows


def test_dense_frames_cluster_around_label_boundaries(tmp_path):
    raw_dir = tmp_path / "raw"
    _record(raw_dir, "op01", "correct", duration=3.0, script="1.0:1,2.0:0")
    take = discover_takes(raw_dir)[0]
    boundaries = [fi for fi, _, _ in take.label_track().items()]
    assert len(boundaries) >= 2

    out_dir = tmp_path / "frames"
    rc = main(
        [
            "--raw-dir", str(raw_dir), "--out-dir", str(out_dir),
            "--dense-radius-s", "0.2", "--dense-stride", "1", "--sparse-stride", "1000",
            "--max-frames", "0",
        ]
    )  # fmt: skip
    assert rc == 0
    rows = read_manifest(out_dir / "manifest.csv")
    dense_frames = {int(r["frame_idx"]) for r in rows if r["sample"] == "dense"}
    radius = round(0.2 * 15)
    for b in boundaries:
        assert any(abs(fi - b) <= radius for fi in dense_frames)


def test_manifest_rows_match_written_images_and_labels(tmp_path):
    raw_dir = tmp_path / "raw"
    _record(raw_dir, "op01", "correct", duration=2.0, script="0.5:1,1.2:0")
    take = discover_takes(raw_dir)[0]
    track = take.label_track()

    out_dir = tmp_path / "frames"
    main(
        [
            "--raw-dir", str(raw_dir), "--out-dir", str(out_dir),
            "--dense-radius-s", "0.3", "--dense-stride", "2", "--sparse-stride", "5",
            "--max-frames", "0",
        ]
    )  # fmt: skip
    rows = read_manifest(out_dir / "manifest.csv")
    assert rows
    for r in rows:
        assert Path(r["image_path"]).exists()
        assert r["label"] == track.label_at(int(r["frame_idx"]))
        assert r["take"] == take.stem
        assert r["operator"] == "op01"


def test_max_frames_caps_total_across_takes(tmp_path):
    raw_dir = tmp_path / "raw"
    _record(raw_dir, "op01", "correct", duration=2.0)
    _record(raw_dir, "op02", "correct", duration=2.0)
    out_dir = tmp_path / "frames"
    main(
        [
            "--raw-dir", str(raw_dir), "--out-dir", str(out_dir),
            "--sparse-stride", "1", "--dense-radius-s", "0", "--max-frames", "15",
        ]
    )  # fmt: skip
    rows = read_manifest(out_dir / "manifest.csv")
    assert len(rows) == 15


def test_no_takes_found_returns_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["--raw-dir", str(empty), "--out-dir", str(tmp_path / "out")]) == 1
