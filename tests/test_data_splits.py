import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._datalib import write_manifest  # noqa: E402
from tools.make_splits import (  # noqa: E402
    assign_operators,
    build_report,
    check_leakage,
    duration_predicts_scenario,
    main,
)
from tools.record_takes import main as record_main  # noqa: E402


def _record(raw_dir, operator, scenario, *, duration=2.0, fps=15.0):
    record_main(
        [
            "--operator", operator, "--scenario", scenario,
            "--source", f"replay:{scenario}", "--fps", str(fps), "--duration", str(duration),
            "--no-preview", "--out", str(raw_dir),
        ]
    )  # fmt: skip


def _manifest_rows(operator: str, scenario: str, take: str, n: int) -> list[dict]:
    return [
        {
            "image_path": f"data/frames/{take}/frame_{i:06d}.jpg",
            "take": take,
            "operator": operator,
            "scenario": scenario,
            "n": 1,
            "frame_idx": i,
            "t": round(i / 15, 4),
            "label": "idle",
            "sample": "sparse",
        }
        for i in range(n)
    ]


def test_assign_operators_auto_holds_out_last_with_three_or_more():
    split_of, warnings = assign_operators(["op01", "op02", "op03"], test=None, val=None)
    assert split_of == {"op01": "train", "op02": "train", "op03": "test"}
    assert warnings  # the auto choice is flagged


def test_assign_operators_too_few_operators_warns_and_keeps_all_train():
    split_of, warnings = assign_operators(["op01", "op02"], test=None, val=None)
    assert split_of == {"op01": "train", "op02": "train"}
    assert any("only 2 operator" in w for w in warnings)


def test_assign_operators_explicit_test_and_val():
    split_of, _warnings = assign_operators(
        ["op01", "op02", "op03", "op04"], test=["op04"], val=["op03"]
    )
    assert split_of == {"op01": "train", "op02": "train", "op03": "val", "op04": "test"}


def test_assign_operators_rejects_unknown_or_overlapping():
    with pytest.raises(ValueError, match="unknown"):
        assign_operators(["op01"], test=["op99"], val=None)
    with pytest.raises(ValueError, match="both val and test"):
        assign_operators(["op01", "op02"], test=["op01"], val=["op01"])


def test_check_leakage_flags_a_take_split_across_operators():
    rows = [
        {"take": "t1", "operator": "op01"},
        {"take": "t1", "operator": "op02"},  # same take, two different operators: a bad manifest
    ]
    split_of = {"op01": "train", "op02": "test"}
    problems = check_leakage(rows, split_of)
    assert len(problems) == 1 and "t1" in problems[0]


def test_check_leakage_clean_manifest_has_no_problems():
    rows = _manifest_rows("op01", "correct", "op01_correct_01", 5)
    split_of = {"op01": "train"}
    assert check_leakage(rows, split_of) == []


def test_duration_predicts_scenario_flags_non_overlapping_ranges():
    warnings = duration_predicts_scenario({"correct": [20.0, 22.0], "skip_S2": [10.0, 11.0]})
    assert len(warnings) == 1
    assert "correct" in warnings[0] and "skip_S2" in warnings[0]


def test_duration_predicts_scenario_silent_when_ranges_overlap():
    assert duration_predicts_scenario({"correct": [10.0, 22.0], "skip_S2": [15.0, 30.0]}) == []


def test_build_report_counts_and_leakage():
    rows = _manifest_rows("op01", "correct", "op01_correct_01", 4) + _manifest_rows(
        "op02", "correct", "op02_correct_01", 6
    )
    split_of = {"op01": "train", "op02": "test"}
    durations = {"op01_correct_01": 20.0, "op02_correct_01": 21.0}
    report = build_report(rows, split_of, durations)
    assert report["per_split_frames"] == {"train": 4, "test": 6}
    assert report["per_split_takes"] == {"train": 1, "test": 1}
    assert report["leakage_checks"] == []


def test_cli_end_to_end_writes_splits_report_and_dataset_yaml(tmp_path):
    raw_dir = tmp_path / "raw"
    _record(raw_dir, "op01", "correct")
    _record(raw_dir, "op02", "correct")
    _record(raw_dir, "op03", "skip_S2")

    rows = (
        _manifest_rows("op01", "correct", "op01_correct_01", 5)
        + _manifest_rows("op02", "correct", "op02_correct_01", 5)
        + _manifest_rows("op03", "skip_S2", "op03_skip_S2_01", 5)
    )
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, rows)

    out_dir = tmp_path / "splits"
    dataset_yaml = tmp_path / "dataset.yaml"
    rc = main(
        [
            "--manifest", str(manifest_path), "--raw-dir", str(raw_dir),
            "--labels-dir", str(tmp_path / "labels"),
            "--out-dir", str(out_dir), "--dataset-yaml", str(dataset_yaml),
            "--test-operators", "op03",
        ]
    )  # fmt: skip
    assert rc == 0

    train = (out_dir / "train.txt").read_text(encoding="utf-8").splitlines()
    test = (out_dir / "test.txt").read_text(encoding="utf-8").splitlines()
    assert len(train) == 10 and len(test) == 5
    assert set(train) & set(test) == set()

    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report["leakage_checks"] == []
    assert report["operators"] == {"op01": "train", "op02": "train", "op03": "test"}
    assert "## Leakage check: passed" in (out_dir / "report.md").read_text(encoding="utf-8")

    dataset = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    assert dataset["names"] == {
        0: "outer_box_closed",
        1: "outer_box_open",
        2: "lid",
        3: "red_box",
        4: "yellow_box",
    }
    assert Path(dataset["path"]) == REPO_ROOT


def test_cli_empty_manifest_returns_error(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, [])
    rc = main(["--manifest", str(manifest_path), "--raw-dir", str(tmp_path / "raw")])
    assert rc == 1
