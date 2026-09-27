import csv
import importlib.util
import json
import sys

import cv2
import pytest

from actus.core.config import REPO_ROOT

_spec = importlib.util.spec_from_file_location(
    "record_takes", REPO_ROOT / "tools" / "record_takes.py"
)
rt = importlib.util.module_from_spec(_spec)
sys.modules["record_takes"] = rt  # dataclasses look the module up while the class is built
_spec.loader.exec_module(rt)


def _rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _decoded_frames(path):
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def test_take_naming_roundtrip(tmp_path):
    stem = rt.take_stem("op01", "skip_S2", 3)
    assert stem == "op01_skip_S2_03"
    assert rt.parse_take_stem(stem) == {"operator": "op01", "scenario": "skip_S2", "n": 3, "cam": 1}
    assert rt.parse_take_stem(stem + "_cam2")["cam"] == 2
    with pytest.raises(ValueError):
        rt.take_stem("op_01", "correct", 1)  # "_" is the field separator
    (tmp_path / "op01_correct_01.mp4").touch()
    (tmp_path / "op01_correct_04.mp4").touch()
    (tmp_path / "op01_correct_extra_02.mp4").touch()  # another scenario, same prefix
    assert rt.next_take_number(tmp_path, "op01", "correct") == 5
    assert rt.next_take_number(tmp_path, "op02", "correct") == 1


def test_key_map():
    assert rt.key_to_label(ord("1")) == "S1"
    assert rt.key_to_label(ord("6")) == "S6"
    assert rt.key_to_label(ord("0")) == "idle"
    assert rt.key_to_label(ord("x")) == "other"
    assert rt.key_to_label(ord("q")) is None
    assert rt.key_to_label(-1) is None


def test_label_track_undo_and_collapse(tmp_path):
    tr = rt.LabelTrack()
    tr.set(10, 0.5, "S1")
    tr.set(12, 0.6, "S1")  # repeated press: collapsed on save
    tr.set(30, 1.5, "other")
    assert tr.undo() == 30
    assert tr.label_at(40) == "S1"
    assert [r[2] for r in tr.items()] == ["idle", "S1"]
    tr.save(tmp_path / "l.csv")
    back = rt.LabelTrack.load(tmp_path / "l.csv")
    assert back.items() == [(0, 0.0, "idle"), (10, 0.5, "S1")]


def test_record_replay_source_headless(tmp_path):
    rt.main([
        "--operator", "op00", "--scenario", "selftest", "--source", "replay:correct",
        "--fps", "15", "--duration", "3", "--no-preview", "--out", str(tmp_path),
        "--script", "0.5:1,1.5:0,2.0:x,2.2:u",
    ])  # fmt: skip
    stem = tmp_path / "op00_selftest_01"
    meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    frames = _rows(tmp_path / "op00_selftest_01.frames.csv")
    labels = _rows(stem.with_suffix(".csv"))

    n = meta["frames"]
    assert 40 <= n <= 46  # 3 s at 15 fps
    assert [int(r["frame_idx"]) for r in frames] == list(range(n))
    assert _decoded_frames(stem.with_suffix(".mp4")) == n
    cam = meta["cameras"][0]
    assert cam["frames_written"] == n and cam["frame_count_ok"]
    assert meta["simulated"] is True and meta["operator"] == "op00"
    assert {"camera_roll_deg", "lighting", "notes"} <= set(meta)

    assert [r["label"] for r in labels] == ["idle", "S1", "idle"]  # "other" was undone
    assert all(0 <= int(r["frame_idx"]) < n for r in labels)
    ts = [float(r["t"]) for r in frames]
    assert ts == sorted(ts)


def test_toggle_session_records_one_take_per_start_stop(tmp_path):
    # standby 0-0.5 s, take 1 0.5-1.5 s, standby, take 2 2.0-3.0 s, then quit from standby.
    rt.main([
        "--operator", "op00", "--scenario", "selftest", "--source", "replay:correct",
        "--fps", "15", "--no-preview", "--out", str(tmp_path), "--toggle",
        "--script", "0.5:r,0.9:1,1.5:r,2.0:r,2.4:2,3.0:r,3.3:q",
    ])  # fmt: skip
    assert sorted(p.name for p in tmp_path.glob("*.mp4")) == [
        "op00_selftest_01.mp4",
        "op00_selftest_02.mp4",
    ]
    for n, step in ((1, "S1"), (2, "S2")):
        stem = tmp_path / f"op00_selftest_{n:02d}"
        meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["stop_reason"] == "toggle"
        assert 10 <= meta["frames"] <= 20  # ~1 s at 15 fps; standby time is not recorded
        assert meta["cameras"][0]["frame_count_ok"]
        assert _decoded_frames(stem.with_suffix(".mp4")) == meta["frames"]
        frames = _rows(tmp_path / f"op00_selftest_{n:02d}.frames.csv")
        assert [int(r["frame_idx"]) for r in frames] == list(range(meta["frames"]))
        assert [r["label"] for r in _rows(stem.with_suffix(".csv"))] == ["idle", step]


def test_toggle_session_quit_without_recording(tmp_path):
    rt.main([
        "--operator", "op00", "--scenario", "selftest", "--source", "replay:correct",
        "--fps", "15", "--no-preview", "--out", str(tmp_path), "--toggle", "--script", "0.3:q",
    ])  # fmt: skip
    assert list(tmp_path.glob("*.mp4")) == []


def test_label_pending_labels_only_unlabelled_takes(tmp_path):
    # take 01: no keys pressed while recording (pending); take 02: labelled live (not pending);
    # take 03 has a second camera whose file must not be offered separately.
    rt.main([
        "--operator", "op00", "--scenario", "selftest", "--source", "replay:correct",
        "--fps", "15", "--no-preview", "--out", str(tmp_path), "--toggle",
        "--script", "0.3:r,1.3:r,1.6:r,2.0:1,2.6:r,2.9:r,3.9:r,4.2:q",
        "--source2", "replay:correct",
    ])  # fmt: skip
    assert [p.name for p in rt.pending_takes(tmp_path)] == [
        "op00_selftest_01.mp4",
        "op00_selftest_03.mp4",
    ]
    rt.main(["--label-pending", str(tmp_path), "--no-preview", "--script", "0.4:3"])
    for n in (1, 3):
        stem = tmp_path / f"op00_selftest_{n:02d}"
        assert [r["label"] for r in _rows(stem.with_suffix(".csv"))] == ["idle", "S3"]
        assert "labelled_at" in json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    labels_02 = _rows(tmp_path / "op00_selftest_02.csv")
    assert [r["label"] for r in labels_02] == ["idle", "S1"]  # untouched
    assert rt.pending_takes(tmp_path) == []


def test_playback_labelling(tmp_path):
    # make a short file first, then label it in playback mode (no window)
    rt.main([
        "--operator", "op00", "--scenario", "pb", "--source", "replay:correct",
        "--fps", "15", "--duration", "2", "--no-preview", "--out", str(tmp_path),
    ])  # fmt: skip
    video = tmp_path / "op00_pb_01.mp4"
    rt.main(["--label", str(video), "--no-preview", "--force", "--script", "0.4:2,1.0:0"])
    labels = _rows(video.with_suffix(".csv"))
    assert [r["label"] for r in labels] == ["idle", "S2", "idle"]
    assert int(labels[1]["frame_idx"]) == 6  # first frame at t >= 0.4 s at 15 fps
    meta = json.loads(video.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["labelled_with"]["tool"] == "record_takes --label"
