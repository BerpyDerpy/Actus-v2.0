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
