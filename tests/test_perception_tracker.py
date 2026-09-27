import pytest

from actus.core.contracts import Detection, DetectionSet
from actus.perception.tracker import TrackRegistry


def _dets(frame_idx: int, items: list[tuple[int | None, str, float]]) -> DetectionSet:
    return DetectionSet(
        frame_idx,
        float(frame_idx),
        [Detection(tid, cls, conf, (0.0, 0.0, 1.0, 1.0)) for tid, cls, conf in items],
        "test",
    )


def test_new_track_has_zero_age_and_zero_missed():
    reg = TrackRegistry()
    tracks = reg.update(_dets(0, [(1, "red_box", 0.9)]))
    assert tracks[1].first_seen_frame == 0
    assert tracks[1].last_seen_frame == 0
    assert tracks[1].missed_frames == 0
    assert tracks[1].age_frames == 1


def test_track_ages_and_updates_across_frames():
    reg = TrackRegistry()
    reg.update(_dets(0, [(1, "red_box", 0.9)]))
    reg.update(_dets(1, [(1, "red_box", 0.95)]))
    tracks = reg.update(_dets(2, [(1, "red_box", 0.99)]))
    t = tracks[1]
    assert t.first_seen_frame == 0
    assert t.last_seen_frame == 2
    assert t.age_frames == 3
    assert t.missed_frames == 0
    assert t.conf == pytest.approx(0.99)


def test_missed_frames_counts_gap_since_last_seen():
    reg = TrackRegistry(max_missed_frames=10)
    reg.update(_dets(0, [(1, "red_box", 0.9)]))
    tracks = reg.update(_dets(4, []))
    assert tracks[1].missed_frames == 4
    assert tracks[1].last_seen_frame == 0


def test_track_is_dropped_after_max_missed_frames():
    reg = TrackRegistry(max_missed_frames=3)
    reg.update(_dets(0, [(1, "red_box", 0.9)]))
    tracks = reg.update(_dets(4, []))  # gap of 4 > max_missed_frames of 3
    assert 1 not in tracks


def test_untracked_detections_are_ignored():
    reg = TrackRegistry()
    tracks = reg.update(_dets(0, [(None, "red_box", 0.9)]))
    assert tracks == {}


def test_multiple_tracks_independent():
    reg = TrackRegistry()
    reg.update(_dets(0, [(1, "red_box", 0.9), (2, "yellow_box", 0.8)]))
    tracks = reg.update(_dets(1, [(1, "red_box", 0.9)]))  # track 2 missing this frame
    assert tracks[1].missed_frames == 0
    assert tracks[2].missed_frames == 1


def test_out_of_order_frames_rejected():
    reg = TrackRegistry()
    reg.update(_dets(5, [(1, "red_box", 0.9)]))
    with pytest.raises(ValueError):
        reg.update(_dets(4, [(1, "red_box", 0.9)]))
    with pytest.raises(ValueError):
        reg.update(_dets(5, [(1, "red_box", 0.9)]))


def test_reset_clears_state():
    reg = TrackRegistry()
    reg.update(_dets(0, [(1, "red_box", 0.9)]))
    reg.reset()
    tracks = reg.update(_dets(0, [(1, "red_box", 0.9)]))
    assert tracks[1].first_seen_frame == 0


def test_negative_max_missed_frames_rejected():
    with pytest.raises(ValueError):
        TrackRegistry(max_missed_frames=-1)
