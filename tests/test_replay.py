import csv

import numpy as np
import pytest

from actus.capture.replay import REQUIRED_SCENARIOS, SCENARIOS, ReplaySource, main
from actus.core.bus import Bus
from actus.core.contracts import Topic, to_jsonable

P_ENTER, DWELL_S, ASK_LO, ASK_HI = 0.8, 0.8, 0.45, 0.8


def _series(src: ReplaySource):
    ticks = list(src)
    ev = {
        s: np.array([t.evidence[i].p_complete for t in ticks])
        for i, s in enumerate(src.protocol.step_ids)
    }
    vis = np.array([t.predicates.visibility for t in ticks])
    return ticks, ev, vis


def _first_confirm(p: np.ndarray, fps: float, thr=P_ENTER, dwell=DWELL_S) -> int | None:
    """First frame at which p has stayed >= thr for `dwell` seconds (what the engine would do)."""
    need, run = round(dwell * fps), 0
    for i, v in enumerate(p):
        run = run + 1 if v >= thr else 0
        if run >= need:
            return i
    return None


def test_required_scenarios_exist():
    assert set(REQUIRED_SCENARIOS) <= set(SCENARIOS)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_stream_is_well_formed(name):
    src = ReplaySource(name)
    prev_t = -1.0
    for i, tick in enumerate(src):
        assert tick.frame_idx == i == tick.predicates.frame_idx == tick.actions.frame_idx
        assert tick.t > prev_t
        prev_t = tick.t
        assert [e.step_id for e in tick.evidence] == src.protocol.step_ids
        assert set(tick.predicates.values) >= set(src.protocol.predicates)
        assert abs(sum(tick.actions.scores.values()) - 1) < 1e-6
    assert i + 1 == len(src)


def test_deterministic_per_seed():
    def dump(seed):
        return [to_jsonable(t.predicates) for t in ReplaySource("correct", seed=seed)]

    assert dump(3) == dump(3)
    assert dump(3) != dump(4)


def test_correct_confirms_every_step_in_order():
    src = ReplaySource("correct")
    _, ev, _ = _series(src)
    frames = [_first_confirm(ev[s], src.fps) for s in src.protocol.step_ids]
    assert None not in frames
    assert frames == sorted(frames)


def test_evidence_stays_low_before_transition():
    # S6's post-state (box closed) holds at the start, but the transition has not happened yet.
    src = ReplaySource("correct")
    _, ev, _ = _series(src)
    s1 = _first_confirm(ev["S1"], src.fps)
    assert ev["S6"][:s1].max() < ASK_LO


def test_skip_S2_never_evidences_red_steps():
    src = ReplaySource("skip_S2")
    _, ev, _ = _series(src)
    assert ev["S2"].max() < ASK_LO and ev["S3"].max() < ASK_LO
    for s in ("S1", "S4", "S5", "S6"):
        assert _first_confirm(ev[s], src.fps) is not None


def test_S4_before_S2_order_in_evidence():
    src = ReplaySource("S4_before_S2")
    _, ev, _ = _series(src)
    f = {s: _first_confirm(ev[s], src.fps) for s in src.protocol.step_ids}
    assert f["S1"] < f["S4"] < f["S5"] < f["S2"] < f["S3"] < f["S6"]


def test_unsure_S3_sits_in_ask_band():
    src = ReplaySource("unsure_S3")
    _, ev, _ = _series(src)
    assert _first_confirm(ev["S3"], src.fps) is None
    in_band = (ev["S3"] >= ASK_LO) & (ev["S3"] < ASK_HI)
    # longest run inside the band exceeds ask_after_s = 3 s
    best = run = 0
    for b in in_band:
        run = run + 1 if b else 0
        best = max(best, run)
    assert best / src.fps > 3.0
    assert _first_confirm(ev["S4"], src.fps) is not None


def test_occluded_collapses_visibility_then_recovers():
    src = ReplaySource("occluded")
    ticks, ev, vis = _series(src)
    s3 = [i for i, t in enumerate(ticks) if t.label == "S3"]
    low = np.where(vis < 0.3)[0]
    assert len(low) / src.fps > 1.5
    assert low[0] > s3[0]
    # while blind, S3 evidence is uncertain (near 0.5), and no detections are reported
    blind = low[len(low) // 2]
    assert 0.35 < ev["S3"][blind] < ASK_HI
    assert ticks[blind].detections.detections == ()
    # after the view clears, S3 is confirmed
    confirm = _first_confirm(ev["S3"], src.fps)
    assert confirm is not None and confirm > low[-1]


def test_unexpected_reverts_S2_evidence():
    src = ReplaySource("unexpected")
    ticks, ev, _ = _series(src)
    ret = [i for i, t in enumerate(ticks) if t.label == "other"]
    assert ret
    assert ev["S2"][ret[0] - 1] > P_ENTER  # red was out
    after = ret[-1] + 2
    assert ev["S2"][after] < ASK_LO  # red back in the outer box: transition undone


def test_annotations_follow_trec_csv_convention(tmp_path):
    src = ReplaySource("skip_S2")
    path = tmp_path / "truth.csv"
    src.write_annotations(path)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows[0] == {"frame_idx": "0", "t": "0.0000", "label": "idle"}
    assert [r["label"] for r in rows if r["label"] != "idle"] == ["S1", "S4", "S5", "S6"]


def test_run_publishes_to_bus():
    bus = Bus()
    subs = {
        t: bus.subscribe(t, maxsize=10_000)
        for t in (Topic.FRAMES, Topic.PREDICATES, Topic.EVIDENCE, Topic.ACTIONS, Topic.DETECTIONS)
    }
    src = ReplaySource("correct", fps=10)
    n = src.run(bus, realtime=False, close_bus=True)
    assert n == len(src)
    assert len(subs[Topic.PREDICATES].drain()) == n
    assert len(subs[Topic.EVIDENCE].drain()) == n * 6
    assert len(subs[Topic.FRAMES].drain()) == 0  # render off


def test_rendered_frames_carry_detectable_aruco_markers():
    cv2 = pytest.importorskip("cv2")
    src = ReplaySource("correct", render=True)
    frame = next(iter(src)).frame
    assert frame is not None and frame.image.shape == (480, 640, 3)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    _, ids, _ = cv2.aruco.ArucoDetector(d).detectMarkers(frame.image)
    assert ids is not None and sorted(ids.ravel().tolist()) == [0, 1, 2, 3]


def test_cli_prints_predicate_stream(capsys):
    assert main(["--scenario", "skip_S2", "--stride", "30"]) == 0
    out = capsys.readouterr().out
    assert "scenario skip_S2" in out and "SIMULATED" in out
    rows = [ln for ln in out.splitlines() if ln and not ln.startswith(("#", "frame"))]
    assert len(rows) == 8


def test_cli_jsonl(capsys):
    assert (
        main(
            [
                "--scenario",
                "correct",
                "--format",
                "jsonl",
                "--stream",
                "evidence",
                "--stride",
                "100",
            ]
        )
        == 0
    )
    lines = capsys.readouterr().out.strip().splitlines()
    assert all('"type":"StepEvidence"' in ln for ln in lines)
