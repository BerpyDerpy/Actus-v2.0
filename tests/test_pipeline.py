"""T-INT: the pipeline harness end to end, without cameras, audio or neural models.

The replay source stands in for the camera (rendered frames with real ArUco markers, watermarked
SIMULATED) and for the detector (its boxes, with rack positions stripped so the spatial lane has
to project them through the rack it solves). Everything downstream -- rack frame, world model,
evidence, engine, event log, downlink, overlays -- is the real code.
"""

import dataclasses
import itertools
import json
import types

import cv2
import numpy as np
import pytest

from actus.app import overlay
from actus.app.pipeline import (
    CameraSource,
    FileSource,
    LiveApp,
    PipelineClock,
    PipelineCore,
    ReplayFrameSource,
    Runner,
    VoiceSink,
    build_detector,
    build_handpose,
    load_plugin,
    run_offline,
)
from actus.capture.replay import ReplaySource
from actus.core.bus import Bus
from actus.core.config import load_config
from actus.core.contracts import (
    Alert,
    AlertKind,
    Answer,
    DetectionSet,
    FramePacket,
    Prompt,
    Stats,
    Topic,
)
from actus.io.eventlog import EventLog, verify_log
from actus.protocol.engine import ProtocolEngine, describe, run_ticks
from actus.spatial.estimator import SpatialEstimator
from actus.voice.tts import NullTTS


def _cfg(tmp_path, *extra):
    return load_config(
        overrides=[f"run.runs_dir={tmp_path.as_posix()}", "models.enabled=false", *extra]
    )


def _statuses(outcomes):
    return {r["step_id"]: r["status"] for r in outcomes}


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _strip_rack(d: DetectionSet) -> DetectionSet:
    dets = [dataclasses.replace(x, bbox_rack_xy=None) for x in d.detections]
    return DetectionSet(d.frame_idx, d.t, dets, d.backend)


# --------------------------------------------------------------------------------------------
# The core on the replay
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", ["correct", "skip_S2", "S4_before_S2"])
def test_core_matches_the_spatial_chain_it_wires(scenario, tmp_path):
    """The pipeline adds wiring, not behaviour: its engine outputs equal the engine driven
    directly by the spatial lane over the same rendered frames."""
    res = run_offline(f"replay:{scenario}", _cfg(tmp_path), run_id="r", video=False, models=None)

    src = ReplaySource(scenario, render=True)
    est = SpatialEstimator.from_config(load_config(), protocol=src.protocol)
    ticks = []
    for tk in src:
        sf = est.update(tk.frame_idx, tk.t, frame=tk.frame, detections=_strip_rack(tk.detections))
        ticks.append(
            types.SimpleNamespace(
                frame_idx=tk.frame_idx,
                t=tk.t,
                evidence=sf.evidence,
                predicates=sf.predicates,
                actions=tk.actions,
            )
        )
    ref = run_ticks(ProtocolEngine(src.protocol), ticks)
    assert [describe(o) for o in res.summary.outputs] == [describe(o) for o in ref]


def test_core_outcomes_on_the_replay(tmp_path):
    cfg = _cfg(tmp_path, "pipeline.overlay=false")
    ok = run_offline("replay:correct", cfg, run_id="ok", models=None)
    assert set(_statuses(ok.summary.outcomes).values()) == {"verified"}

    skip = run_offline("replay:skip_S2", cfg, run_id="skip", models=None)
    st = _statuses(skip.summary.outcomes)
    assert st["S2"] == st["S3"] == "skipped"
    kinds = [a.kind for a in skip.summary.alerts]
    assert AlertKind.OUT_OF_ORDER in kinds and AlertKind.SKIP in kinds


# --------------------------------------------------------------------------------------------
# Offline runs: determinism, artefacts, scripted answers
# --------------------------------------------------------------------------------------------


def test_run_offline_is_byte_identical_across_reruns(tmp_path):
    cfg = _cfg(tmp_path)
    a = run_offline("replay:S4_before_S2", cfg, run_id="a", models=None)
    b = run_offline("replay:S4_before_S2", cfg, run_id="b", models=None)
    assert a.events_path.read_bytes() == b.events_path.read_bytes()
    assert a.events_sha256 == b.events_sha256
    assert verify_log(a.events_path).ok
    assert 0 < a.downlink_bytes <= 4096
    assert sorted(p.name for p in (a.run_dir / "video").glob("annotated_*.mp4"))
    meta = json.loads((a.run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["simulated"] is True and meta["events_sha256"] == a.events_sha256
    recs = _records(a.events_path)
    assert recs[-1]["type"] == "run_end" and recs[0]["ts"].startswith("1970-01-01T00:00:00")


def test_run_offline_refuses_to_overwrite_without_force(tmp_path):
    cfg = _cfg(tmp_path, "pipeline.overlay=false")
    first = run_offline("replay:correct", cfg, run_id="same", models=None)
    with pytest.raises(FileExistsError):
        run_offline("replay:correct", cfg, run_id="same", models=None)
    again = run_offline("replay:correct", cfg, run_id="same", models=None, force=True)
    assert again.events_sha256 == first.events_sha256  # the chain restarted, not appended


def test_scripted_answer_confirms_an_occluded_step(tmp_path):
    cfg = _cfg(tmp_path, "pipeline.overlay=false")
    res = run_offline(
        "replay:occluded", cfg, run_id="occ", models=None, evidence="replay", answer="yes"
    )
    assert _statuses(res.summary.outcomes)["S3"] == "operator_confirmed"
    answers = [r for r in _records(res.events_path) if r["type"] == "Answer"]
    assert len(answers) == 1 and answers[0]["frame_idx"] is not None
    assert answers[0]["value"] == "yes"


class _ReplayDetector:
    """Stands in for a trained detector on a video rendered from the replay: returns that
    replay's boxes for the frame index, without their rack positions."""

    backend_name = "test-replay"

    def __init__(self, by_frame):
        self.by_frame = by_frame

    def detect(self, frame: FramePacket) -> DetectionSet:
        d = self.by_frame[frame.frame_idx]
        return DetectionSet(frame.frame_idx, frame.t_capture, _strip_rack(d).detections, "test")


def _write_replay_video(path, scenario):
    src = ReplaySource(scenario, render=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), src.fps, (640, 480))
    assert writer.isOpened()
    by_frame = {}
    for tk in src:
        writer.write(tk.frame.image)
        by_frame[tk.frame_idx] = tk.detections
    writer.release()
    return by_frame


def test_a_recorded_file_runs_deterministically_on_its_own_clock(tmp_path):
    video = tmp_path / "op99_correct_01.mp4"
    by_frame = _write_replay_video(video, "correct")
    (tmp_path / "op99_correct_01.json").write_text(
        json.dumps({"started_at": "2026-09-28T10:00:00+05:30"}), encoding="utf-8"
    )

    src = FileSource(video)
    ts = [sf.packet.t_capture for sf in src]
    assert len(ts) == len(by_frame) == src.frames_read
    assert all(b >= a for a, b in itertools.pairwise(ts))
    assert ts[-1] == pytest.approx((len(ts) - 1) / 15.0, abs=0.07)  # container time, 15 fps

    cfg = _cfg(tmp_path)
    runs = [
        run_offline(
            video,
            cfg,
            run_id=f"file{i}",
            detector=_ReplayDetector(by_frame),
            handpose=None,
            models=None,
        )
        for i in range(2)
    ]
    assert runs[0].events_path.read_bytes() == runs[1].events_path.read_bytes()
    assert set(_statuses(runs[0].summary.outcomes).values()) == {"verified"}
    first = _records(runs[0].events_path)[0]
    assert first["ts"].startswith("2026-09-28T04:30:00")  # take start (UTC) + t
    assert runs[0].simulated is False


def test_file_source_refuses_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        FileSource(tmp_path / "nope.mp4")


# --------------------------------------------------------------------------------------------
# The Runner on a bus: operator input, stats, annotated frames
# --------------------------------------------------------------------------------------------


def _replay_core(tmp_path, **kw):
    return PipelineCore.from_config(
        _cfg(tmp_path), detector=None, handpose=None, models=None, simulated=True, **kw
    )


def test_gui_answers_on_the_bus_reach_the_engine_and_are_logged_once(tmp_path):
    bus = Bus()
    core = _replay_core(tmp_path, evidence_mode="replay")
    asks = bus.subscribe(Topic.ASKS, maxsize=16, name="test-gui")
    stats = bus.subscribe(Topic.STATS, maxsize=4096, name="test-stats")
    annotated = bus.subscribe(Topic.ANNOTATED, maxsize=4, name="test-annotated")
    log = EventLog(tmp_path / "run", catch_signals=False)

    def gui(res):  # a person pressing "Yes" on the Ask card as soon as it appears
        for env in asks.drain():
            bus.publish(Topic.ANSWERS, Answer(env.payload.question_id, "yes", "gui", t=res.t))

    summary = Runner(
        core, ReplayFrameSource("occluded"), bus=bus, eventlog=log, on_result=gui
    ).run()
    log.close()
    assert _statuses(summary.outcomes)["S3"] == "operator_confirmed"
    answers = [r for r in _records(log.path) if r["type"] == "Answer"]
    assert len(answers) == 1 and answers[0]["via"] == "gui"

    got = [e.payload for e in stats.drain()]
    pipe = [s for s in got if isinstance(s, Stats) and s.source == "pipeline"]
    assert pipe, "the Runner publishes pipeline stats"
    last = pipe[-1].values
    assert last["simulated"] is True and last["frames"] == summary.frames
    assert "ms.spatial" in last and "ms.engine" in last and last["dropped"] == 0
    assert any(s.source == "spatial" for s in got)
    frame = annotated.drain()[-1].payload
    assert frame.image.shape == (480, 640, 3)


def test_scripted_answers_published_on_the_bus_are_not_applied_twice(tmp_path):
    bus = Bus()
    seen = bus.subscribe(Topic.ANSWERS, maxsize=16, name="test-answers")
    log = EventLog(tmp_path / "run", catch_signals=False)
    core = _replay_core(tmp_path, evidence_mode="replay")
    Runner(core, ReplayFrameSource("occluded"), bus=bus, eventlog=log, answer="yes").run()
    log.close()
    n_bus = len(seen.drain())
    n_log = sum(r["type"] == "Answer" for r in _records(log.path))
    assert n_bus == n_log >= 1


def test_camera_source_counts_the_frames_it_drops(tmp_path):
    from actus.capture.camera import Camera

    frames = [tk.frame.image for tk in ReplaySource("correct", render=True)][:60]

    class FakeCapture:
        def __init__(self):
            self.i = 0

        def isOpened(self):
            return True

        def read(self):
            if self.i >= len(frames):
                return False, None
            self.i += 1
            return True, frames[self.i - 1]

        def get(self, prop):
            return 0.0

        def set(self, prop, value):
            return True

        def release(self):
            pass

    bus = Bus()
    cam = Camera(0, capture_factory=FakeCapture, max_live_read_failures=1)
    source = CameraSource(cam, bus, maxsize=2)
    summary = Runner(_replay_core(tmp_path), source, bus=bus).run()
    assert summary.frames >= 1
    assert summary.frames + summary.dropped == len(frames)


# --------------------------------------------------------------------------------------------
# Live app smoke run (replay source; no GUI, audio or network)
# --------------------------------------------------------------------------------------------


def test_live_app_writes_every_run_artefact(tmp_path):
    cfg = _cfg(tmp_path)
    app = LiveApp(
        cfg,
        source="replay",
        source_arg="correct",
        run_id="live",
        ui=False,
        voice=False,
        stream=False,
        record=True,
        realtime=False,
    )
    summary, run_dir = app.run()
    assert set(_statuses(summary.outcomes).values()) == {"verified"}
    assert verify_log(run_dir / "events.jsonl").ok
    assert (run_dir / "downlink.json").stat().st_size <= 4096
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["simulated"] is True and meta["sinks"] == ["record"]
    names = {p.name for p in (run_dir / "video").iterdir()}
    assert {"raw_000.mp4", "annotated_000.mp4"} <= names


# --------------------------------------------------------------------------------------------
# Plug-ins, voice, overlay, clock
# --------------------------------------------------------------------------------------------


class _NullDetector:
    backend_name = "null"

    def detect(self, frame):
        return DetectionSet(frame.frame_idx, frame.t_capture, [], "null")


def make_null_detector(cfg):
    return _NullDetector()


def test_stages_are_chosen_by_config(tmp_path):
    cfg = _cfg(tmp_path, "pipeline.detector=test_pipeline:make_null_detector")
    assert isinstance(build_detector(cfg), _NullDetector)
    assert build_detector(_cfg(tmp_path, "pipeline.detector=none")) is None
    assert build_handpose(_cfg(tmp_path, "pipeline.handpose=none")) is None
    with pytest.raises(ValueError):
        load_plugin("no_colon_here", cfg)
    with pytest.raises(ModuleNotFoundError):
        load_plugin("actus.no_such_module:factory", cfg)


def test_disabled_stages_report_did_not_run(tmp_path):
    core = PipelineCore.from_config(
        _cfg(tmp_path), detector=_NullDetector(), handpose=None, models=None
    )
    frames = iter(ReplayFrameSource("correct"))
    first = dataclasses.replace(next(frames), detections=None)  # make the detector run
    assert core.step(first).detections is not None
    core.disabled.add("detector")
    second = dataclasses.replace(next(frames), detections=None)
    assert core.step(second).detections is None


def test_voice_sink_orders_alerts_first_and_tracks_the_open_ask(tmp_path):
    bus = Bus()
    sink = VoiceSink(bus, _cfg(tmp_path).voice, backend=NullTTS(), asr=False)
    sink.handle(Prompt("S2", "Take out the red box", frame_idx=1, t=0.1))
    sink.handle(Alert("out_of_order", "Out of sequence", ("S4", "S2"), frame_idx=2, t=0.2))
    assert sink.queue.get(timeout=0).kind == "alert"
    assert sink.queue.get(timeout=0).kind == "prompt"
    assert sink.stats().values["tts"] == "injected"
    sink.stop()


def test_overlay_draws_on_a_copy():
    img = np.zeros((480, 640, 3), np.uint8)
    out = overlay.render(
        img,
        step_ids=["S1", "S2"],
        statuses={"S1": "verified"},
        prompt="Take out the red box",
        alert="Out of sequence: yellow box removed before red box.",
        fps=15.0,
        simulated=True,
        tags=["FAULT: occlusion"],
    )
    assert out.shape == img.shape and out.any() and not img.any()
    grey = overlay.render(np.zeros((120, 160), np.uint8), ask="Did you place it?")
    assert grey.shape == (120, 160, 3)


def test_event_log_clock_makes_timestamps_a_function_of_the_input(tmp_path):
    clock = PipelineClock(1_790_000_000.0)
    logs = []
    for name in ("a", "b"):
        clock.t = 0.0
        log = EventLog(tmp_path / name, catch_signals=False, clock=clock)
        clock.t = 2.5
        log.append(Prompt("S1", "Open the outer box", frame_idx=0, t=2.5))
        log.close()
        logs.append((tmp_path / name / "events.jsonl").read_bytes())
    assert logs[0] == logs[1]
    assert json.loads(logs[0].splitlines()[-1])["duration_s"] == 2.5
