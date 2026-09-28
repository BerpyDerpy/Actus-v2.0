# T-INT: decisions made outside the build plan

2026-09-28 · covers T-INT (the pipeline harness: `actus/app/pipeline.py`, `tools/run_offline.py`,
`scripts/run_demo.sh`) · read with `build-plan.md` §4–5 and the other `Vault/*.md` notes

The T-INT card specifies the lane in outline. This file lists every decision made during
implementation that the card does not state, so that the lanes plugging into the pipeline (T-FAULT,
T-EVAL, T-DET-TRAIN, T-HOI-TEMP, T-PKG, T-DEMO) can rely on them or overturn them on purpose.

**Scope, as agreed:** this is the harness. Every stage is wired and swappable, and the whole chain
runs end to end on the SIMULATED replay. Plugging in the real models is left for later (§3 says
how). Nothing is committed.

Status at handover:
- 18 new tests in `tests/test_pipeline.py` pass. The full suite has 1 failure, and it predates this
  lane (§7). `ruff check` and `ruff format --check` are clean on the whole repo.
- No dependency was added.
- Files outside T-INT's own that were edited are listed in §6. All the edits are small and
  backwards compatible.

---

## 0. Status against the card's "done when"

| Criterion | Status |
|---|---|
| `run_offline.py` reproduces an identical `events.jsonl` across two runs | **Verified on the replay** (`--verify-determinism` prints `IDENTICAL`) and **on a real encoded MP4**: a test renders the replay to mp4v, decodes it through `FileSource`, and uses a stand-in detector. **Not verified with YOLO + MediaPipe**: the `perception` extra isn't installed here and there are no takes under `data/raw/` on this machine. |
| `run_demo.sh --live` runs the box experiment at ≥ 15 FPS on the M2, with voice, GUI, RTSP and recording | **Not verified.** There is no M2, no perception extra, no ffmpeg/MediaMTX and no ui/voice extras here. The harness's own cost was measured instead: **~7.5 ms/frame** on the replay (240 frames in 1.8 s). That covers render, ArUco, spatial, engine and overlay, with no neural stages. At 15 FPS that leaves ~59 ms per frame for the detector and hands/pose. |
| Wire capture → perception → spatial → models → engine → voice / log / UI / stream / record | Done. Each sink was started for real, and each one degrades cleanly when its dependency is missing (checked here with all of them missing). |
| Stage FPS and drops go to `stats` | Done (§4). |
| Decision thresholds tuned on training-operator takes only | **Not done, and blocked.** Recorded footage gives no valid rack frame and no red-box detections (`Vault/T-HOI-TEMP.md` §0.1–0.2), so the engine gets no evidence to tune on. Tuned values go in the new `engine:` config section (§6). |

## 1. Files and entry points

| File | What it holds |
|---|---|
| `actus/app/pipeline.py` | `PipelineCore`, the sources (`FileSource`, `CameraSource`, `ReplayFrameSource`), `Runner`, `VoiceSink`, `UIServer`, `LiveApp`, `run_offline()`, the plug-in loader, and the CLI. |
| `actus/app/overlay.py` | Draws the annotated frame (§5). Not listed on the card. It was split out so T-FAULT and T-UI can reuse it. |
| `tools/run_offline.py` | A thin CLI over `pipeline.run_offline()`. |
| `scripts/run_demo.sh` | `--live` / `--file` / `--replay` / `--offline`. It `cd`s to the repo root and puts `bin/` on `PATH`. |
| `tests/test_pipeline.py` | See §8. |

```sh
scripts/run_demo.sh --replay skip_S2                        # SIMULATED; GUI on :8000
scripts/run_demo.sh --live                                  # capture.source; add --source 1 / rtsp://...
scripts/run_demo.sh --file data/raw/op01_correct_01.mp4     # a take, at its own rate
scripts/run_demo.sh --offline data/raw/op01_correct_01.mp4 --verify-determinism
uv run python tools/run_offline.py replay:occluded --evidence replay --answer yes
uv run python -m actus.app.pipeline --help                  # --no-ui/--no-voice/--no-stream/--no-record,
                                                            # --answer, --seconds, --no-realtime, --override
```

**T-EVAL should import `run_offline` rather than shell out to the tool:**
`from actus.app.pipeline import run_offline`. It returns `OfflineResult(run_dir, summary,
events_path, events_sha256, downlink_bytes, simulated)`, and `summary.outputs` holds every engine
output in order.

## 2. Architecture

Three layers, so that live, offline and test runs share one code path:

- **`PipelineCore.step(SourceFrame) -> FrameResult`** makes one synchronous call per frame. It does
  no I/O and starts no threads. The order inside it is fixed:
  1. frame hooks;
  2. **rack** (solved first, because both the models and spatial need it);
  3. detector;
  4. hands/pose;
  5. **models** (`ModelStack`, before spatial, because spatial takes `hoi`);
  6. spatial;
  7. fusion (`ModelStack.fuse`);
  8. engine;
  9. overlay.

  `FrameResult.data_messages()` lists what gets published: rack-projected detections, hands and
  pose (the UI mini-map needs the rack versions), rack, predicates, the **fused** evidence, actions,
  and the per-frame `spatial` and `models` stats.
- **Sources** yield `SourceFrame`s, or `None` on an idle tick so the Runner can still serve operator
  input:
  - `CameraSource` wraps T-PERC's threaded `Camera`. The camera publishes every frame on `frames`,
    so the recorder and the raw RTSP push see all of them. The pipeline reads through its own
    `drop_oldest` subscription (maxsize 2), so when processing falls behind it works on the freshest
    frame and counts what it skipped. A test checks that processed + dropped = captured.
  - `FileSource` decodes in sequence, never drops or loops, and uses the container clock (§4).
  - `ReplayFrameSource` renders the P0 world. It passes the replay's boxes on as the "detector"
    output **with `bbox_rack_xy` stripped**, so spatial has to solve the rack from the rendered
    ArUco markers and project the boxes itself. That is the real chain minus the neural perception.
    It is always marked SIMULATED.
- **`Runner`** handles one frame at a time:
  1. it applies whatever operator input is due;
  2. it runs `core.step`;
  3. it publishes the data messages;
  4. it logs and publishes the engine outputs;
  5. it publishes and/or writes the annotated frame;
  6. it publishes stats.

  At the end of the stream the outputs of `engine.finish()` go the same way. The bus, event log and
  video writer are all optional; offline runs have no bus.
- **`LiveApp`** is a Runner plus the sinks (recorder, streamer, `VoiceSink`, `UIServer`), with setup
  and teardown in one place. It also writes `run.json` and `downlink.json`.

**The engine is driven directly by the core, not through `EngineRunner`.** The core already has
every step's evidence for the frame, so batching on the bus would only add latency and a
cross-thread hazard. The engine stays single-threaded, since only the Runner's thread touches it,
and deterministic.

**Evidence mode.** `pipeline.evidence: spatial` (the default) is the real chain. `replay` feeds the
replay's stand-in evidence and predicates to the engine instead, but only when the source is the
replay. It exists because the replay publishes no hands, and without hands spatial cannot observe
the handling in `occluded` or the brief placement in `unexpected` (`Vault/T-SPATIAL.md` §8). Use it
to demo the ask path and the unexpected alert on the replay. Both modes are labelled SIMULATED.

**Operator input**
- Bus `answers` and `commands` (from the GUI or the ASR) are drained between frames and applied
  through `core.answer` / `core.command`.
- An `Answer` without a `frame_idx` is stamped with the last processed frame before it is logged, as
  the contract's "stamped by the receiver" rule allows.
- Scripted answers (`--answer yes|no`, or a callable) follow `engine.run_ticks`'s semantics: they
  are due `delay` after the Ask and applied before the first frame at or after that time. They are
  also published on `answers`, so the UI and the ASR gate see them. The Runner remembers which ones
  it published, so they are not applied or logged twice (tested).
- `Command`s are not logged, because `EventLog` doesn't accept them. Their outputs are logged.

## 3. Plugging in a better model

Every stage is a plain object, chosen in `configs/default.yaml` under `pipeline:`. The engine stays
the judge; a model only ever reaches it as evidence.

| Key | Values | Interface the object must have |
|---|---|---|
| `pipeline.detector` | `config` (T-PERC `Detector`, per `detector.backend`), `none`, or `"pkg.module:factory"` | `detect(FramePacket) -> DetectionSet`; optionally `backend_name` and `reset()` |
| `pipeline.handpose` | `config` (T-PERC `RotationTTA`), `none`, or a factory | `process(FramePacket) -> (HandSet \| None, PoseSet \| None)` |
| `pipeline.rack` | `aruco` (the measured `rack.yaml`), `selfcal` (T-HOI-TEMP's `MarkerFrameFitter`), or a factory | `update(FramePacket) -> RackFrame` |
| `models.hoi` / `models.temporal` | path stems | Drop in `<stem>.npz` (and `.onnx`). `ModelStack` loads them; if they are missing it falls back to rule evidence. |

- A factory is called as `factory(cfg)` with the whole `Config`.
- `load_plugin` accepts `"pkg.module:Class.attr"`. Nothing is registered anywhere.
- Python callers (tests, T-EVAL) can also pass stage objects straight to
  `PipelineCore.from_config(cfg, detector=..., handpose=..., models=...)` or to
  `run_offline(...)`. Passing `None` means "no such stage".
- Swapping the detector for T-DET-TRAIN's model needs no pipeline change:
  `detector.backend: custom` already does it.

```python
# my_models/det.py  ->  pipeline.detector: "my_models.det:make"
def make(cfg):
    return MyDetector(cfg.detector.custom_weights)  # .detect(FramePacket) -> DetectionSet
```

- `detector_every_n` / `handpose_every_n` run a stage on every Nth processed frame. On the other
  frames the stage reports "did not run" (`None`). Spatial then only predicts, which is different
  from "ran and saw nothing".
- **Selfcal caveat:** it gives a metric, rotation-invariant frame on unmeasured rigs, but its origin
  is wherever the first-seen marker lies. The protocol's zones **will not line up**, so S3/S5
  evidence is meaningless under it. It is for looking at footage, not for judging it.

## 4. Decisions the card leaves open

**Clocks**
- `FileSource` takes `t` from `CAP_PROP_POS_MSEC`. If the container has no usable timestamps (its
  second frame reports 0), it uses `frame_idx / container_fps`: the container's declared rate, not a
  guessed one. A file with neither is refused. `t` is forced to be non-decreasing, because the
  engine raises on a decreasing `t`.
- `CameraSource` keeps T-PERC's `time.monotonic()`.

**A deterministic offline log**
- `EventLog` gained an optional `clock` (§6). `run_offline` passes a `PipelineClock`: the take's
  start time plus the `t` of the frame being processed.
- The start time comes from the T-REC metadata (`<stem>.json: started_at`, with its timezone). With
  no metadata it falls back to the epoch, so a log starts at `1970-01-01T00:00:00Z + t`. That looks
  odd but keeps the log a pure function of the input. The replay always uses the epoch.
- `run.json` records which clock was used (`log_clock`).
- The downlink's `started_at` / `ended_at` use the same clock.
- `run.json` also holds wall-clock data (`wall_s`) and is deliberately not part of the determinism
  claim.
- **`run_offline` refuses a non-empty run dir unless `force`.** `EventLog` resumes an existing
  chain, so a rerun into the same dir would otherwise append to the old log instead of reproducing
  it. With `force`, the old events, downlink, `run.json` and video are removed first.

**Run directories**
- Live runs go to `runs/<YYYYmmdd-HHMMSS>`, or `runs/sim-<...>` for the replay.
- Offline runs go to `runs/offline-<stem>-<ts>`; `--verify-determinism` adds `<id>-rerun`.
- `--run-id` overrides, and so does `run.run_id` in the config.
- Every run writes `events.jsonl`, `downlink.json` (checked against `log.downlink_max_bytes`),
  `run.json` and `video/`. Live runs record `raw_NNN.mp4` and `annotated_NNN.mp4` through T-IO's
  `Recorder`. Offline runs write only `annotated_NNN.mp4`, because the input is the raw video.

**Failure tolerance (live)**
- Each sink starts inside its own try block. A missing `ui` extra, TTS or ASR package, audio device,
  Vosk model, or ffmpeg/MediaMTX binary is logged as a warning, and the run continues.
- TTS falls back from Kokoro to pyttsx3 to text-only (`NullTTS`).
- The streamer's `ManagedProcess` keeps retrying with backoff and never crashes the run.
- **`run.json` records `sink_status`**, which is each sink's final stats. A "started" streamer with
  no binary therefore shows `running: false` and `last_error: ... cannot find the file`, rather
  than implying the stream went out.
- Recording and the event log never depend on the network.

**Interrupts**
- Ctrl-C and SIGTERM (which `LiveApp` turns into `KeyboardInterrupt` while it runs) end the source
  loop. The engine's `finish()` outputs still reach the log, and `run_end` records the reason
  `interrupted`.
- For this reason `LiveApp` builds `EventLog(catch_signals=False)`. If `EventLog` caught the signal
  itself, it would close the log *before* the skip outputs of `finish()` were written.

**Voice**
- `VoiceSink` speaks alerts, asks and prompts through T-VOICE's queue (alert > ask > prompt).
- It drives the ASR `AskGate` itself instead of calling `start_bus_gate`. That way the gate also
  closes on a later `StepEvent` for the asked step, which is the engine's convention for "resolved
  by evidence" (`Vault/T-ENGINE.md` §5). A newer Ask replaces the open one.
- **Push-to-talk isn't wired**, because nothing reads the key. Voice commands work only while an Ask
  is open; the GUI Yes/No buttons always work.

## 5. Stats and overlay

**`stats`**
- The Runner publishes `Stats("pipeline")` every `pipeline.stats_interval_s` (1 s). The first keys
  are `fps`, `dropped` and `simulated`, because the StatsStrip shows only the first three. After
  those come:
  - `frames`, `source`, `evidence`;
  - `stage.detector`, `stage.handpose`, `stage.rack`, `models.enabled`;
  - `ms.<stage>` (an EMA per stage) and `ms.total`;
  - `bus.dropped`, and `tags` when a fault is active.
- The sinks add `Stats("voice")`, `Stats("recorder")` and `Stats("streamer")` at the same interval.
  The streamer's includes the target IP.
- `Stats("spatial")` is published **every frame**, because it drives the mini-map, and so is
  `Stats("models")` whenever a model stack is present.

**The annotated frame** (`overlay.render`) shows:
- the rack board, axes, zones and container footprints, plus the rack status line (from T-SPATIAL's
  `draw_overlay`);
- detection boxes with track ids;
- hand skeletons, and the upper-body pose where visibility is at least 0.3;
- a checklist chip per step, coloured by status;
- `NEXT:` with the prompt, or `ASK:` with the open question;
- the latest alert, for 6 s of pipeline time;
- fps and tags at the top right, plus **SIMULATED** whenever the input is synthetic;
- on the replay only, the ground-truth label.

Text is reduced to ASCII, because the Hershey fonts can't draw anything else.

## 6. Edits outside T-INT's files (integrator changes)

| File | Change | Why |
|---|---|---|
| `actus/io/eventlog.py` (T-IO) | New `clock` keyword on `EventLog` (default `None` = the old behaviour) and a new `iso_at()` helper. With a clock, `ts`, `started_at` and `duration_s` come from it. | Without it, the card's identical-`events.jsonl` criterion can't be met: `ts` is part of the hash chain. |
| `actus/app/server.py` (T-UI) | `/stream/mjpeg` prefers `annotated` once it has seen an annotated frame. | It subscribed to both frame topics, so with the pipeline publishing both, the fallback feed would flicker between raw and annotated. |
| `configs/default.yaml` (P0) | New `pipeline:` section (§3). New free-form **`engine: {}`** section, passed as `DecisionParams.from_protocol(proto, **cfg.engine)`. | `engine:` applies T-ENGINE's §2 proposal. Unknown keys still fail, because `from_protocol` rejects them. |
| `actus/core/contracts.py` (P0) | **Comments only**: `PoseObs.world_landmarks` is in the axes MediaPipe saw, i.e. rotated by `rotation_used`, not rotated back. `RackFrame.reproj_err` is the RMS of the plane fit that gates `valid`. | This resolves T-SPATIAL's §9 proposals. The first matches what T-PERC's `rotation_tta._unrotate_raw` actually does and T-SPATIAL's default (`world_landmarks_rotated_back: false`). No field or behaviour changed. |

## 7. Open items and known limitations

**Contract proposals still open.** They were deliberately not applied, because each one needs
UI/voice changes to be worth doing:
- T-ENGINE's `AskClosed`. The convention in §4 covers it for now.
- T-SPATIAL's `WorldState` topic. `Stats("spatial")` carries the same data.

**Real footage cannot be judged yet**, for reasons upstream of this lane (`Vault/T-HOI-TEMP.md`
§0):
- no valid rack frame against `rack.yaml`;
- the open-vocab detector never sees the red box.

The harness runs on the takes, but every step stays `pending`. Fix the rig (or calibrate per
session into `rack.yaml`) and land T-DET-TRAIN; neither needs a pipeline change.

**Unverified here** (see §0):
- YOLO/MediaPipe determinism;
- live throughput on the M2;
- a real webcam (the camera path is tested only with a fake capture);
- real audio, a real RTSP push, and the GUI served from the pipeline.

Run `scripts/run_demo.sh --live` on the demo machine once `uv sync --all-extras`,
`scripts/setup_media.sh` and `npm run build` are done.

**Other limitations**
- **Annotated video timing.** The annotated stream runs at the *processing* rate but is recorded at
  the nominal fps. If live processing drops frames, `annotated_NNN.mp4` plays faster than real
  time. The `.frames.csv` sidecar still has the true `t`. The fix is T-IO's: the `Recorder` would
  resample to its clock, as T-REC does.
- **Temporal model auto-loads.** If `models/temporal/temporal.npz` exists, `ModelStack` loads it and
  fusion uses it. The current model loses to the duration baseline (`Vault/T-HOI-TEMP.md` §5).
  Fusion's visibility bound keeps it from inventing a step, but consider `models.enabled: false`
  for the demo until it is retrained.
- **pyttsx3 on Windows** is created on the main thread and spoken from the player thread. SAPI/COM
  may not like that. This is untested, and Kokoro is the primary engine anyway.
- On replay frames the SIMULATED tag appears twice (the replay's watermark and the overlay's). This
  is cosmetic.
- **Hooks for T-FAULT are in place, but `faults.py` isn't written:**
  - `core.frame_hooks` holds `FramePacket -> FramePacket` functions (occlude, dim, freeze);
  - `core.disabled` takes `{"detector", "handpose", "models"}` (drop a stage);
  - `core.tags` holds strings drawn on the frame and listed in the stats (`"FAULT: ..."`);
  - killing the network target is the streamer's job (`stream.target_ip`).

**A failure that predates this lane.** `tests/test_det_train.py::
test_full_image_box_baseline_reports_mean_iou` fails on Windows.
`tools/train_detector.py:351` maps image paths to label paths with
`str(path).replace("/images/", "/labels/")`, which never matches a backslash path. The fix belongs
to T-DET-TRAIN (use `Path.parts`). The `test_io_process` / `test_io_streamer` failures noted in
`Vault/T-HOI-TEMP.md` §7 pass in this session.

## 8. Tests (`tests/test_pipeline.py`, 18)

- **The core adds wiring, not behaviour.** For `correct`, `skip_S2` and `S4_before_S2`, the engine
  outputs from `run_offline("replay:…")` equal those of the engine driven directly by
  `SpatialEstimator` over the same rendered frames.
- **Outcomes:**
  - `correct` → every step is `verified`;
  - `skip_S2` → S2 and S3 are `skipped`, with an out_of_order alert followed by a skip alert.
- **Determinism:**
  - replay reruns give byte-identical logs, the chain verifies, the downlink is at most 4 KB, the
    annotated MP4 exists, and `run.json` is marked SIMULATED;
  - on an **mp4v-encoded file** (stand-in detector) reruns are identical, `t` is the container
    clock, `ts` is the take's `started_at` converted to UTC, and every step is verified.
- `run_offline` refuses to overwrite without `force`; with it, the chain restarts rather than
  appends.
- **The ask path:**
  - scripted "yes" → S3 is `operator_confirmed`, with one logged `Answer` that has a `frame_idx`;
  - a GUI "yes" published on the bus → the same result, logged exactly once;
  - scripted answers echoed on the bus are not applied twice.
- `CameraSource`: processed + dropped equals the number of captured frames.
- `LiveApp` on the replay writes events (chain OK), the downlink, `run.json`, and the raw and
  annotated MP4s.
- Plug-ins load from config; `none` gives no stage; a bad spec raises. A disabled stage reports
  "did not run".
- `VoiceSink` orders an alert before a prompt; the overlay draws on a copy and handles grey input;
  `EventLog`'s clock makes timestamps a function of the input.
