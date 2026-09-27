# ACTUS v2 — build plan (fresh start)

2026-09-27 · supersedes all Stage 3 documents · inputs: `problem-statment.md`,
`reports/SIH 26174 HAR competitor repos.md`

This plan is written to be executed by background agents. Every task has a card with an ID, the
model to run it on, its dependencies, the files it owns, and a "done when" test. A task card is
meant to be pasted as the agent's prompt, together with §2 (principles) and §4 (contracts).

---

## 1. How we win

About 60 teams have published for SIH26174. Nearly all of them ship the same template: YOLO +
MediaPipe → distance rules → hand-written FSM → pyttsx3 → JSONL → MJPEG → dashboard. Many of the
numbers they report are inflated or fake. So pipeline breadth is table stakes, and we win on four
things the field does not have together:

1. **Every listed deliverable, done literally and visibly.** The deliverables are: live
   processing, next-step prompts, a voice alert on skipped or out-of-order steps, a structured
   timestamped log with outcomes, a **real RTSP push to a configurable IP plus local recording**,
   a GUI, **a trained model we trained ourselves**, and **dataset generation for detection, pose
   and hand-object interaction (HOI)**. Most rivals miss at least one; the most commonly missed
   is streaming to an IP.
2. **ISRO's own sample experiment**, the boxes, as the primary protocol. A second protocol is
   shown from YAML alone to prove the system works for any procedure.
3. **A reasoning layer rivals don't have.** Steps are *state transitions* verified by evidence,
   and that evidence is kept separate from the protocol's expectations, so a skipped step can
   never be "filled in" by the prior. When the system is unsure it **asks by voice and accepts a
   spoken answer**. Every step gets an outcome status, not just "done".
4. **Credibility.** We report numbers on **operators held out of training**, show a live
   validation report card, provide a fault-injection mode that shows graceful degradation, and
   keep an honest limitations list. In a field where the top repo runs on mocks, that matters.

Plus the optional requirement, demonstrated rather than claimed: **orientation-agnostic tracking
relative to the payload rack.** ArUco markers on the rack define a rack coordinate frame, so
every object, hand and body position is expressed relative to the rack, not the floor or the
camera. Pose runs with rotation test-time augmentation, and the detector is trained with
full-360° rotation augmentation. The demo rolls the camera 90° and 180° and tracking holds.

## 2. Principles every agent follows

- **The protocol engine is the judge; models are sensors.** Models emit evidence with
  confidences. Only `actus/protocol/` decides that a step happened.
- **Evidence and expectation are separate channels.** Skip and out-of-order detection use
  evidence only. The protocol prior may only rank the next-step suggestion and break ties.
- **Frame index is the clock.** Every observation carries `frame_idx` plus a monotonic
  `t_capture`. Never derive time from `index / fps` with a default fps.
- **Everything runs offline at runtime.** No cloud API anywhere in the runtime path. A download
  during setup is fine.
- **Honest by construction.** No placeholder metrics, no mock detectors left in any demo path.
  Anything simulated is labelled "SIMULATED" in the UI and the logs.
- **Contracts in `actus/core/contracts.py` are frozen after P0.** A lane that needs a change
  proposes it in its PR description; the integrator (Opus) applies it.
- **Each lane owns its files.** Don't edit another lane's files; stub against the contract.
- Python 3.11, managed with `uv`. MediaPipe does not support the machine's default Python 3.14.
  Lint with `ruff` and test with `pytest`; every lane ships tests.

## 3. Assumptions to confirm

These don't block the start. Correct any that are wrong and the affected cards change.

| # | Assumption | If wrong |
|---|---|---|
| A1 | Dev machine is the M2 Mac. Detector training can run on an RTX 4070 laptop if one is available, and YOLO11n also trains acceptably on the M2 (MPS). | Only T-DET-TRAIN's time changes |
| A2 | **"Working demo soon" = a recorded walkthrough within ~48 h, plus a live-capable system.** The finale (December) gets the polish tier. | Cut to M1 (§7) if the video is due sooner |
| A3 | The team can build props and film at least **4 different operators** within the first ~12 h. | The held-out evaluation weakens to held-out takes; say so honestly |
| A4 | The sample experiment reads as: open outer box → take out red box → place in Zone A → take out yellow box → place in Zone B → close outer box. The official text is truncated at "red and yello…" in every copy, including the NoBugNinja dump. | Edit `protocols/bas_boxes.yaml` only; nothing else changes |

## 4. Architecture and contracts

```
camera / file ──► capture (frame_idx, t) ──┬─► detector (YOLO11n custom | YOLO-World fallback) ─┐
                                           ├─► hands + pose (MediaPipe, rotation TTA) ──────────┤
                                           └─► rack frame (ArUco PnP) ──────────────────────────┤
                                                                                                ▼
                                            spatial: rack coordinates, zones, tracking, world state
                                                   │ predicates (e.g. red.in_zone_A: 0.93)
                         ┌─────────────────────────┼──────────────────────────┐
                         ▼                         ▼                          ▼
               HOI classifier (trained)   temporal action model (trained)   rule predicates
                         └──────────────► StepEvidence (prior-free) ◄──────────┘
                                                   ▼
                  protocol engine: state machine + decision layer (dwell, hysteresis, abstain/ask)
                    │ StepEvent / Alert / Prompt / Ask
     ┌──────────────┼───────────────┬────────────────┬─────────────────┐
     ▼              ▼               ▼                ▼                 ▼
 voice out/in   event log      GUI (FastAPI+WS    RTSP push to IP   local recorder
 (Kokoro/Vosk)  (JSONL, hash-  + React)           (MediaMTX,        (segmented MP4,
                chained,                          raw + annotated)  raw + annotated)
                downlink pkt)
```

**Repo layout** (created in P0):

```
actus/core/        contracts.py  bus.py  config.py
actus/capture/     camera.py  replay.py
actus/perception/  detector.py  handpose.py  rotation_tta.py  tracker.py
actus/spatial/     rack_frame.py  zones.py  world_state.py  predicates.py
actus/models/      hoi.py  temporal.py            (runtime wrappers, ONNX)
actus/protocol/    schema.py  engine.py  decision.py
actus/voice/       tts.py  asr.py  voice_queue.py
actus/io/          streamer.py  recorder.py  eventlog.py  downlink.py
actus/app/         pipeline.py  server.py  faults.py
ui/                React + Vite, built to static files and served by server.py
protocols/         bas_boxes.yaml  sample_mixing.yaml
configs/           default.yaml  rack.yaml
tools/             record_takes.py  extract_frames.py  autolabel.py  review_labels.py
                   make_splits.py  synth_augment.py  train_detector.py  train_hoi.py
                   train_temporal.py  evaluate.py  run_offline.py
data/  models/     (gitignored)
tests/
```

**Contracts** (P0 writes these as frozen dataclasses; this list is the spec):

- `FramePacket(frame_idx, t_capture, image, source_id, rotation_applied)`
- `Detection(track_id, cls, conf, bbox_xyxy, bbox_rack_xy | None)`
- `HandObs(side, landmarks_px[21], landmarks_rack | None, conf)` and
  `PoseObs(landmarks_px[33], world_landmarks[33], torso_vec_rack | None, conf, rotation_used)`
- `RackFrame(valid, H_img_to_rack, rvec, tvec, markers_seen, reproj_err)`
- `Predicates(frame_idx, t, values: dict[str, float], visibility: float)`, where each value is a
  probability, e.g. `outer.open`, `red.in_outer`, `red.held`, `red.in_zone_A`
- `StepEvidence(frame_idx, t, step_id, p_complete, sources: dict[str, float])`, prior-free
- `ActionScores(frame_idx, t, scores: dict[step_id|"idle", float])` from the temporal model
- Engine outputs: `StepEvent(step_id, status, t_start, t_end, confidence, evidence)`,
  `Alert(kind: skip|out_of_order|unexpected|timeout|too_fast, message, step_ids)`,
  `Prompt(next_step_id, text)`, `Ask(question_id, step_id, text)`, `Answer(question_id, yes|no, via)`
- Step status values: `verified`, `operator_confirmed`, `skipped`, `out_of_order`,
  `timed_out`, `pending`
- `bus.py`: an in-process pub/sub with bounded queues per subscriber that counts drops
  explicitly. Topics: `frames`, `detections`, `hands`, `pose`, `rack`, `predicates`,
  `evidence`, `actions`, `events`, `alerts`, `prompts`, `asks`, `answers`, `stats`.

**Protocol YAML** (`protocols/bas_boxes.yaml`). Steps are pre-/post-condition transitions over
predicates:

```yaml
id: bas_boxes_v1
title: "BAS sample experiment: box transfer"
zones: {A: [[0.55,0.10],[0.80,0.35]], B: [[0.55,0.45],[0.80,0.70]]}   # rack coords, metres
steps:
  - {id: S1, text: "Open the outer box",             post: {outer.open: true}}
  - {id: S2, text: "Take out the red box",           pre: {red.in_outer: true},    post: {red.in_outer: false},    requires_hoi: red}
  - {id: S3, text: "Place the red box in Zone A",    post: {red.in_zone_A: true}}
  - {id: S4, text: "Take out the yellow box",        pre: {yellow.in_outer: true}, post: {yellow.in_outer: false}, requires_hoi: yellow}
  - {id: S5, text: "Place the yellow box in Zone B", post: {yellow.in_zone_B: true}}
  - {id: S6, text: "Close the outer box",            post: {outer.open: false}}
order: strict            # or a partial order: [[S1,S2],[S1,S4],...]
timing: {dwell_s: 0.8, timeout_s: 60, min_duration_s: {S3: 1.0}}
decision: {p_enter: 0.8, p_exit: 0.6, ask_band: [0.45, 0.8], ask_after_s: 3.0}
```

## 5. Task board

Model key: **[OPUS]** = ambiguous spec, cross-cutting design, subtle state or timing logic, ML
methodology, integration and review. **[SONNET]** = well-specified module against a frozen
contract, tooling, UI, IO, scripts, packaging. **[HUMAN]** = physical-world work that agents
can't do.

Parallel groups: tasks in the same wave with no dependency between them run at the same time,
each in its own git worktree on its own branch. The integrator merges them.

### Wave 0 — start immediately, all in parallel (T+0 → T+2h)

**P0 · Repo skeleton + contracts · [OPUS] · blocks Waves 1–3**
- Owns `pyproject.toml`, `actus/core/*`, `actus/protocol/schema.py`, `protocols/bas_boxes.yaml`,
  `configs/*`, `CLAUDE.md` (the agent rules from §2),
  `tests/test_contracts.py`, and CI with ruff + pytest.
- Also writes the `replay` source: a synthetic predicate stream generator that scripts
  scenarios (correct, skip S2, S4 before S2, unsure S3, occluded) with noise. It lets every
  downstream lane test without cameras.
- Done when: `uv sync && pytest` passes on the M2; the protocol YAML loads and validates;
  `python -m actus.capture.replay --scenario skip_S2` prints a predicate stream.

**T-REC · Take recorder + live step annotator · [SONNET] · no dependencies (uses a plain
CSV convention)**
- `tools/record_takes.py`: records the webcam (and optionally a second camera or a phone RTSP
  feed) to `data/raw/{operator}_{scenario}_{n}.mp4` at a fixed fps, with an on-screen overlay.
- The operator's helper presses `1`–`6` at each step start, `0` for idle and `x` for a
  non-step action. It writes a sidecar CSV `frame_idx,t,label`.
- **Playback mode:** `--label existing.mp4` plays a file already recorded (pause, step, 0.5×)
  and takes the same keys. This is the fallback when takes were shot with a phone camera app.
- Key map, file naming and step-boundary rules follow the recording protocol doc (see H-FILM).
- The session metadata JSON records operator id, scenario, camera roll angle, lighting and a
  notes field.
- Done when a 30 s test take produces an mp4 and a CSV whose frame indices match
  `ffprobe`'s frame count.

**H-PROPS · Build the rig · [HUMAN] · no dependencies**
- **Boxes:** an outer box with a hinged or removable lid, a red box and a yellow box, each
  small enough to hold in one hand. Use matte colours, not glossy.
- **"Payload rack" board:** a rigid board behind or under the work area. Put 4 ArUco markers
  (DICT_4X4_50, ids 0–3, 60 mm) at the corners, printed by T-RACK's generator or any online
  generator. Mark Zone A and Zone B with tape.
- **Camera:** a webcam on a fixed mount looking at the rack. Optionally, a phone as a second
  view.
- Measure the marker-centre positions in mm and put them in `configs/rack.yaml`.

### Wave 1 — parallel after P0 (T+2h → ~T+14h)

| ID | Task | Model | Needs |
|---|---|---|---|
| T-ENGINE | Protocol engine + decision layer | OPUS | P0 |
| T-SPATIAL | Rack frame, zones, world state, predicates | OPUS | P0 |
| T-PERC | Capture, detector wrapper, hands/pose, rotation TTA, tracker | SONNET | P0 |
| T-VOICE | TTS + bounded voice queue + offline ASR | SONNET | P0 |
| T-IO | RTSP push, recorder, hash-chained log, downlink packet | SONNET | P0 |
| T-UI | FastAPI + WebSocket server and React dashboard | SONNET | P0 |
| T-DATA | Frame extraction, auto-labelling, review tool, splits, synthetic augmentation | SONNET | P0, T-REC |
| H-FILM | Film the dataset | HUMAN | T-REC, H-PROPS |

Up to 7 agents run at once here. If you're limited to fewer, the priority order is: T-ENGINE,
T-SPATIAL, T-PERC, T-IO, T-VOICE, T-DATA, T-UI.

**T-ENGINE · [OPUS]** · owns `actus/protocol/engine.py`, `decision.py`, `tests/test_engine*.py`
- A state machine over the protocol YAML. It consumes `StepEvidence` and `ActionScores` and
  emits `StepEvent`, `Alert`, `Prompt` and `Ask`.
- **Decision layer:**
  - Per step, `p_complete` must cross `p_enter` and hold for `dwell_s` (hysteresis with
    `p_exit`).
  - If `p_complete` sits in `ask_band` for `ask_after_s`, or visibility collapses during an
    expected step, emit `Ask("I couldn't confirm the red box is in Zone A. Did you place
    it?")`.
  - A yes `Answer` → `operator_confirmed`. A no → re-prompt that step.
- **Evidence/expectation split:**
  - Completion is decided from evidence only.
  - The expected-next step (the prior) only chooses what `Prompt` says, and breaks ties when
    two steps' evidence is within ε.
  - It never raises `p_complete`.
- **Alerts:**
  - `out_of_order`: a later step is verified while an earlier required step is pending.
    Message: "Out of sequence: yellow box removed before red box. Expected: take out the red
    box."
  - `skip`: a step's successor is verified and the step itself was never verified. Marked
    retroactively in the log.
  - `unexpected`: a transition not in the protocol, e.g. the red box goes back into the outer
    box.
  - `timeout` and `too_fast` from `timing`.
- A prompt is issued at start and after every verified or confirmed step.
- Done when the replay tests pass for all 5 P0 scenarios with exact expected event sequences,
  plus property tests: no step ever becomes `verified` without evidence above `p_enter`; the
  prior never changes a completion; each alert fires exactly once per incident.

**T-SPATIAL · [OPUS]** · owns `actus/spatial/*`, `tests/test_spatial*.py`
- `rack_frame.py`: ArUco detection → solvePnP against `configs/rack.yaml`. Also the
  image→rack-plane homography, validity checks (≥3 markers, reprojection error), and temporal
  smoothing. The last good pose is held when markers are briefly occluded.
- Also a generator that writes printable marker sheets, `tools/make_markers.py`.
- `zones.py`: point-in-zone tests in rack coordinates.
- `world_state.py`: a tracked object-state estimator.
  - Each object carries a location belief {in_outer, held, zone_A, zone_B, elsewhere,
    unknown}, updated from detections (projected to rack coordinates), hand proximity and HOI
    output.
  - Occlusion logic: an object that disappears while held stays `held`, not `gone`. An object
    that disappears inside the outer box's footprint while the box is open becomes `in_outer`.
- `predicates.py`: world state → `Predicates` with calibrated probabilities and a
  `visibility` score. Also `StepEvidence` for each step's post-condition, with `sources`
  broken down.
- The body relative to the rack: pose world landmarks → torso vector expressed in rack
  coordinates, published for the UI and the log.
- Done when:
  - unit tests on synthetic geometry pass, including a camera rolled 90°/180° giving identical
    rack-coordinate positions within 1 cm;
  - an occlusion test passes: the held object stays `held` through a 1.5 s dropout.

**T-PERC · [SONNET]** · owns `actus/capture/camera.py`, `actus/perception/*`,
`tests/test_perception*.py`
- `camera.py`: a threaded capture of a webcam, a file or an RTSP input.
  - The monotonic `frame_idx` comes from the capture thread.
  - File mode: EOF ends the stream and never loops or reopens.
  - Frames are published to the bus.
- `detector.py`: a single `Detector` interface with two backends.
  - `custom`: a YOLO11n `.pt` or `.onnx` from T-DET-TRAIN.
  - `open_vocab`: YOLO-World with the prompts "red box", "yellow box", "cardboard box", "box
    lid". This lets the whole pipeline run before training finishes, and it is the fallback.
  - Uses the ultralytics tracker (ByteTrack) for `track_id`.
- `handpose.py`: MediaPipe Tasks Hand Landmarker (2 hands) and Pose Landmarker (lite), in
  VIDEO mode, with world landmarks.
- `rotation_tta.py`:
  - When mean landmark confidence is below a threshold, or every N frames, run
    hands/pose at 0/90/180/270°.
  - Keep the rotation with the best confidence, rotate the landmarks back to the original image
    coordinates, and record `rotation_used`.
  - Cache the winning rotation so it isn't re-searched every frame.
- Done when:
  - `sanity` scripts run live at ≥15 FPS on the M2 with the detector + hands + pose;
  - the test passes: a frame rotated 180° yields landmarks within 3% of the upright result
    after the back-rotation.

**T-VOICE · [SONNET]** · owns `actus/voice/*`, `tests/test_voice*.py`
- `tts.py`: Kokoro-82M via `kokoro-onnx` (Apache-2.0, offline, natural voice), falling back
  to `pyttsx3`. It runs in its own process or thread.
- `voice_queue.py`:
  - A bounded priority queue ordered alert > ask > prompt > info.
  - It deduplicates identical text within 5 s and drops messages that are stale (older than
    their TTL).
  - An alert interrupts a prompt that is playing.
- `asr.py`: Vosk small English model with a restricted grammar ("yes", "no", "next",
  "repeat", "what's next", "confirm", "cancel").
  - It listens only while an `Ask` is open, or when push-to-talk is pressed.
  - Recognised speech is published as `Answer`, or as a `command` for next/repeat.
- Done when a scripted burst of 20 prompts + 3 alerts plays in under 15 s with every alert
  spoken, and a spoken "yes" is recognised offline with the network off.

**T-IO · [SONNET]** · owns `actus/io/*`, `scripts/setup_media.sh`, `tests/test_io*.py`
- `streamer.py`:
  - Manages a bundled **MediaMTX** binary plus an **FFmpeg** process that pushes H.264 RTSP
    to `rtsp://<target_ip>:8554/actus/raw` and `/actus/annotated`, with the IP set in config.
  - The same MediaMTX serves WebRTC for the GUI.
  - Reconnect with backoff; stream health is published to `stats`.
- `recorder.py`: segmented MP4 (60 s segments) of raw and annotated video to
  `runs/<run_id>/video/`, with frame-index sidecars.
- `eventlog.py`:
  - JSONL at `runs/<run_id>/events.jsonl` with `schema_version`, ISO timestamp, `frame_idx`,
    event type, `step_id`, status, confidence and evidence breakdown.
  - Each line carries `prev_hash` (SHA-256 chain), making the log tamper-evident.
  - A `run_end` summary is written, even on Ctrl-C (atexit + signal handler).
- `downlink.py`: a ≤4 KB JSON "downlink packet" per run: the step outcome table, alerts,
  timings and the log hash. It is the bandwidth-restricted artefact the problem statement
  implies.
- `scripts/setup_media.sh` downloads ffmpeg (via brew on macOS) and MediaMTX.
- Done when:
  - VLC on a second machine plays the RTSP stream at the configured IP;
  - the MP4 segments are playable;
  - a log verification script detects a one-byte edit;
  - the downlink packet is under 4 KB for a 6-step run.

**T-UI · [SONNET]** · owns `actus/app/server.py`, `ui/*`
- FastAPI serves the built React app and a WebSocket relaying bus topics. The UI is
  single-page, dark "mission console" style.
- **Live pane:** annotated video (WebRTC from MediaMTX, MJPEG fallback) with boxes, hand
  skeletons and the zones overlaid.
- **Procedure checklist:** step, status chip, confidence bar, duration.
- **Next-step card**, large.
- **Alert banner** with a sound.
- **Ask card:** the question with Yes/No buttons, which also accepts a spoken answer.
- **Rack mini-map:** a top-down view of the rack with Zone A/B, the live object positions and
  the torso orientation arrow. This visualises "relative to the rack, not the floor".
- **Event log tail** and a **stats strip:** FPS per stage, dropped frames, stream status,
  recording status and the active detector backend.
- **Validation tab:** renders `runs/eval/report.json` from T-EVAL.
- Built against a mock WebSocket feed from the P0 replay, so it doesn't wait on the other
  lanes.
- Done when the dashboard runs from replay for all 5 scenarios, is usable at 1366×768, and
  `npm run build` output is served offline.

**T-DATA · [SONNET]** · owns `tools/extract_frames.py`, `autolabel.py`, `review_labels.py`,
`make_splits.py`, `synth_augment.py`
- `extract_frames.py`: samples frames around labelled step boundaries (dense) and elsewhere
  (sparse), about 1,500 frames in total.
- `autolabel.py`: YOLO-World (or GroundingDINO) proposes boxes for the 5 classes
  `outer_box_closed`, `outer_box_open`, `lid`, `red_box`, `yellow_box`, and writes YOLO
  format.
- `review_labels.py`: a fast OpenCV reviewer.
  - Keys: accept, delete box, relabel, redraw.
  - Target throughput: 400 frames/hour per human.
- `make_splits.py`: **splits by operator** (train, val, test = held-out operators), never by
  frame. Leakage checks:
  - the same take never lands in two splits;
  - print duration statistics per scenario; if take length alone predicts the scenario, warn.
- `synth_augment.py`: the synthetic data generation story.
  - Full-360° rotations.
  - Copy-paste of segmented box crops onto public-domain ISS interior backgrounds (NASA
    imagery), with scale, lighting and blur randomisation.
  - Output as a separate `synthetic` split, so it can be ablated.
- Done when the dataset YAML trains for one epoch without errors and the split report shows
  zero operator overlap.

**H-FILM · [HUMAN]** · follows the **ACTUS recording protocol** doc
(https://claude.ai/code/artifact/125c36a0-8648-407f-9422-85433b47786c): 5 operators ×
(14 Task A takes + a sweep clip + a distractor clip), plus 10 Task B takes, which comes to about
90 clips and about 90 minutes of footage.
- op05 is held out.
- The minimum viable fallback is 3 operators × 10 takes.
- Filming does not wait for T-REC: phone-recorded takes are labelled afterwards in playback
  mode.

### Wave 2 — models and integration (starts when footage exists, ~T+10h → T+36h)

| ID | Task | Model | Needs | Parallel with |
|---|---|---|---|---|
| T-DET-TRAIN | Train the custom detector | SONNET | T-DATA, H-FILM, human review | T-HOI-TEMP, T-INT |
| T-HOI-TEMP | HOI classifier + temporal action model | OPUS | T-PERC, T-SPATIAL, H-FILM | T-DET-TRAIN, T-INT |
| T-INT | Integration: `pipeline.py`, end to end | OPUS | all of Wave 1 | both above |
| T-EVAL | Evaluation harness + report card | OPUS | T-ENGINE, P0; real runs from T-INT | T-INT (starts on replay) |
| T-FAULT | Fault-injection mode | SONNET | T-INT's pipeline hooks | T-EVAL |

**T-DET-TRAIN · [SONNET]** · owns `tools/train_detector.py`, `models/detector/`
- Train YOLO11n from COCO weights at 640 px:
  - with `degrees=180` and flips on;
  - once on real data only, and once on real + synthetic;
  - early stopping.
- Evaluate on the **held-out operator** split: mAP50, mAP50-95 and per-class results.
- Export to ONNX. Write `models/detector/card.md` with the data counts, split and metrics,
  including an honest failure-case gallery.
- Done when the held-out mAP50 is reported, whatever its value. It must be ≥0.85 to replace
  the open-vocab backend by default.

**T-HOI-TEMP · [OPUS]** · owns `actus/models/*`, `tools/train_hoi.py`, `tools/train_temporal.py`
- **Offline feature extraction** over all takes with the real perception stack. Per frame: hand
  landmarks in rack coordinates, the object boxes in rack coordinates, hand-to-object
  distances and their velocities, the upper-body pose. Cache the features keyed by frame
  index.
- **HOI classifier:** per (hand, object) pair it predicts {none, touching, holding}.
  - The training labels are weak labels. They come from the step annotations plus geometry;
    for example, between the S2 start and the S3 end the red box is held.
  - Hand-verify 200 test frames.
  - Model: a small MLP or GBDT, exported to ONNX or treelite.
- **Temporal action model:** a 1D-TCN (or GRU) over 2 s feature windows, predicting the step
  in progress ∪ idle. This is the "HAR" model the problem statement asks for.
  - Split by operator; seed-averaged.
  - Report frame accuracy, segmental F1@{10,25,50} and a confusion matrix.
  - Baselines to beat: majority class and duration only.
- Both models plug in as evidence sources; the engine stays the judge.
- Done when both models export and run in real time, and both model cards report held-out
  numbers next to the baselines.

**T-INT · [OPUS]** · owns `actus/app/pipeline.py`, `tools/run_offline.py`, `scripts/run_demo.sh`.
This is the integrator role: it merges Wave 1 branches and resolves contract proposals.
- Wires capture → perception → spatial → models → engine → voice / log / UI / stream / record.
- Stage FPS and drops go to `stats`.
- `run_offline.py` processes a recorded take deterministically (same output every run),
  including overlays, so the walkthrough can never fail live.
- Tunes the decision thresholds on training-operator takes only.
- Done when:
  - `scripts/run_demo.sh --live` runs the full box experiment at ≥15 FPS on the M2 with voice,
    GUI, RTSP and recording;
  - `run_offline.py` reproduces identical `events.jsonl` across two runs.

**T-EVAL · [OPUS]** · owns `tools/evaluate.py`, `runs/eval/`
- Replays every held-out take through `run_offline.py` and compares the events with the
  annotations. Metrics:
  - step-completion precision and recall at ±2 s;
  - skip and out-of-order alert precision and recall;
  - false alerts per minute on correct takes;
  - median detection latency;
  - ask rate, and ask-then-correct rate.
- Writes `report.json` (for the UI) and `report.md` (for the pitch), with the take-level table
  and the definitions of every metric.
- The success criteria are written down **before** the first real numbers exist.
- Done when the report generates from the held-out split, and every number in it can be
  traced to a take.

**T-FAULT · [SONNET]** · owns `actus/app/faults.py`
- Hotkeys and GUI toggles:
  - occlude the camera (black box over the region);
  - dim lighting (gamma);
  - drop the detector;
  - freeze a frame;
  - kill the network target.
- Each fault is shown in the UI as "FAULT INJECTED". The expected behaviour is abstain or ask,
  or "stream down, still recording", never a false "verified".
- Done when each fault is demonstrated on a replayed take with the expected safe behaviour
  logged.

### Wave 3 — demo-complete and pitch (~T+36h → T+48h)

| ID | Task | Model | Parallel |
|---|---|---|---|
| T-PROTO2 | Second protocol from YAML only (`sample_mixing.yaml`: cup, bottle, book; standard COCO classes; Task B in the recording doc) + a demo run | SONNET | yes |
| T-PKG | One-command setup and run, an offline bundle (weights, Vosk and Kokoro models, MediaMTX), README with a hardware table | SONNET | yes |
| T-REVIEW | Code review of all Sonnet lanes: `/code-review high` per merged branch, then fix the confirmed findings | OPUS | yes |
| T-DEMO | Walkthrough script, shot list and limitations slide text, drafted from `report.md` | SONNET drafts, HUMAN records | after T-EVAL |
| H-REC | Record the walkthrough, plus a fallback screen recording as soon as anything works end to end | HUMAN | continuous |

**Walkthrough (3–4 min):**
1. Architecture in one diagram, 15 s: "the models are sensors; the protocol engine is the
   judge".
2. The live correct run: next-step voice prompts, checklist, rack mini-map.
3. Out of order: yellow before red, and the voice alert fires.
4. Skip: the retroactive skip is logged.
5. The ask path: the helper occludes the placement, ACTUS asks "Did you place it?", the
   operator says "yes", and the step is logged `operator_confirmed`.
6. Camera rolled 180° mid-run: the mini-map and zones don't move.
7. The fault-injection montage.
8. The artefacts: the hash-chained log with the tamper check, the downlink packet at under
   4 KB, VLC on a second laptop playing the RTSP stream, the recorded MP4s.
9. A second protocol loaded from YAML.
10. The Validation tab with held-out-operator numbers, and the limitations.

### Wave 4 — finale tier (after the demo; October–December)

| ID | Task | Model |
|---|---|---|
| F-3D | Full 3D body in rack coordinates: pose world landmarks + PnP, and a body-orientation readout relative to the rack in the UI | OPUS |
| F-HMR | Time-boxed spike: an offline HMR model (e.g. HMR2.0) on recorded takes as an "analysis view". Note the SMPL non-commercial licence. Keep it only if it runs above 5 FPS on the demo machine. | OPUS |
| F-ORBIT | Robustness test on public NASA/ESA astronaut footage: hands/pose + rotation TTA on real microgravity clips, as a qualitative figure | SONNET |
| F-EDGE | ONNX Runtime/CoreML/TensorRT export and an FPS table on 2–3 machines; a Jetson Orin build if one is available | SONNET |
| F-DATA+ | More operators and a second camera view; retrain; publish the dataset and model card on Hugging Face (no team has done this) | SONNET + HUMAN |
| F-CAL | Temperature calibration of every evidence source and a reliability plot | OPUS |

## 6. Critical path and parallelism at a glance

```
T+0   P0[OPUS] ─────┐        T-REC[SONNET] ──► H-FILM[HUMAN] ──────────┐
      H-PROPS[HUMAN] ────────────────────────┘                          │
T+2         ├─ T-ENGINE[OPUS]   ┐                                       │
            ├─ T-SPATIAL[OPUS]  │                                       │
            ├─ T-PERC[SONNET]   │                                       ▼
            ├─ T-VOICE[SONNET]  ├─ all parallel ─► T-INT[OPUS] ◄── T-DATA[SONNET] ─► review[HUMAN]
            ├─ T-IO[SONNET]     │                   │                        │
            └─ T-UI[SONNET]     ┘                   │               T-DET-TRAIN[SONNET]
T+12                                                │               T-HOI-TEMP[OPUS]
T+24  ── M1: live loop works end to end (open-vocab detector) ──────────────┘
T+36  T-EVAL[OPUS] ∥ T-FAULT[SONNET] ∥ T-PROTO2[SONNET] ∥ T-PKG[SONNET] ∥ T-REVIEW[OPUS]
T+48  ── M2: demo-complete, walkthrough recorded ──
```

**The long pole is physical:** props → filming → label review → training. Every agent lane is
designed to proceed without it (replay streams, the open-vocab detector), so filming never
blocks code, and code never blocks filming.

**Peak concurrency:** 7 agents in Wave 1 (2 Opus + 5 Sonnet), 4–5 in Wave 2.

**Opus sessions:** keep them to P0, T-ENGINE, T-SPATIAL, T-HOI-TEMP, T-INT, T-EVAL and
T-REVIEW. The integrator should be **one long-lived Opus session** that owns merges.

## 7. Milestones and cut lines

| Milestone | Contains | If time runs out, cut in this order |
|---|---|---|
| **M1 (~T+24h)** | Live box experiment with the open-vocab detector, the engine with alerts, voice prompts and alerts, the JSONL log, recording, RTSP push, the GUI | — this is the floor; don't cut below it |
| **M2 (~T+48h)** | + custom detector, HOI + temporal models, the ask path with a voice answer, the rotated-camera demo, the held-out report card, fault injection, protocol 2, the recorded walkthrough | 1. protocol 2 → 2. synthetic augmentation → 3. fault injection → 4. the temporal model (keep HOI) → 5. voice *input* (keep GUI Yes/No) |
| **Finale** | Wave 4 | — |

**Never cut:** RTSP push + recording, the ask path, held-out evaluation, the limitations slide.

## 8. How to run the agents

- One git worktree per task: `git worktree add ../actus-<ID> -b <ID>`. Start the agent in it
  with `--model opus` or `--model sonnet` per the card. The agent's prompt is: the card + §2 +
  §4 + "Read CLAUDE.md first."
- Every lane opens a PR against `main`. The Opus integrator merges in this order: P0 →
  T-ENGINE/T-SPATIAL → T-PERC → T-IO/T-VOICE → T-UI → the rest. It rebases later lanes as
  needed.
- A Sonnet lane that stalls for more than ~45 min on a design question (not a bug) escalates:
  hand that question to the Opus integrator rather than switching the lane's model.
- Humans keep a single `STATUS.md` with one line per task: owner, state, blocker.

## 9. Honesty list (goes on the limitations slide as-is)

- The step logic is a protocol state machine over trained perception. The HAR model provides
  evidence and does not decide alone.
- The dataset is small: N operators, ~1 h of footage, filmed on Earth. The numbers are on
  held-out operators, not held-out environments.
- The rotated-camera test emulates a lack of fixed "up"; it is not microgravity footage.
- The orientation-agnostic claim covers the rack frame, rotation augmentation and TTA. It is
  not full 3D HMR unless F-HMR ships.
- The sample-experiment steps are our reading of a truncated official description.
- Throughput is measured on the M2 (and the other listed machines), not on flight hardware.

## 10. Operational

- Both GitHub repos (`Actus`, `Actus-v2.0`) currently return 404 to anonymous requests, so they
  are private. Keep them private until submission.
- Idea submission closes **30 Sep 2026**. If the idea video must show the demo, M1 is the target
  for it.
- Licences to state in the README:
  - Ultralytics is AGPL-3.0.
  - MediaPipe, Kokoro, Vosk and MediaMTX are permissive (Apache/MIT).
  - FFmpeg is LGPL/GPL depending on the build.
  - SMPL is non-commercial; this only matters if F-HMR ships.
