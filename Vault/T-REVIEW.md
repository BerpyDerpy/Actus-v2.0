# T-REVIEW: decisions made outside the build plan

2026-09-29 · covers T-REVIEW (code review of every Sonnet lane, then fixes for the confirmed
findings) · read with `build-plan.md` §5 (Wave 3) and the other `Vault/*.md` notes

The card says: "`/code-review high` per merged branch, then fix the confirmed findings". This file
records how that was done, what each finding turned out to be, and every behaviour that changed, so
that the lane owners and T-INT can rely on the fixes or overturn them on purpose.

**Scope, as agreed:** the harness. Nothing is committed. **No code was run** by me while fixing
(that was the instruction): no pytest, no ruff, no tools. Two review sub-agents did run read-only
checks while reviewing: `ruff format --check` on the T-PKG files, and a one-line PowerShell repro
of the `Invoke-Native` bug. Both are noted where they apply. Nothing below is otherwise verified.

---

## 0. What to run first, and what is most likely to fail

```sh
uv run ruff check . && uv run ruff format --check .     # every edit was formatted by hand
uv run pytest tests/test_perception_camera.py tests/test_perception_handpose.py \
  tests/test_perception_rotation_tta.py tests/test_voice_queue.py tests/test_voice_asr.py \
  tests/test_voice_tts.py tests/test_io_*.py tests/test_ui_server.py tests/test_faults.py \
  tests/test_data_*.py tests/test_record_takes.py tests/test_det_train.py \
  tests/test_proto2_demo.py tests/test_pkg_*.py tests/test_pipeline.py
uv run pytest                                            # then everything
bash -n scripts/setup_media.sh scripts/setup.sh scripts/run_demo_protocol2.sh
(cd ui && npx tsc --noEmit)                              # the four UI files changed
```

Most likely to fail, in rough order:

1. **`ruff format --check`.** About 40 files were edited by hand. Expect a few joined or split
   lines; `ruff format .` fixes them and changes no behaviour.
2. **`tests/test_faults.py`.** The toggle semantics changed (§3, T-FAULT). Existing tests were
   read against the new rules and should still hold, but this is the densest logic change. The new
   test `test_a_failed_network_cut_ends_the_window_instead_of_claiming_an_outage` reaches into
   `ctrl._pool` to wait for the worker.
3. **`tests/test_io_eventlog.py`.** The crash-resume tests write CRLF logs on Windows (the log is
   opened in text mode); `_resume` works on bytes and accepts both endings, but this is the first
   code that truncates a log file.
4. **`tests/test_proto2_demo.py` and every protocol-2 run.** `check_scene_protocol` now refuses a
   scene whose `DEST` is not inside the zone the step names. The T-PROTO2 note says every end point
   is "well inside its zone"; if `Zone.contains` disagrees at an edge, the demo now raises instead
   of running.
5. **`tests/test_perception_rotation_tta.py`.** One expectation changed with the half-pixel fix
   (§3, T-PERC); the real-MediaPipe 180 degree test (skipped without the model files) now runs
   rotation 180 on a cloned estimator.

## 1. How the review was run

- `/code-review high` ran once per lane, as forked sub-agents, over the lane's files as they stand
  (the lanes are merged on `main`, and the working tree is not a per-lane branch). Lanes reviewed:
  T-REC (+ `tools/check_markers.py`), T-PERC, T-VOICE, T-IO (Python, and `setup_media.sh`
  separately), T-UI, T-DATA, T-DET-TRAIN, T-FAULT, T-PROTO2, T-PKG. That is 11 reviews and about 110
  findings.
- The reviews ran without a verification pass. **Every finding was checked by hand against the
  code** before anything was changed. "Confirmed" below means I read the code path and agree that
  the failure scenario happens. It does not mean it was reproduced.
- Opus lanes (P0, T-ENGINE, T-SPATIAL, T-HOI-TEMP, T-INT, T-EVAL) were not reviewed; that was not
  on the card.

## 2. Verdicts at a glance

| Lane | Findings | Fixed | Not fixed |
|---|---|---|---|
| T-PERC | 10 | 9 | 1 (cleanup only) |
| T-VOICE | 10 | 10 | the push-to-talk key is still unread (T-INT's, §5) |
| T-IO `setup_media.sh` | 10 | 10 (by delegating to T-PKG's fetcher) | — |
| T-IO Python | 10 | 10 | — |
| T-UI | 10 | 10 | `create_app(faults: Any)` stays untyped |
| T-DATA | 10 | 10 | `make_splits.py` still lists unreviewed frames (§5) |
| T-REC | 10 | 9 | 1 rejected: layout is deliberately not part of READY |
| T-DET-TRAIN | 10 | 10 | — |
| T-FAULT | 10 | 10 | the stale-frame guard still compares whole frames |
| T-PROTO2 | 10 | 5, plus 1 partial | 3 refactors deferred, 1 rejected |
| T-PKG | 10 | 10 | — |

## 3. Findings and fixes, per lane

### T-PERC (`actus/capture/camera.py`, `actus/perception/*`)

| Finding | Fix |
|---|---|
| A real 0 ms `POS_MSEC` on frame 0 was treated as "missing", so the frame fell back to wall time and `t` could go backwards | 0 ms counts for frame 0. A container with no timestamps uses `frame_idx / container_fps`, as `pipeline.FileSource` does. Wall time is only the last resort, and `Camera.clock` then says `"wall"`. File `t` is forced non-decreasing |
| A live source that failed N reads ended with `error=None`, so a disconnect looked like a clean end | Sets `error = CameraError(...)`. `CameraSource` then raises "camera failed", so a dropped webcam or RTSP feed stops the run loudly |
| Failed live reads retried with no pause (30 attempts in about 1 ms) | New `read_retry_s=0.05`: the budget is now about 1.5 s. It uses `_stop.wait`, so `stop()` stays prompt |
| `/dev/video0` and GStreamer strings were treated as files | Digit strings, `/dev/...` and `... ! ...` are live. A digit string opens as an int index |
| A failed `start()` never set `ended`; `ended` fired before the device was released | Both fixed |
| `RawEstimate.confidence` counted a disabled pose landmarker as "found nothing" | `RawEstimate` carries `hands_enabled` / `pose_enabled`; a disabled stage is left out |
| One VIDEO-mode MediaPipe instance was shared across the probe rotations | Each rotation gets its own `HandPoseEstimator` (`clone()`, created lazily). `RotationTTA.close()` closes them all |
| **MediaPipe's handedness assumes a mirrored image** (MediaPipe's own docs say to swap it for a non-mirrored camera) | New `perception.mirrored_input: false` (in `configs/default.yaml`). By default the label is swapped, so `HandObs.side` is the operator's own hand. **Behaviour change, §4** |
| Landmarks were un-rotated with the pixel-index convention (`w-1`), a 1 px bias | `_unrotate_points` shifts by half a pixel around `unrotate_point`, so continuous coordinates map exactly. `rotate_point` itself is unchanged (it is tested against `cv2.rotate` on the pixel grid) |
| Rotation mapping duplicated in three files | **Not fixed**: a cleanup, with no bug |

### T-VOICE (`actus/voice/*`)

| Finding | Fix |
|---|---|
| One exception in `backend.speak` killed the only player thread for good | Caught, logged, and counted (`VoicePlayer.errors`, `last_error`); the worker carries on |
| Dedup dropped the engine's deliberate re-issues for "repeat", "what's next" and a "no" | `VoiceQueue.forget_recent()` plus `put(..., dedupe=False)`. `VoiceSink` (T-INT's file, §6) now subscribes to `commands` and calls `forget_recent()` on every `Command` and `Answer` |
| An interrupt during synthesis did nothing; a late interrupt could cut off the alert itself | The interrupt hook is now `fn(seq)`: `VoicePlayer` stops the backend only if that item is still playing. `KokoroTTS.stop()` also cancels an utterance still being synthesised |
| A "yes" spoken to question 1 could be recorded as the answer to question 2 | A question change mid-utterance flushes and **discards** the partial |
| The mic was open while Actus read the question aloud (it could answer itself) | `AskGate(mute_while=...)`; `VoiceSink` passes `lambda: player.speaking`. What was heard while muted is discarded |
| Releasing push-to-talk threw away the flushed text | A plain close handles the flushed text, as a `Command` only |
| Vosk decoded inside the PortAudio callback | The callback only queues bytes (bounded, drops counted); a worker thread decodes, and errors never kill the stream. `flush()` uses `Reset()` when Vosk has it |
| pyttsx3 was used across threads (SAPI/COM) | `Pyttsx3TTS` owns one thread that creates and runs the engine. `stop()` sets a flag that a `started-word` callback acts on |
| Eviction removed the newest of equal-priority items, contrary to the docstring | Removes the oldest |
| `start_bus_gate` never closed on a resolving `StepEvent` and leaked its subscription | `AskGate.on_step_event()`; the gate also listens on `events`. `stop()` unsubscribes and joins |

### T-IO (`actus/io/*`, `scripts/setup_media.sh`)

- **`setup_media.sh` is now a thin wrapper around `tools/fetch_models.py --group media`** (T-PKG).
  - The standalone script had drifted from the fetcher. It built the wrong ffmpeg arch names, never
    matched MediaMTX's `arm64v8`, died silently under `pipefail`, leaked a temp dir, did no hash
    check or pinning, and was not idempotent for ffmpeg.
  - The fetcher already does all of this correctly, on all three operating systems, and T-PKG's note
    already said "prefer `fetch_models.py`". One copy is better than two.
  - The wrapper needs `uv`, or a Python with PyYAML. Extra flags pass through (`--force`,
    `--dry-run`).
- `Streamer` passes `bin/mediamtx.yml` to MediaMTX when it exists. MediaMTX reads its config from
  the working directory, not from next to the binary, so the setup scripts' copy was being ignored.
- `ManagedProcess`:
  - the backoff resets only after `stable_s` (10 s) of uptime, so a crash-looping ffmpeg now backs
    off to 30 s instead of respawning every second;
  - `stop()` and the watchdog share the lock, and `_spawn` checks `_stop`, so no orphan is left
    holding a port;
  - `write_stdin` also catches `ValueError` (the pipe was closed by `stop`);
  - `stop()` closes stdin off-thread with a 0.5 s timeout, then allows up to 1 s for a clean EOF
    exit before `terminate()`. This also makes `test_managed_process_write_stdin_reaches_the_child`
    less racy.
- `Streamer._run` survives an exception on one frame (`errors` and `last_error` appear in
  `stats()`).
- `EventLog`:
  - **Crash resume:** a last line cut off mid-write is moved to `events.jsonl.partial`, and the
    chain resumes from the last complete record. A bad record anywhere *before* the end raises
    instead: it is not truncated.
  - **No signal-handler deadlock:** a signal that lands while the same thread holds the lock is
    deferred until the write finishes. `SIG_IGN` is honoured.
  - `close()` unregisters the atexit hook and restores the previous signal handlers.
  - **`append()` after `close()` raises** `RuntimeError`, and `final_hash` is always the true chain
    tip. `pipeline.py` already guards with `not eventlog.closed`.
  - `verify_log(require_end=, expect_hash=)`. **The CLI now requires a final `run_end` by default**
    (`--allow-open` accepts a run still in progress), and it can pin the tip (`--expect-hash`, or
    `--downlink runs/<id>/downlink.json`). Deleting records from the end is now detected. Library
    callers (`evaluate.py`, `run_offline.py`, `faults.py`) keep the old default.
- `encode_downlink` no longer fails a long run. It drops alert messages first, then the oldest
  alerts. When it drops alerts it adds `alerts_total`, `alerts_dropped` and `alert_counts` (per
  kind), and it raises only if the step table alone is over budget.
- `SegmentWriter`: a failed open backs off for 5 s (frames skipped meanwhile are counted as
  `frames_dropped`), the segment number is no longer burned, and errors are capped at the last 50
  (`error_count` keeps the total).

### T-UI (`actus/app/server.py`, `ui/src/*`) and the T-FAULT GUI parts

- **The fault buttons and hotkeys send the state the user wants** (`on: !active`), never a
  server-side toggle. Key repeat is ignored.
- `api.setFault` never rejects. It returns `{ok, error}`, and the panel shows the error.
- `api.faults()` returns `"absent"` on a 404, and polling then stops for good (the routes are fixed
  when the server starts).
- `useBusStore` drops the `faults` stats entry on disconnect, so a finished run never leaves
  "FAULT INJECTED" lit.
- The banner and the panel say **SIMULATED**, and the network target says "(SIMULATED target)".
  `FaultController.stats()` gained `simulated` and `network_simulated`, both after the first three
  keys the stats strip shows.
- `types.ts` `FaultsState` / `FaultInfo` now mirror `controller.state()` (`defaults`, `windows`,
  `network.target`).
- MJPEG uses two subscriptions (annotated first, raw as the fallback); once an annotated frame
  arrives, the raw one is dropped.
- A malformed WebSocket message is logged and skipped; it no longer ends the connection.
- `/api/demo/start` and `/api/demo/stop` hold a lock, so a double click cannot leave two replays
  running.

### T-DATA (`tools/_datalib.py`, `extract_frames.py`, `autolabel.py`, `review_labels.py`, `synth_augment.py`)

- **`synth_augment.py` crops only boxes a human accepted, from training operators only.**
  - It reads `review_status.csv` and `make_splits.py`'s `report.json`, and refuses to run without
    them.
  - `--include-unreviewed` and `--all-operators` opt out, and the report records it
    (`crop_sources`).
  - Before this, val/test operators' boxes were pasted into training data, which leaked the
    held-out split.
- `synthetic.txt` lists exactly the images this run wrote; stale images and labels from a larger
  earlier run are deleted.
- **`autolabel.py` never overwrites a reviewed frame's labels** (`--overwrite-reviewed` to force
  it). The report has `n_frames_skipped_reviewed`.
- `review_labels.py`: `q`/Esc and `b` save the current frame's edits (only space/Enter marks a
  frame *accepted*). Drawn boxes are clipped to the image.
- `write_yolo_label` clips every box to the image and drops a box with nothing inside it.
- `extract_frames.py`:
  - `t` is the container timestamp read while decoding (the old `fi / fps` used a default fps,
    against CLAUDE.md);
  - planned frames past the real end are dropped from the manifest, with a warning, instead of
    crashing `write_manifest`;
  - `--takes` merges into the existing manifest rather than replacing it;
  - `--max-frames` thins sparse frames evenly across all takes.
- New shared helper: `_datalib.read_reviewed(status_path)`.

### T-REC (`tools/record_takes.py`, `tools/check_markers.py`)

- `LabelTrack.undo` restores exactly what one key press changed (history holds the replaced row).
  Before, it could delete an unrelated label, or two at once.
- A labelling pass with no step keys no longer sets `labelled_at`, so the take stays pending and
  `evaluate.py` never scores an all-idle "truth".
- The marker summary is taken when the take ends, before frame counting (which can take seconds
  while the monitor keeps checking).
- `--check-ready`:
  - headless with no `--duration` now checks for 5 s (`CHECK_READY_HEADLESS_S`) instead of looping
    forever;
  - a missing output drive reports FAIL rather than a traceback;
  - `--source2` is opened and must deliver a frame;
  - camera errors are FAIL lines.
- File sources are paced to their own fps (they were decoded as fast as possible).
- The console prints `[SIMULATED]` for `replay:` sources.
- `has_new()` reads only the sequence number (`latest_seq()`), with no rotate per poll.
- **Rejected:** "READY despite off-layout markers". The module docstring and
  `test_marker_off_layout_is_flagged_but_still_ready` make layout deliberately not part of READY
  while `rack.yaml` is a placeholder. The problems list still flags it in orange.

### T-DET-TRAIN (`tools/train_detector.py`)

- `label_path_for()` swaps the `images` path part. The old `str.replace("/images/", ...)` never
  matched on Windows, which was the pre-existing test failure T-INT noted (§7 there).
- Gallery:
  - ground truth is decoded with the dataset yaml's `names`, predictions with the model's own
    `result.names`;
  - only boxes are kept in memory, and the n images shown are re-read;
  - files are named `<rank>_<take>_<frame>.jpg`, and the old gallery is cleared.
- Card links are relative to the card, so any `--out-dir` works.
- **The model card no longer carries numbers from an earlier run.** The old limitations section
  hard-coded 0.221 vs. 0.415, 13 vs. 2559, 0.007, "op05 used orange" and "op01 excluded".
  - Every bullet is now computed: the review coverage (`review_status.csv`), the classes with no
    held-out ground truth, the excluded operators, and the real vs. synthetic comparison.
  - Those run-specific observations belong in a Vault note, not in generated output.
- `--relabel --exclude-operators` now really excludes: the operator's label files are deleted and
  its frames removed from the split lists (`drop_operators_from_splits`). It warns that the
  synthetic set was built from the old labels.
- The full-frame baseline is described as a localization reference, not an mAP floor; the two are
  on different scales.
- `_score_frame` matches the highest-confidence prediction first.
- `n_train_images` is parsed once, as UTF-8, and ignores blank lines.
- New `--synthetic-list`, which builds the real+synthetic yaml with the existing, previously
  uncalled `build_synth_dataset_yaml`.
- The docstring now says only the chosen run is exported to ONNX.

### T-FAULT (`actus/app/faults.py`)

- **New switching rules:**
  - A manual switch overrides the schedule for the rest of the run: `on=False` really turns off a
    fault that a scheduled window holds on.
  - `on=None` toggles the fault's **actual** state, and several toggles in one frame count as one.
  - New params for an active fault restart it with those params (logged as "params changed").
- Ready-made `ScheduledFault` objects are validated when the controller is built, and
  `run_fault_demo` builds the controller before the baseline run, so a bad `--param` fails at once.
- An empty or unparseable value (`gamma=`, `gamma=[1`) is a `ValueError`, never a raw `TypeError`
  or `YAMLError`.
- `spec()` round-trips lists (`region=0,0,0.5,1`) and leaves out `None` params.
- A failed asynchronous network cut ends the window at the next frame, marks the fault refused
  (`cut failed: ...`), and logs `unavailable`, so the log never claims an outage that did not
  happen.
- The POSIX hotkeys read the fd with `os.read`, so two quick keys are both handled.
- `reset()`: attaching the controller to a new core starts a new run.
- CLI: `--window` is validated, and `--param` with `--demo all` is an error.
- Occlusion uses a cached seeded noise tile plus a `frame_idx % 4` stamp, so there is no
  full-frame RNG draw per frame. Consecutive frames still differ, so the stale guard is not fooled.
  The guard's own whole-frame compare was left alone.

### T-PROTO2 (`tools/demo_protocol2.py`, `configs/sample_mixing.yaml`, `scripts/run_demo_protocol2.sh`)

- **Fixed:**
  - The caption moved below the board: at `IMG_H - 70` it ran over ArUco marker 3.
  - `check_scene_protocol()` refuses a protocol without cup/bottle/book, or whose zones no longer
    contain the scene's `DEST`. This is **partial** for the "hard-coded geometry" finding: it
    catches drift rather than deriving `DEST` from the YAML.
  - `run_scenario` closes the log with `reason="error"` if the run raises.
  - The config header and the wrapper point to `tools/fetch_models.py yolo11n_coco` (pinned and
    verified) instead of an ad-hoc Ultralytics one-liner.
  - The wrapper lets a user `--config` win, rejects `replay:` inputs and `--replay`/`--source`, and
    its help range was updated.
- **Deferred (they touch other lanes' files):**
  - letting `pipeline.run_offline` accept a prebuilt source, so the demo stops re-implementing
    about 80 lines of it (T-INT's file);
  - sharing the board/marker renderer with `replay.py` (P0's);
  - caching the static background per frame (performance only).
- **Rejected:** "`t = fi / fps` breaks the clock rule". The scene is a synthetic source with its own
  declared rate, exactly like the P0 replay, which `Vault/Wave0.md` §8 explicitly allows.

### T-PKG (`scripts/setup.*`, `tools/_pkglib.py`, `bundle.py`, `doctor.py`, `fetch_models.py`)

- **`Invoke-Native` pipes the command's output to `Out-Host`.** Before, a function returned
  everything the command printed plus the boolean, so any command that printed a line counted as
  success, and its output was hidden. A review sub-agent reproduced this in PowerShell 5.1. It
  affected `uv sync`, `npm`, `fetch_models` and the doctor gate before `-Demo`.
- `fetch_one` turns `OSError`, `BadZipFile`, `TarError` and `EOFError` into a per-asset `failed`,
  and the other assets still get fetched.
- `extract_selected` writes `<name>.part` and renames it into place, so a failure never leaves a
  half-written binary.
- The ultralytics fallback download is checked against the pin. A download that ends up
  mismatching its pin is `failed`, not `downloaded`.
- `--lock <ids>` pins only those ids, and never re-pins a file that differs from its existing pin
  without `--force`. The "latest" media binaries are exempt, since a newer release is the expected
  reason for a mismatch there.
- `--from` picks the newest archive in natural version order, and ignores `.sha256sum` side files.
- `doctor`:
  - an OpenCV that is installed but fails to import is a FAIL line, not a crash;
  - rack YAML errors are a FAIL line;
  - port checks probe with a connect first (which catches a listener on all interfaces on Windows),
    then bind with `SO_REUSEADDR` on POSIX only.
- The offline `setup.sh` retries the uv wheel install with `--break-system-packages` on
  PEP 668 Pythons (Debian 12, Ubuntu 23.04+). That flag only touches the user site.
- `ruff format` nits in `_pkglib.py` and `bundle.py` are fixed. These were confirmed by the review
  sub-agent's `ruff format --check`.

## 4. Behaviour changes other lanes must know about

- **`HandObs.side` now means the operator's own hand by default** (the MediaPipe label is swapped
  for a non-mirrored camera).
  - Anything trained or cached on the old labels has its left/right slots swapped: T-HOI-TEMP's
    feature caches, and the HOI and temporal weights. Retrain, or set
    `perception.mirrored_input: true` to get the old labels back.
  - The HOI model is per (hand, object) pair, so the effect may be small, but it has not been
    measured.
- **The voice interrupt hook is now `fn(seq)`**, not `fn()`. Only `VoicePlayer` sets it.
- `EventLog.append` after `close()` raises. The `eventlog verify` CLI requires `run_end` by default.
- The downlink packet may carry `alerts_total`, `alerts_dropped` and `alert_counts`.
- `Camera` reports a live disconnect as `error`, so `CameraSource` raises at the end of the stream.
- `setup_media.sh` needs `uv` (or a Python with PyYAML).
- `synth_augment.py` needs `review_status.csv` and `data/splits/report.json`, unless you pass
  opt-out flags.
- Fault switching: a manual switch now overrides the schedule, and toggles use the actual state.
  T-DEMO's walkthrough steps that press fault keys should still read naturally, but check them.

## 5. Not done, and known limits

- **Push-to-talk is still not wired.** `voice.asr.push_to_talk_key` is never read. That is T-INT's
  `VoiceSink` (noted in `Vault/T-INT.md` §4). The ASR side now handles a release correctly.
- `make_splits.py` still lists unreviewed frames in `train.txt`. Whether training may use weak
  labels is a T-DET-TRAIN decision: the card now states the review coverage honestly instead.
- `doctor.check_modules` still uses `find_spec` for the heavy modules (mediapipe, ultralytics).
  Importing each one would make the preflight slow. Only `cv2` is import-checked.
- `create_app(faults: Any)` was left untyped, to avoid a FastAPI import-time dependency on
  `faults.py`.
- The three T-PROTO2 refactors in §3.
- Nothing was run. See §0.

## 6. Edits outside the reviewed lanes' own files

| File | Change | Why |
|---|---|---|
| `actus/app/pipeline.py` (T-INT) | `VoiceSink`: subscribes to `commands` too; `forget_recent()` on every `Command` and `Answer`; the ASR gate is built as `AskGate(mute_while=lambda: self.player.speaking)` | The voice fixes (dedup vs. repeat, the echo) need the sink's wiring. Three small hunks, backwards compatible |
| `configs/default.yaml` (P0/T-PERC section) | `perception.mirrored_input: false` | The handedness fix (§4) |

No contract in `actus/core/contracts.py` changed, no dependency was added, and `uv.lock` is
untouched.
