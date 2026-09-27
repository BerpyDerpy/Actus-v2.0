# Wave 0: decisions made outside the build plan

2026-09-27 · covers P0 (repo skeleton, contracts, replay) and T-REC (take recorder) · read with
`build-plan.md`

The build plan specifies Wave 0 only partly. This file lists every decision made during
implementation that the plan does not state, so that later waves can rely on them or overturn
them on purpose. Where a decision is still provisional, it says so.

Status at handover: all 93 tests pass, and `ruff check` and `ruff format --check` are clean.
Nothing is committed. H-PROPS (the physical rig) is human work and was not done.

---

## 1. Environment and verification

- **The dev machine is Windows 11, not the M2 (assumption A1 is wrong for this workspace).**
  Every "done when" check ran on Windows. The M2-specific checks are still unverified there.
- `uv` was installed with `pip install --user uv`. The binary is at
  `C:\Users\nandu\AppData\Roaming\Python\Python313\Scripts\uv.exe`, which is **not on PATH**.
  Python 3.11.16 is managed by uv, and the project venv is `.venv`.
- The system Python is 3.13, not the 3.14 the plan mentions.
- The console is cp1252. **Keep printed CLI output ASCII**; a `→` in a `print` crashed a tool.
- There is no ffmpeg or ffprobe on PATH. T-REC's frame-count check was verified with an
  ephemeral ffmpeg: `uv run --no-project --with imageio-ffmpeg`. It counted 450 frames and
  30.00 s, matching the CSV. T-IO's `setup_media.sh` should install the real binaries.
- The webcam works: index 0, MSMF backend, 1280×720 at about 30 fps.
- **CI runs on Ubuntu only**, because macOS runners cost 10× the minutes on a private repo. CI
  runs `uv sync --locked`, `ruff check`, **`ruff format --check`**, and `pytest`. It installs
  `libgl1` and `libglib2.0-0` for OpenCV.

## 2. Packaging (`pyproject.toml`, `uv.lock`)

- `requires-python = ">=3.11,<3.13"`, and `.python-version` is `3.11`.
- The base dependencies are `numpy`, `pyyaml`, `pydantic>=2.7` and **`opencv-contrib-python`**
  (not `opencv-python`). The contrib build is a superset that includes `cv2.aruco`, and it is
  what mediapipe pulls in.
- **Each lane's heavy dependencies are pre-declared as optional extras**, so that lanes don't
  fight over `uv.lock`. They are already resolved in the lock:
  - `perception`: ultralytics, mediapipe, onnxruntime;
  - `voice`: kokoro-onnx, pyttsx3, vosk, sounddevice;
  - `ui`: fastapi, `uvicorn[standard]`.

  Install one with `uv sync --extra <name>`. The resolved versions are mediapipe 1.0.1 and
  numpy 2.x.
- The dev group is pytest, **hypothesis** (for T-ENGINE's property tests) and ruff.
- Ruff settings: line length 100, rules `E F W I B UP SIM RUF`. `RUF001–003` are ignored so that
  unicode is allowed in strings and comments. `ui`, `data`, `models` and `runs` are excluded.
- The build backend is hatchling, packaging `actus`.

## 3. Repo layout

- `actus/*/__init__.py` exists for every package in the plan, but **there are no stub modules**
  for other lanes' files. Each lane creates its own files, which avoids merge conflicts.
- The `.gitignore` patterns are **anchored to the root** (`/data/`, `/models/`, `/runs/*`,
  `/bin/`). An unanchored `models/` would have ignored `actus/models/`. `!/runs/eval/` is kept
  for T-EVAL, and `/bin/` is reserved for the MediaMTX binary.
- `README.md` was left untouched, because T-PKG owns it. `STATUS.md` was not created, because
  humans own it.
- Tests besides `tests/test_contracts.py`: `test_bus.py`, `test_config.py`,
  `test_protocol_schema.py`, `test_replay.py` and `test_record_takes.py`.

## 4. Contracts (`actus/core/contracts.py`): additions to the §4 spec

**General**
- Every contract is `@dataclass(frozen=True, slots=True)`. Its `__post_init__` validates and
  normalises the input:
  - probabilities must be in [0, 1], and NaN is rejected;
  - `frame_idx` must be a non-negative int (bool, float and str are rejected);
  - every `t` must be finite;
  - rotations must be in {0, 90, 180, 270};
  - lists become tuples;
  - arrays become read-only copies with a checked shape and dtype;
  - dict fields become private copies;
  - strings become enums.

  Invalid input raises `ValueError`.
- `FramePacket.image` is **not copied** (for performance) and not validated beyond having 2 or 3
  dimensions.
- Types that hold arrays use `eq=False`: `FramePacket`, `HandObs`, `PoseObs` and `RackFrame`.
- There is a **JSON codec**. `to_jsonable(obj)` and `from_jsonable(data)` produce dicts tagged
  `"type": "<ClassName>"`. Enums become their values, arrays become lists, and NaN or inf become
  `null`, so the output is strict JSON. `FramePacket` is deliberately not encodable. The event
  log and the UI WebSocket should use this codec.
- Other module-level names: `SCHEMA_VERSION = 1`, `IDLE = "idle"`, `OTHER = "other"`,
  `ROTATIONS`, `ENGINE_OUTPUTS` and `contract_types()`.
- **Clock conventions** (in the module docstring): `t` is seconds on the pipeline clock. That
  is `time.monotonic()` for a live source, the container timestamp for a file, and synthetic
  seconds for replay. Wall-clock ISO time is added only by the event log.

**New types**
- **Per-frame batch types**, which the plan does not define:
  - `DetectionSet(frame_idx, t, detections, backend)`;
  - `HandSet(frame_idx, t, hands, rotation_used=0)`;
  - `PoseSet(frame_idx, t, poses)`.

  These are what the `detections`, `hands` and `pose` topics carry. The per-object types from
  the plan have no frame index, and a batch also makes "nothing detected this frame" explicit.
- **`Command(name, via, *, t, frame_idx=None)`**, with names `next`, `repeat`, `whats_next`,
  `confirm` and `cancel`. T-VOICE's non-answer speech needs somewhere to go.
- **`Stats(source, values, *, t, frame_idx=None)`**, the payload of the `stats` topic.
- Enums: `StepStatus`, `AlertKind`, `AnswerValue` (yes, no), `InputVia` (voice, gui),
  `CommandName` and `Topic`.

**New fields and field semantics**
- **Engine outputs carry the clock as required keyword-only fields.** `Alert`, `Prompt` and
  `Ask` require `frame_idx` and `t`. `StepEvent` requires `frame_idx` and `t` too, and adds
  `retroactive: bool = False`, which a retroactive `skipped` uses. For `StepEvent`, `frame_idx`
  and `t` mean *when the engine decided*. `t_start` and `t_end` can be `None`, for example on a
  skipped step.
- `Answer` names its yes/no field **`value`**. It requires `t` as a keyword, and `frame_idx` is
  optional (the receiver stamps it).
- `Prompt.next_step_id` can be `None` once the procedure is complete.
- `RackFrame` gets keyword-only fields `frame_idx`, `t` and **`age_frames`**. `age_frames` is 0
  for a fresh marker solve and counts up while the last good pose is being held through an
  occlusion. A valid `RackFrame` must have `H_img_to_rack`. `H` is (3, 3) float64, and `rvec`
  and `tvec` are (3,).
- `Detection`:
  - `bbox_xyxy` is a float tuple, and x1 ≤ x2, y1 ≤ y2 is checked;
  - `bbox_rack_xy` is **the box centre projected onto the rack plane**;
  - `track_id` may be `None`.
- `HandObs`: `side` must be `"left"` or `"right"`. `landmarks_px` is (21, 2) pixels **in the
  original, un-rotated image**, and `landmarks_rack` is (21, 3).
- `PoseObs`: `landmarks_px` is (33, 2), `world_landmarks` is (33, 3), and `torso_vec_rack` is
  (3,), defined as the hips→shoulders unit vector. It also gets an optional
  **`visibility`** field of shape (33,).
- `StepEvidence.sources` values must be probabilities. `sources` defaults to `{}`.
- `ActionScores` has a `top()` helper. Its scores are not forced to sum to 1.

**Topics**
- Two topics beyond the plan's 14:
  - **`annotated`**: a `FramePacket` with overlays, for the recorder, the stream and the UI;
  - **`commands`**: `Command`.

## 5. Bus (`actus/core/bus.py`)

- `subscribe(topics, *, maxsize=64, policy="drop_oldest"|"drop_newest", name="")` accepts a
  single topic or several.
- Subscribers receive **`Envelope(topic, seq, t_pub, payload)`**, not the bare payload. `seq` is
  a sequence number that is global across the bus.
- `publish` never blocks. It delivers under the bus lock, so every subscriber sees the same
  global order.
- There is **no blocking policy**. Frames should use `drop_oldest`. Logs and events should use
  a large `maxsize`, and `drop_newest` if history matters more. Drops are counted per
  subscriber and appear in `bus.stats()`.
- `Subscription` methods:
  - `get(timeout)` returns an envelope, or `None` on timeout. It raises `BusClosed` once the bus
    is closed and the queue is drained;
  - `get_nowait()`, `drain()`, iteration (which ends on close), `unsubscribe()` and `stats()`.
- `bus.close()` is the end-of-stream signal. Publishing after it raises `BusClosed`.

## 6. Config (`actus/core/config.py`, `configs/`)

- `load_config(path=None, *, overrides=[...], strict=True)` layers three sources:
  `configs/default.yaml`, then an optional override file, then `key.path=value` strings whose
  values are parsed as YAML.
- It returns a read-only `Config` that supports attribute access, mapping access,
  `get("a.b.c", default)` and `to_dict()`.
- **Strict keys:** an override or file key that is not in `default.yaml` is a `KeyError`. The
  exception is keys under a section whose default is `{}`, which **marks a free-form section**.
  The free-form sections today are `spatial: {}` and `faults: {}`.
- Relative paths are resolved against the repo root with `resolve_path()`.
- `default.yaml` has one section per lane, with guessed defaults that each lane owns and can
  change:
  - top-level sections: `run`, `protocol`, `capture`, `replay`, `detector`, `perception`,
    `rack`, `spatial`, `models`, `voice`, `stream`, `record`, `log`, `ui` and `faults`;
  - notable guesses: the detector backend defaults to `open_vocab` with `yolov8s-worldv2.pt`,
    Kokoro uses voice `af_heart`, the stream target IP defaults to `127.0.0.1`, and the UI
    serves on port 8000.
- **`configs/rack.yaml` is a placeholder with `measured: false`.** It holds:
  - a 0.90 × 0.80 m board;
  - DICT_4X4_50 markers, 0.060 m;
  - marker centres at (0.05, 0.05), (0.85, 0.05), (0.85, 0.75) and (0.05, 0.75) for ids 0–3;
  - an **`outer_box_footprint`**, [[0.15, 0.30], [0.40, 0.55]], for T-SPATIAL's in-outer logic;
  - a **`camera:`** block with `calibrated: false`, `approx_hfov_deg: 70`, and `K`/`dist` set
    to null.
- **Rack frame convention (new):** metres, with the origin at the board's top-left corner as
  seen facing the rack. +x points right and +y points down, which makes **+z point into the
  rack** (right-handed). Protocol zones use this frame.

## 7. Protocol schema and YAML

- The schema is a set of pydantic v2 models with `extra="forbid"` and `frozen=True`. You can
  validate from the shell: `python -m actus.protocol.schema <path>`.
- Protocol fields beyond the plan:
  - `version` (int);
  - **`objects`**: `{name: {label, classes}}`, where `label` is spoken (e.g. "red box") and
    `classes` are the detector classes that map to the object. If `objects` is given, every
    predicate's object must be declared. This is what makes protocol 2 possible from YAML
    alone;
  - **`initial`**: predicate values at the start.
- New step field: **`ask`**, an optional custom confirmation question.
- **`effective_pre`**: when a step has no `pre`, its implicit precondition is the negation of
  `post`. This is what "a step is a transition" means in code.
- Zones are either 2 points, read as the rectangle [[xmin, ymin], [xmax, ymax]], or 3 or more
  points, read as a polygon.
- Naming rules:
  - predicates are `<object>.<predicate>`;
  - a predicate named `in_zone_<Z>` requires zone `Z` to be declared;
  - step ids match `^[A-Za-z][A-Za-z0-9_]*$`, and `idle` is reserved.
- Validation rejects:
  - duplicate step ids;
  - an unknown step in `order` or `min_duration_s`;
  - self-loops and cycles in the order;
  - an empty `post`;
  - a `requires_hoi` that names an unknown object;
  - `p_exit ≥ p_enter`;
  - an `ask_band` that isn't 0 ≤ low < high ≤ p_enter.
- Query helpers: `step()`, `index()`, `step_ids`, `predicates`, `object_names`,
  `object_label()`, `edges()`, `direct_predecessors()`, `direct_successors()`,
  `required_before()` (the transitive closure), `expected_next(done)` (in protocol order).
- Additions to `protocols/bas_boxes.yaml` beyond the plan's snippet:
  - `objects`: outer → [outer_box_closed, outer_box_open, lid], red → [red_box], yellow →
    [yellow_box];
  - `initial`: the box is closed, both small boxes are inside it and not held, and neither is in
    a zone;
  - `ask` texts on S3 and S5.

## 8. Replay source (`actus/capture/replay.py`)

- `ReplaySource(scenario, *, protocol, fps=15, seed=0, noise=0.04, render=False, t0=0,
  source_id="replay")`. It is deterministic for a given (scenario, fps, seed, noise).
- You can iterate it to get **`ReplayTick`**s, or call `run(bus, realtime=, stop=,
  close_bus=)`. `run` publishes to `frames`, `detections`, `predicates`, `evidence` (one message
  per step per frame) and `actions`.
- `t = t0 + frame_idx / fps`, using the replay's own declared fps. This is synthetic time, so
  it doesn't break the "never guess fps" rule.
- The simulator covers only the BAS box world. It refuses a protocol whose predicates it
  can't simulate.
- **Predicates emitted:** `outer.open`; for each of red and yellow, `in_outer`, `held`,
  `in_zone_A` and `in_zone_B`. That is 9 in total, a superset of what the protocol uses.
- **Noise model:** a true value targets 0.93 and a false one targets 0.05. Each predicate lags
  its target with a 0.15 s time constant, gets AR(1) noise plus white noise, and has 1.5× the
  noise if it is a `.held` predicate. Visibility is normally about 0.95. Occlusion pulls each
  probability toward 0.5: `p_obs = v·p + (1−v)·0.5`.
- **Evidence semantics (important for T-ENGINE and T-SPATIAL):**
  - `p_complete` for a step rises only after its transition happened in the ground truth
    (effective pre, then post). It then follows min(prob of each post predicate), and it falls
    back to low when the post-state is undone.
  - A post-state that held from the start is not evidence. S6's "box closed" is low until the
    box has been opened and closed again.
  - Occlusion regresses evidence toward 0.5.
  - `sources` is `{"replay": p}`.
- **Scenario interpretations** (the plan names them only):
  - `correct`: 22 s.
  - `skip_S2`: **the red box is never taken out, so S2 and S3 are both skipped**. You can't
    place a box you never removed. The run is S1, S4, S5, S6.
  - `S4_before_S2`: S1, S4, S5, S2, S3, S6.
  - `unsure_S3`: `red.in_zone_A` is forced to about 0.62 from the moment of placement to the end
    of the run, so it sits inside the ask band for more than 3 s.
  - `occluded`: visibility is 0.15 from 30% of the way through S3 until 2 s after S3 ends. After
    that, S3 confirms.
- **Two scenarios beyond the plan:**
  - `unexpected`: the red box is put back into the outer box after S3, labelled `other` with the
    name `return_red`. S2 and S3 are then redone;
  - `timeout`: a 65 s pause before S4.
- `Scenario.truth` records what really happened, e.g. `{"performed": [...], "skipped": [...]}`.
  The exact expected engine events are still T-ENGINE's to define.
- Action timings: 2 s idle at the start, 2.5 s per step, 1 s gaps before S2, S4 and S6, and 2 s
  idle at the end. Each step changes the world state at fixed fractions of its duration; for
  example, in S2 the box is held at 25% and out of the outer box at 60%.
- `ActionScores`: the ground-truth label gets about 0.75, the rest is shared out, and the scores
  are normalised to sum to 1. An `other` label is spread across idle and two random steps.
- **Rendered frames** (`render=True`): 640×480 BGR, drawn with a fixed orthographic map from
  rack to pixels (`px = 60 + 520·x`, `py = 25 + 520·y`). They show the board, **real
  DICT_4X4_50 markers with ids 0–3** (detectable by `cv2.aruco`, which a test asserts), the
  zones, the outer box and its lid, the coloured boxes, a "hand" disc while an object is held,
  and an occluder overlay. Each frame carries a **"SIMULATED" watermark** and a caption with the
  scenario, frame and truth label.
- Replay detections have `backend="replay"` and track ids outer=1, red=2, yellow=3. A box inside
  a closed outer box is not detected. When visibility is below 0.5, no detections are produced.
- Layout constants: outer box at (0.275, 0.425). The red and yellow boxes carry their own
  in-box positions; `ELSEWHERE_XY` = (0.45, 0.72).
- `ReplaySource.write_annotations(path)` writes ground truth in the T-REC CSV convention.
  `segments()` and `labels()` give per-frame truth.
- Command-line flags: `--scenario`, `--list`, `--format table|jsonl`,
  `--stream predicates|evidence|actions|detections|all`, `--stride`, `--realtime`,
  `--truth-csv`, `--seed`, `--noise`, `--fps`.

## 9. T-REC (`tools/record_takes.py`)

**Provisional:** the ACTUS recording-protocol artifact could not be read, because it isn't
shared with this account. The key map, the naming and the step-boundary rule below are **our
own conventions**. Reconcile them with that doc when it becomes available.

**Naming**
- Takes are named `{operator}_{scenario}_{n:02d}.mp4`, with **`n` zero-padded to 2 digits** and
  auto-incremented.
- **The operator id may not contain `_`**, which is the field separator. The scenario may
  contain `_`, e.g. `skip_S2`.
- A second camera's files are named `<stem>_cam2.*`.
- `parse_take_stem()` is exposed for T-DATA.

**Keys and the label CSV**
- Key map: `1`–`9` start S1–S9 (the plan says 1–6; this was widened for other protocols). `0`
  means idle and `x` means other. **`u` or Backspace undoes the last label.** `q` or Esc saves
  and quits.
- Playback-only keys:
  - space pauses and resumes;
  - `.` and `,` step one frame forward or back while paused;
  - `s` cycles the speed through 1×, 0.5× and 0.25×;
  - `]` and `[` jump 5 s forward or back.
- **Step-boundary rule (invented):** press the key on the frame where the hand starts moving
  toward that step's object, and press `0` once the end state is reached and the hand
  withdraws.
- **Label CSV semantics:**
  - rows record label *changes* (`frame_idx,t,label`), and each label holds until the next row;
  - the first row is always `0,0.0000,idle`;
  - repeated presses of the same label collapse;
  - labels are `S1`…`S9`, `idle` and `other`;
  - the file is autosaved on every key press.

**Other output files**
- **`<stem>.frames.csv`** (new): one row per written frame, with columns
  `frame_idx,t,t_capture,cam_seq,dup`. `t` is when the frame was written, as monotonic seconds
  since the start. `t_capture` is when the camera delivered it. `dup=1` marks a repeated frame.
- **`<stem>.json` metadata** also records:
  - the fps, codec, ISO start time, duration, frame count and stop reason;
  - the key map;
  - `simulated`;
  - per camera: its source and backend, resolution, frames written, duplicates, camera frames
    skipped and read, and a **frame-count check** (an OpenCV decode count, plus ffprobe's count
    if it's on PATH) with a `frame_count_ok` result.

**Recording behaviour**
- **Fixed-fps resampling:** a capture thread keeps only the newest camera frame. At each tick
  of the fixed-fps clock, the newest frame is written. Missed ticks are filled with duplicates,
  so the video keeps wall time.
  - There is a **grace window of 40% of a frame period**, during which the recorder waits for a
    late camera frame before writing a duplicate. Without it, equal camera and output rates
    beat into duplicate-plus-skip pairs (a 25% duplicate rate in testing).
  - The warning about a slow camera fires only when the *net* shortfall, duplicates minus
    skips, is above 10%.
- **The overlay appears only in the preview window. Nothing is burned into the mp4**, which is
  training data.
- The codec defaults to **`mp4v`** and can be changed with `--codec`, e.g. to `avc1` where
  available.
- Default flags: `--fps 30`, `--width 1280`, `--height 720`, `--api any|dshow|msmf|avfoundation|ffmpeg`.
  Metadata flags: `--roll` (degrees), `--lighting` and `--notes`.
- **Test and headless flags (new):**
  - `--no-preview` runs without a window;
  - `--duration` stops after a set time;
  - `--script "t:key,..."` injects scripted key presses;
  - `--source replay:<scenario>` records the P0 replay's rendered SIMULATED frames.
- The playback mode (`--label`):
  - counts frame indices in decode order, re-anchoring from the container only after a seek;
  - uses `CAP_PROP_POS_MSEC` for `t`;
  - refuses a file whose container reports no usable fps;
  - reloads an existing label CSV (`--force` starts fresh);
  - keeps a ring of 300 frames for stepping back;
  - creates or updates the metadata JSON (`labelled_at`, `labelled_with`).

## 10. `CLAUDE.md`

- It holds the §2 principles, the commands (including the `--extra` installs), a table of what
  P0 provides, and the key conventions. Those include the rack frame, predicate naming, the
  meaning of evidence and the annotation CSV.
- Lane conventions it sets:
  - tests go in `tests/test_<lane>*.py` and prefer the replay source;
  - heavy optional imports go inside functions or behind `pytest.importorskip`;
  - run outputs go under `runs/<run_id>/`;
  - to add a dependency, put it in the right extra, run `uv lock`, and say so in the PR.

## 11. Open items for the next wave

- Get the recording-protocol doc shared, and reconcile T-REC's key map, naming and boundary
  rule with it.
- H-PROPS must measure the board and replace the placeholders in `configs/rack.yaml`, then set
  `measured: true`.
- Verify `uv sync && pytest` and the T-REC take on the M2, if the M2 is still the demo machine.
- Before Wave 1 fans out, review the contract additions in §4 above. After that they are frozen.
