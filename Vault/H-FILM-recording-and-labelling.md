# H-FILM: recording and labelling workflow

2026-09-27 · changes to `tools/record_takes.py` (T-REC) made for H-FILM · read with
`build-plan.md` §5 (T-REC, H-FILM) and `Vault/Wave0.md`

H-FILM films the dataset, and the step labels are human ground truth. The system never labels its
own training footage. This note records what was added to the take recorder for the filming
sessions, and why.

Status: all recorder tests pass (`tests/test_record_takes.py`, 8 tests), and `ruff check` and
`ruff format --check` are clean. Nothing is committed.

---

## 1. Workflow

1. **Record** in toggle mode with the phone camera (the IP Webcam Android app) as the source:

   ```powershell
   .venv\Scripts\python tools\record_takes.py --operator op01 --scenario correct --toggle --source http://<phone-ip>:8080/video
   ```

   `r` starts a take and `r` again stops and saves it. `q` / Esc quits. Nobody needs to press
   step keys while filming.

2. **Label** afterwards, all unlabelled takes in one pass:

   ```powershell
   .venv\Scripts\python tools\record_takes.py --label-pending
   ```

   Each take opens in turn in a 640 px labeller window. `q` saves it and opens the next take;
   Esc saves it and stops the batch. Running the command again resumes with the takes that are
   still pending.

## 2. Changes

### 2.1 Record toggle (`--toggle`)

- The camera opens in a live **STANDBY** preview, and nothing is written.
- `r` starts a take and `r` stops and saves it. Each start/stop pair is **one take** with the
  next free number (`op01_correct_01`, `_02`, ...). This was chosen over pausing and resuming a
  single file because the dataset naming, splits and label CSVs are per take.
- `q` / Esc during a take saves it and ends the session. `--duration` is a per-take limit; in
  toggle mode it returns to standby instead of quitting.
- Standby time is never recorded. Each take's `frame_idx` and `t` start at 0, as before.
- `stop_reason` in the take JSON is `toggle` for a take stopped with `r`.
- Without `--toggle` the recorder behaves exactly as before (it records immediately and `q` stops).
- Code: `record()` was split into `open_sources()`, `record_take()` (one take from
  already-running sources) and `record_session()` (the standby loop), so a single camera
  connection serves the whole session.

### 2.2 Phone camera source (no code change)

- `--source http://<phone-ip>:8080/video` (IP Webcam's MJPEG stream) already works, because the
  source string goes straight to OpenCV's FFMPEG backend.
- It was verified against a local fake MJPEG server that behaves like IP Webcam: 640×480, 58
  frames written, frame count OK. The real phone was not tested from this session.
- `--width` / `--height` do not apply to URL sources; set the resolution in the app (1280×720
  recommended).
- If the Wi-Fi stream drops, the take ends with `stop_reason: source_ended` and the session
  ends. There is no reconnect, which is the T-REC rule ("never loop or reopen"). What was
  recorded is kept.
- MJPEG over Wi-Fi often delivers fewer than 30 fps. If a take ends with a duplicated-frames
  warning, record with `--fps 20` or `--fps 15`.

### 2.3 Batch labeller (`--label-pending [DIR]`)

- Default folder: `data/raw`.
- **What counts as pending:** the take's JSON has no `labelled_at` (so it was never labelled in
  playback), and its CSV holds nothing but the initial `idle` row (nobody pressed step keys
  while recording). A take labelled live, or labelled once in playback, is skipped.
- `_cam2` files are never offered: a second camera's take shares camera 1's labels, because
  their frame indices are aligned.
- Files that don't follow the `{operator}_{scenario}_{n}` naming (e.g. a phone file copied in)
  are still offered.
- Takes are opened in filename order. The window shows `take i/N` and the file name.
- Keys are the same as `--label` mode, except: `q` saves and opens the next take, Esc saves and
  stops the batch.
- Code: `label_existing()` now wraps `label_video(args, video, progress, help_text)`, which
  returns `(labels_path, esc_pressed)`. New helpers: `is_pending()`, `pending_takes()`,
  `label_pending()`.

### 2.4 Smaller labeller window (`--view-width`, default 640)

- The playback labeller window is now 640 px wide by default (it was up to 960). Change it with
  `--view-width 960`.
- The recording preview is unchanged (960 px maximum).

## 3. Considered and not done

These were offered and declined for now. Each is a small addition if it's wanted later.

- Opening the labeller automatically after each take in toggle mode.
- Clicking the timeline to seek.
- A key that deletes the label under the playhead. Currently `u` only undoes presses from the
  current session. A wrong label saved in an earlier session can be overridden by pressing the
  correct key at the same frame, but not removed.
- Keys to jump to the next or previous label boundary.

## 4. Open items

- The ACTUS recording-protocol doc is still unread (see `Vault/Wave0.md`), so the scenario names
  (`correct`, `skip_S2`, `S4_before_S2`, ...) follow the replay scenarios, not the doc.
- `configs/rack.yaml` is still `measured: false`. The rig is reported as built and measured, but
  the numbers have not been entered yet. Recording doesn't need them; live step detection
  (T-SPATIAL zones, T-INT) does.
