# ACTUS v2

On-board human-activity recognition that checks an experiment protocol as it is performed
(ISRO's BAS sample experiment: the boxes). It watches through a camera, tells the operator the
next step by voice, raises an alert on a skipped or out-of-order step, asks when it is unsure and
accepts a spoken answer, and writes a hash-chained, timestamped log with an outcome for every
step. Video is pushed over RTSP to a configurable IP and recorded locally. SIH 2026, problem
26174.

The design rule: **models are sensors, the protocol engine is the judge.** Detectors, pose and
the trained models emit evidence with confidences; only `actus/protocol/` decides that a step
happened. Read [build-plan.md](build-plan.md) for the architecture and [CLAUDE.md](CLAUDE.md)
for the rules the code follows.

> **Status.** The pipeline harness is complete and runs end to end on the SIMULATED replay. The
> trained detector, HOI and temporal models are not plugged in yet, and nothing in this README's
> setup path has been run on a second machine. Section [Where it has and has not been
> verified](#where-it-has-and-has-not-been-verified) says exactly what is known. Anything
> simulated is labelled SIMULATED in the GUI, the overlay and the logs.

## Quick start

```sh
git clone <this repo> Actus-v2.0 && cd Actus-v2.0
bash scripts/setup.sh --demo            # macOS / Linux / Git Bash
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Demo      # native Windows
```

`setup` installs `uv` if it is missing, builds the Python environment, builds the GUI, downloads
the models and the MediaMTX and FFmpeg binaries, primes the open-vocabulary detector, and runs
`tools/doctor.py`, which says what works, what degrades and how to fix each gap. `--demo` then
starts the SIMULATED replay (no camera needed) with the GUI on <http://127.0.0.1:8000>.

Then, for real use:

```sh
bash scripts/run.sh --live                  # your camera (capture.source in configs/default.yaml)
bash scripts/run.sh --live --source 1       # another webcam index, or an rtsp:// URL
bash scripts/run.sh --file data/raw/op01_correct_01.mp4      # a recorded take
bash scripts/run.sh --replay skip_S2        # SIMULATED scenario: correct skip_S2 S4_before_S2
                                            #   unsure_S3 occluded unexpected timeout
bash scripts/run.sh doctor                  # the preflight again, any time
bash scripts/run.sh --live --config configs/sample_mixing.yaml   # the second protocol (cup, bottle, book)
```

On Windows use `scripts\run.ps1` with the same arguments. Other flags go to
`python -m actus.app.pipeline` (`--help`): `--no-stream`, `--no-voice`, `--answer yes`,
`--override stream.target_ip=192.168.1.20`, `--faults`, `--fault occlude@9-12`.

Setup has flags for the common variations. `setup.sh`: `--all-models` (also the optional pose
variants), `--no-media`, `--no-ui`, `--no-warm`, `--skip-models`, `--extras perception,ui`.
`setup.ps1`: `-AllModels`, `-NoMedia`, `-NoUi`, `-NoWarm`, `-SkipModels`.

## What you need

| | |
|---|---|
| Python | 3.11. `uv` fetches it; MediaPipe has no wheels for the newest Pythons, so the project pins 3.11 (`.python-version`). |
| `uv` | Installed by `setup` if missing (its official installer). |
| Node 18+ | Only to build the GUI (`ui/`). Not needed if you install a bundle that carries `ui/dist`. |
| `git` | Only for YOLO-World's `clip` package, which installs from git (see Limitations). |
| Disk | The model assets are about 0.9 GB (sizes are in `configs/assets.yaml`). The Python environment adds several GB, since PyTorch comes with `ultralytics`. |
| Camera | A webcam on a fixed mount looking at the rack. A phone over RTSP also works (`--source rtsp://...`). |
| Rack | A board with four ArUco markers (DICT_4X4_50, ids 0-3, 60 mm) at the corners, measured into `configs/rack.yaml`. `tools/make_markers.py` prints the markers; `tools/check_markers.py` shows whether the camera sees all four. |
| Audio | Speakers for prompts and alerts; a microphone if you want to answer by voice. Both are optional. |
| Network | Setup only. Nothing in the run-time path uses it. |

`configs/rack.yaml` ships with `measured: false` and placeholder numbers. Until you measure your
board and set `measured: true`, the zones will not line up with it and steps S3 and S5 cannot
verify. `tools/doctor.py` warns about this.

## What the run gives you

| Deliverable | Where |
|---|---|
| Live processing, next-step prompts, alerts, asks | GUI (<http://127.0.0.1:8000>) and voice |
| Structured log with outcome per step | `runs/<run_id>/events.jsonl`, SHA-256 hash-chained |
| Bandwidth-restricted summary | `runs/<run_id>/downlink.json`, at most 4 KB |
| Recorded video, raw and annotated | `runs/<run_id>/video/*.mp4`, 60 s segments |
| RTSP push to a configurable IP | `rtsp://<stream.target_ip>:8554/actus/raw` and `/actus/annotated` |
| Run metadata and per-sink status | `runs/<run_id>/run.json` |

Watch the stream from another machine with VLC: *Media > Open Network Stream*, then
`rtsp://<target_ip>:8554/actus/annotated`. Set the target with
`--override stream.target_ip=<ip>`; the pusher keeps retrying with backoff if nothing listens
there, and recording never depends on it.

Check that a log was not edited:

```sh
uv run python -m actus.io.eventlog verify runs/<run_id>/events.jsonl
```

## Offline installs (the bundle)

Setup needs the network once. To bring up a machine that never has it (a lab PC, the demo
laptop on a bad network), build a bundle on a connected machine **of the same OS and CPU
architecture**, then install it from a file.

```sh
# on the connected machine, after a normal setup
uv run python tools/fetch_models.py --lock              # pin the hashes; commit configs/assets.lock.json
uv run python tools/bundle.py make --wheels             # dist/actus-bundle-<platform>-<date>.zip (+ .sha256)
uv run python tools/bundle.py verify dist/actus-bundle-*.zip

# on the offline machine: copy the repo (git clone or a zip) and the bundle, then
bash scripts/setup.sh --bundle actus-bundle-linux-amd64-20260930.zip
# Windows: powershell -File scripts\setup.ps1 -Bundle actus-bundle-windows-amd64-20260930.zip
```

The bundle holds everything under `models/` (downloaded weights and the models trained here, with
their cards), MediaMTX and FFmpeg, the built GUI, the CLIP text encoder, the asset pins, and with
`--wheels` every Python wheel of the locked environment. It does not hold the source code.
Installing checks every file's SHA-256 while unpacking and renames nothing until all of them
verified, so a corrupted or altered bundle changes nothing. `bundle.py install` needs only a bare
system Python (3.9+).

Two things it cannot do for you: the offline machine needs **Python 3.11 already installed**
(uv cannot download it offline), and a bundle made on one platform will not install on another
(binaries and wheels differ; `--portable-only` takes just the models and the GUI). Without
`--wheels` the bundle assumes a working `.venv` is already there. With `--wheels` expect 1-4 GB
more, depending on the platform (PyTorch's Linux wheels are the large part).

After an offline install, `scripts/run.*` set `UV_OFFLINE` and `UV_NO_SYNC` (from the
`.actus-offline` marker) so `uv run` neither tries the network nor re-syncs the environment.

## Configuration you will actually touch

| To change | Set |
|---|---|
| The camera | `capture.source` (index, file, or `rtsp://`), or `--source` |
| Where the video is pushed | `stream.target_ip`, `stream.port` |
| The procedure | `protocol.path` (a YAML like `protocols/bas_boxes.yaml`); `configs/sample_mixing.yaml` selects the second protocol and its stock COCO detector (`models/coco/yolo11n.pt`, fetched by setup) |
| The detector | `detector.backend`: `open_vocab` (YOLO-World, the fallback) or `custom` (our trained YOLO11n; the default only once its held-out mAP50 is at least 0.85) |
| Voice | `voice.enabled`, `voice.tts.engine` (`kokoro` or `pyttsx3`), `voice.asr.enabled` |
| The models trained here | `models.enabled`; they load from `models/hoi/hoi.npz` and `models/temporal/temporal.npz` when present |

Site overrides go in a YAML passed with `--config`, or as `--override key.path=value`. Unknown
keys are errors (`actus/core/config.py`). Swapping any pipeline stage for a better model needs no
pipeline change; see `Vault/T-INT.md` section 3.

## Tools

| Tool | For |
|---|---|
| `tools/doctor.py` | Preflight: modules, model files, binaries, ports, devices, rack geometry |
| `tools/fetch_models.py` | Download, verify (`--verify`) and pin (`--lock`) the runtime assets; `--list` shows the state |
| `tools/bundle.py` | Make, verify, inspect and install the offline bundle |
| `tools/record_takes.py`, `check_markers.py`, `make_markers.py` | Filming and labelling takes; the rack markers |
| `tools/extract_frames.py`, `autolabel.py`, `review_labels.py`, `make_splits.py`, `synth_augment.py` | Dataset generation (detection, pose, HOI) |
| `tools/train_detector.py`, `train_hoi.py`, `train_temporal.py` | The models we train ourselves |
| `tools/run_offline.py` | Process a take deterministically, with overlays |
| `tools/evaluate.py` | Held-out-operator report card (`runs/eval/report.json`, shown in the GUI's Validation tab) |
| `python -m actus.app.faults --demo all` | Fault injection against ground truth |

## Where it has and has not been verified

This is the honest table. Fill in a row when someone measures it; do not round up.

| Machine | Role | Setup path verified | Live FPS (detector + hands + pose) | Notes |
|---|---|---|---|---|
| Windows 11 laptop (current dev machine) | development | unit tests written, not yet run for the packaging lane | not measured | The pipeline harness alone costs about 7.5 ms per frame on the replay (no neural stages), per `Vault/T-INT.md`. |
| Apple M2 MacBook | target demo machine | not verified | not measured | The plan's bar is at least 15 FPS live with voice, GUI, RTSP and recording. ultralytics chooses its own device; MediaPipe runs on CPU here. |
| Laptop with an RTX 4070 | detector training | not verified | not applicable | The plan says YOLO11n also trains acceptably on the M2 (MPS). |
| Ubuntu x86-64 | CI | the workflow is disabled (`.github/workflows/ci.yml.disabled`) | not applicable | Runs ruff and pytest when enabled. |
| NVIDIA Jetson Orin | finale (F-EDGE) | not built | not measured | ONNX/TensorRT export is a finale task. |

To measure a row:

```sh
uv run python -m actus.perception.detector --seconds 10        # detector FPS from the camera
bash scripts/run.sh --live                                     # then read the GUI's stats strip
```

Run the test suite with `uv run pytest`, and lint with `uv run ruff check . && uv run ruff format
--check .`. The packaging lane's tests are `tests/test_pkg_*.py`; they use local files and a
folder standing in for the internet, never the network.

## Limitations

Copied from the plan's honesty list, plus what the current state adds. They go on the slide as
they are.

- The step logic is a protocol state machine over trained perception. The HAR model provides
  evidence and does not decide alone.
- The dataset is small: a handful of operators, about an hour of footage, filmed on Earth. Numbers
  are on **held-out operators**, not held-out environments.
- The rotated-camera test emulates a lack of a fixed "up"; it is not microgravity footage.
- The orientation-agnostic claim covers the rack frame, rotation augmentation and rotation
  test-time augmentation. It is not full 3D human mesh recovery.
- The sample-experiment steps are our reading of a truncated official description
  (`protocols/bas_boxes.yaml`).
- Throughput is measured on the machines in the table above, not on flight hardware.
- Until the custom detector is trained and the rack is measured, real footage cannot be judged:
  the open-vocabulary detector does not find the red box reliably and the placeholder rack
  geometry gives no valid frame (`Vault/T-HOI-TEMP.md`, `Vault/T-INT.md`). The harness runs; the
  steps stay pending.
- YOLO-World needs OpenAI's CLIP text encoder and the `clip` Python package, which installs from
  git. `setup` fetches both, but the package is not in `uv.lock`, so a later plain `uv sync`
  removes it; re-run `setup` (or `tools/fetch_models.py --warm`). Moving it into the
  `perception` extra is a proposed follow-up (`Vault/T-PKG.md`).

## Licences

ACTUS depends on:

| Component | Licence |
|---|---|
| Ultralytics (YOLO11, YOLO-World) | AGPL-3.0 |
| MediaPipe | Apache-2.0 |
| Kokoro-82M and `kokoro-onnx` | Apache-2.0 |
| Vosk and its small English model | Apache-2.0 |
| MediaMTX | MIT |
| OpenAI CLIP | MIT |
| FFmpeg | LGPL-2.1+ or GPL-2.0+/3.0 depending on the build; the Windows and Linux download here is BtbN's GPL build |

Ultralytics' AGPL-3.0 carries obligations when the software is distributed or offered over a
network; check them before publishing or handing the code on. SMPL, which is non-commercial,
matters only if the finale's mesh-recovery spike ships; it is not used today.

## Repository layout

```
actus/core/        contracts, bus, config          actus/protocol/   schema, engine, decision layer
actus/capture/     camera, replay                  actus/spatial/    rack frame, zones, world state
actus/perception/  detector, hands+pose, TTA       actus/models/     HOI + temporal runtime
actus/voice/       TTS, voice queue, ASR           actus/io/         streamer, recorder, log, downlink
actus/app/         pipeline, server, faults, overlay
ui/                React dashboard (built to ui/dist)
protocols/         bas_boxes, sample_mixing        configs/          default, rack, assets (+ overlays)
tools/             recording, dataset, training, evaluation, packaging
scripts/           setup.sh|ps1, run.sh|ps1, run_demo.sh, setup_media.sh
Vault/             per-lane decision notes          tests/            pytest suite
data/ models/ runs/ bin/ dist/ wheels/               gitignored
```

`scripts/setup_media.sh` (from the IO lane) is the older, standalone way to fetch MediaMTX and
FFmpeg. `tools/fetch_models.py` does the same for every platform, including Windows, and also
pins and verifies; prefer it.
