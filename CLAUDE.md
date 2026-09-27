# ACTUS v2 — agent rules

Read this first. The full plan is `build-plan.md`; your task card is your prompt.

## Principles (build plan §2)

- **The protocol engine is the judge; models are sensors.** Models emit evidence with
  confidences. Only `actus/protocol/` decides that a step happened.
- **Evidence and expectation are separate channels.** Skip and out-of-order detection use
  evidence only. The protocol prior may only rank the next-step suggestion and break ties. It
  never raises `p_complete`.
- **Frame index is the clock.** Every observation carries `frame_idx` plus a monotonic
  `t_capture`/`t`. Never derive time from `index / fps` with a default fps.
- **Everything runs offline at runtime.** No cloud API anywhere in the runtime path. A download
  during setup is fine.
- **Honest by construction.** No placeholder metrics, no mock detectors left in any demo path.
  Anything simulated is labelled "SIMULATED" in the UI and the logs.
- **Contracts in `actus/core/contracts.py` are frozen.** A lane that needs a change proposes it
  in its PR description; the integrator (Opus) applies it. Don't edit it yourself.
- **Each lane owns its files** (see its task card). Don't edit another lane's files; stub
  against the contract instead.
- Python 3.11, managed with `uv`. Lint with `ruff`, test with `pytest`. Every lane ships tests.

## Commands

```sh
uv sync                              # base + dev deps (numpy, opencv, pydantic, pytest, ruff)
uv sync --extra perception           # + ultralytics, mediapipe, onnxruntime   (T-PERC)
uv sync --extra voice                # + kokoro-onnx, pyttsx3, vosk, sounddevice (T-VOICE)
uv sync --extra ui                   # + fastapi, uvicorn                       (T-UI)
uv run pytest                        # all tests
uv run ruff check . && uv run ruff format --check .   # what CI runs
uv run python -m actus.protocol.schema protocols/bas_boxes.yaml       # validate a protocol
uv run python -m actus.capture.replay --list                          # replay scenarios
uv run python -m actus.capture.replay --scenario skip_S2              # predicate stream
uv run python -m actus.capture.replay --scenario occluded --stream evidence
```

Adding a dependency: add it to the right extra in `pyproject.toml`, run `uv lock`, and say so in
the PR description (the integrator resolves `uv.lock` conflicts).

## What P0 provides

| Module | Use it for |
|---|---|
| `actus/core/contracts.py` | Every message type, the `Topic` enum, `StepStatus`, `AlertKind`, and `to_jsonable` / `from_jsonable` (JSON with a `"type"` tag). Constructors validate probabilities, clocks and array shapes. |
| `actus/core/bus.py` | `Bus().subscribe(topics, maxsize=, policy="drop_oldest"\|"drop_newest")`, `bus.publish(topic, msg)`. Subscribers get `Envelope(topic, seq, t_pub, payload)`. Publishing never blocks; drops are counted per subscriber (`sub.dropped`, `bus.stats()`). `bus.close()` ends iteration. |
| `actus/core/config.py` | `load_config(path=None, overrides=["stream.target_ip=10.0.0.5"])` over `configs/default.yaml`. Your section of `default.yaml` is yours to extend; unknown keys are errors except under `{}` (free-form) sections. |
| `actus/protocol/schema.py` | `load_protocol(path)` → validated `Protocol` with `step()`, `required_before()`, `direct_successors()`, `expected_next(done)`, `effective_pre`. |
| `actus/capture/replay.py` | `ReplaySource(scenario, fps=, seed=, render=)`: a deterministic synthetic stream of `Predicates`, `StepEvidence`, `ActionScores`, `DetectionSet` and optional rendered `FramePacket`s (with real ArUco markers, watermarked SIMULATED). Iterate it, or `run(bus)`. Scenarios: `correct`, `skip_S2`, `S4_before_S2`, `unsure_S3`, `occluded`, plus `unexpected`, `timeout`. |

Key conventions (details in the module docstrings):

- Rack frame: metres, origin at the board's top-left corner (facing the rack), +x right, +y down,
  +z into the rack. Zones in protocol YAML use it.
- Predicate names are `<object>.<predicate>`: `outer.open`, `red.in_outer`, `red.held`,
  `red.in_zone_A`, ...
- `StepEvidence.p_complete` is prior-free evidence that the step's *transition* happened
  (effective `pre` → `post`); a post-state that held from the start is not evidence.
- Annotation CSVs (T-REC sidecars, `ReplaySource.write_annotations`) are label-change rows
  `frame_idx,t,label` with labels `S1`…`S9`, `idle`, `other`.

## Conventions

- Tests go in `tests/test_<lane>*.py` (e.g. `tests/test_engine_replay.py`). Use the replay source
  rather than cameras or recorded files wherever you can, so CI runs anywhere.
- Heavy optional imports (ultralytics, mediapipe, kokoro, vosk, fastapi) go inside functions or
  behind `pytest.importorskip`, so the base install and CI stay light.
- Paths in config are relative to the repo root; resolve them with `actus.core.config.resolve_path`.
- Run outputs go under `runs/<run_id>/`. `data/`, `models/`, `runs/` and `bin/` are gitignored.
- Keep `ruff format` clean (line length 100).
