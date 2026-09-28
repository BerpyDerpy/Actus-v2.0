# T-FAULT: decisions made outside the build plan

2026-09-28 · covers T-FAULT (fault injection: `actus/app/faults.py`) · read with `build-plan.md`
§4–5 and `Vault/T-INT.md`

The T-FAULT card lists five faults and a "done when". This file records every decision made while
implementing it that the card does not state, so that T-INT, T-EVAL, T-UI and T-DEMO can rely on
them or overturn them on purpose.

**Scope, as agreed:** the harness. Nothing is committed. **No code was run** while writing this
lane, so nothing below is verified; §0 lists what to run first and which assertions are the most
likely to fail.

## 0. Status against the card's "done when", and what to run first

| Criterion | Status |
|---|---|
| Hotkeys and GUI toggles for the five faults | Written. Console keys `o d f x n` (`HotkeyListener`), GUI buttons and the same keys in the browser (`ui/src/components/FaultControls.tsx`), `POST /api/faults/{name}`, a schedule (`--fault occlude@9-12`) and code (`FaultController.request`). **Unrun.** |
| Each fault shown as "FAULT INJECTED" | Written: an overlay tag on the annotated frame and stream (`FAULT INJECTED: <label>`), a red pulsing pill in the GUI header, and `Stats("faults")`. **Unrun.** |
| Expected behaviour is abstain or ask, or "stream down, still recording", never a false `verified` | Enforced by construction for `drop_detector` and `freeze` (§2.2) and by the existing world model for `occlude` and `dim`. **Not demonstrated yet.** |
| Each fault demonstrated on a replayed take, with the safe behaviour logged | `python -m actus.app.faults --demo all` writes `faults.jsonl`, `faults_report.json` and `faults_report.md` per fault (§5). **Unrun.** |

Run these, in this order:

```sh
uv run ruff check . && uv run ruff format .          # formatted by hand; expect small diffs
uv run pytest tests/test_faults.py tests/test_pipeline.py tests/test_config.py
uv run python -m actus.app.faults --demo all
(cd ui && npm run build)                             # the new component was never type-checked
```

Most likely to fail, because each rests on an assumption about another lane that I could not
check by running it:

1. `test_a_capped_pipeline_verifies_nothing_and_the_engine_asks`: the final assertion that an
   `Ask` appears once visibility is capped to 0 for 1.5 s while S1 is expected (T-ENGINE's
   `vis_ask_after_s`).
2. `test_every_fault_demo_ends_safe_on_the_replay[occlude|dim]`: these rely on the spatial world
   model not inventing a placement while detections are missing (`Vault/T-SPATIAL.md`). If S3 is
   ever `verified` inside the window, `no_fabricated_verification` fails, and that is a real
   finding about the spatial lane, not a test bug.
3. Anything on the network path with a real streamer (§4): only `SimulatedTarget` is exercised.

## 1. Files and entry points

| File | What it holds |
|---|---|
| `actus/app/faults.py` | The catalogue (`FAULTS`), the five faults, `FaultController`, `StaleFrameGuard`, `StreamerTarget` / `SimulatedTarget`, schedules, `HotkeyListener`, `mount_fault_routes`, the safety oracle (`evaluate_safety`), the demo runner (`run_fault_demo`, `run_fault_montage`) and the CLI. |
| `tests/test_faults.py` | See §7. |
| `ui/src/components/FaultControls.tsx` | Header buttons and the FAULT INJECTED pill. |

```sh
python -m actus.app.faults --list                            # faults, keys, expected behaviour
python -m actus.app.faults --demo all                        # every fault, offline, vs ground truth
python -m actus.app.faults --demo occlude --window 9-12 --param region=0,0,0.5,1
scripts/run_demo.sh --replay correct --fault occlude@9-12    # live app, GUI panel, hotkeys
scripts/run_demo.sh --live --faults                          # camera; toggle by hand
```

T-EVAL can pass a controller to `run_offline(..., faults=FaultController(schedule=[...]))`. A
scheduled controller keeps the run byte-deterministic (the schedule is keyed on the frames' own
clock).

## 2. How a fault reaches the pipeline

### 2.1 Where it is injected

`PipelineCore.step` now runs `core.source_hooks` first: callables `SourceFrame -> SourceFrame`.
`FaultController.process` is one. It runs after the Runner has published the raw frame, so **the
raw stream and recording keep the true image and only the annotated one shows the fault** (the
same reason T-INT put the hooks in the core). The `frame_hooks` T-INT left for T-FAULT stayed
unused: they only see a `FramePacket`, and the replay's detections travel in the `SourceFrame`.

Requests from other threads (hotkeys, GUI) are queued and applied at the **next frame**, stamped
with that frame's index and time. A fault is active while it is switched on manually or a
scheduled window covers the frame.

### 2.2 Blindness, and why `drop_detector` and `freeze` cannot verify

`PipelineCore.visibility_cap` (new): an upper bound on how much the sensors can currently see,
for a *known* outage. While it is set, `cap_visibility` clips the visibility handed to the engine
and pulls every step evidence and predicate toward 0.5 by `1 - cap`, the same mixing the replay
uses for an occlusion. At `cap = 0` no step can reach `p_enter` and the engine sees a blind
sensor and asks about the expected step.

Without the cap this was a real hole: a stage that does not run leaves the spatial lane's own
visibility untouched (`ran` is False, so `_update_visibility` is skipped), and its beliefs would
be free to predict a placement nobody observed.

| Fault | What the pipeline sees |
|---|---|
| `occlude` | Region filled with seeded noise (0-3), detections that are ≥ 50 % covered dropped. Blindness comes out of the real chain: markers lost, rack held then invalid, visibility falls. |
| `dim` | Gamma LUT on the frame. Emulated detections: confidence × `0.5 ** (gamma - 1)`, dropped below `detector.conf`. |
| `freeze` | The previous delivered frame again (and its detections, restamped). The **stale-frame guard** then trips (§2.3). |
| `drop_detector` | `core.disabled += {"detector"}`, detections `None`, cap 0. |
| `kill_network` | Acts on the RTSP target only (§4). The frames are untouched. |

### 2.3 The stale-frame guard

`StaleFrameGuard`: `stale_after` (3) bit-identical frames in a row = the source is stuck. It then
disables `detector` and `handpose`, caps visibility at 0 and tags the frame
`FAULT DETECTED: stale frames`. It is the *system's* defence and is judged independently of the
fault that provoked it: it looks only at the frames the pipeline is about to perceive.

- The first `stale_after` repeats still reach the pipeline (a 3-frame latency).
- **The guard runs only while a controller is attached.** A real recording with duplicated frames
  (phone video at variable rate) would otherwise be mistaken for a fault and starve the engine.
  A real camera freezing during a live demo is therefore *not* covered unless faults are on. If
  T-INT wants it always on, attach a controller with no faults; `faults.stale_frames: 0` turns it
  off.
- An occlusion fills its box with seeded noise for this reason: a bit-exact black frame would trip
  the guard and mislabel a blackout as a frozen camera.

## 3. Decisions the card leaves open

- **The fault list is fixed at five, as on the card.** `drop_handpose` and `drop_models` would be
  ten lines each on top of `core.disabled`, but were not asked for.
- **Emulation on stand-in detections.** The replay supplies its own detections, so an image fault
  alone would change nothing downstream. The same physical effect is applied to them, but only for
  frames whose `SourceFrame.detections` is set (i.e. the SIMULATED replay); a real detector sees
  the black box or the dark frame in the pixels. The emulation constants (50 % coverage, the dim
  curve) are models, not measurements.
- **`evidence_mode=replay` bypasses faults**, except the cap. That mode feeds the replay's own
  evidence straight to the engine, so `attach` logs a warning. Demos force `evidence="spatial"`.
- **Manual switch vs schedule.** Manual on/off wins while it is on. A manual off does not cancel a
  scheduled window that still covers the frame.
- **A refused fault stays refused only while it is being asked for**, and is logged once
  (`unavailable`), e.g. `kill_network` with no streamer.
- **Hotkeys are `o d f x n`**, listed in `FAULTS[...].key`. They use `msvcrt` on Windows and
  `termios` cbreak elsewhere (Ctrl-C still works), and do nothing when stdin is not a terminal.
- **Config (`faults:`)** replaced the empty free-form section: `enabled`, `hotkeys`,
  `stale_frames`, `schedule`, and per-fault parameter defaults. Schedules read
  `name@start-end[:key=value;key=value]`, in seconds of pipeline time since the first frame, or
  `f<frame>` indices (`dim@f90-135:gamma=4`); regions are `region=0,0,0.5,1`.

## 4. The network fault

`kill_network` cuts and restores whatever `set_network_target` was given:

- **Live:** `StreamerTarget` calls T-IO's `Streamer.stop()` and `.start(fps=)`, so MediaMTX and both
  FFmpeg pushers go away and come back. That is what a dead target looks like from here. It runs
  on a single worker thread (`stop()` can take seconds and must not stall the frame loop), so cut
  and restore happen in the order asked. **The recorder, the event log and the GUI's MJPEG feed do
  not use the streamer and carry on.** No edit to `streamer.py` was needed.
- **Offline demos:** `SimulatedTarget` counts what it would have delivered, frame by frame. The
  report says SIMULATED. It cannot show VLC reconnecting; only a live run can.
- Without a target the fault is refused and logged.
- `LiveApp` restores the network on a normal `close()` but skips it at shutdown
  (`restore_network=False`), since the sinks are being torn down anyway.

## 5. The fault log and the demo report

**`runs/<id>/faults.jsonl`**: a `FaultEvent` per line (`start`, `stop`, `unavailable`, `detected`,
`cleared`) with the fault, its parameters, the source (`manual` / `schedule` / `guard`), the frame
index and `t`, `blinding`, the **expected safe behaviour**, `simulated`, and `events_tip`, the event
log's chain tip at that moment. It is the same SHA-256 chain, so
`python -m actus.io.eventlog verify runs/<id>/faults.jsonl` works, and the two files order against
each other. A run always starts a fresh file. Under `run_offline` its timestamps come from the same
`PipelineClock` as `events.jsonl`, so it is deterministic too.

Why a separate file: `EventLog` accepts only the five contracts in `LOGGABLE`. Proposal for the
integrator (not applied, it needs a contract change): let `EventLog` log a `FaultEvent`, or add
`faults_injected` and the fault-log hash to the downlink packet. **Today the downlink packet does
not say a fault was injected.**

**`run_fault_demo(fault, ...)`** runs a replay scenario twice, fault-free and faulted, and judges
the faulted run against the replay's **ground truth** (`truth_completion` reads when each step's
transition really completed from the scenario script). Hard checks:

- `no_false_verified`: nothing `verified` that never happened;
- `no_fabricated_verification`: for a *blinding* window (a full occlusion, dim at gamma ≥ 3,
  freeze, `drop_detector`), no step is `verified` inside the window when its transition also
  completed inside it. A step that completed just before the window, or is verified after it, or
  is `operator_confirmed`, is fine. Mild faults are not held to this;
- `fault_window_logged`, `logs_verify` (both chains intact);
- `freeze`: `guard_tripped`; `kill_network`: `engine_unaffected` (every engine output equals the
  fault-free run's), `recording_continued` (annotated frames recorded = frames processed) and
  `target_saw_the_outage`.

Reported but **not** failed on: the asks raised, the steps that completed in the window and how
they ended, and `alerts_added_by_the_fault`. A fault can legitimately cause a retroactive `skip`
(the operator may have answered no), and whether that is "false" depends on the operator, so T-EVAL
should decide how to score it.

The default plan (`DEMO_PLAN`) injects each fault over 9-12 s of the `correct` replay, which covers
S3's placement at t = 10.0, with a scripted "yes" to any Ask.

## 6. Edits outside T-FAULT's own file

| File | Change | Why |
|---|---|---|
| `actus/app/pipeline.py` (T-INT) | `PipelineCore.source_hooks` and `.visibility_cap`; module function `cap_visibility`; `faults=` on `run_offline` and `LiveApp` (attach, open the log, point `kill_network` at the streamer, hotkeys, teardown, `run.json["faults"]`, `Stats("faults")` in the sink stats); `UIServer(faults=)`; CLI `--faults` and `--fault SPEC`. | §2. All optional: with no controller nothing changes. |
| `actus/app/server.py` (T-UI) | `create_app(..., faults=None)` mounts `/api/faults` and `POST /api/faults/{name}` before the static mount. | GUI toggles. |
| `configs/default.yaml` | `faults:` filled in (§3). | Documents the keys. `tests/test_config.py` still passes: it only overrides `faults.occlude=true`, which replaces the mapping. |
| `ui/src/{types.ts,api.ts,App.tsx,styles.css}`, `ui/src/components/FaultControls.tsx` | Types, two API calls, the controls and the pill in the header. | GUI. Renders nothing when `/api/faults` 404s. Needs `npm run build`. |
| `scripts/run_demo.sh` | Usage text only. | |

## 7. Tests (`tests/test_faults.py`)

- Catalogue, schedule parsing and its errors, `is_blinding`.
- Each fault's effect on a frame and its detections (region, dim curve, freeze restamp, drop).
  A full-frame occlusion is not bit-identical from frame to frame.
- The guard: trips after N repeats, clears on a fresh frame, equal-but-distinct arrays count.
- The controller against a fake core: schedule by seconds and by frame, manual toggle at the next
  frame, validation in the caller's thread, stage / cap / tag bookkeeping (someone else's
  `disabled` entry survives), freeze → guard → release with the exact log sequence, `kill_network`
  refused / cut and restored frame-exactly (`SimulatedTarget`) / in order on a slow target, `close`.
- The fault log: chain verifies, deterministic, a one-byte edit is caught.
- The oracle on synthetic outputs (fabricated vs legitimate), and `truth_completion`.
- The pipeline hooks: `cap_visibility` maths, `source_hooks` order, a capped core verifies nothing.
- **End to end on the replay (unrun):** a scheduled fault through `run_offline` is byte-identical
  across reruns; `LiveApp` with a controller writes the log and `run.json["faults"]`; every fault's
  demo passes its checks; the GUI routes list, toggle and reject (`importorskip("fastapi")`).

## 8. Open items and known limitations

- **Nothing has been executed** (§0).
- The guard is opt-in for real runs (§2.3) and costs `stale_after` frames of latency.
- Emulated detector effects on the replay are a model. The claim "the system degrades gracefully"
  is only as strong as the fault-free baseline it is compared with; on real footage the baseline
  cannot be judged yet (`Vault/T-INT.md` §7).
- `dim` is blinding at gamma ≥ 3 by a rule of thumb (`DIM_BLIND_GAMMA`), not a measurement.
- A real `kill_network` takes as long as `Streamer.stop()` (up to a few seconds) and a viewer such
  as VLC must reconnect by itself when the stream returns. Not seen live.
- `Vault/T-INT.md` §7's bullet "Hooks for T-FAULT are in place, but `faults.py` isn't written" is
  now stale.
- Push-to-talk and voice are not involved: the Ask a fault provokes is answered by the GUI's
  Yes/No, the ASR or, in demos, a scripted answer.
