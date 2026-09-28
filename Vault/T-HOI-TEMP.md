# T-HOI-TEMP: decisions, and two blockers that reach past this lane

2026-09-28 · covers T-HOI-TEMP (offline feature extraction, HOI classifier, temporal action
model, evidence fusion) · read with `build-plan.md` §5 (Wave 2), `Vault/T-SPATIAL.md` and
`Vault/T-ENGINE.md`

The card specifies the lane in outline. This file records what the card does not state: the
decisions taken during implementation, and — more urgently — two upstream problems that surfaced
only when the real perception stack was run over the real footage. Both change what other lanes
can deliver, so they are at the top.

---

## 0. Two blockers, for the integrator

### 0.1 The marker rig does not match `configs/rack.yaml`, and it changed between operators

`RackFrame.valid` is **False on every frame of every recorded take** when solving against
`configs/rack.yaml`. Two independent causes:

1. **The markers are not mounted upright on a common board.** They are printed on loose sheets
   laid on the floor at arbitrary individual rotations — measured top-edge angles in one frame
   were 49°, 92°, −58° and −132°. `RackGeometry.marker_corners` assumes each marker is upright,
   so the corner correspondences are wrong. Fitting the configured model to the detected corners
   gives **42 px RMS against a 4 px acceptance gate**.
2. **The rig was re-laid per operator, at different sizes.** Reconstructed from the footage
   (marker 0's known 60 mm square defines the floor plane):

   | operator | reconstructed layout | notes |
   |---|---|---|
   | op01 | ~1.09 × 0.74 m | marker 0 rarely visible |
   | op02 | ~0.65 × 0.58 m | sheets nudged mid-session on two takes |
   | op03 | ~0.75 × 0.37 m | |
   | op04 | ~0.60 × 0.31 m | |
   | op05 | ~1.11 × 0.78 m | |

   Within an operator the layout repeats to ±1–2 cm, so per-session calibration is recoverable.
   No single `rack.yaml` can describe the dataset.

**Who this hits:** everyone, not just this lane. With no valid rack frame there is no
`bbox_rack_xy`, no `landmarks_rack`, no `torso_vec_rack` and no zone test, so `WorldModel` runs
blind and the engine gets no usable evidence from recorded footage. T-INT, T-EVAL and the
walkthrough all depend on this.

**What this lane did about it:** `actus/models/rackfit.py`, a self-calibrating frame that derives
the plane from the markers' own printed size instead of a configured layout, learns the rest of
the rig as it sees it, and emits the frozen `RackFrame` contract so nothing downstream needs a
special case. It solves on **97.1% of frames** across all 48 takes, is rotation invariant, and
works from a single visible marker. `calibration()` returns exactly the `markers:` block a
measured `rack.yaml` would hold for that session.

**What it does not fix:** the frame is anchored to whichever marker id was seen first, and the
ids sit in different physical corners per operator. So rack coordinates are consistent *within*
a session and not comparable *across* operators. Relative features are unaffected; absolute
positions are not. This is measurable in the results (§5).

**Recommended fix, for T-SPATIAL/H-PROPS to own:** either mount the markers upright on a rigid
board and measure it once into `rack.yaml`, or add a per-session calibration step that writes a
`rack.yaml` per operator. The reconstruction code in `rackfit.py` already does the arithmetic.

### 0.2 The open-vocab detector cannot see the red and yellow boxes

Over all 48 takes at stride 2 (20,647 frames) with `detector.backend: open_vocab`:

| class | per-frame detection rate |
|---|---|
| `outer_box_closed` | 0.189 |
| `outer_box_open` | 0.207 |
| `lid` | 0.434 |
| `yellow_box` | 0.080 |
| **`red_box`** | **0.000** |

`red_box` is never detected, and the `yellow_box` hits are largely the big brown outer box
matching the prompt rather than the small yellow carton. Prompt variants ("red carton", "small
red package", "red packet") at confidence 0.05 peak at 0.06–0.14 confidence on the real box. The
props are small printed product cartons, not matte primary-coloured boxes.

**Consequence for this lane:** the HOI classifier scores a (hand, object) pair only when *both*
were observed. Pair coverage is **0.000 for red** and **0.066 for yellow**, so there is nothing
to train on. `tools/train_hoi.py` refuses to train and says why (§3).

**Consequence beyond this lane:** steps S2–S5 are about the red and yellow boxes. Without those
detections `WorldModel` cannot track them, so the rule evidence for four of the six steps has
nothing behind it either. **T-DET-TRAIN is on the critical path for the demo**, not a
nice-to-have.

**Deliberately not done:** lowering the detector confidence until red boxes appear, or training
HOI from hand features alone. The first produces detections that are mostly noise; the second
would let the model infer an object's state from where we are in the procedure, which is the
protocol prior leaking into evidence that is supposed to be prior-free. Either one would produce
a model with good-looking numbers and no meaning.

---

## 1. Files and entry points

| File | What it holds |
|---|---|
| `actus/models/features.py` | **The feature spec.** `FrameFeaturizer`, `FeatureLayout`, `FeatureParams`, `SPEC_VERSION`. One definition used by both offline extraction and live inference. |
| `actus/models/nn.py` | `MLP` and `TCN` as numpy forward passes, the torch builders they are exported from, the `.npz` format, and ONNX export. |
| `actus/models/hoi.py` | `HOIModel`, `HOIResult`. Per (hand, object) → {none, touching, holding}; `holding()` is the `{object: p}` mapping the world model takes. |
| `actus/models/temporal.py` | `TemporalModel`: streaming TCN over a fixed-rate feature grid → `ActionScores`. |
| `actus/models/fusion.py` | `EvidenceFusion`: folds the temporal model into `StepEvidence`, capped so it cannot invent a step. |
| `actus/models/runtime.py` | **`ModelStack`**, the one-call-per-frame façade for T-INT. |
| `actus/models/rackfit.py` | The self-calibrating rack frame of §0.1. |
| `actus/models/weaklabels.py` | Weak HOI labels from step annotations + protocol + geometry. |
| `actus/models/dataset.py` | Feature-cache reader, operator splits, and the metrics (frame accuracy, macro F1, segmental F1). |
| `tools/extract_features.py` | The offline pass over the takes. Writes `data/features/`. |
| `tools/train_hoi.py`, `tools/train_temporal.py` | Training, evaluation, model cards. |
| `tests/test_models_*.py` | See §6. |

Three files the card does not list. `runtime.py` is the façade that ties the lane together, the
same role `estimator.py` plays for T-SPATIAL. `rackfit.py` exists because of §0.1. `dataset.py`
and `weaklabels.py` hold logic both training tools need and that the tests should be able to
reach without importing a CLI.

## 2. How the models plug in

Order matters: the models run **before** the spatial estimator, because `WorldModel` takes `hoi`
as an input.

```python
stack = ModelStack.from_config(cfg, protocol)
out = stack.update(frame_idx, t, rack=rack, detections=dets, hands=hands, pose=pose)
sf = spatial.update(
    frame_idx, t, rack=rack, detections=dets, hands=hands, pose=pose, hoi=out.hoi
)  # sharpens <object>.held
evidence = stack.fuse(sf.evidence)  # temporal joins as one bounded source
for topic, msg in out.messages():  # publishes ActionScores on Topic.ACTIONS
    bus.publish(topic, msg)
```

Nothing here is required. With no model files present `ModelStack.enabled` is False, `hoi` is
`{}`, no `ActionScores` are published and `fuse` returns its input unchanged — the engine runs on
rule evidence exactly as it does today. `status()` reports which models actually loaded, so the
UI can show it rather than implying a model is contributing when none is.

**Contract note, no change requested:** `ModelStack` re-projects detections and hand landmarks
into rack coordinates using the public `img_to_rack`, duplicating what
`SpatialEstimator._project_detections` does privately. If T-SPATIAL would rather make those
helpers public, this lane will use them; it is a few lines either way and not worth a contract
change on its own.

## 3. Decisions the card does not state

**The feature spec is versioned and enforced.** `SPEC_VERSION` is written into every cache and
every model file, and loading refuses a model trained against a different version rather than
feeding it columns it has never seen. Change the features and you must re-extract and retrain;
the error says so.

**Rack coordinates, normalised by the rig's own extent.** Positions divide by the calibrated
board size, so a bigger rig does not shift every feature. Pixel space appears only where it is
genuinely scale-relative (hand keypoints inside a grown bounding box, grip shape over palm size).

**Rates are per second, never per frame.** Every velocity is differenced over `vel_window_s` of
wall time, so a model trained on 30 fps takes behaves the same on a 15 fps camera. Tested.

**Missing is a value.** Every block carries a presence flag and its payload is cleared when
absent, and "the detector did not run" is distinguishable from "the detector saw nothing".

**The temporal model consumes a fixed-rate grid**, not the camera's frames, so the camera rate
does not change the model's input. Streaming `update()` reproduces the offline `predict_sequence()`
exactly — including during warm-up, because the TCN is causal and a prefix gives the same answer
as the full sequence. That equality is what makes T-EVAL's offline numbers describe the live
system; it is a test, not a claim.

**Two implementations of each network, deliberately.** Training needs autograd; the runtime must
not. The numpy `MLP`/`TCN` and the torch modules they export from live side by side in `nn.py`
and a test pins them together to ~1e-7. The runtime prefers ONNX and falls back to numpy, so the
models run — and the tests cover them — on the base install with neither torch nor onnxruntime.
The `.npz` is the model of record; a failed ONNX export costs speed, not correctness.

**`other`-labelled frames are folded into `idle`** for the temporal model, because `ActionScores`
is defined over the protocol's steps plus `idle`. They stay distinct in the annotations.
Annotation labels outside the protocol (op01's takes carry stray `S7` and `S8`) become `other`
rather than being dropped or folded into a real step.

**One camera roll per take, voted once.** `RotationTTA` searches every frame, which is right live
where the roll can change, but within a take it is fixed — four landmark passes per frame for an
answer that does not move. Extraction votes once over 12 sampled frames and pins the result,
which is ~4× faster. The vote counts *detections* (hands, then poses) and uses confidence only to
break ties: confidence alone picked a different orientation for consecutive takes from the same
fixed camera, and MediaPipe is nearly indifferent between 0° and 180° while losing roughly half
its hand detections at 90°/270°. Measured effect of the fix: the choice became consistent within
an operator (op02's five takes all settle on 180° instead of scattering over 0/90/180/270), but
mean hand presence was unchanged (1.020 → 1.003 over the sampled takes). MediaPipe's hand
detector is more rotation-tolerant on this footage than expected, so this bought reproducibility,
not coverage. Recorded because the null result is worth knowing before someone retries it.

**The detector runs upright, at the image as captured**, matching what `Detector.detect` does
live. An earlier version ran it at the TTA-winning rotation and rotated the boxes back; that was
removed because it made offline features differ from online ones, which is the one thing this
lane cannot afford.

## 4. Evidence fusion, and the limit on it

    offset      = peak * (1 - current)                     # the action happened, and is over
    post_eff    = max(post,    weight * offset * (1 - visibility))
    handled_eff = max(handled, weight * offset)            # 1 if the step has no `handled` term
    fused       = max(rule, min(armed, post_eff, handled_eff))

`armed`, `post`, `handled` and `visibility` all come from the spatial lane's own `sources`.
Three bounds, each doing a distinct job:

- `armed` — pure observation that the world was seen in the step's pre-state.
- `post_eff` — **the one that matters.** The temporal model may substitute for an unobserved
  post-condition only in proportion to how blind the sensors were. At visibility 1 the
  substitution term is 0, so **with a clear view the model cannot raise a step's evidence at
  all**: a low `post` under good sight is positive evidence that the transition did not happen,
  and learned familiarity with the usual order may not overrule it. When the sensors are blind
  an unobserved `post` is uninformative, and that is when the action model earns its place.
- `handled_eff` — the model may stand in for "the object was seen in a hand", the term the
  detector most often fails to supply. Still capped by `post_eff`.

**This was tightened during implementation and the first version was wrong.** Capping only by
`armed` looked sufficient — "no temporal confidence completes a step whose precondition was never
observed" — but a *skipped* step's precondition genuinely does hold, so a maximally confident
wrong model drove S2 in the `skip_S2` replay to **0.867 against a 0.8 threshold**, caught only by
the engine's dwell requirement. With the visibility-gated `post` bound the same adversarial model
moves it from 0.210 to **0.211**. The guarantee is now structural rather than incidental.

Tested directly: `test_a_clear_view_stops_the_model_raising_anything`, a randomised bound over
200 draws, and `tests/test_models_engine_chain.py`, which drives the **real engine** over the
replay and asserts that (a) an idle action model leaves the event sequence byte-identical and
(b) a model insisting the skipped step happened, on every frame, still yields the skip.

## 5. Results

`models/temporal/card.md` and `models/temporal/metrics.json` are generated, not transcribed.
Headline, 48 takes, 20,647 frames at 15 Hz, 3 seeds per split:

| split | test operator | frame acc | macro F1 (steps) | seg F1@25 | duration baseline (macro F1) |
|---|---|---|---|---|---|
| fixed | op05 | 0.102 | 0.096 | 0.041 | 0.602 |
| loo_op01 | op01 | 0.212 | 0.128 | 0.133 | 0.492 |
| loo_op02 | op02 | 0.101 | 0.065 | 0.101 | 0.490 |
| loo_op03 | op03 | 0.150 | 0.132 | 0.066 | 0.442 |
| loo_op04 | op04 | 0.361 | 0.365 | 0.253 | 0.475 |
| loo_op05 | op05 | 0.178 | 0.141 | 0.093 | 0.575 |

**The model loses to the duration-only baseline on every fold.** That is the honest headline and
it is in the model card, not buried.

Throughput, one `update()` per frame over a warm streaming buffer: **0.07 ms/frame (ONNX)** and
**0.13 ms/frame (numpy)** on CPU, so 7,000–14,000 fps. "Runs in real time" is not the problem.

### Why, and what to do about it

A within-operator diagnostic — train and test on *different takes of the same operators*, which
leaks operator identity and the rig and is therefore not a result — reaches **0.386 frame
accuracy / 0.395 macro F1**, still below that split's 0.528 duration baseline. So this is **not
primarily a cross-operator transfer problem: the extracted features themselves are too weak.**

That is exactly what §0.2 predicts. Of the 94 per-frame features, the object channels are
effectively dead (`red_box` detected on 0.0% of frames, `yellow_box` on 8%), and hands are
present on roughly half of frames — 2–12% for op05, the fixed split's test operator, which is why
that fold is the worst of the six. **The temporal model is bottlenecked on perception, not on
architecture, seeds or training budget** — the same bottleneck that blocks HOI. Re-extract and
retrain once T-DET-TRAIN lands; `tools/train_temporal.py --loo` reproduces the whole table.

### One caveat about the baseline, for whoever writes the pitch

The duration prior scores well *because the takes are stereotyped* — six steps, same order,
similar pace — and its segmental F1 (0.83 on the fixed split) is flattered further because it
emits exactly six clean segments in the canonical order by construction.

It is also useless for the actual job. It predicts the canonical procedure whatever the operator
does, reporting every step as completed, in order, always, so it is structurally incapable of
detecting a skip or an out-of-order step. It is the right baseline to report and the wrong thing
to ship. Do not let "the baseline wins" become "we should use the baseline"; and note that
neither model is the judge — the engine decides from evidence, and §4 is what keeps a weak model
from doing damage there.

## 6. Tests

`tests/test_models_nn.py`, `_features.py`, `_runtime.py`, `_weaklabels.py`. All run on the base
install; the torch and ONNX parity tests skip themselves when those packages are absent.

The ones worth knowing about:

- numpy and torch agree to ~1e-7, for both architectures;
- the TCN is causal, and scoring a prefix equals scoring the whole sequence at that position;
- streaming `update()` reproduces `predict_sequence()` exactly;
- rack features are unchanged by a 90° camera roll, and speeds are unchanged by frame rate;
- fusion cannot exceed `armed`, for any weight, over randomised draws;
- HOI abstains on unseen objects rather than guessing;
- `rackfit` recovers a synthetic rig's layout, is rotation invariant, holds through an occlusion
  then expires, and ignores a one-off spurious marker.

## 7. Open items

- **HOI is untrainable until §0.2 is fixed.** The pipeline, weak labelling, training, export,
  card and tests are all in place and gated; `tools/train_hoi.py --coverage` reports the state.
- **200 hand-verified frames are not done** (human work). `tools/train_hoi.py
  --write-verify-sheet` draws a class-balanced sample into `data/hoi_verify/sheet.csv`; fill in
  `true_label`, save as `verified.csv`, and the run reports against it. Until then every HOI
  number is agreement with weak labels, which share the annotations' mistakes.
- **Two takes are unlabelled** and skipped by extraction: `op02_secondary_01` (a Task B take) and
  `op05_correct_01`. Their CSVs hold nothing but the initial `idle` row.
- **`op01_correct_02` never solved a rack frame** even with `rackfit` — marker 0 is not visible
  and too few others recur. It stays in the cache with `rack_valid` false throughout.
- **Pre-existing test failures, not from this lane:** `tests/test_io_process.py::
  test_managed_process_write_stdin_reaches_the_child` and `tests/test_io_streamer.py::
  test_streamer_pipes_bus_frames_to_the_ffmpeg_pusher` fail on this machine in isolation as well
  as in the full run.
- **Dependency added:** a `train` extra in `pyproject.toml` holding `onnx>=1.16`, needed only by
  `torch.onnx.export` in the training tools. `uv.lock` is updated. Not a runtime dependency —
  the models also ship as `.npz` and run on numpy alone.
