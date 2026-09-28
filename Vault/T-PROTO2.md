# T-PROTO2: decisions made outside the build plan

2026-09-28 · covers T-PROTO2 (second protocol from YAML only + a demo run) · read with
`build-plan.md` §4–5, `Vault/T-ENGINE.md` and `Vault/T-SPATIAL.md`

The T-PROTO2 card says: "second protocol from YAML only (`sample_mixing.yaml`: cup, bottle, book;
standard COCO classes; Task B in the recording doc) + a demo run". This file records every
decision made while doing it that the card does not state, so that T-INT, T-EVAL, T-HOI-TEMP,
T-DEMO and T-PKG can rely on them or overturn them on purpose.

**Scope, as agreed:** the harness; nothing is committed. **No code was run** while writing this
lane (that was the instruction), so nothing below is verified; §0 says what to run first and
which assertions are the likeliest to fail. **No other lane's file was edited** and no contract
changed (proposals are in §7).

## 0. Status against the card, and what to run first

| Deliverable | Status |
|---|---|
| `protocols/sample_mixing.yaml` (cup, bottle, book, COCO classes) | Written. Validated by reading against `actus/protocol/schema.py` and the world-model vocabulary, not by loading it. |
| "From YAML only": nothing outside the YAML changes | Held to. A config overlay `configs/sample_mixing.yaml` selects the protocol and the detector (a config file, not code). No engine, spatial, pipeline or UI file was touched. |
| A demo run | `tools/demo_protocol2.py` and `scripts/run_demo_protocol2.sh`: five SIMULATED scenarios through the real `PipelineCore` (§4). **Unrun.** |
| Tests | `tests/test_proto2_protocol.py` (schema, design rules, config, and the real engine on hand-built evidence) and `tests/test_proto2_demo.py` (scene geometry and the runs). **Unrun.** |

Run these, in this order:

```sh
uv run ruff check . && uv run ruff format .          # formatted by hand; expect small diffs
uv run pytest tests/test_proto2_protocol.py          # no cameras, no cv2 scene, fast
uv run pytest tests/test_proto2_demo.py
uv run python tools/demo_protocol2.py all --no-video # read the output against §5
uv run python tools/demo_protocol2.py occluded_S5    # writes the annotated MP4 to runs/
chmod +x scripts/run_demo_protocol2.sh               # written on Windows: no executable bit
```

Most likely to fail, because each rests on an assumption about another lane that I could not
check by running it:

1. `test_correct_run_verifies_every_step_with_no_alerts_or_asks` (and everything that reads
   `runs("correct")`). It needs the hand-driven world model to carry all five transitions through
   `p_enter` for `dwell_s`. T-SPATIAL proved this for a single carry (`test_spatial_world.py`) and
   for the replay's hand-less scene, but never the two together over a whole procedure. If it
   fails, look at `python -m actus.spatial.estimator`-style output of the per-step evidence first
   (`Timeline` in the tool has the ground-truth times).
2. `test_a_blocked_view_makes_it_ask_and_a_spoken_yes_is_operator_confirmed`. It needs S4 to be
   resolved before the blackout starts (0.4 s after the book's arrival; the carry is 2.0 s and
   S4's evidence should cross `p_enter` about 0.5 s into it), and it needs the world model's
   visibility to fall under 0.5 within a second so the engine's blindness timer runs (§5).
3. `test_an_uncallable_placement_is_never_verified`. It only asserts the safe half (not
   verified); whether an Ask is also issued depends on the belief settling inside the ask band
   [0.45, 0.8) for 3 s. `Vault/T-SPATIAL.md` pins "a box on a zone edge reads 0.3–0.7", which
   straddles the band's lower edge.
4. `test_the_hand_never_passes_close_to_an_object_it_is_not_handling`: I checked the paths by
   hand (closest approach 0.136 m, cup withdrawing past the bottle in `bottle_first`); an
   arithmetic slip would show here first.
5. Style: `ruff format` was not run.

## 1. The procedure (protocols/sample_mixing.yaml)

**The recording doc could not be opened** (the same artefact `Vault/Wave0.md` records as
unreadable), so the steps below are **our reading of Task B**, in the spirit of assumption A4.
If Task B differs, edit the YAML only; keep the step ids `S1`…`S5` because T-REC's annotation keys
`1`–`9`, the CSV labels and T-EVAL all use them.

| Step | Text (spoken as-is) | pre → post |
|---|---|---|
| S1 | Place the cup on the mixing pad | `cup.in_zone_M` becomes true |
| S2 | Take the bottle from the stowage area | `bottle.in_zone_S` true → false, `requires_hoi: bottle` |
| S3 | Place the bottle next to the cup | `bottle.in_zone_M` becomes true |
| S4 | Take the book from the stowage area | `book.in_zone_S` true → false, `requires_hoi: book` |
| S5 | Place the book in the record area | `book.in_zone_R` becomes true |

`order: strict`, the box protocol's timing and decision numbers unchanged (`dwell_s 0.8`,
`p_enter 0.8`, `p_exit 0.6`, ask band [0.45, 0.8), `ask_after_s 3.0`). The placement steps carry a
tailored `ask:` text, as the box protocol's S3 and S5 do.

**Pouring is not observed.** Every predicate is a *place*: nothing in COCO geometry says a liquid
moved, and the world model only supports `held`, `in_zone_<Z>` and `in_<container>`/`open`. So the
protocol verifies the staging around the (unseen) pour, and the title says so ("stage the cup and
bottle, log the book"). That belongs on the limitations slide (§8).

**Design rule: every post-state persists to the end of the run.** This is the one non-obvious
constraint, and `test_post_states_persist_to_the_end` enforces it. The engine flags a completed
step whose post-state is later undone as `unexpected`, unless a later step with a contradicting
post is *completing at that instant* (`ProtocolEngine._undo_expected`). A carry between two places
is never simultaneous: the bottle leaving the mixing pad would raise an `unexpected` alert for S3
in the middle of the operator's next move. So there is no "put the bottle back" step. A pick-up is
expressed, as in the box protocol, as *leaving the source* (`bottle.in_zone_S: false`), which stays
true afterwards. Anyone extending this protocol with a return step should include the
contradicting predicate in that step's `post` **and** expect the timing race described above.

**Zones** (rack metres): `M` mixing pad `[[0.10,0.30],[0.45,0.60]]`, `R` record area
`[[0.55,0.10],[0.80,0.35]]`, `S` stowage `[[0.55,0.45],[0.80,0.70]]`. `R` and `S` are the box
protocol's Zone A and Zone B tape, so H-PROPS re-tapes only `M` (its footprint is where the outer
box used to sit). A test checks that none of the three overlaps another or a marker.

**Initial state:** the bottle and book start in `S`, the cup starts beside the board (in no zone).
So `initial` says `cup.in_zone_M: false`, the bottle and book are in `S`, nothing is held.

**Step phrasing comes from the YAML alone.** T-ENGINE builds alert text from the predicate names
(`Vault/T-ENGINE.md` §3). For this protocol that gives, for example: "Out of sequence: bottle moved
out of Zone S before cup. Expected: place the cup on the mixing pad." and "Skipped: place the cup
on the mixing pad." (`_step_phrase` reads the *first* `post` key; all steps here have one key, so
that is unambiguous.) The engine tests assert these strings.

## 2. Detector: COCO through the existing `custom` backend

`cup`, `bottle` and `book` are COCO classes, so the stock YOLO11n COCO checkpoint detects them
with no training. It plugs in as `detector.backend: custom` with `custom_weights:
models/coco/yolo11n.pt`; `_to_detection_set` reports `result.names`, which are the COCO names, and
the protocol's `objects:` map those to `cup`/`bottle`/`book` directly. Other COCO detections
(`person`, `chair`, …) are ignored by association: the world model only looks at each object's
declared classes.

- **Setup download** (runtime stays offline): `uv run --extra perception python -c "from
  ultralytics import YOLO; YOLO('models/coco/yolo11n.pt')"`. Ultralytics fetches the file to that
  path when it is missing. It is in the header of `configs/sample_mixing.yaml`; T-PKG should bundle
  it with the other weights. AGPL-3.0, like the rest of the stack.
- **Why not `open_vocab`:** `detector.prompts` is a fixed mapping in `default.yaml`. A config
  overlay can *add to* it but the strict loader rejects new keys under a non-empty mapping, and it
  can't *remove* the BAS prompts. Proposal in §7.
- COCO's `book` and `cup` are the classes least likely to be robust on a webcam at a slant, and a
  cup seen from above may be labelled `bowl`. **Untested on footage.** If Task B footage shows it,
  the fix is data, not code: the same `tools/extract_frames.py` → `autolabel.py` → train chain with
  a three-class dataset.

## 3. Config overlay (configs/sample_mixing.yaml)

Only three settings: `protocol.path`, the detector above, and `models.enabled: false`.
`load_config` is strict (every key must exist in `default.yaml`), and a test loads it that way.
`models.enabled: false` because the HOI and temporal weights were trained on the box protocol's
features and steps; loading them here would be wrong, not just useless. `ModelStack`'s feature
layout is derived from the protocol (`FeatureLayout.from_protocol`), so retraining for Task B would
need no code change, only Task B annotations. That has not been checked against `train_hoi.py` and
`train_temporal.py`, which I did not open.

## 4. The demo run (tools/demo_protocol2.py)

`ReplaySource` (P0) only simulates the box world and refuses any protocol whose predicates it
can't simulate, so it cannot demo this protocol, and it isn't mine to change. The tool brings its
own small scene instead:

- a scripted operator (`Act(obj)`: reach, grab, carry 2 s, let go, withdraw) over a rack board
  drawn with **real ArUco markers**, using the replay's camera map (`rack_to_px`, 640×480);
- scripted **detections** (pixel boxes only, with position noise σ = 3 mm and confidence
  0.90 ± 0.03) and **hand keypoints** (21 points within 2 cm of the hand). The spatial lane has to
  solve the rack from the rendered markers and project both, as it does for real perception;
- everything downstream is the real code: `PipelineCore` (rack solve, `WorldModel`, `EvidenceModel`,
  `ProtocolEngine`, overlay), `Runner`, the hash-chained `EventLog`, the downlink packet, the
  annotated MP4. `SceneHands` is a `handpose` plug-in (the pipeline's documented extension point);
  the detections go through `SourceFrame.detections`, which replaces the detector, as the replay does.

What is simulated is only the camera and the neural perception, and it is labelled: every frame is
watermarked SIMULATED, the overlay's SIMULATED tag is on (`simulated=True`), and `run.json` says
`"simulated": true` with a note. **Nothing this tool prints is a metric for the report card.**

`run_scenario` duplicates the ~50 lines of plumbing in `run_offline` (run dir, log, downlink,
`run.json`), because `run_offline` only accepts a video file or `replay:<scenario>`. Proposal in
§7. Output is deterministic for a seed; the log clock is 1970-01-01 + the scene's `t`, as an
offline run of a take without metadata.

## 5. Scenarios and the behaviour to expect

`python tools/demo_protocol2.py --list`. All run about 20 s of scene time at 15 fps.

| Scenario | What happens | Expected result |
|---|---|---|
| `correct` | cup, bottle, book | S1…S5 `verified`; no alerts, no asks; prompts S1…S5 then "Procedure complete." |
| `skip_cup` | the cup is never placed | `out_of_order` when the bottle is taken (S2); at the end S1 `skipped` (retroactive) and one `skip` alert: "Skipped: place the cup on the mixing pad." |
| `bottle_first` | bottle staged before the cup | one `out_of_order` alert ("bottle moved out of Zone S before cup"), then the cup is done and the run completes; no skip |
| `unsure_S5` | the book is left exactly on the record area's edge | S5 evidence ≈ 0.5: **never `verified`**; an Ask about S5 if the belief sits in the band long enough |
| `occluded_S5` (answer "yes") | a helper blocks the view 0.4 s after the book arrives, for 6 s | a blindness Ask about S5 in the YAML's words; the scripted "yes" gives S5 `operator_confirmed` |

The occlusion is a real one for the pipeline: during the window there are no detections and no
hands (the hand model "ran and saw nothing"), and the frame shows a dark occluder over the work
area, with the four corner markers left visible so the rack frame stays solved. This is the same
mechanism `Vault/T-SPATIAL.md` §8 describes for the replay's `occluded`.

**The scene is built so that no hand comes within 10 cm of an object it is not handling.**
The start and end positions are inside their zones with at least 4 cm of margin, except the
deliberate edge case. This isolates the procedure logic from hand-brush noise, which is a different
(and worthwhile) test of the world model on real footage. The test asserts the 10 cm.

## 6. Running it on real Task B footage (when it exists)

```sh
uv run python tools/run_offline.py data/raw/op05_secondary_01.mp4 --config configs/sample_mixing.yaml
uv run python tools/evaluate.py --config configs/sample_mixing.yaml --takes op05_secondary_01,op05_secondary_02 --no-publish
scripts/run_demo_protocol2.sh --live       # camera; scripts/run_demo.sh with the overlay
```

- The step-annotation CSVs for Task B takes use `S1`…`S5` for the same reason as §1. T-HOI-TEMP
  noted `op02_secondary_01` as unlabelled; it needs labelling in T-REC's playback mode before it
  can be scored.
- `tools/evaluate.py --takes` bypasses the spec's `exclude_scenarios: [secondary]`, so a Task B
  evaluation needs no change to `runs/eval/criteria.yaml` (which is frozen). The criteria's
  targets were pre-registered for the box protocol; a Task B report should say it is a
  generalisation check on a handful of takes, not a scored benchmark.
- A live run needs `models/coco/yolo11n.pt` (§2) and the rack measured (`configs/rack.yaml`,
  `measured: false` still): the zones assume the box protocol's board, and `M` needs taping.

## 7. Proposals for the integrator and other lanes (none applied)

1. **T-PERC / `default.yaml`:** make `detector.prompts` a free-form section (`{}`), or derive
   the open-vocab prompts from the protocol's `objects:`. Then a second protocol could use
   `open_vocab` for objects that COCO doesn't have.
2. **T-INT:** let `run_offline` (or a sibling `run_source`) accept a pre-built `FrameSource`
   plus its `simulated` flag and log clock. `tools/demo_protocol2.py::run_scenario` would shrink
   to building the source.
3. **P0 / replay:** a protocol-generic scene (the tool's `Timeline`/`SceneSource` is one; it could
   move to `actus/capture/` and grow hands, as `Vault/T-SPATIAL.md` §8 already proposes for the
   replay) would let the engine CLI and T-EVAL's `--replay` run any protocol.
4. **T-ENGINE:** `_step_phrase` reads only the first `post` key. If a later protocol has a
   multi-key post, name the *placement* first, or teach the phrase builder to skip the negated
   key.
5. **T-UI:** the UI's mini-map and checklist are protocol-driven in principle; nobody has looked at
   it with a protocol other than the box one. Check `ui/` with `--config configs/sample_mixing.yaml`
   once T-UI can run.

## 8. Limitations to state honestly

- The protocol verifies where things are, not what was poured. Pouring itself is invisible to it.
- The steps are our reading of Task B; the recording doc was unavailable when this was written.
- The demo run is a simulation. It shows the sequencing, alerts, ask path and logs on a procedure
  nobody wrote code for; it is not evidence about detection or pose quality on real footage.
- COCO's `cup`/`book` detections are untested on this rig; the numbers on real takes may need a
  small custom dataset.
- The hand-set thresholds (world model, decision layer) are the box protocol's, unchanged. They
  were tuned on nothing for this protocol, which is the point of a generalisation check and also
  its weakness.
