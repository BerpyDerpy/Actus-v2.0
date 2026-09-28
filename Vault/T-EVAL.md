# T-EVAL: decisions made outside the build plan

2026-09-28 · covers T-EVAL (the evaluation harness and report card: `tools/evaluate.py`,
`runs/eval/`) · read with `build-plan.md` §5 (Wave 2), `Vault/T-ENGINE.md` §5 and `Vault/T-INT.md`

The T-EVAL card defines the lane in outline. This file lists every decision made during
implementation that the card does not state, so that the lanes that read the report (T-UI, T-DEMO,
the pitch) or change what it measures (T-DET-TRAIN, T-HOI-TEMP, T-INT tuning) can rely on them or
overturn them on purpose.

**Scope, as agreed:** this is the harness. It runs any take through the offline pipeline with
whatever models the config plugs in, scores it, and writes the report card. Better models are
evaluated by changing the config (`Vault/T-INT.md` §3), not this lane. Nothing is committed.

Status at handover:
- **Nothing was run.** Per the brief, the code and tests were written but not executed: not
  `pytest`, not `ruff`, and not the harness itself. Test expectations on the replay were derived
  from T-ENGINE's documented event sequences (`Vault/T-ENGINE.md` §4) and T-INT's tests. Run
  `uv run pytest tests/test_eval.py` and `uv run ruff check . && uv run ruff format --check .`
  before relying on it.
- No dependency was added.
- One file outside T-EVAL's own was edited: `actus/app/pipeline.py` (§6). The edit is two lines
  and backwards compatible.

---

## 0. Status against the card's "done when"

| Criterion | Status |
|---|---|
| The report generates from the held-out split | **Code path written, not run.** `tools/evaluate.py` with no arguments reads the `test` split from `data/splits/report.json` and evaluates every labelled take of those operators. There are no takes on this machine, and real footage cannot be judged yet anyway (`Vault/T-INT.md` §7: no valid rack frame, no red-box detections), so a held-out report today would be honest but all-zero. |
| Every number in it can be traced to a take | **By construction** (§3): every number is summed from per-take counts, every per-take count comes from `score.json`, which names the log `seq` of every event it used, and each take directory keeps the hash-verified log, the annotations, and the clock mapping. `--rescore` rebuilds the report from those files alone. |
| Success criteria written down before the first real numbers | **Done**: `runs/eval/criteria.yaml`, v1, frozen 2026-09-28. **The targets are my proposal. Review them before the first held-out run**; after that, change them only by bumping the version (§2). |

## 1. Files and entry points

| File | What it holds |
|---|---|
| `tools/evaluate.py` | Ground truth from annotations, reading and verifying logs, scoring, aggregation, the report (JSON and Markdown), take selection, running takes, the CLI. |
| `runs/eval/criteria.yaml` | The pre-registered targets, tolerances and take selection. Committed: `.gitignore` keeps `runs/eval/`. |
| `runs/eval/<eval_id>/` | One evaluation: `eval.json`, `report.json`, `report.md`, and `takes/<take>/` (§3). |
| `runs/eval/report.json`, `report.md` | The published copy the Validation tab reads (`/api/report`). |
| `tests/test_eval.py` | See §8. |

```sh
uv run python tools/evaluate.py                                  # the held-out (test) split
uv run python tools/evaluate.py --operators op05                 # explicit operators
uv run python tools/evaluate.py --takes op05_correct_01,op05_skip_S2_01
uv run python tools/evaluate.py --replay all --seeds 0,1,2 --evidence replay   # SIMULATED
uv run python tools/evaluate.py --rescore runs/eval/<eval_id>    # rebuild from the logs only
uv run python tools/evaluate.py --override detector.backend=custom   # evaluate another model
```

Other flags: `--config`, `--override` (repeatable, passed to every run), `--answer
oracle|yes|no|none`, `--video` (annotated MP4s, off by default because `runs/eval/` is committed),
`--max-frames`, `--force`, `--publish` / `--no-publish`, `--eval-id`, `--out`.

**Python API** (tests, notebooks): `run_evaluation(inputs, eval_dir, spec=..., stages={"models":
None}, ...)`, `rescore(eval_dir, spec=...)`, `collect_takes(...)`, `replay_takes(...)`,
`derive_truth(...)`, `score_take(...)`, `build_report(...)`.

## 2. Pre-registered criteria

- They live in `runs/eval/criteria.yaml`, not in `configs/` (P0's) and not in code. `runs/eval/`
  is the lane's own directory and is committed, so the file's git history is the record of when
  a target was set.
- The file's SHA-256 goes into every report, next to its version and freeze date, so an edit
  after the fact shows.
- It holds three kinds of thing, all fixed in advance, because each one changes the numbers:
  - the **targets** (one per headline metric);
  - the **tolerances** (completion ±2 s from the card; alert ±5 s);
  - the **take selection** (the split, and excluded scenarios).
- **n/a is not a pass.** A metric with no cases (for example, no skip takes filmed among the
  held-out operators) is reported as n/a and counts against "criteria met".
- The v1 targets:

  | Metric | Target |
  |---|---|
  | step-completion precision / recall | ≥ 0.90 / ≥ 0.80 |
  | skip-alert precision / recall | ≥ 0.80 / ≥ 0.80 |
  | out-of-order-alert precision / recall | ≥ 0.80 / ≥ 0.80 |
  | false alerts per minute on correct takes | ≤ 0.5 |
  | median detection latency | ≤ 2.0 s |
  | ask rate | ≤ 1.0 per minute |
  | ask-then-correct rate | ≥ 0.90 |

  Precision on completions is set higher than recall on purpose. A missed step should fall to an
  Ask; a step verified when it didn't happen is the one failure the system must not have.

## 3. How a take is evaluated

1. **Select.** Every labelled camera-1 take in `data/raw` of the held-out operators, excluding the
   spec's `exclude_scenarios` (`secondary`, the Task B takes, which follow a different protocol).
   Takes with no CSV and takes still pending labelling (T-REC's `is_pending`) are listed in the
   report as excluded, with the reason. `--takes` names takes explicitly and bypasses the scenario
   exclusion.
2. **Run** through `actus.app.pipeline.run_offline`, which uses the same core as the live system,
   is deterministic, and never drops a frame. The run directory is
   `runs/eval/<eval_id>/takes/<take_id>/`. Whatever `pipeline.*`, `detector.*` and `models.*` say
   is what gets evaluated, and `report.md` lists the stages that actually ran (from each take's
   `run.json`).
3. **Store** next to T-INT's `events.jsonl`, `run.json` and `downlink.json`:
   - `annotations.csv`: a copy of the take's labels (for the replay, the scenario script);
   - `clock.json`: the pipeline `t` of every annotated frame, plus the first and last frame;
   - `take.json`: the take's identity, the annotation source and its SHA-256, the answer policy
     and evidence mode, and the log's SHA-256.
4. **Score from the files, never from memory.** `score_take_dir` reads only that directory. It
   verifies the hash chain first, and **a log that fails verification is refused**. It writes
   `score.json`: the ground truth it derived, every match with the log `seq` of the event used, and
   every miss, false claim, alert and Ask. The run and `--rescore` share this code path.
5. **Aggregate** by summing counts over takes (micro-averaging), then judge the totals against the
   spec.

## 4. Ground truth: decisions the card leaves open

**It comes from the annotations alone, never from the scenario name.** The filmed takes don't
always do what their name says, and the replay proves nothing about naming. The annotated step
sequence determines everything else:

- **Performance:** one label segment of a protocol step. Two segments of the same step separated
  **only by idle** are merged: that is a fumble, not a repeat. Separated by anything else (an
  `other` action, another step), they count as two performances, as in the replay's `unexpected`
  scenario, where S2 and S3 are really redone.
- **Annotated end:** the frame where the next label begins. Per the recording protocol's
  step-boundary rule that is "end state reached and hand withdrawn", so the engine usually
  crosses *before* it. Signed latencies can therefore be negative, and they are reported as such.
- **Labels outside the protocol** (op01's stray `S7`/`S8`) fold into `other`, as T-HOI-TEMP does.
  Rows past the end of the decoded video are ignored.
- **Never performed:** protocol steps with no performance, in a take that performed at least one
  step. This is the skip ground truth.
- **Incidents** mirror the engine's rule on the annotated sequence. Walk the first performances in
  order of their end: a step whose required predecessors are not all done yet raises an incident
  naming the newly missing steps. Steps already missing don't raise a second one.
  - `out_of_order` if any missing step was performed later;
  - `skip` if none was. This is the live precursor of a skip (see §5).
- **Correct take:** every protocol step performed exactly once, no incident, no `other` segment,
  and no stretch without a completion as long as the protocol timeout (`timeout_s`, including
  `engine:` overrides). Only these takes feed "false alerts per minute".
- **Clip with no procedure** (no step labels, e.g. a distractor clip): every claimed completion is
  false; there is no skip ground truth; it is not a correct take.

**Clocks.** Annotations are keyed by frame index. The pipeline's `t` for each annotated frame is
recorded during the run (`clock.json`), and scoring happens on that clock. Nothing divides a frame
index by an fps. The one exception is the simulated operator, which must answer *during* the run:
it uses the annotation CSV's own `t` column, which is the take's own timeline (§7).

## 5. Scoring: decisions the card leaves open

**Completions** are non-retroactive `StepEvent`s with a done status (`verified`, `out_of_order`,
`operator_confirmed`). A retroactive event rewrites a row; it isn't a new claim.
- **Matching** is one-to-one per step, closest first, with deterministic tie-breaks.
- An evidence completion matches when `|t_end − annotated end| ≤ 2 s`. `t_end` is the crossing
  (T-ENGINE §5), not `t`, which includes the dwell.
- **An operator-confirmed completion matches one-sidedly.** It matches a performance that ended no
  later than 2 s after it; how much earlier doesn't matter. Its `t_end` is the time of the Ask,
  and a band Ask comes `ask_after_s` (3 s) after the placement. With a symmetric window, every
  correct confirmation after a band Ask would score as one false completion plus one miss.
- **Latency** is `StepEvent.t` (the decision the operator sees, dwell included) minus the annotated
  end. Only evidence completions count; a confirmation's timing is the Ask's.

**Skip alerts** are scored per step, not per alert. A skip alert naming S2 and S3 makes two claims.
This follows T-ENGINE §5: "count a skip incident by its skip alert."

**Out-of-order alerts** are scored per incident. An alert matches an incident if it names one of
the incident's missing steps (`step_ids[1:]`) and fires within ±5 s of the annotated end of the
step that jumped ahead. When an alert matches an incident of kind `skip`, it is the live precursor
of that skip: at that moment, S4-before-S2 and skip-S2 look identical (`Vault/T-ENGINE.md` §3).
Such an alert is **neutral**. It counts toward neither precision nor recall, and it appears in
`detail.counts.ooo_alerts_crediting_a_skip`. The skip itself is scored by the skip alert. An alert
that matches nothing is a false positive.

**Asks.**
- Distinct Asks are counted by `question_id`, so a `repeat` doesn't count twice.
- The **resolution** of an Ask is the first later completion or skip of the asked step.
- It is **right** if it agrees with whether the step was performed by then (within 2 s). An Ask
  still unresolved at the end of the run is right only if the step was never performed. This makes
  "asked, the operator said yes, the step was never really done" wrong, and "asked, the step stays
  pending forever although it was done" wrong too.

**False alerts per minute** counts **every** alert kind on correct takes, including `too_fast` and
`unexpected`, because none of them should fire on a correct take.

## 6. Edit outside T-EVAL's files

| File | Change | Why |
|---|---|---|
| `actus/app/pipeline.py` (T-INT) | `run_offline(..., on_result=None)`, passed through to `Runner(on_result=...)`, which already had the hook. Default `None` = the old behaviour. | T-EVAL needs the pipeline `t` of each frame to put annotation frame indices on the engine's clock (§4). The alternatives were `frame_idx / fps` (forbidden by §2 of the plan) or re-decoding every video. |

No contract change is needed. Two proposals, neither blocking:
- **`StepStatus` "unresolved at end".** T-ENGINE §7 already floats this. Today a step still asked
  about at `finish()` stays `pending`, which the report shows as `p`. It is scored correctly
  either way.
- **A `take_kind` field in T-REC's take JSON** (procedure / distractor / sweep). The harness infers
  "no procedure" from the absence of step labels, which is right for distractor clips but can't
  tell a distractor clip from a procedure take that was never labelled. Unlabelled takes are
  excluded by `is_pending`, so this only matters if someone labels a procedure take entirely as
  `other`.

## 7. Open items and known limitations

- **Nothing here has been executed** (see the status above).
- **Real footage can't be judged yet**, for the upstream reasons in `Vault/T-INT.md` §7 and
  `Vault/T-HOI-TEMP.md` §0. The harness will run and report honestly, but on today's rig every
  step stays `pending`, so recall is ~0. Fix the rig or calibrate `rack.yaml` per session, land
  T-DET-TRAIN, then run `tools/evaluate.py`.
- **The simulated operator is an oracle.** It answers truthfully from the annotations, so
  ask-then-correct measures what the engine does with a truthful answer, not how real operators
  answer. The report states the policy on every page. `--answer yes|no|none` gives the other
  bounds: `none` shows what happens with nobody answering.
- **The oracle's clock.** The oracle compares Ask times against the annotation CSV's `t`, because
  it has to answer before the frames after the Ask are processed. For T-REC takes that is the
  take's own timeline and matches the container clock to within capture jitter. The window of
  `answer_delay_s + completion_tol_s` absorbs it. Scoring doesn't use this clock.
- **Threshold tuning must not use this split.** T-INT tunes the decision thresholds on
  training-operator takes (card). If you want the harness for tuning, run it with `--operators` on
  training operators into a separate `--eval-id`, and never publish that report.
- **`--video` is off by default**, because `runs/eval/` is committed and annotated MP4s are large.
  Per-take `events.jsonl`, annotations, clock and score files are small and are meant to be
  committed with the report; they are what makes the report traceable.
- **Small n.** With one held-out operator (op05) and about 16 takes, a single take moves a ratio
  by several points. The report adds a caveat whenever fewer than 10 takes are evaluated. If time
  allows, leave-one-operator-out (as T-HOI-TEMP's `--loo` does) would give a spread; it isn't
  implemented.
- **The Validation tab** (T-UI) renders the report generically: top-level strings and numbers
  become cards (numbers with 3 decimals, which is why counts are strings), top-level arrays become
  tables, and `detail` is ignored. The report is shaped for that (§8 tests it). The first card is
  `data`, which reads `SIMULATED replay: …` for the replay.
- **Publishing:** a real evaluation overwrites `runs/eval/report.{json,md}` by default. A SIMULATED
  one only does so with `--publish`, so the tab never shows replay numbers unless someone asks it
  to (and then it says SIMULATED).

## 8. Tests (`tests/test_eval.py`, written, not run)

- **Ground truth:**
  - a correct take;
  - skip vs out-of-order incidents;
  - merging fumbles, and folding `S7` into `other`;
  - rows past the end of the video are ignored;
  - a 67 s pause makes a take not correct;
  - label rows round-trip;
  - `FrameClock` is exact at known frames and interpolates elsewhere.
- **Scoring:**
  - one-to-one matching within ±2 s, with retroactive events ignored;
  - the one-sided window for confirmations;
  - the live out-of-order precursor is credited to the skip;
  - out-of-order alerts matched per incident and in time;
  - every alert on a correct take is false;
  - ask-then-correct, with a `repeat` counted once and an unresolved Ask scored wrong;
  - a clip with no procedure;
  - metrics with no cases are `None`, not 0.
- **Spec, log, report:**
  - the spec covers exactly the headline metrics;
  - a tampered log is refused;
  - the report is flat where the UI reads it, and is JSON-serialisable.
- **End to end on the replay** (`--evidence replay`, no models):
  - `correct` + `skip_S2` + `S4_before_S2` score 1.0 on completions, skips and out-of-order, with
    no false alerts;
  - every per-take file exists, and `rescore` reproduces the report;
  - `occluded` with the oracle ends with S3 `operator_confirmed`, and ask-then-correct is 1.0;
  - an existing evaluation isn't overwritten without `--force`.
