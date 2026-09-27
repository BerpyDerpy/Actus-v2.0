# T-ENGINE: decisions made outside the build plan

2026-09-27 · covers T-ENGINE (protocol engine + decision layer) · read with `build-plan.md` §4–5
and `Vault/Wave0.md`

The T-ENGINE card defines the engine's rules only in outline. This file lists every decision made
during implementation that the card does not state, so that the lanes that consume the engine
(T-INT, T-UI, T-VOICE, T-IO, T-EVAL) can rely on them or overturn them on purpose.

Status at handover: all T-ENGINE tests pass (116 across 4 files), and so do the P0 tests they sit
beside. `ruff check` and `ruff format --check` are clean on every T-ENGINE file. Nothing is
committed. No P0-owned file was edited: contracts, schema, config and `pyproject.toml` are
unchanged, and no dependency was added (`hypothesis` was already in the dev group).

---

## 1. Files and entry points

| File | What it holds |
|---|---|
| `actus/protocol/decision.py` | `DecisionParams`, `DwellGate`, `SustainTimer`, `Candidate`, `order_simultaneous`. Pure, clock-driven, no protocol order. |
| `actus/protocol/engine.py` | `ProtocolEngine`, `run_ticks`, `EngineRunner`, `publish`, `describe`, and a CLI. |
| `tests/test_engine_decision.py` | Gate, timer, tie-break and parameter unit tests. |
| `tests/test_engine_replay.py` | Exact event sequences for every replay scenario × 5 seeds, plus timing, messages, too_fast, and the bus runner. |
| `tests/test_engine_unit.py` | Hand-written evidence streams, one rule per test. |
| `tests/test_engine_properties.py` | The three hypothesis properties from the card. |

```sh
uv run python -m actus.protocol.engine --scenario skip_S2
uv run python -m actus.protocol.engine --scenario occluded --answer yes
uv run pytest tests/test_engine_*.py
```

**API for integrators**

- `ProtocolEngine(protocol, params=None)`. `params` defaults to `DecisionParams.from_protocol`.
  - `update(frame_idx, t, evidence, *, visibility=None, actions=None) -> list[Output]` processes
    one frame. `evidence` is an iterable of `StepEvidence`, or a mapping from step id to
    `StepEvidence`. `visibility` is a float or a `Predicates`. A step with no new evidence keeps
    its last value. Evidence for unknown steps is ignored.
  - **`frame_idx` must strictly increase, and `t` must not decrease. Either violation raises
    `ValueError`.**
  - `answer(Answer)` and `command(Command)` take effect immediately. Their outputs are stamped
    with the last processed frame's `frame_idx` and `t`, not with `Answer.t`.
  - `finish()` is the end-of-stream call.
  - Queries: `statuses`, `finished`, `open_ask`, `prompt_target`, and `outcomes()`. `outcomes()`
    is the step outcome table that T-IO's run summary and downlink packet need.
  - The engine is **single-threaded**. Drive it from one thread.
- `Output = StepEvent | Alert | Prompt | Ask`. `OUTPUT_TOPICS` maps each type to its bus topic,
  and `publish(bus, outputs)` publishes them.
- `run_ticks(engine, ticks, *, answer=None|"yes"|"no"|callable, answer_delay_s=1.0,
  use_actions=True, use_visibility=True, finish=True)` drives the engine from `ReplaySource`
  ticks, or anything shaped like them. It is meant for tests, T-EVAL and offline runs, and it
  can script the operator's answers.
- `EngineRunner(engine, bus)` drives the engine from the bus:
  - it subscribes to `evidence`, `predicates`, `actions`, `answers` and `commands`, and
    publishes to `events`, `alerts`, `prompts` and `asks`;
  - it processes a frame as soon as evidence for every step has arrived, or when evidence for a
    later frame arrives;
  - late evidence is counted in `runner.late`;
  - visibility and action scores are matched to the latest frame at or before the evidence
    frame;
  - `run(stop=None)` blocks until the bus closes, then flushes and calls `finish()`. Outputs
    produced after the bus closes are returned but can't be published;
  - **`pump()`** processes whatever is queued without blocking, for a single-threaded
    `run_offline.py` loop.
- `describe(output)` returns a one-line ASCII form such as `event S2 skipped (retro)` or
  `alert out_of_order S4,S2,S3`. The tests and the CLI use it.

## 2. Decision parameters

- `DecisionParams.from_protocol(proto, **overrides)` takes these from the protocol YAML:
  `p_enter`, `p_exit`, `dwell_s`, `ask_band`, `ask_after_s`, `timeout_s` and `min_duration_s`.
- **The engine adds defaults that are not in the YAML schema**, which has `extra="forbid"` and
  is owned by P0:

  | Param | Default | Meaning |
  |---|---|---|
  | `vis_low` | 0.5 | `Predicates.visibility` below this counts as blind |
  | `vis_ask_after_s` | 1.5 | Blind this long while a step is expected → Ask |
  | `tie_eps` | 0.05 | Evidence within this is a tie (the prior breaks it) |
  | `action_min` | 0.5 | `ActionScores` top score needed to count as "step in progress" |
  | `ask_cooldown_s` | 10 | After "no" or cancel, don't re-ask that step for this long |
  | `ask_supersede_s` | 15 | An unanswered Ask older than this may be replaced (matches T-VOICE's ask TTL) |

  Two more thresholds are derived rather than set. **Undo detection mirrors completion on
  `1 − p`**: the evidence must fall to ≤ `1 − p_enter` (0.2) and stay ≤ `1 − p_exit` (0.4) for
  `dwell_s`.
- **Proposal for the integrator:** add an `engine:` section to `configs/default.yaml` for these
  overrides, and have T-INT pass it as `DecisionParams.from_protocol(proto,
  **cfg.engine.to_dict())`. I didn't add it, because P0 owns `configs/*`.
  `vis_ask_after_s = 1.5` was chosen shorter than `ask_after_s = 3.0` because an occlusion
  during the demo lasts only a few seconds. With 3 s, the ask would come just as the view
  returned.

## 3. Engine semantics the card leaves open

**Completion (evidence only)**
- A step completes when its `p_complete` crosses `p_enter` and then stays ≥ `p_exit` for
  `dwell_s`. Completion gates also keep running for skipped and timed-out steps.
- The resulting `StepEvent` is filled in as follows:
  - `t_end` is the frame where the evidence crossed `p_enter`, i.e. when the transition was
    observed. `t` is when the dwell finished (the decision time).
  - `confidence` is the mean `p_complete` over the dwell window.
  - `evidence` is the step's `sources` on the decision frame, or `{"p_complete": p}` if
    `sources` is empty.
  - `t_start` is the onset of the step in `ActionScores`: the first frame where the step is the
    top label with a score ≥ `action_min`. The search starts at the latest `t_end` of any other
    completed step that is not after this step's `t_end`.
  - With no action evidence, `t_start` is that reference time itself. That makes `too_fast`
    conservative when no temporal model is running: it can only under-report.
- **Same-frame completions are ordered by evidence first**: earlier crossing, then higher
  evidence. Evidence within `tie_eps` is a tie, broken by the prior. This is the only place the
  prior touches completions, and it only changes the order within one frame.

**Out of order vs. skip (the key design choice)**
- A step is **flagged missing** when a step that requires it has completed while it was pending,
  **and** its own evidence is *clearly low*. Clearly low means all of these:
  - below `ask_low` for `dwell_s`;
  - the sensors are not blind;
  - there is no open Ask about the step.

  An unsure step, with evidence in the band or an open Ask, is never accused. Newly flagged steps
  produce one `out_of_order` alert. Its `step_ids` are the triggering completed step, then the
  missing steps in protocol order.
- A completed step's status is `out_of_order` if any step it requires is still flagged missing.
  Otherwise it is `verified`.
- **A skip is decided only when the procedure can go no further**, that is, when every step is
  complete, skipped or flagged. At that point the flagged steps become `skipped` with
  `retroactive=True`, and one `skip` alert covers them. `finish()` does the same at end of
  stream.
  - Why: at the moment S4 completes, "skip S2" and "S4 before S2" are indistinguishable. So a
    skip surfaces live as `out_of_order`, with the card's message ("yellow box removed before
    red box"), and is confirmed as a skip when S6 closes the box. `skip_S2` and `S4_before_S2`
    produce the same live alert.
  - Consequence for the outcome table: in `skip_S2`, S4 and S5 end as `out_of_order`, and S6
    ends as `verified` because its predecessors had been resolved as skipped by then.
  - A step that is flagged and skipped in the same frame (e.g. jumping from S1 straight to S6)
    gets only the `skip` alert, not an `out_of_order` alert as well.
  - A skipped step that is done later becomes `out_of_order`, with no new alert.
- A reopened step (see Unexpected below) is only flagged against completions that came *after*
  it reopened, so undoing S2 after S5 doesn't accuse S4 and S5 retroactively.

**Unexpected**
- Detected from evidence: a completed step's post-state is undone (the undo gate above) while the
  sensors can see. This relies on T-SPATIAL's evidence **un-latching** when the post-state is
  undone, which is what replay does and what `CLAUDE.md` implies.
- **Legitimate undos are exempt.** Each step has a list of "undoers": steps whose `post`
  contradicts its `post` (S6 "close" contradicts S1 "open"). An undo is not unexpected if an
  undoer completed after the step, or is completing now. Reopening the box after S6 therefore
  *is* unexpected ("Unexpected: outer box opened."), because S1 completed before S6.
- The undone step reopens: a `StepEvent` with status `pending`, confidence `1 − p`, and no times.
- **Incidents are grouped by object.** Undos of steps that share an object, while an incident is
  open, give one alert. The incident closes when all of its reopened steps are done again. In
  the `unexpected` replay scenario, S3 is undone first and S2 about 1.5 s later: one alert
  ("Unexpected: red box moved out of Zone A."), two pending events.
- A step completed by operator confirmation is only watched for undo after its evidence has been
  seen ≥ `p_enter`.

**Timeout and too_fast**
- Timeout is measured from the last *progress*: a completion or a reopen (the engine start
  counts as the first). After `timeout_s` with no progress, the prompted step gets `timed_out`
  (if it was pending) and one alert. The timer re-arms only on progress. There is no timeout
  once the procedure is finished. A timed-out step that later completes becomes `verified` or
  `out_of_order` as usual.
- `too_fast`: `t_end − t_start < min_duration_s[step]` gives one alert. The status stays
  `verified`. **It is not checked for operator-confirmed steps**, because their `t_end` is only
  the time of the ask.

**Asks**
- **Band ask:** a step that is not done has evidence in `[ask_low, ask_high)` for `ask_after_s`
  while the sensors see. Short gaps are tolerated, up to a grace of `dwell_s`. Any open step can
  be asked about; when several qualify, the prior picks which comes first.
- **Blindness ask:** blind for `vis_ask_after_s` while the procedure is not finished. It asks
  about the *prompted* step (the prior chooses what to ask), once per blind episode.
- Band asks and missing-step flagging are both suspended while the sensors are blind.
- **One open Ask at a time.** An unanswered Ask can be superseded after `ask_supersede_s`.
  Question ids are `q<n>-<step>`, e.g. `q1-S3`. The text is the step's `ask:` from the YAML if it
  has one, else "I couldn't confirm it. Did you <step text>?".
- "yes" gives `operator_confirmed`:
  - `t_end` is the time of the ask;
  - `confidence` is the machine's own `p_complete`, reported honestly;
  - `evidence` is the sources plus `"operator": 1.0`.
  - The out-of-order rules still apply to the confirmed step.
- "no" re-prompts the step, with a `Prompt` for that step, and then cools down for
  `ask_cooldown_s`.
- **An operator-confirmed step is never upgraded to `verified`** when evidence arrives later.
- **Asks have no explicit "withdrawn" message** (see §6). The convention is that an Ask is closed
  by any later `StepEvent` for its `step_id`, or by being superseded. The engine ignores answers
  to closed questions.

**Prompts**
- A prompt is issued:
  - on the first frame;
  - after every completion or skip;
  - whenever the prompted step changes, e.g. after a reopen;
  - after "no", as a one-off re-prompt that doesn't move the prompt target.
- The target is the first step, in protocol order, of `expected_next(done ∪ skipped)`. Flagged
  steps stay in that list, so after S4-before-S2 the prompt keeps saying "Take out the red box".
- `Prompt.text` is the step's text exactly as written. At the end, `next_step_id=None` with the
  text "Procedure complete." or "Procedure finished with N skipped step(s).".
- If every other step is resolved but an unsure step is still open, the prompt keeps pointing at
  that step. In `unsure_S3` with no answer, the run ends on "prompt S3" and S3 stays `pending`.
  That is honest: the step is neither verified nor confirmed.

**Commands**
- `repeat` re-issues the open Ask with the same `question_id`, or else the prompt.
- `next` and `whats_next` re-issue the prompt. **They never mark a step done**, because the
  engine is the judge.
- `confirm` means "yes" to the open Ask. `cancel` dismisses it, with the cooldown and no
  re-prompt.

**Output order within one call**
- Outputs come out in this order: completion events, reopen events, retroactive skip events,
  then alerts (out_of_order, unexpected, too_fast, skip), then the prompt, then timeout (the
  event and the alert), then the Ask.

**Messages** (short, spoken as-is; ASCII except the step text)
- `out_of_order`: "Out of sequence: <what was done> before <missing step's object>. Expected:
  <missing step>." This matches the card word for word for S4-before-S2. When the two steps
  share an object, it reads "Out of sequence: <what was done>. Expected: …".
- `skip`: "Skipped: take out the red box; place the red box in Zone A."
- `unexpected`: "Unexpected: red box moved out of Zone A."
- `timeout`: "Timeout: no progress for 60 seconds. Next step: take out the yellow box."
- `too_fast`: "Too fast: place the red box in Zone A took 0.6 s (minimum 1 s). Please check it."
- The phrases come from the predicate names:
  - `in_zone_<Z>` → "placed in Zone Z" / "moved out of Zone Z";
  - `open` and `closed`, and `held` → "picked up" / "put down";
  - `in_<container>` → "removed" / "put back in the <container label>";
  - anything else → "<label> <pred>" / "<label> no longer <pred>".

  A second protocol therefore gets usable messages from its YAML alone. A test checks this with
  a cup/bottle protocol.

## 4. Verified behaviour

**Expected sequences** (compact form; `tests/test_engine_replay.py::EXPECTED`)
- The exact sequence is asserted for every replay scenario at noise 0.04 over seeds 0–4. A
  manual sweep also found it identical over seeds 0–9 at noise 0.04 and 0.08, for every scenario
  except `unsure_S3`.
- `correct`: P S1 · E S1 verified · P S2 · … · E S6 verified · P –
- `skip_S2`: … E S4 out_of_order · A out_of_order S4,S2,S3 · P S2 · E S5 out_of_order · P S2 ·
  E S6 verified · E S2 skipped (retro) · E S3 skipped (retro) · A skip S2,S3 · P –
- `S4_before_S2`: … E S4 out_of_order · A out_of_order S4,S2,S3 · P S2 · E S5 out_of_order ·
  P S2 · E S2 verified · P S3 · E S3 verified · P S6 · E S6 verified · P –
- `unsure_S3`: … P S3 · Q S3 · E S4 verified · P S3 · … With the answer "yes", E S3
  operator_confirmed follows the answer.
- `occluded`: … P S3 · Q S3 (blindness ask) · E S3 verified (the view returns) … With the answer
  "yes", E S3 operator_confirmed.
- `unexpected`: … E S3 verified · P S4 · E S3 pending · A unexpected S3 · P S3 · E S2 pending ·
  P S2 · E S2 verified · … P –
- `timeout`: … P S4 · E S4 timed_out · A timeout S4 · E S4 verified · …

**Robustness**
- At twice the replay noise (0.08), `unsure_S3` legitimately varies: sometimes the evidence
  really does hold above 0.8, and sometimes the answer lands before or after S4. The test checks
  only the invariants: S3 is always asked about, and there are no alerts.
- Before the "clearly low for `dwell_s`" rule and the band grace were added, one seed in ten
  produced a false out_of_order/skip on S3. That is why both rules exist.

**Latency** (15 fps replay)
- The decision comes `dwell_s` (0.8 s) after the crossing. The crossing comes 0.1–0.3 s after the
  true state change, because of replay's predicate lag.
- `t_start` is within one frame of the true step start when `ActionScores` are available.

**Property tests** (150 random streams each)
- No evidence-based completion without a crossing above `p_enter` held for the dwell.
- The same completions (frame, step, `t_end`) under a strict order, the reversed order, and no
  order at all.
- Each alert fires once per incident.
- A sample of 300 streams hit every alert kind, reopens, skips and asks, so the properties are
  not vacuous.

## 5. Notes for other lanes

- **T-SPATIAL:** evidence must fall back when a post-state is undone, because the unexpected
  alert depends on it. It must also regress toward 0.5, or be accompanied by low `visibility`,
  under occlusion; otherwise a blind camera looks like "clearly low" and could flag a step as
  missing. Publish `Predicates.visibility` every frame, since the blindness ask and every
  suspension rule use it.
- **T-INT:**
  - Use `EngineRunner` (threaded `run()`, or `pump()` in an offline loop), or call the engine
    directly.
  - Call `finish()` at end of stream, before the bus closes, if its outputs must reach the log.
  - The engine output is deterministic for a given input order, which is what
    `run_offline.py`'s identical `events.jsonl` requires.
  - Tune the §2 parameters on training-operator takes only.
- **T-UI / T-VOICE:**
  - Clear the Ask card and stop listening on any later `StepEvent` for the Ask's `step_id`, or
    when a newer `Ask` arrives.
  - A `StepEvent` with status `pending` means "reopened". A `retroactive` event rewrites a row
    of the checklist.
  - Speak alert messages as they are.
- **T-EVAL:**
  - Count a skip incident by its `skip` alert. It is preceded by one `out_of_order` alert naming
    the same steps, which should not be scored as a false out-of-order.
  - The ask rate is the number of `Ask` outputs; the ask-then-correct rate compares
    `operator_confirmed` against the ground truth.
  - `StepEvent.t_end` is the right timestamp for "step completion at ±2 s"; `t` includes the
    dwell.
- **T-IO:** `engine.outcomes()` returns step id, text, status, `t_start`, `t_end` and confidence,
  which is the downlink packet's step table.

## 6. Contract proposals (for the integrator; not applied)

- **Ask withdrawal:** there is no message that says an Ask is closed without an answer
  (evidence resolved it, it was superseded, or it was cancelled). Today that is a convention
  (§3). An explicit `AskClosed(question_id, reason, frame_idx, t)` on the `asks` topic, or an
  `Ask.closes: str | None`, would let the UI and the voice queue drop stale questions without
  inferring it. Nothing blocks on this.

## 7. Open items and known limitations

- A "no" answer given *after* dependent steps have completed does not retroactively flag the
  step or re-alert. The operator is re-prompted, and that is all.
- An operator-confirmed step is never upgraded to `verified`, and never downgraded, by later
  evidence. It is only watched for undo once its evidence has been seen high.
- Steps that are still unsure at `finish()` stay `pending`, with no status for "unresolved at
  end". If T-EVAL or the downlink wants one, it's a contract change (a new `StepStatus` value).
- The engine uses only `ActionScores.top()`, for `t_start`. It never uses action scores to
  complete, rank or ask. If T-HOI-TEMP wants the temporal model to count as evidence, it goes
  into `StepEvidence.sources` upstream.
- Environment: in this workspace `uv` is at
  `C:\Users\sairi\AppData\Roaming\Python\Python313\Scripts\uv.exe`, which is on PATH in Git Bash
  here. That path differs from the one recorded in `Wave0.md`.
- A T-SPATIAL session was working in the same checkout at the same time
  (`actus/spatial/*`, `tests/test_spatial_*.py`). At handover, some of its tests and lint checks
  were failing. They are not T-ENGINE's and were not touched.
