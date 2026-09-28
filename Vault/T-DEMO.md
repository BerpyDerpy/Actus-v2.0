# T-DEMO: decisions made outside the build plan

2026-09-28 · covers T-DEMO (walkthrough script, shot list and limitations slide text, drafted from
`report.md`) · read with `build-plan.md` §5 (Wave 3) and `demo-plan.md`'s shot-by-shot list, and
`Vault/T-EVAL.md`, `Vault/T-INT.md`, `Vault/T-FAULT.md`, `Vault/T-PROTO2.md`, `Vault/T-PKG.md`

The T-DEMO card is one line: "Walkthrough script, shot list and limitations slide text, drafted
from `report.md`". This file records every decision made turning that into a harness, so that
H-REC (who films it) and whoever regenerates it later can rely on them or overturn them on
purpose.

**Scope, as agreed:** the harness. Nothing is committed. **No code was run** while writing this
lane (that was the instruction), so nothing below is verified; §0 lists what to run first and
which assertions are most likely to fail.

---

## 0. Status against the card, and what to run first

| Deliverable | Status |
|---|---|
| Walkthrough script | Written as a generator, not a fixed document: `demo/shots.yaml` (the 12 shots, in order) + `tools/make_demo_pack.py` → `demo/out/walkthrough.md`. **Unrun.** |
| Shot list | Same tool → `demo/out/shot-list.md`: a preflight table of what the machine has, then per shot who/what/command/what-to-see/when-to-retake. **Unrun.** |
| Limitations slide text | `demo/limitations.yaml` (the honesty-list items, some conditional) → `demo/out/limitations.md`. **Unrun.** |
| "Drafted from `report.md`" | Reinterpreted as "drafted from `report.json`" (§1). No `report.json` exists on this machine, so every number in the generated documents today is a `[NOT MEASURED ...]` placeholder, not a draft number. This was checked by reading the code path, not by running it. |

Run these, in this order:

```sh
uv run ruff check . && uv run ruff format .          # formatted by hand; expect small diffs
uv run pytest tests/test_demo_pack.py
uv run python tools/make_demo_pack.py --check        # validates demo/shots.yaml + limitations.yaml
uv run python tools/make_demo_pack.py                # writes demo/out/*.md, demo/out/manifest.json
uv run python tools/make_demo_pack.py --strict; echo $?   # expect 1: D9 is always simulated
```

Most likely to fail, because each rests on an assumption about another lane that I could not
check by running it:

1. **The gate checks in `check_gates`** (rack measured, detector trained, media binaries, GUI
   built, ASR/TTS models, `engine:` tuned, temporal model) each read one or two files under
   `configs/default.yaml`'s own paths. If a lane changes where its asset lives without changing
   the config (T-DET-TRAIN's `models/detector/best.onnx`, T-PKG's `bin/mediamtx`), the gate will
   under- or over-report readiness. `test_each_gate_follows_exactly_its_own_files` pins today's
   paths; if it fails after another lane lands, the gate needs updating, not the test.
2. **`detector_custom` requires `detector.backend: custom`, not just the files.** This is a
   judgement call (§4) that nobody has exercised against a real `models/detector/card.md`.
3. **The word-per-second timing budget** (2.5 words/s) on every narration is arithmetic I did by
   eye when writing `demo/shots.yaml`; the test suite checks it holds, but nobody has timed a
   human actually reading D5 or D7 aloud.
4. Style: `ruff format` was not run on either new file.

## 1. Files and entry points

| File | What it holds |
|---|---|
| `demo/shots.yaml` | The 12 shots (D1-D11, with D8 split into D8a/D8b), in running order: title, timing, who does what, both narrations (real and simulated), both commands, what must be seen and heard, when to retake, and the gates that decide how each can be filmed. |
| `demo/limitations.yaml` | The honesty-list items for the limitations slide. Some are unconditional (the build plan's §9 list); some appear only while a named gap is open, or only once something exists to caveat. |
| `tools/make_demo_pack.py` | The generator: gate-checking (`check_gates`), shot/limitation parsing and validation, report-token resolution (`Tokens`), rendering, and the CLI. |
| `tests/test_demo_pack.py` | See §6. |
| `demo/out/` | Generated, not committed (§6 proposes a `.gitignore` line). `walkthrough.md`, `shot-list.md`, `limitations.md`, `manifest.json`. |

```sh
uv run python tools/make_demo_pack.py                       # demo/out/ from the real report
uv run python tools/make_demo_pack.py --check                # validate only, write nothing
uv run python tools/make_demo_pack.py --strict                # exit 1 unless every shot is READY
uv run python tools/make_demo_pack.py --report path/to/other.json --out somewhere/
```

## 2. Why a generator, not a document

`report.md` (T-EVAL) does not exist until a real evaluation has run, and the machine filming the
walkthrough will not always have every asset (a trained detector, a measured rack, the media
binaries, the built GUI). A script committed as prose would either lie about what is ready or go
stale the moment any of that changes. So the deliverable is the **generator plus its two input
files**, and the actual script is regenerated right before recording, from whatever is really on
the machine at that moment. This mirrors T-EVAL's own report (a generated artefact with a fixed
shape) and T-PKG's doctor (WARN/FAIL from what is on disk, never from memory).

**Consequence:** "the walkthrough script" is not one file to hand to H-REC. It is
`demo/out/walkthrough.md`, regenerated, plus this Vault note explaining why some shots in it may
say `BLOCKED`.

## 3. Decisions the card leaves open

**Reading `report.md` as `report.json`.** The card says "drafted from `report.md`". The generator
reads `runs/eval/report.json` instead, because `report.md` is prose rendered *from* the JSON
(`tools/evaluate.py::render_markdown`, `Vault/T-EVAL.md` §1) and the generator needs the
structured `criteria` rows and metric values, not Markdown to re-parse. Nothing downstream reads
`report.md`.

**A report only counts if it is real.** `read_report` refuses to treat a report as usable unless
it exists, parses as a JSON object, is **not** `simulated`, and evaluated at least one take
(`takes_evaluated >= 1`). This directly implements the build plan's "no placeholder metrics"
principle and T-EVAL's own rule that a SIMULATED report is not a result
(`Vault/T-EVAL.md` §0, §7). An unusable report makes every metric token in the walkthrough and
shot list read as `[NOT MEASURED - <reason>]` or `[SIMULATED - DO NOT QUOTE]`, visibly, rather
than silently substituting a stale or fake number. `tests/test_demo_pack.py` asserts this end to
end (`test_a_simulated_report_never_reaches_the_documents`,
`test_on_a_bare_machine_the_pack_claims_nothing`).

**Numbers are never typed by hand.** Every quoted metric in `demo/shots.yaml` and
`demo/limitations.yaml` is a `{{m:<metric>}}` / `{{t:<metric>}}` / `{{v:<metric>}}` /
`{{criteria_met}}` / ... token (§5), resolved only from the report at generation time. A reviewer
can grep `demo/*.yaml` for a bare digit next to a metric name and expect not to find one.

**Gates decide how a shot can be filmed, and the generator computes them from disk, not from
memory.** `check_gates` looks at `configs/default.yaml` (and the files it points to) fresh every
run: is the rack measured, is a trained detector present *and* selected, are the media binaries
found, is the GUI built, are the Vosk/Kokoro models present, is `engine:` non-empty (T-INT/T-ENGINE
tuning has happened), does a temporal model exist. Ten gates in total (`GATES` in the tool). Three
states per shot, derived from a shot's declared gates:

| State | Meaning | How a shot gets it |
|---|---|---|
| `READY` | Film it for real. | Every `needs_any` and `needs_real` gate is met. |
| `SIMULATED` | Film it from the replay; its narration says "simulated" out loud. | A `needs_real` gate is unmet but the shot is marked `replay_ok` (or `always_simulated`), and every `needs_any` gate is met. |
| `BLOCKED` | Cannot be filmed honestly yet. | A `needs_any` gate is unmet, or a `needs_real` gate is unmet and the shot cannot be simulated. |

`needs_any` gates block even the simulated path (e.g. the GUI must be built for almost every
shot, since they're all shown on the dashboard); `needs_real` gates are what real footage would
need but the replay can stand in for; `wants` is a soft gate that only adds a "Degrades without"
note (e.g. D2 wants the Kokoro voice; without it the voice falls back to pyttsx3 per
`Vault/T-INT.md` §4, and the shot still plays). A `BLOCKED` shot still gets a drafted narration in
the walkthrough, clearly marked, so the document is reviewable before every gate is met — but the
tool's `--strict` flag and the shot-list's own text say plainly not to record it.

**Ten gates, and why each is checked the way it is (`check_gates`):**
- `report_real`: as in §3 above.
- `rack_measured`: `configs/rack.yaml`'s `measured: true`, per `Vault/T-HOI-TEMP.md` §0.1 (the
  filmed rig never matched the placeholder, and differs per operator).
- `detector_custom`: **not just file presence.** It requires `models/detector/best.onnx` and its
  `card.md` to exist, *and* `configs/default.yaml`'s `detector.backend` to already say `custom`.
  T-DET-TRAIN's card ("done when the held-out mAP50 is reported, whatever its value... must be
  ≥0.85 to replace the open-vocab backend by default", build plan §5) is a numeric judgement call
  this tool cannot make from a card it isn't allowed to interpret; requiring the config to already
  say `custom` means a human made that call, not the generator.
- `media_binaries`: `stream.mediamtx_bin` and `stream.ffmpeg_bin` resolved the same way
  T-IO's own binaries are (a repo-relative path, a `.exe` sibling on Windows, or `PATH`) — see
  `actus/io/process.py::resolve_binary`, which this tool's `_find_binary` mirrors rather than
  imports, to keep T-DEMO from depending on T-IO's module surface for something this small.
- `ui_built`: `ui.static_dir`'s `index.html` exists (T-UI's `npm run build` output).
- `asr_model` / `tts_model`: the Vosk directory and the Kokoro model+voices files exist.
- `engine_tuned`: `configs/default.yaml`'s `engine:` section is non-empty. This is a proxy, not a
  guarantee — an empty `engine: {}` certainly means untuned (`Vault/T-INT.md` §0 says tuning is
  "not done, and blocked"), but a non-empty one only means *someone* set an override, not that it
  was tuned honestly on training-operator takes only. The limitations item this gates
  (`thresholds_untuned`) is worded to match what the gate can actually know.
- `temporal_model`: `models/temporal/temporal.npz` exists. This gates nothing about *shots* (no
  shot needs it); it only gates whether the `temporal_small_data` limitation item is worth
  showing (§4).

**A shot's narration is validated, not just filled in.** `_validated` (called while parsing
`demo/shots.yaml`) enforces, at load time, before any gate is even checked:
- a shot has `narration` unless it is `always_simulated`;
- a shot that can ever be simulated (`replay_ok` or `always_simulated`) has both
  `narration_sim` and `cmd_sim`;
- **`narration_sim` must contain the word "simulated"**, case-insensitively. This is the harness's
  answer to the build plan's "Anything simulated is labelled SIMULATED in the UI and the logs"
  principle, extended to the spoken word: a viewer who only hears the audio, not the on-screen
  watermark, must still be told.
- a shot that can never be simulated does not carry a `narration_sim` (so nobody can quietly make
  a shot "simulatable" by adding text without also marking `replay_ok`).

These are `PackError`s, not warnings: a shot that fails validation stops the whole build, because
a broken shot is worse than no walkthrough.

**The time budget is enforced, not advisory.** Every narration is checked against
`MAX_WORDS_PER_SECOND = 2.5` (150 words/minute) for its shot's allotted seconds, and the total of
all shots' `seconds` must land in `TOTAL_SECONDS_RANGE = (180, 240)` (the plan's "3-4 min",
`demo-plan.md`'s walkthrough section, and build plan §5's Wave 3 table). A `{{token}}` counts as
one word for this purpose (`timing_text`), since its rendered length is not known until generation
time and a placeholder like `[NOT MEASURED - no report]` must not be allowed to blow the budget by
accident. These are warnings, not `PackError`s (`timing_warnings`), because a slightly tight shot
is something a human should look at, not something that should stop the pack from generating.

**Limitations follow the gaps, both ways.** `demo/limitations.yaml` items can carry
`show_if_unmet: [gate, ...]` (shown only while at least one named gate is unmet — the item
*describes* the gap) or `show_if_met: [gate, ...]` (shown only once every named gate *is* met —
the item is only meaningful once something exists, e.g. "the held-out set is N takes" only means
something once `report_real` holds). An item with neither is the build plan's unconditional
honesty list (§9) plus this lane's own standing caveats (protocol 2 is always simulated, faults
are software-emulated, etc.) — these never disappear, because they are not gaps to close, they are
what the system honestly is.

**The slide holds six items; nothing is dropped.** `SLIDE_MAX_ITEMS = 6` matches the plan's "20 s"
budget for the limitations shot (D11) at a readable pace. Every *shown* item beyond the first six
goes on a "backup slide" in the same document, in the same order, so a presenter fielding a
question in Q&A has the rest ready without anything having been silently discarded. The gap items
(no numbers yet, detector untrained, rack unmeasured, thresholds untuned) are ordered first in
`demo/limitations.yaml`, so they are the ones most likely to occupy the visible six while the
build is incomplete — deliberately: they are also the ones a judge is most likely to ask about.

## 4. Decisions specific to individual shots

- **D2 (the correct run) is also H-REC's fallback recording.** Its `notes` field says to keep the
  first successful take regardless of quality, per the build plan's "record the walkthrough, plus
  a fallback screen recording as soon as anything works end to end" (H-REC, §5 Wave 3).
- **D6 (the 180° roll) has no simulated form**, because the replay has no camera to roll — it
  renders a fixed orthographic map (`Vault/Wave0.md` §8). Its `notes` field gives the honest
  fallback (`--override capture.rotation=180` on a file or a live source) and says the narration
  must then admit the rotation happened in post, matching `demo-plan.md`'s own honesty note ("the
  rotation is applied in post, so it is a controlled test rather than new footage").
- **D5 (the ask path) uses `--evidence replay` for its simulated form**, not the default replay
  evidence mode, because `Vault/T-INT.md` §2 explains the plain replay publishes no hands, and
  without hands the occlusion scenario has nothing for spatial to lose track of.
- **D7 (fault injection) also points at `python -m actus.app.faults --demo all`**, T-FAULT's own
  ground-truth-checked report (`Vault/T-FAULT.md` §5), rather than asking the presenter to judge
  live behaviour by eye.
- **D8 is split into D8a (log + downlink) and D8b (RTSP + recordings)**, because the plan's shot 8
  ("the artefacts") bundles three different things to demonstrate, each with its own command and
  its own failure mode. D8b's `notes` flags that `stream.target_ip` must be a real LAN address for
  a second machine to see the stream (127.0.0.1 only loops back to the same host), which is my
  reading of `actus/io/streamer.py`'s docstring, not something T-IO or T-FAULT confirmed by
  running it (`Vault/T-INT.md` §0 also lists the real RTSP push as unverified).
- **D9 (the second protocol) is `always_simulated`.** `Vault/T-PROTO2.md` §4 is explicit that its
  demo tool draws its own scene and "nothing this tool prints is a metric for the report card";
  there is no real-footage path for it until Task B takes exist and are scored. This is also why
  `--strict` can never return 0 on this build: D9 can never reach `READY` (verified by
  `test_strict_fails_until_every_shot_is_ready`, which checks this holds even on a fully-equipped
  fixture root).
- **D10 (the Validation tab) can never be simulated.** It has no `replay_ok` and no
  `narration_sim`: showing the Validation tab with replay numbers would contradict T-EVAL's own
  rule that a SIMULATED report is not a result, so if `report_real` is unmet the shot is simply
  `BLOCKED`, with no fallback.

## 5. Tokens (`Tokens` in `tools/make_demo_pack.py`)

`{{name}}` in any narration or limitation slide resolves against the loaded report:

- `{{m:<metric>}}`, `{{t:<metric>}}`, `{{v:<metric>}}` — a headline metric's value, its
  pre-registered target, or its PASS/FAIL/n/a verdict, exactly as `tools/evaluate.py::judge`
  reports them. `<metric>` must be one of the ten in `tools/evaluate.py::METRICS`
  (`test_metric_names_match_the_evaluation_harness` pins the two lists together, so a metric
  T-EVAL renames breaks this test rather than silently going stale).
- `{{data_kind}}`, `{{n_takes}}`, `{{n_operators}}`, `{{operators}}`, `{{criteria_met}}`,
  `{{criteria_version}}`, `{{eval_id}}` — the report's own top-level fields.
- **An unknown token is a build error, not a silent blank.** `build_pack` resolves every token in
  every shot's narrations before rendering anything, so a typo in `demo/shots.yaml` is caught at
  `--check` time, not discovered by a human reading the output.
- **Every token, resolved or not, is checked into the time budget as exactly one word**
  (`timing_text`), because a real value's word count isn't known until the report exists.

## 6. Cross-lane notes, and what is not done

- **No other lane's file was edited.** This lane reads `configs/default.yaml` and
  `runs/eval/report.json`; it writes nothing outside `demo/`.
- **Proposal, not applied: add `/demo/out/` to `.gitignore`.** P0 owns `.gitignore`
  (`Vault/Wave0.md` §3); `demo/out/` is generated and machine-state-dependent, so it shouldn't be
  committed the way `runs/eval/` deliberately is. I did not add it myself, matching the standing
  rule that a lane doesn't edit files it doesn't own without saying so here.
- **The generator does not import `tools/evaluate.py` or `actus/io/process.py`.** It re-implements
  the small pieces it needs (reading the report JSON, resolving a binary) rather than depending on
  their internals, so a refactor in T-EVAL or T-IO can't silently break demo generation; the price
  is that `test_metric_names_match_the_evaluation_harness` is the only thing keeping the metric
  list in sync, and it will fail loudly (not silently) if T-EVAL renames a metric.
- **Not done:** actually filming anything (H-REC's job); tuning the demo machine's `stream.target_ip`
  for a real second-laptop VLC test (D8b); checking that `ruff format` leaves the two YAML-adjacent
  Python files unchanged; confirming the 2.5 words/s budget against a real read-aloud timing.
- **Known limitation of the harness itself:** the `engine_tuned` gate (§3) can be fooled by an
  `engine:` override that exists but wasn't tuned honestly; there is no way to check "tuned on
  training-operator takes only" from the filesystem alone. The `thresholds_untuned` limitation
  item is worded around this rather than claiming more than the gate can know.
