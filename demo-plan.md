# ACTUS: the 2-day demo plan

2026-09-27 · execution plan · architecture rationale in `Stage3-Architecture.md`

## Reality check, first

**Target:** recorded walkthrough. Both halves shown working: the Earth pipeline end-to-end, and
a real astronaut clip end-to-end, with MicroG-4M visibly contributing action scores as one
evidence stream.

**This is achievable in 2 days for exactly one reason:** a recorded walkthrough means everything
runs **offline and in batch**. No real-time budget, no frame-drop handling, no live camera risk,
no throughput spike. A 3-minute clip at a 1-second hop is ~180 windows; even at a full second
per window that is 3 minutes of compute. **Compute is a non-issue.** That single fact is what
makes the rest of this plan fit.

**It is achievable only with three hard cuts.** These are deadline decisions, not reversals of
the architecture doc:

| Cut | Why | What replaces it |
|---|---|---|
| **No new filming** | A shoot plus labelling is 2+ days on its own | `takes/take1*` (bottle, already filmed, 3 step CSVs incl. a skip take and an unsure take) + the chosen astronaut clip |
| **No ST-GCN++ / MMAction2** | Two reasons. There is **no training data** in 2 days, so a learned action classifier has nothing to learn from. And `mmcv` installation against the macOS pin chain (mediapipe 0.10.21 → numpy<2 → opencv 4.x) is a half-day risk on its own. | **Skeleton *features*, not a skeleton *model*.** Normalised joint angles keep the entire Earth→orbit bridging argument from `Stage3-Architecture.md` §1–2, because that comes from the representation, not the GCN. ST-GCN++ is the first post-demo upgrade. |
| **No trained step classifier** | Same reason: no labelled data beyond 4 clips | **Rules over pretrained action evidence.** Hand-written thresholds on normalised joint angles, object relations, and the action model's scores. Hours, not weeks, fully debuggable, and honest if described as such. |

**What stays:** the two-channel belief split, the decision layer, the abstain/ask path, the
orientation-agnostic pose front end, MicroG-4M as a frozen evidence stream, and `sequencer.py`
untouched.

**State this in the writeup, do not bury it:** the step logic is rule-based over pretrained
action evidence, not a trained step classifier. The trained AI models in the loop are YOLO, the
pose estimator and MicroG-4M. The step-classifier fine-tune is the stated next step.

## The one thing that can sink this: MicroG-4M loadability

Unverified, and it is now a hard requirement rather than a T2 nicety. **Time-box it to 3 hours**
on the morning of day 1, and walk this ladder:

1. **Do not install PySlowFast.** It is CUDA-oriented and effectively unmaintained against torch
   2.14. Instead build the architecture from `pytorchvideo` (pip-installable, has SlowFast and
   I3D R50) and load the published checkpoint's `state_dict` into it, renaming keys as needed.
   This is the most likely path to work and it avoids the whole framework.
2. If key names will not reconcile, check Hugging Face for any `transformers`-compatible
   checkpoint trained on this dataset — a one-line load if it exists.
3. **Fallback, if both fail:** substitute an off-the-shelf video action model that loads in
   minutes (a Kinetics-pretrained VideoMAE or TimeSformer via `transformers`), wire it into the
   identical stream slot, and **label it honestly** in the GUI and the writeup as a substitute
   pending MicroG-4M integration. The architecture is unchanged; only the weights differ. Avoid
   VideoMAE-*small* specifically if anything is published — CC-BY-NC-4.0.

The stream slot is defined by an interface, not by a model, so this swap costs nothing
downstream. **Decide at hour 3 and move on** — do not let this consume day 1.

## Clip selection, and the gate that must pass first

If the step file is supplied with the clip, use it. If not, authoring one takes ten minutes with
a tool that already exists: `python tools/run_fsm.py --video clip.mp4`, tap `0`–`5` as each step
begins, and it writes `clip_steps.csv` on exit.

**Gate, before committing to any clip:** run the pose estimator on it and check a skeleton is
found on a usable fraction of frames. If pose fails on the astronaut, the whole spine fails and
no amount of downstream work rescues it. This check is 15 minutes and it must come before
everything else. Try another clip rather than trying to fix it.

Selection criteria, in priority order for a 2-day build: astronaut's body mostly visible;
camera roughly fixed; 5–6 visually distinguishable steps; 1–3 minutes long; NASA/ESA public
domain. Generic objects (bags, cables, tools, containers) so stock COCO classes can see them —
there is no time to fine-tune a detector.

## Schedule

Roughly 20 working hours. Ordered so that stopping early still leaves a working demo.

### Day 1

| Hours | Work | Gate |
|---|---|---|
| 1–2 | Choose astronaut clip. Author or ingest its step CSV. **Run the pose gate.** Write `configs/step_graph_iss.yaml`. | Pose found on the astronaut, or pick another clip |
| 3–5 | **MicroG-4M load spike**, time-boxed per the ladder above. Get per-window logits *and* the pooled embedding out of one test clip. | Decide at hour 5: real checkpoint or substitute. Do not overrun. |
| 6–8 | `tools/extract_evidence.py` — the offline batch pass. Per frame: YOLO (person + COCO objects), pose. Per 3 s window at 1 s hop: action model. Writes one **frame-index-keyed** cache per video. | A cache file exists for both clips |
| 9–10 | `src/perception/skeleton.py` — 4-rotation pose pick by mean landmark confidence, hip-centring, scale normalisation, torso canonicalisation, joint angles. Keep the pre-normalisation torso vector as its own feature. | Rotated clip yields the same normalised skeleton as upright |

### Day 2

| Hours | Work | If behind, cut |
|---|---|---|
| 11–13 | `src/fsm/belief.py` (rules → `p(step)`, two channels) and `src/fsm/decide.py` (gate, dwell, hysteresis, abstain). Add `CachedBeliefSource` to `sources.py` implementing `.due(t)`. | Cut hysteresis and the prior channel before cutting abstain |
| 14 | Bound the voice queue (drop stale utterances). Add confidence to `EventLog`'s `detail` dict. | Neither — the voice queue will misbehave without this |
| 15–16 | GUI: confidence as a 4th `StepList` column; action-score panel using the `VideoWorker.stats` second-channel pattern. | Cut the action-score panel; keep the confidence column |
| 17–18 | `src/io/recorder.py` (local write) + `src/io/streamer.py` (IP). Listed deliverable, currently 0% built, no research risk. | Cut the streamer, keep local write, and say so |
| 19 | End-to-end runs on all four clips. Fix. | — |
| 20 | Record the walkthrough. Write the limitations section. | — |

**Do not start the GUI work before the belief and decision layers run headless.** `run_fsm.py`
gives a console view of every event, which is enough to debug with, and the GUI is the most
tempting and least load-bearing piece.

## Files to create

```
tools/extract_evidence.py          offline batch pass -> per-video cache
src/perception/skeleton.py         4-rotation pose, normalise, joint angles
src/perception/action_model.py     MicroG-4M wrapper + substitute behind one interface
src/fsm/belief.py                  evidence -> p(step), C1 prior-free and C2 prior-informed
src/fsm/decide.py                  gate / dwell / hysteresis / abstain -> observe() or ask()
src/fsm/sources.py                 ADD CachedBeliefSource, implementing .due(t)
src/io/recorder.py                 local mp4 write
src/io/streamer.py                 push to an IP
configs/step_graph_iss.yaml        the astronaut procedure's steps
configs/pipeline_bottle.yaml       which streams, which weights, which thresholds
configs/pipeline_iss.yaml          same, for the orbit path
```

Existing code that must **not** change: `src/fsm/sequencer.py`, `src/fsm/session.py`. `Session`
accepts any object with `.due(t)` and no isinstance check, so `CachedBeliefSource` drops in.
Two small exceptions: thread `index` through `Session.advance()` if the cache is index-keyed,
and add the confidence field to `EventLog`.

## Traps already identified — avoid these by construction

- **Key the cache by frame index, not time.** `t = index / fps` with `fps = cap.get(...) or
  30.0` silently invents 30 fps on a file with unreadable metadata, and drifts on
  variable-frame-rate sources. An ISS clip from YouTube is exactly the kind of file that bites.
- **Do not route video files through `VideoSource`.** On a file, EOF is indistinguishable from a
  disconnected camera, so it reopens and loops forever. Use the existing synchronous file
  readers and put a `deque(maxlen=90)` in that loop.
- **`ScriptSource` ignores the `frame` column** and reads only `time_s`. Be consistent when
  adding the new source.
- **Bound the voice queue before wiring the model.** It is unbounded with no dedup; a noisy
  belief stream produces bursts and speech falls progressively further behind. This is the
  difference between sounding intelligent and sounding broken.
- **Stock COCO YOLO, not a fine-tune.** No time, and stock weights generalise across appearance
  far better than a few-hundred-image fine-tune would.

## The walkthrough, shot by shot

Write this down before building, because it defines what actually has to work. Target 3–4
minutes.

1. **Architecture, 15 s.** One diagram: evidence streams → belief → decision → FSM → alerts.
   Say plainly that the FSM is the judge and the models are sensors.
2. **Bottle clip, correct run.** GUI with overlays, step tracking, next-step prompts, voice.
   Proves the basic-section requirements.
3. **Skip alert.** `take1_ERROR_cap_pour_steps.csv` — pours without unscrewing, voice fires.
   This is a listed deliverable and it demos in 15 seconds.
4. **Abstain.** The unsure take: *"I can't see clearly. Is the cap off?"* → answer yes → step
   counted. Already implemented and exercised; reads as sophistication.
5. **Astronaut clip.** Same GUI, real orbit footage. Person box, skeleton, **MicroG-4M action
   scores visible in the panel**, step tracking against the step file.
6. **The money shot: rotate the astronaut clip 90° and 180°** and show it still tracks. This is
   the optional orientation-agnostic requirement, demonstrated rather than claimed. Note in the
   narration that the rotation is applied in post, so it is a controlled test rather than new
   footage.
7. **Artifacts.** The JSONL log with timestamps and outcomes, the locally stored video, the
   stream endpoint.
8. **Limitations, 20 s.** Read from the honesty list below. Saying it yourself is stronger than
   being asked.

## Honesty list for the writeup

- Step inference is **rule-based over pretrained action evidence**, not a trained step
  classifier. No labelled procedure data existed within the build window.
- **N = 1 astronaut clip.** This is an existence proof, not a generalisation claim.
- MicroG-4M is used **frozen and zero-shot**. It was not fine-tuned — deliberately, since
  fine-tuning it on Earth footage would overwrite the microgravity prior that is its whole value.
- If the substitute model was used, name it and say why.
- The detector uses **stock COCO classes**, not a procedure-specific fine-tune.
- The rotation test is **synthetic** (rotated in post), not independently captured footage.
- Everything runs **offline in batch**; real-time performance is untested.

## What day 3 would buy, in priority order

1. Film 15–25 Earth takes of the astronaut procedure, mirrored with props → the first real
   training data.
2. Replace the rules with a trained step head on that data. Keep the rules as the baseline to
   beat.
3. Add ST-GCN++ via MMAction2 (fine-tuned from the published NTU checkpoint — the checkpoint,
   not the 1.3–2.3 TB dataset).
4. The three-way ablation: geometry only / + skeleton model / + MicroG-4M. This is what turns
   the transfer hypothesis from an assertion into a measurement.
5. A second astronaut clip, so N > 1.