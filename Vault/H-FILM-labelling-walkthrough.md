# Labelling walkthrough

For whoever is labelling recorded takes — you don't need to know the codebase, just this page.
Run everything from the repo root (`Actus-v2.0/`). See `Vault/H-FILM-recording-and-labelling.md`
for the engineering log behind these tools if you want the "why".

## 1. What you're doing

Each take is a video of one operator running the box-transfer experiment
(`protocols/bas_boxes.yaml`). Labelling means marking **when each step starts**, frame by frame,
by watching the video and pressing a key the instant the operator's hand starts moving toward the
next object. The steps are:

| Key | Step | What starts it |
|-----|------|-----------------|
| `1` | S1 | Hand moves to open the outer box |
| `2` | S2 | Hand moves to take out the red box |
| `3` | S3 | Hand moves to place the red box in Zone A |
| `4` | S4 | Hand moves to take out the yellow box |
| `5` | S5 | Hand moves to place the yellow box in Zone B |
| `6` | S6 | Hand moves to close the outer box |
| `0` | idle | Hand withdraws / nothing is happening (press once a step's end state is reached) |
| `x` | other | Anything that isn't a step above (adjusting something, a distraction, etc.) |

**Rule of thumb:** press the step key the moment the hand *starts* moving toward the object, and
press `0` once the step is *finished* and the hand has withdrawn. Don't wait for the object to
fully settle before pressing — err on catching the start of the motion.

## 2. Before you start

```sh
uv sync                                    # once, if you haven't already
uv run python tools/record_takes.py --label-pending
```

That's it — `--label-pending` defaults to `data/raw` and finds every take nobody has fully
labelled yet. The very first thing it does is print any naming/file problems it finds (missing
`.csv`/`.json`/`.frames.csv`, or a file that doesn't follow `{operator}_{scenario}_{n}.mp4`) —
if you see a `WARNING` line, flag it to whoever recorded that take before labelling around it.

You'll then see something like:

```
20 take(s) to label in data/raw  (q = save + next, Esc = save + stop)
[1/20] op02_correct_01.mp4
```

## 3. Labelling one take

**Standby.** A window opens showing the filename and **"press r to begin"**. Nothing plays yet —
this is your moment to get ready, position the window, whatever. Press **`r`** when you're ready.
(`q` or `Esc` here skips the take untouched — it stays pending for next time.)

**Playing.** The video plays at normal speed. Press the step keys (`1`–`6`, `0`, `x`) as described
above, as the moments happen. A few things on screen help:

- The **timeline bar** at the bottom fills in as you label — a quick visual check that you're not
  missing anything.
- The **big label in the corner** shows what's currently active at the playhead.
- `space` pauses/resumes, `.`/`,` step one frame at a time while paused, `[`/`]` jump 5 seconds,
  `s` cycles playback speed (1× → 0.5× → 0.25×) if the action is too fast to catch.
- `u` undoes your last key press *while the video is still playing*.

**End of the file.** The video pauses on the last frame and the screen changes to
`END  u restart   q/Esc save+quit`. Now you have a choice:

- **Happy with it?** Press `q` — it saves and moves straight to the next take.
- **Not happy?** Press `u` — this **discards every label you just made on this take** and plays
  it again from frame 0, back in standby ("press r to begin"). Use this if you missed a step, got
  the timing badly wrong, or just want a clean second attempt. There's no partial-undo at this
  point — `u` at the end always means "start this take over," not "undo one label."

## 4. Finishing a batch session

- `q` at the end of a take → saves it, opens the next pending take automatically.
- `Esc` at the end of a take → saves it, and **stops the whole batch** (the rest stay pending).
- You can stop any time (close the terminal, `Esc`) and resume later — just run
  `uv run python tools/record_takes.py --label-pending` again; it picks up exactly where you left
  off, skipping anything already labelled.

## 5. Key reference

| Key | Standby | Playing | End of file |
|-----|---------|---------|-------------|
| `r` | begin playback | — | — |
| `1`–`6` | — | mark step S1–S6 starting | — |
| `0` | — | mark idle | — |
| `x` | — | mark a non-step action | — |
| `u` | — | undo last label | **discard labels, restart this take** |
| `space` | — | pause / resume | — |
| `.` / `,` | — | step forward / back a frame (while paused) | — |
| `[` / `]` | — | jump 5 s back / forward | — |
| `s` | — | cycle speed 1× / 0.5× / 0.25× | — |
| `q` | skip take (stays pending) | save, next take | save, next take |
| `Esc` | skip take (stays pending) | save, stop batch | save, stop batch |

## 6. After labelling

Once a take's `.csv` has real step rows (not just `idle`), it's done — you'll never be asked to
label it again unless someone explicitly deletes its `.json`'s `labelled_at` field. The next
pipeline stage (`tools/extract_frames.py` → `autolabel.py` → `review_labels.py` →
`make_splits.py`) reads labelled takes to build the training set; that's handled separately, not
part of labelling.

## 7. Common mistakes to avoid

- **Don't** run `--label-pending` with `--script` or `--no-preview` unless you know what you're
  doing — those are for automated tests, not real labelling, and skip the standby screen and the
  end-of-file safety pause entirely.
- **Don't** panic if you mis-click mid-video — `u` undoes one press; you don't need to restart
  unless you want to.
- **Do** use the restart-at-end (`u` after `END`) liberally if your first pass felt rushed. It's
  cheap — the video is still right there, it just replays.
