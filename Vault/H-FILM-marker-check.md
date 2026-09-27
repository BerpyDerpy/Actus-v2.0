# H-FILM: ArUco marker check

2026-09-27 · `tools/check_markers.py` (new) and `--markers` in `tools/record_takes.py` (T-REC) ·
read with `Vault/H-FILM-setup-commands.md`

The rack frame is solved from four ArUco markers on the board. If they're out of shot, covered
or too small in the recordings, the spatial lane can't place anything on that footage. This check
shows a live window before you record, so you can move the markers or the camera until all four
are detected. It's a live video stream, not a snapshot.

Run all commands from the repo root. Replace `192.168.1.23` with the address the IP Webcam app
shows.

---

## 1. One-time setup

Nothing new to install: the check uses OpenCV's ArUco module, which `uv sync` already installs.

```sh
uv sync                                               # if you haven't already (see H-FILM-setup-commands.md, section 1)
uv run python tools/make_markers.py                   # writes data/markers/markers_a4_p1.svg and board_layout.svg
uv run python tools/check_markers.py --help           # check that it works: prints the checker's options
```

Print the markers:

1. Print `data/markers/markers_a4_p1.svg` at **100 % / actual size**, not "fit to page".
   Measure the 100 mm scale bar with a ruler, and each black square should be 60 mm.
2. Cut along the dashed lines, keeping the white border around each marker. The detector needs
   that border.
3. Stick them on the board as `data/markers/board_layout.svg` shows. Each marker's red arrow
   points to the **top** of the board.

| Marker id | Where on the board |
|---|---|
| 0 | top-left |
| 1 | top-right |
| 2 | bottom-right |
| 3 | bottom-left |

Positions and ids come from `configs/rack.yaml`. That file is still placeholder geometry
(`measured: false`), so the checker prints a note about it. This doesn't stop the check from
working (see `fit` in section 3).

## 2. Run the check (before recording)

Windows PowerShell, without `uv` on PATH, from the repo folder:

```powershell
.venv\Scripts\python tools\check_markers.py --source http://192.168.1.23:8080/video
.venv\Scripts\python tools\check_markers.py --source 0

```

With `uv`:

```sh
uv run python tools/check_markers.py --source http://192.168.1.23:8080/video   # phone (IP Webcam app)
uv run python tools/check_markers.py --source 0                                 # laptop webcam
uv run python tools/check_markers.py --source http://192.168.1.23:8080/video --view-width 960   # a bigger window
uv run python tools/check_markers.py --source replay:correct                    # self-test with SIMULATED frames, no camera needed
```

A small window stays on top of other windows and shows the live feed. Keep adjusting until the
banner is green and reads **READY 4/4 markers**. Then press `q` (or close the window) and start
recording:

```powershell
.venv\Scripts\python tools\record_takes.py --operator op01 --scenario correct --toggle --markers --source http://192.168.1.23:8080/video
```

Close the checker before you start the recorder. A laptop webcam can only be opened by one
program at a time, and two programs reading the phone stream share its frame rate.

## 3. Reading the window

**Banner** (top):

| Banner | Meaning |
|---|---|
| green **READY** | All 4 markers detected, each at least 20 px on a side. |
| amber **PARTIAL** | 3 of 4 usable. The rack frame still solves, but a hand covering one more marker loses it. Fix this before recording. |
| red **NOT READY** | Fewer than 3 usable. |

**On the feed**, every detected marker is outlined, with its id next to it:

| Outline | Meaning |
|---|---|
| green | Good. |
| red | Detected but too small (under 20 px). |
| orange | Detected, but not where `rack.yaml` puts it relative to the others. |
| grey | An id that isn't one of the rack markers (e.g. a box marker). It is ignored. |

When the markers solve the rack, the board outline and Zones A/B are drawn too, so you can check
where the boxes will go.

**Panel** (bottom):

- `0: 62px   1: 62px   2: --   3: 58px`: size of each marker in pixels, or `--` when it isn't
  detected.
- `fit 1.2px`: how well the visible markers match the layout in `rack.yaml`. It is not part of
  READY. With placeholder geometry it can be large or `-`, which is expected until the board is
  measured and `rack.yaml` updated.
- Up to 3 hints in yellow (see section 5).
- `checks/s` and `ms`: how often the window updates, and how long one detection takes.

The console prints a line each time the result changes (e.g.
`PARTIAL: 3/4 markers, missing 2 (missing 2: out of view, covered, glare or wrong dictionary)`).
The last line is the final result.

## 4. Live check while recording (`--markers`)

Adding `--markers` to `record_takes.py` runs the same check during the session. Measured on this
Windows laptop, one detection on a 1280×720 frame takes about 4 ms. The check runs on a
background thread about 8 times a second, so it doesn't slow the recording or cause duplicated
frames.

- The preview shows `MARKERS 0 1 2 3 READY` above the key help, in STANDBY and while recording.
  Each id is green when detected and red when missing. `MARKERS ?` means no recent check (the
  camera stalled).
- The strip and outlines are drawn on the preview only. **The saved video never contains them.**
- Each take's `.json` gets a `markers` section, for example:

  ```json
  "markers": {"checks": 96, "ready_frac": 0.94, "partial_frac": 0.06, "not_ready_frac": 0.0,
              "seen_frac": {"0": 1.0, "1": 1.0, "2": 0.94, "3": 1.0}, "hz": 8.0, ...}
  ```

  `ready_frac` is the fraction of checks with all 4 markers usable. A hand passing over a marker
  during a step lowers it a little, which is normal. If it drops below 50 %, the recorder prints
  a `WARNING markers:` line after the take. Consider re-framing and recording that take again.
- Without `--markers` the recorder behaves exactly as before.

So the separate check in section 2 is optional if you always record with `--markers`. The
STANDBY preview then does the same job. The standalone window gives more detail (sizes, fit,
hints, zones), so use it when first setting up the rig.

## 5. Fixing problems

| What you see | Try |
|---|---|
| `missing N`, marker is in shot | Glare: tilt the board or move the light. Blur: tap to focus in IP Webcam. Check the white border wasn't cut off. |
| `missing N`, marker out of shot | Move the camera back or widen the framing. All 4 corners must be in view. |
| `N is 14px (< 20): move camera closer` | Move the camera closer, or raise the resolution to 1280×720 in IP Webcam. |
| orange outline, `not where rack.yaml puts it: swapped or rotated?` | Check that id and its corner in the table in section 1, and that its red arrow points to the top of the board. |
| `markers don't fit the rack.yaml layout` | The board isn't built to `rack.yaml`. It's fine for recording, but tell whoever owns `configs/rack.yaml` (H-PROPS) so it can be measured. |
| No markers at all, even close up | Printed from the wrong dictionary. Reprint with `tools/make_markers.py` (DICT_4X4_50). |
| `no frames from ...; check --source` | Same as the recorder: the phone and PC must be on the same Wi-Fi, and `/video` must open in the PC's browser. |

## 6. Options

```text
--source S        webcam index, http/rtsp URL, video file, or replay:<scenario> (SIMULATED)
--view-width N    window width in pixels (default 640)
--min-side-px N   smallest usable marker side in pixels (default 20)
--min-markers N   markers needed for PARTIAL (default from configs/default.yaml rack.min_markers = 3)
--rack PATH       rack geometry (default configs/rack.yaml)
--duration S      stop after S seconds
--no-preview      no window; prints the result changes only
```

Exit status: 0 when the last check was READY, 1 otherwise. A script can chain it, e.g.
`uv run python tools/check_markers.py --source 0 && uv run python tools/record_takes.py ...`.

## 7. For developers

- `tools/check_markers.py`: `MarkerChecker` (detection with the rack solver's settings, plus the
  READY/PARTIAL/NOT READY rule), `MarkerMonitor` (background thread used by the recorder),
  `render_check` (standalone window) and `draw_marker_strip` (recorder strip). It reuses
  `RackFrameEstimator.solve` from `actus/spatial/rack_frame.py` (T-SPATIAL, unchanged) with no
  hold and no smoothing, so each check judges its own frame only.
- `tools/record_takes.py`: new `--markers` flag; `start_marker_monitor()` and
  `draw_markers_on()`; `record_take()` takes an optional `monitor`.
- Tests: `tests/test_check_markers.py`, 10 tests: replay frames are READY; covering markers
  gives PARTIAL, then NOT READY; small markers don't count; a displaced marker is flagged off
  layout; unknown ids are ignored; rendering; monitor stats; CLI exit status; `--markers` writes
  the JSON summary and its absence doesn't. The full suite and `ruff check` / `ruff format
  --check` pass.
- Not tested here: a real phone stream and printed markers. The window was run on the SIMULATED
  replay only.
