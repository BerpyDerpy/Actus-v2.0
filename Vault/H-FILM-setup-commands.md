# H-FILM: recording setup commands

Everything you need to record and label ACTUS takes, one command per line. Run the commands from
the repo root. They work the same in PowerShell (Windows) and a terminal (macOS/Linux).

The full explanation of the workflow is in `Vault/H-FILM-recording-and-labelling.md`.

---

## 1. One-time setup

```sh
git clone https://github.com/BerpyDerpy/Actus-v2.0.git   # get the code (private repo: you need access)
cd Actus-v2.0                                             # every command below runs from here
pip install uv                                            # install uv, the Python project manager
uv sync                                                   # create .venv with Python 3.11 and the recording dependencies
uv run python tools/record_takes.py --help                # check that it works: prints the recorder's options
```

If `uv` is "not recognized" after `pip install uv`, pip's Scripts folder isn't on PATH. Run it as
`python -m uv sync` instead. After that, replace `uv run python` in every command below with the
venv's Python:

```powershell
.venv\Scripts\python tools\record_takes.py --help         # Windows, when `uv` isn't on PATH
```

```sh
.venv/bin/python tools/record_takes.py --help             # macOS/Linux, when `uv` isn't on PATH
```

**Not the recording tool:** `tools\run_ava_interaction.py` in the older `SIH-2\Actus\har-bas`
project is a viewer that shows AVA action predictions on a video that has already been recorded.
It doesn't record or label takes, so it isn't used for H-FILM.

## 2. Phone camera (IP Webcam app, Android)

```text
Install "IP Webcam" from the Play Store                  # turns the phone into a network camera
Video preferences -> Video resolution -> 1280x720         # the recorder can't change a phone's resolution, so set it here
Start server                                              # the app shows the address, e.g. http://192.168.1.23:8080
Open http://<phone-ip>:8080/video in the PC's browser     # check it: you should see the live feed (phone and PC on the same Wi-Fi)
```

Keep the phone on the charger with the screen on. If the Wi-Fi drops, the recording session ends.
The take recorded so far is kept; start the command again.

## 3. Record

Use a new `--operator` per person (`op01`, `op02`, ...; no underscores). Use one `--scenario` per
command; change it by quitting (`q`) and starting again.

Replace `<phone-ip>` with the address the IP Webcam app shows, **without the angle brackets**,
e.g. `--source http://192.168.1.23:8080/video`. PowerShell rejects a literal `<` ("The '<'
operator is reserved for future use").

For example, operator 1 on Windows without `uv` on PATH, from the repo folder:

```powershell
.venv\Scripts\python tools\record_takes.py --operator op01 --scenario correct --toggle --source http://192.168.1.23:8080/video
```

```sh
uv run python tools/record_takes.py --operator op01 --scenario correct --toggle --source http://<phone-ip>:8080/video       # record correct runs from the phone
uv run python tools/record_takes.py --operator op01 --scenario skip_S2 --toggle --source http://<phone-ip>:8080/video       # record takes where step S2 is skipped
uv run python tools/record_takes.py --operator op01 --scenario S4_before_S2 --toggle --source http://<phone-ip>:8080/video  # record takes with the yellow box taken out before the red one
uv run python tools/record_takes.py --operator op01 --scenario correct --toggle --source 0                                  # use the laptop webcam instead of the phone
uv run python tools/record_takes.py --operator op01 --scenario correct --toggle --source 0 --source2 http://<phone-ip>:8080/video  # record the laptop webcam and the phone at the same time
uv run python tools/record_takes.py --operator op01 --scenario correct --toggle --source http://<phone-ip>:8080/video --fps 15     # use a lower fps if you see a "duplicated frames" warning
uv run python tools/record_takes.py --operator op02 --scenario correct --toggle --source http://<phone-ip>:8080/video --roll 90 --lighting dim --notes "lid sticky"  # record camera angle, lighting and notes with the take
```

Keys in the recording window (click the window first):

| Key | Action |
|---|---|
| `r` | Start a take; press again to stop and save it |
| `q` / `Esc` | Save any running take and quit |

The window opens in **STANDBY**, and nothing is saved until you press `r`. Each `r` on/off makes
one take: `op01_correct_01`, `op01_correct_02`, ...

## 4. Label (after recording)

```sh
uv run python tools/record_takes.py --label-pending                                    # open every unlabelled take in data/raw, one after another
uv run python tools/record_takes.py --label-pending --view-width 960                   # the same, in a bigger window
uv run python tools/record_takes.py --label data/raw/op01_correct_01.mp4               # label (or re-check) one specific take, keeping its existing labels
uv run python tools/record_takes.py --label data/raw/op01_correct_01.mp4 --force       # re-label a take from scratch: its existing labels are discarded
```

`--force` can't be undone. The old labels are overwritten on the first key press, or when you
quit with `q` even if you pressed nothing. To keep a copy, duplicate the take's `.csv` first.

Keys in the labeller:

| Key | Action |
|---|---|
| `1`–`6` | The operator starts step S1–S6 at this frame |
| `0` | Idle: the step is done and the hand has pulled back |
| `x` | A non-step action (something outside the procedure) |
| `u` | Undo the last press |
| `space` | Pause or resume |
| `.` / `,` | One frame forward or back (while paused) |
| `s` | Playback speed 1× → 0.5× → 0.25× |
| `]` / `[` | Jump 5 s forward or back |
| `q` | Save and open the next take |
| `Esc` | Save and stop (run the command again to carry on) |

**When to press:** press the step key on the frame where the hand *starts moving toward the
object*. Press `0` once the step's end state is reached *and the hand pulls back*. Labels are
saved on every key press.

The steps:

| Key | Step |
|---|---|
| `1` | S1: Open the outer box |
| `2` | S2: Take out the red box |
| `3` | S3: Place the red box in Zone A |
| `4` | S4: Take out the yellow box |
| `5` | S5: Place the yellow box in Zone B |
| `6` | S6: Close the outer box |

## 5. Check and hand over

```sh
uv run python -c "import pathlib; [print(p.name) for p in sorted(pathlib.Path('data/raw').glob('*.mp4'))]"   # list the takes you recorded
uv run python -c "import sys; sys.path.insert(0, '.'); from tools.record_takes import pending_takes; from pathlib import Path; print([p.name for p in pending_takes(Path('data/raw'))])"   # list takes still waiting for labels (should be [])
```

`data/` is not in git. Hand over the whole `data/raw/` folder, not just the videos: every take is
a set of files (`.mp4`, `.csv`, `.frames.csv`, `.json`). Copy it to the team's shared drive.
