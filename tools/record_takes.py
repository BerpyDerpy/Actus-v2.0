"""T-REC: record takes with a live step annotator, or label an existing video.

Record (webcam, optional second camera / phone RTSP feed):

    uv run python tools/record_takes.py --operator op01 --scenario correct
    uv run python tools/record_takes.py --operator op02 --scenario skip_S2 --source 1 \\
        --source2 rtsp://192.168.1.50:8554/live --roll 90 --lighting dim --notes "lid sticky"

Toggle mode: the camera opens in a live standby preview; ``r`` starts a take, ``r`` stops and
saves it, and the next ``r`` starts the next take number. ``q`` ends the session.

    uv run python tools/record_takes.py --operator op01 --scenario correct --toggle

Label a file already recorded (e.g. shot with a phone camera app):

    uv run python tools/record_takes.py --label data/raw/op03_correct_02.mp4

Label every take that has no labels yet, one after another (``q`` saves and opens the next,
``Esc`` saves and stops the batch):

    uv run python tools/record_takes.py --label-pending            # data/raw
    uv run python tools/record_takes.py --label-pending data/other_dir

Keys (both modes)
-----------------
    1–9     step S1–S9 starts at this frame         0   idle (no step in progress)
    x       a non-step action starts ("other")      u   undo the last label
    q / Esc finish and save
  toggle mode only:
    r       start a take (standby) / stop and save it (recording)
  playback only:
    space   pause / resume      .  ,   step one frame forward / back (while paused)
    s       cycle speed 1× → 0.5× → 0.25×          ]  [   jump 5 s forward / back
  --label-pending only:
    q       save and open the next pending take     Esc save and stop the batch

Step-boundary rule: press the key on the frame where the operator's hand starts moving toward
the object for that step, and `0` once the step's end state is reached and the hand withdraws.

Outputs, in ``data/raw/`` by default, for take ``{operator}_{scenario}_{n:02d}``:

* ``<stem>.mp4``          the raw video at a fixed fps, NO overlay burned in.
* ``<stem>.csv``          label changes ``frame_idx,t,label`` (labels ``S1``…``S9``, ``idle``,
                          ``other``). The label holds from its frame until the next row.
* ``<stem>.frames.csv``   per written frame: ``frame_idx,t,t_capture,cam_seq,dup``. ``t`` is
                          seconds since the take started when the frame was written, and
                          ``t_capture`` is when the camera delivered it. ``dup=1`` marks a repeated
                          frame (the camera was slower than --fps).
* ``<stem>.json``         session metadata: operator, scenario, camera roll, lighting, notes, fps,
                          counts, and a frame-count check of the written file.
* ``<stem>_cam2.*``       the same for --source2. Frame indices are aligned with cam 1.

The written video has exactly one frame per ``frame_idx``. At every tick of the fixed-fps
clock the newest camera frame is written; a missed tick is filled with a duplicate, so the
video's timeline matches the wall clock.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TOOL_VERSION = 1
IDLE, OTHER = "idle", "other"
OPERATOR_RE = re.compile(r"^[A-Za-z0-9-]+$")  # no underscores: "_" separates name fields
SCENARIO_RE = re.compile(r"^[A-Za-z0-9_-]+$")
STEM_RE = re.compile(
    r"^(?P<op>[A-Za-z0-9-]+)_(?P<sc>[A-Za-z0-9_-]+?)_(?P<n>\d+)(?:_cam(?P<cam>\d+))?$"
)
SPEEDS = (1.0, 0.5, 0.25)
GRACE = 0.4  # fraction of a frame period to wait for a late camera frame
KEY_HELP = "1-9 step  0 idle  x other  u undo  q quit"
TOGGLE_HELP_STANDBY = "r start recording  q quit"
TOGGLE_HELP_RECORDING = "1-9 step  0 idle  x other  u undo  r stop take  q stop+quit"
BATCH_HELP = "1-9 step  0 idle  x other  u undo  q save+next  Esc save+stop"
WINDOW = "ACTUS take recorder"


# --------------------------------------------------------------------------------------------
# Naming and labels
# --------------------------------------------------------------------------------------------


def take_stem(operator: str, scenario: str, n: int) -> str:
    if not OPERATOR_RE.match(operator):
        raise ValueError(f"operator {operator!r} may use letters, digits and '-' only")
    if not SCENARIO_RE.match(scenario):
        raise ValueError(f"scenario {scenario!r} may use letters, digits, '_' and '-' only")
    return f"{operator}_{scenario}_{n:02d}"


def parse_take_stem(stem: str) -> dict:
    """``op01_skip_S2_03_cam2`` → ``{operator: op01, scenario: skip_S2, n: 3, cam: 2}``."""
    m = STEM_RE.match(stem)
    if not m:
        raise ValueError(f"{stem!r} does not follow {{operator}}_{{scenario}}_{{n}}")
    return {
        "operator": m["op"],
        "scenario": m["sc"],
        "n": int(m["n"]),
        "cam": int(m["cam"]) if m["cam"] else 1,
    }


def next_take_number(out_dir: Path, operator: str, scenario: str) -> int:
    taken = set()
    for p in out_dir.glob(f"{operator}_{scenario}_*.mp4"):
        try:
            info = parse_take_stem(p.stem)
        except ValueError:
            continue
        if info["operator"] == operator and info["scenario"] == scenario:
            taken.add(info["n"])
    return max(taken, default=0) + 1


def key_to_label(key: int) -> str | None:
    if key < 0:
        return None
    ch = chr(key & 0xFF).lower()
    if ch in "123456789":
        return f"S{ch}"
    if ch == "0":
        return IDLE
    if ch == "x":
        return OTHER
    return None


@dataclass
class LabelTrack:
    """Label changes keyed by frame index. The first row is always ``0, idle``."""

    rows: dict[int, tuple[float, str]] = field(default_factory=lambda: {0: (0.0, IDLE)})
    history: list[int] = field(default_factory=list)

    def set(self, frame_idx: int, t: float, label: str) -> None:
        self.rows[frame_idx] = (t, label)
        self.history.append(frame_idx)

    def undo(self) -> int | None:
        while self.history:
            fi = self.history.pop()
            if fi in self.rows and fi not in self.history:
                if fi == 0:
                    self.rows[0] = (0.0, IDLE)
                else:
                    del self.rows[fi]
                return fi
        return None

    def label_at(self, frame_idx: int) -> str:
        keys = [k for k in self.rows if k <= frame_idx]
        return self.rows[max(keys)][1] if keys else IDLE

    def items(self) -> list[tuple[int, float, str]]:
        out, prev = [], None
        for fi in sorted(self.rows):
            t, lab = self.rows[fi]
            if lab != prev:  # collapse repeated presses of the same key
                out.append((fi, t, lab))
            prev = lab
        return out

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx", "t", "label"])
            for fi, t, lab in self.items():
                w.writerow([fi, f"{t:.4f}", lab])
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> LabelTrack:
        track = cls()
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                track.rows[int(row["frame_idx"])] = (float(row["t"]), row["label"])
        return track


# --------------------------------------------------------------------------------------------
# Frame sources
# --------------------------------------------------------------------------------------------


class CaptureThread(threading.Thread):
    """Reads a cv2 source as fast as it delivers and keeps only the newest frame."""

    def __init__(self, source: str | int, width: int, height: int, fps: float, api: str):
        super().__init__(daemon=True)
        self.source = source
        api_id = {
            "any": cv2.CAP_ANY,
            "dshow": cv2.CAP_DSHOW,
            "msmf": cv2.CAP_MSMF,
            "avfoundation": cv2.CAP_AVFOUNDATION,
            "ffmpeg": cv2.CAP_FFMPEG,
        }[api]
        self.cap = cv2.VideoCapture(source, api_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video source {source!r}")
        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        self._lock = threading.Lock()
        self._latest: tuple[int, np.ndarray, float] | None = None
        self._stop = threading.Event()
        self.eof = False
        self.read_count = 0

    def run(self) -> None:
        seq = 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            t = time.monotonic()
            if not ok:
                self.eof = True  # a file ended or a stream died: never loop or reopen
                break
            with self._lock:
                self._latest = (seq, frame, t)
            seq += 1
            self.read_count = seq
        self.cap.release()

    def latest(self) -> tuple[int, np.ndarray, float] | None:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop.set()

    def describe(self) -> dict:
        return {"source": str(self.source), "backend": self.cap.getBackendName()}


class ReplayFrames(threading.Thread):
    """``replay:<scenario>``: rendered SIMULATED frames from the P0 replay source (self-test)."""

    def __init__(self, scenario: str, fps: float):
        super().__init__(daemon=True)
        from actus.capture.replay import ReplaySource

        self.scenario = scenario
        self.src = ReplaySource(scenario, fps=fps, render=True)
        self.fps = fps
        self._lock = threading.Lock()
        self._latest: tuple[int, np.ndarray, float] | None = None
        self._stop = threading.Event()
        self.eof = False
        self.read_count = 0

    def run(self) -> None:
        start = time.monotonic()
        for tick in self.src:
            if self._stop.is_set():
                return
            delay = start + tick.frame_idx / self.fps - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            with self._lock:
                self._latest = (tick.frame_idx, tick.frame.image, time.monotonic())
            self.read_count = tick.frame_idx + 1
        self.eof = True

    def latest(self):
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop.set()

    def describe(self) -> dict:
        return {"source": f"replay:{self.scenario}", "backend": "replay", "simulated": True}


def open_source(spec: str, args) -> CaptureThread | ReplayFrames:
    if spec.startswith("replay:"):
        return ReplayFrames(spec.split(":", 1)[1], args.fps)
    source: str | int = int(spec) if spec.isdigit() else spec
    return CaptureThread(source, args.width, args.height, args.fps, args.api)


# --------------------------------------------------------------------------------------------
# Writing and verification
# --------------------------------------------------------------------------------------------


def open_writer(path: Path, fps: float, size: tuple[int, int], codec: str) -> cv2.VideoWriter:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
    if not w.isOpened():
        raise RuntimeError(f"could not open a {codec!r} writer for {path}")
    return w


def count_frames(path: Path) -> dict:
    """Decode-count the frames of a written file with OpenCV, and with ffprobe if present."""
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    out: dict = {"opencv_decoded": n, "ffprobe": None}
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            res = subprocess.run(
                [ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
                 "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
                capture_output=True, text=True, timeout=120, check=True,
            )  # fmt: skip
            out["ffprobe"] = int(res.stdout.strip().split(",")[0])
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            out["ffprobe_error"] = str(e)
    return out


class CamRecorder:
    """One camera's writer plus its per-frame sidecar."""

    def __init__(self, src, stem_path: Path, fps: float, codec: str):
        self.src = src
        self.video_path = stem_path.with_suffix(".mp4")
        self.frames_path = stem_path.with_name(stem_path.name + ".frames.csv")
        self.fps, self.codec = fps, codec
        self.writer: cv2.VideoWriter | None = None
        self.size: tuple[int, int] | None = None
        self._frames_f = open(self.frames_path, "w", newline="", encoding="utf-8")  # noqa: SIM115 (closed in close())
        self._frames = csv.writer(self._frames_f)
        self._frames.writerow(["frame_idx", "t", "t_capture", "cam_seq", "dup"])
        self.last_seq: int | None = None
        self.written = self.dups = self.cam_skipped = 0

    def has_new(self) -> bool:
        latest = self.src.latest()
        return latest is not None and latest[0] != self.last_seq

    def write(self, frame_idx: int, t_rel: float, t0: float) -> np.ndarray:
        seq, img, t_cap = self.src.latest()
        if self.writer is None:
            self.size = (img.shape[1], img.shape[0])
            self.writer = open_writer(self.video_path, self.fps, self.size, self.codec)
        if (img.shape[1], img.shape[0]) != self.size:
            img = cv2.resize(img, self.size)
        dup = self.last_seq == seq
        if self.last_seq is not None and not dup:
            self.cam_skipped += max(0, seq - self.last_seq - 1)
        self.last_seq = seq
        self.writer.write(img)
        self._frames.writerow([frame_idx, f"{t_rel:.4f}", f"{t_cap - t0:.4f}", seq, int(dup)])
        self.written += 1
        self.dups += int(dup)
        return img

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
        self._frames_f.close()

    def summary(self) -> dict:
        return {
            "video": self.video_path.name,
            "frames_csv": self.frames_path.name,
            **self.src.describe(),
            "resolution": list(self.size) if self.size else None,
            "frames_written": self.written,
            "duplicated_frames": self.dups,
            "camera_frames_skipped": self.cam_skipped,
            "camera_frames_read": self.src.read_count,
        }


# --------------------------------------------------------------------------------------------
# Overlay
# --------------------------------------------------------------------------------------------


def _put(img, text, org, scale=0.6, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def overlay(
    img: np.ndarray,
    lines: list[str],
    label: str,
    rec: bool,
    max_w: int = 960,
    help_text: str = KEY_HELP,
):
    scale = min(1.0, max_w / img.shape[1])
    view = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img.copy()
    h, w = view.shape[:2]
    if rec:
        cv2.circle(view, (22, 24), 9, (0, 0, 255), -1)
    y = 30
    for ln in lines:
        _put(view, ln, (40 if y == 30 and rec else 12, y))
        y += 24
    color = (80, 220, 80) if label.startswith("S") else (200, 200, 200)
    _put(view, label.upper(), (w - 150, 44), 1.3, color, 3)
    _put(view, help_text, (12, h - 12), 0.5, (220, 220, 220))
    return view


def timeline_bar(view: np.ndarray, track: LabelTrack, pos: int, total: int) -> None:
    h, w = view.shape[:2]
    y0, y1 = h - 44, h - 30
    cv2.rectangle(view, (10, y0), (w - 10, y1), (60, 60, 60), -1)
    items = track.items()
    for i, (fi, _, lab) in enumerate(items):
        end = items[i + 1][0] if i + 1 < len(items) else total
        if lab == IDLE:
            continue
        x0 = 10 + int((w - 20) * fi / max(total, 1))
        x1 = 10 + int((w - 20) * end / max(total, 1))
        hue = (int(lab[1:]) * 40) % 180 if lab.startswith("S") else 0
        col = cv2.cvtColor(np.uint8([[[hue, 200, 230]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        cv2.rectangle(view, (x0, y0), (max(x1, x0 + 1), y1), col, -1)
    xp = 10 + int((w - 20) * pos / max(total, 1))
    cv2.line(view, (xp, y0 - 4), (xp, y1 + 4), (255, 255, 255), 2)


# --------------------------------------------------------------------------------------------
# Record mode
# --------------------------------------------------------------------------------------------


def parse_script(spec: str | None) -> list[tuple[float, str]]:
    """``"2.0:1,5.5:0,7:x"`` → scripted key presses at elapsed seconds (for tests / dry runs)."""
    if not spec:
        return []
    out = []
    for item in spec.split(","):
        t, _, key = item.partition(":")
        out.append((float(t), key.strip()))
    return sorted(out)


def open_sources(args) -> list:
    """Open --source (and --source2), start their threads and wait for a first frame."""
    sources = [open_source(args.source, args)]
    if args.source2:
        sources.append(open_source(args.source2, args))
    for s in sources:
        s.start()
    deadline = time.monotonic() + args.warmup_timeout
    while any(s.latest() is None for s in sources):
        if time.monotonic() > deadline or any(s.eof for s in sources):
            for s in sources:
                s.stop()
            raise SystemExit("no frames from the video source(s); check --source / --source2")
        time.sleep(0.01)
    return sources


def record(args) -> Path:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = args.n or next_take_number(out_dir, args.operator, args.scenario)
    stem = take_stem(args.operator, args.scenario, n)
    if (out_dir / f"{stem}.mp4").exists() and not args.force:
        raise SystemExit(f"{out_dir / stem}.mp4 exists; pass --force or another --n")

    sources = open_sources(args)
    try:
        meta_path, _ = record_take(args, sources, out_dir, n, parse_script(args.script))
    finally:
        for s in sources:
            s.stop()
        if not args.no_preview:
            cv2.destroyAllWindows()
    return meta_path


def record_session(args) -> list[Path]:
    """``--toggle``: the camera stays live in a standby preview; ``r`` starts a take and ``r``
    again stops and saves it. Each start/stop is one take with the next free number."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    take_stem(args.operator, args.scenario, 1)  # validate the names before opening cameras
    sources = open_sources(args)
    script = parse_script(args.script)
    preview = not args.no_preview
    t_session = time.monotonic()
    saved: list[Path] = []
    first_n = args.n
    print(f"standby: r = start/stop a take, q = quit  (operator {args.operator}, "
          f"scenario {args.scenario}) -> {out_dir}")  # fmt: skip
    try:
        while not any(s.eof for s in sources):
            n = first_n or next_take_number(out_dir, args.operator, args.scenario)
            stem = take_stem(args.operator, args.scenario, n)
            keys: list[int] = []
            elapsed = time.monotonic() - t_session
            while script and script[0][0] <= elapsed:
                keys.append(ord(script.pop(0)[1][0]))
            if preview:
                latest = sources[0].latest()
                if latest is not None:
                    lines = [f"STANDBY  next take: {stem}", "press r to start recording"]
                    view = overlay(latest[1], lines, "ready", False, help_text=TOGGLE_HELP_STANDBY)
                    cv2.imshow(WINDOW, view)
                keys.append(cv2.waitKey(1))
            else:
                time.sleep(1 / args.fps)
            if any(k in (ord("q"), 27) for k in keys):
                break
            if ord("r") not in keys:
                continue
            first_n = None
            if (out_dir / f"{stem}.mp4").exists() and not args.force:
                print(f"{out_dir / stem}.mp4 exists; pass --force or another --n")
                continue
            meta_path, quit_session = record_take(
                args, sources, out_dir, n, script, script_t0=t_session, toggle=True
            )
            saved.append(meta_path)
            if quit_session:
                break
    except KeyboardInterrupt:
        pass
    finally:
        for s in sources:
            s.stop()
        if preview:
            cv2.destroyAllWindows()
    print(f"session done: {len(saved)} take(s) saved")
    for p in saved:
        print(f"  {p.with_suffix('.mp4')}")
    return saved


def record_take(
    args,
    sources: list,
    out_dir: Path,
    n: int,
    script: list[tuple[float, str]],
    script_t0: float | None = None,
    toggle: bool = False,
) -> tuple[Path, bool]:
    """Record one take from already-running sources. Returns (metadata path, quit requested).

    ``script`` times are seconds since ``script_t0`` (default: the take's start). In toggle mode
    ``r`` ends the take and returns to standby; ``q`` / Esc also ends the session."""
    stem = take_stem(args.operator, args.scenario, n)
    cams = [CamRecorder(sources[0], out_dir / stem, args.fps, args.codec)]
    if len(sources) > 1:
        cams.append(CamRecorder(sources[1], out_dir / f"{stem}_cam2", args.fps, args.codec))

    labels_path = out_dir / f"{stem}.csv"
    track = LabelTrack()
    preview = not args.no_preview
    help_text = TOGGLE_HELP_RECORDING if toggle else KEY_HELP
    wall_start = dt.datetime.now().astimezone()
    t0 = time.monotonic()
    clock0 = t0 if script_t0 is None else script_t0
    frame_idx = -1
    last_t = 0.0
    stop_reason = "key"
    quit_requested = True
    stop_hint = "r to stop, q to stop and quit" if toggle else "q to stop"
    print(f"recording {stem} at {args.fps:g} fps -> {out_dir}  ({stop_hint})")

    try:
        while True:
            elapsed = time.monotonic() - t0
            if args.duration and elapsed >= args.duration:
                stop_reason = "duration"
                quit_requested = not toggle
                break
            if any(s.eof for s in sources):
                stop_reason = "source_ended"
                break
            # One written frame per fixed-fps tick. If the loop fell behind (e.g. the window was
            # dragged), the missed ticks are filled with duplicates so the video keeps wall time.
            target = math.floor(elapsed * args.fps)
            img = None
            waiting = False
            while frame_idx < target:
                # Equal camera and output rates beat against each other; give a late camera
                # frame up to 40% of a period before duplicating, to avoid dup+skip pairs.
                lag = elapsed - (frame_idx + 1) / args.fps
                if lag < GRACE * (1 / args.fps) and not cams[0].has_new():
                    waiting = True
                    break
                frame_idx += 1
                last_t = time.monotonic() - t0
                for cam in cams:
                    written = cam.write(frame_idx, last_t, t0)
                    img = written if cam is cams[0] else img

            keys: list[int] = []
            script_elapsed = time.monotonic() - clock0
            while script and script[0][0] <= script_elapsed:
                keys.append(ord(script.pop(0)[1][0]))
            if preview:
                if img is not None:
                    lines = [
                        f"{stem}  f{frame_idx}  {elapsed:6.1f}s",
                        f"roll {args.roll}  light {args.lighting}  dup {cams[0].dups}",
                    ]
                    view = overlay(img, lines, track.label_at(frame_idx), True, help_text=help_text)
                    cv2.imshow(WINDOW, view)
                keys.append(cv2.waitKey(1))
            else:
                next_tick = t0 + (frame_idx + 1) / args.fps
                time.sleep(0.002 if waiting else max(0.0, next_tick - time.monotonic()))

            if any(k in (ord("q"), 27) for k in keys):
                break
            if toggle and ord("r") in keys:
                stop_reason = "toggle"
                quit_requested = False
                break
            for key in keys:
                if key in (ord("u"), 8):
                    track.undo()
                    track.save(labels_path)
                lab = key_to_label(key)
                if lab and frame_idx >= 0:
                    track.set(frame_idx, last_t, lab)
                    track.save(labels_path)  # autosave: a crash never loses labels
                    print(f"  f{frame_idx:6d} {last_t:7.2f}s  {lab}")
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    finally:
        for cam in cams:
            cam.close()
    track.save(labels_path)

    duration = (frame_idx + 1) / args.fps
    meta = {
        "tool": "record_takes",
        "tool_version": TOOL_VERSION,
        "take": stem,
        "mode": "record",
        "operator": args.operator,
        "scenario": args.scenario,
        "n": n,
        "camera_roll_deg": args.roll,
        "lighting": args.lighting,
        "notes": args.notes,
        "fps": args.fps,
        "codec": args.codec,
        "started_at": wall_start.isoformat(timespec="seconds"),
        "duration_s": round(duration, 3),
        "frames": frame_idx + 1,
        "stop_reason": stop_reason,
        "labels_csv": labels_path.name,
        "label_rows": len(track.items()),
        "key_map": {"1-9": "S1-S9 start", "0": IDLE, "x": OTHER},
        "simulated": any(s.describe().get("simulated") for s in sources),
        "cameras": [],
    }
    for cam in cams:
        info = cam.summary()
        counts = count_frames(cam.video_path)
        info["frame_count_check"] = counts
        info["frame_count_ok"] = counts["opencv_decoded"] == cam.written and (
            counts["ffprobe"] is None or counts["ffprobe"] == cam.written
        )
        meta["cameras"].append(info)
        # dup+skip pairs are timing jitter; only a net shortfall means the camera is too slow
        dup_ratio = (cam.dups - cam.cam_skipped) / max(cam.written, 1)
        if dup_ratio > 0.1:
            print(f"WARNING {cam.video_path.name}: {dup_ratio:.0%} duplicated frames; "
                  f"the camera delivers fewer than {args.fps:g} fps (lower --fps?)")  # fmt: skip
        if not info["frame_count_ok"]:
            print(f"WARNING {cam.video_path.name}: frame count mismatch {counts} vs {cam.written}")
    meta_path = out_dir / f"{stem}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    ok = all(c["frame_count_ok"] for c in meta["cameras"])
    print(f"saved {stem}: {frame_idx + 1} frames, {duration:.1f}s, "
          f"{len(track.items())} label rows, frame count {'OK' if ok else 'MISMATCH'}")  # fmt: skip
    return meta_path, quit_requested


# --------------------------------------------------------------------------------------------
# Playback (label an existing file)
# --------------------------------------------------------------------------------------------


class _Player:
    """Sequential reader with a ring of recent frames for stepping back. Frame indices are
    counted by decode order (re-anchored from the container only after a seek)."""

    def __init__(self, cap: cv2.VideoCapture, ring: int):
        self.cap = cap
        self.ring: list[tuple[int, float, np.ndarray]] = []
        self.ring_max = max(2, ring)
        self.ri = -1
        self.next_idx = 0

    @property
    def current(self) -> tuple[int, float, np.ndarray]:
        return self.ring[self.ri]

    def _read(self) -> bool:
        ok, frame = self.cap.read()
        if not ok:
            return False
        t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        self.ring.append((self.next_idx, t, frame))
        self.next_idx += 1
        if len(self.ring) > self.ring_max:
            self.ring.pop(0)
            self.ri -= 1
        return True

    def forward(self) -> bool:
        if self.ri < len(self.ring) - 1:
            self.ri += 1
            return True
        if self._read():
            self.ri = len(self.ring) - 1
            return True
        return False

    def back(self) -> None:
        self.ri = max(0, self.ri - 1)

    def seek(self, target: int) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target))
        self.next_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        keep = self.ring[self.ri] if self.ring else None
        self.ring, self.ri = [], -1
        if not self.forward() and keep is not None:  # seeked past the end: stay put
            self.ring, self.ri = [keep], 0


def is_pending(video: Path) -> bool:
    """A take still needs labelling if it was never labelled in playback and its CSV holds no
    label other than the initial ``idle`` row (i.e. nobody pressed step keys while recording)."""
    meta_path = video.with_suffix(".json")
    if meta_path.exists() and "labelled_at" in json.loads(meta_path.read_text(encoding="utf-8")):
        return False
    labels_path = video.with_suffix(".csv")
    if not labels_path.exists():
        return True
    return all(lab == IDLE for _, _, lab in LabelTrack.load(labels_path).items())


def pending_takes(folder: Path) -> list[Path]:
    """Camera-1 takes in ``folder`` that still need labels (cam 2 shares cam 1's labels)."""
    out = []
    for video in sorted(folder.glob("*.mp4")):
        try:
            if parse_take_stem(video.stem)["cam"] != 1:
                continue
        except ValueError:
            pass  # not our naming (e.g. a phone file): still labellable
        if is_pending(video):
            out.append(video)
    return out


def label_pending(args) -> list[Path]:
    folder = Path(args.label_pending)
    if not folder.is_dir():
        raise SystemExit(f"{folder} is not a folder")
    todo = pending_takes(folder)
    if not todo:
        print(f"nothing to label in {folder}")
        return []
    print(f"{len(todo)} take(s) to label in {folder}  (q = save + next, Esc = save + stop)")
    done = []
    for i, video in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {video.name}")
        labels_path, stop = label_video(
            args, video, progress=f"take {i}/{len(todo)}", help_text=BATCH_HELP
        )
        done.append(labels_path)
        if stop:
            print(f"stopped; {len(todo) - i} take(s) still pending")
            break
    return done


def label_existing(args) -> Path:
    labels_path, _ = label_video(args, Path(args.label))
    return labels_path


def label_video(
    args, video: Path, progress: str = "", help_text: str = KEY_HELP
) -> tuple[Path, bool]:
    """Play ``video`` and label it. Returns (labels CSV path, Esc pressed)."""
    if not video.exists():
        raise SystemExit(f"{video} not found")
    labels_path = video.with_suffix(".csv")
    track = (
        LabelTrack.load(labels_path) if labels_path.exists() and not args.force else LabelTrack()
    )
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 240:
        raise SystemExit(f"{video}: container reports no usable fps; re-mux it with ffmpeg first")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    window = f"ACTUS labeller - {video.name}"
    script = parse_script(args.script)
    preview = not args.no_preview
    player = _Player(cap, args.ring)
    if not player.forward():
        raise SystemExit(f"{video}: no decodable frames")

    paused, speed_i = False, 0
    esc = False
    next_due = time.monotonic() + 1.0 / fps
    while True:
        # Without a window there is nothing to pace for: run through the file (tests, dry runs).
        if not paused and (not preview or time.monotonic() >= next_due):
            if not player.forward():
                if not preview or args.exit_at_end:
                    break
                paused = True  # hold on the last frame; q saves and quits
            next_due = time.monotonic() + 1.0 / (fps * SPEEDS[speed_i])
        pos, t, img = player.current

        keys: list[int] = []
        while script and script[0][0] <= t:
            keys.append(ord(script.pop(0)[1][0]))
        if preview:
            state = f"  {SPEEDS[speed_i]:g}x" + ("  PAUSED" if paused else "")
            lines = [f"f{pos}/{total}  {t:7.2f}s{state}", "space pause  . , step  [ ] 5s  s speed"]
            if progress:
                lines.insert(0, f"{progress}  {video.name}")
            view = overlay(
                img, lines, track.label_at(pos), False, max_w=args.view_width, help_text=help_text
            )
            timeline_bar(view, track, pos, total)
            cv2.imshow(window, view)
            keys.append(cv2.waitKey(1 if not paused else 30))

        if any(k in (ord("q"), 27) for k in keys):
            esc = 27 in keys
            break
        for key in keys:
            if key == ord(" "):
                paused = not paused
            elif key == ord("s"):
                speed_i = (speed_i + 1) % len(SPEEDS)
            elif key == ord(".") and paused:
                player.forward()
            elif key == ord(",") and paused:
                player.back()
            elif key in (ord("]"), ord("[")):
                player.seek(pos + round((5 if key == ord("]") else -5) * fps))
            elif key in (ord("u"), 8):
                track.undo()
                track.save(labels_path)
            elif lab := key_to_label(key):
                track.set(pos, t, lab)
                track.save(labels_path)
                print(f"  f{pos:6d} {t:7.2f}s  {lab}")

    cap.release()
    if preview:
        cv2.destroyAllWindows()
    track.save(labels_path)

    meta_path = video.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    try:
        name_info = parse_take_stem(video.stem)
    except ValueError:
        name_info = {}
    meta.setdefault("take", video.stem)
    meta.setdefault("operator", args.operator or name_info.get("operator"))
    meta.setdefault("scenario", args.scenario or name_info.get("scenario"))
    meta.setdefault("camera_roll_deg", args.roll)
    meta.setdefault("lighting", args.lighting)
    meta.setdefault("notes", args.notes)
    meta.setdefault("fps", fps)
    meta.setdefault("mode", "playback_labelled")
    meta["labels_csv"] = labels_path.name
    meta["label_rows"] = len(track.items())
    meta["labelled_at"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    meta["labelled_with"] = {"tool": "record_takes --label", "tool_version": TOOL_VERSION}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {labels_path.name}: {len(track.items())} label rows")
    return labels_path, esc


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Record takes with a live step annotator, or label an existing video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Keys: 1-9 step starts, 0 idle, x other, u undo, q quit. See the module docstring.",
    )
    ap.add_argument("--label", metavar="VIDEO", help="playback mode: label an existing file")
    ap.add_argument(
        "--label-pending",
        metavar="DIR",
        nargs="?",
        const="data/raw",
        help="label every take in DIR (default data/raw) that has no labels yet, one by one",
    )
    ap.add_argument(
        "--view-width", type=int, default=640, help="playback: labeller window width (default 640)"
    )
    ap.add_argument("--operator", help="operator id, e.g. op01 (no underscores)")
    ap.add_argument("--scenario", help="scenario name, e.g. correct, skip_S2, sweep, distractor")
    ap.add_argument("--n", type=int, help="take number (default: next free)")
    ap.add_argument("--out", default="data/raw", help="output directory (default data/raw)")
    ap.add_argument("--source", default="0", help="webcam index, file, rtsp:// URL, replay:<name>")
    ap.add_argument("--source2", help="optional second camera (index, URL, file)")
    ap.add_argument("--fps", type=float, default=30.0, help="fixed output fps (default 30)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument(
        "--api", default="any", choices=["any", "dshow", "msmf", "avfoundation", "ffmpeg"]
    )
    ap.add_argument("--codec", default="mp4v", help="fourcc for the mp4 (default mp4v)")
    ap.add_argument("--roll", type=int, default=0, help="camera roll angle, degrees")
    ap.add_argument("--lighting", default="normal", help="e.g. normal, dim, bright, mixed")
    ap.add_argument("--notes", default="", help="free text for the session metadata")
    ap.add_argument("--duration", type=float, help="stop after this many seconds (per take)")
    ap.add_argument(
        "--toggle",
        action="store_true",
        help="start in a live standby preview; r starts/stops each take (one file per take)",
    )
    ap.add_argument("--no-preview", action="store_true", help="no window (headless / tests)")
    ap.add_argument("--script", help='scripted keys "t:key,...", t = seconds (tests, dry runs)')
    ap.add_argument("--force", action="store_true", help="overwrite / ignore existing outputs")
    ap.add_argument("--warmup-timeout", type=float, default=15.0)
    ap.add_argument("--ring", type=int, default=300, help="playback: frames kept for stepping back")
    ap.add_argument("--exit-at-end", action="store_true", help="playback: quit at end of file")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fps <= 0:
        raise SystemExit("--fps must be > 0")
    if args.label:
        label_existing(args)
        return 0
    if args.label_pending:
        label_pending(args)
        return 0
    if not args.operator or not args.scenario:
        raise SystemExit("record mode needs --operator and --scenario (or use --label VIDEO)")
    if args.toggle:
        record_session(args)
    else:
        record(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
