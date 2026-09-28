"""Preflight: what will work on this machine, what will degrade, and what is broken (T-PKG).

    uv run python tools/doctor.py                     # check the default config
    uv run python tools/doctor.py --config configs/site.yaml --override stream.target_ip=10.0.0.5
    uv run python tools/doctor.py --camera            # also open capture.source and read a frame
    uv run python tools/doctor.py --strict            # warnings fail the run (CI, packaging)
    uv run python tools/doctor.py --json              # machine-readable

It reads the same config the pipeline reads and checks the things the pipeline needs:

* Python and the optional extras (ultralytics, mediapipe, kokoro, vosk, fastapi ...);
* every model file the config points at (and the hidden run-time downloads YOLO-World would
  otherwise make: the ``clip`` package and its text encoder);
* the MediaMTX and FFmpeg binaries, and that FFmpeg can encode H.264 (``libx264``);
* the built GUI, the protocol, and whether ``configs/rack.yaml`` still holds placeholders;
* the ports, the RTSP target, the audio devices, free disk space.

The pipeline degrades on purpose when a sink or a model is missing (``Vault/T-INT.md`` section
4), so a missing optional piece is a **WARN** that says what you lose, not a failure. A **FAIL**
is something without which nothing can run. Exit status: 0 unless there is a FAIL (or, with
``--strict``, a WARN); 2 if the doctor itself could not run.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from actus.core.config import Config, load_config  # noqa: E402
from tools._pkglib import load_manifest, platform_tag, say  # noqa: E402

FETCH = "uv run python tools/fetch_models.py"

# (module, extra, severity if missing, what is lost)
MODULES: tuple[tuple[str, str, str, str], ...] = (
    ("numpy", "base", "fail", "nothing runs"),
    ("yaml", "base", "fail", "nothing runs"),
    ("pydantic", "base", "fail", "the protocol cannot load"),
    ("cv2", "base", "fail", "no video capture, ArUco rack frame or overlay"),
    ("ultralytics", "perception", "warn", "no object detector"),
    ("mediapipe", "perception", "warn", "no hand or pose landmarks"),
    ("onnxruntime", "perception", "warn", "ONNX models fall back to their .npz numpy versions"),
    ("kokoro_onnx", "voice", "warn", "TTS falls back to pyttsx3, then to text only"),
    ("vosk", "voice", "warn", "no spoken answers; the GUI Yes/No still works"),
    ("sounddevice", "voice", "warn", "no audio output or input"),
    ("fastapi", "ui", "warn", "no GUI"),
    ("uvicorn", "ui", "warn", "no GUI"),
)


@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail | info
    detail: str
    effect: str = ""  # what is lost until this is fixed
    fix: str = ""


def _ok(name: str, detail: str) -> Check:
    return Check(name, "ok", detail)


def _info(name: str, detail: str, effect: str = "", fix: str = "") -> Check:
    return Check(name, "info", detail, effect, fix)


def _warn(name: str, detail: str, effect: str, fix: str = "") -> Check:
    return Check(name, "warn", detail, effect, fix)


def _fail(name: str, detail: str, effect: str, fix: str = "") -> Check:
    return Check(name, "fail", detail, effect, fix)


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _rp(root: Path, p: str | Path) -> Path:
    p = Path(str(p)).expanduser()
    return p if p.is_absolute() else root / p


def _file(name: str, path: Path, effect: str, fix: str, *, is_dir: bool = False) -> Check:
    found = path.is_dir() if is_dir else path.is_file()
    if found:
        return _ok(name, str(path))
    return _warn(name, f"{path} is missing", effect, fix)


# --------------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------------


def check_python() -> list[Check]:
    v = sys.version_info
    text = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) < (3, 11):
        return [_fail("python", f"{text}; ACTUS needs 3.11+", "nothing runs", "uv sync")]
    if (v.major, v.minor) >= (3, 13):
        return [
            _warn(
                "python",
                f"{text}; the project pins 3.11 (.python-version)",
                "MediaPipe may have no wheels for this Python",
                "uv sync",
            )
        ]
    return [_ok("python", text)]


def check_modules(cfg: Config, has_module: Callable[[str], bool]) -> list[Check]:
    out: list[Check] = []
    voice_on = bool(cfg.get("voice.enabled", True))
    for module, extra, severity, effect in MODULES:
        if extra == "voice" and not voice_on:
            continue
        if has_module(module):
            out.append(_ok(f"module {module}", extra))
        else:
            fix = "uv sync" if extra == "base" else f"uv sync --extra {extra}"
            out.append(Check(f"module {module}", severity, f"not installed ({extra})", effect, fix))
    if has_module("cv2"):
        try:
            import cv2
        except Exception as e:  # installed but broken, e.g. libGL.so.1 missing on headless Linux
            out.append(
                _fail(
                    "module cv2",
                    f"installed but does not import ({type(e).__name__}: {e})",
                    "no camera, no rack frame, no video",
                    "Linux: apt install libgl1 libglib2.0-0; otherwise reinstall with uv sync",
                )
            )
            return out
        if not hasattr(cv2, "aruco"):
            out.append(
                _fail(
                    "cv2.aruco",
                    "OpenCV has no aruco module",
                    "no rack frame, so no spatial evidence",
                    "uv sync (the project needs opencv-contrib-python, not opencv-python)",
                )
            )
    return out


def _asset_id_for(root: Path, path: Path) -> str | None:
    """The manifest asset that is downloaded to ``path``, if there is one."""
    try:
        assets = load_manifest(root / "configs" / "assets.yaml")
    except Exception:  # a broken manifest must not stop the doctor from reporting the rest
        return None
    return next((a.id for a in assets if a.kind != "local" and a.dest_path(root) == path), None)


def _detector_checks(cfg: Config, root: Path, has_module: Callable[[str], bool]) -> list[Check]:
    if cfg.get("detector.backend", "open_vocab") == "custom":
        weights = _rp(root, cfg.get("detector.custom_weights", ""))
        asset = _asset_id_for(root, weights)
        if asset:
            fix = f"{FETCH} {asset}"
        else:
            fix = "train it (tools/train_detector.py) or set detector.backend=open_vocab"
        return [_file("detector weights", weights, "the custom detector cannot load", fix)]
    weights = _rp(root, cfg.get("detector.open_vocab_weights", "yolov8s-worldv2.pt"))
    out = [
        _file(
            "detector weights",
            weights,
            "ultralytics downloads it at first use (needs the network)",
            f"{FETCH} yolo_world_s",
        )
    ]
    if has_module("clip"):
        out.append(_ok("clip package", "importable"))
    else:
        out.append(
            _warn(
                "clip package",
                "not installed",
                "YOLO-World installs it from git at first use (needs the network and git)",
                f"{FETCH} --warm",
            )
        )
    out.append(
        _file(
            "CLIP text encoder",
            Path("~/.cache/clip/ViT-B-32.pt").expanduser(),
            "YOLO-World downloads it at first use (needs the network)",
            f"{FETCH} clip_vit_b32",
        )
    )
    return out


def _perception_checks(cfg: Config, root: Path) -> list[Check]:
    out: list[Check] = []
    if cfg.get("perception.hands", True):
        hand = cfg.get("perception.hand_model_path") or "models/perception/hand_landmarker.task"
        fix = f"{FETCH} hand_landmarker"
        out.append(_file("hand landmarker", _rp(root, hand), "no hand landmarks", fix))
    if cfg.get("perception.pose", True):
        variant = cfg.get("perception.pose_model", "lite")
        default = f"models/perception/pose_landmarker_{variant}.task"
        pose = cfg.get("perception.pose_model_path") or default
        fix = f"{FETCH} pose_landmarker_{variant}"
        out.append(_file("pose landmarker", _rp(root, pose), "no pose landmarks", fix))
    return out


def _voice_checks(cfg: Config, root: Path) -> list[Check]:
    out: list[Check] = []
    if not cfg.get("voice.enabled", True):
        return out
    if cfg.get("voice.tts.engine", "kokoro") == "kokoro":
        effect = "Kokoro cannot load; TTS falls back to pyttsx3, then to text only"
        for key in ("kokoro_model", "kokoro_voices"):
            path = _rp(root, cfg.get(f"voice.tts.{key}", key))
            out.append(_file(key.replace("_", " "), path, effect, f"{FETCH} {key}"))
    if cfg.get("voice.asr.enabled", True):
        model = _rp(root, cfg.get("voice.asr.model", ""))
        effect = "no spoken answers; the GUI Yes/No still works"
        out.append(_file("vosk model", model, effect, f"{FETCH} vosk_small_en", is_dir=True))
    return out


def _trained_checks(cfg: Config, root: Path) -> list[Check]:
    out: list[Check] = []
    if not cfg.get("models.enabled", True):
        return out
    for key in ("hoi", "temporal"):
        stem = _rp(root, cfg.get(f"models.{key}", ""))
        found = [s for s in (".npz", ".onnx") if stem.with_suffix(s).is_file()]
        if found:
            out.append(_ok(f"{key} model", str(stem.with_suffix(found[0]))))
        else:
            out.append(
                _info(
                    f"{key} model",
                    f"{stem}.npz is not there",
                    "the engine runs on rule evidence alone",
                    f"train it (tools/train_{key}.py)",
                )
            )
    return out


def check_models(cfg: Config, root: Path, has_module: Callable[[str], bool]) -> list[Check]:
    return [
        *_detector_checks(cfg, root, has_module),
        *_perception_checks(cfg, root),
        *_voice_checks(cfg, root),
        *_trained_checks(cfg, root),
    ]


def check_protocol_and_rack(cfg: Config, root: Path) -> list[Check]:
    out: list[Check] = []
    proto = _rp(root, cfg.get("protocol.path", "protocols/bas_boxes.yaml"))
    if not proto.is_file():
        out.append(_fail("protocol", f"{proto} is missing", "nothing to validate against"))
    else:
        try:
            from actus.protocol.schema import load_protocol

            load_protocol(proto)
            out.append(_ok("protocol", str(proto)))
        except Exception as e:  # the schema raises its own error types
            out.append(_fail("protocol", f"{proto}: {e}", "the engine cannot start"))

    rack = _rp(root, cfg.get("rack.config", "configs/rack.yaml"))
    try:
        import yaml

        with open(rack, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("the top level is not a mapping")
        measured = bool(data.get("measured", False))
    except Exception as e:  # unreadable, bad YAML (a YAMLError is not a ValueError), wrong shape
        out.append(_fail("rack geometry", f"{rack}: {type(e).__name__}: {e}", "no rack frame"))
    else:
        if measured:
            out.append(_ok("rack geometry", f"{rack} (measured)"))
        else:
            out.append(
                _warn(
                    "rack geometry",
                    f"{rack} still holds placeholder numbers",
                    "zones will not line up with your board, so S3 and S5 cannot verify",
                    "measure the printed markers (H-PROPS), fill the file, set measured: true",
                )
            )

    static = _rp(root, cfg.get("ui.static_dir", "ui/dist"))
    if (static / "index.html").is_file():
        out.append(_ok("GUI build", str(static)))
    else:
        out.append(
            _warn(
                "GUI build",
                f"{static}/index.html is missing",
                "the GUI port serves an error page",
                "cd ui && npm ci && npm run build (or install a bundle that carries ui/dist)",
            )
        )
    return out


def find_binary(spec: str, root: Path, search_path: str | None = None) -> str | None:
    """Where the pipeline would find ``spec``: at that path (relative to the repo, which is the
    working directory it runs in), or on PATH with ``bin/`` first (``scripts/run.*`` put ``bin/``
    on PATH before they start). ``search_path`` replaces PATH; tests pass ``""``."""
    p = Path(spec)
    for cand in (p if p.is_absolute() else root / p, root / (spec + ".exe")):
        if cand.is_file():
            return str(cand)
    dirs = [str(root / "bin")]
    extra = os.environ.get("PATH", "") if search_path is None else search_path
    if extra:
        dirs.append(extra)
    return shutil.which(spec, path=os.pathsep.join(dirs))


def _ffmpeg_encoders(ffmpeg: str) -> str:
    try:
        run = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return run.stdout


def check_binaries(
    cfg: Config,
    root: Path,
    encoders: Callable[[str], str] = _ffmpeg_encoders,
    search_path: str | None = None,
) -> list[Check]:
    if not cfg.get("stream.enabled", True):
        return [_info("streaming", "stream.enabled is false: no RTSP push")]
    out: list[Check] = []
    mtx = find_binary(cfg.get("stream.mediamtx_bin", "bin/mediamtx"), root, search_path)
    if mtx:
        out.append(_ok("mediamtx", mtx))
    else:
        out.append(
            _warn(
                "mediamtx",
                "not found",
                "no RTSP or WebRTC server: the GUI falls back to MJPEG, recording still works",
                f"{FETCH} mediamtx",
            )
        )
    ff = find_binary(cfg.get("stream.ffmpeg_bin", "ffmpeg"), root, search_path)
    if not ff:
        effect = "no RTSP push (recording still works)"
        out.append(_warn("ffmpeg", "not found", effect, f"{FETCH} ffmpeg"))
    elif "libx264" not in encoders(ff):
        out.append(
            _warn(
                "ffmpeg",
                f"{ff} has no libx264 encoder",
                "the RTSP push fails at start",
                f"{FETCH} ffmpeg --force (the BtbN build has libx264)",
            )
        )
    else:
        out.append(_ok("ffmpeg", f"{ff} (libx264)"))
    return out


def _port_free(host: str, port: int) -> bool:
    """Nothing is listening on ``port`` and a server could bind it.

    A connect probe comes first: on Windows a bind to 127.0.0.1 succeeds even while another
    program listens on all interfaces. On POSIX the bind sets ``SO_REUSEADDR``, as uvicorn and
    MediaMTX do, so TIME_WAIT leftovers from the previous run do not count as "in use". (On
    Windows that option would let the bind succeed over a live listener, so it is not set.)"""
    probe = "127.0.0.1" if host in ("", "0.0.0.0") else host
    if _reachable(probe, port):
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if os.name != "nt":
            with contextlib.suppress(OSError):
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def _reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.5):
            return True
    except OSError:
        return False


def _local_port_check(label: str, port: int) -> Check:
    if _port_free("127.0.0.1", port):
        return _ok(label, f"127.0.0.1:{port} is free for MediaMTX")
    return _warn(
        label,
        f"127.0.0.1:{port} is in use",
        "MediaMTX cannot bind (is another one already running?)",
        "stop it, or --override stream.port=<a free port>",
    )


def check_network(cfg: Config) -> list[Check]:
    out: list[Check] = []
    host, port = str(cfg.get("ui.host", "127.0.0.1")), int(cfg.get("ui.port", 8000))
    if _port_free(host, port):
        out.append(_ok("GUI port", f"{host}:{port} is free"))
    else:
        out.append(
            _warn(
                "GUI port",
                f"{host}:{port} is in use",
                "the GUI cannot start",
                "stop the other program, or --override ui.port=8001",
            )
        )
    if not cfg.get("stream.enabled", True):
        return out
    target = str(cfg.get("stream.target_ip", "127.0.0.1"))
    rtsp = int(cfg.get("stream.port", 8554))
    if target in ("127.0.0.1", "localhost", "::1"):
        out.append(_local_port_check("RTSP port", rtsp))
        out.append(_local_port_check("WebRTC port", int(cfg.get("stream.webrtc_port", 8889))))
        view = f"this machine; view with: vlc rtsp://127.0.0.1:{rtsp}/actus/annotated"
        out.append(_info("stream target", view))
    elif _reachable(target, rtsp):
        out.append(_ok("stream target", f"{target}:{rtsp} accepts connections"))
    else:
        out.append(
            _warn(
                "stream target",
                f"nothing listens at {target}:{rtsp}",
                "the push keeps retrying with backoff; recording is unaffected",
                "start MediaMTX on that machine, or check stream.target_ip",
            )
        )
    return out


def _camera_check(source: Any) -> Check:
    try:
        import cv2

        src: Any = int(source) if str(source).lstrip("-").isdigit() else str(source)
        cap = cv2.VideoCapture(src)
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
    except Exception as e:  # any backend error means "not usable"
        return _warn("camera", f"{source!r}: {e}", "no live input", "check capture.source")
    if ok and frame is not None:
        h, w = frame.shape[:2]
        return _ok("camera", f"{source!r} delivers {w}x{h}")
    return _warn(
        "camera",
        f"{source!r} opened but gave no frame",
        "no live input",
        "close other apps using it, or set capture.source / --source",
    )


def check_devices(cfg: Config, has_module: Callable[[str], bool], camera: bool) -> list[Check]:
    out: list[Check] = []
    if cfg.get("voice.enabled", True) and has_module("sounddevice"):
        try:
            import sounddevice as sd

            devices = list(sd.query_devices())
            n_out = sum(1 for d in devices if d.get("max_output_channels", 0) > 0)
            n_in = sum(1 for d in devices if d.get("max_input_channels", 0) > 0)
        except Exception as e:  # PortAudio raises OSError or its own error type
            out.append(_warn("audio", f"cannot query devices: {e}", "no voice in or out"))
        else:
            if n_out:
                out.append(_ok("audio output", f"{n_out} device(s)"))
            else:
                out.append(_warn("audio output", "none", "prompts and alerts are text only"))
            if n_in or not cfg.get("voice.asr.enabled", True):
                out.append(_ok("microphone", f"{n_in} device(s)"))
            else:
                out.append(_warn("microphone", "none", "no spoken answers; use the GUI Yes/No"))
    if camera:
        out.append(_camera_check(cfg.get("capture.source", 0)))
    return out


def check_system(root: Path) -> list[Check]:
    out: list[Check] = []
    free_gb = shutil.disk_usage(root).free / 2**30
    if free_gb < 2.0:
        out.append(
            _warn(
                "disk space",
                f"{free_gb:.1f} GB free",
                "recordings need room",
                "free some space or set run.runs_dir",
            )
        )
    else:
        out.append(_ok("disk space", f"{free_gb:.0f} GB free"))
    if (root / ".actus-offline").is_file():
        out.append(_info("install mode", "offline (scripts/run.* set UV_OFFLINE and UV_NO_SYNC)"))
    return out


def run_checks(
    cfg: Config,
    root: Path = REPO_ROOT,
    *,
    has_module: Callable[[str], bool] = _has_module,
    encoders: Callable[[str], str] = _ffmpeg_encoders,
    search_path: str | None = None,
    network: bool = True,
    camera: bool = False,
) -> list[Check]:
    out = check_python()
    out += check_modules(cfg, has_module)
    out += check_models(cfg, root, has_module)
    out += check_protocol_and_rack(cfg, root)
    out += check_binaries(cfg, root, encoders, search_path)
    if network:
        out += check_network(cfg)
    out += check_devices(cfg, has_module, camera)
    out += check_system(root)
    return out


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------

_TAGS = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]", "info": "[INFO]"}


def verdict(checks: list[Check]) -> tuple[str, list[str]]:
    """("NOT READY" | "READY (DEGRADED)" | "READY", what is broken or lost)."""
    if any(c.status == "fail" for c in checks):
        return "NOT READY", [c.name for c in checks if c.status == "fail"]
    lost = [c.effect for c in checks if c.status == "warn" and c.effect]
    return ("READY (DEGRADED)" if lost else "READY"), lost


def render(checks: list[Check]) -> list[str]:
    lines: list[str] = []
    for c in checks:
        lines.append(f"{_TAGS[c.status]} {c.name:22} {c.detail}")
        if c.status != "ok" and c.effect:
            lines.append(f"       loses: {c.effect}")
        if c.status in ("warn", "fail") and c.fix:
            lines.append(f"       fix:   {c.fix}")
    state, lost = verdict(checks)
    lines.append("")
    lines.append(f"verdict: {state}")
    if state == "READY (DEGRADED)":
        lines += [f"  - {item}" for item in dict.fromkeys(lost)]
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", help="override YAML on top of configs/default.yaml")
    ap.add_argument("--override", action="append", default=[], help="key.path=value")
    ap.add_argument("--camera", action="store_true", help="open capture.source and read a frame")
    ap.add_argument("--strict", action="store_true", help="warnings also fail the run")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-network", action="store_true", help="skip port and RTSP-target probes")
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config, overrides=args.override)
        checks = run_checks(cfg, network=not args.no_network, camera=args.camera)
    except Exception as e:  # a config error is reported, not raised
        say(f"doctor: {type(e).__name__}: {e}", err=True)
        return 2

    state, lost = verdict(checks)
    if args.json:
        report = {
            "platform": platform_tag(),
            "verdict": state,
            "lost": lost,
            "checks": [asdict(c) for c in checks],
        }
        say(json.dumps(report, indent=2))
    else:
        say(f"ACTUS doctor ({platform_tag()}, python {sys.version.split()[0]})")
        say()
        for line in render(checks):
            say(line)
    if state == "NOT READY" or (args.strict and state == "READY (DEGRADED)"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
