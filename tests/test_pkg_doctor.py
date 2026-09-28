import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.doctor as doctor  # noqa: E402
from actus.core.config import load_config  # noqa: E402
from tools.doctor import (  # noqa: E402
    Check,
    find_binary,
    main,
    render,
    run_checks,
    verdict,
)

EXE = ".exe" if sys.platform == "win32" else ""
X264 = "V....D libx264   H.264 / AVC"


def _touch(path: Path, data: bytes = b"x", *, executable: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if executable:
        path.chmod(0o755)
    return path


@pytest.fixture
def root(tmp_path):
    """A repo root with a valid protocol and a placeholder rack, and nothing else."""
    r = tmp_path / "repo"
    shutil.copy(REPO_ROOT / "protocols" / "bas_boxes.yaml", _touch(r / "protocols/bas_boxes.yaml"))
    _touch(r / "configs/rack.yaml", b"measured: false\n")
    return r


def _run(root, cfg=None, **kw):
    kw.setdefault("has_module", lambda name: name != "sounddevice")
    kw.setdefault("encoders", lambda ffmpeg: X264)
    kw.setdefault("search_path", "")  # hermetic: whatever is on this machine's PATH is ignored
    return run_checks(cfg or load_config(), root, network=False, **kw)


def _by_name(checks):
    return {c.name: c for c in checks}


def _install_everything(root: Path) -> None:
    for rel in (
        "models/perception/hand_landmarker.task",
        "models/perception/pose_landmarker_lite.task",
        "models/voice/kokoro-v1.0.onnx",
        "models/voice/voices-v1.0.bin",
        "models/hoi/hoi.npz",
        "models/temporal/temporal.npz",
        "yolov8s-worldv2.pt",
        "ui/dist/index.html",
    ):
        _touch(root / rel)
    _touch(root / "models/voice/vosk-model-small-en-us-0.15/am/final.mdl")
    _touch(root / f"bin/mediamtx{EXE}", executable=True)
    _touch(root / f"bin/ffmpeg{EXE}", executable=True)


# --- verdicts ---------------------------------------------------------------------------------


def test_a_bare_checkout_is_degraded_not_broken(root):
    checks = _by_name(_run(root))
    for name in (
        "hand landmarker",
        "pose landmarker",
        "kokoro model",
        "kokoro voices",
        "vosk model",
        "detector weights",
        "GUI build",
        "mediamtx",
        "ffmpeg",
    ):
        assert checks[name].status == "warn", name
        assert checks[name].effect, name  # every warning says what it costs
        assert checks[name].fix, name  # ... and how to fix it
    assert checks["hoi model"].status == "info"
    assert checks["rack geometry"].status == "warn"
    assert checks["protocol"].status == "ok"
    state, lost = verdict(list(checks.values()))
    assert state == "READY (DEGRADED)"
    assert "no hand landmarks" in lost


def test_with_everything_installed_the_file_checks_pass(root):
    _install_everything(root)
    checks = _by_name(_run(root))
    for name in (
        "hand landmarker",
        "pose landmarker",
        "kokoro model",
        "kokoro voices",
        "vosk model",
        "detector weights",
        "hoi model",
        "temporal model",
        "GUI build",
        "mediamtx",
        "ffmpeg",
    ):
        assert checks[name].status == "ok", (name, checks[name])
    assert "libx264" in checks["ffmpeg"].detail


def test_a_missing_core_module_is_a_failure(root):
    checks = _run(root, has_module=lambda name: name != "cv2")
    assert _by_name(checks)["module cv2"].status == "fail"
    assert verdict(checks)[0] == "NOT READY"


def test_a_missing_optional_extra_only_degrades(root):
    checks = _run(root, has_module=lambda name: name not in ("mediapipe", "sounddevice"))
    by_name = _by_name(checks)
    assert by_name["module mediapipe"].status == "warn"
    assert by_name["module mediapipe"].fix == "uv sync --extra perception"
    assert verdict(checks)[0] != "NOT READY"


def test_voice_checks_disappear_when_voice_is_off(root):
    cfg = load_config(overrides=["voice.enabled=false"])
    names = _by_name(_run(root, cfg))
    assert "kokoro model" not in names
    assert "vosk model" not in names
    assert "module vosk" not in names


def test_an_invalid_protocol_is_a_failure(root):
    (root / "protocols" / "bas_boxes.yaml").write_text("id: broken\nsteps: []\n", encoding="utf-8")
    protocol = _by_name(_run(root))["protocol"]
    assert protocol.status == "fail"
    assert verdict([protocol])[0] == "NOT READY"


def test_a_measured_rack_passes(root):
    (root / "configs" / "rack.yaml").write_text("measured: true\n", encoding="utf-8")
    assert _by_name(_run(root))["rack geometry"].status == "ok"


def test_custom_backend_checks_the_custom_weights(root):
    cfg = load_config(overrides=["detector.backend=custom"])
    check = _by_name(_run(root, cfg))["detector weights"]
    assert check.status == "warn"
    assert "best.onnx" in check.detail
    _touch(root / "models/detector/best.onnx")
    assert _by_name(_run(root, cfg))["detector weights"].status == "ok"


def test_the_fix_names_the_asset_that_provides_the_configured_weights(root):
    shutil.copy(REPO_ROOT / "configs" / "assets.yaml", _touch(root / "configs/assets.yaml"))
    cfg = load_config(
        overrides=["detector.backend=custom", "detector.custom_weights=models/coco/yolo11n.pt"]
    )
    check = _by_name(_run(root, cfg))["detector weights"]
    assert check.status == "warn"
    assert check.fix.endswith("fetch_models.py yolo11n_coco")


def test_ffmpeg_without_libx264_is_flagged(root):
    _install_everything(root)
    check = _by_name(_run(root, encoders=lambda ffmpeg: "V..... mpeg4"))["ffmpeg"]
    assert check.status == "warn"
    assert "libx264" in check.detail


def test_streaming_off_skips_the_binaries(root):
    cfg = load_config(overrides=["stream.enabled=false"])
    names = _by_name(_run(root, cfg))
    assert "mediamtx" not in names
    assert names["streaming"].status == "info"


# --- helpers and output -----------------------------------------------------------------------


def test_find_binary_prefers_the_repo_bin(root):
    made = _touch(root / f"bin/ffmpeg{EXE}", executable=True)
    assert Path(find_binary("ffmpeg", root, "")) == made
    assert find_binary("definitely-not-a-real-binary-xyz", root, "") is None


def test_render_is_ascii_and_shows_loss_and_fix(root):
    lines = render(_run(root))
    text = "\n".join(lines)
    assert all(ord(ch) < 128 for ch in text)
    assert "loses:" in text
    assert "fix:" in text
    assert lines[-1].startswith("  - ") or lines[-1].startswith("verdict")


def test_verdict_of_an_all_ok_run_is_ready():
    assert verdict([Check("a", "ok", "fine"), Check("b", "info", "note", "x")]) == ("READY", [])


def test_cli_json_report_has_the_documented_shape(capsys):
    code = main(["--json", "--no-network"])
    assert code in (0, 1)
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {"platform", "verdict", "lost", "checks"}
    assert report["verdict"] in ("READY", "READY (DEGRADED)", "NOT READY")
    assert {"name", "status", "detail", "effect", "fix"} <= set(report["checks"][0])


@pytest.mark.parametrize(
    ("status", "plain", "strict"),
    [("ok", 0, 0), ("warn", 0, 1), ("fail", 1, 1)],
)
def test_cli_exit_status_and_strict(monkeypatch, capsys, status, plain, strict):
    checks = [Check("x", status, "detail", "what is lost")]
    monkeypatch.setattr(doctor, "run_checks", lambda cfg, **kw: checks)
    assert main(["--json", "--no-network"]) == plain
    assert main(["--json", "--no-network", "--strict"]) == strict
    capsys.readouterr()
