import json
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._pkglib import Asset, PkgError, platform_tag  # noqa: E402
from tools.bundle import (  # noqa: E402
    MANIFEST_NAME,
    PAYLOAD,
    collect_files,
    install_bundle,
    is_allowed_key,
    main,
    make_bundle,
    verify_bundle,
)

CLIP = Asset(id="clip", dest="~/.cache/clip/ViT-B-32.pt", url="https://x.invalid/clip")


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _repo(base: Path, name: str = "repo") -> Path:
    """A tiny checkout: weights, a trained model, both media binaries, a built GUI, a lock."""
    root = base / name
    _write(root / "models/perception/hand_landmarker.task", b"hand" * 50)
    _write(root / "models/voice/vosk-x/am/final.mdl", b"vosk" * 50)
    _write(root / "models/detector/card.md", b"# card")
    _write(root / "models/detector/runs/train/weights.pt", b"big" * 50)
    _write(root / "bin/mediamtx", b"#!mtx")
    _write(root / "bin/mediamtx.yml", b"rtspAddress: :8554")
    _write(root / "ui/dist/index.html", b"<html></html>")
    _write(root / "configs/assets.lock.json", b'{"schema_version": 1, "assets": {}}\n')
    empty_manifest = yaml.safe_dump({"schema_version": 1, "assets": []}).encode()
    _write(root / "configs/assets.yaml", empty_manifest)
    return root


def _home(base: Path, name: str = "home") -> Path:
    home = base / name
    _write(home / ".cache/clip/ViT-B-32.pt", b"clip" * 50)
    return home


def _make(tmp_path, **kw):
    root = _repo(tmp_path)
    home = _home(tmp_path)
    out = tmp_path / "dist" / "b.zip"
    manifest = make_bundle(out, root=root, assets=[CLIP], home=home, **kw)
    return root, home, out, manifest


def _rewrite(src: Path, dst: Path, edits: dict) -> None:
    """Copy a zip, passing named members through a bytes -> bytes function."""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename in edits:
                data = edits[info.filename](data)
            zout.writestr(info.filename, data)


def _keys(manifest) -> set:
    return {f["path"] for f in manifest["files"]}


# --- what goes in -----------------------------------------------------------------------------


def test_make_bundles_models_binaries_gui_lock_and_the_clip_encoder(tmp_path):
    _, _, out, manifest = _make(tmp_path)
    keys = _keys(manifest)
    assert {
        "models/perception/hand_landmarker.task",
        "models/voice/vosk-x/am/final.mdl",
        "models/detector/card.md",
        "bin/mediamtx",
        "bin/mediamtx.yml",
        "ui/dist/index.html",
        "configs/assets.lock.json",
        "home/.cache/clip/ViT-B-32.pt",
    } <= keys
    assert manifest["platform"] == platform_tag()
    assert manifest["wheels"]["included"] is False
    sidecar = out.with_name(out.name + ".sha256").read_text(encoding="utf-8")
    assert sidecar.strip().endswith("  b.zip")
    assert verify_bundle(out) == []


def test_excludes_and_no_ui(tmp_path):
    _, _, _, manifest = _make(tmp_path, excludes=["models/detector/runs/*"], include_ui=False)
    keys = _keys(manifest)
    assert "models/detector/runs/train/weights.pt" not in keys
    assert not any(k.startswith("ui/") for k in keys)
    assert "models/detector/card.md" in keys


def test_notes_say_what_an_offline_machine_would_miss(tmp_path):
    root = tmp_path / "bare"
    _write(root / "models/a.bin", b"a")
    needed = Asset(id="kokoro_model", dest="models/voice/kokoro.onnx", url="https://x.invalid/k")
    _, notes = collect_files(root, [needed], home=tmp_path / "nohome")
    text = "\n".join(notes)
    assert "kokoro_model is missing" in text
    assert "ui/dist" in text
    assert "assets.lock.json" in text


def test_partial_downloads_and_caches_are_never_bundled(tmp_path):
    root = _repo(tmp_path)
    _write(root / "models/voice/big.onnx.part", b"half")
    _write(root / "models/perception/__pycache__/x.pyc", b"c")
    files, _ = collect_files(root, [])
    assert not [k for k in files if k.endswith(".part") or "__pycache__" in k]


# --- the path allow-list ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "models/a.bin",
        "bin/mediamtx.exe",
        "ui/dist/assets/index.js",
        "wheels/numpy-2.0-cp311-win_amd64.whl",
        "home/.cache/clip/ViT-B-32.pt",
        "configs/assets.lock.json",
        "yolov8s-worldv2.pt",
    ],
)
def test_allowed_keys(key):
    assert is_allowed_key(key)


@pytest.mark.parametrize(
    "key",
    [
        "",
        "../evil",
        "models/../../evil",
        "/etc/passwd",
        "C:/Windows/x",
        "models\\a.bin",
        "models//a.bin",
        "home/.ssh/authorized_keys",
        "actus/core/config.py",
        "configs/default.yaml",
        "tools/doctor.py",
        "scripts/setup.sh",
        "evil.pt/../x",
        "yolo/x.pt",
    ],
)
def test_refused_keys(key):
    assert not is_allowed_key(key)


def test_a_bundle_naming_a_path_outside_the_allow_list_is_refused(tmp_path):
    evil = tmp_path / "evil.zip"
    entry = {"path": "../evil.txt", "sha256": "0" * 64, "bytes": 1, "mode": 420}
    manifest = {"kind": "actus-asset-bundle", "schema_version": 1, "files": [entry]}
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(manifest))
        zf.writestr(PAYLOAD + "../evil.txt", "x")
    repo = tmp_path / "target"
    repo.mkdir()
    with pytest.raises(PkgError, match="outside the allowed"):
        install_bundle(evil, repo, home=tmp_path / "h")
    assert not (tmp_path / "evil.txt").exists()


# --- install ----------------------------------------------------------------------------------


def test_install_round_trip_is_exact_and_idempotent(tmp_path):
    root, home, out, manifest = _make(tmp_path)
    target, target_home = tmp_path / "offline", tmp_path / "offline-home"
    target.mkdir()
    written = install_bundle(out, target, home=target_home)
    assert set(written) == _keys(manifest)
    for key in _keys(manifest):
        src = home / key[len("home/") :] if key.startswith("home/") else root / key
        dst = target_home / key[len("home/") :] if key.startswith("home/") else target / key
        assert dst.read_bytes() == src.read_bytes(), key
    assert install_bundle(out, target, home=target_home) == []  # nothing left to do


def test_install_replaces_a_stale_file_unless_told_not_to(tmp_path):
    _, _, out, _ = _make(tmp_path)
    target = tmp_path / "offline"
    stale = _write(target / "models/detector/card.md", b"old card")
    install_bundle(out, target, home=tmp_path / "h", overwrite=False)
    assert stale.read_bytes() == b"old card"
    install_bundle(out, target, home=tmp_path / "h")
    assert stale.read_bytes() == b"# card"


def test_dry_run_writes_nothing(tmp_path):
    _, _, out, manifest = _make(tmp_path)
    target = tmp_path / "offline"
    target.mkdir()
    written = install_bundle(out, target, home=tmp_path / "h", dry_run=True)
    assert set(written) == _keys(manifest)
    assert not list(target.rglob("*"))


def test_a_tampered_bundle_installs_nothing(tmp_path):
    _, _, out, _ = _make(tmp_path)
    bad = tmp_path / "bad.zip"
    _rewrite(out, bad, {PAYLOAD + "ui/dist/index.html": lambda b: b.replace(b"html", b"HTML")})

    problems = verify_bundle(bad)
    assert any("ui/dist/index.html" in p and "hash mismatch" in p for p in problems)

    target = tmp_path / "offline"
    target.mkdir()
    with pytest.raises(PkgError, match="hash mismatch"):
        install_bundle(bad, target, home=tmp_path / "h")
    # files that sort before the bad one were staged, then rolled back: none may remain
    assert not (target / "bin" / "mediamtx").exists()
    assert not list(target.rglob("*.part"))


def test_an_unlisted_member_is_a_problem(tmp_path):
    _, _, out, _ = _make(tmp_path)
    extra = tmp_path / "extra.zip"
    _rewrite(out, extra, {})
    with zipfile.ZipFile(extra, "a") as zf:
        zf.writestr(PAYLOAD + "models/smuggled.bin", b"x")
    assert any("unlisted member" in p for p in verify_bundle(extra))


def test_a_bundle_from_another_platform_is_refused_unless_portable_only(tmp_path):
    _, _, out, _ = _make(tmp_path)

    def retarget(data: bytes) -> bytes:
        manifest = json.loads(data)
        manifest["platform"] = "plan9-mips"
        return json.dumps(manifest).encode()

    foreign = tmp_path / "foreign.zip"
    _rewrite(out, foreign, {MANIFEST_NAME: retarget})
    target = tmp_path / "offline"
    target.mkdir()
    with pytest.raises(PkgError, match="plan9-mips"):
        install_bundle(foreign, target, home=tmp_path / "h")
    assert not list(target.rglob("*"))

    written = install_bundle(foreign, target, home=tmp_path / "h", portable_only=True)
    assert "models/detector/card.md" in written
    assert not any(k.startswith(("bin/", "wheels/")) for k in written)
    assert not (target / "bin").exists()


def test_not_a_bundle_is_a_clear_error(tmp_path):
    junk = tmp_path / "junk.zip"
    with zipfile.ZipFile(junk, "w") as zf:
        zf.writestr("hello.txt", "hi")
    with pytest.raises(PkgError, match="not an ACTUS bundle"):
        verify_bundle(junk)


# --- the command line -------------------------------------------------------------------------


def test_cli_make_info_verify_install(tmp_path, capsys):
    root = _repo(tmp_path)
    out = tmp_path / "dist" / "cli.zip"
    assert main(["make", "--root", str(root), "--out", str(out)]) == 0
    assert out.is_file()
    capsys.readouterr()

    assert main(["info", str(out)]) == 0
    assert "actus-asset-bundle" in capsys.readouterr().out
    assert main(["verify", str(out)]) == 0

    target = tmp_path / "offline"
    target.mkdir()
    assert main(["install", str(out), "--repo", str(target)]) == 0
    assert (target / "models" / "detector" / "card.md").read_bytes() == b"# card"


def test_cli_reports_a_bad_bundle_without_a_traceback(tmp_path, capsys):
    missing = tmp_path / "nope.zip"
    assert main(["verify", str(missing)]) == 2
    assert "bundle:" in capsys.readouterr().err
