import hashlib
import io
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._pkglib import (  # noqa: E402
    KINDS,
    MEDIAMTX_DOWNLOAD,
    Asset,
    DownloadError,
    PkgError,
    check_asset,
    download,
    expected_hash,
    extract_selected,
    extract_zip,
    ffmpeg_download,
    load_lock,
    load_manifest,
    media_assets,
    mediamtx_pinned_url,
    pick_ffmpeg_member,
    pick_mediamtx_asset,
    pick_mediamtx_member,
    rel_key,
    resolve_dest,
    save_lock,
    sha256_tree,
)


def _write(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _manifest(tmp_path: Path, body: dict) -> Path:
    p = tmp_path / "assets.yaml"
    p.write_text(yaml.safe_dump(body), encoding="utf-8")
    return p


# --- the shipped manifest ---------------------------------------------------------------------


def test_the_shipped_manifest_is_valid():
    assets = load_manifest()
    ids = [a.id for a in assets]
    assert len(ids) == len(set(ids))
    for needed in (
        "hand_landmarker",
        "pose_landmarker_lite",
        "yolo_world_s",
        "clip_vit_b32",
        "kokoro_model",
        "kokoro_voices",
        "vosk_small_en",
    ):
        assert needed in ids
    assert {a.kind for a in assets} <= set(KINDS)
    for a in assets:
        if a.kind != "local":
            assert a.url and a.url.startswith("https://"), a.id


def test_manifest_paths_match_the_default_config():
    """The manifest and configs/default.yaml name the same files; a drift here means a fresh
    setup would fetch a file the pipeline never reads."""
    from actus.core.config import load_config

    cfg = load_config()
    by_id = {a.id: a for a in load_manifest()}
    assert by_id["hand_landmarker"].dest == cfg.perception.hand_model_path
    assert by_id["kokoro_model"].dest == cfg.voice.tts.kokoro_model
    assert by_id["kokoro_voices"].dest == cfg.voice.tts.kokoro_voices
    assert by_id["vosk_small_en"].dest == cfg.voice.asr.model
    assert by_id["yolo_world_s"].dest == cfg.detector.open_vocab_weights
    assert by_id["detector_custom"].dest == cfg.detector.custom_weights
    assert by_id["hoi_model"].dest == cfg.models.hoi + ".npz"
    assert by_id["temporal_model"].dest == cfg.models.temporal + ".npz"
    pose = cfg.perception.pose_model
    assert by_id[f"pose_landmarker_{pose}"].dest == f"models/perception/pose_landmarker_{pose}.task"


def test_the_second_protocol_overlay_points_at_a_manifest_asset():
    from actus.core.config import load_config

    cfg = load_config("configs/sample_mixing.yaml")
    by_id = {a.id: a for a in load_manifest()}
    assert by_id["yolo11n_coco"].dest == cfg.detector.custom_weights
    assert by_id["yolo11n_coco"].group == "coco"


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"id": "a", "dest": "x", "url": "https://e/x", "bogus": 1}, "unknown keys"),
        ({"id": "a", "dest": "x"}, "needs a url"),
        ({"id": "a", "dest": "x", "url": "u", "kind": "weird"}, "kind must be"),
        ({"id": "a", "dest": "x", "url": "u", "sha256": "abc"}, "64 hex"),
        ({"dest": "x", "url": "u"}, "missing 'id'"),
    ],
)
def test_manifest_rejects_bad_entries(tmp_path, entry, message):
    path = _manifest(tmp_path, {"schema_version": 1, "assets": [entry]})
    with pytest.raises(PkgError, match=message):
        load_manifest(path)


def test_manifest_rejects_duplicate_ids_and_wrong_schema(tmp_path):
    dup = {"id": "a", "dest": "x", "url": "https://e/x"}
    with pytest.raises(PkgError, match="duplicate"):
        load_manifest(_manifest(tmp_path, {"schema_version": 1, "assets": [dup, dup]}))
    with pytest.raises(PkgError, match="schema_version"):
        load_manifest(_manifest(tmp_path, {"schema_version": 2, "assets": []}))


def test_local_assets_need_no_url(tmp_path):
    body = {"schema_version": 1, "assets": [{"id": "m", "dest": "models/m.npz", "kind": "local"}]}
    (asset,) = load_manifest(_manifest(tmp_path, body))
    assert asset.kind == "local"
    assert asset.url is None


# --- paths and keys ---------------------------------------------------------------------------


def test_dest_and_bundle_keys(tmp_path):
    assert resolve_dest("models/a.bin", tmp_path) == tmp_path / "models" / "a.bin"
    assert resolve_dest("~/x/y", tmp_path) == Path("~/x/y").expanduser()
    assert rel_key("~/.cache/clip/ViT-B-32.pt") == "home/.cache/clip/ViT-B-32.pt"
    assert rel_key("models\\a.bin") == "models/a.bin"


# --- hashing and pins -------------------------------------------------------------------------


def test_tree_hash_depends_on_names_and_content(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        _write(root / "f.txt", b"1")
        _write(root / "sub" / "g.txt", b"2")
    assert sha256_tree(a) == sha256_tree(b)
    _write(b / "sub" / "g.txt", b"3")
    assert sha256_tree(a) != sha256_tree(b)
    _write(b / "sub" / "g.txt", b"2")
    (b / "sub" / "g.txt").rename(b / "sub" / "h.txt")
    assert sha256_tree(a) != sha256_tree(b)


def test_tree_hash_ignores_partial_downloads(tmp_path):
    root = tmp_path / "t"
    _write(root / "f.txt", b"1")
    before = sha256_tree(root)
    _write(root / "f.txt.part", b"junk")
    assert sha256_tree(root) == before


def test_pin_precedence_is_manifest_then_url_then_lock(tmp_path):
    url_hash, lock_hash, manifest_hash = "a" * 64, "b" * 64, "c" * 64
    lock = {"x": {"sha256": lock_hash}}
    from_url = Asset(id="x", dest="d", url=f"https://h/{url_hash}/f.bin", sha256_from_url=True)
    assert expected_hash(from_url, lock) == (url_hash, "url")
    from_url.sha256 = manifest_hash.upper()
    assert expected_hash(from_url, lock) == (manifest_hash, "manifest")
    plain = Asset(id="x", dest="d", url="https://h/f.bin")
    assert expected_hash(plain, lock) == (lock_hash, "lock")
    assert expected_hash(plain, {}) == (None, "none")


def test_the_url_hash_is_only_used_when_it_looks_like_a_hash():
    a = Asset(id="x", dest="d", url="https://h/models/f.bin", sha256_from_url=True)
    assert expected_hash(a, {}) == (None, "none")


def test_check_asset_verdicts(tmp_path):
    a = Asset(id="w", dest="models/w.bin", url="https://h/w.bin")
    assert check_asset(a, {}, tmp_path).verdict == "missing"
    _write(tmp_path / "models" / "w.bin", b"data")
    assert check_asset(a, {}, tmp_path).verdict == "unpinned"
    good = {"w": {"sha256": hashlib.sha256(b"data").hexdigest()}}
    assert check_asset(a, good, tmp_path).verdict == "ok"
    assert check_asset(a, {"w": {"sha256": "0" * 64}}, tmp_path).verdict == "mismatch"
    assert check_asset(a, good, tmp_path, hash_it=False).verdict == "present"


def test_lock_file_round_trips_sorted(tmp_path):
    path = tmp_path / "assets.lock.json"
    assert load_lock(path) == {}
    save_lock({"b": {"sha256": "2"}, "a": {"sha256": "1"}}, path)
    assert list(load_lock(path)) == ["a", "b"]
    assert path.read_text(encoding="utf-8").endswith("\n")


# --- downloading ------------------------------------------------------------------------------


def test_download_is_atomic_and_returns_the_hash(tmp_path):
    src = _write(tmp_path / "src.bin", b"hello world" * 1000)
    dest = tmp_path / "out" / "f.bin"
    digest = download(src.as_uri(), dest)
    assert digest == hashlib.sha256(src.read_bytes()).hexdigest()
    assert dest.read_bytes() == src.read_bytes()
    assert not dest.with_name("f.bin.part").exists()


def test_download_discards_a_file_that_fails_its_pin(tmp_path):
    src = _write(tmp_path / "src.bin", b"payload")
    dest = tmp_path / "out" / "f.bin"
    with pytest.raises(PkgError, match="mismatch"):
        download(src.as_uri(), dest, expected_sha256="0" * 64)
    assert not dest.exists()
    assert not dest.with_name("f.bin.part").exists()


def test_download_accepts_the_right_pin(tmp_path):
    src = _write(tmp_path / "src.bin", b"payload")
    dest = tmp_path / "f.bin"
    download(src.as_uri(), dest, expected_sha256=hashlib.sha256(b"payload").hexdigest().upper())
    assert dest.read_bytes() == b"payload"


def test_download_of_a_missing_source_is_a_download_error(tmp_path):
    with pytest.raises(DownloadError):
        download((tmp_path / "nope.bin").as_uri(), tmp_path / "f.bin", retries=1)
    assert not (tmp_path / "f.bin").exists()


# --- archives ---------------------------------------------------------------------------------


def test_extract_zip_refuses_paths_outside_the_target(tmp_path):
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with pytest.raises(PkgError, match="unsafe"):
        extract_zip(evil, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_extract_zip_unpacks_nested_files(tmp_path):
    z = tmp_path / "m.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("model/a.txt", "A")
        zf.writestr("model/sub/b.txt", "B")
    extract_zip(z, tmp_path / "out")
    assert (tmp_path / "out" / "model" / "sub" / "b.txt").read_text() == "B"


def test_pickers_choose_only_the_binaries():
    assert pick_mediamtx_member("mediamtx") == "mediamtx"
    assert pick_mediamtx_member("./mediamtx.yml") == "mediamtx.yml"
    assert pick_mediamtx_member("mediamtx.exe") == "mediamtx.exe"
    assert pick_mediamtx_member("LICENSE") is None
    assert pick_mediamtx_member("docs/mediamtx") is None
    assert pick_ffmpeg_member("ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe") == "ffmpeg.exe"
    assert pick_ffmpeg_member("ffmpeg-x/bin/ffprobe") == "ffprobe"
    assert pick_ffmpeg_member("ffmpeg-x/doc/ffmpeg.html") is None
    assert pick_ffmpeg_member("ffmpeg-x/bin/ffplay.exe") is None


def test_extract_selected_from_zip_and_tar(tmp_path):
    z = tmp_path / "ff.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("ffmpeg-x/bin/ffmpeg.exe", "EXE")
        zf.writestr("ffmpeg-x/bin/ffprobe.exe", "PROBE")
        zf.writestr("ffmpeg-x/doc/readme.txt", "DOC")
    got = extract_selected(z, pick_ffmpeg_member, tmp_path / "bin1")
    assert sorted(p.name for p in got) == ["ffmpeg.exe", "ffprobe.exe"]
    assert (tmp_path / "bin1" / "ffmpeg.exe").read_text() == "EXE"

    t = tmp_path / "mtx.tar.gz"
    with tarfile.open(t, "w:gz") as tf:
        for name, data in (("mediamtx", b"BIN"), ("mediamtx.yml", b"CFG"), ("LICENSE", b"L")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    got = extract_selected(t, pick_mediamtx_member, tmp_path / "bin2")
    assert sorted(p.name for p in got) == ["mediamtx", "mediamtx.yml"]
    assert (tmp_path / "bin2" / "mediamtx").read_bytes() == b"BIN"


# --- platform pickers -------------------------------------------------------------------------


def test_pick_mediamtx_asset_handles_linux_arm64v8():
    def item(name):
        return {"name": name, "browser_download_url": f"https://x/{name}"}

    release = {
        "assets": [
            item("mediamtx_v1.9.3_linux_arm64v8.tar.gz"),
            item("mediamtx_v1.9.3_linux_amd64.tar.gz"),
            item("mediamtx_v1.9.3_windows_amd64.zip"),
            item("mediamtx_v1.9.3_darwin_arm64.tar.gz"),
            item("checksums.sha256"),
        ]
    }
    name, url = pick_mediamtx_asset(release, "linux", "arm64")
    assert name == "mediamtx_v1.9.3_linux_arm64v8.tar.gz"
    assert url == "https://x/" + name
    assert pick_mediamtx_asset(release, "windows", "amd64")[0].endswith("windows_amd64.zip")
    assert pick_mediamtx_asset(release, "darwin", "arm64")[0].endswith("darwin_arm64.tar.gz")
    with pytest.raises(PkgError, match="no MediaMTX"):
        pick_mediamtx_asset(release, "linux", "riscv64")


def test_mediamtx_pinned_url():
    name, url = mediamtx_pinned_url("v1.9.3", "windows", "amd64")
    assert name == "mediamtx_v1.9.3_windows_amd64.zip"
    assert url == f"{MEDIAMTX_DOWNLOAD}/v1.9.3/{name}"
    assert mediamtx_pinned_url("v1.9.3", "linux", "arm64")[0].endswith("linux_arm64v8.tar.gz")


def test_ffmpeg_download_leaves_macos_to_the_package_manager():
    assert ffmpeg_download("darwin", "arm64") is None
    win = ffmpeg_download("windows", "amd64")
    assert win is not None
    assert win[0].endswith("win64-gpl.zip")
    linux = ffmpeg_download("linux", "arm64")
    assert linux is not None
    assert linux[0].endswith("linuxarm64-gpl.tar.xz")


def test_media_assets_name_the_platform_and_the_exe_suffix():
    win = {a.resolver: a for a in media_assets("windows", "amd64")}
    assert win["mediamtx"].dest == "bin/mediamtx.exe"
    assert win["ffmpeg"].id == "ffmpeg-windows-amd64"
    lin = {a.resolver: a for a in media_assets("linux", "arm64")}
    assert lin["mediamtx"].dest == "bin/mediamtx"
    assert all(a.group == "media" for a in lin.values())
