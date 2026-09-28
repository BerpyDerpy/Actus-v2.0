import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._pkglib import Asset, PkgError, sha256_tree  # noqa: E402
from tools.fetch_models import Context, fetch_one, main, select_assets  # noqa: E402

PAYLOAD = b"model-bytes" * 100
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def mirror(tmp_path):
    """A folder standing in for the internet: `--from` reads files from it by name."""
    folder = tmp_path / "mirror"
    folder.mkdir()
    (folder / "w.bin").write_bytes(PAYLOAD)
    with zipfile.ZipFile(folder / "vosk-tiny.zip", "w") as zf:
        zf.writestr("vosk-tiny/a.txt", "A")
        zf.writestr("vosk-tiny/sub/b.txt", "B")
    return folder


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    return r


def _ctx(root, mirror=None, **kw):
    return Context(root=root, lock=kw.pop("lock", {}), src_dir=mirror, **kw)


def _weights(**kw):
    return Asset(id="w", dest="models/w.bin", url="https://example.invalid/dl/w.bin", **kw)


def _zip_asset(**kw):
    url = "https://example.invalid/dl/vosk-tiny.zip"
    return Asset(id="v", dest="models/voice/vosk-tiny", kind="zip_dir", url=url, **kw)


# --- fetch_one --------------------------------------------------------------------------------


def test_fetch_from_a_mirror_then_skip_when_present(root, mirror):
    ctx = _ctx(root, mirror)
    first = fetch_one(_weights(), ctx)
    assert first.status == "downloaded"
    assert first.detail == "unpinned"  # no manifest pin, no lock entry: trusted, and said so
    assert (root / "models" / "w.bin").read_bytes() == PAYLOAD
    assert fetch_one(_weights(), ctx).status == "present"


def test_a_wrong_pin_fails_and_leaves_nothing_behind(root, mirror):
    result = fetch_one(_weights(sha256="0" * 64), _ctx(root, mirror))
    assert result.status == "failed"
    assert "mismatch" in result.detail
    assert not (root / "models" / "w.bin").exists()
    assert not list(root.rglob("*.part"))


def test_the_right_pin_is_accepted(root, mirror):
    result = fetch_one(_weights(sha256=PAYLOAD_SHA), _ctx(root, mirror))
    assert (result.status, result.detail) == ("downloaded", "ok")


def test_a_corrupt_file_on_disk_is_replaced_when_the_lock_disagrees(root, mirror):
    dest = root / "models" / "w.bin"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"corrupt")
    ctx = _ctx(root, mirror, lock={"w": {"sha256": PAYLOAD_SHA}})
    result = fetch_one(_weights(), ctx)
    assert result.status == "downloaded"
    assert dest.read_bytes() == PAYLOAD


def test_force_refetches_a_present_file(root, mirror):
    dest = root / "models" / "w.bin"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"old")
    assert fetch_one(_weights(), _ctx(root, mirror)).status == "present"
    assert fetch_one(_weights(), _ctx(root, mirror, force=True)).status == "downloaded"
    assert dest.read_bytes() == PAYLOAD


def test_zip_dir_unpacks_into_the_named_directory(root, mirror):
    result = fetch_one(_zip_asset(), _ctx(root, mirror))
    assert result.status == "downloaded"
    dest = root / "models" / "voice" / "vosk-tiny"
    assert (dest / "sub" / "b.txt").read_text() == "B"
    assert not list((root / "models" / "voice").glob("actus-fetch-*"))  # temp dir cleaned up


def test_zip_dir_checks_the_tree_hash_pin(root, mirror):
    good = _zip_asset()
    fetch_one(good, _ctx(root, mirror))
    tree = sha256_tree(root / "models" / "voice" / "vosk-tiny")

    other = root / "other"
    assert fetch_one(_zip_asset(sha256=tree), _ctx(other, mirror)).detail == "ok"

    third = root / "third"
    bad = fetch_one(_zip_asset(sha256="1" * 64), _ctx(third, mirror))
    assert bad.status == "failed"
    assert not (third / "models" / "voice" / "vosk-tiny").exists()


def test_a_local_asset_is_reported_not_fetched(root, mirror):
    asset = Asset(id="m", dest="models/m.npz", kind="local", produced_by="T-HOI-TEMP")
    result = fetch_one(asset, _ctx(root, mirror))
    assert result.status == "manual"
    assert "T-HOI-TEMP" in result.detail


def test_a_file_missing_from_the_mirror_is_a_clear_failure(root, mirror):
    asset = Asset(id="n", dest="models/n.bin", url="https://example.invalid/dl/not-there.bin")
    result = fetch_one(asset, _ctx(root, mirror))
    assert result.status == "failed"
    assert "--from" in result.detail


def test_dry_run_downloads_nothing(root, mirror):
    result = fetch_one(_weights(size_mb=12), _ctx(root, mirror, dry_run=True))
    assert result.status == "would-fetch"
    assert "12" in result.detail
    assert not (root / "models").exists()


# --- selection --------------------------------------------------------------------------------


def _catalogue():
    return [
        Asset(id="a", dest="a", url="u", group="perception"),
        Asset(id="b", dest="b", url="u", group="perception", optional=True),
        Asset(id="c", dest="c", url="u", group="voice"),
        Asset(id="d", dest="d", kind="local", group="trained"),
        Asset(id="mediamtx-linux-amd64", dest="e", group="media", resolver="mediamtx"),
    ]


def _ids(assets):
    return [a.id for a in assets]


def test_default_selection_skips_optional_and_other_groups():
    groups = ["perception", "voice"]
    got = select_assets(_catalogue(), ids=[], groups=groups, include_optional=False)
    assert _ids(got) == ["a", "c"]


def test_optional_assets_come_with_all():
    got = select_assets(_catalogue(), ids=[], groups=["perception"], include_optional=True)
    assert _ids(got) == ["a", "b"]


def test_named_ids_ignore_groups_and_prefix_match_media():
    got = select_assets(_catalogue(), ids=["b", "mediamtx"], groups=[], include_optional=False)
    assert _ids(got) == ["b", "mediamtx-linux-amd64"]


def test_an_unknown_id_is_an_error():
    with pytest.raises(PkgError, match="unknown asset"):
        select_assets(_catalogue(), ids=["nope"], groups=[], include_optional=False)


# --- the command line -------------------------------------------------------------------------


@pytest.fixture
def repo_with_manifest(root):
    """A repo root whose manifest has one downloadable and one local asset, in group `test`."""
    (root / "configs").mkdir()
    body = {
        "schema_version": 1,
        "assets": [
            {"id": "w", "group": "test", "dest": "models/w.bin", "url": "https://x.invalid/w.bin"},
            {"id": "m", "group": "test", "kind": "local", "dest": "models/m.npz"},
        ],
    }
    (root / "configs" / "assets.yaml").write_text(yaml.safe_dump(body), encoding="utf-8")
    return root


def _cli(root, *args):
    return main(["--root", str(root), "--group", "test", *args])


def test_cli_fetch_from_a_mirror_lock_and_verify(repo_with_manifest, mirror):
    root = repo_with_manifest
    assert _cli(root, "--from", str(mirror), "w") == 0
    assert (root / "models" / "w.bin").read_bytes() == PAYLOAD

    assert _cli(root, "--lock") == 0
    lock = json.loads((root / "configs" / "assets.lock.json").read_text(encoding="utf-8"))
    assert lock["assets"]["w"]["sha256"] == PAYLOAD_SHA
    assert "m" not in lock["assets"]  # trained models change on retrain and are never pinned

    assert _cli(root, "--verify") == 0
    (root / "models" / "w.bin").write_bytes(b"tampered")
    assert _cli(root, "--verify") == 1


def test_cli_verify_fails_when_a_required_asset_is_missing(repo_with_manifest):
    assert _cli(repo_with_manifest, "--verify") == 1


def test_cli_list_shows_every_selected_asset(repo_with_manifest, capsys):
    assert _cli(repo_with_manifest, "--list") == 0
    out = capsys.readouterr().out
    assert "w" in out.split()
    assert "models/m.npz" in out
    assert "missing" in out


def test_cli_dry_run_and_unknown_id(repo_with_manifest, capsys):
    assert _cli(repo_with_manifest, "--dry-run", "w") == 0
    assert "would-fetch" in capsys.readouterr().out
    assert _cli(repo_with_manifest, "nope") == 2


# --- review fixes (T-REVIEW) --------------------------------------------------------------------


def test_a_bad_archive_fails_that_asset_only(root, tmp_path):
    folder = tmp_path / "bad-mirror"
    folder.mkdir()
    (folder / "vosk-tiny.zip").write_bytes(b"<html>503 Service Unavailable</html>")
    result = fetch_one(_zip_asset(), _ctx(root, folder))
    assert result.status == "failed"
    assert "BadZipFile" in result.detail


def test_find_local_archive_picks_the_newest_archive_not_a_checksum(tmp_path):
    from tools.fetch_models import _find_local_archive

    for name in (
        "mediamtx_v1.9.3_linux_amd64.tar.gz",
        "mediamtx_v1.11.3_linux_amd64.tar.gz",
        "mediamtx_v1.11.3_linux_amd64.tar.gz.sha256sum",
    ):
        (tmp_path / name).write_bytes(b"x")
    got = _find_local_archive(tmp_path, ["mediamtx_*_linux_amd64.*"])
    assert got.name == "mediamtx_v1.11.3_linux_amd64.tar.gz"


def test_lock_does_not_repin_a_file_that_differs_from_its_pin(root, mirror, tmp_path):
    from tools.fetch_models import cmd_lock

    dest = root / "models" / "w.bin"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"corrupted")
    lock = {"w": {"sha256": PAYLOAD_SHA, "bytes": len(PAYLOAD), "dest": "models/w.bin"}}
    lock_path = tmp_path / "assets.lock.json"
    assert cmd_lock([_weights()], _ctx(root, lock=lock), lock_path) == 0
    saved = json.loads(lock_path.read_text(encoding="utf-8"))["assets"]
    assert saved["w"]["sha256"] == PAYLOAD_SHA  # the good pin survives
    cmd_lock([_weights()], _ctx(root, lock=lock, force=True), lock_path)
    saved = json.loads(lock_path.read_text(encoding="utf-8"))["assets"]
    assert saved["w"]["sha256"] == hashlib.sha256(b"corrupted").hexdigest()
