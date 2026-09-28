"""Fetch, verify and pin the runtime assets: models, MediaMTX and FFmpeg (T-PKG).

    uv run python tools/fetch_models.py                 # everything in the default set
    uv run python tools/fetch_models.py --list          # what is there, missing, pinned
    uv run python tools/fetch_models.py --all           # plus the optional assets (pose full/heavy)
    uv run python tools/fetch_models.py kokoro_model    # just this asset (ids from --list)
    uv run python tools/fetch_models.py --group voice   # a group: perception voice open_vocab media
    uv run python tools/fetch_models.py --verify        # re-hash what is on disk; no network
    uv run python tools/fetch_models.py --lock          # pin what is on disk (assets.lock.json)
    uv run python tools/fetch_models.py --warm          # load the open-vocab detector once, online
    uv run python tools/fetch_models.py --from D:/mirror    # read the files from a folder
    uv run python tools/fetch_models.py --dry-run       # print what would be downloaded

The manifest is ``configs/assets.yaml``. Setup is the only time the network is used: nothing here
runs at run time, and everything it fetches lands where ``configs/default.yaml`` already looks.

Trust model. A download is verified against the manifest's ``sha256``, else a hash embedded in
its URL, else ``configs/assets.lock.json``. With none of those the first download is trusted and
reported as "unpinned"; ``--lock`` then records its hash so that every later fetch, and
``tools/bundle.py``, can detect a changed or corrupted file. Commit the lock file to share the
pins with the team. The MediaMTX and FFmpeg binaries come from "latest" releases, so a newer
release than the lock's is reported, not rejected.

``--warm`` exists because the open-vocabulary detector (YOLO-World) needs OpenAI's CLIP text
encoder and the ``clip`` Python package, and ultralytics installs the package with
``pip install git+https://github.com/ultralytics/CLIP.git`` and downloads the weights the first
time ``set_classes`` runs. That is a hidden run-time download, so ``--warm`` triggers it now.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._pkglib import (  # noqa: E402
    MEDIAMTX_API,
    Asset,
    DownloadError,
    PkgError,
    check_asset,
    detect_platform,
    download,
    expected_hash,
    extract_selected,
    extract_zip,
    fetch_json,
    ffmpeg_download,
    format_mb,
    hash_path,
    load_lock,
    load_manifest,
    make_executable,
    media_assets,
    mediamtx_arch_candidates,
    mediamtx_pinned_url,
    pick_ffmpeg_member,
    pick_mediamtx_asset,
    pick_mediamtx_member,
    save_lock,
    say,
    sha256_tree,
    size_of,
)

DEFAULT_GROUPS = ("perception", "voice", "open_vocab", "coco", "media")
CLIP_GIT = "git+https://github.com/ultralytics/CLIP.git"


@dataclass
class Context:
    root: Path
    lock: dict[str, dict[str, Any]]
    src_dir: Path | None = None
    force: bool = False
    dry_run: bool = False
    mediamtx_version: str | None = None
    opener: Any = None  # urllib opener injected by tests


@dataclass
class FetchResult:
    asset_id: str
    status: str  # downloaded | present | manual | failed | would-fetch
    detail: str = ""


# --------------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------------


def all_assets(root: Path = REPO_ROOT) -> list[Asset]:
    """The manifest plus the MediaMTX and FFmpeg binaries for this platform."""
    os_name, arch = detect_platform()
    return [*load_manifest(root / "configs" / "assets.yaml"), *media_assets(os_name, arch)]


def select_assets(
    assets: list[Asset], *, ids: list[str], groups: list[str], include_optional: bool
) -> list[Asset]:
    if ids:
        picked: list[Asset] = []
        for name in ids:
            # "mediamtx" names "mediamtx-<os>-<arch>", so the same command works everywhere
            hits = [a for a in assets if a.id == name or a.id.startswith(name + "-")]
            if not hits:
                raise PkgError(f"unknown asset {name!r}; run with --list to see the ids")
            picked.extend(h for h in hits if h not in picked)
        return picked
    return [a for a in assets if a.group in groups and (include_optional or not a.optional)]


# --------------------------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------------------------


def _source_url(url: str, ctx: Context) -> str:
    """The URL to read: the asset's own, or its file name inside the ``--from`` folder."""
    if ctx.src_dir is None:
        return url
    local = ctx.src_dir / PurePosixPath(urlparse(url).path).name
    if not local.is_file():
        raise PkgError(f"{local.name} is not in {ctx.src_dir} (--from)")
    return local.as_uri()


def _ultralytics_download(dest: Path) -> None:
    try:
        from ultralytics.utils.downloads import attempt_download_asset
    except ImportError as e:
        raise PkgError("ultralytics is not installed (uv sync --extra perception)") from e
    dest.parent.mkdir(parents=True, exist_ok=True)
    attempt_download_asset(str(dest))
    if not dest.is_file():
        raise PkgError(f"ultralytics did not produce {dest}")


def _fetch_file(asset: Asset, ctx: Context) -> None:
    dest = asset.dest_path(ctx.root)
    expected, _ = expected_hash(asset, ctx.lock)
    try:
        download(
            _source_url(asset.url or "", ctx),
            dest,
            expected_sha256=expected,
            label=asset.id,
            opener=ctx.opener,
        )
    except DownloadError:
        if asset.fallback != "ultralytics" or ctx.src_dir is not None:
            raise
        say(f"  {asset.id}: direct download failed; asking ultralytics to fetch it")
        _ultralytics_download(dest)
        # The fallback bypasses download()'s hash check: apply the pin here.
        if expected and hash_path(dest) != expected:
            dest.unlink(missing_ok=True)
            raise PkgError(f"{asset.id}: the ultralytics download does not match its pin") from None


def _fetch_zip_dir(asset: Asset, ctx: Context) -> None:
    dest = asset.dest_path(ctx.root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected, _ = expected_hash(asset, ctx.lock)
    with tempfile.TemporaryDirectory(prefix="actus-fetch-", dir=dest.parent) as tmp:
        archive = Path(tmp) / "archive.zip"
        download(_source_url(asset.url or "", ctx), archive, label=asset.id, opener=ctx.opener)
        stage = Path(tmp) / "unpacked"
        extract_zip(archive, stage)
        inner = stage / dest.name
        if not inner.is_dir():
            entries = list(stage.iterdir())
            if len(entries) != 1 or not entries[0].is_dir():
                raise PkgError(f"{asset.id}: the archive has no {dest.name}/ folder")
            inner = entries[0]
        if expected and sha256_tree(inner) != expected:
            raise PkgError(f"{asset.id}: unpacked tree does not match its pinned hash")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(inner), str(dest))


_ARCHIVE_SUFFIXES = (".zip", ".tar.gz", ".tgz", ".tar.xz", ".tar.bz2")


def _version_key(path: Path) -> list:
    """Natural sort: v1.11.3 after v1.9.3 (plain string order gets that backwards)."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def _find_local_archive(src_dir: Path, patterns: list[str]) -> Path:
    for pattern in patterns:
        # Only archives: a mirror may also hold .sha256sum files that match the same pattern.
        hits = [p for p in src_dir.glob(pattern) if p.name.endswith(_ARCHIVE_SUFFIXES)]
        if hits:
            return max(hits, key=_version_key)
    raise PkgError(f"no archive matching {patterns} in {src_dir} (--from)")


def _fetch_mediamtx(asset: Asset, ctx: Context) -> None:
    os_name, arch = detect_platform()
    with tempfile.TemporaryDirectory(prefix="actus-fetch-") as tmp:
        if ctx.src_dir is not None:
            alts = mediamtx_arch_candidates(os_name, arch)
            archive = _find_local_archive(
                ctx.src_dir, [f"mediamtx_*_{os_name}_{alt}.*" for alt in alts]
            )
        else:
            if ctx.mediamtx_version:
                name, url = mediamtx_pinned_url(ctx.mediamtx_version, os_name, arch)
            else:
                release = fetch_json(MEDIAMTX_API, opener=ctx.opener)
                name, url = pick_mediamtx_asset(release, os_name, arch)
            archive = Path(tmp) / name
            say(f"  {asset.id}: {name}")
            download(url, archive, label=asset.id, opener=ctx.opener)
        written = extract_selected(archive, pick_mediamtx_member, ctx.root / "bin")
    binaries = [p for p in written if p.suffix != ".yml"]
    if not binaries:
        raise PkgError(f"{asset.id}: no mediamtx binary inside {archive.name}")
    for p in binaries:
        make_executable(p)


def _brew_install_ffmpeg() -> None:
    brew = shutil.which("brew")
    if not brew:
        raise PkgError("ffmpeg: Homebrew not found (https://brew.sh), then: brew install ffmpeg")
    say("  ffmpeg: brew install ffmpeg")
    if subprocess.run([brew, "install", "ffmpeg"], check=False).returncode != 0:
        raise PkgError("ffmpeg: `brew install ffmpeg` failed")


def _fetch_ffmpeg(asset: Asset, ctx: Context) -> None:
    os_name, arch = detect_platform()
    found = ffmpeg_download(os_name, arch)
    if found is None:
        if os_name == "darwin":
            _brew_install_ffmpeg()
            return
        raise PkgError(f"no FFmpeg build for {os_name}/{arch}; use your package manager")
    name, url = found
    with tempfile.TemporaryDirectory(prefix="actus-fetch-") as tmp:
        if ctx.src_dir is not None:
            archive = _find_local_archive(ctx.src_dir, [name])
        else:
            archive = Path(tmp) / name
            download(url, archive, label=asset.id, opener=ctx.opener)
        written = extract_selected(archive, pick_ffmpeg_member, ctx.root / "bin")
    if not any(p.stem == "ffmpeg" for p in written):
        raise PkgError(f"{asset.id}: no ffmpeg binary inside {archive.name}")
    for p in written:
        make_executable(p)


def fetch_one(asset: Asset, ctx: Context) -> FetchResult:
    state = check_asset(asset, ctx.lock, ctx.root)
    if state.present and not ctx.force:
        # A pinned mismatch is re-fetched, except for the "latest"-release binaries, where a
        # newer release than the lock's is expected and only worth a note.
        if state.verdict != "mismatch" or asset.resolver:
            detail = state.verdict + (" (on PATH)" if state.via_path else "")
            if state.verdict == "mismatch":
                detail += " (lock is stale: a newer release? run --lock)"
            return FetchResult(asset.id, "present", detail)
        say(f"  {asset.id}: on disk but does not match its {state.pin_source} pin; fetching again")
    if asset.kind == "local":
        return FetchResult(asset.id, "manual", f"not downloadable; made by {asset.produced_by}")
    if ctx.dry_run:
        size = f"~{asset.size_mb:g} MB" if asset.size_mb else "size unknown"
        return FetchResult(asset.id, "would-fetch", f"{size} -> {asset.dest}")
    try:
        if asset.resolver == "mediamtx":
            _fetch_mediamtx(asset, ctx)
        elif asset.resolver == "ffmpeg":
            _fetch_ffmpeg(asset, ctx)
        elif asset.kind == "zip_dir":
            _fetch_zip_dir(asset, ctx)
        else:
            _fetch_file(asset, ctx)
    except PkgError as e:
        return FetchResult(asset.id, "failed", str(e))
    except (OSError, zipfile.BadZipFile, tarfile.TarError, EOFError) as e:
        # A bad archive (an HTML error page served as a zip), a locked or running binary, a full
        # disk: this asset failed, the others still get fetched.
        return FetchResult(asset.id, "failed", f"{type(e).__name__}: {e}")
    after = check_asset(asset, ctx.lock, ctx.root)
    if after.verdict == "mismatch" and not asset.resolver:
        detail = f"downloaded, but differs from its {after.pin_source} pin"
        return FetchResult(asset.id, "failed", detail)
    return FetchResult(asset.id, "downloaded", after.verdict)


# --------------------------------------------------------------------------------------------
# --warm: make the open-vocabulary detector's hidden run-time downloads happen now
# --------------------------------------------------------------------------------------------


def ensure_clip_package() -> bool:
    """Install the ``clip`` package ultralytics needs for YOLO-World, if it is missing."""
    if importlib.util.find_spec("clip") is not None:
        return True
    if shutil.which("git") is None:
        say("  the clip package installs from git, and git is not on PATH", err=True)
        return False
    uv = shutil.which("uv")
    cmd = (
        [uv, "pip", "install", "--python", sys.executable, CLIP_GIT]
        if uv
        else [sys.executable, "-m", "pip", "install", CLIP_GIT]
    )
    say("  installing the clip package: " + " ".join(cmd[-1:]))
    return subprocess.run(cmd, check=False).returncode == 0


def warm_open_vocab() -> bool:
    """Load YOLO-World and call ``set_classes`` exactly as ``OpenVocabDetector`` does."""
    from actus.core.config import load_config, resolve_path

    cfg = load_config()
    weights = resolve_path(cfg.detector.open_vocab_weights)
    if not weights.is_file():
        say(f"  {weights} is missing; run tools/fetch_models.py first", err=True)
        return False
    if not ensure_clip_package():
        say("  YOLO-World will try to install the clip package when it first runs", err=True)
        return False
    try:
        from ultralytics import YOLO

        model = YOLO(str(weights))
        model.set_classes(list(dict(cfg.detector.prompts).values()))
    except Exception as e:  # report whatever ultralytics or CLIP raised, not just our own errors
        say(f"  open-vocab warm-up failed: {type(e).__name__}: {e}", err=True)
        return False
    say("  open-vocab detector loads; its CLIP text encoder is cached")
    return True


# --------------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------------


def _state_word(state: Any) -> str:
    if not state.present:
        return "missing"
    if state.via_path:
        return "on PATH"
    return "ok" if state.verdict == "present" else state.verdict


def cmd_list(assets: list[Asset], ctx: Context) -> int:
    say(f"{'id':26} {'group':11} {'state':10} {'pin':9} {'size':>9}  dest")
    for a in assets:
        s = check_asset(a, ctx.lock, ctx.root, hash_it=False)
        if s.present and not s.via_path:
            size = format_mb(size_of(s.path))
        else:
            size = f"~{a.size_mb:g} MB" if a.size_mb else "-"
        opt = " (optional)" if a.optional else ""
        say(f"{a.id:26} {a.group:11} {_state_word(s):10} {s.pin_source:9} {size:>9}  {a.dest}{opt}")
    return 0


def cmd_verify(assets: list[Asset], ctx: Context) -> int:
    bad = 0
    for a in assets:
        s = check_asset(a, ctx.lock, ctx.root)
        if s.via_path:
            say(f"[ -- ] {a.id}: found on PATH (not hashed)")
        elif not s.present:
            hard = not a.optional and a.kind != "local"
            say(f"[{'MISS' if hard else ' -- '}] {a.id}: {'missing' if hard else 'not present'}")
            bad += hard
        elif a.kind == "local":
            say(f"[ ok ] {a.id}: present (a trained model; not pinned, it changes on retrain)")
        elif s.verdict == "mismatch":
            say(f"[FAIL] {a.id}: hash differs from its {s.pin_source} pin")
            bad += 1
        elif s.verdict == "unpinned":
            say(f"[ -- ] {a.id}: present, unpinned (run --lock to record its hash)")
        else:
            say(f"[ ok ] {a.id}: matches its {s.pin_source} pin")
    say(f"{bad} problem(s)" if bad else "all present assets verify")
    return 1 if bad else 0


def cmd_lock(assets: list[Asset], ctx: Context, lock_path: Path) -> int:
    """Pin ``assets`` (the ids or groups asked for; everything when none were named). A file that
    already has a pin and no longer matches it is not re-pinned without ``--force``: a corrupted
    or tampered file must not quietly become the new truth. The "latest" media binaries are
    exempt, since a newer release is the normal reason for a mismatch there."""
    entries = dict(ctx.lock)
    n = 0
    for a in assets:
        if a.kind == "local":
            continue
        path = a.dest_path(ctx.root)
        if not path.exists():
            continue
        state = check_asset(a, ctx.lock, ctx.root)
        if state.verdict == "mismatch" and not a.resolver and not ctx.force:
            say(
                f"  {a.id}: differs from its {state.pin_source} pin; not re-pinned "
                "(re-fetch it, or pass --force to accept the file on disk)"
            )
            continue
        entries[a.id] = {
            "sha256": hash_path(path),
            "bytes": size_of(path),
            "dest": a.dest,
            "kind": a.kind,
        }
        n += 1
    save_lock(entries, lock_path)
    say(f"pinned {n} asset(s) in {lock_path}")
    say("commit it so the rest of the team verifies against the same hashes")
    return 0


def cmd_fetch(assets: list[Asset], ctx: Context) -> int:
    todo_mb = sum(
        a.size_mb or 0
        for a in assets
        if a.kind != "local" and not check_asset(a, ctx.lock, ctx.root, hash_it=False).present
    )
    say(f"{len(assets)} asset(s) selected; about {todo_mb:g} MB to download")
    results = [fetch_one(a, ctx) for a in assets]
    say()
    for r in results:
        say(f"  {r.status:12} {r.asset_id:26} {r.detail}")
    failed = [r for r in results if r.status == "failed"]
    unpinned = [r for r in results if r.status == "downloaded" and r.detail == "unpinned"]
    if unpinned:
        say()
        say("downloaded without a pin; run `tools/fetch_models.py --lock` to record the hashes")
    if failed:
        say()
        say(f"{len(failed)} asset(s) failed. The pipeline degrades gracefully without them;")
        say("run tools/doctor.py to see what each one costs.")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ids", nargs="*", help="asset ids (default: the default set)")
    ap.add_argument("--group", action="append", help="perception|voice|open_vocab|coco|media")
    ap.add_argument("--all", action="store_true", help="include optional assets")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--verify", action="store_true", help="re-hash what is on disk")
    mode.add_argument("--lock", action="store_true", help="pin what is on disk")
    mode.add_argument("--warm", action="store_true", help="prime the open-vocab detector")
    ap.add_argument("--from", dest="src_dir", type=Path, help="read files from this folder")
    ap.add_argument("--force", action="store_true", help="fetch even if present")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mediamtx-version", help="pin a MediaMTX release tag, e.g. v1.11.3")
    ap.add_argument("--root", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    root = args.root.resolve()
    try:
        if args.warm:
            return 0 if warm_open_vocab() else 1
        groups = [g for chunk in (args.group or []) for g in chunk.split(",")]
        every = all_assets(root)
        inspecting = args.list or args.verify  # these look at everything, optional included
        if not groups:
            groups = [*DEFAULT_GROUPS, "trained"] if inspecting else list(DEFAULT_GROUPS)
        chosen = select_assets(
            every, ids=args.ids, groups=groups, include_optional=inspecting or args.all
        )
        ctx = Context(
            root=root,
            lock=load_lock(root / "configs" / "assets.lock.json"),
            src_dir=args.src_dir.resolve() if args.src_dir else None,
            force=args.force,
            dry_run=args.dry_run,
            mediamtx_version=args.mediamtx_version,
        )
        if args.list:
            return cmd_list(chosen, ctx)
        if args.verify:
            return cmd_verify(chosen, ctx)
        if args.lock:
            targets = chosen if (args.ids or args.group) else every
            return cmd_lock(targets, ctx, root / "configs" / "assets.lock.json")
        return cmd_fetch(chosen, ctx)
    except PkgError as e:
        say(f"fetch_models: {e}", err=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
