"""Pack the runtime assets into one verified archive, and unpack it on an offline machine (T-PKG).

    uv run python tools/bundle.py make                    # dist/actus-bundle-<platform>-<date>.zip
    uv run python tools/bundle.py make --wheels           # + the Python wheels, for a bare machine
    uv run python tools/bundle.py make --no-ui --exclude "models/detector/runs/*"
    uv run python tools/bundle.py info    dist/actus-bundle-windows-amd64-20260930.zip
    uv run python tools/bundle.py verify  dist/actus-bundle-windows-amd64-20260930.zip
    python tools/bundle.py install dist/actus-bundle-windows-amd64-20260930.zip   # no environment

The bundle carries what git does not: everything under ``models/`` (the downloaded weights and
the models trained here, with their cards), the MediaMTX and FFmpeg binaries in ``bin/``, the
built GUI (``ui/dist``), the CLIP text encoder from ``~/.cache/clip``, the pins in
``configs/assets.lock.json`` and, with ``--wheels``, every Python wheel the locked environment
needs. The source code is not in it: copy or clone the repo first, then install the bundle into it.
Bring a machine up offline with ``scripts/setup.sh --bundle <file>`` (``setup.ps1 -Bundle``).

Layout: ``bundle.json`` (schema, creation time, platform, Python, git commit, and every file's
path, SHA-256, size and mode) plus the files under ``payload/``. ``install`` checks each file's
hash while unpacking, stages it as ``<name>.part`` and renames only when the whole bundle
verified, so a corrupt or altered bundle changes nothing. It also refuses any path outside
``models/``, ``bin/``, ``ui/dist/``, ``wheels/``, ``configs/assets.lock.json``, a root-level
``yolo*.pt`` and ``~/.cache/clip/``, so a hostile bundle cannot write elsewhere.

Binaries and wheels are platform-specific. A bundle made on one platform refuses to install on
another unless ``--portable-only`` says to take just the platform-independent files.

``install`` uses only the standard library and runs on any Python >= 3.9, so it works before
``uv sync`` has created the environment. ``make`` needs the project environment (and, for
``--wheels``, ``uv`` on PATH and network access).
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._pkglib import (  # noqa: E402
    CHUNK,
    Asset,
    PkgError,
    detect_platform,
    format_mb,
    load_manifest,
    media_assets,
    platform_tag,
    rel_key,
    say,
    sha256_file,
)

BUNDLE_KIND = "actus-asset-bundle"
SCHEMA = 1
MANIFEST_NAME = "bundle.json"
PAYLOAD = "payload/"
ALLOWED_PREFIXES = ("models/", "bin/", "ui/dist/", "wheels/", "home/.cache/clip/")
PLATFORM_PREFIXES = ("bin/", "wheels/")
STORED_SUFFIXES = (".whl", ".zip", ".gz", ".xz", ".mp4", ".png", ".jpg")
SKIP_PARTS = ("__pycache__",)
CLIP_GIT = "git+https://github.com/ultralytics/CLIP.git"


def is_allowed_key(key: str) -> bool:
    """True for a relative POSIX path that stays inside the places a bundle may write to."""
    if not key or "\\" in key or key.startswith("/") or re.match(r"^[A-Za-z]:", key):
        return False
    if any(part in ("", ".", "..") for part in key.split("/")):
        return False
    return (
        key.startswith(ALLOWED_PREFIXES)
        or key == "configs/assets.lock.json"
        or re.fullmatch(r"yolo[A-Za-z0-9_.\-]*\.pt", key) is not None
    )


# --------------------------------------------------------------------------------------------
# make
# --------------------------------------------------------------------------------------------


def _add_tree(files: dict[str, Path], base: Path, prefix: str) -> None:
    if not base.is_dir():
        return
    for p in sorted(base.rglob("*")):
        if p.is_file():
            files[f"{prefix}/{p.relative_to(base).as_posix()}"] = p


def collect_files(
    root: Path,
    assets: list[Asset],
    *,
    include_ui: bool = True,
    excludes: list[str] | None = None,
    home: Path | None = None,
) -> tuple[dict[str, Path], list[str]]:
    """Bundle key -> source file, plus notes about anything an offline machine would miss."""
    files: dict[str, Path] = {}
    notes: list[str] = []
    for a in assets:
        if a.dest.startswith("~") and home is not None:
            path = home / a.dest[1:].lstrip("/\\")
        else:
            path = a.dest_path(root)
        if path.is_dir():
            _add_tree(files, path, rel_key(a.dest))
        elif path.is_file():
            files[rel_key(a.dest)] = path
        elif not a.optional and a.kind != "local":
            if a.resolver == "ffmpeg" and shutil.which("ffmpeg"):
                notes.append("ffmpeg is on PATH, not in bin/: the bundle does not carry it")
            else:
                notes.append(f"{a.id} is missing ({a.dest}); run tools/fetch_models.py first")
    _add_tree(files, root / "models", "models")
    _add_tree(files, root / "bin", "bin")
    if include_ui:
        if (root / "ui" / "dist").is_dir():
            _add_tree(files, root / "ui" / "dist", "ui/dist")
        else:
            notes.append("ui/dist is missing: no GUI (cd ui && npm ci && npm run build)")
    lock = root / "configs" / "assets.lock.json"
    if lock.is_file():
        files["configs/assets.lock.json"] = lock
    else:
        notes.append("configs/assets.lock.json is missing: the install cannot verify pins")
    root_pt = sorted(root.glob("yolo*.pt"))
    for p in root_pt:
        files.setdefault(p.name, p)
    patterns = excludes or []
    kept = {
        k: p
        for k, p in sorted(files.items())
        if not k.endswith(".part")
        and not any(s in k.split("/") for s in SKIP_PARTS)
        and not any(fnmatch.fnmatch(k, pat) for pat in patterns)
    }
    return kept, notes


def _dos_time(mtime: float) -> tuple[int, int, int, int, int, int]:
    t = time.localtime(max(mtime, 315532800.0))  # zip cannot store dates before 1980
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)


def _copy_hashing(src: Any, dst: Any) -> str:
    """Copy ``src`` to ``dst`` in chunks; returns the SHA-256 of what was copied."""
    h = hashlib.sha256()
    for chunk in iter(lambda: src.read(CHUNK), b""):
        h.update(chunk)
        dst.write(chunk)
    return h.hexdigest()


def _add_file(zf: zipfile.ZipFile, key: str, path: Path) -> dict[str, Any]:
    st = path.stat()
    info = zipfile.ZipInfo(PAYLOAD + key, date_time=_dos_time(st.st_mtime))
    info.compress_type = (
        zipfile.ZIP_STORED if key.endswith(STORED_SUFFIXES) else zipfile.ZIP_DEFLATED
    )
    mode = (st.st_mode & 0o777) or 0o644
    info.external_attr = mode << 16
    with open(path, "rb") as src, zf.open(info, "w", force_zip64=True) as dst:
        digest = _copy_hashing(src, dst)
    return {"path": key, "sha256": digest, "bytes": st.st_size, "mode": mode}


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _run(cmd: list[str], *, cwd: Path) -> None:
    say("  $ " + " ".join(cmd))
    if subprocess.run(cmd, cwd=cwd, check=False).returncode != 0:
        raise PkgError(f"command failed: {' '.join(cmd)}")


def build_wheels(root: Path, out_dir: Path, python: str) -> list[str]:
    """Download every wheel the locked environment needs, plus what an offline install needs to
    bootstrap (uv, the build backend, the ``clip`` package). Returns notes for the manifest."""
    import tomllib

    uv = shutil.which("uv")
    if uv is None:
        raise PkgError("--wheels needs uv on PATH")
    notes: list[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    req = out_dir / "requirements.txt"
    _run(
        [uv, "export", "--frozen", "--all-extras", "--no-dev", "--no-emit-project",
         "--no-hashes", "-o", str(req)],
        cwd=root,
    )  # fmt: skip
    with open(root / "pyproject.toml", "rb") as f:
        build_requires = tomllib.load(f).get("build-system", {}).get("requires", [])
    pip = [uv, "tool", "run", "--python", python, "pip"]
    dl = ["download", "--only-binary=:all:", "--disable-pip-version-check", "-d", str(out_dir)]
    _run([*pip, *dl, "-r", str(req)], cwd=root)
    _run([*pip, *dl, *build_requires, "uv", "ftfy", "regex", "tqdm"], cwd=root)
    if shutil.which("git"):
        try:
            _run(
                [*pip, "wheel", "--no-deps", "--disable-pip-version-check",
                 "-w", str(out_dir), CLIP_GIT],
                cwd=root,
            )  # fmt: skip
        except PkgError:
            notes.append("the clip package wheel could not be built; YOLO-World needs it offline")
    else:
        notes.append("git is not on PATH, so the clip package wheel was not built")
    return notes


def make_bundle(
    out: Path,
    *,
    root: Path = REPO_ROOT,
    assets: list[Asset] | None = None,
    include_ui: bool = True,
    excludes: list[str] | None = None,
    wheels: bool = False,
    python: str = "3.11",
    home: Path | None = None,
) -> dict[str, Any]:
    """Write the bundle to ``out`` (plus ``<out>.sha256``); returns the manifest it embeds."""
    if assets is None:
        os_name, arch = detect_platform()
        assets = [*load_manifest(root / "configs" / "assets.yaml"), *media_assets(os_name, arch)]
    files, notes = collect_files(root, assets, include_ui=include_ui, excludes=excludes, home=home)
    out.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="actus-wheels-") as tmp:
        if wheels:
            notes += build_wheels(root, Path(tmp), python)
            for p in sorted(Path(tmp).iterdir()):
                if p.is_file():
                    files[f"wheels/{p.name}"] = p
        part = out.with_name(out.name + ".part")
        with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            for key, path in files.items():
                entries.append(_add_file(zf, key, path))
                say(f"  + {key}  ({format_mb(entries[-1]['bytes'])})")
            manifest = {
                "kind": BUNDLE_KIND,
                "schema_version": SCHEMA,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "platform": platform_tag(),
                "python": ".".join(map(str, sys.version_info[:3])),
                "git_commit": _git_commit(root),
                "wheels": {"included": wheels, "python": python},
                "notes": notes,
                "files": entries,
            }
            zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2) + "\n")
        os.replace(part, out)
    sidecar = out.with_name(out.name + ".sha256")
    sidecar.write_text(f"{sha256_file(out)}  {out.name}\n", encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------------------------
# info / verify / install
# --------------------------------------------------------------------------------------------


def read_manifest(zf: zipfile.ZipFile) -> dict[str, Any]:
    try:
        manifest = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))
    except (KeyError, ValueError) as e:
        raise PkgError(f"not an ACTUS bundle: cannot read {MANIFEST_NAME} ({e})") from e
    if manifest.get("kind") != BUNDLE_KIND or manifest.get("schema_version") != SCHEMA:
        raise PkgError("not an ACTUS bundle, or a schema this tool does not understand")
    for entry in manifest.get("files", []):
        if not is_allowed_key(str(entry.get("path", ""))):
            raise PkgError(f"refusing bundle: {entry.get('path')!r} is outside the allowed paths")
    return manifest


class _Discard:
    """A write-only sink, so hashing a 300 MB member does not hold it in memory."""

    def write(self, data: bytes) -> int:
        return len(data)


def _hash_member(zf: zipfile.ZipFile, name: str) -> str:
    with zf.open(name) as src:
        return _copy_hashing(src, _Discard())


def verify_bundle(bundle: Path) -> list[str]:
    """Every problem found (an empty list means the bundle is intact)."""
    problems: list[str] = []
    with zipfile.ZipFile(bundle) as zf:
        manifest = read_manifest(zf)
        listed = {PAYLOAD + e["path"]: e for e in manifest["files"]}
        problems += [
            f"unlisted member {n}" for n in zf.namelist() if n != MANIFEST_NAME and n not in listed
        ]
        for name, entry in listed.items():
            try:
                got = _hash_member(zf, name)
            except KeyError:
                problems.append(f"listed but missing: {entry['path']}")
                continue
            except (zipfile.BadZipFile, OSError) as e:
                problems.append(f"{entry['path']}: unreadable ({e})")
                continue
            if got != entry["sha256"]:
                problems.append(f"{entry['path']}: hash mismatch")
    return problems


def _target(key: str, repo: Path, home: Path) -> Path:
    return home / key[len("home/") :] if key.startswith("home/") else repo / key


def _apply_mode(target: Path, key: str, mode: int) -> None:
    if sys.platform == "win32":
        return
    if key.startswith("bin/") and not key.endswith(".yml"):
        mode |= 0o755
    with contextlib.suppress(OSError):
        target.chmod(mode or 0o644)


def install_bundle(
    bundle: Path,
    repo: Path = REPO_ROOT,
    *,
    portable_only: bool = False,
    overwrite: bool = True,
    dry_run: bool = False,
    home: Path | None = None,
) -> list[str]:
    """Unpack ``bundle`` into ``repo`` (and ``~/.cache/clip``); returns the paths written.

    Hash-verifies every file while unpacking and renames nothing until all of them verified.
    """
    home = home or Path.home()
    written: list[str] = []
    with zipfile.ZipFile(bundle) as zf:
        manifest = read_manifest(zf)
        files = list(manifest["files"])
        if manifest.get("platform") != platform_tag():
            specific = [f for f in files if f["path"].startswith(PLATFORM_PREFIXES)]
            if specific and not portable_only:
                raise PkgError(
                    f"this bundle was built for {manifest.get('platform')}, this machine is "
                    f"{platform_tag()}: its binaries and wheels will not run here. Build a "
                    "bundle on this platform, or pass --portable-only to take just the "
                    "platform-independent files (models, GUI)."
                )
            files = [f for f in files if not f["path"].startswith(PLATFORM_PREFIXES)]
        staged: list[tuple[Path, Path, dict[str, Any]]] = []
        try:
            for entry in files:
                key = entry["path"]
                target = _target(key, repo, home)
                if target.is_file() and (not overwrite or sha256_file(target) == entry["sha256"]):
                    continue  # kept as it is, or already identical
                if dry_run:
                    written.append(key)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                part = target.with_name(target.name + ".part")
                try:
                    with zf.open(PAYLOAD + key) as src, open(part, "wb") as dst:
                        digest = _copy_hashing(src, dst)
                    if digest != entry["sha256"]:
                        raise PkgError(f"{key}: hash mismatch; the bundle is corrupt or altered")
                except KeyError as e:
                    raise PkgError(f"{key}: listed but missing from the archive") from e
                except BaseException:
                    with contextlib.suppress(OSError):
                        part.unlink()
                    raise
                staged.append((part, target, entry))
            for part, target, entry in staged:
                os.replace(part, target)
                _apply_mode(target, entry["path"], int(entry.get("mode", 0o644)))
                written.append(entry["path"])
            staged = []
        finally:
            for part, _, _ in staged:
                with contextlib.suppress(OSError):
                    part.unlink()
    return written


def _describe(manifest: dict[str, Any]) -> list[str]:
    files = manifest["files"]
    by_top: dict[str, list[int]] = {}
    for f in files:
        top = f["path"].split("/")[0] if "/" in f["path"] else "(repo root)"
        n, size = by_top.setdefault(top, [0, 0])
        by_top[top] = [n + 1, size + int(f["bytes"])]
    lines = [
        f"kind      {manifest['kind']} v{manifest['schema_version']}",
        f"created   {manifest.get('created_at')}",
        f"platform  {manifest.get('platform')}   python {manifest.get('python')}",
        f"commit    {manifest.get('git_commit') or 'unknown'}",
        f"wheels    {'yes' if manifest.get('wheels', {}).get('included') else 'no'}",
        f"files     {len(files)}, {format_mb(sum(int(f['bytes']) for f in files))}",
    ]
    lines += [f"  {top:12} {n:4} files  {format_mb(size):>10}" for top, (n, size) in by_top.items()]
    lines += [f"note      {n}" for n in manifest.get("notes", [])]
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    mk = sub.add_parser("make", help="build a bundle from this checkout")
    mk.add_argument("--out", type=Path, help="default: dist/actus-bundle-<platform>-<date>.zip")
    mk.add_argument("--wheels", action="store_true", help="include the Python wheels (large)")
    mk.add_argument("--no-ui", action="store_true", help="leave out ui/dist")
    mk.add_argument("--exclude", action="append", default=[], help="glob on bundle paths")
    mk.add_argument("--python", help="wheel target version, default: .python-version")
    mk.add_argument("--root", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)

    for name, help_ in (("info", "describe a bundle"), ("verify", "check every hash")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("bundle", type=Path)

    ins = sub.add_parser("install", help="unpack a bundle into this checkout (stdlib only)")
    ins.add_argument("bundle", type=Path)
    ins.add_argument("--repo", type=Path, default=REPO_ROOT)
    ins.add_argument("--portable-only", action="store_true")
    ins.add_argument("--no-overwrite", action="store_true", help="keep files already on disk")
    ins.add_argument("--dry-run", action="store_true")

    args = ap.parse_args(argv)
    try:
        if args.command == "make":
            root = args.root.resolve()
            pyver = args.python
            if pyver is None:
                pin = root / ".python-version"
                pyver = pin.read_text(encoding="utf-8").strip() if pin.is_file() else "3.11"
            stamp = time.strftime("%Y%m%d")
            out = args.out or root / "dist" / f"actus-bundle-{platform_tag()}-{stamp}.zip"
            manifest = make_bundle(
                out,
                root=root,
                include_ui=not args.no_ui,
                excludes=args.exclude,
                wheels=args.wheels,
                python=pyver,
            )
            say()
            for line in _describe(manifest):
                say(line)
            say(f"\nwrote {out} ({format_mb(out.stat().st_size)}) and {out.name}.sha256")
            return 0
        if args.command == "info":
            with zipfile.ZipFile(args.bundle) as zf:
                for line in _describe(read_manifest(zf)):
                    say(line)
            return 0
        if args.command == "verify":
            problems = verify_bundle(args.bundle)
            for p in problems:
                say(f"PROBLEM  {p}")
            say(f"{len(problems)} problem(s)" if problems else "bundle verifies")
            return 1 if problems else 0
        written = install_bundle(
            args.bundle,
            args.repo.resolve(),
            portable_only=args.portable_only,
            overwrite=not args.no_overwrite,
            dry_run=args.dry_run,
        )
        verb = "would install" if args.dry_run else "installed"
        say(f"{verb} {len(written)} file(s); the rest were already in place")
        return 0
    except (PkgError, zipfile.BadZipFile, OSError) as e:
        say(f"bundle: {e}", err=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
