"""Shared helpers for the T-PKG tools (``fetch_models.py``, ``bundle.py``, ``doctor.py``).

Standard library only, apart from PyYAML, which ``load_manifest`` imports on first use, so that
``tools/bundle.py install`` can run on a bare system Python before ``uv sync`` has built an
environment. Written for Python >= 3.9 for the same reason. Console output is ASCII-only
(``say``): the Windows console is cp1252, and a stray arrow in a ``print`` crashes a tool.

What lives here:

* the asset manifest (``configs/assets.yaml``) and the lock file (``configs/assets.lock.json``);
* hashing (a file, or a whole directory tree) and the pin rules (manifest > URL > lock > none);
* a small resumable-free downloader (atomic ``.part`` rename, retries, hash while streaming);
* safe archive extraction (zip-slip guarded);
* platform detection and the MediaMTX / FFmpeg release pickers (pure functions, so they can be
  tested without a network).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "assets.yaml"
LOCK_PATH = REPO_ROOT / "configs" / "assets.lock.json"
SCHEMA_VERSION = 1
CHUNK = 1 << 20
USER_AGENT = "actus-setup/1"

MEDIAMTX_API = "https://api.github.com/repos/bluenviron/mediamtx/releases/latest"
MEDIAMTX_DOWNLOAD = "https://github.com/bluenviron/mediamtx/releases/download"
FFMPEG_BASE = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest"

KINDS = ("file", "zip_dir", "local")


class PkgError(RuntimeError):
    """Anything a T-PKG tool wants to report to the user without a traceback."""


class DownloadError(PkgError):
    """The transfer failed (network, 404). A hash mismatch is a plain ``PkgError`` instead."""


class _Truncated(OSError):
    """The server closed the connection before Content-Length bytes arrived (retryable)."""


def say(msg: str = "", *, err: bool = False) -> None:
    stream = sys.stderr if err else sys.stdout
    stream.write(msg.encode("ascii", "replace").decode("ascii") + "\n")
    stream.flush()


def format_mb(n_bytes: float) -> str:
    return f"{n_bytes / (1 << 20):.1f} MB"


# --------------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------------


@dataclass
class Asset:
    """One file (or directory) ACTUS needs at runtime that git does not carry.

    ``dest`` is relative to the repo root, or starts with ``~`` for the user's home (the CLIP
    text encoder lives in ``~/.cache/clip`` because that is where the ``clip`` package looks).
    ``sha256`` pins the file at ``dest`` (for ``zip_dir``: the tree hash, see ``sha256_tree``).
    ``resolver`` is filled in code for the platform-specific media binaries, never in YAML.
    """

    id: str
    dest: str
    kind: str = "file"  # file | zip_dir (a .zip unpacked to a directory) | local (not fetchable)
    url: str | None = None
    sha256: str | None = None
    sha256_from_url: bool = False  # the URL's parent folder is the file's SHA-256 (OpenAI CLIP)
    group: str = "misc"
    optional: bool = False  # fetched only with --all, or when named explicitly
    size_mb: float | None = None
    licence: str = ""
    description: str = ""
    produced_by: str = ""  # for kind: local, which lane makes it
    resolver: str = ""  # "" | "mediamtx" | "ffmpeg"
    fallback: str = ""  # "" | "ultralytics": use its downloader if the URL fails

    def dest_path(self, root: Path = REPO_ROOT) -> Path:
        return resolve_dest(self.dest, root)


def resolve_dest(dest: str, root: Path = REPO_ROOT) -> Path:
    if dest.startswith("~"):
        return Path(dest).expanduser()
    p = Path(dest)
    return p if p.is_absolute() else root / p


def rel_key(dest: str) -> str:
    """The path an asset has inside a bundle: repo-relative POSIX, or ``home/...`` for ``~/...``."""
    if dest.startswith("~"):
        return "home/" + dest[1:].lstrip("/\\").replace("\\", "/")
    return dest.replace("\\", "/")


def load_manifest(path: Path = MANIFEST_PATH) -> list[Asset]:
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if data.get("schema_version") != SCHEMA_VERSION:
        raise PkgError(f"{path}: schema_version must be {SCHEMA_VERSION}")
    allowed = {f.name for f in fields(Asset)} - {"resolver"}
    seen: set[str] = set()
    out: list[Asset] = []
    for i, raw in enumerate(data.get("assets") or []):
        if not isinstance(raw, dict):
            raise PkgError(f"{path}: assets[{i}] must be a mapping")
        unknown = set(raw) - allowed
        if unknown:
            raise PkgError(f"{path}: assets[{i}] has unknown keys {sorted(unknown)}")
        for req in ("id", "dest"):
            if not raw.get(req):
                raise PkgError(f"{path}: assets[{i}] is missing {req!r}")
        asset = Asset(**raw)
        if asset.id in seen:
            raise PkgError(f"{path}: duplicate asset id {asset.id!r}")
        seen.add(asset.id)
        if asset.kind not in KINDS:
            raise PkgError(f"{path}: {asset.id}: kind must be one of {KINDS}")
        if asset.kind != "local" and not asset.url:
            raise PkgError(f"{path}: {asset.id}: a fetchable asset needs a url")
        if asset.sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", str(asset.sha256)):
            raise PkgError(f"{path}: {asset.id}: sha256 must be 64 hex characters")
        out.append(asset)
    return out


# --------------------------------------------------------------------------------------------
# Hashing, pins and the lock file
# --------------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(root: Path) -> str:
    """Deterministic hash of a directory: every file's relative POSIX path and SHA-256, sorted."""
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix != ".part"):
        h.update(f"{f.relative_to(root).as_posix()}\0{sha256_file(f)}\n".encode())
    return h.hexdigest()


def hash_path(path: Path) -> str | None:
    if path.is_dir():
        return sha256_tree(path)
    if path.is_file():
        return sha256_file(path)
    return None


def size_of(path: Path) -> int:
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return path.stat().st_size if path.is_file() else 0


def load_lock(path: Path = LOCK_PATH) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise PkgError(f"{path}: unsupported lock schema {data.get('schema_version')!r}")
    return dict(data.get("assets") or {})


def save_lock(entries: dict[str, dict[str, Any]], path: Path = LOCK_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"schema_version": SCHEMA_VERSION, "assets": {k: entries[k] for k in sorted(entries)}}
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(body, f, indent=2)
        f.write("\n")


def expected_hash(asset: Asset, lock: dict[str, dict[str, Any]]) -> tuple[str | None, str]:
    """The hash this asset must have, and where that pin came from.

    Precedence: the manifest's ``sha256``, then the hash embedded in the URL (OpenAI publishes
    CLIP weights under their own SHA-256), then the lock file, else no pin ("none": the first
    download is trusted and ``fetch_models.py --lock`` records it).
    """
    if asset.sha256:
        return str(asset.sha256).lower(), "manifest"
    if asset.sha256_from_url and asset.url:
        seg = asset.url.rstrip("/").split("/")[-2]
        if re.fullmatch(r"[0-9a-f]{64}", seg):
            return seg, "url"
    entry = lock.get(asset.id)
    if entry and entry.get("sha256"):
        return str(entry["sha256"]).lower(), "lock"
    return None, "none"


@dataclass
class AssetState:
    asset: Asset
    path: Path
    present: bool
    via_path: bool = False  # FFmpeg found on PATH rather than in bin/
    actual: str | None = None
    expected: str | None = None
    pin_source: str = "none"

    @property
    def verdict(self) -> str:
        """missing | ok | mismatch | unpinned | present (present, hash not computed)"""
        if not self.present:
            return "missing"
        if self.expected is None:
            return "unpinned"
        if self.actual is None:
            return "present"
        return "ok" if self.actual == self.expected else "mismatch"


def check_asset(
    asset: Asset,
    lock: dict[str, dict[str, Any]],
    root: Path = REPO_ROOT,
    *,
    hash_it: bool = True,
) -> AssetState:
    path = asset.dest_path(root)
    expected, source = expected_hash(asset, lock)
    present = path.exists()
    via_path = False
    if not present and asset.resolver == "ffmpeg" and shutil.which("ffmpeg"):
        present, via_path = True, True
    actual = hash_path(path) if (present and not via_path and hash_it) else None
    return AssetState(asset, path, present, via_path, actual, expected, source)


# --------------------------------------------------------------------------------------------
# Downloading and archives
# --------------------------------------------------------------------------------------------


class _Progress:
    """A one-line progress readout, drawn only on an interactive terminal."""

    def __init__(self, label: str, total: int) -> None:
        self.label, self.total = label, total
        self.on = sys.stderr.isatty()
        self._last = 0.0

    def update(self, got: int) -> None:
        now = time.monotonic()
        if not self.on or now - self._last < 0.25:
            return
        self._last = now
        of = f" / {format_mb(self.total)}" if self.total else ""
        sys.stderr.write(f"\r  {self.label}: {format_mb(got)}{of}   ")
        sys.stderr.flush()

    def done(self) -> None:
        if self.on:
            sys.stderr.write("\r" + " " * 72 + "\r")
            sys.stderr.flush()


def _unlink(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink()


def _stream_to(url: str, part: Path, timeout: float, label: str, opener: Any) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    h = hashlib.sha256()
    got = 0
    with opener(req, timeout=timeout) as resp, open(part, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        progress = _Progress(label, total)
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            out.write(chunk)
            h.update(chunk)
            got += len(chunk)
            progress.update(got)
        progress.done()
    if total and got != total:
        raise _Truncated(f"got {got} of {total} bytes")
    return h.hexdigest()


def download(
    url: str,
    dest: Path,
    *,
    expected_sha256: str | None = None,
    retries: int = 3,
    timeout: float = 30.0,
    label: str = "",
    opener: Callable[..., Any] | None = None,
) -> str:
    """Fetch ``url`` to ``dest`` atomically (``dest.part`` then rename); returns its SHA-256.

    Retries transient failures with backoff, but never a 4xx (except 429): a 404 will not fix
    itself. A hash mismatch deletes the download and raises; nothing wrong is ever left at
    ``dest``. ``file://`` URLs work, which is how ``--from DIR`` and the tests avoid the network.
    """
    opener = opener or urllib.request.urlopen
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    digest: str | None = None
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            digest = _stream_to(url, part, timeout, label or dest.name, opener)
            break
        except OSError as e:  # URLError, HTTPError, timeouts and _Truncated are all OSErrors
            last = e
            _unlink(part)
            fatal = isinstance(e, urllib.error.HTTPError) and 400 <= e.code < 500 and e.code != 429
            if fatal or attempt == retries:
                break
            say(f"  {label or dest.name}: {e}; retrying ({attempt}/{retries - 1})")
            time.sleep(min(2**attempt, 20))
    if digest is None:
        raise DownloadError(f"download failed: {url}: {last}")
    if expected_sha256 and digest != expected_sha256.lower():
        _unlink(part)
        raise PkgError(
            f"{dest.name}: SHA-256 mismatch (got {digest}, expected {expected_sha256}); "
            "the download was discarded"
        )
    os.replace(part, dest)
    return digest


def fetch_json(url: str, *, opener: Callable[..., Any] | None = None) -> Any:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    opener = opener or urllib.request.urlopen
    try:
        with opener(urllib.request.Request(url, headers=headers), timeout=30.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError) as e:
        raise PkgError(f"could not read {url}: {e}") from e


def _check_inside(root: Path, target: Path, name: str) -> None:
    root_r, target_r = root.resolve(), target.resolve()
    if target_r != root_r and root_r not in target_r.parents:
        raise PkgError(f"unsafe path in archive: {name!r}")


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    """Unpack a zip into ``out_dir``, refusing any member that would land outside it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target = out_dir / info.filename
            _check_inside(out_dir, target, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def extract_selected(
    archive: Path, pick: Callable[[str], str | None], out_dir: Path
) -> list[Path]:
    """Copy the archive members ``pick`` names (member path -> output file name) into ``out_dir``.

    Works on ``.zip`` and ``.tar.*``. Only regular files are read and every output name is a bare
    file name chosen by ``pick``, so a hostile archive cannot write anywhere else. Each file is
    written to ``<name>.part`` and renamed into place, so a failure (a corrupt archive, the old
    binary still running) never leaves a half-written executable behind.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _place(src, name: str) -> None:
        part = out_dir / (name + ".part")
        try:
            with open(part, "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.replace(part, out_dir / name)
        finally:
            part.unlink(missing_ok=True)
        written.append(out_dir / name)

    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                name = None if info.is_dir() else pick(info.filename)
                if name:
                    with zf.open(info) as src:
                        _place(src, name)
    else:
        with tarfile.open(archive, "r:*") as tf:
            for member in tf:
                name = pick(member.name) if member.isfile() else None
                if name:
                    src = tf.extractfile(member)
                    if src is None:
                        continue
                    with src:
                        _place(src, name)
    return written


def make_executable(path: Path) -> None:
    if sys.platform != "win32":
        path.chmod(path.stat().st_mode | 0o111)


# --------------------------------------------------------------------------------------------
# Platform and the media binaries
# --------------------------------------------------------------------------------------------

_ARCHES = {
    "x86_64": "amd64",
    "amd64": "amd64",
    "aarch64": "arm64",
    "arm64": "arm64",
    "armv7l": "armv7",
}
# MediaMTX names its 64-bit Linux ARM builds "arm64v8"; keep "arm64" as a fallback.
_MEDIAMTX_ARCH = {("linux", "arm64"): ("arm64v8", "arm64"), ("linux", "armv7"): ("armv7",)}
_FFMPEG_BUILDS = {
    ("windows", "amd64"): "ffmpeg-master-latest-win64-gpl.zip",
    ("windows", "arm64"): "ffmpeg-master-latest-winarm64-gpl.zip",
    ("linux", "amd64"): "ffmpeg-master-latest-linux64-gpl.tar.xz",
    ("linux", "arm64"): "ffmpeg-master-latest-linuxarm64-gpl.tar.xz",
}


def detect_platform() -> tuple[str, str]:
    """``("windows" | "linux" | "darwin", "amd64" | "arm64" | ...)`` for this machine."""
    machine = platform.machine().lower()
    return platform.system().lower(), _ARCHES.get(machine, machine)


def platform_tag() -> str:
    os_name, arch = detect_platform()
    return f"{os_name}-{arch}"


def mediamtx_arch_candidates(os_name: str, arch: str) -> tuple[str, ...]:
    return _MEDIAMTX_ARCH.get((os_name, arch), (arch,))


def pick_mediamtx_asset(release: dict[str, Any], os_name: str, arch: str) -> tuple[str, str]:
    """(file name, download URL) of the MediaMTX release asset for this platform."""
    for alt in mediamtx_arch_candidates(os_name, arch):
        rx = re.compile(rf"^mediamtx_v[\d.]+_{os_name}_{alt}\.(?:tar\.gz|zip)$")
        for a in release.get("assets", []):
            if rx.match(str(a.get("name", ""))):
                return str(a["name"]), str(a["browser_download_url"])
    raise PkgError(
        f"no MediaMTX release asset for {os_name}/{arch}; "
        "see https://github.com/bluenviron/mediamtx/releases"
    )


def mediamtx_pinned_url(version: str, os_name: str, arch: str) -> tuple[str, str]:
    alt = mediamtx_arch_candidates(os_name, arch)[0]
    ext = "zip" if os_name == "windows" else "tar.gz"
    name = f"mediamtx_{version}_{os_name}_{alt}.{ext}"
    return name, f"{MEDIAMTX_DOWNLOAD}/{version}/{name}"


def ffmpeg_download(os_name: str, arch: str) -> tuple[str, str] | None:
    """(file name, URL) of a static FFmpeg build, or None where the package manager is the
    right source (macOS: ``brew install ffmpeg``)."""
    name = _FFMPEG_BUILDS.get((os_name, arch))
    return (name, f"{FFMPEG_BASE}/{name}") if name else None


_FFMPEG_EXES = ("ffmpeg", "ffmpeg.exe", "ffprobe", "ffprobe.exe")


def pick_mediamtx_member(name: str) -> str | None:
    parts = PurePosixPath(name).parts
    if len(parts) == 1 and parts[0] in ("mediamtx", "mediamtx.exe", "mediamtx.yml"):
        return parts[0]
    return None


def pick_ffmpeg_member(name: str) -> str | None:
    parts = PurePosixPath(name).parts
    if len(parts) >= 2 and parts[-2] == "bin" and parts[-1] in _FFMPEG_EXES:
        return parts[-1]
    return None


def media_assets(os_name: str, arch: str) -> list[Asset]:
    """The two platform-specific binaries the streamer needs, as ordinary assets."""
    exe = ".exe" if os_name == "windows" else ""
    return [
        Asset(
            id=f"mediamtx-{os_name}-{arch}",
            dest=f"bin/mediamtx{exe}",
            group="media",
            resolver="mediamtx",
            size_mb=25,
            licence="MIT (MediaMTX)",
            description="RTSP and WebRTC server the streamer manages",
        ),
        Asset(
            id=f"ffmpeg-{os_name}-{arch}",
            dest=f"bin/ffmpeg{exe}",
            group="media",
            resolver="ffmpeg",
            size_mb=130,
            licence="GPL-3.0 for the BtbN build, LGPL/GPL for a package-manager build",
            description="H.264 RTSP push encoder (libx264 is required)",
        ),
    ]

