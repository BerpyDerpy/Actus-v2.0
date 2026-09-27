#!/usr/bin/env bash
# Fetches the two binaries T-IO's streamer.py manages: MediaMTX (RTSP + WebRTC server) and
# FFmpeg (RTSP push encoder), into bin/ at the repo root ("/bin/" is gitignored on purpose).
#
# Usage: scripts/setup_media.sh
#
# Linux and macOS are fully automated (macOS via Homebrew for ffmpeg). Windows needs no MediaMTX
# build script step beyond the download below, but ffmpeg has no single trustworthy unattended
# download URL that stays valid over time, so this script prints where to get it instead of
# guessing one.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="$REPO_ROOT/bin"
mkdir -p "$BIN_DIR"

os="$(uname -s)"
arch="$(uname -m)"

case "$arch" in
  x86_64|amd64) mediamtx_arch="amd64" ;;
  aarch64|arm64) mediamtx_arch="arm64" ;;
  armv7l) mediamtx_arch="armv7" ;;
  *) echo "setup_media: unsupported architecture $arch" >&2; exit 1 ;;
esac

case "$os" in
  Linux*)   mediamtx_os="linux" ;;
  Darwin*)  mediamtx_os="darwin" ;;
  MINGW*|MSYS*|CYGWIN*) mediamtx_os="windows" ;;
  *) echo "setup_media: unsupported OS $os" >&2; exit 1 ;;
esac

echo "== MediaMTX =="
mediamtx_bin="$BIN_DIR/mediamtx"
[ "$mediamtx_os" = "windows" ] && mediamtx_bin="$mediamtx_bin.exe"

if [ -x "$mediamtx_bin" ]; then
  echo "already at $mediamtx_bin, skipping (delete it to re-fetch)"
else
  api_url="https://api.github.com/repos/bluenviron/mediamtx/releases/latest"
  echo "looking up the latest release from $api_url"
  release_json="$(curl -fsSL "$api_url")"
  asset_pattern="mediamtx_.*_${mediamtx_os}_${mediamtx_arch}\.(tar\.gz|zip)"
  asset_url="$(printf '%s' "$release_json" \
    | grep -oE '"browser_download_url": *"[^"]+"' \
    | sed -E 's/.*"(https:[^"]+)"/\1/' \
    | grep -E "$asset_pattern" | head -n1)"
  if [ -z "$asset_url" ]; then
    echo "setup_media: could not find a MediaMTX asset for ${mediamtx_os}/${mediamtx_arch}" >&2
    echo "check https://github.com/bluenviron/mediamtx/releases manually" >&2
    exit 1
  fi
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  echo "downloading $asset_url"
  archive="$tmp/mediamtx.archive"
  curl -fsSL "$asset_url" -o "$archive"
  case "$asset_url" in
    *.tar.gz) tar -xzf "$archive" -C "$tmp" ;;
    *.zip)    unzip -q "$archive" -d "$tmp" ;;
  esac
  extracted="$(find "$tmp" -maxdepth 1 -type f \( -name "mediamtx" -o -name "mediamtx.exe" \) | head -n1)"
  if [ -z "$extracted" ]; then
    echo "setup_media: mediamtx binary not found inside the downloaded archive" >&2
    exit 1
  fi
  cp "$extracted" "$mediamtx_bin"
  chmod +x "$mediamtx_bin"
  # MediaMTX's default config ships alongside the binary in the same archive; keep it if present.
  config_src="$(find "$tmp" -maxdepth 1 -type f -name "mediamtx.yml" | head -n1)"
  [ -n "$config_src" ] && cp "$config_src" "$BIN_DIR/mediamtx.yml"
  echo "installed $mediamtx_bin"
fi

echo
echo "== FFmpeg =="
if command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg already on PATH at $(command -v ffmpeg); nothing to do"
  echo "(set stream.ffmpeg_bin in configs/default.yaml if you'd rather pin a specific binary)"
elif [ "$mediamtx_os" = "darwin" ]; then
  if command -v brew >/dev/null 2>&1; then
    echo "installing ffmpeg via Homebrew"
    brew install ffmpeg
  else
    echo "Homebrew not found; install it from https://brew.sh, then run: brew install ffmpeg" >&2
    exit 1
  fi
elif [ "$mediamtx_os" = "linux" ]; then
  echo "downloading a static ffmpeg build from johnvansickle.com/ffmpeg (well-known static builds)"
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  build_url="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-${arch}-static.tar.xz"
  curl -fsSL "$build_url" -o "$tmp/ffmpeg.tar.xz"
  tar -xJf "$tmp/ffmpeg.tar.xz" -C "$tmp"
  extracted="$(find "$tmp" -maxdepth 2 -type f -name "ffmpeg" | head -n1)"
  if [ -z "$extracted" ]; then
    echo "setup_media: ffmpeg binary not found inside the downloaded archive" >&2
    exit 1
  fi
  cp "$extracted" "$BIN_DIR/ffmpeg"
  chmod +x "$BIN_DIR/ffmpeg"
  echo "installed $BIN_DIR/ffmpeg"
else
  cat >&2 <<'EOF'
setup_media: no unattended ffmpeg download for Windows (build URLs there rotate too often to
hardcode reliably). Get a build yourself:
  - https://www.ffmpeg.org/download.html#build-windows
Then either put ffmpeg.exe on PATH, or point stream.ffmpeg_bin in configs/default.yaml (or
configs/rack.yaml overrides) at wherever you unzipped it, e.g. bin/ffmpeg.exe.
EOF
  exit 1
fi

echo
echo "done. Set stream.mediamtx_bin / stream.ffmpeg_bin in configs/default.yaml if you moved"
echo "either binary out of bin/."
