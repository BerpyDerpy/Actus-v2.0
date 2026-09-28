#!/usr/bin/env bash
# Fetches the two binaries T-IO's streamer.py manages: MediaMTX (RTSP + WebRTC server) and
# FFmpeg (RTSP push encoder), into bin/ at the repo root ("/bin/" is gitignored on purpose).
#
# Usage: scripts/setup_media.sh [extra fetch_models.py flags, e.g. --force --dry-run]
#
# This is a thin wrapper over T-PKG's fetcher (`tools/fetch_models.py --group media`), which
# already handles Windows, Linux (amd64 and arm64v8) and macOS (Homebrew ffmpeg), verifies
# downloads against configs/assets.lock.json, and skips binaries that are already in place. An
# earlier standalone version of this script duplicated that logic and had drifted (wrong ffmpeg
# arch names, no Linux ARM MediaMTX, no Windows ffmpeg, no hash check); see Vault/T-REVIEW.md.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if command -v uv >/dev/null 2>&1; then
  exec uv run python tools/fetch_models.py --group media "$@"
fi

# No uv: any Python with PyYAML can run the fetcher (it is stdlib otherwise).
for py in python3 python; do
  if command -v "$py" >/dev/null 2>&1 && "$py" -c "import yaml" >/dev/null 2>&1; then
    exec "$py" tools/fetch_models.py --group media "$@"
  fi
done

echo "setup_media: needs uv (or a Python with PyYAML) to run tools/fetch_models.py." >&2
echo "Run scripts/setup.sh first, or install uv: https://docs.astral.sh/uv/" >&2
exit 1
