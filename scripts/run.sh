#!/usr/bin/env bash
# One-command run. Same modes as run_demo.sh, plus the offline handling and a preflight.
#
# Usage:
#   bash scripts/run.sh                          # = --live (camera per capture.source)
#   bash scripts/run.sh --live --source 1        # another webcam index, or an rtsp:// URL
#   bash scripts/run.sh --file data/raw/op01_correct_01.mp4
#   bash scripts/run.sh --replay skip_S2         # SIMULATED, no camera or models needed
#   bash scripts/run.sh --offline <take.mp4 | replay:SCENARIO> [--verify-determinism]
#   bash scripts/run.sh doctor [--camera]        # what works, what degrades, what is broken
#
# Everything else goes to run_demo.sh / `python -m actus.app.pipeline` (see their --help), for
# example --no-stream, --answer yes, --override stream.target_ip=192.168.1.20, --faults.
#
# After `setup.sh --bundle` (an offline install) this script sets UV_OFFLINE and UV_NO_SYNC, so
# `uv run` neither reaches for the network nor tries to re-sync the environment.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$REPO_ROOT/bin:$PATH"

if [ -f "$REPO_ROOT/.actus-offline" ]; then
  export UV_OFFLINE=1 UV_NO_SYNC=1
fi

command -v uv >/dev/null 2>&1 || { echo "run: uv not found; run: bash scripts/setup.sh" >&2; exit 1; }

case "${1:-}" in
  -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
  doctor)    shift; exec uv run python tools/doctor.py "$@" ;;
  "")        set -- --live ;;
esac

exec bash "$REPO_ROOT/scripts/run_demo.sh" "$@"
