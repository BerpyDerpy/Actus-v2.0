#!/usr/bin/env bash
# Run ACTUS end to end: capture -> perception -> spatial -> engine -> voice / GUI / RTSP / MP4.
#
# Usage:
#   scripts/run_demo.sh --live                         # camera per configs/default.yaml capture.source
#   scripts/run_demo.sh --live --source 1              # another webcam index, or an rtsp:// URL
#   scripts/run_demo.sh --file data/raw/op01_correct_01.mp4    # a recorded take, at its own rate
#   scripts/run_demo.sh --replay skip_S2               # SIMULATED replay, no camera or models
#   scripts/run_demo.sh --offline <take.mp4 | replay:SCENARIO> [--verify-determinism ...]
#
# Other flags go to `python -m actus.app.pipeline` (see --help), e.g.
#   --no-stream  --no-voice  --answer yes  --override stream.target_ip=192.168.1.20
#
# Fault injection (T-FAULT): add --faults (hotkeys o d f x n + GUI buttons) or a schedule, e.g.
#   scripts/run_demo.sh --replay correct --fault occlude@9-12
#   python -m actus.app.faults --demo all              # every fault vs ground truth, offline
#
# The GUI is on http://127.0.0.1:8000 (ui.host / ui.port). Outputs go to runs/<run_id>/.
# One-time setup: uv sync --all-extras; scripts/setup_media.sh; (cd ui && npm ci && npm run build)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PATH="$REPO_ROOT/bin:$PATH"   # ffmpeg / mediamtx installed by scripts/setup_media.sh

mode="${1:-}"
case "$mode" in
  --offline)
    shift
    exec uv run --extra perception python tools/run_offline.py "$@"
    ;;
  --replay)
    exec uv run --extra ui --extra voice python -m actus.app.pipeline "$@"
    ;;
  --live|--file|--source)
    exec uv run --extra perception --extra voice --extra ui python -m actus.app.pipeline "$@"
    ;;
  -h|--help|"")
    sed -n '2,19p' "$0"
    exit 0
    ;;
  *)
    echo "run_demo: first argument must be --live, --file, --replay or --offline (got $mode)" >&2
    exit 2
    ;;
esac
