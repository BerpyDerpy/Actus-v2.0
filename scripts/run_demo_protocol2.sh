#!/usr/bin/env bash
# Run the second protocol (T-PROTO2: protocols/sample_mixing.yaml, cup / bottle / book).
#
# Usage:
#   scripts/run_demo_protocol2.sh                        # every SIMULATED scenario, no video
#   scripts/run_demo_protocol2.sh correct                # one scenario, with the annotated MP4
#   scripts/run_demo_protocol2.sh occluded_S5            # the ask path (scripted "yes")
#   scripts/run_demo_protocol2.sh --list
#   scripts/run_demo_protocol2.sh --live                 # camera, this protocol, COCO detector
#   scripts/run_demo_protocol2.sh --file data/raw/op05_secondary_01.mp4
#   scripts/run_demo_protocol2.sh --offline data/raw/op05_secondary_01.mp4 --verify-determinism
#
# The SIMULATED scenarios need no camera, no model weights and no audio: they run the real
# pipeline core (rack solve, world state, evidence, engine, hash-chained log, downlink packet) on a
# drawn scene. --live / --file / --offline hand over to scripts/run_demo.sh with the config overlay
# configs/sample_mixing.yaml (or your own --config, which then replaces it); the detector is then
# the stock YOLO11n COCO checkpoint, which must exist at models/coco/yolo11n.pt:
#   uv run python tools/fetch_models.py yolo11n_coco
# The box-world replay (replay:<scenario>) does not apply to this protocol; the SIMULATED
# scenarios above are its replay. Outputs go to runs/<run_id>/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

case "${1:-}" in
  --live|--file|--offline)
    for arg in "$@"; do
      case "$arg" in
        replay:*)
          echo "run_demo_protocol2: $arg is the box-world replay; for this protocol run a" \
               "SIMULATED scenario instead, e.g. scripts/run_demo_protocol2.sh correct" >&2
          exit 2
          ;;
      esac
    done
    for arg in "$@"; do
      if [ "$arg" = "--config" ] || [ "${arg#--config=}" != "$arg" ]; then
        exec "$REPO_ROOT/scripts/run_demo.sh" "$@"  # the caller's own config wins
      fi
    done
    exec "$REPO_ROOT/scripts/run_demo.sh" "$@" --config configs/sample_mixing.yaml
    ;;
  --replay|--source)
    echo "run_demo_protocol2: $1 is not available here; use --live, --file, --offline or a" \
         "SIMULATED scenario name (see --list)" >&2
    exit 2
    ;;
  -h|--help)
    sed -n '2,20p' "$0"
    exit 0
    ;;
  "")
    exec uv run python tools/demo_protocol2.py all --no-video
    ;;
  *)
    exec uv run python tools/demo_protocol2.py "$@"
    ;;
esac
