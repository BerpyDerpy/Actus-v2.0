#!/usr/bin/env bash
# One-command setup: Python environment, GUI build, models, MediaMTX and FFmpeg, then a preflight.
#
# Usage:
#   bash scripts/setup.sh                       # online: everything, then `tools/doctor.py`
#   bash scripts/setup.sh --demo                # ... then start the SIMULATED replay demo
#   bash scripts/setup.sh --bundle FILE.zip     # offline: install from a bundle (tools/bundle.py)
#   bash scripts/setup.sh --all-models          # also the optional assets (pose full/heavy)
#   bash scripts/setup.sh --extras perception,ui    # only these Python extras (default: all)
#   bash scripts/setup.sh --no-media | --no-ui | --no-warm | --skip-models
#
# macOS, Linux and Git Bash. On native Windows use scripts/setup.ps1. The network is used only
# here; --bundle needs none (it does need Python 3.11 on the machine, see README "Offline").
# Afterwards: bash scripts/run.sh --live | --replay correct | doctor

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PATH="$REPO_ROOT/bin:$PATH"

BUNDLE=""; DEMO=0; ALL_MODELS=0; MEDIA=1; UI=1; WARM=1; MODELS=1; EXTRAS="all"
while [ $# -gt 0 ]; do
  case "$1" in
    --bundle)      BUNDLE="${2:?--bundle needs a file}"; shift 2 ;;
    --demo)        DEMO=1; shift ;;
    --all-models)  ALL_MODELS=1; shift ;;
    --extras)      EXTRAS="${2:?--extras needs a comma-separated list}"; shift 2 ;;
    --no-media)    MEDIA=0; shift ;;
    --no-ui)       UI=0; shift ;;
    --no-warm)     WARM=0; shift ;;
    --skip-models) MODELS=0; shift ;;
    -h|--help)     sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "setup: unknown option $1 (try --help)" >&2; exit 2 ;;
  esac
done

step() { printf '\n== %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die()  { printf 'setup: %s\n' "$*" >&2; exit 1; }

find_python() {
  local c
  for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null; then
      echo "$c"; return 0
    fi
  done
  return 1
}

OFFLINE=0

# --- 1. offline bundle -------------------------------------------------------------------------
if [ -n "$BUNDLE" ]; then
  step "Installing the offline bundle $BUNDLE"
  PY="$(find_python)" || die "no Python 3.9+ found to unpack the bundle"
  "$PY" tools/bundle.py install "$BUNDLE" --repo "$REPO_ROOT"
  OFFLINE=1
fi

# --- 2. uv -------------------------------------------------------------------------------------
step "Checking uv"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) die "uv is not installed; on Windows run scripts/setup.ps1 instead" ;;
  esac
  if [ "$OFFLINE" = 1 ]; then
    [ -d wheels ] || die "uv is not installed and the bundle has no wheels/ (make it with --wheels)"
    PY="$(find_python)" || die "no Python 3.9+ found to install uv"
    # A distro Python marked externally managed (PEP 668: Debian 12, Ubuntu 23.04+) refuses a
    # plain --user install; --break-system-packages only touches the user site, never the OS.
    "$PY" -m pip install --user --no-index --find-links wheels uv \
      || "$PY" -m pip install --user --break-system-packages --no-index --find-links wheels uv \
      || die "could not install uv from wheels/"
    export PATH="$("$PY" -m site --user-base)/bin:$PATH"
  else
    echo "installing uv with its official installer (https://astral.sh/uv)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
fi
command -v uv >/dev/null 2>&1 || die "uv is still not on PATH; open a new shell and rerun"
uv --version

# --- 3. Python environment ---------------------------------------------------------------------
if [ "$OFFLINE" = 1 ]; then
  step "Python environment, offline (from wheels/)"
  export UV_OFFLINE=1
  if [ -d wheels ] && [ -f wheels/requirements.txt ]; then
    [ -d .venv ] || uv venv --python "$(cat .python-version 2>/dev/null || echo 3.11)"
    uv pip install --no-index --find-links wheels -r wheels/requirements.txt
    uv pip install --no-index --find-links wheels --no-deps -e .
    for clip in wheels/clip-*.whl; do
      [ -e "$clip" ] && uv pip install --no-index --find-links wheels --no-deps "$clip"
    done
  else
    [ -d .venv ] || die "the bundle has no wheels/ and there is no .venv: build the bundle with --wheels"
    warn "the bundle has no wheels/; using the existing .venv as it is"
  fi
  : > .actus-offline          # scripts/run.* read this to stay offline and skip syncing
  export UV_NO_SYNC=1
else
  step "Python environment (uv sync)"
  rm -f .actus-offline
  if [ "$EXTRAS" = "all" ]; then
    uv sync --all-extras
  else
    extra_args=()
    IFS=',' read -r -a picked <<< "$EXTRAS"
    for e in "${picked[@]}"; do extra_args+=(--extra "$e"); done
    uv sync "${extra_args[@]}"
  fi
fi

# --- 4. GUI ------------------------------------------------------------------------------------
if [ "$UI" = 1 ]; then
  step "GUI build"
  if [ -f ui/dist/index.html ]; then
    echo "ui/dist is already built"
  elif command -v npm >/dev/null 2>&1; then
    (cd ui && npm ci && npm run build) || warn "the GUI build failed; the GUI port will show an error page"
  else
    warn "npm not found and ui/dist is missing: build it elsewhere and copy ui/dist here, or install a bundle that has it"
  fi
fi

# --- 5. models, MediaMTX, FFmpeg ---------------------------------------------------------------
if [ "$MODELS" = 1 ]; then
  if [ "$OFFLINE" = 1 ]; then
    step "Verifying the installed assets"
    uv run python tools/fetch_models.py --verify || warn "some assets are missing or differ from their pins"
  else
    step "Fetching models and media binaries"
    fetch_args=()
    [ "$ALL_MODELS" = 1 ] && fetch_args+=(--all)
    [ "$MEDIA" = 0 ] && fetch_args+=(--group perception,voice,open_vocab,coco)
    uv run python tools/fetch_models.py ${fetch_args[@]+"${fetch_args[@]}"} \
      || warn "some assets failed to download; the doctor below says what each one costs"
    if [ "$WARM" = 1 ]; then
      step "Priming the open-vocabulary detector (its CLIP text encoder downloads once)"
      uv run python tools/fetch_models.py --warm || warn "warm-up failed; YOLO-World will try again on first use"
    fi
  fi
fi

# --- 6. preflight ------------------------------------------------------------------------------
step "Preflight"
status=0
uv run python tools/doctor.py || status=$?

if [ "$DEMO" = 1 ] && [ "$status" = 0 ]; then
  step "Starting the SIMULATED replay demo (GUI on http://127.0.0.1:8000)"
  exec bash "$REPO_ROOT/scripts/run.sh" --replay correct
fi

echo
if [ "$status" = 0 ]; then
  echo "setup complete. Next:"
  echo "  bash scripts/run.sh --replay correct     # SIMULATED, needs no camera"
  echo "  bash scripts/run.sh --live               # your camera (see README for the rack)"
else
  echo "setup finished, but the preflight found problems (above)."
fi
exit "$status"
