# One-command run on Windows. The counterpart of scripts/run.sh; needs no bash.
#
# Usage (from the repo root, or anywhere):
#   scripts\run.ps1                          # = --live (camera per capture.source)
#   scripts\run.ps1 --live --source 1        # another webcam index, or an rtsp:// URL
#   scripts\run.ps1 --file data\raw\op01_correct_01.mp4
#   scripts\run.ps1 --replay skip_S2         # SIMULATED, no camera or models needed
#   scripts\run.ps1 --offline <take.mp4 | replay:SCENARIO> [--verify-determinism]
#   scripts\run.ps1 doctor [--camera]        # what works, what degrades, what is broken
#
# Everything else goes to `python -m actus.app.pipeline` (see its --help), for example
# --no-stream, --answer yes, --override stream.target_ip=192.168.1.20, --faults.
#
# The mode-to-extras mapping below mirrors scripts/run_demo.sh; keep the two in step.
# After `setup.ps1 -Bundle` (an offline install) this script sets UV_OFFLINE and UV_NO_SYNC.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PATH = "$env:USERPROFILE\.local\bin;$Root\bin;$env:PATH"

if (Test-Path "$Root\.actus-offline") {
  $env:UV_OFFLINE = "1"
  $env:UV_NO_SYNC = "1"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "run: uv not found; run: powershell -File scripts\setup.ps1" -ForegroundColor Red
  exit 1
}

$rest = @($args)
if ($rest.Count -eq 0) { $rest = @("--live") }
$mode = $rest[0]
$tail = @()
if ($rest.Count -gt 1) { $tail = $rest[1..($rest.Count - 1)] }

switch ($mode) {
  { $_ -in "-h", "--help" } {
    Get-Content $PSCommandPath -TotalCount 14 | ForEach-Object { $_ -replace "^# ?", "" }
    exit 0
  }
  "doctor" { uv run python tools/doctor.py @tail }
  "--offline" { uv run --extra perception python tools/run_offline.py @tail }
  "--replay" { uv run --extra ui --extra voice python -m actus.app.pipeline @rest }
  { $_ -in "--live", "--file", "--source" } {
    uv run --extra perception --extra voice --extra ui python -m actus.app.pipeline @rest
  }
  default {
    Write-Host "run: first argument must be --live, --file, --replay, --offline or doctor (got $mode)" -ForegroundColor Red
    exit 2
  }
}
exit $LASTEXITCODE
