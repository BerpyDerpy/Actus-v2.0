<#
.SYNOPSIS
  One-command setup on Windows: Python environment, GUI build, models, MediaMTX and FFmpeg,
  then a preflight. The counterpart of scripts/setup.sh.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Demo
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Bundle actus-bundle-windows-amd64-20260930.zip

.NOTES
  -Bundle installs offline from a bundle made by tools\bundle.py (it still needs Python 3.11 on
  the machine). -AllModels adds the optional assets (pose full/heavy). -NoMedia, -NoUi, -NoWarm
  and -SkipModels skip those steps. The network is used only here.
  Afterwards: scripts\run.ps1 --live | --replay correct | doctor
#>
[CmdletBinding()]
param(
  [string]$Bundle = "",
  [switch]$Demo,
  [switch]$AllModels,
  [switch]$NoMedia,
  [switch]$NoUi,
  [switch]$NoWarm,
  [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PATH = "$Root\bin;$env:PATH"

function Step($msg) { Write-Host ""; Write-Host "== $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "warning: $msg" -ForegroundColor Yellow }
function Die($msg) { Write-Host "setup: $msg" -ForegroundColor Red; exit 1 }

# Windows PowerShell 5.1 does not throw when a native command fails; check the exit code.
# The command's output goes to the host, not the pipeline: a function returns everything written
# to its output stream, so without Out-Host any printed line would turn the result into a
# (truthy) array and hide the output too.
function Invoke-Native([scriptblock]$Block) {
  & $Block | Out-Host
  return ($LASTEXITCODE -eq 0)
}

function Find-Python {
  foreach ($c in @("python", "py")) {
    if (Get-Command $c -ErrorAction SilentlyContinue) {
      $probe = @("-c", "import sys; sys.exit(sys.version_info < (3, 9))")
      if ($c -eq "py") { $probe = @("-3") + $probe }
      try {
        & $c @probe 2>$null
        if ($LASTEXITCODE -eq 0) { return $c }
      } catch {
        continue   # e.g. the Microsoft Store python stub
      }
    }
  }
  return $null
}

$Offline = $false

# --- 1. offline bundle ---------------------------------------------------------------------
if ($Bundle -ne "") {
  Step "Installing the offline bundle $Bundle"
  $py = Find-Python
  if (-not $py) { Die "no Python 3.9+ found to unpack the bundle" }
  $pyArgs = @("tools\bundle.py", "install", $Bundle, "--repo", $Root)
  if ($py -eq "py") { $pyArgs = @("-3") + $pyArgs }
  if (-not (Invoke-Native { & $py @pyArgs })) { Die "the bundle did not install" }
  $Offline = $true
}

# --- 2. uv ---------------------------------------------------------------------------------
Step "Checking uv"
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  if ($Offline) {
    if (-not (Test-Path "wheels")) { Die "uv is not installed and the bundle has no wheels\ (make it with --wheels)" }
    $py = Find-Python
    if (-not $py) { Die "no Python 3.9+ found to install uv" }
    $pipArgs = @("-m", "pip", "install", "--user", "--no-index", "--find-links", "wheels", "uv")
    if ($py -eq "py") { $pipArgs = @("-3") + $pipArgs }
    if (-not (Invoke-Native { & $py @pipArgs })) { Die "could not install uv from wheels\" }
    $scripts = & $py -c "import sysconfig; print(sysconfig.get_path('scripts', 'nt_user'))"
    $env:PATH = "$scripts;$env:PATH"
  } else {
    Write-Host "installing uv with its official installer (https://astral.sh/uv)"
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
  }
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Die "uv is still not on PATH; open a new shell and rerun" }
uv --version

# --- 3. Python environment -----------------------------------------------------------------
if ($Offline) {
  Step "Python environment, offline (from wheels\)"
  $env:UV_OFFLINE = "1"
  if ((Test-Path "wheels\requirements.txt")) {
    if (-not (Test-Path ".venv")) {
      $pin = "3.11"
      if (Test-Path ".python-version") { $pin = (Get-Content ".python-version" -TotalCount 1).Trim() }
      if (-not (Invoke-Native { uv venv --python $pin })) { Die "could not create .venv with Python $pin" }
    }
    $ok = Invoke-Native { uv pip install --no-index --find-links wheels -r wheels\requirements.txt }
    if ($ok) { $ok = Invoke-Native { uv pip install --no-index --find-links wheels --no-deps -e . } }
    if (-not $ok) { Die "the offline install failed" }
    foreach ($whl in @(Get-ChildItem "wheels\clip-*.whl" -ErrorAction SilentlyContinue)) {
      $path = $whl.FullName
      Invoke-Native { uv pip install --no-index --find-links wheels --no-deps $path } | Out-Null
    }
  } elseif (-not (Test-Path ".venv")) {
    Die "the bundle has no wheels\ and there is no .venv: build the bundle with --wheels"
  } else {
    Warn "the bundle has no wheels\; using the existing .venv as it is"
  }
  New-Item -ItemType File -Force -Path ".actus-offline" | Out-Null
  $env:UV_NO_SYNC = "1"
} else {
  Step "Python environment (uv sync)"
  Remove-Item ".actus-offline" -ErrorAction SilentlyContinue
  if (-not (Invoke-Native { uv sync --all-extras })) { Die "uv sync failed" }
}

# --- 4. GUI --------------------------------------------------------------------------------
if (-not $NoUi) {
  Step "GUI build"
  if (Test-Path "ui\dist\index.html") {
    Write-Host "ui\dist is already built"
  } elseif (Get-Command npm -ErrorAction SilentlyContinue) {
    Push-Location ui
    $built = (Invoke-Native { npm ci }) -and (Invoke-Native { npm run build })
    Pop-Location
    if (-not $built) { Warn "the GUI build failed; the GUI port will show an error page" }
  } else {
    Warn "npm not found and ui\dist is missing: build it elsewhere and copy ui\dist here, or install a bundle that has it"
  }
}

# --- 5. models, MediaMTX, FFmpeg -----------------------------------------------------------
if (-not $SkipModels) {
  if ($Offline) {
    Step "Verifying the installed assets"
    if (-not (Invoke-Native { uv run python tools/fetch_models.py --verify })) {
      Warn "some assets are missing or differ from their pins"
    }
  } else {
    Step "Fetching models and media binaries"
    $fetch = @()
    if ($AllModels) { $fetch += "--all" }
    if ($NoMedia) { $fetch += @("--group", "perception,voice,open_vocab,coco") }
    if (-not (Invoke-Native { uv run python tools/fetch_models.py @fetch })) {
      Warn "some assets failed to download; the doctor below says what each one costs"
    }
    if (-not $NoWarm) {
      Step "Priming the open-vocabulary detector (its CLIP text encoder downloads once)"
      if (-not (Invoke-Native { uv run python tools/fetch_models.py --warm })) {
        Warn "warm-up failed; YOLO-World will try again on first use"
      }
    }
  }
}

# --- 6. preflight --------------------------------------------------------------------------
Step "Preflight"
$healthy = Invoke-Native { uv run python tools/doctor.py }

if ($Demo -and $healthy) {
  Step "Starting the SIMULATED replay demo (GUI on http://127.0.0.1:8000)"
  & "$Root\scripts\run.ps1" --replay correct
  exit $LASTEXITCODE
}

Write-Host ""
if ($healthy) {
  Write-Host "setup complete. Next:"
  Write-Host "  scripts\run.ps1 --replay correct     # SIMULATED, needs no camera"
  Write-Host "  scripts\run.ps1 --live               # your camera (see README for the rack)"
  exit 0
}
Write-Host "setup finished, but the preflight found problems (above)."
exit 1
