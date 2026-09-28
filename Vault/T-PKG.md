# T-PKG: decisions made outside the build plan

2026-09-28 · covers T-PKG (one-command setup and run, the offline bundle, the README with a
hardware table) · read with `build-plan.md` §5 (Wave 3) and §10, `Vault/T-INT.md` and
`Vault/T-PROTO2.md`

The T-PKG card is one line. This file records every decision made while implementing it that the
card does not state, so that T-INT, T-DEMO, T-REVIEW and whoever sets up the demo machine can rely
on them or overturn them on purpose.

**Scope, as agreed:** the harness. Nothing is committed. **No code was run** while writing this
lane (that was the instruction), so nothing below is verified; §0 lists what to run first and what
is most likely to fail.

Status at handover:
- New: `configs/assets.yaml`, `tools/_pkglib.py`, `tools/fetch_models.py`, `tools/bundle.py`,
  `tools/doctor.py`, `scripts/setup.sh`, `setup.ps1`, `run.sh`, `run.ps1`, and four test files
  (`tests/test_pkg_{lib,fetch,bundle,doctor}.py`, 75 tests).
- Rewritten: `README.md` (T-PKG owns it, `Vault/Wave0.md` §3).
- One file outside T-PKG's own was edited: `.gitignore`, four lines appended (§6).
- No dependency was added and no contract changed. `uv.lock` is untouched.

---

## 0. Status against the card, and what to run first

| Deliverable | Status |
|---|---|
| One-command setup | Written: `scripts/setup.sh` and `setup.ps1`. Installs uv, builds the environment and the GUI, fetches models and binaries, primes YOLO-World, runs the doctor. **Unrun.** |
| One-command run | Written: `scripts/run.sh` and `run.ps1`. The modes are `run_demo.sh`'s, plus `doctor` and the offline handling. **Unrun.** |
| Offline bundle (weights, Vosk and Kokoro models, MediaMTX) | Written: `tools/bundle.py make / info / verify / install`, hash-verified end to end, with optional Python wheels. **Unrun.** |
| README with a hardware table | Written. The table's FPS column is **"not measured"** on every row, on purpose (§5). |

Run these, in this order:

```sh
uv run ruff check . && uv run ruff format .          # formatted by hand; expect small diffs
uv run pytest tests/test_pkg_lib.py tests/test_pkg_fetch.py tests/test_pkg_bundle.py tests/test_pkg_doctor.py
bash -n scripts/setup.sh && bash -n scripts/run.sh    # shell syntax only
powershell -NoProfile -Command "[void][scriptblock]::Create((Get-Content scripts\setup.ps1 -Raw)); [void][scriptblock]::Create((Get-Content scripts\run.ps1 -Raw))"
uv run python tools/fetch_models.py --list            # no network
uv run python tools/fetch_models.py --dry-run         # no network: what setup would download
uv run python tools/doctor.py                         # a real preflight of this machine
chmod +x scripts/*.sh                                 # written on Windows: no executable bit
```

Then, on a connected machine, the real thing: `bash scripts/setup.sh`, then
`uv run python tools/fetch_models.py --lock`, then `tools/bundle.py make --wheels`, then install
that bundle on a second machine with the network off.

Most likely to fail, because each rests on something I could not check by running it:

1. **The download URLs.** Written from memory of each project's release layout, never fetched.
   In rough order of doubt: the OpenAI CLIP URL on `openaipublic.azureedge.net` (and the SHA-256 it
   embeds); `yolov8s-worldv2.pt` and `yolo11n.pt` under the `ultralytics/assets` `v8.3.0` release
   (both have an ultralytics-downloader fallback, §3); BtbN's `…/releases/download/latest/…` FFmpeg
   aliases; the MediaMTX asset names (`linux_arm64v8`, not `arm64`); the Kokoro
   `model-files-v1.0` release; the MediaPipe `float16/1` model paths. A wrong URL is a 404 or a
   hash mismatch, never a wrong file. `fetch_models.py` reports each failure and carries on.
2. **`uv` flags in `bundle.py make --wheels`.** `uv export --frozen --all-extras --no-dev
   --no-emit-project --no-hashes -o` and `uv tool run --python 3.11 pip download --only-binary=:all:`
   are from memory. `--only-binary` will also refuse a package that ships no wheel for the
   platform (`lap` is the one to watch on Windows); that is the intended failure, at bundle time
   rather than on the offline machine.
3. **The offline install (`setup.sh --bundle`).** `uv pip install --no-index --find-links wheels -r
   wheels/requirements.txt` on a `uv export` file with platform markers, then `-e .` with build
   isolation resolving `hatchling` from `wheels/`. This is the least certain path in the lane.
4. **`test_manifest_paths_match_the_default_config`** and
   **`test_the_second_protocol_overlay_points_at_a_manifest_asset`**: they fail if
   `configs/default.yaml` or `configs/sample_mixing.yaml` change a path. That is their job.
5. **`test_cli_json_report_has_the_documented_shape`** runs the doctor on the real checkout, so it
   depends on the environment (it only checks the shape, and that the exit code is 0 or 1).
6. Anything with `Path.as_uri()` on Windows (`file:///D:/…`): `urllib` should read it, and the
   tests use it in place of a network.
7. Style: `ruff format` was not run.

## 1. Files and entry points

| File | What it holds |
|---|---|
| `configs/assets.yaml` | The manifest: every downloadable or trained file, its destination, URL, group, licence. |
| `configs/assets.lock.json` | Not created. `fetch_models.py --lock` writes it (§2). Commit it. |
| `tools/_pkglib.py` | Hashing, pins, the downloader, safe archive extraction, the MediaMTX/FFmpeg pickers. Stdlib only. |
| `tools/fetch_models.py` | Fetch, `--verify`, `--lock`, `--list`, `--warm`, `--from DIR`, `--dry-run`. |
| `tools/bundle.py` | `make`, `info`, `verify`, `install`. |
| `tools/doctor.py` | The preflight. |
| `scripts/setup.sh`, `setup.ps1` | The one-command setup. |
| `scripts/run.sh`, `run.ps1` | The one-command run. |

```sh
bash scripts/setup.sh --demo                    # everything, then the SIMULATED replay
bash scripts/setup.sh --bundle FILE.zip         # offline
bash scripts/run.sh --live | --replay correct | --file take.mp4 | doctor
uv run python tools/fetch_models.py --list      # / --verify / --lock / --all / --group voice
uv run python tools/bundle.py make --wheels     # / info / verify / install
uv run python tools/doctor.py --strict          # warnings fail: for a packaging check
```

## 2. The asset manifest and the pins

**Nothing is pinned by hand.** A hash typed in without downloading the file is worse than none, and
I could not download anything. So the trust model is trust-on-first-use with a committed lock:

- `fetch_models.py` verifies a download against, in order, the manifest's `sha256`, a hash embedded
  in the URL (`sha256_from_url`, used for CLIP, whose URL folder *is* its SHA-256, which is how
  OpenAI's own loader checks it), then `configs/assets.lock.json`. With none of them the first
  download is accepted and reported as `unpinned`.
- `fetch_models.py --lock` hashes what is on disk and writes the lock. **Commit that file**; from
  then on every fetch, `--verify`, and `bundle.py` (through the same pins) detect a changed or
  corrupted file.
- A `zip_dir` asset (Vosk) is pinned by a **tree hash**: the SHA-256 of the sorted relative paths and
  file hashes. The archive itself is not pinned, because the archive is not what runs.
- Trained models (`kind: local`: the custom detector, HOI, temporal) are **never pinned**. They
  change on every retrain, so a pin would only produce false alarms. They are listed so the doctor
  reports them and the bundle carries them.
- **MediaMTX and FFmpeg come from "latest".** A pin in the lock is per platform (the asset id has the
  OS and architecture in it). A newer release than the lock's is reported as "lock is stale", not
  rejected, because "latest" is the point. `--mediamtx-version v1.x.y` pins a release.

## 3. Decisions the card leaves open

**Media binaries in Python, not `setup_media.sh`.** T-IO's script does not handle Windows (no
`unzip` in Git Bash, no FFmpeg URL) or Linux ARM (its `arm64` pattern misses MediaMTX's
`arm64v8`). `fetch_models.py` does MediaMTX and FFmpeg for Windows, Linux and macOS, so the same
command works on the dev machine. FFmpeg is BtbN's GPL build on Windows and Linux (it has
`libx264`, which the streamer needs; the doctor checks) and `brew install ffmpeg` on macOS.
`setup_media.sh` is left untouched and still works standalone. **The two overlap; prefer
`fetch_models.py`.**

**YOLO-World has hidden run-time downloads, and this lane found them.** `OpenVocabDetector` calls
`model.set_classes(...)`. In ultralytics that loads OpenAI's CLIP text encoder through the `clip`
package, which (a) is installed on first use with `pip install git+https://github.com/ultralytics/
CLIP.git` and (b) downloads `ViT-B-32.pt` into `~/.cache/clip`. That contradicts "everything runs
offline at runtime". (This is from my memory of ultralytics' internals and is unverified.) So:
- `configs/assets.yaml` lists the CLIP weights (`~/.cache/clip/ViT-B-32.pt`), and the bundle carries
  them under `home/.cache/clip/`, the only place outside the repo that a bundle may write.
- `fetch_models.py --warm` installs the `clip` package (`uv pip install`, needs `git`) and then runs
  `YOLO(weights).set_classes(prompts)` exactly as the detector does, so the encoder is cached.
- `bundle.py make --wheels` builds a wheel of the `clip` package (`pip wheel`) plus its small
  dependencies, so an offline machine can install it.
- **Wrinkle:** the `clip` package is not in `uv.lock`, so a later plain `uv sync` (which is exact)
  uninstalls it. `uv run` is inexact and leaves it. The README says to re-run setup or `--warm`.
- **Proposal for T-INT (it owns `pyproject.toml` changes and the lock):** add `clip @
  git+https://github.com/ultralytics/CLIP.git` to the `perception` extra and run `uv lock`. That
  removes the wrinkle and makes `--warm`'s install step unnecessary. I did not do it because it
  needs `uv lock`, and nothing was to be run.

**YOLO-World's weights live at the repo root.** `detector.open_vocab_weights` is the bare file name
`yolov8s-worldv2.pt`, which ultralytics resolves against the working directory (`run_demo.sh` runs
from the repo root). `tools/autolabel.py` uses the same bare name. Moving it under `models/` would
mean editing `default.yaml` (T-PERC's section) and `autolabel.py` (T-DATA's file), so it stays,
with `/yolo*.pt` added to `.gitignore` (§6). The bundle allows exactly a root-level `yolo*.pt`.

**The second protocol's detector is a bundled asset.** `Vault/T-PROTO2.md` §2 asks T-PKG to bundle
the stock COCO `yolo11n.pt`. It is asset `yolo11n_coco` (group `coco`, in the default set), at
`models/coco/yolo11n.pt`, which is what `configs/sample_mixing.yaml` points at. A test compares
the two paths.

**ultralytics-downloader fallback.** For the two ultralytics assets, if the direct URL fails
(`DownloadError`, not a hash mismatch) `fetch_models.py` asks
`ultralytics.utils.downloads.attempt_download_asset` to fetch the file to the same path. That
survives a release-tag change I could not check. It is skipped under `--from`.

**`--from DIR`.** Reads every asset by file name from a folder instead of the network: a LAN mirror,
a USB stick, and how the tests stand in for the internet. It is the manual offline path when a
bundle is overkill.

**Doctor: WARN, not FAIL.** `Vault/T-INT.md` §4 makes every sink and model degrade on purpose, so a
missing optional piece is a WARN that states what it costs (`loses:`) and how to fix it (`fix:`). A
FAIL is only what stops everything (core modules, `cv2.aruco`, an invalid protocol, a missing
`rack.yaml`). The verdict is `READY`, `READY (DEGRADED)` or `NOT READY`. Exit status is 1 only for
`NOT READY`, or with `--strict` for `READY (DEGRADED)` too. Trained models absent are INFO. A
placeholder `configs/rack.yaml` (`measured: false`) is a WARN, because it makes S3 and S5
unverifiable (`Vault/T-INT.md` §7).
- The doctor reads the *config*, not the manifest, for what the pipeline needs, so an overlay
  (`--config configs/sample_mixing.yaml`) is checked against its own paths. Where a configured
  file matches a manifest asset's destination, the fix names that asset.
- It never opens the camera unless `--camera` is passed, and it does not import ultralytics or
  mediapipe (`find_spec` only), so it is fast and cannot hang on a bad driver.

## 4. The offline bundle

- **Format:** one `.zip`: `bundle.json` (kind, schema, creation time, platform, Python, git commit,
  and every file's path, SHA-256, size and mode) plus the files under `payload/`. A `<bundle>.sha256`
  sidecar covers transfer. `tools/bundle.py info` and `verify` read it.
- **Contents:** all of `models/` (weights and trained models, with their cards), `bin/`,
  `ui/dist`, `configs/assets.lock.json`, the CLIP weights, a root `yolo*.pt`, and with `--wheels`
  `wheels/` (every wheel of the locked environment, the build backend, `uv`, and the `clip` wheel).
  **Not the source code**: clone or copy the repo first, then install the bundle into it. `make` prints
  notes about anything an offline machine would miss (no `ui/dist`, no lock, an asset not fetched, FFmpeg
  on PATH rather than in `bin/`); they are also stored in the manifest.
- **Install is all or nothing.** Every member is hashed while it is unpacked into `<name>.part`; a
  file is renamed into place only after all of them verified, so a corrupt or altered bundle changes
  nothing. Members not listed in `bundle.json` are never extracted, and `verify` reports them.
- **Where a bundle may write:** `models/`, `bin/`, `ui/dist/`, `wheels/`, `configs/assets.lock.json`, a
  root-level `yolo*.pt`, `~/.cache/clip/`. Anything else, an absolute path, `..`, a backslash or a
  drive letter, is refused when the manifest is read. Tested.
- **Platform-specific parts.** `bin/` and `wheels/` are platform-specific; a bundle from another
  platform is refused unless `--portable-only` (models and GUI only). Wheels are for the Python that
  `.python-version` names (3.11), fetched with `uv tool run --python 3.11 pip download`.
- **Prerequisite the bundle cannot remove:** Python 3.11 on the offline machine. `uv` cannot fetch it
  offline. `uv` itself is bootstrapped from the bundle's wheels.
- **Size.** Models are about 0.9 GB, from the sizes in `configs/assets.yaml`. Wheels add 1-4 GB
  (an estimate: PyTorch's Linux wheels dominate, and I did not build one).
- `install` and everything it imports are stdlib-only and written for Python 3.9+, so they run on a bare
  system Python before any environment exists. `make` needs the project environment (and for
  `--wheels`, `uv` and the network).
- **Offline runs.** `setup.* --bundle` writes an `.actus-offline` marker; `scripts/run.*` then export
  `UV_OFFLINE=1` and `UV_NO_SYNC=1`, so `uv run` does not try to reach the network or re-sync an
  environment installed from wheels. `run_demo.sh` is not edited; the environment variables reach it.

## 5. README, and the hardware table

The README states what has and has not been verified, in a table, rather than a number.
**Every FPS cell says "not measured".** The only measured number in the repo is T-INT's harness cost
(about 7.5 ms per frame on the replay, on the Windows dev machine, no neural stages), and it is
labelled as that. The table's rows are the machines the plan names (the Windows dev machine, the
M2, the RTX 4070 laptop, Ubuntu CI, the finale's Jetson Orin), each with its status. Someone fills a
cell by measuring: `python -m actus.perception.detector --seconds 10`, then `run.sh --live` and the GUI
stats strip. The README also carries the plan's honesty list (§9), the licences (§10), and this
lane's own caveats.

## 6. Cross-lane edits and what other lanes should know

- **`.gitignore`** (P0's file): appended `/wheels/`, `/.actus-offline`, `/yolo*.pt`. `/dist/` was
  already there, so bundles land in an ignored place. `configs/assets.lock.json` is deliberately
  **not** ignored.
- **`scripts/run_demo.sh`, `setup_media.sh`, `configs/default.yaml`, `pyproject.toml`: not edited.**
- `scripts/run.ps1` **duplicates `run_demo.sh`'s mode-to-extras mapping**, so that Windows needs no
  bash. If T-INT changes a mode or an extra, change both. It is noted at the top of the file.
- **T-INT / T-DEMO:** the demo machine's setup is `bash scripts/setup.sh` (or `-Bundle` offline) then
  `bash scripts/run.sh --live`. `doctor` is a good thing to run on camera at the start of the
  walkthrough: it shows what the machine can and cannot do, and it never lies about it.
- **T-REVIEW:** the packaging tools are stdlib-heavy and shell out. Look hardest at `bundle.py
  install` (the trust boundary), the `--wheels` commands, and the PowerShell scripts (5.1 does not
  throw on a failed native command, which is why `Invoke-Native` checks `$LASTEXITCODE`).

## 7. Not done, and known limits

- **Nothing was run.** See §0.
- **CI:** `.github/workflows/ci.yml.disabled` stays disabled. These tests need no network and no
  optional extra, and would run there.
- **Bundle signing.** The bundle is hash-verified against its own manifest, which detects corruption
  and a partial edit, not a hostile bundle with a rewritten manifest. Transfer integrity is the
  `.sha256` sidecar, which the recipient must check against a hash they got another way. There is no
  signature.
- **No cross-platform bundle.** One bundle per OS and architecture.
- **Python is not bundled.** Nor is `npm`/Node: the GUI is bundled built.
- **`setup.sh` on native Windows** refuses if `uv` is missing and points at `setup.ps1`; it works in
  Git Bash when `uv` is already installed.
- **The first `uv sync` is large.** PyTorch arrives with `ultralytics`; setup does not try to select a
  CPU-only build.
- **Hardware table cells are empty** by design until someone measures them (§5).
- **F-EDGE** (the Jetson build and FPS table on 2-3 machines) is a finale task and untouched.
- `tools/fetch_models.py --warm` prints and returns a failure rather than raising, so `setup` treats a
  failed warm-up as a warning: YOLO-World will simply try the same downloads on first use, online.
