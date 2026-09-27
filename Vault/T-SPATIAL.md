# T-SPATIAL: decisions made outside the build plan

2026-09-27 · covers T-SPATIAL (rack frame, zones, world state, predicates, marker sheets) · read
with `build-plan.md` §4–5, `Vault/Wave0.md` and `Vault/T-ENGINE.md`

The T-SPATIAL card specifies the lane in outline. This file lists every decision made during
implementation that the card does not state, so that the lanes consuming spatial output (T-INT,
T-UI, T-HOI-TEMP, T-EVAL, T-PROTO2) can rely on them or overturn them on purpose.

Status at handover: 81 T-SPATIAL tests pass across 5 files, and the full suite (P0 + T-ENGINE +
T-SPATIAL) passes: 290 tests. `ruff check` and `ruff format --check` are clean. Nothing is
committed. No dependency was added. The only file outside `actus/spatial/*`, `tests/test_spatial*`
and `tools/make_markers.py` that was touched is the `rack:` section of `configs/default.yaml`
(two new keys, see §2).

Both "done when" tests pass:
- a camera rolled 90°/180° gives the same rack positions within 1 cm. This is tested on pure
  geometry, on real ArUco detection in perspective-rendered boards (physical roll), and on
  `cv2.rotate`d frames (software rotation);
- a held object stays `held` through a 1.5 s dropout, both when only the object vanishes and in a
  total blackout (hands gone too).

---

## 1. Files and entry points

| File | What it holds |
|---|---|
| `actus/spatial/zones.py` | `Zone` (signed distance, `contains`, soft `membership`, `translated`), `as_polygon`, `zones_from_protocol`. |
| `actus/spatial/rack_frame.py` | `RackGeometry` (reads `configs/rack.yaml`), `CameraSpec`, `RackFrameEstimator`, `img_to_rack` / `rack_to_img` / `apply_homography`, `torso_vec_rack`, `rotate_camera_matrix`, `draw_overlay`, and a live board-check CLI. |
| `actus/spatial/world_state.py` | `SpatialParams`, `HandInput`, `WorldModel` (the per-object filter), `WorldSnapshot` / `ObjectState`, `parse_vocabulary`, `expm`. |
| `actus/spatial/predicates.py` | `EvidenceModel` (world → `Predicates` + `StepEvidence`), `calibrate`, `fit_temperature`. |
| `actus/spatial/estimator.py` | **`SpatialEstimator`**, the one-call-per-frame façade for T-INT, and `SpatialFrame`. There is also a CLI that runs the lane on the replay. |
| `tools/make_markers.py` | Printable SVG marker sheets and a board-layout plan. |
| `tests/test_spatial_rack.py`, `_world.py`, `_evidence.py`, `_markers.py`, `_engine_chain.py` | See §7. |

`estimator.py` is a file the card doesn't list. The card names four files, and the façade that
ties them together didn't belong in any one of them.

```sh
uv run python -m actus.spatial.estimator --scenario skip_S2       # lane on the SIMULATED replay
uv run python -m actus.spatial.rack_frame --source 0              # live board check (H-PROPS)
uv run python -m actus.spatial.rack_frame --image photo.jpg       # solve one image, write overlay
uv run python tools/make_markers.py                               # -> data/markers/*.svg
uv run pytest tests/test_spatial*.py
```

**API for T-INT**

```python
est = SpatialEstimator.from_config(cfg)  # cfg.rack + cfg.spatial + cfg.protocol.path
sf = est.update(
    frame_idx,
    t,
    frame=packet,  # or rack=RackFrame if solved elsewhere
    detections=dets,
    hands=hands,
    pose=pose,
    hoi={"red": 0.9},
)
for topic, msg in sf.messages():  # rack, predicates, evidence×N, stats
    bus.publish(topic, msg)
```

- `update` is single-threaded and stateful. Feed it frames in order, with `t` non-decreasing.
- `sf.detections`, `sf.hands` and `sf.pose` are the inputs with rack coordinates filled in. Publish
  them on `detections`, `hands` and `pose` if the UI should get the rack-frame versions. They are
  not in `messages()`, because T-PERC may already publish the originals on those topics.
- `detections=None` means "the detector didn't run this frame": the filter only predicts. An
  empty `DetectionSet` means "ran and saw nothing".
- `hands=None` means "no hand model". An empty `HandSet` means "the model ran and saw no hand".
  The two are treated very differently (§4).
- `hoi` is `{object: p(holding)}`. The HOI wrapper from T-HOI-TEMP should reduce its (hand,
  object) pairs to a max over hands.

## 2. Rack frame (`rack_frame.py`)

- **The plane map is a direct homography, not PnP.** `H_img_to_rack` is fitted to every marker
  corner: RANSAC first (threshold `max_reproj_err_px`), then least squares on the inliers. It
  doesn't use the camera intrinsics, so rack-plane positions are exact even though
  `configs/rack.yaml` has no calibration (`calibrated: false`). `reproj_err` is the RMS of this
  fit.
- **Validity** needs at least `min_markers` *inlier* markers (a marker counts only if all 4 of its
  corners are inliers) and `reproj_err ≤ max_reproj_err_px`. Unknown ids are ignored. If an id is
  detected twice, the larger instance is kept.
- **3D pose**: `rvec`/`tvec` come from IPPE PnP on the four smoothed board corners, using the
  calibrated `K` if present, otherwise one built from `approx_hfov_deg` across the long side
  (square pixels, centred principal point). `tvec` must have z > 0. The PnP error against the raw
  corners is in `estimator.last_info["pnp_err_px"]`, not in the contract.
- **Lens distortion is ignored for the plane map** and used only for PnP. The reprojection check
  guards against the resulting error; with a typical webcam it is small. If a strongly
  distorting lens is ever used, undistort first. That would be a change to the meaning of
  `H_img_to_rack` and needs the integrator.
- **Software rotation.** `FramePacket.rotation_applied` is honoured for the intrinsics:
  `CameraSpec.matrix` rotates a calibrated `K` and its tangential distortion (tested against
  `cv2.rotate` + `projectPoints`). The homography needs nothing special: the frame is re-solved
  on whatever image comes in, which is what makes the rack frame orientation-agnostic.
- **Smoothing** filters where the four board corners land in the image, with the time constant
  `smooth_tau_s` (0.08 s, taken from the real `t`, not per frame). The more the corners jump,
  the faster it follows, and a jump beyond `snap_px` (12 px) snaps straight to the new pose. A
  camera rolled 180° mid-run is therefore correct on the very next frame. A test shows jitter
  down by more than 40%.
- **Hold.** When no fresh solve is possible, the last good pose is returned `valid` with
  `age_frames` = frames since the last *fresh* solve. The hold lasts `hold_frames` (45) after the
  last frame on which it was **confirmed**. Confirmation is a fresh solve, or at least one
  still-visible known marker agreeing with the held pose (≤ 2× the error limit). So an arm
  covering two markers keeps the frame alive indefinitely, while a full blackout expires it. A
  visible marker that **disagrees** (> 3× the limit, i.e. the camera moved) drops the hold
  immediately. `last_info["status"]` is one of `fresh`, `held`, `expired`, `contradicted` or
  `invalid`.
- An invalid `RackFrame` has `H_img_to_rack`, `rvec` and `tvec` all `None`.
- New keys in the `rack:` section of `configs/default.yaml`: `smooth_tau_s: 0.08` and
  `snap_px: 12.0`.
- **Marker corner convention:** each marker is printed *upright* on the board, and its OpenCV
  corners TL, TR, BR, BL sit at centre ± size/2 in rack x/y. The marker sheets draw a red arrow
  on each marker's top edge for this reason.

## 3. Zones (`zones.py`)

- Soft membership is `Φ(d / pos_sigma_m)`, where `d` is the signed distance to the polygon
  boundary (positive inside) and `pos_sigma_m` is 0.02 m. A box on a zone edge reads about 0.5,
  which lands in the engine's ask band instead of flickering between 0 and 1. A test pins this
  down.
- 2-point zones are rectangles and 3+ points are polygons, exactly as the protocol schema says.

## 4. World state (`world_state.py`)

**Vocabulary.** The vocabulary is read from the protocol's predicates. Any other form raises
`ValueError` at construction, listing the supported forms.

| Predicate | Meaning |
|---|---|
| `<obj>.held` | in a hand |
| `<obj>.in_zone_<Z>` | resting in protocol zone Z |
| `<obj>.in_<container>` | inside a container (`red.in_outer` makes `outer` a container) |
| `<container>.open` | the container is open |

- Declared objects that are not containers are "movables". An object can't be both (e.g.
  `outer.held` would need a movable container, which is not supported yet).
- A container named by `in_<c>` must be declared in `objects:` if the protocol has an `objects:`
  block.
- Every combination is emitted, a superset of the protocol's predicates, as the replay does:
  `red.in_zone_B`, `yellow.in_zone_A` and so on.
- **T-PROTO2 note:** a cup/bottle/book protocol works as long as it uses these forms (a test
  covers a container-less protocol). Something like `book.open` makes `book` a container whose
  open state is evidenced only by `<cls>_open` / `<cls>_closed` detector classes (below). With
  plain COCO `book` it would stay at its prior.

**Detector classes → evidence**
- A movable object's classes are `objects.<obj>.classes`, or `[obj]` if none are given.
- For a container, a class ending in `_open` or `_closed` is open/closed evidence. A class named
  `lid` or ending in `_lid` is a lid: a lid lying more than 3 cm outside the footprint is open
  evidence, and one on it is weak closed evidence.
- Association: per object, the best detection of its classes by conf × Gaussian(distance to last
  position, 15 cm), with a small bonus for the same `track_id`.

**Container footprint**
- From `configs/rack.yaml: outer_box_footprint` when the protocol has exactly one container.
  Otherwise use `spatial.footprints: {name: polygon}`. A container with neither gets a 25 cm
  square around its first detection.
- The footprint re-centres on the container's detections (conf ≥ 0.4, τ = 0.5 s). It snaps on
  first sight.

**The filter: a per-object input-output HMM** (the full rationale is in the module docstring)
- States: `in_<container>`, `held`, `zone_<Z>`, `elsewhere`.
- `location` is the argmax state, or **`unknown`** when no state reaches 0.5. "Unknown" is a
  *label* of the belief, not a state. There is no "gone": an object that vanishes keeps its
  belief.
- **Transitions are hand-gated.**
  - Picking up (`X → held`) needs p(hand near the object), and an open container.
  - Putting down (`held → X`) needs the object's last-seen position, or the position of the
    hand carrying it, to be over X, *and* the hand to have let go (1 − p(near)).
  - With the hand model running but no hand visible, both gates are 0.1: a blinded camera does
    not move objects. With no hand model at all, both are 0.5.
  - A 0.02/s correction rate connects every pair of states.
  - Transition probabilities use a matrix exponential of the real `dt` (capped at 5 s).
- **Emissions**
  - Detection position vs. soft zone/footprint membership (a box seen "inside" a closed
    container is near-impossible).
  - Speed over a 0.25 s window: only `held` explains motion, and a still object weakly argues
    against `held` (0.4).
  - A visible hand far from a detected object argues against `held` (0.15). A hand model that
    sees no hand gives 0.6.
  - **Miss likelihoods:** resting in view 0.1; held 0.35 with a hand visible or 0.6 without; in
    an open container 0.5; in a closed container 0.98. All are discounted by scene visibility.
  - HOI p(holding), if given.
- **Frame-rate independence:** each frame's likelihoods are tempered by `min(1, dt / 0.2 s)`.
  Evidence accumulates per second, and consecutive frames are not treated as independent. A test
  checks 10/15/30/60 fps give the same result.
- **`initial` prior:** a flat weight of 0.2 for any state contradicting `initial:` (not
  compounded per contradicted fact), so the first clear observations overturn a wrong prior
  within a frame or two. The normal case doesn't depend on the prior: a closed box makes "not
  detected" likely only for `in_outer`.
- **Visibility** = (detected + 0.5) / (expected detections + 0.5), where expected comes from the
  beliefs and the miss likelihoods; × 0.8 for a held rack frame and × 0.3 for an invalid one;
  smoothed with τ = 0.3 s. Everything expected in view vanishing drops it within about 1 s.
- **Hand points:** 8 keypoints per hand (wrist, 4 MCP knuckles, 3 fingertips). Near = logistic of
  the rack distance (0.5 at 8 cm, width 2 cm), or 0.9 if any keypoint is inside the object's bbox
  grown by 20% (this works without a rack frame).
- `HandObs.landmarks_rack` is the **intersection of the pixel ray with the rack plane, z = 0**.
  It is not true hand depth, because HandObs has no hand world landmarks.

## 5. Predicates and step evidence (`predicates.py`)

- **Reported predicates:** belief → per-predicate temperature (`spatial.temperatures`, identity
  by default) → `v·p + (1−v)·0.5` (the same regression the replay applies) → clip to
  [0.01, 0.99]. The internal belief stays sticky; only the report goes to "don't know" when blind.
  This is what `Vault/T-ENGINE.md` §5 asked for.
- **`p_complete` (the "rule" source)** = min(armed, P(post), handled):
  - **armed** = the running max of a *sustained* `min(P(pre), 1 − P(post)) × support`.
    "Sustained" means its minimum over the last `arm_dwell_s` (0.25 s). `support` decays
    (`arm_memory_s`, 5 s) with the time since the pre-objects were last observed: detected, or
    for an object believed inside a container, a hand inside the open container's footprint.
    Together these mean a pre-state asserted only by `initial:`, or surviving only a frame or two
    before observations overturn it, cannot arm a step. A test shows a wrong `initial:` can't
    fabricate S2/S3: they stay below the ask band with no hand model, and below 0.1 with one,
    because then `held` is ruled out.
  - **handled** (steps with `requires_hoi`) = running max of min(armed, P(obj.held)).
  - **Un-latching:** once `rule ≥ 0.8` and then P(post) ≤ 0.2, both latches reset. Evidence falls
    as soon as the post-state is undone, and a redo must re-arm and be handled again (the engine's
    unexpected-alert relies on this).
- The same regression and clip are then applied to `p_complete`.
- `sources`: `armed`, `post`, `handled` (for `requires_hoi` steps), `rule` (= `p_complete`) and
  `visibility`. All are probabilities, as the contract requires. When T-HOI-TEMP fuses in `hoi`
  and `temporal`, it should build a new `StepEvidence` with the fused `p_complete` and these
  sources plus its own.
- **Prior-free, tested:** the evidence stream is bit-identical under a different `order:`.
- `fit_temperature(probs, labels)` is an NLL grid search for F-CAL and T-EVAL (it recovers T = 3
  on synthetic overconfident data).

## 6. Body relative to the rack

- `torso_vec_rack` = unit (mean shoulders − mean hips) from the MediaPipe world landmarks,
  rotated into the rack frame with the PnP rotation (`R.T @ v`). It is `None` without a valid
  rack pose, or if any of those 4 landmarks has visibility < 0.3.
- **Assumption about T-PERC (to confirm):** `PoseObs.world_landmarks` are MediaPipe's raw output,
  i.e. in the axes of the image MediaPipe actually saw, rotated by `rotation_used`. Spatial
  therefore undoes `rotation_used` first. If T-PERC rotates world landmarks back itself, set
  `spatial.world_landmarks_rotated_back: true`. Both paths are tested. The contract comment
  ("MediaPipe hip-centred world coordinates") doesn't say which, so **the integrator should pin
  it down in the contract docstring.**
- MediaPipe's world axes are taken to follow the camera image: x right, y down, z away. This is
  approximate by nature (a learned 3D estimate), so the torso vector is for display and logging.
  No step logic uses it.

## 7. Verified behaviour

- **Rack** (`test_spatial_rack.py`, 35 tests):
  - rolls of 0/90/180/270/37° give sub-mm error with exact corners and < 1 cm with 0.4 px corner
    noise, and the recovered R, t match;
  - rendered boards give < 1 cm for physical rolls and for `cv2.rotate` 90/180/270;
  - RANSAC rejects a 40 px-displaced marker;
  - the hold, expiry, partial-marker extension and contradiction cases behave as in §2;
  - smoothing reduces jitter, and a 180° roll mid-run snaps;
  - the torso vector holds under camera roll × TTA rotation.
- **World** (`test_spatial_world.py`, 18 tests):
  - held through a 1.5 s dropout, both object-only and total blackout (P(held) ≥ 0.87
    throughout);
  - an object vanishing into the open box → `in_outer`;
  - a lost detection with the hand far away is not invented as `held`;
  - a zone edge gives ≈ 0.5;
  - the box opens only on evidence;
  - the footprint follows the container;
  - HOI drives `held`; visibility collapses when the view does; the vocabulary errors fire.
- **Evidence** (`test_spatial_evidence.py`, 13 tests):
  - on the replay, with the rack solved by ArUco from the rendered SIMULATED frames, projected
    positions match the replay's truth within 1 cm, and steps cross `p_enter` in S1…S6 order,
    never before the true step starts;
  - `skip_S2`: S2 and S3 never exceed 0.5;
  - `S4_before_S2` has the right order;
  - `unexpected` un-latches and re-latches S2 and S3;
  - occlusion regresses S3 into 0.3–0.7 with visibility < 0.4, then it verifies;
  - prior-freeness, a wrong `initial:`, calibration, hand/pose projection and `from_config` are
    all covered.
- **Chain into the real engine** (`test_spatial_engine_chain.py`, 8 tests; skipped where
  `actus.protocol.engine` is absent): for `correct`, `skip_S2`, `S4_before_S2` and `timeout` over
  2 seeds, the engine's event sequence from spatial's evidence is *identical* to its sequence from
  the replay's stand-in evidence. This was also checked by hand with the rack solved from
  rendered pixels, for seeds 0–1.
- **Markers** (`test_spatial_markers.py`, 7 tests): sheet cells decode to the right id through
  `cv2.aruco`; the SVG is true size (A4 = 210 mm, marker = 60 mm) and well-formed.

## 8. Known divergences from the replay's stand-in evidence

Both come from the replay publishing **no hands** and moving objects on a 0.4 s exponential creep.
Without hands, spatial reads "still creeping" as "still carried".

- **`unexpected`:** the red box rests in Zone A for only about 2 s. Spatial's S3 evidence crosses
  0.8 about 1.6 s after the placement: the creep keeps it "moving" for about 0.9 s, then release
  has to be inferred without a hand. It falls again before the engine's 0.8 s dwell completes. The
  engine therefore never verifies the first S3, and the return is not "unexpected". It just sees
  S2/S3 completed once. With hand input, release is detected about 0.4 s after the hand withdraws
  (`test_spatial_world.py`).
- **`occluded`:** the yellow box is picked up *during* the blackout. S4's `requires_hoi` handling
  is not observed until the S5 carry, so the engine adds an `ask S4` (answered "yes" → S4
  `operator_confirmed`). That is the honest outcome for an unobserved step.
- `unsure_S3` forces only the replay's *predicate stream* to 0.62. Its detections still put the
  box mid-zone, so spatial (which works from detections) verifies S3. The boundary case is tested
  separately with synthetic geometry.
- **Proposal for P0 / T-INT:** have `ReplaySource` also emit a synthetic `HandSet` (it already
  renders a hand disc while an object is held) and give placements a short deceleration instead
  of the exponential creep. The whole chain could then be exercised on `unexpected` and
  `occluded` too.

## 9. Contract proposals (for the integrator; not applied)

- **World-state message.** There is no topic for object locations. For now `SpatialFrame.stats()`
  flattens them into `Stats(source="spatial")` on `stats`, with keys:
  - `visibility`;
  - `rack.state`, `rack.markers`, `rack.age_frames`, `rack.reproj_px`;
  - `<obj>.loc`, `<obj>.x`, `<obj>.y`, `<container>.p_open`;
  - `torso.x`, `torso.y`, `torso.z`.

  This is enough for the UI's rack mini-map. A proper `WorldState(frame_idx, t, objects: {name:
  {location, pos, belief}}, footprints, visibility)` on a `world` topic would be cleaner.
- **`PoseObs.world_landmarks` axes** under rotation TTA (§6): state it in the contract docstring.
- **`RackFrame.reproj_err`**: document it as the RMS of the plane homography fit, which is what
  gates validity.

## 10. Open items and known limitations

- `configs/rack.yaml` is still placeholder geometry (`measured: false`). H-PROPS must measure the
  board. `tools/make_markers.py` and `python -m actus.spatial.rack_frame --source 0` exist to
  help: the live view overlays the zones, footprint and axes, with the marker count and error.
- **All filter parameters are hand-set** (`SpatialParams` defaults). They were tuned only against
  synthetic scenes and the replay, never on real footage. T-INT should tune them on
  training-operator takes only, and F-CAL should fit `temperatures`.
- **Hand ≠ 3D.** `landmarks_rack` is a ray–plane intersection, so a hand high above the board
  appears displaced along the camera ray. Proximity tolerates this at 8 cm, but a camera at a
  steep angle to a deep work area would need true hand depth.
- **Detections use the bbox centre**, projected onto the board plane. A tall object seen at an
  angle is displaced the same way. Zone margins of a few cm absorb it on the BAS board.
- An object that is placed and then *missed by the detector* while the hand withdraws can be
  believed "carried away invisibly" (held, with a miss likelihood of 0.35). The zone evidence then
  never rises, and the engine asks. That is the intended fail-safe, not a false verify.
- `SpatialEstimator` isn't a bus node. T-INT wires it (join detections/hands/pose by
  `frame_idx`, call `update`, publish `messages()`). A threaded node was left out on purpose,
  since T-INT owns the pipeline's timing.
- Environment: this session's `uv` is at `C:\Users\sairi\AppData\Local\Programs\Python\Python313\Scripts\uv`
  and is on PATH in Git Bash. OpenCV here is **5.0.0** (`cv2.aruco.ArucoDetector`,
  `DetectorParameters` and `CORNER_REFINE_SUBPIX` all work).
