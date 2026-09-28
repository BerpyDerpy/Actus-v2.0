#!/usr/bin/env python3
"""Run the real perception stack over every recorded take and cache the features (T-HOI-TEMP).

    uv run python tools/extract_features.py                     # all takes, every frame
    uv run python tools/extract_features.py --stride 3          # every 3rd frame (quick pass)
    uv run python tools/extract_features.py --takes op05        # one operator
    uv run python tools/extract_features.py --report            # summarise an existing cache

This is the slow, once-per-dataset step that ``train_hoi.py`` and ``train_temporal.py`` read
from. Per frame it solves the rack frame from the ArUco markers, runs the detector, runs
MediaPipe hands and pose through rotation TTA, and hands all four to
``actus.models.features.FrameFeaturizer`` -- the same featurizer the live pipeline uses, so the
models see identical inputs offline and online.

Output, one ``.npz`` per take in ``data/features/`` plus ``manifest.json``:

    frame_idx (N,) int32     the clock; matches the video and the label CSV
    t         (N,) float32   seconds, read from the take's .frames.csv, never index/fps
    frame     (N, F)         per-frame features for the temporal model
    pair      (N, H, O, P)   per (hand, object) features for the HOI classifier
    pair_valid(N, H, O) bool both the hand and the object were actually observed
    label     (N,) int8      index into the manifest's `label_names` (steps, idle, other)
    rack_valid(N,) bool      the rack frame solved (or was held) this frame
    det_seen  (N, K) bool    which detector classes fired, for the coverage report

Caveats worth knowing before you trust the numbers, both reported by ``--report``:

* Rotation TTA finds the orientation of the *hands and pose*; the detector is then run at that
  same rotation and its boxes are rotated back. The takes' ``camera_roll_deg`` metadata is 0
  even for the rolled takes, so the roll is recovered from the image, not read off the label.
* Everything is keyed to the detector backend, which is recorded in the manifest. Change the
  detector and this cache is stale -- re-extract and retrain.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from actus.core.contracts import IDLE, OTHER, FramePacket  # noqa: E402
from actus.models.features import (  # noqa: E402
    SPEC_VERSION,
    FeatureLayout,
    FeatureParams,
    FrameFeaturizer,
)
from actus.protocol.schema import load_protocol  # noqa: E402
from tools._datalib import CLASSES, discover_takes, rel_to_repo  # noqa: E402

DEFAULT_OUT = "data/features"


# ----------------------------------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------------------------------


def label_names(protocol) -> list[str]:
    """Label vocabulary in index order: the protocol's steps, then idle, then other.

    Annotations may carry step ids the protocol doesn't define (op01's takes have stray ``S7``
    and ``S8``). Those become ``other`` rather than being silently dropped or, worse, folded
    into a real step.
    """
    return [*protocol.step_ids, IDLE, OTHER]


def read_times(take) -> dict[int, float]:
    """``frame_idx -> t`` from the take's ``.frames.csv`` sidecar, if T-REC wrote one."""
    import csv

    path = take.video_path.with_suffix(".frames.csv")
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        return {int(r["frame_idx"]): float(r["t"]) for r in csv.DictReader(f)}


# ----------------------------------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------------------------------


class Extractor:
    """Holds the perception models open across takes (loading them is the expensive part)."""

    def __init__(
        self,
        cfg,
        protocol,
        *,
        use_detector: bool = True,
        use_tta: bool = True,
        rack_mode: str = "auto",
        vote_rotation: bool = True,
    ) -> None:
        from actus.perception.detector import Detector
        from actus.spatial.rack_frame import RackFrameEstimator

        self.cfg = cfg
        self.protocol = protocol
        self.geometry_cfg = cfg["rack"]
        self.rack = RackFrameEstimator.from_config(self.geometry_cfg)
        self.geometry = self.rack.geometry
        self.rack_mode = rack_mode
        self.vote_rot = vote_rotation
        self.detector = Detector.from_config(cfg["detector"]) if use_detector else None
        self.backend = self.detector.backend_name if self.detector else "none"
        self.tta = self._build_handpose(cfg, use_tta)
        self.layout = FeatureLayout.from_protocol(protocol)
        self.params = FeatureParams()

    @staticmethod
    def _build_handpose(cfg, use_tta: bool):
        from actus.perception.handpose import HandPoseEstimator
        from actus.perception.rotation_tta import RotationTTA

        if use_tta:
            return RotationTTA.from_config(cfg["perception"])
        return RotationTTA(HandPoseEstimator.from_config(cfg["perception"]), candidates=(0,))

    def vote_rotation(self, take, *, samples: int = 12) -> int:
        """Pick one camera roll for the whole take by scoring a few frames at all four rotations.

        The camera does not roll *within* a take, so searching every frame (what RotationTTA does
        live, where the roll can change) costs four landmark passes per frame for an answer that
        is fixed.

        Scoring counts what was actually found -- hands, then poses -- and only uses mean
        confidence to break ties. Confidence alone is not discriminative on this footage: it
        picked a different orientation for consecutive takes from the same fixed camera, and
        MediaPipe's hand detector is nearly indifferent between 0 and 180 degrees while losing
        roughly half its detections at 90 and 270. Counting detections avoids paying that.
        """
        import cv2

        from actus.perception.rotation_tta import ROTATIONS, rotate_image

        cap = cv2.VideoCapture(str(take.video_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        hands = dict.fromkeys(ROTATIONS, 0)
        poses = dict.fromkeys(ROTATIONS, 0)
        confs = dict.fromkeys(ROTATIONS, 0.0)
        ts = 0
        seen = 0
        for k in range(samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k * n / max(1, samples)))
            ok, img = cap.read()
            if not ok:
                continue
            seen += 1
            for r in ROTATIONS:
                ts += 1
                raw = self.tta.estimator.estimate(rotate_image(img, r), ts)
                hands[r] += len(raw.hands.hands)
                poses[r] += 1 if raw.pose.landmarks_px is not None else 0
                confs[r] += raw.confidence
        cap.release()
        if not seen:
            return 0
        max_hands = max(max(hands.values()), 1)
        score = {r: hands[r] / max_hands + poses[r] / seen + confs[r] / seen for r in ROTATIONS}
        return max(score, key=lambda r: score[r])

    def featurizer(self, board_size: tuple[float, float] | None = None) -> FrameFeaturizer:
        fp = self.geometry.outer_box_footprint
        return FrameFeaturizer(
            self.protocol,
            params=self.params,
            layout=self.layout,
            board_size=board_size or self.geometry.board_size,
            outer_footprint={"outer": fp} if fp else None,
        )

    # -- rack frame ----------------------------------------------------------------------------

    def calibrate_rack(self, take, *, stride: int = 5):
        """Learn this take's marker layout in one cheap ArUco-only pass, then freeze it.

        The rig was re-laid between operators and does not match configs/rack.yaml, so the frame
        is derived from the markers' own (known, printed) size rather than a configured layout.
        Calibrating first means frame 0 already has the full layout instead of the frame drifting
        while markers are discovered. See actus/models/rackfit.py.
        """
        import cv2

        from actus.models.rackfit import MarkerFrameFitter, board_size_from

        fitter = MarkerFrameFitter(marker_size=self.geometry.marker_size)
        cap = cv2.VideoCapture(str(take.video_path))
        idx = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                fitter.update(FramePacket(idx, idx / 30.0, img, take.stem))
            idx += 1
        cap.release()
        calibration = fitter.calibration()
        fitter.frozen = True
        fitter._H = None
        fitter._last_fresh = None
        return fitter, calibration, board_size_from(calibration)

    def close(self) -> None:
        if self.tta is not None:
            self.tta.estimator.close()

    # -- one take ------------------------------------------------------------------------------

    def run_take(self, take, *, stride: int = 1, max_frames: int | None = None) -> dict:
        import cv2

        names = label_names(self.protocol)
        name_to_idx = {n: i for i, n in enumerate(names)}
        track = take.label_track()
        times = read_times(take)
        fps = float(take.meta().get("fps") or 30.0)

        rotation = self.vote_rotation(take) if self.vote_rot else None
        if rotation is not None:
            self.tta.candidates = (rotation,)
            self.tta.rotation_used = rotation
        if self.rack_mode == "auto":
            fitter, calibration, board = self.calibrate_rack(take)
            solve_rack = fitter.update
        else:
            self.rack.reset()
            fitter, calibration, board = None, {}, self.geometry.board_size
            solve_rack = self.rack.update
        if self.detector is not None:
            self.detector.reset()
        fz = self.featurizer(board)

        cap = cv2.VideoCapture(str(take.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {take.video_path}")

        rows: list[tuple] = []
        rotations: dict[int, int] = {}
        idx = 0
        t0 = time.time()
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                t = times.get(idx, idx / fps)
                packet = FramePacket(idx, t, img, take.stem)
                rack = solve_rack(packet)
                hands, pose = self.tta.process(packet, timestamp_ms=round(t * 1000))
                rot = hands.rotation_used
                rotations[rot] = rotations.get(rot, 0) + 1
                dets = None if self.detector is None else self.detector.detect(packet)
                ff = fz.update(idx, t, rack=rack, detections=dets, hands=hands, pose=pose)
                raw = track.label_at(idx)
                label = name_to_idx.get(raw, name_to_idx[OTHER])
                seen = np.array(
                    [any(d.cls == c for d in (dets.detections if dets else ())) for c in CLASSES],
                    dtype=bool,
                )
                rows.append((idx, t, ff.frame, ff.pair, ff.pair_valid, label, ff.rack_valid, seen))
                if max_frames is not None and len(rows) >= max_frames:
                    break
            idx += 1
        cap.release()

        if not rows:
            raise RuntimeError(f"{take.stem}: no frames read")
        return self._pack(take, rows, rotations, idx, time.time() - t0, calibration, board)

    def _pack(self, take, rows, rotations, n_video_frames, secs, calibration, board) -> dict:
        idxs = np.array([r[0] for r in rows], dtype=np.int32)
        ts = np.array([r[1] for r in rows], dtype=np.float32)
        frame = np.stack([r[2] for r in rows]).astype(np.float32)
        pair = np.stack([r[3] for r in rows]).astype(np.float32)
        pair_valid = np.stack([r[4] for r in rows]).astype(bool)
        label = np.array([r[5] for r in rows], dtype=np.int8)
        rack_valid = np.array([r[6] for r in rows], dtype=bool)
        det_seen = np.stack([r[7] for r in rows]).astype(bool)
        info = {
            "take": take.stem,
            "operator": take.operator,
            "scenario": take.scenario,
            "n": take.n,
            "frames": len(idxs),
            "video_frames": int(n_video_frames),
            "seconds": round(secs, 1),
            "rack_valid_rate": round(float(rack_valid.mean()), 4),
            "rack_mode": self.rack_mode,
            "rack_board_size": [round(float(b), 4) for b in board],
            "rack_calibration": {
                str(k): [round(float(v[0]), 4), round(float(v[1]), 4)]
                for k, v in calibration.items()
            },
            "rotations": {str(k): v for k, v in sorted(rotations.items())},
            "det_rate": {c: round(float(det_seen[:, i].mean()), 4) for i, c in enumerate(CLASSES)},
            "pair_valid_rate": {
                o: round(float(pair_valid[:, :, i].any(axis=1).mean()), 4)
                for i, o in enumerate(self.layout.objects)
            },
            "label_counts": {
                n: int((label == i).sum()) for i, n in enumerate(label_names(self.protocol))
            },
        }
        arrays = {
            "frame_idx": idxs,
            "t": ts,
            "frame": frame,
            "pair": pair,
            "pair_valid": pair_valid,
            "label": label,
            "rack_valid": rack_valid,
            "det_seen": det_seen,
        }
        return {"info": info, "arrays": arrays}


# ----------------------------------------------------------------------------------------------
# Manifest / report
# ----------------------------------------------------------------------------------------------


def write_manifest(out_dir: Path, extractor: Extractor, takes_info: list[dict], args) -> Path:
    manifest = {
        "feature_spec_version": SPEC_VERSION,
        "detector_backend": extractor.backend,
        "detector_classes": list(CLASSES),
        "protocol": rel_to_repo(args.protocol),
        "label_names": label_names(extractor.protocol),
        "hands": list(extractor.layout.hands),
        "objects": list(extractor.layout.objects),
        "containers": list(extractor.layout.containers),
        "zones": list(extractor.layout.zones),
        "frame_names": list(extractor.layout.frame_names),
        "pair_names": list(extractor.layout.pair_names),
        "feature_params": vars(extractor.params),
        "rack_mode": args.rack,
        "rack_measured": bool(extractor.geometry.measured),
        "camera_calibrated": bool(extractor.geometry.camera.calibrated),
        "stride": args.stride,
        "rotation_tta": not args.no_tta,
        "rotation_per_take_vote": not args.no_tta and not args.search_rotation,
        "takes": takes_info,
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return path


def report(out_dir: Path) -> int:
    path = out_dir / "manifest.json"
    if not path.exists():
        print(f"no manifest at {path}; run the extractor first")
        return 1
    m = json.loads(path.read_text(encoding="utf-8"))
    takes = m["takes"]
    print(f"cache: {out_dir}  spec v{m['feature_spec_version']}  detector={m['detector_backend']}")
    print(f"takes: {len(takes)}   frames: {sum(t['frames'] for t in takes)}")
    if not m.get("rack_measured"):
        print("WARNING configs/rack.yaml is measured: false -- rack metres are approximate")
    if not m.get("camera_calibrated"):
        print("WARNING camera intrinsics are approximated from the image size")

    def mean(key: str, sub: str) -> float:
        vals = [t[key][sub] for t in takes if sub in t[key]]
        return float(np.mean(vals)) if vals else 0.0

    print("\nper-frame detection rate (share of frames the class fired):")
    for c in m["detector_classes"]:
        print(f"  {c:20s} {mean('det_rate', c):.3f}")
    print("\nHOI pair coverage (share of frames with hand AND object seen):")
    for o in m["objects"]:
        print(f"  {o:20s} {mean('pair_valid_rate', o):.3f}")
    rack = float(np.mean([t["rack_valid_rate"] for t in takes]))
    print(f"\nrack frame valid: {rack:.3f} of frames")
    counts: dict[str, int] = {}
    for t in takes:
        for k, v in t["label_counts"].items():
            counts[k] = counts.get(k, 0) + v
    total = sum(counts.values()) or 1
    print("\nlabel distribution:")
    for k in m["label_names"]:
        print(f"  {k:8s} {counts.get(k, 0):7d}  {counts.get(k, 0) / total:6.1%}")
    rot: dict[str, int] = {}
    for t in takes:
        for k, v in t.get("rotations", {}).items():
            rot[k] = rot.get(k, 0) + v
    print(f"\nrotation TTA winners: {dict(sorted(rot.items()))}")
    return 0


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--raw-dir", default="data/raw", help="takes directory (default: data/raw)")
    p.add_argument("--out", default=DEFAULT_OUT, help=f"cache directory (default: {DEFAULT_OUT})")
    p.add_argument("--protocol", default="protocols/bas_boxes.yaml")
    p.add_argument("--config", default=None, help="config file (default: configs/default.yaml)")
    p.add_argument("--stride", type=int, default=1, help="keep every Nth frame (default: 1)")
    p.add_argument("--takes", nargs="*", default=None, help="substring filters on the take name")
    p.add_argument("--max-frames", type=int, default=None, help="stop each take after N frames")
    p.add_argument("--force", action="store_true", help="re-extract takes already cached")
    p.add_argument("--no-detector", action="store_true", help="skip the detector (hands/pose only)")
    p.add_argument(
        "--no-tta", action="store_true", help="disable rotation TTA entirely (upright only)"
    )
    p.add_argument(
        "--search-rotation",
        action="store_true",
        help="search rotations every frame instead of voting once per take (slower)",
    )
    p.add_argument(
        "--rack",
        choices=("auto", "config"),
        default="auto",
        help="auto: self-calibrate from the markers (default; see actus/models/rackfit.py). "
        "config: solve against configs/rack.yaml (needs a measured rig)",
    )
    p.add_argument("--report", action="store_true", help="summarise the existing cache and exit")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    from actus.core.config import load_config, resolve_path

    args = parse_args(argv)
    out_dir = resolve_path(args.out)
    if args.report:
        return report(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    protocol = load_protocol(resolve_path(args.protocol))

    takes = discover_takes(resolve_path(args.raw_dir))
    if args.takes:
        takes = [t for t in takes if any(f in t.stem for f in args.takes)]
    unlabelled = [t for t in takes if len(t.label_track().items()) <= 1]
    takes = [t for t in takes if t not in unlabelled]
    if unlabelled:
        print(f"skipping {len(unlabelled)} unlabelled take(s): {[t.stem for t in unlabelled]}")
    if not takes:
        print("no labelled takes found")
        return 1

    extractor = Extractor(
        cfg,
        protocol,
        use_detector=not args.no_detector,
        use_tta=not args.no_tta,
        rack_mode=args.rack,
        vote_rotation=not args.no_tta and not args.search_rotation,
    )
    print(
        f"detector backend: {extractor.backend}   features: {extractor.layout.n_frame} frame, "
        f"{extractor.layout.n_pair} pair   stride: {args.stride}"
    )

    infos: list[dict] = []
    try:
        for i, take in enumerate(takes, 1):
            dest = out_dir / f"{take.stem}.npz"
            if dest.exists() and not args.force:
                info = json.loads((out_dir / f"{take.stem}.json").read_text(encoding="utf-8"))
                infos.append(info)
                print(f"[{i}/{len(takes)}] {take.stem}: cached")
                continue
            packed = extractor.run_take(take, stride=args.stride, max_frames=args.max_frames)
            np.savez_compressed(dest, **packed["arrays"])
            (out_dir / f"{take.stem}.json").write_text(
                json.dumps(packed["info"], indent=1), encoding="utf-8"
            )
            infos.append(packed["info"])
            info = packed["info"]
            print(
                f"[{i}/{len(takes)}] {take.stem}: {info['frames']} frames in {info['seconds']}s  "
                f"rack={info['rack_valid_rate']:.2f}  "
                f"pairs={info['pair_valid_rate']}"
            )
    finally:
        extractor.close()

    write_manifest(out_dir, extractor, infos, args)
    print(f"\nwrote {len(infos)} take(s) to {rel_to_repo(out_dir)}")
    return report(out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
