"""T-DATA: sample frames from T-REC takes for labelling.

Dense samples are taken around every step-boundary in a take's label CSV (the frames a human
annotator would most want to see); sparse samples cover the rest. Frames are decoded
sequentially, never seeked, so ``frame_idx`` matches T-REC's own decode-order convention.

    uv run python tools/extract_frames.py --raw-dir data/raw --out-dir data/images

Writes ``<out-dir>/<take>/frame_{frame_idx:06d}.jpg`` and ``<out-dir>/manifest.csv`` (one row per
extracted frame: image_path, take, operator, scenario, n, frame_idx, t, label, sample). The
manifest is the source of truth for downstream tools; ``--max-frames`` thins it globally (across
all takes) if the plan below produces more than that, dense frames first, and the rest evenly
across every take.

``t`` is the container's own timestamp for the frame (``CAP_PROP_POS_MSEC``), read while
decoding; ``frame_idx / container_fps`` only if the container has no timestamps; ``--fps`` is a
last resort, and a warning says so. ``--takes`` re-extracts only those takes and *merges* them
into an existing manifest; the other takes' rows are kept. A planned frame that turns out not to
exist (the container over-reported its frame count) is dropped from the manifest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._datalib import (  # noqa: E402
    Take,
    discover_takes,
    read_manifest,
    rel_to_repo,
    to_repo_path,
    write_manifest,
)


def sample_plan(
    boundaries: list[int],
    n_frames: int,
    *,
    fps: float,
    dense_radius_s: float,
    dense_stride: int,
    sparse_stride: int,
) -> dict[int, str]:
    """``{frame_idx: "dense"|"sparse"}`` for one take. Dense wins where the two would overlap."""
    radius = max(0, round(dense_radius_s * fps))
    dense: set[int] = set()
    for b in boundaries:
        lo, hi = max(0, b - radius), min(n_frames - 1, b + radius)
        dense.update(range(lo, hi + 1, max(1, dense_stride)))
    sparse = set(range(0, n_frames, max(1, sparse_stride))) - dense
    return {**dict.fromkeys(sparse, "sparse"), **dict.fromkeys(dense, "dense")}


def _thin_to_budget(rows: list[dict], max_frames: int | None) -> list[dict]:
    if max_frames is None or len(rows) <= max_frames:
        return rows
    dense = [r for r in rows if r["sample"] == "dense"]
    sparse = [r for r in rows if r["sample"] == "sparse"]
    if len(dense) >= max_frames:
        # even the boundary frames alone don't fit: thin them uniformly, drop all sparse frames
        stride = len(dense) / max_frames
        keep = {round(i * stride) for i in range(max_frames)}
        return [r for i, r in enumerate(dense) if i in keep]
    budget = max_frames - len(dense)
    # Evenly spaced over the whole sparse list (every take), not a floored stride plus a
    # truncation that would starve the takes listed last.
    n = len(sparse)
    keep = {min(n - 1, int(i * n / budget)) for i in range(budget)}
    return dense + [r for i, r in enumerate(sparse) if i in keep]


def _video_fps_and_count(video_path: Path, fallback_fps: float) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        print(
            f"warning: {video_path.name} reports no fps; using --fps {fallback_fps:g} for the "
            "dense window (t comes from the container timestamps where it has them)",
            file=sys.stderr,
        )
        fps = fallback_fps
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:  # some containers under-report; discover by decoding once
        n = 0
        while cap.grab():
            n += 1
    cap.release()
    return fps, n


def candidate_rows(
    take: Take,
    *,
    fallback_fps: float,
    dense_radius_s: float,
    dense_stride: int,
    sparse_stride: int,
) -> tuple[float, list[dict]]:
    """The frames this take *would* contribute, before any global budget thinning."""
    track = take.label_track()
    boundaries = [fi for fi, _, _ in track.items()]
    fps, n_frames = _video_fps_and_count(take.video_path, fallback_fps)
    plan = sample_plan(
        boundaries,
        n_frames,
        fps=fps,
        dense_radius_s=dense_radius_s,
        dense_stride=dense_stride,
        sparse_stride=sparse_stride,
    )
    rows = [
        {
            "take": take.stem,
            "operator": take.operator,
            "scenario": take.scenario,
            "n": take.n,
            "frame_idx": fi,
            "t": round(fi / fps, 4),  # replaced by the container timestamp when decoded
            "label": track.label_at(fi),
            "sample": sample,
        }
        for fi, sample in sorted(plan.items())
    ]
    return fps, rows


def write_selected_frames(
    take: Take, rows: list[dict], out_dir: Path, *, fmt: str, quality: int, overwrite: bool
) -> None:
    """Decode ``take`` once and write the images for ``rows`` (mutated in place with
    ``image_path``)."""
    by_idx = {r["frame_idx"]: r for r in rows}
    take_dir = out_dir / take.stem
    take_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(take.video_path))
    fi = 0
    ok, frame = cap.read()
    while ok:
        if fi in by_idx:
            img_path = take_dir / f"frame_{fi:06d}.{fmt}"
            if overwrite or not img_path.exists():
                params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == "jpg" else []
                cv2.imwrite(str(img_path), frame, params)
            by_idx[fi]["image_path"] = rel_to_repo(img_path)
            msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if (msec and msec > 0) or fi == 0:  # 0 ms is real only for the first frame
                by_idx[fi]["t"] = round((msec or 0.0) / 1000.0, 4)
        fi += 1
        ok, frame = cap.read()
    cap.release()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/images")
    ap.add_argument("--fps", type=float, default=30.0, help="fallback if a container reports none")
    ap.add_argument("--dense-radius-s", type=float, default=0.5)
    ap.add_argument("--dense-stride", type=int, default=2)
    ap.add_argument("--sparse-stride", type=int, default=15)
    ap.add_argument("--max-frames", type=int, default=1500, help="global cap; 0 disables it")
    ap.add_argument("--format", choices=["jpg", "png"], default="jpg")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--takes", help="comma-separated take stems to restrict to")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    raw_dir = to_repo_path(args.raw_dir)
    out_dir = to_repo_path(args.out_dir)

    takes = discover_takes(raw_dir)
    if args.takes:
        wanted = set(args.takes.split(","))
        takes = [t for t in takes if t.stem in wanted]
    if not takes:
        print(f"no takes with a label CSV found in {raw_dir}", file=sys.stderr)
        return 1

    per_take: dict[str, tuple[Take, float]] = {}
    all_rows: list[dict] = []
    for take in takes:
        fps, rows = candidate_rows(
            take,
            fallback_fps=args.fps,
            dense_radius_s=args.dense_radius_s,
            dense_stride=args.dense_stride,
            sparse_stride=args.sparse_stride,
        )
        per_take[take.stem] = (take, fps)
        all_rows.extend(rows)

    all_rows = _thin_to_budget(all_rows, args.max_frames or None)
    kept_by_take: dict[str, list[dict]] = {}
    for r in all_rows:
        kept_by_take.setdefault(r["take"], []).append(r)

    for stem, (take, _fps) in per_take.items():
        rows = kept_by_take.get(stem, [])
        if rows:
            write_selected_frames(
                take, rows, out_dir, fmt=args.format, quality=args.quality, overwrite=args.overwrite
            )
        missing = [r for r in rows if "image_path" not in r]
        if missing:
            print(
                f"warning: {stem}: {len(missing)} planned frames past the real end of the video "
                "(the container over-reported its frame count); left out of the manifest",
                file=sys.stderr,
            )
        n_dense = sum(1 for r in rows if r["sample"] == "dense")
        print(f"{stem}: {len(rows)} frames ({n_dense} dense, {len(rows) - n_dense} sparse)")
    all_rows = [r for r in all_rows if "image_path" in r]

    manifest_path = out_dir / "manifest.csv"
    if args.takes and manifest_path.exists():
        # Only these takes were re-extracted: keep every other take's rows.
        redone = set(per_take)
        kept = [r for r in read_manifest(manifest_path) if r["take"] not in redone]
        all_rows = kept + all_rows
    write_manifest(manifest_path, all_rows)
    n_dense = sum(1 for r in all_rows if r["sample"] == "dense")
    print(
        f"wrote {len(all_rows)} frames ({n_dense} dense, {len(all_rows) - n_dense} sparse) "
        f"from {len({r['take'] for r in all_rows})} takes to {rel_to_repo(manifest_path)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
