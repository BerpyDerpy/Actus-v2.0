"""Process a recorded take (or a SIMULATED replay) deterministically, with overlays.

    uv run python tools/run_offline.py data/raw/op01_correct_01.mp4
    uv run python tools/run_offline.py replay:skip_S2
    uv run python tools/run_offline.py replay:occluded --evidence replay --answer yes
    uv run python tools/run_offline.py data/raw/op01_correct_01.mp4 --verify-determinism

It is the live pipeline (``actus.app.pipeline``) with sequential decode, no dropped frames, no
pacing, and no voice / RTSP / GUI. Outputs go to ``runs/<run_id>/``: ``events.jsonl`` (whose
timestamps come from the take's own clock, so reruns are byte-identical), ``downlink.json``,
``run.json`` and ``video/annotated_NNN.mp4``. ``--verify-determinism`` runs the input twice and
compares the two logs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from actus.app.pipeline import outcome_line, run_offline
from actus.core.config import load_config
from actus.io.eventlog import verify_log


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input", help="a video file, or replay:<scenario> (SIMULATED)")
    ap.add_argument("--config", help="override YAML on top of configs/default.yaml")
    ap.add_argument("--override", action="append", default=[], help="key.path=value")
    ap.add_argument("--run-id")
    ap.add_argument("--force", action="store_true", help="overwrite an existing run dir")
    ap.add_argument("--answer", choices=["yes", "no"], help="scripted answer to every Ask")
    ap.add_argument("--answer-delay", type=float, default=1.0, help="seconds after the Ask")
    ap.add_argument("--evidence", choices=["spatial", "replay"], help="replay only: stand-in")
    ap.add_argument("--max-frames", type=int, help="files only: stop after this many frames")
    ap.add_argument("--no-video", action="store_true", help="skip the annotated MP4")
    ap.add_argument(
        "--verify-determinism",
        action="store_true",
        help="run twice and compare events.jsonl byte for byte",
    )
    args = ap.parse_args(argv)
    cfg = load_config(args.config, overrides=args.override)

    def once(run_id: str | None):
        return run_offline(
            args.input,
            cfg,
            run_id=run_id,
            answer=args.answer,
            answer_delay_s=args.answer_delay,
            video=not args.no_video,
            evidence=args.evidence,
            max_frames=args.max_frames,
            force=args.force,
        )

    res = once(args.run_id)
    tag = " [SIMULATED]" if res.simulated else ""
    check = verify_log(res.events_path)
    print(f"# {args.input}{tag}: {res.summary.frames} frames in {res.summary.wall_s:.1f}s")
    print(f"# outcomes: {outcome_line(res.summary.outcomes)}")
    print(f"# alerts: {len(res.summary.alerts)}  events: {len(res.summary.outputs)}")
    print(f"# log: {res.events_path} ({'chain OK' if check.ok else check.error})")
    print(f"# events sha256: {res.events_sha256}")
    print(f"# downlink: {res.downlink_bytes} bytes")
    if not check.ok:
        return 1
    if args.verify_determinism:
        again = once(res.run_id + "-rerun")
        same = again.events_sha256 == res.events_sha256
        print(f"# rerun {again.run_dir.name}: {'IDENTICAL' if same else 'DIFFERENT'} events.jsonl")
        if not same:
            _first_diff(res.events_path, again.events_path)
            return 1
    return 0


def _first_diff(a: Path, b: Path) -> None:
    la = a.read_text(encoding="utf-8").splitlines()
    lb = b.read_text(encoding="utf-8").splitlines()
    for i, (x, y) in enumerate(zip(la, lb, strict=False), start=1):
        if x != y:
            print(f"# first difference at line {i}:\n<  {x[:300]}\n>  {y[:300]}")
            return
    print(f"# line counts differ: {len(la)} vs {len(lb)}")


if __name__ == "__main__":
    sys.exit(main())
