"""T-DATA: split extracted frames by operator (never by frame), with leakage checks.

    uv run python tools/make_splits.py --manifest data/images/manifest.csv \\
        --test-operators op05 --val-operators op04

Writes ``data/splits/{train,val,test}.txt`` (image paths, repo-relative, one per line),
``data/splits/report.json`` / ``report.md`` (per-operator/per-scenario counts and durations, the
leakage check, and a warning if take duration alone predicts the scenario), and
``data/dataset.yaml`` (an Ultralytics-format dataset config over the classes in
``data/labels/classes.txt``).

If ``--test-operators`` / ``--val-operators`` aren't given, the last one or two operators
(sorted) are held out automatically — with a warning, since that choice should usually be
deliberate (the build plan holds out ``op05``).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._datalib import (  # noqa: E402
    CLASSES,
    discover_takes,
    read_classes,
    read_manifest,
    rel_to_repo,
    to_repo_path,
)

SPLITS = ("train", "val", "test")


def assign_operators(
    operators: list[str], *, test: list[str] | None, val: list[str] | None
) -> tuple[dict[str, str], list[str]]:
    """``{operator: "train"|"val"|"test"}``, plus any warnings about the choice."""
    warnings = []
    operators = sorted(set(operators))
    auto = test is None
    if test is None:
        if len(operators) >= 3:
            test = operators[-1:]
            warnings.append(f"no --test-operators given; auto-held-out {test} (last, sorted)")
        else:
            warnings.append(
                f"only {len(operators)} operator(s) present; nothing held out for test "
                "(pass --test-operators to force a split)"
            )
            test = []
    if val is None:
        remaining = [o for o in operators if o not in test]
        if len(remaining) >= 3:
            val = remaining[-1:]
            if not auto:  # test was explicit but val wasn't: still worth flagging
                warnings.append(f"no --val-operators given; auto-held-out {val} (last, sorted)")
        else:
            val = []
    unknown = (set(test) | set(val)) - set(operators)
    if unknown:
        raise ValueError(f"--test-operators/--val-operators name unknown operators: {unknown}")
    overlap = set(test) & set(val)
    if overlap:
        raise ValueError(f"an operator can't be both val and test: {overlap}")
    split_of = {o: ("test" if o in test else "val" if o in val else "train") for o in operators}
    return split_of, warnings


def check_leakage(rows: list[dict], split_of: dict[str, str]) -> list[str]:
    """Every take must land in exactly one split. Splits are assigned per operator, so this is
    guaranteed by construction; checked explicitly in case a manifest was hand-edited."""
    problems = []
    take_splits: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        take_splits[r["take"]].add(split_of[r["operator"]])
    for take, splits in take_splits.items():
        if len(splits) > 1:
            problems.append(f"take {take!r} spans multiple splits: {sorted(splits)}")
    return problems


def duration_predicts_scenario(durations: dict[str, list[float]]) -> list[str]:
    """Scenario pairs whose take-duration ranges never overlap (so a length threshold alone
    would perfectly separate them — a sign the model could learn duration, not the activity)."""
    warnings = []
    scenarios = sorted(durations)
    for i, a in enumerate(scenarios):
        for b in scenarios[i + 1 :]:
            da, db = durations[a], durations[b]
            if not da or not db:
                continue
            if max(da) < min(db) or max(db) < min(da):
                warnings.append(
                    f"{a} ({min(da):.1f}-{max(da):.1f}s) and {b} ({min(db):.1f}-{max(db):.1f}s) "
                    "never overlap in duration: length alone predicts the scenario"
                )
    return warnings


def load_take_durations(raw_dir: Path) -> dict[str, float]:
    """take stem -> duration_s, from T-REC's session metadata (falls back to the label track's
    last timestamp, e.g. for hand-built fixtures with no metadata JSON)."""
    out = {}
    for take in discover_takes(raw_dir):
        d = take.duration_s()
        if d is None:
            items = take.label_track().items()
            d = items[-1][1] if items else 0.0
        out[take.stem] = d
    return out


def build_report(rows: list[dict], split_of: dict[str, str], durations: dict[str, float]) -> dict:
    by_take: dict[str, dict] = {}
    for r in rows:
        info = by_take.setdefault(
            r["take"], {"operator": r["operator"], "scenario": r["scenario"], "frames": 0}
        )
        info["frames"] += 1
    for stem, info in by_take.items():
        info["duration_s"] = durations.get(stem)
        info["split"] = split_of[info["operator"]]

    scenario_durations: dict[str, list[float]] = defaultdict(list)
    for info in by_take.values():
        if info["duration_s"] is not None:
            scenario_durations[info["scenario"]].append(info["duration_s"])

    per_split_takes: dict[str, int] = defaultdict(int)
    per_split_frames: dict[str, int] = defaultdict(int)
    for info in by_take.values():
        per_split_takes[info["split"]] += 1
        per_split_frames[info["split"]] += info["frames"]

    return {
        "operators": split_of,
        "takes": by_take,
        "per_split_takes": dict(per_split_takes),
        "per_split_frames": dict(per_split_frames),
        "leakage_checks": check_leakage(rows, split_of),
        "duration_warnings": duration_predicts_scenario(scenario_durations),
    }


def _report_md(report: dict) -> str:
    lines = ["# Dataset split report", "", "## Operators"]
    for op, split in sorted(report["operators"].items()):
        lines.append(f"- {op}: {split}")
    lines += ["", "## Frames / takes per split"]
    for split in SPLITS:
        lines.append(
            f"- {split}: {report['per_split_frames'].get(split, 0)} frames, "
            f"{report['per_split_takes'].get(split, 0)} takes"
        )
    lines.append("")
    if report["leakage_checks"]:
        lines.append("## Leakage check: FAILED")
        lines += [f"- {w}" for w in report["leakage_checks"]]
    else:
        lines.append(
            "## Leakage check: passed — no take spans two splits (splits are assigned per "
            "operator, so no operator does either)"
        )
    if report["duration_warnings"]:
        lines += ["", "## Duration warnings"]
        lines += [f"- {w}" for w in report["duration_warnings"]]
    return "\n".join(lines) + "\n"


def _split_arg(spec: str | None) -> list[str] | None:
    if spec is None:
        return None
    return [o for o in spec.split(",") if o]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default="data/images/manifest.csv")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--labels-dir", default="data/labels")
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--dataset-yaml", default="data/dataset.yaml")
    ap.add_argument("--test-operators", help="comma-separated operator ids")
    ap.add_argument("--val-operators", help="comma-separated operator ids")
    ap.add_argument("--synthetic", help="synthetic split file to reference in dataset.yaml")
    args = ap.parse_args(argv)

    manifest_path = to_repo_path(args.manifest)
    raw_dir = to_repo_path(args.raw_dir)
    labels_dir = to_repo_path(args.labels_dir)
    out_dir = to_repo_path(args.out_dir)

    rows = read_manifest(manifest_path)
    if not rows:
        print(f"{manifest_path} is empty", file=sys.stderr)
        return 1
    operators = sorted({r["operator"] for r in rows})
    split_of, warnings = assign_operators(
        operators, test=_split_arg(args.test_operators), val=_split_arg(args.val_operators)
    )

    durations = load_take_durations(raw_dir)
    report = build_report(rows, split_of, durations)
    for w in [*warnings, *report["duration_warnings"], *report["leakage_checks"]]:
        print(f"WARNING {w}")

    out_dir.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_split[split_of[r["operator"]]].append(r["image_path"])
    for split in SPLITS:
        paths = by_split.get(split, [])
        (out_dir / f"{split}.txt").write_text(
            "\n".join(paths) + ("\n" if paths else ""), encoding="utf-8"
        )

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(_report_md(report), encoding="utf-8")

    classes_path = labels_dir / "classes.txt"
    classes = read_classes(classes_path) if classes_path.exists() else CLASSES
    # Anchored at the repo root, not "data/": image paths in the split files are already
    # repo-relative (extract_frames.py writes them that way), and --out-dir need not live
    # under data/ (e.g. a test's tmp_path).
    dataset = {
        "path": str(REPO_ROOT),
        "train": rel_to_repo(out_dir / "train.txt"),
        "val": rel_to_repo(out_dir / "val.txt"),
        "test": rel_to_repo(out_dir / "test.txt"),
        "names": dict(enumerate(classes)),
    }
    if args.synthetic:
        dataset["synthetic"] = rel_to_repo(to_repo_path(args.synthetic))
    dataset_yaml_path = to_repo_path(args.dataset_yaml)
    dataset_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_yaml_path.write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")

    print(
        f"train {report['per_split_frames'].get('train', 0)}f / "
        f"val {report['per_split_frames'].get('val', 0)}f / "
        f"test {report['per_split_frames'].get('test', 0)}f "
        f"across {len(operators)} operators -> {dataset_yaml_path}"
    )
    return 1 if report["leakage_checks"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
