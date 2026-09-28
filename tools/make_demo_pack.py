"""T-DEMO: draft the walkthrough script, shot list and limitations slide from the report card.

    uv run python tools/make_demo_pack.py                    # writes demo/out/
    uv run python tools/make_demo_pack.py --strict           # exit 1 unless every shot is READY
    uv run python tools/make_demo_pack.py --report runs/eval/report.json --out demo/out

The text lives in ``demo/shots.yaml`` and ``demo/limitations.yaml`` (hand-written, reviewable).
This tool renders it into three documents and never invents a result:

* ``walkthrough.md``  the narration, shot by shot, with a running clock;
* ``shot-list.md``    a preflight of what the machine has, then for each shot who does what, the
                      command, exactly what to see and hear, and when to retake;
* ``limitations.md``  the slide text and the backup slide, each item with its source;
* ``manifest.json``   the inputs' hashes, every gate and every shot's status, and any warnings.

Honesty is enforced here, not left to whoever records:

* **Numbers come only from a real report.** A metric token resolves from ``runs/eval/report.json``
  only if that report exists, is not SIMULATED and evaluated at least one take. Otherwise it
  resolves to a visible ``[NOT MEASURED ...]`` placeholder, so a draft can never quote a number
  that does not exist.
* **Each shot has a status computed from what is on disk** (the "gates": the rack measured, the
  custom detector present, the media binaries, the built GUI, ...):
  ``READY`` (film it for real), ``SIMULATED`` (it can only be filmed from the replay, so its
  narration says "simulated" out loud) or ``BLOCKED`` (it cannot be filmed honestly yet).
* **Limitations follow the gaps.** An item describing a gap appears only while the gap exists.

The output is a pure function of its inputs: no timestamps, sorted keys, so regenerating an
unchanged pack reproduces it byte for byte. Console output is ASCII.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

SPEC_VERSION = 1
MAX_WORDS_PER_SECOND = 2.5  # 150 words a minute, a comfortable narration pace
TOTAL_SECONDS_RANGE = (180, 240)  # the plan's "3-4 min" walkthrough; the script aims for 3:00-4:00
SLIDE_MAX_ITEMS = 6
SLIDE_ITEM_MAX_WORDS = 30

# The ten headline metrics of tools/evaluate.py (tests pin this list to evaluate.METRICS).
METRICS: tuple[str, ...] = (
    "step_completion_precision",
    "step_completion_recall",
    "skip_alert_precision",
    "skip_alert_recall",
    "out_of_order_alert_precision",
    "out_of_order_alert_recall",
    "false_alerts_per_min_correct_takes",
    "median_detection_latency_s",
    "ask_rate_per_min",
    "ask_then_correct_rate",
)
STATIC_TOKENS: tuple[str, ...] = (
    "data_kind",
    "n_takes",
    "n_operators",
    "operators",
    "criteria_met",
    "criteria_version",
    "eval_id",
)
TOKEN_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*(?::[A-Za-z_][A-Za-z0-9_]*)?)\s*\}\}")

READY, SIMULATED, BLOCKED = "READY", "SIMULATED", "BLOCKED"


class PackError(ValueError):
    """The demo pack's inputs are inconsistent. Raised loudly rather than papered over."""


# --------------------------------------------------------------------------------------------
# Files and configuration
# --------------------------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise PackError(f"{path}: file not found") from e
    except yaml.YAMLError as e:
        raise PackError(f"{path}: not valid YAML: {e}") from e
    if not isinstance(data, dict):
        raise PackError(f"{path}: expected a mapping at the top level")
    return data


def _read_yaml_or_empty(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _dig(data: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _mmss(seconds: int) -> str:
    return f"{seconds // 60}:{seconds % 60:02d}"


# --------------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportFacts:
    """``runs/eval/report.json`` as far as the demo pack may rely on it."""

    path: Path | None
    sha256: str | None
    data: Mapping[str, Any]
    usable: bool
    reason: str  # why it is not usable; "" when it is
    placeholder: str  # what a token resolves to when it is not usable


def read_report(path: Path | None) -> ReportFacts:
    """Load the report card. Usable means: present, valid, not SIMULATED, at least one take."""

    def unusable(reason: str, placeholder: str, data: Mapping[str, Any] | None = None):
        return ReportFacts(path, sha, data or {}, False, reason, placeholder)

    sha = _sha256(path) if path is not None else None
    if path is None or sha is None:
        return unusable("no report at runs/eval/report.json", "[NOT MEASURED - no report]")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return unusable("the report is not valid JSON", "[NOT MEASURED - unreadable report]")
    if not isinstance(data, dict):
        return unusable("the report is not a JSON object", "[NOT MEASURED - unreadable report]")
    if data.get("simulated"):
        return unusable(
            "the report is SIMULATED (replay numbers exercise the harness, not perception)",
            "[SIMULATED - DO NOT QUOTE]",
            data,
        )
    try:
        n_takes = int(data.get("takes_evaluated", 0))
    except (TypeError, ValueError):
        n_takes = 0
    if n_takes < 1:
        return unusable("the report evaluated no takes", "[NOT MEASURED - no takes]", data)
    return ReportFacts(path, sha, data, True, "", "")


class Tokens:
    """Resolves ``{{name}}`` in the shots' and limitations' text from a report.

    ``m:<metric>`` is the value, ``t:<metric>`` the pre-registered target and ``v:<metric>`` the
    verdict (PASS / FAIL / n/a), all as the report states them. An unknown token is an error.
    """

    def __init__(self, report: ReportFacts) -> None:
        self.report = report
        d = report.data
        operators = str(d.get("operators", "-"))
        self._static = {
            "data_kind": str(d.get("data", "")),
            "n_takes": str(d.get("takes_evaluated", "")),
            "n_operators": str(
                0 if operators.strip() in ("", "-") else len([o for o in operators.split(",") if o])
            ),
            "operators": operators,
            "criteria_met": str(d.get("criteria_met", "")),
            "criteria_version": str(d.get("criteria_version", "")),
            "eval_id": str(d.get("eval_id", "")),
        }
        rows = d.get("criteria") if isinstance(d.get("criteria"), list) else []
        self._rows = {r["metric"]: r for r in rows if isinstance(r, dict) and "metric" in r}

    @staticmethod
    def names_in(text: str) -> list[str]:
        return TOKEN_RE.findall(text)

    def resolve(self, name: str) -> str:
        if ":" in name:
            kind, metric = name.split(":", 1)
            if kind not in ("m", "t", "v") or metric not in METRICS:
                raise PackError(f"unknown token {{{{{name}}}}}")
            if not self.report.usable:
                return self.report.placeholder
            row = self._rows.get(metric)
            if row is None:
                return "[NOT MEASURED - metric missing from the report]"
            field = {"m": "value", "t": "target", "v": "result"}[kind]
            return str(row.get(field, "n/a"))
        if name not in STATIC_TOKENS:
            raise PackError(f"unknown token {{{{{name}}}}}")
        return self.report.placeholder if not self.report.usable else self._static[name]

    def render(self, text: str) -> str:
        return TOKEN_RE.sub(lambda m: self.resolve(m.group(1)), text)


def timing_text(text: str) -> str:
    """The text with every token counted as one spoken word, for the time budget."""
    return TOKEN_RE.sub("0.00", text)


# --------------------------------------------------------------------------------------------
# Gates: what is on disk decides what can be filmed
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Gate:
    name: str
    ok: bool
    detail: str
    fix: str


GATE_FIXES: dict[str, str] = {
    "report_real": (
        "Run `uv run python tools/evaluate.py` on the held-out split (it needs labelled takes "
        "of the test operators, a measured rack and the custom detector), then regenerate."
    ),
    "rack_measured": (
        "H-PROPS: measure the board into configs/rack.yaml and set `measured: true` "
        "(Vault/T-HOI-TEMP.md section 0.1)."
    ),
    "detector_custom": (
        "T-DET-TRAIN: `uv run python tools/train_detector.py`; models/detector/best.onnx and "
        "card.md must exist. Read the card: only if its held-out mAP50 is >= 0.85, set "
        "`detector.backend: custom` in configs/default.yaml (the plan's rule for replacing the "
        "open-vocabulary detector). This gate does not read the card's numbers."
    ),
    "media_binaries": "`uv run python tools/fetch_models.py` (or scripts/setup.sh / setup.ps1).",
    "ui_built": "`(cd ui && npm ci && npm run build)`.",
    "asr_model": "`uv run python tools/fetch_models.py --group voice` (the Vosk model).",
    "tts_model": (
        "`uv run python tools/fetch_models.py --group voice` (the Kokoro model and voices). "
        "Without it the voice falls back to pyttsx3."
    ),
    "engine_tuned": (
        "Tune the decision thresholds on training-operator takes only (T-INT) and record them "
        "under `engine:` in the configuration."
    ),
    "temporal_model": "`uv run python tools/train_temporal.py` (only if the model is wanted).",
}
GATES: tuple[str, ...] = tuple(GATE_FIXES)


def _find_binary(root: Path, spec: str) -> Path | None:
    p = Path(spec)
    candidates = [p] if p.is_absolute() else [root / p, root / "bin" / p.name]
    for c in candidates:
        for cand in (c, c.with_name(c.name + ".exe")):
            if cand.is_file():
                return cand
    if len(p.parts) != 1:  # a path is looked up under the root only, never in the cwd
        return None
    found = shutil.which(spec)
    return Path(found) if found else None


def _under(root: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else root / q


def check_gates(root: Path, report: ReportFacts) -> dict[str, Gate]:
    """Every gate, from the files under ``root`` and the report. Never runs anything."""
    cfg = _read_yaml_or_empty(root / "configs" / "default.yaml")
    g: dict[str, Gate] = {}

    def add(name: str, ok: bool, detail: str) -> None:
        g[name] = Gate(name, ok, detail, GATE_FIXES[name])

    add("report_real", report.usable, "usable" if report.usable else report.reason)

    rack_path = _under(root, str(_dig(cfg, "rack.config", "configs/rack.yaml")))
    measured = _read_yaml_or_empty(rack_path).get("measured") is True
    add(
        "rack_measured",
        measured,
        f"{_rel(rack_path, root)} has measured: true"
        if measured
        else f"{_rel(rack_path, root)} is a placeholder (measured: false or missing)",
    )

    # The gate is about what the pipeline will actually run: trained weights that exist, with
    # their model card, AND a configuration that selects them (the default is open_vocab).
    weights = _under(root, str(_dig(cfg, "detector.custom_weights", "models/detector/best.onnx")))
    card = weights.parent / "card.md"
    problems = [f"missing {_rel(p, root)}" for p in (weights, card) if not p.is_file()]
    backend = str(_dig(cfg, "detector.backend", "open_vocab"))
    if backend != "custom":
        problems.append(f"detector.backend is {backend}, not custom")
    add(
        "detector_custom",
        not problems,
        "trained weights, model card and detector.backend: custom"
        if not problems
        else "; ".join(problems),
    )

    mediamtx = _find_binary(root, str(_dig(cfg, "stream.mediamtx_bin", "bin/mediamtx")))
    ffmpeg = _find_binary(root, str(_dig(cfg, "stream.ffmpeg_bin", "ffmpeg")))
    lacking = [n for n, p in (("mediamtx", mediamtx), ("ffmpeg", ffmpeg)) if p is None]
    add(
        "media_binaries",
        not lacking,
        "mediamtx and ffmpeg found" if not lacking else "not found: " + ", ".join(lacking),
    )

    index = _under(root, str(_dig(cfg, "ui.static_dir", "ui/dist"))) / "index.html"
    add(
        "ui_built",
        index.is_file(),
        f"{_rel(index, root)} present" if index.is_file() else f"{_rel(index, root)} missing",
    )

    asr = _under(root, str(_dig(cfg, "voice.asr.model", "models/voice/vosk")))
    asr_ok = asr.is_dir() and any(asr.iterdir())
    add(
        "asr_model",
        asr_ok,
        f"{_rel(asr, root)} present" if asr_ok else f"{_rel(asr, root)} missing or empty",
    )

    kokoro = [
        _under(root, str(_dig(cfg, "voice.tts.kokoro_model", "models/voice/kokoro.onnx"))),
        _under(root, str(_dig(cfg, "voice.tts.kokoro_voices", "models/voice/voices.bin"))),
    ]
    absent = [_rel(p, root) for p in kokoro if not p.is_file()]
    add(
        "tts_model",
        not absent,
        "Kokoro model and voices present" if not absent else "missing " + ", ".join(absent),
    )

    engine = _dig(cfg, "engine", {})
    tuned = isinstance(engine, Mapping) and len(engine) > 0
    add(
        "engine_tuned",
        tuned,
        "engine: overrides are set" if tuned else "engine: is empty (protocol defaults)",
    )

    stem = _under(root, str(_dig(cfg, "models.temporal", "models/temporal/temporal")))
    npz = stem.with_suffix(".npz")
    add(
        "temporal_model",
        npz.is_file(),
        f"{_rel(npz, root)} present" if npz.is_file() else f"{_rel(npz, root)} missing",
    )

    assert tuple(g) == GATES
    return g


# --------------------------------------------------------------------------------------------
# Shots
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Shot:
    id: str
    title: str
    seconds: int
    who: str = ""
    screen: str = ""
    narration: str | None = None
    narration_sim: str | None = None
    cmd_real: str | None = None
    cmd_sim: str | None = None
    expect: str = ""
    retake_if: str = ""
    notes: str = ""
    needs_any: tuple[str, ...] = ()
    needs_real: tuple[str, ...] = ()
    wants: tuple[str, ...] = ()
    replay_ok: bool = False
    always_simulated: bool = False


_SHOT_TEXT_KEYS = ("who", "screen", "expect", "retake_if", "notes")
_SHOT_OPT_TEXT_KEYS = ("narration", "narration_sim", "cmd_real", "cmd_sim")
_SHOT_GATE_KEYS = ("needs_any", "needs_real", "wants")
_SHOT_KEYS = {
    "id",
    "title",
    "seconds",
    "replay_ok",
    "always_simulated",
    *_SHOT_TEXT_KEYS,
    *_SHOT_OPT_TEXT_KEYS,
    *_SHOT_GATE_KEYS,
}


def _clean(text: Any) -> str:
    return " ".join(str(text).split()) if text is not None else ""


def _clean_block(text: Any) -> str:
    """Collapse folded prose to one line, but keep a multi-line command block as written."""
    s = str(text).strip() if text is not None else ""
    return s if "\n" in s else _clean(s)


def _gate_list(row: Mapping[str, Any], key: str, where: str) -> tuple[str, ...]:
    raw = row.get(key) or []
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise PackError(f"{where}: {key} must be a list of gate names")
    unknown = [x for x in raw if x not in GATES]
    if unknown:
        raise PackError(f"{where}: unknown gate(s) {unknown}; known: {list(GATES)}")
    return tuple(raw)


def parse_shots(data: Mapping[str, Any], where: str = "shots") -> list[Shot]:
    if data.get("version") != SPEC_VERSION:
        raise PackError(f"{where}: version must be {SPEC_VERSION}")
    rows = data.get("shots")
    if not isinstance(rows, list) or not rows:
        raise PackError(f"{where}: `shots` must be a non-empty list")
    shots: list[Shot] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise PackError(f"{where}: shot #{i + 1} is not a mapping")
        label = f"{where}: shot {row.get('id', f'#{i + 1}')}"
        extra = set(row) - _SHOT_KEYS
        if extra:
            raise PackError(f"{label}: unknown key(s) {sorted(extra)}")
        for key in ("id", "title", "seconds"):
            if key not in row:
                raise PackError(f"{label}: missing `{key}`")
        seconds = row["seconds"]
        if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds <= 0:
            raise PackError(f"{label}: `seconds` must be a positive integer")
        kwargs: dict[str, Any] = {
            "id": str(row["id"]),
            "title": _clean(row["title"]),
            "seconds": seconds,
            "replay_ok": bool(row.get("replay_ok", False)),
            "always_simulated": bool(row.get("always_simulated", False)),
        }
        for key in _SHOT_TEXT_KEYS:
            kwargs[key] = _clean(row.get(key))
        for key in _SHOT_OPT_TEXT_KEYS:
            kwargs[key] = _clean_block(row[key]) if row.get(key) else None
        for key in _SHOT_GATE_KEYS:
            kwargs[key] = _gate_list(row, key, label)
        shots.append(_validated(Shot(**kwargs), label))
    ids = [s.id for s in shots]
    dupes = sorted({x for x in ids if ids.count(x) > 1})
    if dupes:
        raise PackError(f"{where}: duplicate shot id(s) {dupes}")
    return shots


def _validated(s: Shot, label: str) -> Shot:
    if s.always_simulated and s.narration:
        raise PackError(f"{label}: an always-simulated shot has only `narration_sim`")
    if not s.always_simulated and not s.narration:
        raise PackError(f"{label}: `narration` is required")
    if (s.replay_ok or s.always_simulated) and not s.narration_sim:
        raise PackError(f"{label}: a shot that can be simulated needs `narration_sim`")
    if s.narration_sim and "simulated" not in s.narration_sim.lower():
        raise PackError(f"{label}: `narration_sim` must say the word 'simulated' out loud")
    if (s.replay_ok or s.always_simulated) and not s.cmd_sim:
        raise PackError(f"{label}: a shot that can be simulated needs `cmd_sim`")
    if s.narration_sim and not (s.replay_ok or s.always_simulated):
        raise PackError(f"{label}: `narration_sim` given but the shot cannot be simulated")
    return s


def load_shots(path: Path) -> list[Shot]:
    return parse_shots(_load_yaml(path), str(path))


@dataclass(frozen=True)
class ShotStatus:
    shot: Shot
    state: str
    unmet_any: tuple[str, ...]
    unmet_real: tuple[str, ...]
    unmet_wants: tuple[str, ...]

    @property
    def variant(self) -> str:
        """Which text applies: ``real`` or ``sim``. A blocked shot shows the draft it will get."""
        if self.state == SIMULATED or (self.shot.always_simulated and self.state != READY):
            return "sim"
        return "real" if self.shot.narration else "sim"


def shot_status(shot: Shot, gates: Mapping[str, Gate]) -> ShotStatus:
    def unmet(names: Sequence[str]) -> tuple[str, ...]:
        return tuple(n for n in names if not gates[n].ok)

    unmet_any, unmet_real, unmet_wants = (
        unmet(shot.needs_any),
        unmet(shot.needs_real),
        unmet(shot.wants),
    )
    if unmet_any:
        state = BLOCKED
    elif shot.always_simulated:
        state = SIMULATED
    elif not unmet_real:
        state = READY
    elif shot.replay_ok:
        state = SIMULATED
    else:
        state = BLOCKED
    return ShotStatus(shot, state, unmet_any, unmet_real, unmet_wants)


def narration_for(st: ShotStatus, tokens: Tokens) -> str:
    text = st.shot.narration_sim if st.variant == "sim" else st.shot.narration
    return tokens.render(text or "")


def command_for(st: ShotStatus) -> str:
    text = st.shot.cmd_sim if st.variant == "sim" else st.shot.cmd_real
    return text or st.shot.cmd_real or st.shot.cmd_sim or ""


def timing_warnings(shots: Sequence[Shot]) -> list[str]:
    """Narrations that do not fit their time, and a total outside the walkthrough's length."""
    out = []
    for s in shots:
        for label, text in (("narration", s.narration), ("narration_sim", s.narration_sim)):
            if not text:
                continue
            n = _words(timing_text(text))
            if n > s.seconds * MAX_WORDS_PER_SECOND:
                out.append(
                    f"{s.id} {label}: {n} words in {s.seconds} s is over "
                    f"{MAX_WORDS_PER_SECOND:g} words/s; shorten it or give the shot more time"
                )
    total = sum(s.seconds for s in shots)
    lo, hi = TOTAL_SECONDS_RANGE
    if not lo <= total <= hi:
        out.append(f"total running time {_mmss(total)} is outside {_mmss(lo)}-{_mmss(hi)}")
    return out


# --------------------------------------------------------------------------------------------
# Limitations
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Limitation:
    id: str
    slide: str
    note: str
    show_if_unmet: tuple[str, ...] = ()
    show_if_met: tuple[str, ...] = ()


def parse_limitations(data: Mapping[str, Any], where: str = "limitations") -> list[Limitation]:
    if data.get("version") != SPEC_VERSION:
        raise PackError(f"{where}: version must be {SPEC_VERSION}")
    rows = data.get("items")
    if not isinstance(rows, list) or not rows:
        raise PackError(f"{where}: `items` must be a non-empty list")
    keys = {"id", "slide", "note", "show_if_unmet", "show_if_met"}
    items: list[Limitation] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise PackError(f"{where}: item #{i + 1} is not a mapping")
        label = f"{where}: item {row.get('id', f'#{i + 1}')}"
        extra = set(row) - keys
        if extra:
            raise PackError(f"{label}: unknown key(s) {sorted(extra)}")
        if not row.get("id") or not row.get("slide") or not row.get("note"):
            raise PackError(f"{label}: `id`, `slide` and `note` are required")
        items.append(
            Limitation(
                id=str(row["id"]),
                slide=_clean(row["slide"]),
                note=_clean(row["note"]),
                show_if_unmet=_gate_list(row, "show_if_unmet", label),
                show_if_met=_gate_list(row, "show_if_met", label),
            )
        )
    ids = [x.id for x in items]
    dupes = sorted({x for x in ids if ids.count(x) > 1})
    if dupes:
        raise PackError(f"{where}: duplicate item id(s) {dupes}")
    return items


def load_limitations(path: Path) -> list[Limitation]:
    return parse_limitations(_load_yaml(path), str(path))


def limitation_shown(item: Limitation, gates: Mapping[str, Gate]) -> tuple[bool, str]:
    """Whether the item is on the slide, and the reason in one line."""
    if item.show_if_unmet and not any(not gates[n].ok for n in item.show_if_unmet):
        return False, "hidden: " + ", ".join(item.show_if_unmet) + " is met"
    if item.show_if_met and not all(gates[n].ok for n in item.show_if_met):
        unmet = [n for n in item.show_if_met if not gates[n].ok]
        return False, "hidden: " + ", ".join(unmet) + " is not met"
    if item.show_if_unmet:
        unmet = [n for n in item.show_if_unmet if not gates[n].ok]
        return True, "shown: " + ", ".join(unmet) + " is not met"
    if item.show_if_met:
        return True, "shown: " + ", ".join(item.show_if_met) + " is met"
    return True, "always shown"


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------


def _quote(text: str) -> str:
    return "\n".join("> " + line for line in textwrap.wrap(text, width=94)) or ">"


def _table(rows: Sequence[Sequence[str]], header: Sequence[str]) -> str:
    def cell(v: str) -> str:
        return str(v).replace("|", "\\|").replace("\n", " ")

    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _banner(report: ReportFacts, statuses: Sequence[ShotStatus]) -> str:
    counts = {s: sum(1 for x in statuses if x.state == s) for s in (READY, SIMULATED, BLOCKED)}
    lines = [
        "> **DRAFT.** Generated by `tools/make_demo_pack.py` from `demo/shots.yaml`, "
        "`demo/limitations.yaml` and the machine's current state. Regenerate it right before "
        "recording.",
        f"> Shots: {counts[READY]} ready, {counts[SIMULATED]} simulated only, "
        f"{counts[BLOCKED]} blocked.",
    ]
    if report.usable:
        lines.append(
            f"> Numbers come from `{report.data.get('eval_id', '?')}` "
            f"({report.data.get('data', '')})."
        )
    else:
        lines.append(
            f"> **No number in this document is a result.** Report: {report.reason}. "
            f"Metric tokens read `{report.placeholder}`."
        )
    return "\n".join(lines)


def render_walkthrough(
    statuses: Sequence[ShotStatus], tokens: Tokens, warnings: Sequence[str]
) -> str:
    total = sum(s.shot.seconds for s in statuses)
    lines = [
        "# ACTUS walkthrough script (DRAFT)",
        "",
        _banner(tokens.report, statuses),
        "",
        f"Running time {_mmss(total)}. Read the narration as written; the shot list "
        "(`shot-list.md`) says what to film. A `SIMULATED` shot is filmed from the replay and "
        "its narration says so out loud; a `BLOCKED` shot cannot be filmed honestly yet.",
        "",
    ]
    if warnings:
        lines += ["## Warnings", ""] + [f"- {w}" for w in warnings] + [""]
    rows = []
    at = 0
    for st in statuses:
        span = f"{_mmss(at)}-{_mmss(at + st.shot.seconds)}"
        rows.append([st.shot.id, st.shot.title, span, st.state])
        at += st.shot.seconds
    lines += [_table(rows, ["shot", "title", "time", "status"]), ""]
    at = 0
    for st in statuses:
        s = st.shot
        span = f"{_mmss(at)}-{_mmss(at + s.seconds)}"
        lines += [f"## {s.id} - {s.title}  ({span}, {s.seconds} s)", ""]
        at += s.seconds
        if st.state == BLOCKED:
            unmet = st.unmet_any + tuple(g for g in st.unmet_real if g not in st.unmet_any)
            lines += [
                f"**BLOCKED - do not record.** Unmet: {', '.join(unmet)}. The draft below is what "
                "would be said once it can be filmed honestly.",
                "",
            ]
        elif st.state == SIMULATED:
            lines += [
                "**SIMULATED.** Film this from the replay. The SIMULATED watermark stays in "
                "frame and the narration says so out loud.",
                "",
            ]
        lines += [_quote(narration_for(st, tokens)), ""]
    return "\n".join(lines).rstrip() + "\n"


def render_shot_list(
    statuses: Sequence[ShotStatus],
    gates: Mapping[str, Gate],
    tokens: Tokens,
    warnings: Sequence[str],
) -> str:
    lines = [
        "# ACTUS shot list (DRAFT)",
        "",
        _banner(tokens.report, statuses),
        "",
        "## Before you film",
        "",
        "Run `uv run python tools/doctor.py` and `uv run python tools/check_markers.py --source 0` "
        "on the demo machine first. Film the correct run (D2) as soon as it works end to end and "
        "keep it as the fallback screen recording (H-REC).",
        "",
        _table(
            [
                [g.name, "yes" if g.ok else "**NO**", g.detail, "" if g.ok else g.fix]
                for g in gates.values()
            ],
            ["gate", "met", "state", "how to fix"],
        ),
        "",
        "## Rules for every take",
        "",
        "- The SIMULATED watermark is never cropped, covered or edited out. A viewer must be able "
        "to tell a replay from a recording.",
        "- Never film a Validation tab that shows a SIMULATED or empty report.",
        "- A take that fails its `retake_if` is thrown away, unless the failure is itself a "
        "finding: then keep it, and put it on the limitations slide.",
        "- Say a failed target out loud. Do not reword a target to make it pass.",
        "",
    ]
    if warnings:
        lines += ["## Warnings", ""] + [f"- {w}" for w in warnings] + [""]
    at = 0
    for st in statuses:
        s = st.shot
        lines += [
            f"## {s.id} - {s.title}",
            "",
            f"- **Status:** {st.state}"
            + (f" (unmet: {', '.join(st.unmet_any + st.unmet_real)})" if st.state != READY else ""),
            f"- **Time:** {_mmss(at)}-{_mmss(at + s.seconds)} ({s.seconds} s); "
            f"narration {_words(timing_text(narration_for_template(st)))} words",
        ]
        at += s.seconds
        if s.who:
            lines.append(f"- **Who:** {s.who}")
        if s.screen:
            lines.append(f"- **On screen:** {s.screen}")
        cmd = command_for(st)
        if cmd:
            if "\n" in cmd:
                lines += ["- **Command:**", "", "```sh", cmd, "```", ""]
            else:
                lines.append(f"- **Command:** `{cmd}`")
        if st.state == BLOCKED and st.shot.cmd_real:
            lines.append("  (blocked: this is the command for when it can be filmed)")
        if s.expect:
            lines.append(f"- **See and hear:** {s.expect}")
        if s.retake_if:
            lines.append(f"- **Retake if:** {s.retake_if}")
        if st.unmet_wants:
            lines.append(
                "- **Degrades without:** "
                + ", ".join(f"{g} ({gates[g].detail})" for g in st.unmet_wants)
            )
        if s.notes:
            lines.append(f"- **Notes:** {s.notes}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def narration_for_template(st: ShotStatus) -> str:
    text = st.shot.narration_sim if st.variant == "sim" else st.shot.narration
    return text or ""


@dataclass(frozen=True)
class RenderedLimitation:
    item: Limitation
    shown: bool
    why: str
    slide: str


def render_limitations(
    items: Sequence[Limitation], gates: Mapping[str, Gate], tokens: Tokens
) -> tuple[str, list[RenderedLimitation]]:
    rendered = []
    for it in items:
        shown, why = limitation_shown(it, gates)
        rendered.append(RenderedLimitation(it, shown, why, tokens.render(it.slide)))
    shown_items = [r for r in rendered if r.shown]
    on_slide, backup = shown_items[:SLIDE_MAX_ITEMS], shown_items[SLIDE_MAX_ITEMS:]
    lines = [
        "# ACTUS limitations slide (DRAFT)",
        "",
        "> Generated by `tools/make_demo_pack.py`. Which items appear depends on what is fixed "
        "on the machine that generated it; regenerate right before recording. Nothing is "
        "dropped: what does not fit the slide is on the backup slide.",
        "",
        "## Slide",
        "",
        "**What we do not claim**",
        "",
    ]
    lines += [f"{i}. {r.slide}" for i, r in enumerate(on_slide, 1)]
    if backup:
        lines += ["", "## Backup slide", ""]
        lines += [f"{i}. {r.slide}" for i, r in enumerate(backup, len(on_slide) + 1)]
    lines += [
        "",
        "## Where each item comes from",
        "",
        _table(
            [
                [
                    r.item.id,
                    "slide" if r in on_slide else "backup" if r in backup else "-",
                    r.why,
                    r.item.note,
                ]
                for r in rendered
            ],
            ["item", "on", "why", "source"],
        ),
        "",
    ]
    return "\n".join(lines), rendered


# --------------------------------------------------------------------------------------------
# The pack
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pack:
    files: dict[str, str]
    statuses: tuple[ShotStatus, ...]
    gates: dict[str, Gate]
    limitations: tuple[RenderedLimitation, ...]
    warnings: tuple[str, ...]
    report: ReportFacts


def build_pack(
    root: Path,
    *,
    report_path: Path | None = None,
    shots_path: Path | None = None,
    limitations_path: Path | None = None,
) -> Pack:
    """Render the whole pack in memory. Raises ``PackError`` on inconsistent inputs."""
    root = Path(root)
    report_path = report_path or root / "runs" / "eval" / "report.json"
    shots_path = shots_path or root / "demo" / "shots.yaml"
    limitations_path = limitations_path or root / "demo" / "limitations.yaml"

    shots = load_shots(shots_path)
    items = load_limitations(limitations_path)
    report = read_report(report_path)
    tokens = Tokens(report)
    gates = check_gates(root, report)

    for s in shots:  # fail loudly on a token nobody can resolve
        for text in (s.narration, s.narration_sim):
            tokens.render(text or "")
    statuses = tuple(shot_status(s, gates) for s in shots)
    warnings = timing_warnings(shots)

    limitations_md, rendered = render_limitations(items, gates, tokens)
    for r in rendered:
        n = _words(timing_text(r.item.slide))
        if n > SLIDE_ITEM_MAX_WORDS:
            warnings.append(f"limitation {r.item.id}: {n} words is over {SLIDE_ITEM_MAX_WORDS}")

    manifest = {
        "generator": "tools/make_demo_pack.py",
        "spec_version": SPEC_VERSION,
        "inputs": {
            "shots": {"path": _rel(shots_path, root), "sha256": _sha256(shots_path)},
            "limitations": {
                "path": _rel(limitations_path, root),
                "sha256": _sha256(limitations_path),
            },
            "report": {
                "path": _rel(report_path, root),
                "sha256": report.sha256,
                "usable": report.usable,
                "reason": report.reason,
                "eval_id": report.data.get("eval_id") if report.usable else None,
            },
        },
        "gates": {n: {"ok": g.ok, "detail": g.detail} for n, g in gates.items()},
        "shots": [
            {
                "id": st.shot.id,
                "state": st.state,
                "seconds": st.shot.seconds,
                "unmet_any": list(st.unmet_any),
                "unmet_real": list(st.unmet_real),
                "unmet_wants": list(st.unmet_wants),
            }
            for st in statuses
        ],
        "limitations": [{"id": r.item.id, "shown": r.shown, "why": r.why} for r in rendered],
        "total_seconds": sum(s.seconds for s in shots),
        "warnings": list(warnings),
    }
    files = {
        "walkthrough.md": render_walkthrough(statuses, tokens, warnings),
        "shot-list.md": render_shot_list(statuses, gates, tokens, warnings),
        "limitations.md": limitations_md,
        "manifest.json": json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    }
    return Pack(files, statuses, gates, tuple(rendered), tuple(warnings), report)


def write_pack(pack: Pack, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in pack.files.items():
        p = out_dir / name
        p.write_text(text, encoding="utf-8", newline="\n")
        written.append(p)
    return written


def summary(pack: Pack) -> str:
    """The console summary. ASCII only (the Windows console is cp1252)."""
    rows = [f"{st.shot.id:<4} {st.state:<10} {st.shot.title}" for st in pack.statuses]
    unmet = [g for g in pack.gates.values() if not g.ok]
    out = ["shots:"] + [f"  {r}" for r in rows]
    out.append("gates not met:" if unmet else "gates: all met")
    out += [f"  {g.name}: {g.detail}" for g in unmet]
    shown = sum(1 for r in pack.limitations if r.shown)
    out.append(f"limitations shown: {shown} of {len(pack.limitations)}")
    if pack.report.usable:
        out.append(f"report: {pack.report.data.get('eval_id', '?')}")
    else:
        out.append(f"report: NOT USABLE - {pack.report.reason}")
    out += [f"warning: {w}" for w in pack.warnings]
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Draft the walkthrough script, shot list and limitations slide."
    )
    ap.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    ap.add_argument("--report", type=Path, help="default: <root>/runs/eval/report.json")
    ap.add_argument("--shots", type=Path, help="default: <root>/demo/shots.yaml")
    ap.add_argument("--limitations", type=Path, help="default: <root>/demo/limitations.yaml")
    ap.add_argument("--out", type=Path, help="default: <root>/demo/out")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 unless every shot is READY (a SIMULATED or BLOCKED shot fails)",
    )
    ap.add_argument("--check", action="store_true", help="validate and print; write nothing")
    args = ap.parse_args(argv)

    root = args.root.resolve()
    try:
        pack = build_pack(
            root,
            report_path=args.report,
            shots_path=args.shots,
            limitations_path=args.limitations,
        )
    except PackError as e:
        print(f"make_demo_pack: {e}", file=sys.stderr)
        return 2
    print(summary(pack))
    if not args.check:
        out = args.out or root / "demo" / "out"
        for p in write_pack(pack, out):
            print(f"wrote {_rel(p, root)}")
    if args.strict and any(st.state != READY for st in pack.statuses):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
