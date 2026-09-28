"""T-DEMO: the walkthrough / shot list / limitations generator (tools/make_demo_pack.py).

The tests build a throwaway repo root, so what the demo machine happens to have on disk (models,
binaries, a report) never changes the result. The two real input files (demo/shots.yaml and
demo/limitations.yaml) are checked as they stand, because they are the deliverable."""

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.make_demo_pack import (  # noqa: E402
    BLOCKED,
    GATES,
    MAX_WORDS_PER_SECOND,
    METRICS,
    READY,
    SIMULATED,
    SLIDE_ITEM_MAX_WORDS,
    SLIDE_MAX_ITEMS,
    STATIC_TOKENS,
    TOTAL_SECONDS_RANGE,
    Gate,
    PackError,
    Shot,
    Tokens,
    build_pack,
    check_gates,
    load_limitations,
    load_shots,
    main,
    parse_limitations,
    parse_shots,
    read_report,
    shot_status,
    timing_text,
    timing_warnings,
)

SHOTS = REPO_ROOT / "demo" / "shots.yaml"
LIMITS = REPO_ROOT / "demo" / "limitations.yaml"
PRECISION = "0.9137"  # a value nobody would type by hand: if it shows up, it came from a report


# --------------------------------------------------------------------------------------------
# Fixtures: a throwaway repo root and a report
# --------------------------------------------------------------------------------------------


def make_root(
    tmp_path: Path,
    *,
    rack=False,
    detector=False,
    media=False,
    ui=False,
    asr=False,
    tts=False,
    engine=False,
    temporal=False,
) -> Path:
    """A repo root holding only what the flags ask for. The paths mirror configs/default.yaml."""
    cfg = {
        "rack": {"config": "configs/rack.yaml"},
        "detector": {
            "backend": "custom" if detector else "open_vocab",
            "custom_weights": "models/detector/best.onnx",
        },
        "stream": {"mediamtx_bin": "bin/mediamtx", "ffmpeg_bin": "bin/ffmpeg"},
        "ui": {"static_dir": "ui/dist"},
        "voice": {
            "asr": {"model": "models/voice/vosk"},
            "tts": {
                "kokoro_model": "models/voice/kokoro.onnx",
                "kokoro_voices": "models/voice/voices.bin",
            },
        },
        "models": {"temporal": "models/temporal/temporal"},
        "engine": {"vis_ask_after_s": 1.5} if engine else {},
    }
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "configs" / "default.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (tmp_path / "configs" / "rack.yaml").write_text(f"measured: {str(rack).lower()}\n")

    def touch(rel: str) -> None:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")

    if detector:
        touch("models/detector/best.onnx")
        touch("models/detector/card.md")
    if media:
        touch("bin/mediamtx")
        touch("bin/ffmpeg")
    if ui:
        touch("ui/dist/index.html")
    if asr:
        touch("models/voice/vosk/am/final.mdl")
    if tts:
        touch("models/voice/kokoro.onnx")
        touch("models/voice/voices.bin")
    if temporal:
        touch("models/temporal/temporal.npz")
    return tmp_path


def make_ready_root(tmp_path: Path) -> Path:
    return make_root(
        tmp_path,
        rack=True,
        detector=True,
        media=True,
        ui=True,
        asr=True,
        tts=True,
        engine=True,
        temporal=True,
    )


def report_dict(**over) -> dict:
    d = {
        "data": "REAL held-out takes (test split)",
        "criteria_met": "7/10 (1 n/a)",
        "takes_evaluated": "16",
        "operators": "op05",
        "criteria_version": "v1 (frozen 2026-09-28)",
        "eval_id": "eval-fixture",
        "simulated": False,
        "criteria": [
            {"metric": m, "target": ">= 0.9", "value": "0.5", "result": "FAIL"} for m in METRICS
        ],
    }
    d["criteria"][0] = {
        "metric": "step_completion_precision",
        "target": ">= 0.9",
        "value": PRECISION,
        "result": "PASS",
    }
    d.update(over)
    return d


def write_report(tmp_path: Path, **over) -> Path:
    p = tmp_path / "report.json"
    p.write_text(json.dumps(report_dict(**over)), encoding="utf-8")
    return p


def pack(root: Path, report: Path | None = None):
    return build_pack(
        root,
        report_path=report or root / "no-report.json",
        shots_path=SHOTS,
        limitations_path=LIMITS,
    )


def all_text(p) -> str:
    return "\n".join(p.files.values())


def flat(text: str) -> str:
    """Markdown with the blockquote markers and line wrapping undone."""
    return " ".join(text.replace(">", " ").split())


def state_of(p, shot_id: str) -> str:
    return next(s.state for s in p.statuses if s.shot.id == shot_id)


# --------------------------------------------------------------------------------------------
# The shipped input files
# --------------------------------------------------------------------------------------------


def test_metric_names_match_the_evaluation_harness():
    from tools.evaluate import METRICS as EVAL_METRICS

    assert METRICS == EVAL_METRICS


def test_the_input_files_are_ascii():
    # Shot titles reach the Windows console (cp1252) through the summary.
    for p in (SHOTS, LIMITS):
        assert p.read_text(encoding="utf-8").isascii(), p


def test_shots_load_and_fit_the_walkthrough_length():
    shots = load_shots(SHOTS)
    lo, hi = TOTAL_SECONDS_RANGE
    assert lo <= sum(s.seconds for s in shots) <= hi
    assert timing_warnings(shots) == []


def test_the_walkthrough_follows_the_plan_and_ends_on_the_limitations():
    ids = [s.id for s in load_shots(SHOTS)]
    assert ids[0] == "D1" and ids[-1] == "D11"
    # The plan's ten items, in order: architecture, correct run, out of order, skip, ask, roll,
    # faults, artefacts, second protocol, validation + limitations.
    assert ids == ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8a", "D8b", "D9", "D10", "D11"]


def test_every_simulated_narration_says_so_out_loud():
    for s in load_shots(SHOTS):
        if s.narration_sim:
            assert "simulated" in s.narration_sim.lower(), s.id


def test_a_shot_that_can_be_simulated_has_a_command_for_it():
    for s in load_shots(SHOTS):
        if s.replay_ok or s.always_simulated:
            assert s.cmd_sim and s.narration_sim, s.id


def test_the_validation_shot_cannot_be_filmed_from_the_replay():
    d10 = next(s for s in load_shots(SHOTS) if s.id == "D10")
    assert "report_real" in d10.needs_real
    assert not d10.replay_ok and not d10.always_simulated and d10.narration_sim is None


def test_every_token_in_the_shots_and_limitations_is_known():
    tokens = Tokens(read_report(None))
    texts = []
    for s in load_shots(SHOTS):
        texts += [s.narration or "", s.narration_sim or ""]
    texts += [x.slide for x in load_limitations(LIMITS)]
    for text in texts:
        for name in Tokens.names_in(text):
            tokens.resolve(name)  # raises PackError on an unknown token


def test_limitations_are_short_enough_for_a_slide():
    for x in load_limitations(LIMITS):
        assert len(timing_text(x.slide).split()) <= SLIDE_ITEM_MAX_WORDS, x.id


def test_the_always_shown_limitations_include_the_plans_honesty_list():
    ids = {x.id for x in load_limitations(LIMITS) if not x.show_if_met and not x.show_if_unmet}
    assert {
        "judge_not_model",
        "small_data",
        "rotation_emulated",
        "steps_are_our_reading",
        "not_flight_hardware",
    } <= ids


# --------------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------------


def test_a_missing_report_is_not_usable(tmp_path):
    r = read_report(tmp_path / "nope.json")
    assert not r.usable and "NOT MEASURED" in r.placeholder
    assert not read_report(None).usable


def test_a_broken_report_is_not_usable(tmp_path):
    p = tmp_path / "report.json"
    p.write_text("{not json", encoding="utf-8")
    assert not read_report(p).usable
    p.write_text("[1, 2]", encoding="utf-8")
    assert not read_report(p).usable


def test_a_simulated_report_is_never_usable(tmp_path):
    r = read_report(write_report(tmp_path, simulated=True))
    assert not r.usable and "SIMULATED" in r.placeholder


def test_a_report_that_evaluated_no_takes_is_not_usable(tmp_path):
    assert not read_report(write_report(tmp_path, takes_evaluated="0")).usable
    assert not read_report(write_report(tmp_path, takes_evaluated="many")).usable


def test_a_real_report_is_usable(tmp_path):
    r = read_report(write_report(tmp_path))
    assert r.usable and r.reason == "" and r.sha256


# --------------------------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------------------------


def test_tokens_read_the_report(tmp_path):
    t = Tokens(read_report(write_report(tmp_path)))
    assert t.resolve("m:step_completion_precision") == PRECISION
    assert t.resolve("t:step_completion_precision") == ">= 0.9"
    assert t.resolve("v:step_completion_precision") == "PASS"
    assert t.resolve("n_takes") == "16"
    assert t.resolve("n_operators") == "1"
    assert t.resolve("criteria_met") == "7/10 (1 n/a)"
    assert t.render("p={{ m:step_completion_precision }}!") == f"p={PRECISION}!"


def test_tokens_never_produce_a_number_without_a_real_report(tmp_path):
    for report in (None, write_report(tmp_path, simulated=True)):
        t = Tokens(read_report(report))
        for name in [*STATIC_TOKENS, *(f"m:{m}" for m in METRICS)]:
            out = t.resolve(name)
            assert out.startswith("[") and PRECISION not in out, name


def test_an_unknown_token_is_an_error(tmp_path):
    t = Tokens(read_report(write_report(tmp_path)))
    for bad in ("nope", "m:not_a_metric", "x:step_completion_precision"):
        with pytest.raises(PackError):
            t.resolve(bad)
    with pytest.raises(PackError):
        t.render("{{ nope }}")


def test_a_metric_missing_from_a_real_report_is_flagged(tmp_path):
    p = write_report(tmp_path, criteria=[])
    assert "NOT MEASURED" in Tokens(read_report(p)).resolve("m:ask_rate_per_min")


# --------------------------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------------------------


def test_a_bare_root_meets_no_gate(tmp_path):
    gates = check_gates(make_root(tmp_path), read_report(None))
    assert tuple(gates) == GATES
    assert not any(g.ok for g in gates.values())
    assert all(g.detail and g.fix for g in gates.values())


def test_a_ready_root_meets_every_gate(tmp_path):
    root = make_ready_root(tmp_path / "root")
    report = read_report(write_report(tmp_path))
    gates = check_gates(root, report)
    assert all(g.ok for g in gates.values()), {n: g.detail for n, g in gates.items() if not g.ok}


@pytest.mark.parametrize(
    ("flag", "gate"),
    [
        ("rack", "rack_measured"),
        ("detector", "detector_custom"),
        ("media", "media_binaries"),
        ("ui", "ui_built"),
        ("asr", "asr_model"),
        ("tts", "tts_model"),
        ("engine", "engine_tuned"),
        ("temporal", "temporal_model"),
    ],
)
def test_each_gate_follows_exactly_its_own_files(tmp_path, flag, gate):
    gates = check_gates(make_root(tmp_path, **{flag: True}), read_report(None))
    assert gates[gate].ok
    assert not any(g.ok for n, g in gates.items() if n != gate)


def test_a_detector_without_its_model_card_is_not_a_trained_detector(tmp_path):
    root = make_root(tmp_path)
    (root / "models" / "detector").mkdir(parents=True)
    (root / "models" / "detector" / "best.onnx").write_bytes(b"x")
    g = check_gates(root, read_report(None))["detector_custom"]
    assert not g.ok and "card.md" in g.detail


def test_trained_weights_are_not_enough_if_the_config_still_selects_open_vocab(tmp_path):
    root = make_root(tmp_path, detector=True)
    assert check_gates(root, read_report(None))["detector_custom"].ok
    cfg_path = root / "configs" / "default.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["detector"]["backend"] = "open_vocab"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    g = check_gates(root, read_report(None))["detector_custom"]
    assert not g.ok and "open_vocab" in g.detail


def test_a_placeholder_rack_is_not_measured(tmp_path):
    root = make_root(tmp_path)
    (root / "configs" / "rack.yaml").write_text("measured: false\nboard: {}\n")
    assert not check_gates(root, read_report(None))["rack_measured"].ok
    (root / "configs" / "rack.yaml").unlink()
    assert not check_gates(root, read_report(None))["rack_measured"].ok


def test_a_missing_mediamtx_blocks_media_even_with_ffmpeg(tmp_path):
    root = make_root(tmp_path)
    (root / "bin").mkdir()
    (root / "bin" / "ffmpeg").write_bytes(b"x")
    g = check_gates(root, read_report(None))["media_binaries"]
    assert not g.ok and "mediamtx" in g.detail and "ffmpeg" not in g.detail


# --------------------------------------------------------------------------------------------
# Shot status
# --------------------------------------------------------------------------------------------


def _gates(**met) -> dict[str, Gate]:
    return {n: Gate(n, bool(met.get(n, False)), "d", "f") for n in GATES}


def _shot(**kw) -> Shot:
    base = {
        "id": "X",
        "title": "t",
        "seconds": 10,
        "narration": "Words.",
        "narration_sim": "Simulated words.",
        "cmd_real": "real",
        "cmd_sim": "sim",
    }
    return Shot(**{**base, **kw})


def test_a_shot_with_every_gate_met_is_ready():
    st = shot_status(
        _shot(needs_any=("ui_built",), needs_real=("rack_measured",), replay_ok=True),
        _gates(ui_built=True, rack_measured=True),
    )
    assert st.state == READY and st.variant == "real"


def test_an_unmet_real_gate_falls_back_to_the_replay_when_allowed():
    st = shot_status(_shot(needs_real=("rack_measured",), replay_ok=True), _gates())
    assert st.state == SIMULATED and st.variant == "sim"
    assert st.unmet_real == ("rack_measured",)


def test_an_unmet_real_gate_blocks_a_shot_that_cannot_be_simulated():
    st = shot_status(_shot(needs_real=("report_real",), narration_sim=None), _gates())
    assert st.state == BLOCKED


def test_an_unmet_universal_gate_blocks_even_the_replay():
    st = shot_status(_shot(needs_any=("media_binaries",), replay_ok=True), _gates())
    assert st.state == BLOCKED and st.unmet_any == ("media_binaries",)


def test_an_always_simulated_shot_is_never_ready():
    s = _shot(narration=None, always_simulated=True)
    assert shot_status(s, _gates()).state == SIMULATED
    assert shot_status(s, _gates()).variant == "sim"
    gated = _shot(narration=None, always_simulated=True, needs_any=("ui_built",))
    assert shot_status(gated, _gates()).state == BLOCKED


def test_a_soft_gate_never_changes_the_state():
    st = shot_status(_shot(wants=("asr_model",)), _gates())
    assert st.state == READY and st.unmet_wants == ("asr_model",)


# --------------------------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------------------------


def _doc(**shot) -> dict:
    base = {"id": "A", "title": "t", "seconds": 10, "narration": "Words."}
    return {"version": 1, "shots": [{**base, **shot}]}


def test_valid_minimal_shot_parses():
    assert parse_shots(_doc())[0].id == "A"


@pytest.mark.parametrize(
    "bad",
    [
        {"needs_real": ["no_such_gate"]},
        {"wants": "ui_built"},
        {"seconds": 0},
        {"seconds": "ten"},
        {"unknown_key": 1},
        {"narration": None},  # a real shot needs narration
        {"replay_ok": True},  # simulable, but no narration_sim / cmd_sim
        {"replay_ok": True, "narration_sim": "Words.", "cmd_sim": "x"},  # never says simulated
        {"narration_sim": "Simulated words.", "cmd_sim": "x"},  # not simulable
        {"always_simulated": True},  # must not carry a real narration
    ],
)
def test_bad_shots_are_rejected(bad):
    with pytest.raises(PackError):
        parse_shots(_doc(**bad))


def test_duplicate_ids_and_wrong_versions_are_rejected():
    d = _doc()
    d["shots"].append(dict(d["shots"][0]))
    with pytest.raises(PackError):
        parse_shots(d)
    with pytest.raises(PackError):
        parse_shots({"version": 2, "shots": d["shots"][:1]})
    with pytest.raises(PackError):
        parse_shots({"version": 1, "shots": []})


def test_bad_limitations_are_rejected():
    ok = {"id": "a", "slide": "Words.", "note": "src"}
    assert parse_limitations({"version": 1, "items": [ok]})[0].id == "a"
    for bad in (
        {**ok, "show_if_unmet": ["no_such_gate"]},
        {**ok, "extra": 1},
        {"id": "a", "slide": "Words."},
    ):
        with pytest.raises(PackError):
            parse_limitations({"version": 1, "items": [bad]})
    with pytest.raises(PackError):
        parse_limitations({"version": 1, "items": [ok, ok]})


def test_the_time_budget_flags_a_narration_that_cannot_be_spoken_in_time():
    words = " ".join(["word"] * int(10 * MAX_WORDS_PER_SECOND + 5))
    warns = timing_warnings([_shot(narration=words, seconds=10)])
    assert any("words/s" in w for w in warns)
    assert any("total running time" in w for w in warns)


def test_a_token_counts_as_one_word_in_the_budget():
    assert timing_text("a {{ m:ask_rate_per_min }} b").split() == ["a", "0.00", "b"]


def test_an_unknown_token_in_a_shot_fails_the_build(tmp_path):
    root = make_root(tmp_path / "root")
    shots = tmp_path / "shots.yaml"
    shots.write_text(yaml.safe_dump(_doc(narration="Ten {{ nope }} words.")), encoding="utf-8")
    with pytest.raises(PackError):
        build_pack(root, shots_path=shots, limitations_path=LIMITS)


# --------------------------------------------------------------------------------------------
# The pack
# --------------------------------------------------------------------------------------------


def test_on_a_bare_machine_the_pack_claims_nothing(tmp_path):
    p = pack(make_root(tmp_path))
    text = all_text(p)
    assert "No number in this document is a result" in p.files["walkthrough.md"]
    assert "[NOT MEASURED - no report]" in p.files["walkthrough.md"]
    assert state_of(p, "D1") == READY and state_of(p, "D11") == READY
    assert state_of(p, "D10") == BLOCKED
    assert state_of(p, "D2") == BLOCKED  # the GUI is not built
    assert state_of(p, "D9") == SIMULATED  # by design; it needs nothing on disk
    assert "BLOCKED - do not record" in p.files["walkthrough.md"]
    assert PRECISION not in text


def test_with_the_gui_built_the_replay_shots_become_simulated_not_ready(tmp_path):
    p = pack(make_root(tmp_path, ui=True, media=True))
    for sid in ("D2", "D3", "D4", "D5", "D7", "D8a", "D8b"):
        assert state_of(p, sid) == SIMULATED, sid
    assert state_of(p, "D6") == BLOCKED  # cannot be shot from the replay
    assert state_of(p, "D10") == BLOCKED  # replay numbers are not validation
    assert state_of(p, "D9") == SIMULATED
    wt = flat(p.files["walkthrough.md"])
    assert "Simulated scene." in wt and "**SIMULATED.**" in wt


def test_simulated_shots_use_their_simulated_narration(tmp_path):
    p = pack(make_root(tmp_path, ui=True))
    d2 = next(s for s in p.statuses if s.shot.id == "D2")
    assert d2.variant == "sim"
    wt = flat(p.files["walkthrough.md"])
    assert "This is a simulated run" in wt
    assert "Here is ISRO's sample experiment, run live" not in wt


def test_a_ready_machine_with_a_real_report_films_everything_for_real(tmp_path):
    root = make_ready_root(tmp_path / "root")
    p = pack(root, write_report(tmp_path))
    # D9 is the one shot that can only ever be simulated.
    assert [s.shot.id for s in p.statuses if s.state != READY] == ["D9"]
    assert state_of(p, "D9") == SIMULATED
    wt = flat(p.files["walkthrough.md"])
    assert f"Step precision {PRECISION}" in wt
    assert "7/10 (1 n/a) pre-registered targets met" in wt
    assert "[NOT MEASURED" not in wt
    assert "Here is ISRO's sample experiment, run live" in wt
    assert p.warnings == ()


def test_a_simulated_report_never_reaches_the_documents(tmp_path):
    root = make_ready_root(tmp_path / "root")
    p = pack(root, write_report(tmp_path, simulated=True))
    assert PRECISION not in all_text(p)
    assert "[SIMULATED - DO NOT QUOTE]" in flat(p.files["walkthrough.md"])
    assert state_of(p, "D10") == BLOCKED


def test_limitations_follow_the_gaps(tmp_path):
    bare = pack(make_root(tmp_path / "bare"))
    gaps = {"no_numbers_yet", "detector_not_trained", "rack_not_measured", "thresholds_untuned"}
    shown = {r.item.id for r in bare.limitations if r.shown}
    assert gaps <= shown
    assert {"held_out_size", "asks_simulated_operator", "temporal_small_data"}.isdisjoint(shown)
    md = bare.files["limitations.md"]
    assert md.index("No accuracy numbers are claimed yet") < md.index("state machine")

    ready = pack(make_ready_root(tmp_path / "ready"), write_report(tmp_path))
    shown = {r.item.id for r in ready.limitations if r.shown}
    assert gaps.isdisjoint(shown)
    assert {"held_out_size", "asks_simulated_operator", "temporal_small_data"} <= shown
    assert "The held-out set is 16 takes from operators op05." in ready.files["limitations.md"]


def test_the_slide_holds_six_and_nothing_is_dropped(tmp_path):
    p = pack(make_root(tmp_path))
    shown = [r for r in p.limitations if r.shown]
    assert len(shown) > SLIDE_MAX_ITEMS
    md = p.files["limitations.md"]
    slide, _, rest = md.partition("## Backup slide")
    assert slide.count("\n1. ") == 1 and f"\n{SLIDE_MAX_ITEMS}. " in slide
    assert f"\n{SLIDE_MAX_ITEMS + 1}. " not in slide
    for r in shown:
        assert r.slide in md
    assert rest  # the overflow is on the backup slide


def test_the_pack_is_a_pure_function_of_its_inputs(tmp_path):
    root = make_ready_root(tmp_path / "root")
    report = write_report(tmp_path)
    assert pack(root, report).files == pack(root, report).files


def test_the_manifest_records_inputs_gates_and_statuses(tmp_path):
    p = pack(make_root(tmp_path, ui=True))
    m = json.loads(p.files["manifest.json"])
    assert m["gates"]["ui_built"]["ok"] is True and m["gates"]["rack_measured"]["ok"] is False
    assert {s["id"] for s in m["shots"]} >= {"D1", "D10", "D11"}
    assert m["inputs"]["shots"]["sha256"] and m["inputs"]["report"]["usable"] is False
    assert m["total_seconds"] == sum(s.shot.seconds for s in p.statuses)
    assert m["warnings"] == []


def test_the_shot_list_says_what_is_missing_and_how_to_fix_it(tmp_path):
    sl = pack(make_root(tmp_path, ui=True)).files["shot-list.md"]
    assert "## Before you film" in sl and "**NO**" in sl
    assert "measure the board into configs/rack.yaml" in sl
    assert "tools/doctor.py" in sl
    assert "Degrades without" in sl  # D2 wants the Kokoro voice
    assert "```sh" in sl and "printf 'X' | dd" in sl  # the tamper demo is a runnable block


# --------------------------------------------------------------------------------------------
# The CLI
# --------------------------------------------------------------------------------------------


def _cli(root: Path, *extra: str) -> list[str]:
    """The CLI's arguments; a later flag in ``extra`` overrides an earlier one."""
    return [
        "--root",
        str(root),
        "--shots",
        str(SHOTS),
        "--limitations",
        str(LIMITS),
        "--report",
        str(root / "no-report.json"),
        *extra,
    ]


def test_the_cli_writes_the_four_files_and_prints_ascii(tmp_path, capsys):
    root = make_root(tmp_path)
    assert main(_cli(root)) == 0
    out = capsys.readouterr().out
    assert out.isascii() and "BLOCKED" in out and "report: NOT USABLE" in out
    written = sorted(p.name for p in (root / "demo" / "out").iterdir())
    assert written == ["limitations.md", "manifest.json", "shot-list.md", "walkthrough.md"]


def test_strict_fails_until_every_shot_is_ready(tmp_path):
    assert main(_cli(make_root(tmp_path / "a"), "--strict")) == 1
    # Even a fully equipped machine with a real report has D9, which is only ever simulated.
    report = str(write_report(tmp_path))
    ready = make_ready_root(tmp_path / "b")
    assert main(_cli(ready, "--report", report, "--strict")) == 1
    assert main(_cli(ready, "--report", report)) == 0


def test_check_mode_writes_nothing(tmp_path):
    root = make_root(tmp_path)
    assert main(_cli(root, "--check")) == 0
    assert not (root / "demo").exists()


def test_the_cli_exits_2_on_bad_input(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 1\nshots: []\n", encoding="utf-8")
    root = make_root(tmp_path / "root")
    assert main(["--root", str(root), "--shots", str(bad), "--limitations", str(LIMITS)]) == 2
    assert "make_demo_pack" in capsys.readouterr().err
