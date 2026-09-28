"""T-PROTO2 demo run: the second protocol through the real pipeline core, on a SIMULATED scene.

``tools/demo_protocol2.py`` draws a cup / bottle / book scene with real ArUco markers and feeds
scripted detections and hand keypoints through the rack solve, the world-state filter, the step
evidence and the protocol engine (``PipelineCore``), then the hash-chained log and the downlink
packet. Nothing here needs a camera, model weights, or a GPU; nothing here is a perception metric.

The assertions on the run outcomes are about behaviour (which steps end verified / skipped /
operator-confirmed, which alerts fire), not exact frame timings.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.demo_protocol2 as demo  # noqa: E402
from actus.core.config import load_config  # noqa: E402
from actus.core.contracts import AlertKind, Ask, Prompt, StepStatus  # noqa: E402
from actus.io.eventlog import verify_log  # noqa: E402
from actus.protocol.engine import DONE  # noqa: E402
from actus.protocol.schema import load_protocol  # noqa: E402
from actus.spatial.zones import zones_from_protocol  # noqa: E402

STEP_IDS = ["S1", "S2", "S3", "S4", "S5"]
HANDLED = {"S1": "cup", "S2": "bottle", "S3": "bottle", "S4": "book", "S5": "book"}
BOOK_S5_ASK = "I couldn't confirm the book is in the record area. Did you place it?"


@pytest.fixture(scope="module")
def proto():
    return load_protocol(load_config(demo.CONFIG_PATH).protocol.path)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """``runs(name, **kw)`` -> DemoResult; each distinct call runs once for the whole module."""
    root = tmp_path_factory.mktemp("proto2")
    cfg = load_config(demo.CONFIG_PATH, overrides=[f"run.runs_dir={root.as_posix()}"])
    cache: dict = {}

    def get(name, **kw):
        key = (name, tuple(sorted(kw.items())))
        if key not in cache:
            cache[key] = demo.run_scenario(
                name, cfg, run_id=f"{name}-{len(cache)}", video=False, **kw
            )
        return cache[key]

    return get


# ---- the scene, without running anything ---------------------------------------------------------


def test_scene_layout_sits_in_the_intended_zones(proto):
    zones = zones_from_protocol(proto)

    def depth(zone, xy):  # metres inside the zone (negative = outside)
        return float(zones[zone].signed_distance(xy))

    assert depth("M", demo.DEST["cup"]) >= 0.06
    assert depth("M", demo.DEST["bottle"]) >= 0.06
    assert depth("R", demo.DEST["book"]) >= 0.06
    assert depth("S", demo.START["bottle"]) >= 0.04
    assert depth("S", demo.START["book"]) >= 0.04
    assert all(depth(z, demo.START["cup"]) < -0.05 for z in zones)  # the cup starts in no zone
    assert abs(depth("R", demo.EDGE_OF_R)) < 1e-9  # the uncallable placement is on the edge


def test_correct_timeline_moves_each_object_once_from_start_to_place():
    tl = demo.build_timeline(demo.SCENARIOS["correct"].items)
    for o in demo.OBJECTS:
        assert tl.objects_at(0.0)[o] == pytest.approx(demo.START[o])
        assert tl.objects_at(tl.duration_s)[o] == pytest.approx(demo.DEST[o])
    assert tl.hand_at(0.0) == pytest.approx(demo.REST)
    assert tl.hand_at(tl.duration_s) == pytest.approx(demo.REST)
    assert [name for _, _, name in tl.labels] == ["S1", "S1", "S2", "S3", "S4", "S5"]
    assert tl.label_at(0.0) == "idle" and tl.blackouts == ()


def test_occluded_scenario_blocks_the_view_as_the_book_is_placed():
    tl = demo.build_timeline(demo.SCENARIOS["occluded_S5"].items)
    ((t0, t1),) = tl.blackouts
    assert t1 - t0 == pytest.approx(6.0)
    last_start, last_end, last_label = tl.labels[-1]
    assert last_label == "S5" and last_start < t0 < last_end  # starts while the hand withdraws
    assert tl.blocked(t0 + 1.0) and not tl.blocked(t0 - 0.1) and not tl.blocked(t1)


def test_the_hand_never_passes_close_to_an_object_it_is_not_handling():
    """Keeps the scene honest: a hand brushing a resting object would be a real (and different)
    test of the world model, not the procedure."""
    for sc in demo.SCENARIOS.values():
        tl = demo.build_timeline(sc.items)
        for t in np.arange(0.0, tl.duration_s, 0.05):
            handled = HANDLED.get(tl.label_at(t))
            hx, hy = tl.hand_at(t)
            for o, (ox, oy) in tl.objects_at(t).items():
                if o != handled:
                    d = math.hypot(hx - ox, hy - oy)
                    assert d >= 0.10, f"{sc.name} t={t:.2f}: hand {d:.3f} m from the {o}"


def test_scene_frames_are_watermarked_and_the_hand_set_matches_the_frame(proto):
    src = demo.SceneSource("correct", protocol=proto)
    it = iter(src)
    first = next(it)
    assert first.packet.frame_idx == 0 and first.packet.image.shape == (480, 640, 3)
    assert {d.cls for d in first.detections.detections} == set(demo.OBJECTS)
    assert all(d.bbox_rack_xy is None for d in first.detections.detections)  # spatial projects
    assert src.hands_for(0).frame_idx == 0 and len(src.hands_for(0).hands) == 1
    with pytest.raises(LookupError):
        src.hands_for(5)
    assert src.simulated is True


def test_unknown_scenario_and_object_are_rejected():
    with pytest.raises(KeyError, match="nope"):
        demo.get_scenario("nope")
    with pytest.raises(ValueError, match="unknown object"):
        demo.build_timeline([demo.Act("phone")])


# ---- the runs through the real pipeline core ----------------------------------------------------


def test_correct_run_verifies_every_step_with_no_alerts_or_asks(runs):
    res = runs("correct")
    assert res.statuses == dict.fromkeys(STEP_IDS, "verified")
    out = res.summary.outputs
    assert res.summary.alerts == [] and not [o for o in out if isinstance(o, Ask)]
    assert [o.next_step_id for o in out if isinstance(o, Prompt)] == [*STEP_IDS, None]
    verified = [e.step_id for e in res.summary.step_events if e.status is StepStatus.VERIFIED]
    assert verified == STEP_IDS  # in protocol order


def test_run_artifacts_are_labelled_simulated_intact_and_small(runs):
    res = runs("correct")
    assert res.chain_ok and verify_log(res.events_path).ok
    records = [json.loads(line) for line in res.events_path.read_text("utf-8").splitlines()]
    assert records[-1]["type"] == "run_end"
    assert sum(r["type"] == "StepEvent" for r in records) == 5
    assert 0 < res.downlink_bytes <= 4096  # the bandwidth-restricted packet, for a 5-step run
    meta = json.loads((res.run_dir / "run.json").read_text("utf-8"))
    assert meta["simulated"] is True and "SIMULATED" in meta["note"]
    assert meta["protocol"] == "sample_mixing_v1" and meta["scenario"] == "correct"
    assert meta["stages"]["handpose"] == "simulated-hands"


def test_a_tampered_log_is_detected(runs):
    res = runs("correct")
    lines = res.events_path.read_text("utf-8").splitlines()
    i = next(n for n, line in enumerate(lines) if '"StepEvent"' in line)
    edited = res.events_path.with_name("edited.jsonl")
    lines[i] = lines[i].replace("verified", "skipped")
    edited.write_text("\n".join(lines), encoding="utf-8")
    assert not verify_log(edited).ok


def test_skipping_the_cup_raises_out_of_order_then_a_retroactive_skip(runs):
    res = runs("skip_cup")
    kinds = [a.kind for a in res.summary.alerts]
    assert AlertKind.OUT_OF_ORDER in kinds and kinds[-1] is AlertKind.SKIP
    assert res.statuses["S1"] == "skipped"
    assert all(StepStatus(res.statuses[s]) in DONE for s in STEP_IDS[1:]), res.statuses
    skipped = [e for e in res.summary.step_events if e.status is StepStatus.SKIPPED]
    assert [(e.step_id, e.retroactive) for e in skipped] == [("S1", True)]


def test_staging_the_bottle_before_the_cup_is_flagged_and_then_recovers(runs):
    res = runs("bottle_first")
    kinds = [a.kind for a in res.summary.alerts]
    assert kinds[0] is AlertKind.OUT_OF_ORDER and AlertKind.SKIP not in kinds
    assert res.summary.alerts[0].message.startswith(
        "Out of sequence: bottle moved out of Zone S before cup."
    )
    assert all(StepStatus(s) in DONE for s in res.statuses.values()), res.statuses


def test_an_uncallable_placement_is_never_verified(runs):
    res = runs("unsure_S5")
    assert res.statuses["S5"] != "verified"
    assert all(res.statuses[s] == "verified" for s in STEP_IDS[:4]), res.statuses
    asks = [o for o in res.summary.outputs if isinstance(o, Ask)]
    assert all(a.step_id == "S5" for a in asks)  # if it asks, it asks about the book


def test_a_blocked_view_makes_it_ask_and_a_spoken_yes_is_operator_confirmed(runs):
    res = runs("occluded_S5")  # the scenario scripts the operator's "yes"
    asks = [o for o in res.summary.outputs if isinstance(o, Ask)]
    assert asks and asks[0].step_id == "S5" and asks[0].text == BOOK_S5_ASK
    assert res.statuses["S5"] == "operator_confirmed"
    assert all(res.statuses[s] == "verified" for s in STEP_IDS[:4]), res.statuses
    records = [json.loads(line) for line in res.events_path.read_text("utf-8").splitlines()]
    answers = [r for r in records if r["type"] == "Answer"]
    assert [(a["question_id"], a["value"]) for a in answers] == [(asks[0].question_id, "yes")]


def test_a_no_is_never_logged_as_a_confirmation(runs):
    res = runs("occluded_S5", answer="no")  # once the view returns the evidence may still verify it
    assert res.statuses["S5"] != "operator_confirmed"


def test_the_same_seed_gives_a_byte_identical_log(runs, tmp_path):
    first = runs("correct")
    cfg = load_config(demo.CONFIG_PATH, overrides=[f"run.runs_dir={tmp_path.as_posix()}"])
    again = demo.run_scenario("correct", cfg, run_id="again", video=False)
    assert again.events_sha256 == first.events_sha256
    other = demo.run_scenario("correct", cfg, run_id="other-seed", seed=1, video=False)
    assert other.statuses == first.statuses  # a different noise draw, the same outcomes


def test_an_existing_run_dir_is_not_overwritten_without_force(tmp_path):
    cfg = load_config(demo.CONFIG_PATH, overrides=[f"run.runs_dir={tmp_path.as_posix()}"])
    demo.run_scenario("correct", cfg, run_id="r", video=False)
    with pytest.raises(FileExistsError):
        demo.run_scenario("correct", cfg, run_id="r", video=False)
    demo.run_scenario("correct", cfg, run_id="r", video=False, force=True)


def test_output_lines_are_plain_ascii(runs):
    res = runs("skip_cup")
    lines = demo.format_outputs(res.summary.outputs)
    assert lines and all(line.isascii() for line in lines)
    assert any("Skipped: place the cup on the mixing pad." in line for line in lines)


def test_the_scene_refuses_a_protocol_it_was_not_choreographed_for():
    demo.check_scene_protocol(load_protocol("protocols/sample_mixing.yaml"))  # fits
    with pytest.raises(ValueError, match="no object"):
        demo.check_scene_protocol(load_protocol("protocols/bas_boxes.yaml"))
