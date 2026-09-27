import copy

import pytest
import yaml
from pydantic import ValidationError

from actus.core.config import REPO_ROOT
from actus.protocol.schema import Protocol, load_protocol, main

BAS = REPO_ROOT / "protocols" / "bas_boxes.yaml"


@pytest.fixture
def raw():
    return yaml.safe_load(BAS.read_text(encoding="utf-8"))


def test_bas_boxes_loads_and_matches_plan():
    p = load_protocol(BAS)
    assert p.id == "bas_boxes_v1"
    assert p.step_ids == ["S1", "S2", "S3", "S4", "S5", "S6"]
    assert p.step("S3").post == {"red.in_zone_A": True}
    assert p.step("S2").requires_hoi == "red"
    assert p.timing.dwell_s == 0.8 and p.timing.min_duration_s == {"S3": 1.0}
    assert p.decision.ask_band == (0.45, 0.8)
    assert p.zones["A"] == [(0.55, 0.10), (0.80, 0.35)]
    assert p.object_label("yellow") == "yellow box"


def test_effective_pre_is_negated_post_when_missing():
    p = load_protocol(BAS)
    assert p.step("S6").effective_pre == {"outer.open": True}
    assert p.step("S2").effective_pre == {"red.in_outer": True}


def test_strict_order_queries():
    p = load_protocol(BAS)
    assert p.required_before("S4") == {"S1", "S2", "S3"}
    assert p.direct_successors("S2") == {"S3"}
    assert p.expected_next(set()) == ["S1"]
    assert p.expected_next({"S1", "S2"}) == ["S3"]
    assert p.expected_next(set(p.step_ids)) == []


def test_partial_order(raw):
    raw["order"] = [
        ["S1", "S2"],
        ["S1", "S4"],
        ["S2", "S3"],
        ["S4", "S5"],
        ["S3", "S6"],
        ["S5", "S6"],
    ]
    p = Protocol.model_validate(raw)
    assert p.expected_next({"S1"}) == ["S2", "S4"]
    assert p.required_before("S6") == {"S1", "S2", "S3", "S4", "S5"}


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda r: r["steps"].append(copy.deepcopy(r["steps"][0])), "duplicate"),
        (lambda r: r.__setitem__("order", [["S1", "S9"]]), "unknown step"),
        (lambda r: r.__setitem__("order", [["S1", "S2"], ["S2", "S1"]]), "cycle"),
        (lambda r: r["steps"][2]["post"].update({"red.in_zone_C": True}), "zone 'C'"),
        (lambda r: r["steps"][0]["post"].update({"blue.open": True}), "not declared"),
        (lambda r: r["steps"][0].__setitem__("post", {"open": True}), "<object>.<predicate>"),
        (lambda r: r["steps"][0].__setitem__("post", {}), "post must not be empty"),
        (lambda r: r["decision"].__setitem__("p_exit", 0.9), "hysteresis"),
        (lambda r: r["decision"].__setitem__("ask_band", [0.5, 0.9]), "ask_band"),
        (lambda r: r["timing"]["min_duration_s"].update({"S9": 1}), "unknown step"),
        (lambda r: r.__setitem__("typo_field", 1), "typo_field"),
        (lambda r: r["zones"].__setitem__("A", [[0.8, 0.1], [0.5, 0.3]]), "xmin"),
    ],
)
def test_invalid_protocols_rejected(raw, mutate, match):
    mutate(raw)
    with pytest.raises(ValidationError, match=match):
        Protocol.model_validate(raw)


def test_cli_validates(capsys):
    assert main([str(BAS)]) == 0
    assert "OK bas_boxes_v1" in capsys.readouterr().out
