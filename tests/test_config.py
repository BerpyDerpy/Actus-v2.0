import pytest

from actus.core.config import REPO_ROOT, deep_merge, load_config, load_yaml, resolve_path


def test_defaults_load_with_attribute_access():
    cfg = load_config()
    assert cfg.stream.port == 8554
    assert cfg["record"]["segment_s"] == 60
    assert cfg.get("voice.tts.engine") == "kokoro"
    assert cfg.get("voice.nope", "dflt") == "dflt"
    assert resolve_path(cfg.protocol.path).exists()


def test_overrides_parse_yaml_values():
    cfg = load_config(
        overrides=["stream.target_ip=192.168.1.20", "record.enabled=false", "replay.fps=30"]
    )
    assert cfg.stream.target_ip == "192.168.1.20"
    assert cfg.record.enabled is False
    assert cfg.replay.fps == 30


def test_unknown_override_is_an_error():
    with pytest.raises(KeyError):
        load_config(overrides=["stream.target_iip=1.2.3.4"])
    cfg = load_config(overrides=["stream.target_iip=1.2.3.4"], strict=False)
    assert cfg.stream.target_iip == "1.2.3.4"


def test_free_form_sections_accept_new_keys():
    cfg = load_config(overrides=["spatial.held_timeout_s=1.5", "faults.occlude=true"])
    assert cfg.spatial.held_timeout_s == 1.5


def test_override_file(tmp_path):
    site = tmp_path / "site.yaml"
    site.write_text("stream:\n  target_ip: 10.0.0.5\n", encoding="utf-8")
    cfg = load_config(site)
    assert cfg.stream.target_ip == "10.0.0.5" and cfg.stream.port == 8554
    bad = tmp_path / "bad.yaml"
    bad.write_text("stream:\n  tagret_ip: 10.0.0.5\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_config(bad)


def test_config_is_read_only():
    cfg = load_config()
    with pytest.raises(AttributeError):
        cfg.stream = {}  # type: ignore[misc]
    d = cfg.to_dict()
    d["stream"]["port"] = 1
    assert cfg.stream.port == 8554


def test_deep_merge():
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}


def test_rack_yaml_matches_plan():
    rack = load_yaml(REPO_ROOT / "configs" / "rack.yaml")
    assert rack["aruco"]["dictionary"] == "DICT_4X4_50"
    assert sorted(rack["markers"]) == [0, 1, 2, 3]
    assert rack["aruco"]["marker_size"] == pytest.approx(0.06)
