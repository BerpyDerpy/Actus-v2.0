"""Configuration loading: ``configs/default.yaml`` + optional override file + ``key=value`` pairs.

    cfg = load_config()                                    # defaults
    cfg = load_config("configs/site.yaml", overrides=["stream.target_ip=192.168.1.20"])
    cfg.stream.target_ip          # attribute access
    cfg["stream"]["port"]         # mapping access
    cfg.get("voice.tts.engine")   # dotted lookup

Each lane owns the interpretation of its own section. Overriding a key that does not exist in
the defaults is an error (catches typos); add the key to ``default.yaml`` first.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"


def resolve_path(p: str | Path) -> Path:
    """Absolute path; relative paths are taken relative to the repo root."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(resolve_path(path), encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``over`` into a copy of ``base``; non-mapping values replace."""
    out = copy.deepcopy(dict(base))
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Config(Mapping[str, Any]):
    """Read-only nested mapping with attribute access."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_data", dict(data))

    def __getitem__(self, key: str) -> Any:
        v = self._data[key]
        return Config(v) if isinstance(v, Mapping) else v

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError("Config is read-only")

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Config({self._data!r})"

    def get(self, dotted: str, default: Any = None) -> Any:  # type: ignore[override]
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)


def _apply_override(data: dict[str, Any], dotted: str, value: Any, strict: bool) -> None:
    if strict and not _is_known(data, dotted):
        raise KeyError(f"config override: unknown key {dotted!r}")
    parts = dotted.split(".")
    node = data
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        if part not in node:
            node[part] = value if last else {}
        if last:
            node[part] = value
        else:
            if not isinstance(node[part], dict):
                raise KeyError(f"config override {dotted!r}: {part!r} is not a section")
            node = node[part]


def parse_overrides(items: Iterable[str] | Mapping[str, Any]) -> dict[str, Any]:
    """``["a.b=1", "c=hello"]`` → ``{"a.b": 1, "c": "hello"}`` (values parsed as YAML)."""
    if isinstance(items, Mapping):
        return dict(items)
    out: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        out[key.strip()] = yaml.safe_load(raw) if raw.strip() else ""
    return out


def load_config(
    path: str | Path | None = None,
    *,
    overrides: Iterable[str] | Mapping[str, Any] = (),
    strict: bool = True,
    defaults: str | Path = DEFAULT_CONFIG,
) -> Config:
    data = load_yaml(defaults)
    if path is not None:
        extra = load_yaml(path)
        if strict:
            for dotted in _flatten(extra):
                if not _is_known(data, dotted):
                    raise KeyError(f"{path}: unknown config key {dotted!r}")
        data = deep_merge(data, extra)
    for dotted, value in parse_overrides(overrides).items():
        _apply_override(data, dotted, value, strict)
    return Config(data)


def _flatten(d: Mapping[str, Any], prefix: str = "") -> Iterator[str]:
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping) and v:
            yield from _flatten(v, key + ".")
        else:
            yield key


def _is_known(d: Mapping[str, Any], dotted: str) -> bool:
    """True if ``dotted`` exists in ``d``, or lies under a section whose default is ``{}``
    (an empty mapping in default.yaml marks a free-form section)."""
    node: Any = d
    for part in dotted.split("."):
        if isinstance(node, Mapping) and not node and node is not d:
            return True
        if not isinstance(node, Mapping) or part not in node:
            return False
        node = node[part]
    return True
