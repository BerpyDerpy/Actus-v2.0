"""Protocol YAML schema: load, validate, and query step ordering.

A protocol is a set of steps. Each step is a *state transition* over world-state predicates:
``pre`` must hold before it, ``post`` holds after it. When ``pre`` is empty the implicit
precondition is "``post`` does not hold yet".

    proto = load_protocol("protocols/bas_boxes.yaml")
    proto.step("S3").post                  # {"red.in_zone_A": True}
    proto.required_before("S4")            # {"S1", "S2", "S3"} for order: strict
    proto.expected_next({"S1"})            # ["S2"]

Validate from the shell:  ``python -m actus.protocol.schema protocols/bas_boxes.yaml``
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PREDICATE_RE = re.compile(r"^(?P<obj>[A-Za-z][A-Za-z0-9_]*)\.(?P<pred>[A-Za-z][A-Za-z0-9_]*)$")
ZONE_PRED_RE = re.compile(r"^in_zone_(?P<zone>[A-Za-z0-9_]+)$")
STEP_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

Point = tuple[float, float]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_predicates(conds: dict[str, bool], where: str) -> dict[str, bool]:
    for name in conds:
        if not PREDICATE_RE.match(name):
            raise ValueError(f"{where}: predicate {name!r} must look like <object>.<predicate>")
    return conds


class ObjectSpec(_Strict):
    """A tracked object: its spoken label and the detector classes that map to it."""

    label: str
    classes: list[str] = Field(default_factory=list)


class Step(_Strict):
    id: str
    text: str  # imperative, used for prompts: "Open the outer box"
    pre: dict[str, bool] = Field(default_factory=dict)
    post: dict[str, bool]
    requires_hoi: str | None = None  # object that must be handled (HOI evidence) for this step
    ask: str | None = None  # optional custom confirmation question

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not STEP_ID_RE.match(v) or v == "idle":
            raise ValueError(f"invalid step id {v!r}")
        return v

    @model_validator(mode="after")
    def _conds(self) -> Step:
        if not self.post:
            raise ValueError(f"step {self.id}: post must not be empty")
        _check_predicates(self.pre, f"step {self.id} pre")
        _check_predicates(self.post, f"step {self.id} post")
        return self

    @property
    def effective_pre(self) -> dict[str, bool]:
        """``pre`` if given, else the negation of ``post`` (the state before the transition)."""
        return dict(self.pre) if self.pre else {k: not v for k, v in self.post.items()}


class Timing(_Strict):
    dwell_s: float = Field(0.8, ge=0)
    timeout_s: float = Field(60.0, gt=0)
    min_duration_s: dict[str, float] = Field(default_factory=dict)


class Decision(_Strict):
    p_enter: float = Field(0.8, gt=0, le=1)
    p_exit: float = Field(0.6, ge=0, lt=1)
    ask_band: tuple[float, float] = (0.45, 0.8)
    ask_after_s: float = Field(3.0, ge=0)

    @model_validator(mode="after")
    def _bands(self) -> Decision:
        lo, hi = self.ask_band
        if not self.p_exit < self.p_enter:
            raise ValueError("decision: p_exit must be < p_enter (hysteresis)")
        if not 0 <= lo < hi <= self.p_enter:
            raise ValueError("decision: ask_band must satisfy 0 <= low < high <= p_enter")
        return self


class Protocol(_Strict):
    id: str
    title: str
    version: int = 1
    objects: dict[str, ObjectSpec] = Field(default_factory=dict)
    zones: dict[str, list[Point]] = Field(default_factory=dict)  # rack metres
    initial: dict[str, bool] = Field(default_factory=dict)  # predicate values at the start
    steps: list[Step]
    order: Literal["strict"] | list[tuple[str, str]] = "strict"
    timing: Timing = Field(default_factory=Timing)
    decision: Decision = Field(default_factory=Decision)

    # -- validation ---------------------------------------------------------------------------

    @field_validator("zones")
    @classmethod
    def _zones(cls, zones: dict[str, list[Point]]) -> dict[str, list[Point]]:
        for name, pts in zones.items():
            if len(pts) == 2:
                (x1, y1), (x2, y2) = pts
                if not (x1 < x2 and y1 < y2):
                    raise ValueError(f"zone {name}: a 2-point zone is [[xmin,ymin],[xmax,ymax]]")
            elif len(pts) < 3:
                raise ValueError(f"zone {name}: give 2 points (rectangle) or >= 3 (polygon)")
        return zones

    @model_validator(mode="after")
    def _consistency(self) -> Protocol:
        ids = [s.id for s in self.steps]
        if not ids:
            raise ValueError("protocol has no steps")
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate step ids: {sorted(dupes)}")
        known = set(ids)

        if self.order != "strict":
            for a, b in self.order:
                if a not in known or b not in known:
                    raise ValueError(f"order edge {a}->{b} references an unknown step")
                if a == b:
                    raise ValueError(f"order edge {a}->{b} is a self-loop")
            self._topo_order()  # raises on a cycle

        for k in self.timing.min_duration_s:
            if k not in known:
                raise ValueError(f"timing.min_duration_s references unknown step {k!r}")

        preds = set(self.initial)
        for s in self.steps:
            preds |= set(s.pre) | set(s.post)
        _check_predicates(self.initial, "initial")
        for name in preds:
            m = PREDICATE_RE.match(name)
            assert m is not None
            obj, pred = m["obj"], m["pred"]
            if self.objects and obj not in self.objects:
                raise ValueError(f"predicate {name!r}: object {obj!r} is not declared in objects")
            zm = ZONE_PRED_RE.match(pred)
            if zm and zm["zone"] not in self.zones:
                raise ValueError(f"predicate {name!r}: zone {zm['zone']!r} is not declared")
        for s in self.steps:
            if s.requires_hoi is not None:
                objs = {PREDICATE_RE.match(p)["obj"] for p in preds}  # type: ignore[index]
                if s.requires_hoi not in objs and s.requires_hoi not in self.objects:
                    raise ValueError(f"step {s.id}: requires_hoi object {s.requires_hoi!r} unknown")
        return self

    # -- queries ------------------------------------------------------------------------------

    @property
    def step_ids(self) -> list[str]:
        return [s.id for s in self.steps]

    def step(self, step_id: str) -> Step:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(step_id)

    def index(self, step_id: str) -> int:
        return self.step_ids.index(step_id)

    @property
    def predicates(self) -> list[str]:
        """Every predicate the protocol mentions, sorted."""
        names = set(self.initial)
        for s in self.steps:
            names |= set(s.pre) | set(s.post)
        return sorted(names)

    @property
    def object_names(self) -> list[str]:
        names = set(self.objects)
        for p in self.predicates:
            names.add(p.split(".", 1)[0])
        return sorted(names)

    def object_label(self, obj: str) -> str:
        spec = self.objects.get(obj)
        return spec.label if spec else obj

    def edges(self) -> list[tuple[str, str]]:
        """Direct ordering constraints (a must precede b)."""
        if self.order == "strict":
            return list(itertools.pairwise(self.step_ids))
        return [tuple(e) for e in self.order]  # type: ignore[misc]

    def direct_predecessors(self, step_id: str) -> set[str]:
        return {a for a, b in self.edges() if b == step_id}

    def direct_successors(self, step_id: str) -> set[str]:
        return {b for a, b in self.edges() if a == step_id}

    def required_before(self, step_id: str) -> set[str]:
        """All steps that must be done before ``step_id`` (transitive closure)."""
        self.step(step_id)
        out: set[str] = set()
        stack = list(self.direct_predecessors(step_id))
        while stack:
            s = stack.pop()
            if s not in out:
                out.add(s)
                stack.extend(self.direct_predecessors(s))
        return out

    def expected_next(self, done: set[str] | frozenset[str]) -> list[str]:
        """Steps not yet done whose required predecessors are all done, in protocol order."""
        return [s for s in self.step_ids if s not in done and self.required_before(s) <= set(done)]

    def _topo_order(self) -> list[str]:
        ids = self.step_ids
        indeg = {s: 0 for s in ids}
        for _, b in self.edges():
            indeg[b] += 1
        ready = [s for s in ids if indeg[s] == 0]
        out: list[str] = []
        while ready:
            s = ready.pop(0)
            out.append(s)
            for a, b in self.edges():
                if a == s:
                    indeg[b] -= 1
                    if indeg[b] == 0:
                        ready.append(b)
        if len(out) != len(ids):
            raise ValueError("order contains a cycle")
        return out


def load_protocol(path: str | Path) -> Protocol:
    from actus.core.config import resolve_path

    with open(resolve_path(path), encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Protocol.model_validate(data)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate a protocol YAML and print a summary.")
    ap.add_argument("path", nargs="?", default="protocols/bas_boxes.yaml")
    args = ap.parse_args(argv)
    try:
        proto = load_protocol(args.path)
    except Exception as e:  # show validation errors plainly
        print(f"INVALID {args.path}\n{e}", file=sys.stderr)
        return 1
    print(f"OK {proto.id}: {proto.title}")
    print(f"  order: {'strict' if proto.order == 'strict' else proto.edges()}")
    for s in proto.steps:
        pre = s.pre or {k: f"{v} (implicit)" for k, v in s.effective_pre.items()}
        print(f"  {s.id}: {s.text}\n      pre={pre} post={s.post}")
    print(f"  predicates: {', '.join(proto.predicates)}")
    print(f"  zones: {', '.join(proto.zones) or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
