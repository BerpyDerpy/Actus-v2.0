"""Decision layer: turns noisy per-step probability streams into discrete, timed decisions.

These helpers are pure and clock-driven (they are fed ``t`` from the frame clock, never wall
time) and know nothing about protocol order. The protocol engine (``engine.py``) owns one of each
per step and decides what the firings mean.

* ``DwellGate``      hysteresis + dwell: fires once ``p`` has crossed ``enter`` and then stayed
                     at or above ``exit`` for ``dwell_s``. Used for completion (on ``p``) and for
                     "post-state undone" (on ``1 - p``).
* ``SustainTimer``   a condition that has held continuously for ``after_s`` (ask band, blindness).
* ``order_simultaneous``  the one place the protocol prior may act on completions: it orders
                     steps that fire on the same frame. Earlier crossing first; evidence within
                     ``eps`` is a tie, broken by the prior. It never adds or removes a completion.
* ``DecisionParams`` every threshold. The protocol YAML supplies the core ones (``timing`` and
                     ``decision``); the rest are engine defaults that a caller may override.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Any

from actus.protocol.schema import Protocol

T_EPS = 1e-9  # tolerance for comparisons of float clock differences (e.g. 12/15 - 0/15 vs 0.8)


@dataclass(frozen=True)
class DecisionParams:
    # -- from the protocol YAML (timing / decision) ------------------------------------------
    p_enter: float = 0.8  # completion: evidence must cross this ...
    p_exit: float = 0.6  # ... and not fall below this during the dwell (hysteresis)
    dwell_s: float = 0.8
    ask_low: float = 0.45  # ask band [ask_low, ask_high): unsure evidence
    ask_high: float = 0.8
    ask_after_s: float = 3.0  # time in the ask band before asking
    timeout_s: float = 60.0  # no progress for this long -> timeout alert
    min_duration_s: Mapping[str, float] = field(default_factory=dict)  # too_fast thresholds
    # -- engine defaults (not in the YAML schema) --------------------------------------------
    vis_low: float = 0.5  # Predicates.visibility below this = blind
    vis_ask_after_s: float = 1.5  # blind this long during an expected step -> ask
    tie_eps: float = 0.05  # evidence within this is a tie (broken by the prior)
    action_min: float = 0.5  # ActionScores top score needed to count as "step in progress"
    ask_cooldown_s: float = 10.0  # after "no" / cancel, don't re-ask that step for this long
    ask_supersede_s: float = 15.0  # an unanswered ask older than this may be replaced

    def __post_init__(self) -> None:
        if not 0 <= self.p_exit < self.p_enter <= 1:
            raise ValueError("need 0 <= p_exit < p_enter <= 1")
        if not 0 <= self.ask_low < self.ask_high <= self.p_enter:
            raise ValueError("need 0 <= ask_low < ask_high <= p_enter")
        for name in ("dwell_s", "ask_after_s", "vis_ask_after_s", "ask_cooldown_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        if not 0 <= self.vis_low <= 1 or not 0 <= self.action_min <= 1 or self.tie_eps < 0:
            raise ValueError("vis_low and action_min must be in [0, 1]; tie_eps >= 0")
        object.__setattr__(self, "min_duration_s", dict(self.min_duration_s))

    # Undo detection mirrors completion on 1 - p: p must fall to <= 1 - p_enter and stay
    # <= 1 - p_exit for dwell_s.
    @property
    def p_undo_enter(self) -> float:
        return 1.0 - self.p_enter

    @property
    def p_undo_exit(self) -> float:
        return 1.0 - self.p_exit

    @classmethod
    def from_protocol(cls, protocol: Protocol, **overrides: Any) -> DecisionParams:
        """Core thresholds from the protocol YAML; ``overrides`` for anything (e.g. from config)."""
        d, tm = protocol.decision, protocol.timing
        base = cls(
            p_enter=d.p_enter,
            p_exit=d.p_exit,
            dwell_s=tm.dwell_s,
            ask_low=d.ask_band[0],
            ask_high=d.ask_band[1],
            ask_after_s=d.ask_after_s,
            timeout_s=tm.timeout_s,
            min_duration_s=dict(tm.min_duration_s),
        )
        known = {f.name for f in fields(cls)}
        unknown = set(overrides) - known
        if unknown:
            raise KeyError(f"unknown decision parameter(s): {sorted(unknown)}")
        return replace(base, **overrides) if overrides else base


class DwellGate:
    """Hysteresis + dwell on a probability stream.

    ``update`` returns True on every frame on which the gate is satisfied: ``p`` crossed
    ``enter`` at ``t_cross`` and has not dropped below ``exit`` since, for at least ``dwell_s``.
    The owner calls ``reset()`` once it has acted on the firing.
    """

    __slots__ = ("_ps", "dwell_s", "enter", "exit", "frame_cross", "t_cross")

    def __init__(self, enter: float, exit: float, dwell_s: float) -> None:
        if not exit < enter:
            raise ValueError("exit must be < enter")
        self.enter, self.exit, self.dwell_s = enter, exit, dwell_s
        self.reset()

    def reset(self) -> None:
        self.t_cross: float | None = None
        self.frame_cross: int | None = None
        self._ps: list[float] = []

    @property
    def active(self) -> bool:
        """Crossed ``enter`` and still holding (a candidate, possibly not yet dwelled)."""
        return self.t_cross is not None

    @property
    def mean(self) -> float:
        return sum(self._ps) / len(self._ps) if self._ps else 0.0

    @property
    def peak(self) -> float:
        return max(self._ps, default=0.0)

    def update(self, t: float, p: float, frame_idx: int | None = None) -> bool:
        if self.t_cross is None:
            if p < self.enter:
                return False
            self.t_cross, self.frame_cross, self._ps = t, frame_idx, [p]
        elif p < self.exit:
            self.reset()
            return False
        else:
            self._ps.append(p)
        return t - self.t_cross >= self.dwell_s - T_EPS


class SustainTimer:
    """Tracks whether a condition has held continuously for ``after_s``.

    ``update`` returns True while it has. Gaps (condition False) shorter than ``grace_s`` don't
    break the episode, but ``update`` returns False during them. ``fired`` is for the owner: set
    it once it has acted; it clears when the episode breaks (so it fires once per episode).
    """

    __slots__ = ("after_s", "fired", "grace_s", "t0", "t_false")

    def __init__(self, after_s: float, grace_s: float = 0.0) -> None:
        self.after_s, self.grace_s = after_s, grace_s
        self.reset()

    def reset(self) -> None:
        self.t0: float | None = None
        self.t_false: float | None = None  # start of the current gap
        self.fired = False

    def update(self, t: float, cond: bool) -> bool:
        if not cond:
            if self.t0 is not None:
                if self.t_false is None:
                    self.t_false = t
                if t - self.t_false >= self.grace_s - T_EPS:
                    self.reset()
            return False
        self.t_false = None
        if self.t0 is None:
            self.t0 = t
        return t - self.t0 >= self.after_s - T_EPS


@dataclass(frozen=True)
class Candidate:
    """A step whose gate fired this frame. ``rank`` is the prior: lower = more expected."""

    key: str
    t_cross: float
    p: float
    rank: int


def order_simultaneous(cands: Sequence[Candidate], eps: float) -> list[Candidate]:
    """Order same-frame completions: earlier crossing first (evidence); then higher evidence;
    evidence within ``eps`` (or equal crossing times) is a tie broken by the prior ``rank``."""

    def cmp(a: Candidate, b: Candidate) -> int:
        if not math.isclose(a.t_cross, b.t_cross, abs_tol=T_EPS):
            return -1 if a.t_cross < b.t_cross else 1
        if abs(a.p - b.p) >= eps:
            return -1 if a.p > b.p else 1
        return (a.rank > b.rank) - (a.rank < b.rank)

    return sorted(cands, key=functools.cmp_to_key(cmp))
