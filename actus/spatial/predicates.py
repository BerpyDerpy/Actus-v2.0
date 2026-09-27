"""World state → ``Predicates`` and prior-free ``StepEvidence``.

Predicates
----------
Every predicate the world model can estimate (a superset of the protocol's), from the filtered
beliefs, then per-predicate temperature calibration (``spatial.temperatures``, identity until
F-CAL fits them with ``fit_temperature``), then — as the replay source does — regression toward
0.5 by scene visibility, ``v·p + (1−v)·0.5``, then clipping to ``p_clip``. A blinded camera
therefore reports "don't know" rather than a stale certainty; the internal belief keeps the
last state so it can resume when the view returns.

Step evidence (the rule source)
-------------------------------
``p_complete`` says the step's *transition* happened: its effective ``pre`` held (with ``post``
not yet holding), and now ``post`` holds. Per step, two latches over the unregressed beliefs:

* ``armed = max over time of a(t)``, where ``a(t)`` is the minimum over the last
  ``arm_dwell_s`` of ``min(P(pre), 1 − P(post)) × support``: the pre-state must *hold* for a
  moment, not flash for a frame. ``support`` decays with the time since the objects in ``pre``
  were last actually observed (``arm_memory_s``). A pre-state that only the protocol's
  ``initial`` block asserts — or that survives only in the first frames before observations
  overturn it — cannot arm a step, so a wrong initial prior cannot fabricate a completion.
* ``handled = max over time of min(armed, P(<obj>.held))`` for steps with ``requires_hoi``.

``rule = min(armed, P(post), handled)``. After a completion (``rule ≥ complete_hi``) is undone
(``P(post) ≤ complete_lo``), both latches reset, so a redo must re-arm and be handled again.

Nothing here reads the protocol order or which step is expected next: evidence is prior-free.
``sources`` breaks ``p_complete`` down: ``rule`` (= p_complete), ``armed``, ``post``,
``handled`` (steps with ``requires_hoi``) and ``visibility``.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

from actus.core.contracts import Predicates, StepEvidence
from actus.spatial.world_state import HELD, SpatialParams, WorldSnapshot


def calibrate(p: float, temperature: float) -> float:
    """Temperature scaling on the logit: T > 1 softens, T < 1 sharpens."""
    if temperature == 1.0:
        return p
    p = min(max(p, 1e-6), 1 - 1e-6)
    return 1.0 / (1.0 + math.exp(-math.log(p / (1 - p)) / temperature))


def fit_temperature(
    probs: Iterable[float], labels: Iterable[bool], grid: Iterable[float] | None = None
) -> float:
    """Temperature minimising the negative log-likelihood of ``labels`` (F-CAL, T-EVAL)."""
    p = np.clip(np.asarray(list(probs), dtype=np.float64), 1e-6, 1 - 1e-6)
    y = np.asarray(list(labels), dtype=bool)
    if len(p) == 0 or len(p) != len(y):
        raise ValueError("need equally many probabilities and labels")
    logit = np.log(p / (1 - p))
    best_t, best_nll = 1.0, math.inf
    for T in grid if grid is not None else np.geomspace(0.2, 10.0, 121):
        q = np.clip(1 / (1 + np.exp(-logit / T)), 1e-9, 1 - 1e-9)
        nll = float(-np.mean(np.where(y, np.log(q), np.log(1 - q))))
        if nll < best_nll:
            best_t, best_nll = float(T), nll
    return best_t


@dataclass
class _Latch:
    armed: float = 0.0
    handled: float = 0.0
    completed: bool = False
    window: deque[tuple[float, float]] = field(default_factory=deque)  # (t, arm candidate)

    def sustained(self, t: float, a: float, dwell_s: float) -> float:
        """Minimum of ``a`` over the last ``dwell_s`` seconds; 0 until that much history."""
        self.window.append((t, a))
        while len(self.window) > 1 and self.window[1][0] <= t - dwell_s:
            self.window.popleft()
        if t - self.window[0][0] < dwell_s:
            return 0.0
        return min(v for _, v in self.window)


class EvidenceModel:
    """Turns successive ``WorldSnapshot``s into ``Predicates`` and per-step ``StepEvidence``."""

    def __init__(self, protocol, params: SpatialParams | None = None, *, movables=()) -> None:
        self.protocol = protocol
        self.params = params or SpatialParams()
        self._pre_objects = {
            s.id: sorted({k.split(".", 1)[0] for k in s.effective_pre}) for s in protocol.steps
        }
        for s in protocol.steps:
            if s.requires_hoi is not None and movables and s.requires_hoi not in movables:
                raise ValueError(
                    f"step {s.id}: requires_hoi {s.requires_hoi!r} is not a movable object"
                )
        self.reset()

    def reset(self) -> None:
        self._latch = {s.id: _Latch() for s in self.protocol.steps}

    def _report(self, name: str, p: float, v: float) -> float:
        pr = self.params
        p = calibrate(p, float(pr.temperatures.get(name, 1.0)))
        if pr.report_regress:
            p = v * p + (1 - v) * 0.5
        lo, hi = pr.p_clip
        return min(max(p, lo), hi)

    def predicates(self, snap: WorldSnapshot) -> Predicates:
        v = snap.visibility
        values = {k: self._report(k, p, v) for k, p in snap.predicates.items()}
        return Predicates(snap.frame_idx, snap.t, values, v)

    def evidence(self, snap: WorldSnapshot) -> tuple[StepEvidence, ...]:
        pr = self.params
        vals = snap.predicates
        v = snap.visibility
        out = []
        for step in self.protocol.steps:
            la = self._latch[step.id]
            pre = _cond(vals, step.effective_pre)
            post = _cond(vals, step.post)
            support = min(
                (
                    _support(snap.objects[o].supported_t, snap.t, pr.arm_memory_s)
                    if o in snap.objects
                    else 0.0
                )
                for o in self._pre_objects[step.id]
            )
            if la.completed and post <= pr.complete_lo:  # the step was undone
                la.armed, la.handled, la.completed = 0.0, 0.0, False
            a = la.sustained(snap.t, min(pre, 1 - post) * support, pr.arm_dwell_s)
            la.armed = max(la.armed, a)
            rule = min(la.armed, post)
            sources = {"armed": la.armed, "post": post}
            if step.requires_hoi is not None:
                held = vals.get(f"{step.requires_hoi}.{HELD}", 0.0)
                la.handled = max(la.handled, min(la.armed, held))
                rule = min(rule, la.handled)
                sources["handled"] = la.handled
            if rule >= pr.complete_hi:
                la.completed = True
            p = self._report(step.id, rule, v)
            sources.update(rule=p, visibility=v)
            out.append(StepEvidence(snap.frame_idx, snap.t, step.id, p, sources))
        return tuple(out)


def _cond(vals: Mapping[str, float], conds: Mapping[str, bool]) -> float:
    """Fuzzy AND (min) of predicate conditions."""
    return min((vals[k] if want else 1 - vals[k]) for k, want in conds.items())


def _support(supported_t: float | None, t: float, memory_s: float) -> float:
    if supported_t is None:
        return 0.0
    return math.exp(-max(0.0, t - supported_t) / memory_s) if memory_s > 0 else 1.0
