"""Folding the temporal action model into ``StepEvidence``, without letting it invent steps.

The spatial lane already produces prior-free rule evidence per step
(``actus/spatial/predicates.py``): ``rule = min(armed, post, handled)``, where ``armed`` latches
that the step's precondition was *observed* to hold, ``post`` is its post-condition, and
``handled`` (for ``requires_hoi`` steps) that the object was seen in a hand. The temporal model
sees the same scene differently -- it recognises the action itself -- so it can carry a
completion the geometry missed, typically when the operator's own arm hides the placement.

The rule
--------
    offset      = peak * (1 - current)                     # the action happened, and is over
    post_eff    = max(post,    weight * offset * (1 - visibility))
    handled_eff = max(handled, weight * offset)            # 1 if the step has no `handled` term
    fused       = min(armed, post_eff, handled_eff)

Three bounds, each doing a distinct job:

* ``min(..., armed)`` -- pure observation that the world was seen in the step's pre-state.
* ``min(..., post_eff)`` -- and this is the one that matters. The temporal model may only
  substitute for an unobserved post-condition **in proportion to how blind the sensors are**.
  When visibility is 1 the substitution term is 0, so **with a clear view the model cannot raise
  a step's evidence at all**: a low ``post`` under good sight is positive evidence that the
  transition did not happen, and no amount of learned familiarity with the usual order of the
  procedure is allowed to overrule it. When the sensors are blind an unobserved ``post`` is
  uninformative, and that is exactly when the action model earns its place.
* ``min(..., handled_eff)`` -- the model may stand in for "the object was seen in a hand", which
  is the term the detector most often fails to supply. This one needs no visibility gate: it is
  evidence *for* an interaction, and it is still capped by ``post_eff`` above.

An earlier version capped only by ``armed``. That was too weak: a skipped step's precondition
genuinely does hold, so a confident model could drive a step that never happened to 0.87 against
a 0.8 threshold, and it was only the engine's dwell requirement that caught it. The
visibility-gated ``post`` bound is what makes the guarantee structural rather than incidental --
see ``test_a_confident_wrong_model_cannot_erase_a_skip``, which is an end-to-end check against
the real engine.

Fusion never lowers evidence: ``fused = max(rule, min(armed, post_eff, handled_eff))``. So its
*contribution* is bounded by ``armed``, while an incoming ``rule`` above ``armed`` is passed
through rather than suppressed (the spatial lane never produces one, since
``rule = min(armed, post, handled)``, but suppressing evidence is not fusion's job). Evidence carrying no ``armed``/``post`` sources is passed through untouched -- failing
closed for producers this rule was not written against.

``EvidenceFusion`` is stateful because ``peak`` is a running maximum; it resets when the step's
action score collapses after a completion, so a redo is judged afresh.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from actus.core.contracts import IDLE, ActionScores, StepEvidence

DEFAULT_WEIGHT = 0.5


def noisy_or(a: float, b: float) -> float:
    return 1.0 - (1.0 - a) * (1.0 - b)


@dataclass
class _Peak:
    peak: float = 0.0
    completed: bool = False


class EvidenceFusion:
    """Combines per-step rule evidence with the temporal model's action scores."""

    def __init__(
        self,
        step_ids: Iterable[str],
        *,
        weight: float = DEFAULT_WEIGHT,
        reset_below: float = 0.2,
        min_peak: float = 0.3,
        complete_at: float = 0.8,
    ) -> None:
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"weight must be in [0, 1], got {weight}")
        self.step_ids = tuple(step_ids)
        self.weight = float(weight)
        self.reset_below = float(reset_below)
        self.min_peak = float(min_peak)
        self.complete_at = float(complete_at)
        self.reset()

    def reset(self) -> None:
        self._peaks = {s: _Peak() for s in self.step_ids}

    def observe(self, actions: ActionScores | None) -> None:
        """Update the running peak per step. Call once per frame, before ``fuse``."""
        if actions is None:
            return
        for step_id, pk in self._peaks.items():
            score = float(actions.scores.get(step_id, 0.0))
            if score <= self.reset_below and pk.peak > 0.0 and pk.completed:
                pk.peak = 0.0
                pk.completed = False
            pk.peak = max(pk.peak, score)

    def offset(self, step_id: str, actions: ActionScores | None) -> float:
        """Evidence that ``step_id``'s action has been performed and is now over."""
        pk = self._peaks.get(step_id)
        if pk is None or actions is None or pk.peak < self.min_peak:
            return 0.0
        current = float(actions.scores.get(step_id, 0.0))
        return max(0.0, pk.peak * (1.0 - current))

    def fuse(self, evidence: StepEvidence, actions: ActionScores | None) -> StepEvidence:
        """Return ``evidence`` with the temporal source folded in (or unchanged if it can't be)."""
        sources: Mapping[str, float] = evidence.sources
        armed = sources.get("armed")
        post = sources.get("post")
        off = self.offset(evidence.step_id, actions)
        if armed is None or post is None or off <= 0.0 or self.weight <= 0.0:
            return evidence

        rule = float(sources.get("rule", evidence.p_complete))
        visibility = float(sources.get("visibility", 1.0))
        contribution = self.weight * off
        # Blind sensors make an unobserved post-condition uninformative; clear sight makes a low
        # one a real observation that the transition did not happen.
        post_eff = max(float(post), contribution * (1.0 - visibility))
        handled = sources.get("handled")
        handled_eff = 1.0 if handled is None else max(float(handled), contribution)
        fused = min(float(armed), post_eff, handled_eff)
        fused = min(1.0, max(0.0, max(fused, rule)))

        pk = self._peaks.get(evidence.step_id)
        if pk is not None and fused >= self.complete_at:
            pk.completed = True
        merged = dict(sources)
        merged["temporal"] = min(1.0, off)
        merged["fused"] = fused
        return StepEvidence(evidence.frame_idx, evidence.t, evidence.step_id, fused, merged)

    def fuse_all(
        self, evidence: Iterable[StepEvidence], actions: ActionScores | None
    ) -> tuple[StepEvidence, ...]:
        self.observe(actions)
        return tuple(self.fuse(e, actions) for e in evidence)


def idle_score(actions: ActionScores | None) -> float:
    return 0.0 if actions is None else float(actions.scores.get(IDLE, 0.0))
