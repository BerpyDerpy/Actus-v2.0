"""Protocol engine: the judge. Consumes evidence, emits step outcomes, alerts, prompts and asks.

Inputs (per frame): ``StepEvidence`` for each step (prior-free), optionally ``Predicates``
(only ``visibility`` is used) and ``ActionScores``. Operator inputs: ``Answer``, ``Command``.
Outputs: ``StepEvent``, ``Alert``, ``Prompt``, ``Ask`` (all stamped with the frame clock).

    engine = ProtocolEngine(load_protocol("protocols/bas_boxes.yaml"))
    out = engine.update(frame_idx, t, evidence, visibility=0.95, actions=scores)
    out += engine.answer(Answer("q1-S3", "yes", "voice", t=t))
    out += engine.finish()                     # end of stream: pending skips are finalised

Rules (the evidence/expectation split)
--------------------------------------
* **Completion is evidence only.** A step completes when its ``p_complete`` crosses ``p_enter``
  and holds (>= ``p_exit``) for ``dwell_s`` (``decision.DwellGate``), or when the operator
  answers "yes" to an Ask about it. The protocol order never raises or lowers ``p_complete``,
  and never adds, removes or delays a completion.
* **The prior only ranks.** The expected-next step decides what ``Prompt`` says, which step a
  blindness Ask is about, which of several unsure steps is asked about first, and the order of
  completions that fire on the same frame when their evidence ties within ``tie_eps``.
* **Out of order** (alert once per missing step): a step completes while a step required
  before it is still pending *and that step's own evidence is clearly low* (below the ask band
  for ``dwell_s``, sensors not blind, no open Ask about it). Such a step is *flagged missing*.
  The completed step gets status ``out_of_order``; the alert names it first, then the missing
  steps.
  An unsure predecessor (evidence in the band, or an open Ask) is not an accusation.
* **Skip** (retroactive): when the procedure can go no further - every step is complete,
  skipped or flagged missing - the flagged steps become ``skipped`` (``retroactive=True``) with
  one ``skip`` alert. ``finish()`` does the same at end of stream. A skip therefore surfaces
  live as ``out_of_order`` first (at that moment it can't be told apart from a late step) and is
  confirmed as ``skip`` when the procedure ends. A skipped step that is done later becomes
  ``out_of_order`` (no new alert).
* **Unexpected**: a completed step's post-state is undone (evidence falls to
  <= ``1 - p_enter`` and stays <= ``1 - p_exit`` for ``dwell_s``, sensors not blind) and no
  protocol step that legitimately undoes it (a later step with a contradicting post, e.g. S6
  "close" undoes S1 "open") completed after it or is completing now. The step reopens
  (``StepEvent`` ``pending``). Undos of steps sharing an object form one incident, one alert.
* **Timeout**: no progress (completion or reopen) for ``timeout_s``: the prompted step gets
  ``timed_out`` and one alert, until the next progress. **Too fast**: a step's duration
  (``t_end - t_start``) is below ``min_duration_s``; status unchanged, one alert.
* **Ask**: evidence sits in ``ask_band`` for ``ask_after_s`` with the sensors seeing, or the
  sensors are blind for ``vis_ask_after_s`` while a step is expected. One open Ask at a time
  (an unanswered one may be superseded after ``ask_supersede_s``). "yes" -> ``operator_confirmed``;
  "no" -> re-prompt that step and don't re-ask it for ``ask_cooldown_s``. An Ask is closed by
  any later ``StepEvent`` for its ``step_id``; answers to closed asks are ignored.
* **Prompt**: at start, after every completion, after a "no", and whenever the prompted step
  changes (e.g. a reopen). ``next_step_id=None`` once every step is resolved.

Timing fields of a completion: ``t_end`` = the frame the evidence crossed ``p_enter`` (the
transition was observed), ``t`` = when the dwell finished (decision). ``t_start`` = onset of the
step in ``ActionScores`` (top label, score >= ``action_min``) after the previous completion's
``t_end``; without action evidence it is that previous ``t_end`` (conservative for too_fast).

The engine is single-threaded; ``EngineRunner`` drives it from the bus in one thread.
Operator inputs are applied immediately and stamped with the last processed frame.

    python -m actus.protocol.engine --scenario skip_S2
    python -m actus.protocol.engine --scenario unsure_S3 --answer yes
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import itertools
import math
import sys
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any
from typing import Protocol as TypingProtocol

from actus.core.bus import Bus, BusClosed
from actus.core.contracts import (
    IDLE,
    ActionScores,
    Alert,
    AlertKind,
    Answer,
    AnswerValue,
    Ask,
    Command,
    CommandName,
    InputVia,
    Predicates,
    Prompt,
    StepEvent,
    StepEvidence,
    StepStatus,
    Topic,
)
from actus.protocol.decision import (
    Candidate,
    DecisionParams,
    DwellGate,
    SustainTimer,
    order_simultaneous,
)
from actus.protocol.schema import PREDICATE_RE, ZONE_PRED_RE, Protocol, Step

Output = StepEvent | Alert | Prompt | Ask

DONE = frozenset({StepStatus.VERIFIED, StepStatus.OPERATOR_CONFIRMED, StepStatus.OUT_OF_ORDER})
OPEN = frozenset({StepStatus.PENDING, StepStatus.TIMED_OUT})  # not done, not skipped
RESOLVED = DONE | {StepStatus.SKIPPED}

OUTPUT_TOPICS: dict[type, Topic] = {
    StepEvent: Topic.EVENTS,
    Alert: Topic.ALERTS,
    Prompt: Topic.PROMPTS,
    Ask: Topic.ASKS,
}


# --------------------------------------------------------------------------------------------
# Phrasing (spoken and shown as-is; keep it short)
# --------------------------------------------------------------------------------------------


def _lower_first(s: str) -> str:
    return s[:1].lower() + s[1:]


def _phrase(protocol: Protocol, predicate: str, value: bool) -> str:
    """Past-tense description of ``predicate`` becoming ``value``: "yellow box removed"."""
    m = PREDICATE_RE.match(predicate)
    obj, pred = (m["obj"], m["pred"]) if m else ("", predicate)
    label = protocol.object_label(obj)
    zm = ZONE_PRED_RE.match(pred)
    if zm:
        z = zm["zone"]
        return f"{label} placed in Zone {z}" if value else f"{label} moved out of Zone {z}"
    if pred == "open":
        return f"{label} opened" if value else f"{label} closed"
    if pred == "closed":
        return f"{label} closed" if value else f"{label} opened"
    if pred == "held":
        return f"{label} picked up" if value else f"{label} put down"
    if pred.startswith("in_") and len(pred) > 3:
        container = protocol.object_label(pred[3:])
        return f"{label} put back in the {container}" if value else f"{label} removed"
    words = pred.replace("_", " ")
    return f"{label} {words}" if value else f"{label} no longer {words}"


def _step_object(step: Step) -> str:
    if step.requires_hoi:
        return step.requires_hoi
    return next(iter(step.post)).split(".", 1)[0]


def _step_phrase(protocol: Protocol, step: Step) -> str:
    pred, value = next(iter(step.post.items()))
    return _phrase(protocol, pred, value)


# --------------------------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------------------------


@dataclass
class _StepState:
    spec: Step
    index: int
    gate: DwellGate
    undo_gate: DwellGate
    band: SustainTimer
    low: SustainTimer  # evidence clearly low (< ask_low, seeing) for dwell_s
    status: StepStatus = StepStatus.PENDING
    p: float = 0.0
    sources: dict[str, float] = field(default_factory=dict)
    flagged: bool = False  # missing: a step that requires it completed while it was pending
    pending_since: float = -math.inf  # when it last became pending (start or reopen)
    done_t: float | None = None  # decision time of the current completion
    t_start: float | None = None
    t_end: float | None = None
    confidence: float = 0.0
    seen_high: bool = False  # evidence >= p_enter seen since completion (arms undo detection)
    ask_blocked_until: float = -math.inf
    clearly_low: bool = False  # low timer held as of the last frame

    @property
    def id(self) -> str:
        return self.spec.id


@dataclass
class _AskState:
    ask: Ask
    closed: bool = False  # answered, cancelled, or its step was resolved


@dataclass
class _Incident:
    objects: set[str]
    steps: set[str]


# --------------------------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------------------------


class ProtocolEngine:
    """State machine over a protocol. See the module docstring for the rules."""

    def __init__(self, protocol: Protocol, params: DecisionParams | None = None) -> None:
        self.protocol = protocol
        self.params = params or DecisionParams.from_protocol(protocol)
        pr = self.params
        self._steps: dict[str, _StepState] = {
            s.id: _StepState(
                spec=s,
                index=i,
                gate=DwellGate(pr.p_enter, pr.p_exit, pr.dwell_s),
                undo_gate=DwellGate(pr.p_enter, pr.p_exit, pr.dwell_s),  # fed 1 - p
                band=SustainTimer(pr.ask_after_s, grace_s=pr.dwell_s),
                low=SustainTimer(pr.dwell_s),
            )
            for i, s in enumerate(protocol.steps)
        }
        ids = protocol.step_ids
        self._required: dict[str, frozenset[str]] = {
            s: frozenset(protocol.required_before(s)) for s in ids
        }
        self._dependents: dict[str, tuple[str, ...]] = {
            s: tuple(x for x in ids if s in self._required[x]) for s in ids
        }
        self._undoers: dict[str, tuple[str, ...]] = {
            s: tuple(
                y
                for y in ids
                if y != s
                and any(
                    k in protocol.step(y).post and protocol.step(y).post[k] != v
                    for k, v in protocol.step(s).post.items()
                )
            )
            for s in ids
        }
        self._frame_idx: int | None = None
        self._t: float | None = None
        self._visibility: float | None = None
        self._vis_timer = SustainTimer(pr.vis_ask_after_s)
        self._top: str | None = None
        self._top_changes: collections.deque[tuple[float, str | None]] = collections.deque(
            maxlen=512
        )
        self._last_progress_t = 0.0
        self._t0 = 0.0  # t of the first frame
        self._timeout_fired = False
        self._asks: dict[str, _AskState] = {}
        self._open_ask: str | None = None
        self._ask_seq = itertools.count(1)
        self._prompted: str | None = None
        self._finished = False
        self._incidents: list[_Incident] = []

    # -- public queries -----------------------------------------------------------------------

    @property
    def statuses(self) -> dict[str, StepStatus]:
        return {s: st.status for s, st in self._steps.items()}

    @property
    def finished(self) -> bool:
        """Every step is complete or skipped."""
        return self._finished

    @property
    def open_ask(self) -> Ask | None:
        return self._asks[self._open_ask].ask if self._open_ask else None

    @property
    def prompt_target(self) -> str | None:
        return self._target()

    def outcomes(self) -> list[dict[str, Any]]:
        """The step outcome table (for the run summary and the downlink packet)."""
        return [
            {
                "step_id": st.id,
                "text": st.spec.text,
                "status": st.status.value,
                "t_start": st.t_start,
                "t_end": st.t_end,
                "confidence": round(st.confidence, 4),
            }
            for st in self._steps.values()
        ]

    # -- inputs -------------------------------------------------------------------------------

    def update(
        self,
        frame_idx: int,
        t: float,
        evidence: Iterable[StepEvidence] | Mapping[str, StepEvidence] = (),
        *,
        visibility: float | Predicates | None = None,
        actions: ActionScores | None = None,
    ) -> list[Output]:
        """Process one frame. Steps without new evidence keep their last value."""
        if self._frame_idx is not None and frame_idx <= self._frame_idx:
            raise ValueError(f"frame_idx must increase: got {frame_idx} after {self._frame_idx}")
        if self._t is not None and t < self._t:
            raise ValueError(f"t must not decrease: got {t} after {self._t}")
        first = self._frame_idx is None
        self._frame_idx, self._t = frame_idx, t
        pr = self.params

        if isinstance(evidence, Mapping):
            evidence = evidence.values()
        for ev in evidence:
            st = self._steps.get(ev.step_id)
            if st is not None:
                st.p, st.sources = ev.p_complete, dict(ev.sources)
        if isinstance(visibility, Predicates):
            visibility = visibility.visibility
        if visibility is not None:
            self._visibility = float(visibility)
        if actions is not None:
            self._ingest_actions(t, actions)
        blind = self._blind()
        for st in self._steps.values():
            st.clearly_low = st.low.update(t, st.status in OPEN and not blind and st.p < pr.ask_low)

        out: list[Output] = []
        if first:
            self._last_progress_t = self._t0 = t
            out.append(self._prompt())

        # completion gates (evidence only), for every step that is not done
        fired: list[Candidate] = []
        for st in self._steps.values():
            if st.status in DONE:
                continue
            if st.gate.update(t, st.p, frame_idx):
                fired.append(Candidate(st.id, st.gate.t_cross, st.gate.mean, self._rank(st)))
        # undo gates, for done steps whose post-state was seen
        undone: list[_StepState] = []
        for st in self._steps.values():
            if st.status not in DONE:
                continue
            if st.p >= pr.p_enter:
                st.seen_high = True
            if not st.seen_high or blind:
                st.undo_gate.reset()
            elif st.undo_gate.update(t, 1.0 - st.p, frame_idx):
                undone.append(st)

        completions = [(self._steps[c.key], None) for c in order_simultaneous(fired, pr.tie_eps)]
        out += self._settle(completions, undone)
        out += self._check_timeout()
        out += self._check_asks(blind)
        return out

    def answer(self, ans: Answer) -> list[Output]:
        """Apply an operator answer. Unknown or closed questions are ignored."""
        rec = self._asks.get(ans.question_id)
        if rec is None or rec.closed or self._t is None:
            return []
        self._close_ask(rec)
        st = self._steps[rec.ask.step_id]
        if st.status in DONE:
            return []
        if ans.value is AnswerValue.YES:
            return self._settle([(st, rec.ask)], [])
        st.ask_blocked_until = self._t + self.params.ask_cooldown_s
        st.band.reset()
        return [self._stamp(Prompt, st.id, st.spec.text)]  # re-prompt; not the prompt target

    def command(self, cmd: Command) -> list[Output]:
        """``repeat`` re-issues the open Ask (else the prompt); ``next``/``whats_next`` re-issue
        the prompt (they never mark a step done); ``confirm``/``cancel`` answer or dismiss the
        open Ask."""
        if self._t is None:
            return []
        name = cmd.name
        rec = self._asks.get(self._open_ask) if self._open_ask else None
        if name is CommandName.CONFIRM:
            if rec is None:
                return []
            return self.answer(Answer(rec.ask.question_id, AnswerValue.YES, cmd.via, t=cmd.t))
        if name is CommandName.CANCEL:
            if rec is None:
                return []
            self._close_ask(rec)
            st = self._steps[rec.ask.step_id]
            st.ask_blocked_until = self._t + self.params.ask_cooldown_s
            st.band.reset()
            return []
        if name is CommandName.REPEAT and rec is not None:
            a = rec.ask
            return [self._stamp(Ask, a.question_id, a.step_id, a.text)]
        return [self._prompt()]

    def finish(self) -> list[Output]:
        """End of stream: flagged-missing steps become skipped. Other open steps stay pending."""
        if self._t is None:
            return []
        flagged = [st for st in self._steps.values() if st.flagged and st.status in OPEN]
        if not flagged:
            return []
        events, alert = self._skip(flagged)
        self._finished = all(st.status in RESOLVED for st in self._steps.values())
        return [*events, alert]

    # -- core ---------------------------------------------------------------------------------

    def _settle(
        self,
        completions: list[tuple[_StepState, Ask | None]],
        undone: list[_StepState],
    ) -> list[Output]:
        """Apply completions (ask=None: evidence; else operator-confirmed) and undos, then run the
        missing / skip / status / prompt logic. Returns outputs in emission order: completion
        events, reopen events, retroactive skip events, alerts, prompt."""
        t = self._t
        assert t is not None
        pr = self.params
        done_events: list[Output] = []
        reopen_events: list[Output] = []
        alerts: list[Output] = []

        # 1. completions: mark done (status decided in step 5, once flags are known)
        done_now: list[tuple[_StepState, Ask | None, bool]] = []
        for st, ask in completions:
            was_skipped = st.status is StepStatus.SKIPPED
            st.t_end = ask.t if ask is not None else st.gate.t_cross
            st.t_start = self._onset(st.id, st.t_end)
            st.confidence = st.p if ask is not None else st.gate.mean
            st.status = StepStatus.VERIFIED  # provisional
            st.done_t = t
            st.flagged = False
            st.seen_high = ask is None
            st.gate.reset()
            st.undo_gate.reset()
            st.band.reset()
            self._last_progress_t = t
            self._timeout_fired = False
            self._resolve_asks(st.id)
            for inc in self._incidents:
                inc.steps.discard(st.id)
            done_now.append((st, ask, was_skipped))
        self._incidents = [inc for inc in self._incidents if inc.steps]

        # 2. undos (after completions, so a legitimate undoer completing now is seen)
        reopened = False
        for st in undone:
            if self._undo_expected(st):
                st.seen_high = False  # legitimately undone; disarm until seen high again
                st.undo_gate.reset()
                continue
            reopen_events.append(
                self._stamp(
                    StepEvent, st.id, StepStatus.PENDING, None, None, 1.0 - st.p, st.sources
                )
            )
            st.status = StepStatus.PENDING
            st.pending_since = t
            st.flagged = False
            st.done_t = st.t_start = st.t_end = None
            st.seen_high = False
            st.gate.reset()
            st.undo_gate.reset()
            self._last_progress_t = t
            self._timeout_fired = False
            reopened = True
            objs = {k.split(".", 1)[0] for k in st.spec.post}
            inc = next((i for i in self._incidents if i.objects & objs), None)
            if inc is not None:  # same incident: no second alert
                inc.objects |= objs
                inc.steps.add(st.id)
            else:
                self._incidents.append(_Incident(objs, {st.id}))
                pred, value = next(iter(st.spec.post.items()))
                msg = f"Unexpected: {_phrase(self.protocol, pred, not value)}."
                alerts.append(self._stamp(Alert, AlertKind.UNEXPECTED, msg, (st.id,)))

        # 3. missing steps: required by a completed step, own evidence clearly low
        newly_flagged, trigger = self._check_missing()

        # 4. the procedure can go no further -> flagged steps are skipped (no out_of_order
        #    alert for steps flagged and skipped in the same frame: the skip alert covers them)
        skip_events: list[Output] = []
        skip_alert: Alert | None = None
        flagged_open = [st for st in self._steps.values() if st.flagged and st.status in OPEN]
        if flagged_open and all(st.status in RESOLVED or st.flagged for st in self._steps.values()):
            skip_events, skip_alert = self._skip(flagged_open)
            newly_flagged = []

        # 5. statuses of this frame's completions
        for st, ask, was_skipped in done_now:
            missing = any(
                self._steps[m].flagged and self._steps[m].status in OPEN
                for m in self._required[st.id]
            )
            if ask is not None:
                st.status = StepStatus.OPERATOR_CONFIRMED
            elif was_skipped or missing:
                st.status = StepStatus.OUT_OF_ORDER
            else:
                st.status = StepStatus.VERIFIED
            evidence = dict(st.sources) if st.sources else {"p_complete": st.p}
            if ask is not None:
                evidence["operator"] = 1.0
            done_events.append(
                self._stamp(
                    StepEvent, st.id, st.status, st.t_start, st.t_end, st.confidence, evidence
                )
            )
            # too_fast needs an observed end; a confirmation's t_end is only the ask time
            min_d = pr.min_duration_s.get(st.id) if ask is None else None
            if min_d is not None and st.t_start is not None and st.t_end is not None:
                dur = st.t_end - st.t_start
                if dur < min_d:
                    msg = (
                        f"Too fast: {_lower_first(st.spec.text)} took {dur:.1f} s "
                        f"(minimum {min_d:g} s). Please check it."
                    )
                    alerts.append(self._stamp(Alert, AlertKind.TOO_FAST, msg, (st.id,)))

        if newly_flagged:
            assert trigger is not None
            missing_sorted = sorted(newly_flagged, key=lambda s: s.index)
            first = missing_sorted[0]
            did = _step_phrase(self.protocol, trigger.spec)
            if _step_object(trigger.spec) != _step_object(first.spec):
                what = self.protocol.object_label(_step_object(first.spec))
                msg = f"Out of sequence: {did} before {what}."
            else:
                msg = f"Out of sequence: {did}."
            msg += f" Expected: {_lower_first(first.spec.text)}."
            ids = (trigger.id, *(s.id for s in missing_sorted))
            alerts.insert(0, self._stamp(Alert, AlertKind.OUT_OF_ORDER, msg, ids))
        if skip_alert is not None:
            alerts.append(skip_alert)

        out: list[Output] = [*done_events, *reopen_events, *skip_events, *alerts]

        # 6. prompt: after any completion or skip, or when the prompted step changed
        self._finished = all(st.status in RESOLVED for st in self._steps.values())
        if done_now or skip_events or reopened or self._target() != self._prompted:
            out.append(self._prompt())
        return out

    def _check_missing(self) -> tuple[list[_StepState], _StepState | None]:
        if self._blind():
            return [], None
        newly: list[_StepState] = []
        trigger: _StepState | None = None
        for st in self._steps.values():
            if st.status not in OPEN or st.flagged:
                continue
            if not st.clearly_low or self._has_open_ask(st.id):
                continue  # unsure or in progress: not an accusation
            deps = [
                self._steps[x]
                for x in self._dependents[st.id]
                if self._steps[x].status in DONE
                and self._steps[x].done_t is not None
                and self._steps[x].done_t >= st.pending_since
            ]
            if not deps:
                continue
            st.flagged = True
            newly.append(st)
            last = max(deps, key=lambda d: (d.done_t, d.index))
            if trigger is None or (last.done_t, last.index) > (trigger.done_t, trigger.index):
                trigger = last
        return newly, trigger

    def _skip(self, steps: list[_StepState]) -> tuple[list[Output], Alert]:
        steps = sorted(steps, key=lambda s: s.index)
        events: list[Output] = []
        for st in steps:
            st.status = StepStatus.SKIPPED
            st.flagged = False
            events.append(
                self._stamp(
                    StepEvent,
                    st.id,
                    StepStatus.SKIPPED,
                    None,
                    None,
                    1.0 - st.p,
                    st.sources,
                    retroactive=True,
                )
            )
            self._resolve_asks(st.id)
        msg = "Skipped: " + "; ".join(_lower_first(st.spec.text) for st in steps) + "."
        alert = self._stamp(Alert, AlertKind.SKIP, msg, tuple(st.id for st in steps))
        return events, alert

    def _undo_expected(self, st: _StepState) -> bool:
        for y in self._undoers[st.id]:
            ys = self._steps[y]
            if ys.gate.active:
                return True
            if (
                ys.status in DONE
                and ys.done_t is not None
                and st.done_t is not None
                and ys.done_t >= st.done_t
            ):
                return True
        return False

    def _check_timeout(self) -> list[Output]:
        t = self._t
        assert t is not None
        if self._finished or self._timeout_fired:
            return []
        if t - self._last_progress_t < self.params.timeout_s - 1e-9:
            return []
        target = self._target()
        if target is None:
            return []
        self._timeout_fired = True
        st = self._steps[target]
        out: list[Output] = []
        if st.status is StepStatus.PENDING:
            st.status = StepStatus.TIMED_OUT
            out.append(
                self._stamp(StepEvent, st.id, StepStatus.TIMED_OUT, None, None, 1.0, st.sources)
            )
        msg = (
            f"Timeout: no progress for {self.params.timeout_s:g} seconds. "
            f"Next step: {_lower_first(st.spec.text)}."
        )
        out.append(self._stamp(Alert, AlertKind.TIMEOUT, msg, (st.id,)))
        return out

    def _check_asks(self, blind: bool) -> list[Output]:
        t = self._t
        assert t is not None
        pr = self.params
        cands: list[_StepState] = []
        # blindness during an expected step: ask about the prompted step
        vis_held = self._vis_timer.update(t, blind and not self._finished)
        if vis_held and not self._vis_timer.fired:
            target = self._target()
            if target is not None and not self._has_open_ask(target):
                cands.append(self._steps[target])
            else:
                self._vis_timer.fired = True  # nothing to ask this episode
        # unsure evidence, sensors seeing
        band: list[_StepState] = []
        for st in self._steps.values():
            in_band = (
                st.status in OPEN
                and not blind
                and pr.ask_low <= st.p < pr.ask_high
                and t >= st.ask_blocked_until
                and not self._has_open_ask(st.id)
            )
            if st.band.update(t, in_band) and not st.band.fired:
                band.append(st)
        cands += sorted(band, key=self._rank)
        if not cands:
            return []
        if self._open_ask is not None:
            rec = self._asks[self._open_ask]
            if t - rec.ask.t < pr.ask_supersede_s:
                return []  # retry on a later frame
            self._close_ask(rec)
        st = cands[0]
        if st in band:
            st.band.fired = True
        if vis_held:
            self._vis_timer.fired = True
        text = st.spec.ask or f"I couldn't confirm it. Did you {_lower_first(st.spec.text)}?"
        qid = f"q{next(self._ask_seq)}-{st.id}"
        ask = self._stamp(Ask, qid, st.id, text)
        self._asks[qid] = _AskState(ask)
        self._open_ask = qid
        return [ask]

    # -- helpers ------------------------------------------------------------------------------

    def _blind(self) -> bool:
        return self._visibility is not None and self._visibility < self.params.vis_low

    def _resolved(self) -> set[str]:
        return {s for s, st in self._steps.items() if st.status in RESOLVED}

    def _target(self) -> str | None:
        nxt = self.protocol.expected_next(self._resolved())
        return nxt[0] if nxt else None

    def _rank(self, st: _StepState) -> int:
        """The prior: expected-next steps first, then protocol order."""
        expected = self.protocol.expected_next(self._resolved())
        return (0 if st.id in expected else len(self._steps)) + st.index

    def _prompt(self) -> Prompt:
        target = self._target()
        self._prompted = target
        if target is not None:
            return self._stamp(Prompt, target, self._steps[target].spec.text)
        n_skipped = sum(st.status is StepStatus.SKIPPED for st in self._steps.values())
        if self._finished and n_skipped == 0:
            text = "Procedure complete."
        elif self._finished:
            text = f"Procedure finished with {n_skipped} skipped step{'s' * (n_skipped > 1)}."
        else:
            text = "Waiting for the remaining steps to be confirmed."
        return self._stamp(Prompt, None, text)

    def _ingest_actions(self, t: float, actions: ActionScores) -> None:
        if not actions.scores:
            return
        label, score = actions.top()
        lab = label if score >= self.params.action_min and label != IDLE else None
        if lab != self._top:
            self._top = lab
            self._top_changes.append((t, lab))

    def _onset(self, step_id: str, t_end: float) -> float:
        """Start of the step's action after the previous completion (see module docstring).

        The reference is the latest ``t_end`` of another completed step that is not after this
        step's ``t_end`` (so a late operator confirmation isn't measured from a later step)."""
        ref = max(
            (
                s.t_end
                for s in self._steps.values()
                if s.id != step_id and s.status in DONE and s.t_end is not None and s.t_end <= t_end
            ),
            default=self._t0,
        )
        active: str | None = None
        for tc, lab in self._top_changes:
            if tc <= ref:
                active = lab
                continue
            if active == step_id:
                break
            if lab == step_id:
                return tc if tc <= t_end else ref
        return ref

    def _has_open_ask(self, step_id: str) -> bool:
        return any(not r.closed and r.ask.step_id == step_id for r in self._asks.values())

    def _close_ask(self, rec: _AskState) -> None:
        rec.closed = True
        if self._open_ask == rec.ask.question_id:
            self._open_ask = None

    def _resolve_asks(self, step_id: str) -> None:
        for rec in self._asks.values():
            if not rec.closed and rec.ask.step_id == step_id:
                self._close_ask(rec)

    def _stamp(self, cls: type, *args: Any, **kw: Any) -> Any:
        return cls(*args, frame_idx=self._frame_idx, t=self._t, **kw)


# --------------------------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------------------------


class _Tick(TypingProtocol):
    frame_idx: int
    t: float
    evidence: tuple[StepEvidence, ...]
    predicates: Predicates
    actions: ActionScores


AnswerPolicy = Callable[[Ask], tuple[float, AnswerValue | str] | None]


def run_ticks(
    engine: ProtocolEngine,
    ticks: Iterable[_Tick],
    *,
    answer: AnswerPolicy | AnswerValue | str | None = None,
    answer_delay_s: float = 1.0,
    use_actions: bool = True,
    use_visibility: bool = True,
    finish: bool = True,
) -> list[Output]:
    """Drive the engine from replay ticks (``ReplaySource``) or anything shaped like them.

    ``answer`` scripts the operator: ``"yes"``/``"no"`` answers every Ask ``answer_delay_s``
    after it is asked; a callable ``ask -> (delay_s, value) | None`` decides per Ask.
    Answers are applied before the first frame at or after their due time.
    """
    if answer is not None and not callable(answer):
        fixed = AnswerValue(answer)

        def policy(_ask: Ask) -> tuple[float, AnswerValue | str] | None:
            return answer_delay_s, fixed

    else:
        policy = answer  # type: ignore[assignment]

    out: list[Output] = []
    due: list[tuple[float, Answer]] = []
    for tick in ticks:
        for when, ans in sorted(due, key=lambda d: d[0]):
            if when <= tick.t + 1e-9:
                out += engine.answer(ans)
        due = [(w, a) for w, a in due if w > tick.t + 1e-9]
        new = engine.update(
            tick.frame_idx,
            tick.t,
            tick.evidence,
            visibility=tick.predicates.visibility if use_visibility else None,
            actions=tick.actions if use_actions else None,
        )
        out += new
        if policy is not None:
            for o in new:
                if isinstance(o, Ask):
                    decision = policy(o)
                    if decision is not None:
                        delay, value = decision
                        ans = Answer(o.question_id, value, InputVia.VOICE, t=o.t + delay)
                        due.append((o.t + delay, ans))
    if finish:
        out += engine.finish()
    return out


def publish(bus: Bus, outputs: Iterable[Output]) -> None:
    for o in outputs:
        bus.publish(OUTPUT_TOPICS[type(o)], o)


class EngineRunner:
    """Drives an engine from the bus in the calling thread.

    Subscribes to ``evidence``, ``predicates``, ``actions``, ``answers`` and ``commands``;
    publishes to ``events``, ``alerts``, ``prompts`` and ``asks``. A frame is processed as soon
    as evidence for every step has arrived, or when evidence for a later frame arrives. Evidence
    for an already-processed frame is dropped and counted in ``late``. Visibility and action
    scores are matched to the frame (latest at or before it).
    """

    def __init__(self, engine: ProtocolEngine, bus: Bus, *, maxsize: int = 8192) -> None:
        self.engine = engine
        self.bus = bus
        self.sub = bus.subscribe(
            [Topic.EVIDENCE, Topic.PREDICATES, Topic.ACTIONS, Topic.ANSWERS, Topic.COMMANDS],
            maxsize=maxsize,
            policy="drop_newest",
            name="engine",
        )
        self.late = 0
        self.outputs: list[Output] = []
        self._frame: int | None = None
        self._last_done: int | None = None
        self._t = 0.0
        self._buf: dict[str, StepEvidence] = {}
        self._vis: dict[int, float] = {}
        self._actions: dict[int, ActionScores] = {}
        self._n_steps = len(engine.protocol.steps)

    def run(self, stop: threading.Event | None = None, poll_s: float = 0.1) -> list[Output]:
        """Run until the bus closes (or ``stop`` is set), then flush and ``finish()``."""
        while stop is None or not stop.is_set():
            try:
                env = self.sub.get(timeout=poll_s)
            except BusClosed:
                break
            if env is not None:
                self._handle(env.payload)
        self._flush()
        self._emit(self.engine.finish())
        self.sub.unsubscribe()
        return self.outputs

    def pump(self) -> list[Output]:
        """Process everything queued right now without blocking (for single-threaded loops).
        Returns the outputs produced by this call. Does not flush a partial frame."""
        n = len(self.outputs)
        for env in self.sub.drain():
            self._handle(env.payload)
        return self.outputs[n:]

    def _handle(self, msg: object) -> None:
        if isinstance(msg, StepEvidence):
            if self._last_done is not None and msg.frame_idx <= self._last_done:
                self.late += 1
                return
            if self._frame is not None and msg.frame_idx != self._frame:
                if msg.frame_idx < self._frame:
                    self.late += 1
                    return
                self._flush()
            self._frame, self._t = msg.frame_idx, msg.t
            self._buf[msg.step_id] = msg
            if len(self._buf) >= self._n_steps:
                self._flush()
        elif isinstance(msg, Predicates):
            self._vis[msg.frame_idx] = msg.visibility
            _prune(self._vis, msg.frame_idx)
        elif isinstance(msg, ActionScores):
            self._actions[msg.frame_idx] = msg
            _prune(self._actions, msg.frame_idx)
        elif isinstance(msg, Answer):
            self._emit(self.engine.answer(msg))
        elif isinstance(msg, Command):
            self._emit(self.engine.command(msg))

    def _flush(self) -> None:
        if self._frame is None:
            return
        f = self._frame
        self._emit(
            self.engine.update(
                f,
                self._t,
                list(self._buf.values()),
                visibility=_latest_at(self._vis, f),
                actions=_latest_at(self._actions, f),
            )
        )
        self._last_done, self._frame, self._buf = f, None, {}

    def _emit(self, outputs: list[Output]) -> None:
        self.outputs += outputs
        with contextlib.suppress(BusClosed):  # end of stream: run() still returns outputs
            publish(self.bus, outputs)


def _latest_at(d: Mapping[int, Any], f: int) -> Any:
    keys = [k for k in d if k <= f]
    return d[max(keys)] if keys else None


def _prune(d: dict[int, Any], newest: int, keep: int = 64) -> None:
    if len(d) > 2 * keep:
        for k in [k for k in d if k < newest - keep]:
            del d[k]


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def describe(o: Output) -> str:
    """One-line ASCII description of an output (for logs, the CLI and tests)."""
    if isinstance(o, StepEvent):
        retro = " (retro)" if o.retroactive else ""
        return f"event {o.step_id} {o.status.value}{retro}"
    if isinstance(o, Alert):
        return f"alert {o.kind.value} {','.join(o.step_ids)}"
    if isinstance(o, Prompt):
        return f"prompt {o.next_step_id or '-'}"
    return f"ask {o.step_id}"


def main(argv: list[str] | None = None) -> int:
    from actus.capture.replay import DEFAULT_PROTOCOL, SCENARIOS, ReplaySource
    from actus.protocol.schema import load_protocol

    ap = argparse.ArgumentParser(description="Run a replay scenario through the protocol engine.")
    ap.add_argument("--scenario", default="correct", choices=sorted(SCENARIOS))
    ap.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--noise", type=float, default=0.04)
    ap.add_argument("--answer", choices=["yes", "no"], help="auto-answer every Ask")
    ap.add_argument("--answer-delay", type=float, default=1.0)
    args = ap.parse_args(argv)

    proto = load_protocol(args.protocol)
    src = ReplaySource(
        args.scenario, protocol=proto, fps=args.fps, seed=args.seed, noise=args.noise
    )
    engine = ProtocolEngine(proto)
    outputs = run_ticks(engine, src, answer=args.answer, answer_delay_s=args.answer_delay)
    print(f"# scenario {args.scenario} (SIMULATED replay), seed {args.seed}, {len(src)} frames")
    for o in outputs:
        text = o.message if isinstance(o, Alert) else getattr(o, "text", "")
        extra = ""
        if isinstance(o, StepEvent) and o.t_end is not None:
            t0 = f"{o.t_start:.2f}" if o.t_start is not None else "?"
            extra = f" [{t0}-{o.t_end:.2f}s] conf={o.confidence:.2f}"
        line = f"f{o.frame_idx:5d} t={o.t:7.2f}  {describe(o):34s}{extra}  {text}"
        print(line.encode("ascii", "replace").decode("ascii"))
    print("# outcomes: " + ", ".join(f"{r['step_id']}={r['status']}" for r in engine.outcomes()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
