"""T-HOI-TEMP → T-ENGINE: fused evidence must not change what the engine decides.

The temporal model is a sensor. These tests drive the *real* protocol engine over the replay
scenarios and check the two things that matter about letting a learned model touch evidence:

1. With the action model idle, fusion is inert -- the engine's event sequence is byte-identical
   to the one it produces from unfused evidence. Adding the model cannot perturb a system that
   works.
2. With the action model maximally, confidently wrong -- insisting the skipped step was
   performed, on every frame -- the engine still reports the skip. This is the end-to-end
   version of the ``min(armed, ...)`` bound in ``actus/models/fusion.py``: a model that has
   learned the usual order of the procedure cannot fill in a step the sensors never saw.

Mirrors ``tests/test_spatial_engine_chain.py``, which does the same for the spatial lane.
"""

from __future__ import annotations

import types

import pytest

from actus.capture.replay import ReplaySource
from actus.core.contracts import IDLE, ActionScores
from actus.models.fusion import EvidenceFusion
from actus.spatial.estimator import SpatialEstimator

engine = pytest.importorskip("actus.protocol.engine")

SCENARIOS = ["correct", "skip_S2", "S4_before_S2", "timeout"]


def run(ticks, protocol) -> list[str]:
    out = engine.run_ticks(engine.ProtocolEngine(protocol), ticks)
    return [engine.describe(o) for o in out]


def build_ticks(scenario: str, seed: int, *, actions_for=None, weight: float = 0.8):
    """Replay -> spatial evidence -> (optionally) fusion -> engine ticks.

    ``actions_for`` maps a tick to the ``ActionScores`` the temporal model would emit; None
    means no action model at all.
    """
    src = ReplaySource(scenario, seed=seed)
    est = SpatialEstimator(src.protocol)
    fusion = EvidenceFusion(src.protocol.step_ids, weight=weight)
    ticks = []
    for tk in src:
        sf = est.update(tk.frame_idx, tk.t, detections=tk.detections)
        evidence = sf.evidence
        actions = None
        if actions_for is not None:
            actions = actions_for(tk, src.protocol)
            evidence = fusion.fuse_all(evidence, actions)
        ticks.append(
            types.SimpleNamespace(
                frame_idx=tk.frame_idx,
                t=tk.t,
                evidence=evidence,
                predicates=sf.predicates,
                actions=actions,
            )
        )
    return ticks, src.protocol


def all_idle(tk, protocol) -> ActionScores:
    scores = dict.fromkeys(protocol.step_ids, 0.0)
    scores[IDLE] = 1.0
    return ActionScores(tk.frame_idx, tk.t, scores)


def insist(step_id: str):
    """An action model that claims ``step_id`` just finished, on every single frame."""

    def make(tk, protocol) -> ActionScores:
        scores = dict.fromkeys(protocol.step_ids, 0.0)
        scores[IDLE] = 1.0  # "the step is over" is what fusion reads as completion evidence
        return ActionScores(tk.frame_idx, tk.t, scores)

    make.step_id = step_id
    return make


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("seed", [0, 3])
def test_fusion_with_an_idle_model_changes_nothing(scenario, seed):
    plain, protocol = build_ticks(scenario, seed)
    fused, _ = build_ticks(scenario, seed, actions_for=all_idle)
    assert run(fused, protocol) == run(plain, protocol)


@pytest.mark.parametrize("seed", [0, 3])
def test_a_confident_wrong_model_cannot_erase_a_skip(seed):
    """The skip_S2 scenario, with a temporal model insisting S2 was performed throughout."""

    def claim_s2(tk, protocol) -> ActionScores:
        # S2 peaks first so fusion arms its "the action happened" term, then drops to idle so
        # the term reads as "and it is over" -- the strongest signal this model can send.
        scores = dict.fromkeys(protocol.step_ids, 0.0)
        if tk.frame_idx < 12:
            scores["S2"] = 1.0
        else:
            scores[IDLE] = 1.0
        return ActionScores(tk.frame_idx, tk.t, scores)

    fused, protocol = build_ticks("skip_S2", seed, actions_for=claim_s2, weight=1.0)
    events = run(fused, protocol)
    joined = " ".join(events)
    assert "alert skip" in joined or "S2 skipped" in joined
    assert "S2 verified" not in joined


@pytest.mark.parametrize("seed", [0, 3])
def test_the_correct_run_still_completes_with_fusion_on(seed):
    """Fusion must not break the happy path either."""

    def claim_current(tk, protocol) -> ActionScores:
        scores = dict.fromkeys(protocol.step_ids, 0.0)
        scores[IDLE] = 1.0
        return ActionScores(tk.frame_idx, tk.t, scores)

    fused, protocol = build_ticks("correct", seed, actions_for=claim_current, weight=1.0)
    joined = " ".join(run(fused, protocol))
    for step in protocol.step_ids:
        assert f"{step} verified" in joined
    assert "alert skip" not in joined
    assert "alert out_of_order" not in joined
