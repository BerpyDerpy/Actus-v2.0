"""T-SPATIAL → T-ENGINE: spatial evidence is a drop-in for the replay's stand-in evidence.

For each scenario the engine's event sequence from *spatial's* evidence (world model over the
replay detections) must equal the sequence from the replay's own synthetic evidence. Skipped
where the engine lane is not merged yet. ``occluded`` and ``unexpected`` are excluded on
purpose; see ``Vault/T-SPATIAL.md`` (the replay has no hands, so handling during a blackout and
a 2 s placement are not observable the way the stand-in evidence assumes).
"""

import types

import pytest

from actus.capture.replay import ReplaySource
from actus.spatial.estimator import SpatialEstimator

engine = pytest.importorskip("actus.protocol.engine")


def _describe_run(ticks, protocol):
    out = engine.run_ticks(engine.ProtocolEngine(protocol), ticks)
    return [engine.describe(o) for o in out]


@pytest.mark.parametrize("scenario", ["correct", "skip_S2", "S4_before_S2", "timeout"])
@pytest.mark.parametrize("seed", [0, 3])
def test_engine_sees_the_same_procedure_through_spatial(scenario, seed):
    src = ReplaySource(scenario, seed=seed)
    est = SpatialEstimator(src.protocol)
    spatial_ticks, replay_ticks = [], []
    for tk in src:
        sf = est.update(tk.frame_idx, tk.t, detections=tk.detections)
        spatial_ticks.append(
            types.SimpleNamespace(
                frame_idx=tk.frame_idx,
                t=tk.t,
                evidence=sf.evidence,
                predicates=sf.predicates,
                actions=tk.actions,
            )
        )
        replay_ticks.append(tk)
    got = _describe_run(spatial_ticks, src.protocol)
    assert got == _describe_run(replay_ticks, src.protocol)
    assert "alert unexpected" not in " ".join(got)
