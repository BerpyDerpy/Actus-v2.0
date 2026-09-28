"""Trained models that act as evidence sources (T-HOI-TEMP).

* ``features``  -- the one feature spec, shared by offline training and live inference
* ``nn``        -- the two tiny architectures, in torch (training) and numpy (runtime)
* ``hoi``       -- hand-object interaction: {none, touching, holding} per (hand, object)
* ``temporal``  -- the temporal action model (HAR): which step appears to be in progress
* ``fusion``    -- folds the temporal model into StepEvidence, capped so it cannot invent steps
* ``runtime``   -- ``ModelStack``, the one-call-per-frame façade for the pipeline
* ``rackfit``   -- a self-calibrating rack frame for footage whose rig predates a measured
                   ``configs/rack.yaml``

The models are sensors. Only ``actus/protocol/`` decides that a step happened.
"""

from actus.models.features import (
    SPEC_VERSION,
    FeatureLayout,
    FeatureParams,
    FrameFeatures,
    FrameFeaturizer,
)
from actus.models.fusion import EvidenceFusion
from actus.models.hoi import HOI_CLASSES, HOIModel, HOIResult
from actus.models.runtime import ModelFrame, ModelStack
from actus.models.temporal import TemporalModel

__all__ = [
    "HOI_CLASSES",
    "SPEC_VERSION",
    "EvidenceFusion",
    "FeatureLayout",
    "FeatureParams",
    "FrameFeatures",
    "FrameFeaturizer",
    "HOIModel",
    "HOIResult",
    "ModelFrame",
    "ModelStack",
    "TemporalModel",
]
