"""Detector tests use fake Ultralytics-shaped objects rather than real weights, so they run
offline and without the `perception` extra (only ultralytics's *return shapes* are mimicked;
`ultralytics` itself is never imported unless a real `_load()` is exercised, which these tests
avoid by injecting `_model` directly)."""

import numpy as np
import pytest

from actus.core.config import load_config
from actus.core.contracts import FramePacket
from actus.perception.detector import CustomDetector, Detector, OpenVocabDetector


class _T:
    """Mimics the `.cpu().numpy()` chain of a torch.Tensor, as used on ultralytics `Boxes`."""

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class FakeBoxes:
    def __init__(self, xyxy, conf, cls, ids=None):
        self.xyxy = _T(xyxy)
        self.conf = _T(conf)
        self.cls = _T(cls)
        self.id = _T(ids) if ids is not None else None

    def __len__(self):
        return len(self.xyxy._arr)


class FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class FakePredictor:
    def __init__(self, trackers):
        self.trackers = trackers


class FakeTracker:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


class FakeModel:
    def __init__(self, result):
        self._result = result
        self.calls: list[dict] = []
        self.predictor = None

    def track(self, image, **kwargs):
        self.calls.append(kwargs)
        return [self._result]

    def set_classes(self, classes):
        self.set_classes_called_with = classes


def _frame(frame_idx=0, t=0.0) -> FramePacket:
    return FramePacket(frame_idx, t, np.zeros((10, 10, 3), dtype=np.uint8), "cam0")


def test_custom_detector_converts_boxes_to_contract():
    boxes = FakeBoxes(xyxy=[[10.0, 20.0, 30.0, 40.0]], conf=[0.87], cls=[2], ids=[5])
    result = FakeResult(boxes, names={0: "outer_box_closed", 1: "lid", 2: "red_box"})
    model = FakeModel(result)
    det = CustomDetector(weights="models/detector/best.onnx", conf=0.3, imgsz=640)
    det._model = model

    dets = det.detect(_frame(3, 1.5))

    assert (dets.frame_idx, dets.t, dets.backend) == (3, 1.5, "custom")
    assert len(dets.detections) == 1
    d = dets.detections[0]
    assert d.track_id == 5
    assert d.cls == "red_box"
    assert d.conf == pytest.approx(0.87)
    assert d.bbox_xyxy == (10.0, 20.0, 30.0, 40.0)
    assert d.bbox_rack_xy is None
    assert model.calls[0]["conf"] == 0.3
    assert model.calls[0]["persist"] is True


def test_no_boxes_gives_empty_detection_set():
    boxes = FakeBoxes(xyxy=np.zeros((0, 4)), conf=[], cls=[])
    model = FakeModel(FakeResult(boxes, names={}))
    det = CustomDetector(weights="x")
    det._model = model

    dets = det.detect(_frame())
    assert dets.detections == ()


def test_track_id_none_when_tracker_has_no_id():
    boxes = FakeBoxes(xyxy=[[0.0, 0.0, 1.0, 1.0]], conf=[0.5], cls=[0], ids=None)
    model = FakeModel(FakeResult(boxes, names={0: "red_box"}))
    det = CustomDetector(weights="x")
    det._model = model

    dets = det.detect(_frame())
    assert dets.detections[0].track_id is None


def test_open_vocab_detector_reports_canonical_class_names_not_prompt_text():
    boxes = FakeBoxes(xyxy=[[0, 0, 5, 5], [1, 1, 6, 6]], conf=[0.5, 0.6], cls=[0, 1], ids=[10, 11])
    result = FakeResult(boxes, names=["red box", "yellow box"])  # model.names post set_classes
    model = FakeModel(result)
    det = OpenVocabDetector(prompts={"red_box": "red box", "yellow_box": "yellow box"})
    det._model = model
    det._classes = ["red_box", "yellow_box"]  # normally set inside `_load()`

    dets = det.detect(_frame())
    assert [d.cls for d in dets.detections] == ["red_box", "yellow_box"]
    assert dets.backend == "open_vocab"


def test_open_vocab_detector_requires_prompts():
    with pytest.raises(ValueError):
        OpenVocabDetector(prompts={})


def test_reset_resets_trackers_without_reloading_model():
    model = FakeModel(FakeResult(FakeBoxes([], [], []), names={}))
    tracker = FakeTracker()
    model.predictor = FakePredictor([tracker])
    det = CustomDetector(weights="x")
    det._model = model

    det.reset()
    assert tracker.reset_calls == 1
    assert det._model is model  # reset must not force a reload


def test_reset_is_a_noop_before_the_model_is_loaded():
    det = CustomDetector(weights="x")
    det.reset()  # must not raise


def test_detector_from_config_defaults_to_open_vocab():
    cfg = load_config()
    det = Detector.from_config(cfg.detector)
    assert det.backend_name == "open_vocab"
    assert isinstance(det._backend, OpenVocabDetector)
    assert det._backend.prompts["red_box"] == "red box"


def test_detector_from_config_selects_custom():
    cfg = load_config(overrides=["detector.backend=custom"])
    det = Detector.from_config(cfg.detector)
    assert det.backend_name == "custom"
    assert isinstance(det._backend, CustomDetector)


def test_detector_from_config_rejects_unknown_backend():
    cfg = load_config(overrides=["detector.backend=nope"])
    with pytest.raises(ValueError):
        Detector.from_config(cfg.detector)
