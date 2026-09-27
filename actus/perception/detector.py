"""Object detector: one interface, two backends, both tracked with Ultralytics ByteTrack.

* ``custom``: a YOLO11n checkpoint (``.pt`` or ``.onnx``) trained by T-DET-TRAIN on the
  project's own classes (``outer_box_open``, ``outer_box_closed``, ``lid``, ``red_box``,
  ``yellow_box``).
* ``open_vocab``: YOLO-World with open-vocabulary text prompts (``configs/default.yaml``'s
  ``detector.prompts``), so the pipeline runs before the custom detector is trained. This is the
  default and the fallback. Detection class names are the prompt dict's *keys* (the project's
  canonical names), never YOLO-World's raw prompt text, so downstream code never has to care
  which backend produced a ``Detection``.

Both backends call ``model.track(..., persist=True)`` so ``track_id`` comes from Ultralytics'
ByteTrack, not reassigned per frame. Neither backend sets ``bbox_rack_xy`` -- that projection is
``actus/spatial``'s job, done once a rack frame is available.

    det = Detector.from_config(cfg.detector)
    dets = det.detect(frame_packet)   # DetectionSet

Sanity check against a live camera (FPS only)::

    python -m actus.perception.detector --source 0 --seconds 10
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from typing import Protocol as TypingProtocol

from actus.core.config import resolve_path
from actus.core.contracts import Detection, DetectionSet, FramePacket


class DetectorBackend(TypingProtocol):
    def detect(self, frame: FramePacket) -> DetectionSet: ...
    def reset(self) -> None: ...


@dataclass
class CustomDetector:
    """A trained YOLO11n checkpoint, tracked with ByteTrack."""

    weights: str | Path
    conf: float = 0.25
    imgsz: int = 640
    tracker: str = "bytetrack.yaml"
    _model: Any = field(default=None, init=False, repr=False)

    def _load(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(resolve_path(self.weights)))
        return self._model

    def detect(self, frame: FramePacket) -> DetectionSet:
        model = self._load()
        results = model.track(
            frame.image,
            conf=self.conf,
            imgsz=self.imgsz,
            tracker=self.tracker,
            persist=True,
            verbose=False,
        )
        return _to_detection_set(results[0], frame, backend="custom")

    def reset(self) -> None:
        _reset_tracker(self._model)


@dataclass
class OpenVocabDetector:
    """YOLO-World with text prompts. ``prompts`` maps canonical class name -> free-text prompt,
    e.g. ``{"red_box": "red box"}``; ``Detection.cls`` is always the canonical name."""

    prompts: dict[str, str]
    weights: str = "yolov8s-worldv2.pt"
    conf: float = 0.25
    imgsz: int = 640
    tracker: str = "bytetrack.yaml"
    _model: Any = field(default=None, init=False, repr=False)
    _classes: list[str] = field(default=None, init=False, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not self.prompts:
            raise ValueError("OpenVocabDetector needs at least one prompt")

    def _load(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            self._classes = list(self.prompts.keys())
            model = YOLO(self.weights)
            model.set_classes(list(self.prompts.values()))
            self._model = model
        return self._model

    def detect(self, frame: FramePacket) -> DetectionSet:
        model = self._load()
        results = model.track(
            frame.image,
            conf=self.conf,
            imgsz=self.imgsz,
            tracker=self.tracker,
            persist=True,
            verbose=False,
        )
        return _to_detection_set(results[0], frame, backend="open_vocab", class_names=self._classes)

    def reset(self) -> None:
        _reset_tracker(self._model)


def _reset_tracker(model: Any) -> None:
    predictor = getattr(model, "predictor", None) if model is not None else None
    for t in getattr(predictor, "trackers", []):
        t.reset()


def _to_detection_set(
    result: Any, frame: FramePacket, *, backend: str, class_names: list[str] | None = None
) -> DetectionSet:
    dets: list[Detection] = []
    boxes = result.boxes
    if boxes is not None and len(boxes):
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls_idx = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)
        names = class_names if class_names is not None else result.names
        for i in range(len(xyxy)):
            dets.append(
                Detection(
                    track_id=None if ids[i] is None else int(ids[i]),
                    cls=str(names[int(cls_idx[i])]),
                    conf=float(conf[i]),
                    bbox_xyxy=tuple(float(v) for v in xyxy[i]),
                )
            )
    return DetectionSet(frame.frame_idx, frame.t_capture, dets, backend)


class Detector:
    """Single detector interface; wraps whichever backend ``cfg.backend`` selects."""

    def __init__(self, backend: DetectorBackend, backend_name: str) -> None:
        self._backend = backend
        self.backend_name = backend_name

    @classmethod
    def from_config(cls, cfg: Any) -> Detector:
        name = cfg.backend
        if name == "custom":
            backend: DetectorBackend = CustomDetector(
                weights=cfg.custom_weights, conf=cfg.conf, imgsz=cfg.imgsz, tracker=cfg.tracker
            )
        elif name == "open_vocab":
            backend = OpenVocabDetector(
                prompts=dict(cfg.prompts),
                weights=cfg.open_vocab_weights,
                conf=cfg.conf,
                imgsz=cfg.imgsz,
                tracker=cfg.tracker,
            )
        else:
            raise ValueError(f"unknown detector backend {name!r}")
        return cls(backend, name)

    def detect(self, frame: FramePacket) -> DetectionSet:
        return self._backend.detect(frame)

    def reset(self) -> None:
        self._backend.reset()


# --------------------------------------------------------------------------------------------
# CLI: live FPS sanity check
# --------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    from actus.core.config import load_config

    ap = argparse.ArgumentParser(description="Live sanity check: detector FPS from a camera.")
    ap.add_argument("--source", default="0", help="camera index, file path, or rtsp:// URL")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--backend", choices=["custom", "open_vocab"], default=None)
    args = ap.parse_args(argv)
    source: int | str = int(args.source) if args.source.lstrip("-").isdigit() else args.source

    from actus.capture.camera import Camera
    from actus.core.bus import Bus
    from actus.core.contracts import Topic

    cfg = load_config(overrides=(["detector.backend=" + args.backend] if args.backend else []))
    det = Detector.from_config(cfg.detector)
    bus = Bus()
    sub = bus.subscribe(Topic.FRAMES, maxsize=4, policy="drop_oldest", name="sanity")
    cam = Camera(source, source_id="sanity")
    cam.start(bus)
    n, t0 = 0, time.monotonic()
    try:
        while time.monotonic() - t0 < args.seconds:
            env = sub.get(timeout=1.0)
            if env is None:
                continue
            dets = det.detect(env.payload)
            n += 1
            print(f"frame {env.payload.frame_idx}: {len(dets.detections)} detections")
    finally:
        cam.stop()
        bus.close()
    dt = time.monotonic() - t0
    print(f"{n} frames in {dt:.1f}s = {n / dt:.1f} FPS" if dt > 0 else "no frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
