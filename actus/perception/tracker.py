"""Track-continuity bookkeeping over a stream of ``DetectionSet``s.

Ultralytics' ByteTrack (used by ``detector.py``, via ``model.track()``) assigns ``track_id`` per
frame; this module turns that raw per-frame stream into per-track continuity: how long a track
has been seen, and how many consecutive frames it has just been missing. ``world_state.py``
(T-SPATIAL) uses this to decide whether a briefly-missing object is occluded rather than gone --
it owns that interpretation; this module only counts frames.

    reg = TrackRegistry(max_missed_frames=45)
    tracks = reg.update(detection_set)   # {track_id: TrackInfo}, current snapshot
"""

from __future__ import annotations

from dataclasses import dataclass

from actus.core.contracts import Detection, DetectionSet


@dataclass(frozen=True)
class TrackInfo:
    track_id: int
    cls: str
    conf: float
    bbox_xyxy: tuple[float, float, float, float]
    first_seen_frame: int
    last_seen_frame: int
    missed_frames: int  # consecutive frames since last_seen_frame; 0 if seen this frame

    @property
    def age_frames(self) -> int:
        return self.last_seen_frame - self.first_seen_frame + 1


@dataclass
class _State:
    first_seen_frame: int
    last_seen_frame: int
    last_detection: Detection


class TrackRegistry:
    """Per-track continuity over successive ``DetectionSet``s, fed in increasing ``frame_idx``
    order. Detections with ``track_id is None`` are ignored: untracked detections carry no
    continuity to bookkeep."""

    def __init__(self, *, max_missed_frames: int = 45) -> None:
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be >= 0")
        self.max_missed_frames = max_missed_frames
        self._tracks: dict[int, _State] = {}
        self._last_frame_idx: int | None = None

    def update(self, detections: DetectionSet) -> dict[int, TrackInfo]:
        if self._last_frame_idx is not None and detections.frame_idx <= self._last_frame_idx:
            raise ValueError(
                f"TrackRegistry.update got frame_idx {detections.frame_idx} at or before "
                f"{self._last_frame_idx}; frames must be fed in increasing order"
            )
        self._last_frame_idx = detections.frame_idx

        for d in detections.detections:
            if d.track_id is None:
                continue
            state = self._tracks.get(d.track_id)
            if state is None:
                self._tracks[d.track_id] = _State(detections.frame_idx, detections.frame_idx, d)
            else:
                state.last_seen_frame = detections.frame_idx
                state.last_detection = d

        stale = [
            tid
            for tid, s in self._tracks.items()
            if detections.frame_idx - s.last_seen_frame > self.max_missed_frames
        ]
        for tid in stale:
            del self._tracks[tid]

        return {
            tid: TrackInfo(
                track_id=tid,
                cls=s.last_detection.cls,
                conf=s.last_detection.conf,
                bbox_xyxy=s.last_detection.bbox_xyxy,
                first_seen_frame=s.first_seen_frame,
                last_seen_frame=s.last_seen_frame,
                missed_frames=detections.frame_idx - s.last_seen_frame,
            )
            for tid, s in self._tracks.items()
        }

    def reset(self) -> None:
        self._tracks.clear()
        self._last_frame_idx = None
