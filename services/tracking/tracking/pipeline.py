"""Core tracking pipeline extracted from object_tracking.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from supervision import Detections

from bytetrack.byte_track import ByteTrack

VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck
TRACKING_PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = TRACKING_PROJECT_DIR / "yolov8n.pt"


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class VehicleTrack:
    tracking_id: int
    frame: int
    timestamp_sec: float
    class_id: int
    class_name: str
    confidence: float
    bounding_box: BoundingBox


class TrackingPipeline:
    """YOLOv8 detection + ByteTrack — returns structured tracks per frame."""

    def __init__(self, yolo_weights: str | Path = DEFAULT_MODEL_PATH) -> None:
        self.model = YOLO(str(yolo_weights))
        self.model.fuse()
        self.class_names = self.model.model.names
        self.byte_tracker = ByteTrack(
            track_thresh=0.25,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=30,
        )

    def reset(self) -> None:
        self.byte_tracker.reset()

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        fps: float = 30.0,
    ) -> list[VehicleTrack]:
        results = self.model(frame, imgsz=640, verbose=False)[0]
        detections = Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, VEHICLE_CLASS_IDS)]
        detections = self.byte_tracker.update_with_detections(detections)

        timestamp_sec = frame_index / fps if fps > 0 else 0.0
        tracks: list[VehicleTrack] = []

        for confidence, class_id, tracker_id, xyxy in zip(
            detections.confidence,
            detections.class_id,
            detections.tracker_id,
            detections.xyxy,
        ):
            x1, y1, x2, y2 = map(int, xyxy)
            tracks.append(
                VehicleTrack(
                    tracking_id=int(tracker_id),
                    frame=frame_index,
                    timestamp_sec=round(timestamp_sec, 4),
                    class_id=int(class_id),
                    class_name=str(self.class_names[int(class_id)]),
                    confidence=float(confidence),
                    bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )

        return tracks
