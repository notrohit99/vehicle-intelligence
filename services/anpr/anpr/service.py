"""High-level service for processing images and videos through the ANPR pipeline."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from anpr.pipeline import ANPRPipeline, ANPRState, BoundingBox, PlateDetection
from anpr.schemas import ANPRResponse, BoundingBoxSchema, DetectionSchema


def _to_bbox_schema(bbox: BoundingBox) -> BoundingBoxSchema:
    return BoundingBoxSchema(x1=bbox.x1, y1=bbox.y1, x2=bbox.x2, y2=bbox.y2)


def _to_detection_schema(det: PlateDetection) -> DetectionSchema:
    return DetectionSchema(
        plate_number=det.plate_number,
        confidence=det.confidence,
        frame=det.frame,
        timestamp_sec=det.timestamp_sec,
        bounding_box=_to_bbox_schema(det.bounding_box),
        vehicle_id=det.vehicle_id,
        vehicle_bounding_box=(
            _to_bbox_schema(det.vehicle_bounding_box) if det.vehicle_bounding_box else None
        ),
    )


class ANPRService:
    """Orchestrates image/video ingestion and returns structured detections."""

    def __init__(self, pipeline: ANPRPipeline | None = None) -> None:
        self.pipeline = pipeline or ANPRPipeline()

    def process_image_bytes(self, data: bytes) -> ANPRResponse:
        start = time.time()
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image. Upload a valid JPEG or PNG file.")

        detections = self.pipeline.process_frame(
            frame,
            frame_index=0,
            fps=0.0,
            persist_tracking=False,
        )

        return ANPRResponse(
            source_type="image",
            total_frames=1,
            processing_time_sec=round(time.time() - start, 4),
            detections=[_to_detection_schema(d) for d in detections],
        )

    def process_video_bytes(self, data: bytes, filename: str = "upload.mp4") -> ANPRResponse:
        suffix = Path(filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            return self.process_video_path(tmp_path)
        finally:
            import gc
            for _ in range(5):
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                    break
                except Exception:
                    gc.collect()
                    time.sleep(0.3)

    def process_video_path(self, video_path: str) -> ANPRResponse:
        start = time.time()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        state = ANPRState()
        all_detections: list[PlateDetection] = []
        frame_index = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_detections = self.pipeline.process_frame(
                frame,
                frame_index=frame_index,
                fps=fps,
                state=state,
                persist_tracking=True,
            )
            all_detections.extend(frame_detections)
            frame_index += 1

        cap.release()
        del cap
        import gc
        gc.collect()

        return ANPRResponse(
            source_type="video",
            total_frames=frame_index,
            processing_time_sec=round(time.time() - start, 4),
            detections=[_to_detection_schema(d) for d in all_detections],
        )
