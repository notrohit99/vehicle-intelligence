"""High-level tracking service for video and frame processing."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from tracking.pipeline import BoundingBox, TrackingPipeline, VehicleTrack
from tracking.schemas import BoundingBoxSchema, TrackDetectionSchema, TrackingResponse


def _to_track_schema(track: VehicleTrack) -> TrackDetectionSchema:
    bb = track.bounding_box
    return TrackDetectionSchema(
        tracking_id=track.tracking_id,
        frame=track.frame,
        timestamp_sec=track.timestamp_sec,
        class_id=track.class_id,
        class_name=track.class_name,
        confidence=track.confidence,
        bounding_box=BoundingBoxSchema(x1=bb.x1, y1=bb.y1, x2=bb.x2, y2=bb.y2),
    )


class TrackingService:
    def __init__(self, pipeline: TrackingPipeline | None = None) -> None:
        self.pipeline = pipeline or TrackingPipeline()

    def process_frame_bytes(
        self,
        data: bytes,
        frame_index: int = 0,
        fps: float = 30.0,
        *,
        reset_tracker: bool = False,
    ) -> list[TrackDetectionSchema]:
        if reset_tracker:
            self.pipeline.reset()

        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image. Upload a valid JPEG or PNG file.")

        tracks = self.pipeline.process_frame(frame, frame_index=frame_index, fps=fps)
        return [_to_track_schema(t) for t in tracks]

    def process_video_bytes(self, data: bytes, filename: str = "upload.mp4") -> TrackingResponse:
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

    def process_video_path(self, video_path: str) -> TrackingResponse:
        start = time.time()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.pipeline.reset()

        all_tracks: list[TrackDetectionSchema] = []
        frame_index = 0
        frame_stride = 2  # Process every 2nd frame for 2x faster execution

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if frame_index % frame_stride == 0:
                tracks = self.pipeline.process_frame(frame, frame_index=frame_index, fps=fps)
                all_tracks.extend(_to_track_schema(t) for t in tracks)

            frame_index += 1

        cap.release()
        del cap
        import gc
        gc.collect()

        return TrackingResponse(
            source_type="video",
            total_frames=frame_index,
            fps=fps,
            processing_time_sec=round(time.time() - start, 4),
            detections=all_tracks,
        )
