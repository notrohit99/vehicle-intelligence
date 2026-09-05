"""Orchestrates tracking + ANPR OCR into combined vehicle records."""

from __future__ import annotations

import tempfile
import time
import gc
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from clients.anpr_client import ANPRClient
from clients.tracking_client import TrackingClient
from merger import OCRResult, build_vehicle_records
from schemas import ProcessVideoResponse, new_job_id


class VideoOrchestrator:
    def __init__(
        self,
        tracking_client: TrackingClient,
        anpr_client: ANPRClient,
    ) -> None:
        self.tracking_client = tracking_client
        self.anpr_client = anpr_client

    async def process_video(
        self,
        video_bytes: bytes,
        camera_id: str,
        filename: str = "upload.mp4",
    ) -> ProcessVideoResponse:
        start = time.time()
        job_started_at = datetime.now(timezone.utc)
        job_id = new_job_id()

        tracking_result = await self.tracking_client.track_video(video_bytes, filename=filename)
        detections = tracking_result.get("detections", [])
        total_frames = int(tracking_result.get("total_frames", 0))
        fps = float(tracking_result.get("fps", 30.0))

        detections_by_track = _group_detections_by_track(detections)
        ocr_by_tracking_id = await _ocr_tracks_from_video(
            video_bytes=video_bytes,
            filename=filename,
            detections_by_track=detections_by_track,
            anpr_client=self.anpr_client,
            fps=fps,
        )

        vehicles = build_vehicle_records(
            camera_id=camera_id,
            ocr_by_tracking_id=ocr_by_tracking_id,
            job_started_at=job_started_at,
            detections_by_track=detections_by_track,
        )

        return ProcessVideoResponse(
            job_id=job_id,
            camera_id=camera_id,
            total_frames=total_frames,
            processing_time_sec=round(time.time() - start, 4),
            vehicles=vehicles,
        )


def _group_detections_by_track(detections: list[dict]) -> dict[int, list[dict]]:
    """Group detections by tracking_id, highest confidence first."""
    grouped: dict[int, list[dict]] = defaultdict(list)
    for det in detections:
        grouped[int(det["tracking_id"])].append(det)

    for tracking_id in grouped:
        grouped[tracking_id].sort(key=lambda d: float(d["confidence"]), reverse=True)
    return grouped


async def _ocr_tracks_from_video(
    video_bytes: bytes,
    filename: str,
    detections_by_track: dict[int, list[dict]],
    anpr_client: ANPRClient,
    fps: float,
) -> dict[int, OCRResult]:
    suffix = Path(filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    ocr_results: dict[int, OCRResult] = {}

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Could not open video for OCR cropping")

        for tracking_id, candidates in detections_by_track.items():
            # Test at most top 3 highest-confidence frames per track to avoid redundant OCR work
            for det in candidates[:3]:
                frame_index = int(det["frame"])
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = cap.read()
                if not success:
                    continue

                bbox = det["bounding_box"]
                x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                ok, encoded = cv2.imencode(".jpg", crop)
                if not ok:
                    continue

                ocr_response = await anpr_client.ocr_crop(
                    encoded.tobytes(),
                    offset_x=x1,
                    offset_y=y1,
                )
                readings = ocr_response.get("readings", [])
                if not readings:
                    continue

                best_reading = max(readings, key=lambda r: float(r["confidence"]))
                conf = float(best_reading["confidence"])
                timestamp_sec = frame_index / fps if fps > 0 else 0.0

                ocr_results[tracking_id] = OCRResult(
                    plate_number=best_reading["plate_number"],
                    confidence=conf,
                    bbox=best_reading["bbox"],
                    frame=frame_index,
                    timestamp_sec=timestamp_sec,
                    vehicle_bbox=[x1, y1, x2, y2],
                    class_name=str(det.get("class_name", "")),
                )
                # Early exit: if confidence is already good (>= 75%), stop testing more frames for this vehicle
                if conf >= 0.75:
                    break

        cap.release()
        cap = None
        gc.collect()
        time.sleep(0.5)

    finally:
        for _ in range(5):
            try:
                Path(tmp_path).unlink(missing_ok=True)
                break
            except PermissionError:
                gc.collect()
                time.sleep(0.5)
    return ocr_results
