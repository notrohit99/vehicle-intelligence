"""Merge tracking detections with OCR readings into vehicle records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from schemas import VehicleRecordSchema, iso_timestamp


@dataclass
class OCRResult:
    plate_number: str
    confidence: float
    bbox: list[int]
    frame: int
    timestamp_sec: float
    vehicle_bbox: list[int]
    class_name: str


def build_vehicle_records(
    camera_id: str,
    ocr_by_tracking_id: dict[int, OCRResult],
    job_started_at: datetime,
) -> list[VehicleRecordSchema]:
    records: list[VehicleRecordSchema] = []

    for tracking_id in sorted(ocr_by_tracking_id):
        result = ocr_by_tracking_id[tracking_id]
        records.append(
            VehicleRecordSchema(
                global_vehicle_candidate=result.plate_number,
                tracking_id=tracking_id,
                plate_confidence=result.confidence,
                timestamp=iso_timestamp(job_started_at, result.timestamp_sec),
                camera_id=camera_id,
                bbox=result.bbox,
                vehicle_bbox=result.vehicle_bbox,
                class_name=result.class_name,
                frame=result.frame,
            )
        )

    return records
