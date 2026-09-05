"""Merge tracking detections with OCR readings into vehicle records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

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
    detections_by_track: dict[int, list[dict]] | None = None,
) -> list[VehicleRecordSchema]:
    records: list[VehicleRecordSchema] = []
    seen_track_ids: set[int] = set()

    for tracking_id in sorted(ocr_by_tracking_id):
        result = ocr_by_tracking_id[tracking_id]
        seen_track_ids.add(tracking_id)
        records.append(
            VehicleRecordSchema(
                global_vehicle_candidate=result.plate_number,
                plate_number=result.plate_number,
                tracking_id=tracking_id,
                plate_confidence=result.confidence,
                timestamp=iso_timestamp(job_started_at, result.timestamp_sec),
                timestamp_sec=result.timestamp_sec,
                camera_id=camera_id,
                bbox=result.bbox,
                plate_bbox=result.bbox,
                vehicle_bbox=result.vehicle_bbox,
                class_name=result.class_name,
                frame=result.frame,
            )
        )

    if detections_by_track:
        for tracking_id, candidates in sorted(detections_by_track.items()):
            if tracking_id not in seen_track_ids and candidates:
                best = candidates[0]
                bb = best.get("bounding_box", {})
                v_bbox = [
                    int(bb.get("x1", 0)),
                    int(bb.get("y1", 0)),
                    int(bb.get("x2", 0)),
                    int(bb.get("y2", 0)),
                ]
                frame_idx = int(best.get("frame", 0))
                ts_sec = float(best.get("timestamp_sec", 0.0))
                c_name = str(best.get("class_name", "car"))
                records.append(
                    VehicleRecordSchema(
                        global_vehicle_candidate="Not detected",
                        plate_number="Not detected",
                        tracking_id=tracking_id,
                        plate_confidence=None,
                        timestamp=iso_timestamp(job_started_at, ts_sec),
                        timestamp_sec=ts_sec,
                        camera_id=camera_id,
                        bbox=None,
                        plate_bbox=None,
                        vehicle_bbox=v_bbox,
                        class_name=c_name,
                        frame=frame_idx,
                    )
                )

    return records
