"""Integration service schemas."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class VehicleRecordSchema(BaseModel):
    global_vehicle_candidate: str
    tracking_id: int
    plate_confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: str
    camera_id: str
    bbox: list[int] = Field(..., min_length=4, max_length=4)
    vehicle_bbox: list[int] | None = Field(default=None, min_length=4, max_length=4)
    class_name: str | None = None
    frame: int | None = None


class ProcessVideoResponse(BaseModel):
    job_id: str
    camera_id: str
    total_frames: int
    processing_time_sec: float
    vehicles: list[VehicleRecordSchema]


def new_job_id() -> str:
    return str(uuid.uuid4())


def iso_timestamp(base: datetime, offset_sec: float) -> str:
    moment = base.timestamp() + offset_sec
    return datetime.fromtimestamp(moment, tz=timezone.utc).isoformat().replace("+00:00", "Z")
