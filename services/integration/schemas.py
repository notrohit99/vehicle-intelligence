"""Integration service schemas."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class VehicleRecordSchema(BaseModel):
    tracking_id: int
    plate_number: Optional[str] = None
    global_vehicle_candidate: Optional[str] = None
    plate_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    timestamp: Optional[str] = None
    timestamp_sec: Optional[float] = None
    camera_id: str
    vehicle_bbox: Optional[list[int]] = Field(default=None)
    plate_bbox: Optional[list[int]] = Field(default=None)
    bbox: Optional[list[int]] = Field(default=None)
    class_name: Optional[str] = None
    frame: Optional[int] = None


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
