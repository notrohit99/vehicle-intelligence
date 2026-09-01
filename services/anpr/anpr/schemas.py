"""Pydantic schemas for ANPR API request/response."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBoxSchema(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class DetectionSchema(BaseModel):
    plate_number: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    frame: int
    timestamp_sec: float
    bounding_box: BoundingBoxSchema
    vehicle_id: int | None = None
    vehicle_bounding_box: BoundingBoxSchema | None = None


class ANPRResponse(BaseModel):
    source_type: str
    total_frames: int
    processing_time_sec: float
    detections: list[DetectionSchema]


class OCRReadingSchema(BaseModel):
    plate_number: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: list[int] = Field(..., min_length=4, max_length=4)


class OCRResponse(BaseModel):
    readings: list[OCRReadingSchema]
