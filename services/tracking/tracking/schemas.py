"""Pydantic schemas for the tracking API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBoxSchema(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class TrackDetectionSchema(BaseModel):
    tracking_id: int
    frame: int
    timestamp_sec: float
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bounding_box: BoundingBoxSchema


class TrackingResponse(BaseModel):
    source_type: str
    total_frames: int
    fps: float
    processing_time_sec: float
    detections: list[TrackDetectionSchema]
