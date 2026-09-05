"""FastAPI application exposing vehicle tracking as a REST service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from tracking.schemas import TrackDetectionSchema, TrackingResponse
from tracking.service import TrackingService

service: TrackingService | None = None

IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp", "image/bmp"}
VIDEO_TYPES = {"video/mp4", "video/avi", "video/x-msvideo", "video/quicktime", "video/x-matroska"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = TrackingService()
    yield


app = FastAPI(
    title="Vehicle Tracking Service",
    description="Vehicle detection and ByteTrack multi-object tracking",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/track/frame", response_model=list[TrackDetectionSchema])
async def track_single_frame(
    file: UploadFile = File(...),
    frame_index: int = Form(0),
    fps: float = Form(30.0),
    reset_tracker: bool = Form(False),
) -> list[TrackDetectionSchema]:
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if file.content_type and file.content_type not in IMAGE_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported image type: {file.content_type}")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        return service.process_frame_bytes(
            data,
            frame_index=frame_index,
            fps=fps,
            reset_tracker=reset_tracker,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/track/video", response_model=TrackingResponse)
async def track_video(file: UploadFile = File(...)) -> TrackingResponse:
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    ext = Path(file.filename or "").suffix.lower()
    is_valid = (
        (file.content_type in VIDEO_TYPES)
        or (file.content_type in ("application/octet-stream", "", None))
        or (ext in {".mp4", ".avi", ".mov", ".mkv"})
    )
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Unsupported video type: {file.content_type}")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        return service.process_video_bytes(data, filename=file.filename or "upload.mp4")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
