"""FastAPI integration layer connecting tracking and ANPR services."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from clients.anpr_client import ANPRClient
from clients.tracking_client import TrackingClient
from orchestrator import VideoOrchestrator
from schemas import ProcessVideoResponse

TRACKING_URL = os.getenv("TRACKING_SERVICE_URL", "http://127.0.0.1:8001")
ANPR_URL = os.getenv("ANPR_SERVICE_URL", "http://127.0.0.1:8000")

orchestrator: VideoOrchestrator | None = None

VIDEO_TYPES = {"video/mp4", "video/avi", "video/x-msvideo", "video/quicktime", "video/x-matroska"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator
    orchestrator = VideoOrchestrator(
        tracking_client=TrackingClient(TRACKING_URL),
        anpr_client=ANPRClient(ANPR_URL),
    )
    yield


app = FastAPI(
    title="Vehicle Intelligence Integration",
    description="Combines ByteTrack vehicle tracking with ANPR OCR",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health() -> dict:
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    tracking_ok = await orchestrator.tracking_client.health()
    anpr_ok = await orchestrator.anpr_client.health()

    status = "ok" if tracking_ok and anpr_ok else "degraded"
    return {
        "status": status,
        "tracking_service": TRACKING_URL,
        "anpr_service": ANPR_URL,
        "tracking_ok": tracking_ok,
        "anpr_ok": anpr_ok,
    }


@app.post("/v1/process/video", response_model=ProcessVideoResponse)
async def process_video(
    file: UploadFile = File(...),
    camera_id: str = Form("CAM_01"),
) -> ProcessVideoResponse:
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

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
        return await orchestrator.process_video(
            video_bytes=data,
            camera_id=camera_id,
            filename=file.filename or "upload.mp4",
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Pipeline failed: {exc}") from exc
