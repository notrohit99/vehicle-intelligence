"""FastAPI application exposing the ANPR pipeline as a REST service."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from anpr.ocr import PlateOCR
from anpr.schemas import ANPRResponse, OCRReadingSchema, OCRResponse
from anpr.service import ANPRService

service: ANPRService | None = None
ocr_service: PlateOCR | None = None

IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp", "image/bmp"}
VIDEO_TYPES = {"video/mp4", "video/avi", "video/x-msvideo", "video/quicktime", "video/x-matroska"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service, ocr_service
    service = ANPRService()
    ocr_service = PlateOCR()
    yield


app = FastAPI(
    title="ANPR Service",
    description="Automatic Number Plate Recognition using YOLOv8 and PaddleOCR",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/anpr/image", response_model=ANPRResponse)
async def detect_plates_in_image(file: UploadFile = File(...)) -> ANPRResponse:
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if file.content_type and file.content_type not in IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {file.content_type}. Use JPEG or PNG.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        return service.process_image_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/anpr/video", response_model=ANPRResponse)
async def detect_plates_in_video(file: UploadFile = File(...)) -> ANPRResponse:
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if file.content_type and file.content_type not in VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video type: {file.content_type}. Use MP4 or AVI.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        return service.process_video_bytes(data, filename=file.filename or "upload.mp4")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/anpr/ocr", response_model=OCRResponse)
async def ocr_vehicle_crop(
    file: UploadFile = File(...),
    offset_x: int = 0,
    offset_y: int = 0,
) -> OCRResponse:
    """OCR-only endpoint for a vehicle crop (used by the integration layer)."""
    if ocr_service is None:
        raise HTTPException(status_code=503, detail="OCR service not initialized")

    if file.content_type and file.content_type not in IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {file.content_type}. Use JPEG or PNG.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        readings = ocr_service.read_image_bytes(data, offset_x=offset_x, offset_y=offset_y)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return OCRResponse(
        readings=[
            OCRReadingSchema(
                plate_number=r.plate_number,
                confidence=r.confidence,
                bbox=r.bbox,
            )
            for r in readings
        ]
    )
