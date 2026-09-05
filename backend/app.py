import httpx
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware


TRAFFIC_URL = "http://127.0.0.1:8003"
INTEGRATION_URL = "http://127.0.0.1:8002"


app = FastAPI(
    title="Vehicle Intelligence Backend",
    description="Central API for ANPR, tracking and traffic services",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load blacklist from file
BLACKLIST_PATH = Path(__file__).parent / "blacklist.json"
if BLACKLIST_PATH.is_file():
    with open(BLACKLIST_PATH, "r", encoding="utf-8") as f:
        BLACKLIST = json.load(f)
else:
    BLACKLIST = []


@app.get("/blacklist")
async def get_blacklist():
    """Return the list of blacklisted plate numbers."""
    return {"blacklist": BLACKLIST}

# Endpoint to add a plate to blacklist
@app.post("/blacklist")
async def add_blacklist_plate(data: dict):
    """Add a new plate number to the blacklist.
    Expected JSON: {"plate": "XYZ123"}
    """
    plate = data.get("plate")
    if not plate:
        raise HTTPException(status_code=400, detail="Plate number is required")
    if plate not in BLACKLIST:
        BLACKLIST.append(plate)
        # Persist to file
        try:
            with open(BLACKLIST_PATH, "w", encoding="utf-8") as f:
                json.dump(BLACKLIST, f, indent=2)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return {"blacklist": BLACKLIST}


@app.get("/health")
async def health():

    traffic_ok = False
    integration_ok = False

    async with httpx.AsyncClient(timeout=5) as client:

        try:
            response = await client.get(
                f"{TRAFFIC_URL}/health"
            )
            traffic_ok = response.status_code == 200
        except Exception:
            pass

        try:
            response = await client.get(
                f"{INTEGRATION_URL}/health"
            )
            integration_ok = response.status_code == 200
        except Exception:
            pass

    return {
        "status": (
            "ok"
            if traffic_ok and integration_ok
            else "degraded"
        ),
        "backend": True,
        "traffic": traffic_ok,
        "integration": integration_ok,
    }


@app.get("/traffic/predict")
async def traffic_predict():

    try:

        async with httpx.AsyncClient(timeout=30) as client:

            response = await client.post(
                f"{TRAFFIC_URL}/predict"
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail="Traffic service failed",
            )

        return response.json()

    except httpx.RequestError as exc:

        raise HTTPException(
            status_code=502,
            detail=f"Traffic service unavailable: {exc}",
        )


@app.get("/")
async def root():

    return {
        "service": "Vehicle Intelligence Backend",
        "status": "running",
    }


@app.post("/vehicles/process-video")
@app.post("/v1/process/video")
async def process_video(
    file: UploadFile = File(...),
    camera_id: str = Form("CAM_01"),
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No video file provided",
        )

    video_data = await file.read()

    if not video_data:
        raise HTTPException(
            status_code=400,
            detail="Empty video file",
        )

    try:

        async with httpx.AsyncClient(timeout=300) as client:

            response = await client.post(
                f"{INTEGRATION_URL}/v1/process/video",
                files={
                    "file": (
                        file.filename,
                        video_data,
                        file.content_type or "video/mp4",
                    )
                },
                data={
                    "camera_id": camera_id,
                },
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=response.text,
            )

        # Append blacklist flag to each vehicle
        payload = response.json()
        for vehicle in payload.get('vehicles', []):
            plate = vehicle.get('plate_number')
            vehicle['blacklisted'] = plate in BLACKLIST if plate else False
        return payload

    except httpx.RequestError as exc:

        raise HTTPException(
            status_code=502,
            detail=f"Integration service unavailable: {exc}",
        )