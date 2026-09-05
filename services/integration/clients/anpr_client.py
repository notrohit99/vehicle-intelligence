"""HTTP client for the ANPR OCR service."""

from __future__ import annotations

import httpx


class ANPRClient:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def health(self) -> bool:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.base_url}/health")
            return response.status_code == 200

    async def ocr_crop(
        self,
        image_bytes: bytes,
        offset_x: int = 0,
        offset_y: int = 0,
        filename: str = "crop.jpg",
    ) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/anpr/ocr",
                files={"file": (filename, image_bytes, "image/jpeg")},
                params={"offset_x": offset_x, "offset_y": offset_y},
            )
            response.raise_for_status()
            return response.json()


ANPR_URL = "http://localhost:8000"
TRACKING_URL = "http://localhost:8001"
