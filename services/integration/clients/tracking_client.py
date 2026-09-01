"""HTTP client for the tracking service."""

from __future__ import annotations

import httpx


class TrackingClient:
    def __init__(self, base_url: str, timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def health(self) -> bool:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.base_url}/health")
            return response.status_code == 200

    async def track_video(self, video_bytes: bytes, filename: str = "upload.mp4") -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/track/video",
                files={"file": (filename, video_bytes, "video/mp4")},
            )
            response.raise_for_status()
            return response.json()
