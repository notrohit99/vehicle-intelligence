"""PaddleOCR-only engine for license plate reading on vehicle crops."""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
from paddleocr import PaddleOCR

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "50.0"))


@dataclass
class PlateReading:
    plate_number: str
    confidence: float
    bbox: list[int]  # x1, y1, x2, y2 in full-frame coordinates


class PlateOCR:
    """Runs PaddleOCR on a vehicle crop without YOLO detection."""

    def __init__(self, ocr_lang: str = "en", min_confidence: float = DEFAULT_CONFIDENCE_THRESHOLD) -> None:
        self.min_confidence = min_confidence
        self._ocr = PaddleOCR(use_angle_cls=True, lang=ocr_lang, show_log=False)

    def read_image_bytes(
        self,
        data: bytes,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> list[PlateReading]:
        arr = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image. Upload a valid JPEG or PNG file.")
        return self.read_image(image, offset_x=offset_x, offset_y=offset_y)

    def read_image(
        self,
        image: np.ndarray,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> list[PlateReading]:
        if image.size == 0:
            return []

        ocr_result = self._ocr.ocr(image, cls=True)
        readings: list[PlateReading] = []

        if ocr_result and isinstance(ocr_result[0], list):
            for line in ocr_result[0]:
                box = np.array(line[0], dtype=np.int32)
                text = line[1][0]
                score = float(line[1][1])

                if self.min_confidence <= (score * 100.0):
                    x_coords = box[:, 0]
                    y_coords = box[:, 1]
                    readings.append(
                        PlateReading(
                            plate_number=text,
                            confidence=score,
                            bbox=[
                                int(x_coords.min()) + offset_x,
                                int(y_coords.min()) + offset_y,
                                int(x_coords.max()) + offset_x,
                                int(y_coords.max()) + offset_y,
                            ],
                        )
                    )

        return readings
