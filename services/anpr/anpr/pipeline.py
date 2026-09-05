"""Core ANPR pipeline extracted from the original run.py logic."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ANPR_PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ANPR_PROJECT_DIR / "yolov8n.pt"

CONFIDENCE_THRESHOLD = 95.0  # percent, matches original: score * 100 > 95
YOLO_CONF = 0.8
YOLO_IOU = 0.3
VEHICLE_CLASS = 2


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class PlateDetection:
    plate_number: str
    confidence: float
    frame: int
    timestamp_sec: float
    bounding_box: BoundingBox
    vehicle_id: int | None = None
    vehicle_bounding_box: BoundingBox | None = None


@dataclass
class ANPRState:
    """Tracks OCR progress across video frames (replaces original ocr_flag list)."""

    processed_ids: set[int] = field(default_factory=set)
    ocr_flags: dict[int, bool] = field(default_factory=dict)


class ANPRPipeline:
    """Wraps YOLOv8 vehicle tracking and PaddleOCR plate recognition."""

    def __init__(
        self,
        yolo_weights: str | Path = DEFAULT_MODEL_PATH,
        ocr_lang: str = "en",
    ) -> None:
        self.model = YOLO(str(yolo_weights))
        self.ocr = PaddleOCR(use_angle_cls=True, lang=ocr_lang, show_log=False)

    def _run_detection(
        self,
        frame: np.ndarray,
        *,
        persist_tracking: bool = True,
    ):
        if persist_tracking:
            return self.model.track(
                frame,
                persist=True,
                conf=YOLO_CONF,
                classes=[VEHICLE_CLASS],
                iou=YOLO_IOU,
                verbose=False,
            )
        return self.model(
            frame,
            conf=YOLO_CONF,
            classes=[VEHICLE_CLASS],
            iou=YOLO_IOU,
            verbose=False,
        )

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        fps: float = 30.0,
        state: ANPRState | None = None,
        *,
        persist_tracking: bool = True,
    ) -> list[PlateDetection]:
        """Run detection + OCR on a single frame. Returns new plate detections."""
        results = self._run_detection(frame, persist_tracking=persist_tracking)
        return self._extract_detections(
            frame=frame,
            results=results,
            frame_index=frame_index,
            fps=fps,
            state=state,
        )

    def process_frame_with_plot(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        fps: float = 30.0,
        state: ANPRState | None = None,
        *,
        persist_tracking: bool = True,
    ) -> tuple[list[PlateDetection], np.ndarray]:
        """Single YOLO pass — returns detections and annotated frame (for CLI)."""
        results = self._run_detection(frame, persist_tracking=persist_tracking)
        detections = self._extract_detections(
            frame=frame,
            results=results,
            frame_index=frame_index,
            fps=fps,
            state=state,
        )
        return detections, results[0].plot()

    def _extract_detections(
        self,
        frame: np.ndarray,
        results,
        frame_index: int,
        fps: float,
        state: ANPRState | None,
    ) -> list[PlateDetection]:
        if state is None:
            state = ANPRState()

        detections: list[PlateDetection] = []
        timestamp_sec = frame_index / fps if fps > 0 else 0.0

        for result in results[0].boxes:
            track_id = int(result.id) if result.id is not None else None
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            vehicle_bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            car_roi = frame[y1:y2, x1:x2]

            if track_id is not None:
                if track_id not in state.processed_ids or not state.ocr_flags.get(track_id, False):
                    detections.extend(
                        self._ocr_vehicle_region(
                            car_roi=car_roi,
                            frame_index=frame_index,
                            timestamp_sec=timestamp_sec,
                            vehicle_id=track_id,
                            vehicle_bbox=vehicle_bbox,
                            state=state,
                        )
                    )
                    state.processed_ids.add(track_id)
            else:
                detections.extend(
                    self._ocr_vehicle_region(
                        car_roi=car_roi,
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        vehicle_id=None,
                        vehicle_bbox=vehicle_bbox,
                        state=None,
                    )
                )

        return detections

    def _ocr_vehicle_region(
        self,
        car_roi: np.ndarray,
        frame_index: int,
        timestamp_sec: float,
        vehicle_id: int | None,
        vehicle_bbox: BoundingBox,
        state: ANPRState | None,
    ) -> list[PlateDetection]:
        """OCR on a vehicle crop — same logic as original run.py."""
        if car_roi.size == 0:
            return []

        ocr_result = self.ocr.ocr(car_roi, cls=True)
        detections: list[PlateDetection] = []

        if ocr_result and isinstance(ocr_result[0], list):
            for line in ocr_result[0]:
                box = np.array(line[0], dtype=np.int32) + np.array(
                    [vehicle_bbox.x1, vehicle_bbox.y1]
                )
                text = line[1][0]
                score = float(line[1][1])

                if CONFIDENCE_THRESHOLD < (score * 100.0):
                    x_coords = box[:, 0]
                    y_coords = box[:, 1]
                    plate_bbox = BoundingBox(
                        x1=int(x_coords.min()),
                        y1=int(y_coords.min()),
                        x2=int(x_coords.max()),
                        y2=int(y_coords.max()),
                    )

                    if vehicle_id is not None and state is not None:
                        state.ocr_flags[vehicle_id] = True

                    detections.append(
                        PlateDetection(
                            plate_number=text,
                            confidence=score,
                            frame=frame_index,
                            timestamp_sec=round(timestamp_sec, 4),
                            bounding_box=plate_bbox,
                            vehicle_id=vehicle_id,
                            vehicle_bounding_box=vehicle_bbox,
                        )
                    )

        return detections

    def annotate_plates(self, frame: np.ndarray, detections: list[PlateDetection]) -> np.ndarray:
        """Overlay plate labels on a frame (for CLI video output)."""
        annotated = frame.copy()
        for det in detections:
            bb = det.bounding_box
            cv2.rectangle(annotated, (bb.x1, bb.y1), (bb.x2, bb.y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{det.plate_number} ({det.confidence:.2f})",
                (bb.x1, bb.y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return annotated
