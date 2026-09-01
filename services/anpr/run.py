"""Original CLI entry point — unchanged behavior, now uses the shared pipeline."""

import csv
import time

import cv2
import numpy as np

from anpr.pipeline import ANPRPipeline, ANPRState

VIDEO_PATH = "demoVideo.mp4"
CSV_FILE = "Car-List.csv"
OUTPUT_VIDEO = "demoVideo_out.mp4"


def main() -> None:
    pipeline = ANPRPipeline()
    cap = cv2.VideoCapture(VIDEO_PATH)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    video_write = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"MJPG"),
        30,
        size,
    )

    state = ANPRState()
    processing_times: list[float] = []

    print("Program Started")
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Car ID", "License Plate Number"])
        writer.writeheader()

        start = time.time()
        frame_index = 0

        while cap.isOpened():
            start_frame_time = time.time()
            success, frame = cap.read()
            if not success:
                break

            detections, annotated_frame = pipeline.process_frame_with_plot(
                frame,
                frame_index=frame_index,
                fps=30.0,
                state=state,
                persist_tracking=True,
            )

            for det in detections:
                if det.vehicle_id is not None:
                    writer.writerow(
                        {"Car ID": det.vehicle_id, "License Plate Number": det.plate_number}
                    )
                    bb = det.bounding_box
                    text_position = (bb.x1, bb.y1 - 10)
                    cv2.putText(
                        annotated_frame,
                        f"{det.plate_number} ({det.confidence:.2f})",
                        text_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

            end_frame_time = time.time()
            processing_times.append(end_frame_time - start_frame_time)

            video_write.write(annotated_frame)
            cv2.imshow("ANPR using YOLOv8 + PaddleOCR", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_index += 1

    print("Total time: ", time.time() - start)

    if processing_times:
        avg_processing_time = np.mean(processing_times)
        print(f"Average processing time per frame: {avg_processing_time:.4f} seconds")

    print("Program Stopped")

    cap.release()
    video_write.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
