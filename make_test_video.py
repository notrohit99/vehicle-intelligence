import cv2

src = r"services\tracking\assets\video\vehicle-counting.mp4"
out = r"data\test10s.mp4"

cap = cv2.VideoCapture(src)

fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    out,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

max_frames = int(fps * 10)
count = 0

while count < max_frames:
    ok, frame = cap.read()

    if not ok:
        break

    writer.write(frame)
    count += 1

writer.release()
cap.release()

print(f"Created: {out}")