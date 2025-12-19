import os
import time
import cv2
import torch
from ultralytics import YOLO

# Fix OpenMP issue (Windows)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def object_detection_video(
    video_path: str,
    model_id: str = "yolov8l.pt",
    output_dir: str = "."
):
    """
    Run YOLOv8 object detection + tracking on a video and save annotated output.
    """

    # -----------------------------
    # GPU / Device Detection
    # -----------------------------
    if torch.cuda.is_available():
        device = "cuda"
        print("✅ GPU:", torch.cuda.get_device_name(0))
        imgsz = 960
    elif torch.backends.mps.is_available():
        device = "mps"
        print("✅ Using Apple Silicon GPU")
        imgsz = 960
    else:
        device = "cpu"
        print("⚠️ Using CPU")
        imgsz = 640
        model_id = "yolov8s.pt"

    # -----------------------------
    # Load YOLO Model
    # -----------------------------
    model = YOLO(model_id)
    model.to(device)

    if device == "cuda":
        model.fuse()  # speedup

    # -----------------------------
    # Open Input Video
    # -----------------------------
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "❌ Cannot open input video"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        fps = 30.0  # safe fallback

    # -----------------------------
    # Output Video Setup
    # -----------------------------
    os.makedirs(output_dir, exist_ok=True)

    timestamp = int(time.time())
    output_path = f"{output_dir}/video_obj_detection_{timestamp}.avi"

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    assert out.isOpened(), "❌ VideoWriter failed to open"

    # -----------------------------
    # Frame Processing Loop
    # -----------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            source=frame,
            persist=True,
            imgsz=imgsz,
            conf=0.25,
            iou=0.5,
            tracker="botsort.yaml",
            device=device,
            verbose=False
        )

        annotated = results[0].plot()
        out.write(annotated)

    # -----------------------------
    # Cleanup
    # -----------------------------
    cap.release()
    out.release()

    print("✅ Detection complete. Video saved at:", output_path)
