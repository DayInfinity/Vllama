# Object Detection & 3D Commands

---

## `vllama detect_image` — Detect Objects in an Image { #detect-image }

Run YOLO object detection on a single image.

### Syntax

```bash
vllama detect_image [--path <image>] [--url <url>] [--model <yolo>] [--output_dir <dir>]
```

### Parameters

| Parameter | Short | Default | Description |
|---|---|---|---|
| `--path` | | | Path to a local image file |
| `--url` | | | URL of an image to fetch and detect |
| `--model` | `-m` | `yolov8n.pt` | YOLO model variant |
| `--output_dir` | `-o` | current dir | Where to save the annotated output |

### Examples

```bash
# Local file
vllama detect_image --path photo.jpg

# From URL
vllama detect_image --url https://ultralytics.com/images/bus.jpg

# Use a larger, more accurate model
vllama detect_image --path photo.jpg --model yolov8s.pt

# Custom output directory
vllama detect_image --path photo.jpg -o ./detections
```

### YOLO Model Variants

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| `yolov8n.pt` | Nano | Fastest | Lower |
| `yolov8s.pt` | Small | Fast | Medium |
| `yolov8m.pt` | Medium | Moderate | Good |
| `yolov8l.pt` | Large | Slower | High |
| `yolov8x.pt` | XLarge | Slowest | Highest |

---

## `vllama detect_video` — Detect Objects in a Video { #detect-video }

Run YOLO object detection frame-by-frame on a video file.

### Syntax

```bash
vllama detect_video --path <video> [--model <yolo>] [--output_dir <dir>]
```

### Example

```bash
vllama detect_video --path video.mp4 -o ./outputs
vllama detect_video --path clip.mp4 --model yolov8s.pt
```

Output is an annotated video file saved in the output directory.

---

## `vllama image3d` — Image to 3D Model { #image3d }

Generate a 3D `.ply` model from a single image using Kaggle GPU.

### Syntax

```bash
vllama image3d [--path <image>] [--url <url>] [--service kaggle] [--output_dir <dir>]
```

### Example

```bash
vllama image3d --path object_photo.jpg --service kaggle -o ./3d_outputs
vllama image3d --url https://example.com/object.jpg --service kaggle
```

!!! info "Kaggle required"
    `image3d` runs on Kaggle's GPU. Make sure you are logged in with `vllama login --service kaggle`.

---

## `vllama video3d` — Video to 3D Model { #video3d }

Generate a 3D `.ply` model from a video file using Kaggle GPU.

### Syntax

```bash
vllama video3d --path <video> [--service kaggle] [--output_dir <dir>] [--frame_interval <n>]
```

### Parameters

| Parameter | Short | Default | Description |
|---|---|---|---|
| `--path` | | required | Path to input video |
| `--service` | `-s` | | Cloud service (`kaggle`) |
| `--output_dir` | `-o` | current dir | Where to save output |
| `--frame_interval` | `-f` | `5` | Process every Nth frame |

### Example

```bash
vllama video3d --path object_scan.mp4 --service kaggle -o ./3d_outputs
vllama video3d --path scan.mp4 --service kaggle -f 10
```

---

## View the Output

After generating a `.ply` file, use [`vllama view3d`](viewer.md) to open it in an interactive 3D viewer.

```bash
vllama view3d --path output.ply
```
