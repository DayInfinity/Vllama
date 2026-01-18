import os
import json
import time
import shutil
import tempfile
import subprocess
from pathlib import Path


def run_kaggle_video_to_3d(
    video_path: str, 
    output_dir: str, 
    frame_interval: int = 15,
    max_frames_per_batch: int = 25,
    total_frames_limit: int = 100
):
    """
    Runs Pi3 Video-to-3D pipeline on Kaggle GPU with batch processing for large videos.
    
    This version handles large room videos (100MB+, 60+ seconds) by:
    1. Uploading video as Kaggle dataset
    2. Processing frames in multiple batches to avoid OOM
    3. Merging point clouds from all batches
    4. Producing final high-quality 3D model
    
    Perfect for detailed room scans!

    Args:
        video_path: Path to input video file (.mp4, .avi, etc.)
        output_dir: Directory to save output PLY point cloud
        frame_interval: Frame sampling interval (default: 15 for long videos)
        max_frames_per_batch: Max frames to process at once (default: 25)
        total_frames_limit: Max total frames to extract (default: 100)
    
    Returns:
        Path to the generated PLY file
    """

    # ---------------------------------------------------------
    # Resolve and validate paths
    # ---------------------------------------------------------
    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir).resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print("=" * 70)
    print("LARGE VIDEO PROCESSING MODE")
    print("=" * 70)
    print(f"Input video: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Frame interval: {frame_interval}")
    print(f"Max frames per batch: {max_frames_per_batch}")
    print(f"Total frames limit: {total_frames_limit}")

    video_size_mb = video_path.stat().st_size / (1024 * 1024)
    print(f"Video size: {video_size_mb:.2f} MB")

    if video_size_mb > 500:
        print("⚠️  WARNING: Video is very large (>500MB)")
        print("   Consider compressing or trimming the video first.")

    # ---------------------------------------------------------
    # Kaggle setup
    # ---------------------------------------------------------
    subprocess.run(["kaggle", "--version"], check=True)

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    with open(kaggle_json, encoding="utf-8") as f:
        username = json.load(f)["username"]

    # ---------------------------------------------------------
    # Create dataset with video
    # ---------------------------------------------------------
    dataset_slug = f"video-input-{int(time.time())}"
    dataset_dir = Path(tempfile.mkdtemp(prefix="kaggle_dataset_"))
    
    print(f"\n{'=' * 70}")
    print(f"Creating Kaggle dataset: {dataset_slug}")
    print(f"{'=' * 70}")
    
    # Copy video to dataset directory
    video_filename = "input_video" + video_path.suffix
    print("Copying video to dataset directory...")
    shutil.copy2(video_path, dataset_dir / video_filename)
    
    # Create dataset metadata
    dataset_metadata = {
        "title": dataset_slug,
        "id": f"{username}/{dataset_slug}",
        "licenses": [{"name": "CC0-1.0"}]
    }
    
    with open(dataset_dir / "dataset-metadata.json", "w") as f:
        json.dump(dataset_metadata, f, indent=2)
    
    # Upload dataset
    print("Uploading video to Kaggle (this may take a few minutes for large files)...")
    result = subprocess.run(
        ["kaggle", "datasets", "create", "-p", str(dataset_dir)],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("❌ Dataset upload failed!")
        print(result.stderr if result.stderr else result.stdout)
        raise RuntimeError("Failed to upload dataset to Kaggle")
    
    print(f"✅ Dataset uploaded: {username}/{dataset_slug}")
    
    # Clean up dataset temp dir
    shutil.rmtree(dataset_dir, ignore_errors=True)

    # ---------------------------------------------------------
    # Create kernel with batch processing
    # ---------------------------------------------------------
    kernel_dir = Path(tempfile.mkdtemp(prefix="kaggle_pi3_kernel_"))
    kernel_slug = f"video-to-3d-large-{int(time.time())}"

    print(f"\n{'=' * 70}")
    print(f"Creating batch processing kernel: {kernel_slug}")
    print(f"{'=' * 70}")

    try:
        # -----------------------------------------------------
        # Kaggle kernel script - BATCH PROCESSING VERSION
        # -----------------------------------------------------
        script_code = f"""
import subprocess
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

print("=" * 70)
print("BATCH PROCESSING MODE - LARGE VIDEO")
print("=" * 70)

print("\\nInstalling dependencies...")
subprocess.run(["pip", "install", "-q", "--no-cache-dir", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu118"], check=True)
subprocess.run(["pip", "install", "-q", "--no-cache-dir", "opencv-python", "Pillow", "numpy", "huggingface_hub", "safetensors"], check=True)
subprocess.run(["git", "clone", "-q", "https://github.com/yyfz/Pi3.git"], cwd="/kaggle/working", check=True)

import sys
sys.path.insert(0, "/kaggle/working/Pi3")

import torch
import cv2
import numpy as np
from PIL import Image
import gc

# Try Pi3X first, fallback to Pi3
try:
    from pi3.models.pi3x import Pi3X as Pi3Model
    model_name = "yyfz233/Pi3X"
    use_pi3x = True
    print("Using Pi3X (batch processing enabled)")
except:
    from pi3.models.pi3 import Pi3 as Pi3Model
    model_name = "yyfz233/Pi3"
    use_pi3x = False
    print("Using Pi3 (batch processing enabled)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {{device}}")
if torch.cuda.is_available():
    print(f"GPU: {{torch.cuda.get_device_name(0)}}")
    print(f"Total VRAM: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}} GB")

# Find video
VIDEO_PATH = list(Path("/kaggle/input/{dataset_slug}").glob("*"))[0]
print(f"\\nFound video: {{VIDEO_PATH}}")

OUTPUT_DIR = Path("/kaggle/working/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Extract frames with intelligent sampling
print("\\nExtracting frames...")
cap = cv2.VideoCapture(str(VIDEO_PATH))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration = total_frames / fps if fps > 0 else 0

print(f"Video info:")
print(f"  Total frames: {{total_frames}}")
print(f"  FPS: {{fps:.2f}}")
print(f"  Duration: {{duration:.1f}} seconds")
print(f"  Frame interval: {frame_interval}")

# Extract frames
frames = []
frame_idx = 0
extracted_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_idx % {frame_interval} == 0:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        extracted_count += 1
        
        # Limit total frames
        if extracted_count >= {total_frames_limit}:
            print(f"  Reached frame limit: {total_frames_limit}")
            break
    
    frame_idx += 1

cap.release()
print(f"Extracted {{len(frames)}} frames")

if len(frames) == 0:
    raise ValueError("No frames extracted from video")

# Prepare size based on model
if use_pi3x:
    target_size = (518, 518)  # Multiple of 14
else:
    target_size = (512, 512)

print(f"\\nTarget size: {{target_size}}")

# Load model once
print("\\nLoading Pi3 model...")
model = Pi3Model.from_pretrained(model_name).to(device).eval()
print("Model loaded successfully")

# Batch processing setup
max_batch_size = {max_frames_per_batch}
num_frames = len(frames)
num_batches = (num_frames + max_batch_size - 1) // max_batch_size

print(f"\\nBatch processing setup:")
print(f"  Total frames: {{num_frames}}")
print(f"  Frames per batch: {{max_batch_size}}")
print(f"  Number of batches: {{num_batches}}")

# Storage for all results
all_points = []
all_colors = []
all_confidences = []

# Process each batch
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

for batch_idx in range(num_batches):
    start_idx = batch_idx * max_batch_size
    end_idx = min(start_idx + max_batch_size, num_frames)
    batch_frames = frames[start_idx:end_idx]
    
    print(f"\\n{'=' * 70}")
    print(f"Processing batch {{batch_idx + 1}}/{{num_batches}}")
    print(f"Frames {{start_idx}}-{{end_idx - 1}} ({{len(batch_frames)}} frames)")
    print(f"{'=' * 70}")
    
    # Prepare batch
    def prep_batch(frame_list):
        processed = []
        for frame in frame_list:
            pil_img = Image.fromarray(frame).resize(target_size, Image.LANCZOS)
            frame_np = np.array(pil_img).astype(np.float32) / 255.0
            frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1)
            processed.append(frame_tensor)
        return torch.stack(processed)
    
    imgs = prep_batch(batch_frames).to(device)
    print(f"Image tensor: {{imgs.shape}}")
    
    # Run inference
    print("Running Pi3 reconstruction...")
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            results = model(imgs[None])
    
    print("Extraction complete")
    
    # Extract points and colors
    points = results['points'][0]  # (N, H, W, 3)
    conf = torch.sigmoid(results['conf'][0])  # (N, H, W, 1)
    
    N, H, W, _ = points.shape
    points_flat = points.reshape(-1, 3).cpu().numpy()
    conf_flat = conf.reshape(-1).cpu().numpy()
    
    # Get colors
    colors_list = []
    for i in range(len(batch_frames)):
        colors_list.append(cv2.resize(batch_frames[i], (W, H)))
    colors = np.array(colors_list).reshape(-1, 3)
    
    print(f"Batch points: {{len(points_flat):,}}")
    
    # Store results
    all_points.append(points_flat)
    all_colors.append(colors)
    all_confidences.append(conf_flat)
    
    # Clear memory
    del imgs, results, points, conf, points_flat, conf_flat, colors
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"Batch {{batch_idx + 1}} complete, memory cleared")

# Merge all batches
print(f"\\n{'=' * 70}")
print("MERGING ALL BATCHES")
print(f"{'=' * 70}")

all_points = np.concatenate(all_points, axis=0)
all_colors = np.concatenate(all_colors, axis=0)
all_confidences = np.concatenate(all_confidences, axis=0)

print(f"Total points before filtering: {{len(all_points):,}}")

# Apply confidence filtering
conf_threshold = 0.5
mask = all_confidences > conf_threshold
points_filtered = all_points[mask]
colors_filtered = all_colors[mask]

print(f"Points after confidence filtering (>{{conf_threshold}}): {{len(points_filtered):,}}")

# Save final merged PLY
output_ply = OUTPUT_DIR / f"video_to_3d_{{timestamp}}.ply"
print(f"\\nSaving final point cloud to {{output_ply}}...")

vertices = np.zeros(len(points_filtered), 
                   dtype=[('x','f4'),('y','f4'),('z','f4'),
                          ('red','u1'),('green','u1'),('blue','u1')])
vertices['x'] = points_filtered[:, 0]
vertices['y'] = points_filtered[:, 1]
vertices['z'] = points_filtered[:, 2]
vertices['red'] = colors_filtered[:, 0]
vertices['green'] = colors_filtered[:, 1]
vertices['blue'] = colors_filtered[:, 2]

with open(output_ply, 'wb') as f:
    f.write(b"ply\\n")
    f.write(b"format binary_little_endian 1.0\\n")
    f.write(f"element vertex {{len(vertices)}}\\n".encode())
    f.write(b"property float x\\nproperty float y\\nproperty float z\\n")
    f.write(b"property uchar red\\nproperty uchar green\\nproperty uchar blue\\n")
    f.write(b"end_header\\n")
    vertices.tofile(f)

print(f"\\n{'=' * 70}")
print("SUCCESS!")
print(f"{'=' * 70}")
print(f"✅ Final 3D model: {{output_ply}}")
print(f"✅ Total vertices: {{len(vertices):,}}")
print(f"✅ File size: {{output_ply.stat().st_size / 1e6:.2f}} MB")
"""

        # Write kernel files
        (kernel_dir / "kernel.py").write_text(script_code, encoding="utf-8")

        # Kernel metadata with dataset reference
        metadata = {
            "id": f"{username}/{kernel_slug}",
            "title": kernel_slug,
            "code_file": "kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "dataset_sources": [f"{username}/{dataset_slug}"]
        }

        with open(kernel_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # -----------------------------------------------------
        # Push kernel
        # -----------------------------------------------------
        print("\nPushing kernel to Kaggle...")
        
        script_size_kb = (kernel_dir / "kernel.py").stat().st_size / 1024
        print(f"Script size: {script_size_kb:.2f} KB")
        
        result = subprocess.run(
            ["kaggle", "kernels", "push", "-p", str(kernel_dir)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print("❌ Kaggle push failed!")
            print(result.stderr if result.stderr else result.stdout)
            raise RuntimeError("Failed to push kernel to Kaggle")

        kernel_ref = f"{username}/{kernel_slug}"
        print(f"✅ Kernel pushed: {kernel_ref}")
        print(f"   View at: https://www.kaggle.com/code/{kernel_ref}")

        # -----------------------------------------------------
        # Wait for completion
        # -----------------------------------------------------
        print(f"\n{'=' * 70}")
        print("WAITING FOR BATCH PROCESSING TO COMPLETE")
        print("This will take longer for large videos (10-60 minutes)")
        print(f"{'=' * 70}")
        
        max_wait = 7200  # 2 hours max
        start = time.time()
        last_status = ""
        
        while True:
            elapsed = time.time() - start
            
            if elapsed > max_wait:
                raise TimeoutError("Kernel execution timeout (2 hours)")
            
            time.sleep(15)
            
            result = subprocess.run(
                ["kaggle", "kernels", "status", kernel_ref],
                capture_output=True,
                text=True
            )
            
            status = result.stdout.lower()

            if "complete" in status:
                print("\n✅ Kernel completed successfully!")
                break
            elif "error" in status or "failed" in status:
                print(f"\n❌ Kernel failed!")
                print(f"View logs: https://www.kaggle.com/code/{kernel_ref}")
                raise RuntimeError(f"Kernel failed. Check logs at: https://www.kaggle.com/code/{kernel_ref}")
            elif "running" in status:
                mins = elapsed / 60
                if mins - int(mins / 5) * 5 < 0.25:  # Print every 5 minutes
                    if status != last_status:
                        print(f"⏳ Processing... {mins:.1f} minutes elapsed")
                        last_status = status

        # -----------------------------------------------------
        # Download output
        # -----------------------------------------------------
        print(f"\n{'=' * 70}")
        print("DOWNLOADING FINAL 3D MODEL")
        print(f"{'=' * 70}")
        
        output_dir.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            ["kaggle", "kernels", "output", kernel_ref, "-p", str(output_dir)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0 and result.stderr:
            print("⚠️  Download warning (usually harmless):")
            print(result.stderr.strip())

        # Find PLY file
        ply_files = list(output_dir.rglob("video_to_3d_*.ply"))
        
        if not ply_files:
            raise FileNotFoundError(
                f"PLY output not found. Check logs: https://www.kaggle.com/code/{kernel_ref}"
            )

        final_ply = ply_files[0]
        
        print(f"\n{'=' * 70}")
        print("✅ SUCCESS! LARGE VIDEO PROCESSED")
        print(f"{'=' * 70}")
        print(f"✅ 3D Model: {final_ply}")
        print(f"   Size: {final_ply.stat().st_size / 1e6:.2f} MB")
        
        # Read vertex count from PLY
        with open(final_ply, 'rb') as f:
            content = f.read(500).decode('utf-8', errors='ignore')
            for line in content.split('\n'):
                if 'element vertex' in line:
                    vertex_count = int(line.split()[-1])
                    print(f"   Vertices: {vertex_count:,}")
                    break
        
        # Clean up dataset
        print(f"\n🧹 Cleaning up dataset: {username}/{dataset_slug}")
        subprocess.run(
            ["kaggle", "datasets", "delete", "-d", f"{username}/{dataset_slug}"],
            capture_output=True
        )
        
        return final_ply

    finally:
        shutil.rmtree(kernel_dir, ignore_errors=True)


# if __name__ == "__main__":
#     import sys
    
#     if len(sys.argv) < 2:
#         print("Usage: python script.py <video_path> [output_dir] [frame_interval]")
#         sys.exit(1)
    
#     video_path = sys.argv[1]
#     output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output"
#     frame_interval = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    
#     result = run_kaggle_video_to_3d_large(
#         video_path=video_path,
#         output_dir=output_dir,
#         frame_interval=frame_interval,
#         max_frames_per_batch=25,
#         total_frames_limit=100
#     )
    
#     print(f"\n🎉 Complete! 3D model: {result}")