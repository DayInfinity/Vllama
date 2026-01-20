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
    frame_interval: int = 10,
):
    """
    Runs Pi3 Video-to-3D pipeline on Kaggle GPU using OFFICIAL Pi3 inference.

    Kaggle flow:
    1. Upload video as Kaggle dataset
    2. Run Pi3 official inference inside Kaggle kernel
    3. Download result.ply
    """

    # ---------------------------------------------------------
    # Resolve paths
    # ---------------------------------------------------------
    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print("=" * 70)
    print("PI3 VIDEO → 3D (KAGGLE)")
    print("=" * 70)
    print(f"Input video  : {video_path}")
    print(f"Output dir   : {output_dir}")
    print(f"Frame interval: {frame_interval}")

    # ---------------------------------------------------------
    # Kaggle setup
    # ---------------------------------------------------------
    subprocess.run(["kaggle", "--version"], check=True)

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    with open(kaggle_json, encoding="utf-8") as f:
        username = json.load(f)["username"]

    # ---------------------------------------------------------
    # Create Kaggle dataset
    # ---------------------------------------------------------
    dataset_slug = f"pi3-video-{int(time.time())}"
    dataset_dir = Path(tempfile.mkdtemp(prefix="kaggle_dataset_"))

    shutil.copy2(video_path, dataset_dir / video_path.name)

    with open(dataset_dir / "dataset-metadata.json", "w") as f:
        json.dump(
            {
                "title": dataset_slug,
                "id": f"{username}/{dataset_slug}",
                "licenses": [{"name": "CC0-1.0"}],
            },
            f,
            indent=2,
        )

    print("Uploading dataset to Kaggle...")
    result = subprocess.run(
        ["kaggle", "datasets", "create", "-p", str(dataset_dir)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)

    shutil.rmtree(dataset_dir, ignore_errors=True)
    print(f"Dataset uploaded: {username}/{dataset_slug}")

    # ---------------------------------------------------------
    # Create kernel
    # ---------------------------------------------------------
    kernel_dir = Path(tempfile.mkdtemp(prefix="kaggle_pi3_kernel_"))
    kernel_slug = f"pi3-video-to-3d-{int(time.time())}"

    script_code = f"""
import subprocess
from pathlib import Path
import torch
import warnings

warnings.filterwarnings("ignore")

print("=" * 70)
print("PI3 OFFICIAL INFERENCE (KAGGLE)")
print("=" * 70)

# ---------------------------------------------------------
# Install dependencies
# ---------------------------------------------------------
subprocess.run(
    [
        "pip", "install", "--no-cache-dir",
        "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ],
    check=True
)

subprocess.run(
    ["pip", "install", "--no-cache-dir", "git+https://github.com/yyfz/Pi3.git"],
    check=True
)

subprocess.run(
    ["pip", "install", "--no-cache-dir", "plyfile", "safetensors"],
    check=True
)

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3

# ---------------------------------------------------------
# Input / Output
# ---------------------------------------------------------
VIDEO_PATH = next(Path("/kaggle/input").rglob("*.mp4"))
OUTPUT_DIR = Path("/kaggle/working/output")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PLY = OUTPUT_DIR / "result.ply"

print("Video:", VIDEO_PATH)

# ---------------------------------------------------------
# Device
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()

# ---------------------------------------------------------
# Load frames
# ---------------------------------------------------------
imgs = load_images_as_tensor(
    str(VIDEO_PATH),
    interval={frame_interval}
).to(device)

# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------
dtype = (
    torch.bfloat16
    if torch.cuda.get_device_capability()[0] >= 8
    else torch.float16
)

with torch.no_grad():
    with torch.amp.autocast("cuda", dtype=dtype):
        res = model(imgs[None])

# ---------------------------------------------------------
# Mask filtering
# ---------------------------------------------------------
masks = torch.sigmoid(res["conf"][..., 0]) > 0.1
non_edge = ~depth_edge(res["local_points"][..., 2], rtol=0.03)
masks = torch.logical_and(masks, non_edge)[0]

# ---------------------------------------------------------
# Save PLY
# ---------------------------------------------------------
write_ply(
    res["points"][0][masks].cpu(),
    imgs.permute(0, 2, 3, 1)[masks],
    OUTPUT_PLY
)

print("Saved:", OUTPUT_PLY)
"""

    (kernel_dir / "kernel.py").write_text(script_code, encoding="utf-8")

    with open(kernel_dir / "kernel-metadata.json", "w") as f:
        json.dump(
            {
                "id": f"{username}/{kernel_slug}",
                "title": kernel_slug,
                "code_file": "kernel.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": "true",
                "enable_gpu": "true",
                "enable_internet": "true",
                "dataset_sources": [f"{username}/{dataset_slug}"],
            },
            f,
            indent=2,
        )

    # ---------------------------------------------------------
    # Push kernel
    # ---------------------------------------------------------
    subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(kernel_dir)],
        check=True,
    )

    kernel_ref = f"{username}/{kernel_slug}"
    print("Kernel running:", kernel_ref)

    # ---------------------------------------------------------
    # Wait for completion
    # ---------------------------------------------------------
    while True:
        time.sleep(15)
        status = subprocess.run(
            ["kaggle", "kernels", "status", kernel_ref],
            capture_output=True,
            text=True,
        ).stdout.lower()

        if "complete" in status:
            break
        if "error" in status or "failed" in status:
            raise RuntimeError(f"Kernel failed: https://www.kaggle.com/code/{kernel_ref}")

    # ---------------------------------------------------------
    # Download output
    # ---------------------------------------------------------
    print("Downloading kernel output...")

    result = subprocess.run(
        ["kaggle", "kernels", "output", kernel_ref, "-p", str(output_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )

    # Kaggle CLI may error even if download succeeded (Windows issue)
    final_ply = output_dir / "output" / f"result_{int(time.time())}.ply"
    downloaded_ply = output_dir / "output" / "result.ply"


    if not downloaded_ply.exists():
        print("Kaggle CLI output:")
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("Download failed: downloaded_ply not found")
    
    shutil.move(downloaded_ply, final_ply)

    print("Download confirmed:", final_ply)
