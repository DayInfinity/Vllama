import os
import json
import time
import shutil
import tempfile
import subprocess
import base64
from pathlib import Path


def run_kaggle_image_to_3d(image_path: str, output_dir: str):
    """
    Runs Apple's SHARP Image-to-3D Gaussian pipeline on Kaggle GPU
    by embedding the input image directly into the kernel.

    This version is fully hardened against:
    - Kaggle path issues
    - float64 / float32 mismatches
    - Kaggle CLI non-zero output exit codes
    """

    # ---------------------------------------------------------
    # Resolve and validate paths
    # ---------------------------------------------------------
    image_path = Path(image_path).resolve()
    output_dir = Path(output_dir).resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print("Input image:", image_path)
    print("Output directory:", output_dir)

    # ---------------------------------------------------------
    # Encode image as Base64
    # ---------------------------------------------------------
    print("Encoding image to Base64...")
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    # ---------------------------------------------------------
    # Kaggle setup
    # ---------------------------------------------------------
    subprocess.run(["kaggle", "--version"], check=True)

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    with open(kaggle_json, encoding="utf-8") as f:
        username = json.load(f)["username"]

    # ---------------------------------------------------------
    # Create temporary kernel directory
    # ---------------------------------------------------------
    kernel_dir = Path(tempfile.mkdtemp(prefix="kaggle_3dgs_"))
    kernel_slug = f"image-to-3d-{int(time.time())}"

    print("Created temp kernel dir:", kernel_dir)
    print("Kernel slug:", kernel_slug)

    try:
        # -----------------------------------------------------
        # Kaggle kernel script (FLOAT-SAFE & STABLE)
        # -----------------------------------------------------
        script_code = f"""
import base64
import subprocess
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

print("Installing SHARP...")
subprocess.run(
    ["pip", "install", "--no-cache-dir", "git+https://github.com/apple/ml-sharp.git"],
    check=True
)

import torch
import torch.nn.functional as F
from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.utils.gaussians import save_ply, unproject_gaussians

# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------------------------
# Restore image
# -------------------------------------------------
IMAGE_PATH = Path("/kaggle/working/input_image.jpg")
IMAGE_PATH.write_bytes(base64.b64decode(\"\"\"{image_b64}\"\"\"))

print("Image restored:", IMAGE_PATH)

# -------------------------------------------------
# Output
# -------------------------------------------------
OUTPUT_DIR = Path("/kaggle/working/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_ply = OUTPUT_DIR / f"image_to_3d_{{timestamp}}.ply"

# -------------------------------------------------
# Load image
# -------------------------------------------------
image, _, f_px = io.load_rgb(IMAGE_PATH)
h, w = image.shape[:2]

# -------------------------------------------------
# Load model
# -------------------------------------------------
state_dict = torch.hub.load_state_dict_from_url(
    "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt",
    progress=True
)

predictor = create_predictor(PredictorParams())
predictor.load_state_dict(state_dict)
predictor.eval().to(device)

# -------------------------------------------------
# Preprocess (STRICT float32)
# -------------------------------------------------
image_pt = (
    torch.from_numpy(image)
    .permute(2, 0, 1)
    .contiguous()
    .to(device)
    .float()
    / 255.0
)

image_resized = F.interpolate(
    image_pt.unsqueeze(0),
    size=(1536, 1536),
    mode="bilinear",
    align_corners=True
).float()

disparity_factor = torch.tensor(
    [float(f_px / w)],
    device=device,
    dtype=torch.float32
)

# -------------------------------------------------
# Predict
# -------------------------------------------------
print("Running SHARP inference...")
gaussians_ndc = predictor(image_resized, disparity_factor)

# -------------------------------------------------
# Unproject (STRICT float32)
# -------------------------------------------------
intrinsics = torch.tensor(
    [[f_px, 0.0, w / 2.0, 0.0],
     [0.0, f_px, h / 2.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]],
    device=device,
    dtype=torch.float32
)

intrinsics_resized = intrinsics.clone()
intrinsics_resized[0] *= float(1536 / w)
intrinsics_resized[1] *= float(1536 / h)

gaussians = unproject_gaussians(
    gaussians_ndc,
    torch.eye(4, device=device, dtype=torch.float32),
    intrinsics_resized,
    (1536, 1536)
)

# -------------------------------------------------
# Save
# -------------------------------------------------
save_ply(gaussians, f_px, (h, w), output_ply)
print("Saved PLY:", output_ply)
"""

        # Write kernel files
        (kernel_dir / "kernel.py").write_text(script_code, encoding="utf-8")

        metadata = {
            "id": f"{username}/{kernel_slug}",
            "title": kernel_slug,
            "code_file": "kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
        }

        with open(kernel_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # -----------------------------------------------------
        # Push kernel
        # -----------------------------------------------------
        print("Pushing Kaggle kernel...")
        subprocess.run(
            ["kaggle", "kernels", "push", "-p", str(kernel_dir)],
            check=True
        )

        kernel_ref = f"{username}/{kernel_slug}"
        print("Kernel ref:", kernel_ref)

        # -----------------------------------------------------
        # Wait for completion
        # -----------------------------------------------------
        print("Waiting for kernel to finish...")
        while True:
            time.sleep(6)
            status = subprocess.run(
                ["kaggle", "kernels", "status", kernel_ref],
                capture_output=True,
                text=True
            ).stdout.lower()

            if "complete" in status:
                break
            if "failed" in status or "error" in status:
                raise RuntimeError(
                    f"Kaggle kernel failed: https://www.kaggle.com/code/{kernel_ref}"
                )

        # -----------------------------------------------------
        # Download output (SAFE MODE)
        # -----------------------------------------------------
        print("Downloading kernel outputs...")
        output_dir.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            ["kaggle", "kernels", "output", kernel_ref, "-p", str(output_dir)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("⚠ Kaggle CLI returned non-zero exit (harmless)")
            print(result.stderr.strip())

        ply_files = list(output_dir.rglob("image_to_3d_*.ply"))
        if not ply_files:
            raise FileNotFoundError("PLY output not found")

        print("✅ Final 3D model:", ply_files[0])
        return ply_files[0]

    finally:
        shutil.rmtree(kernel_dir, ignore_errors=True)
