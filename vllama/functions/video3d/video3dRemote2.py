import json
import time
import shutil
import tempfile
import subprocess
from pathlib import Path


# ---------------------------------------------------------
# Timeline Logger
# ---------------------------------------------------------
START_TIME = time.time()

def log(msg: str):
    elapsed = time.time() - START_TIME
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    print(f"[{mins:02d}:{secs:02d}] {msg}")


# ---------------------------------------------------------
# Main Function
# ---------------------------------------------------------
def run_kaggle_video_to_3d(
    video_path: str,
    output_dir: str,
    frame_interval: int = 15,
):
    """
    Kaggle-based Pi3 Video → 3D runner with timeline.

    - Uploads video as Kaggle dataset
    - Runs Pi3 on Kaggle GPU
    - Downloads final PLY
    - Saves locally with timing info
    """

    log("Initializing job")

    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    # ---------------------------------------------------------
    # Kaggle user
    # ---------------------------------------------------------
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    with open(kaggle_json) as f:
        username = json.load(f)["username"]

    # ---------------------------------------------------------
    # Create dataset
    # ---------------------------------------------------------
    dataset_slug = f"pi3-video-{int(time.time())}"
    dataset_dir = Path(tempfile.mkdtemp())

    log("Creating Kaggle dataset")
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

    subprocess.run(
        ["kaggle", "datasets", "create", "-p", str(dataset_dir)],
        check=True,
    )

    shutil.rmtree(dataset_dir, ignore_errors=True)
    log("Dataset uploaded")

    # ---------------------------------------------------------
    # Create kernel
    # ---------------------------------------------------------
    kernel_slug = f"pi3-kernel-{int(time.time())}"
    kernel_dir = Path(tempfile.mkdtemp())

    log("Preparing Kaggle kernel")

    script_code = f"""
import subprocess
from pathlib import Path

print("=" * 70)
print("PI3 OFFICIAL PIPELINE")
print("=" * 70)

subprocess.run([
    "pip", "install", "-q", "--no-cache-dir",
    "torch", "torchvision",
    "--index-url", "https://download.pytorch.org/whl/cu118"
], check=True)

subprocess.run(
    ["git", "clone", "https://github.com/yyfz/Pi3.git"],
    cwd="/kaggle/working",
    check=True
)

subprocess.run(
    ["pip", "install", "-q", "--no-cache-dir", "-r", "requirements.txt"],
    cwd="/kaggle/working/Pi3",
    check=True
)

subprocess.run(
    ["pip", "install", "-q", "--no-cache-dir", "plyfile"],
    check=True
)

dataset_dir = Path("/kaggle/input/{dataset_slug}")
video_path = list(dataset_dir.glob("*"))[0]

output_dir = Path("/kaggle/working/output")
output_dir.mkdir(exist_ok=True)

output_ply = output_dir / "result.ply"

subprocess.run(
    [
        "python", "example.py",
        "--data_path", str(video_path),
        "--save_path", str(output_ply),
        "--interval", "{frame_interval}",
        "--device", "cuda"
    ],
    cwd="/kaggle/working/Pi3",
    check=True
)

print("DONE:", output_ply)
"""

    (kernel_dir / "kernel.py").write_text(script_code)

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

    subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(kernel_dir)],
        check=True,
    )

    log("Kernel pushed, waiting for GPU execution")

    # ---------------------------------------------------------
    # Wait for completion
    # ---------------------------------------------------------
    kernel_ref = f"{username}/{kernel_slug}"

    while True:
        time.sleep(20)
        status = subprocess.run(
            ["kaggle", "kernels", "status", kernel_ref],
            capture_output=True,
            text=True,
        ).stdout.lower()

        if "complete" in status:
            log("Kernel completed (3D model generated)")
            break

        if "error" in status or "failed" in status:
            raise RuntimeError(
                f"Kernel failed: https://www.kaggle.com/code/{kernel_ref}"
            )

    # ---------------------------------------------------------
    # Download output
    # ---------------------------------------------------------
    log("Syncing Kaggle outputs (may take 1–2 minutes)")

    tmp_download_dir = output_dir / "_kaggle_tmp"
    tmp_download_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "kaggle",
            "kernels",
            "output",
            kernel_ref,
            "-p",
            str(tmp_download_dir),
            "--force",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    ply_source = tmp_download_dir / "output" / "result.ply"
    final_ply = output_dir / "result.ply"

    if not ply_source.exists():
        raise FileNotFoundError("result.ply not found in kernel output")

    shutil.move(str(ply_source), final_ply)
    shutil.rmtree(tmp_download_dir, ignore_errors=True)

    log(f"3D model saved locally → {final_ply}")

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------
    subprocess.run(
        ["kaggle", "datasets", "delete", "-d", f"{username}/{dataset_slug}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    shutil.rmtree(kernel_dir, ignore_errors=True)

    # ---------------------------------------------------------
    # Final return
    # ---------------------------------------------------------
    total_time = round(time.time() - START_TIME, 2)

    return {
        "path": str(final_ply.resolve()),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_seconds": total_time,
    }
