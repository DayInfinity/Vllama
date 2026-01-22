import os
import subprocess
import warnings
import time
from pathlib import Path

import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Global config
# ---------------------------------------------------------
PATCH = 14

# CUDA memory stability (must be set early)
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128"
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def snap_to_patch(x, patch=PATCH):
    return (x // patch) * patch


def get_system_config():
    """
    Decide safe defaults based on available hardware.
    """
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1024**3

        if vram_gb <= 4:      # very low-end GPU
            return dict(device="cuda", frame_interval=15, max_frames=40, amp=True)
        elif vram_gb <= 8:    # T4 / RTX 2060
            return dict(device="cuda", frame_interval=10, max_frames=80, amp=True)
        else:                # 16GB+
            return dict(device="cuda", frame_interval=6, max_frames=120, amp=True)

    return dict(device="cpu", frame_interval=20, max_frames=25, amp=False)


def pip_install(pkgs):
    subprocess.run(
        ["pip", "install", "--no-cache-dir"] + pkgs,
        check=True
    )


# ---------------------------------------------------------
# Main function
# ---------------------------------------------------------
def run_local_video_to_3d(video_path: str, output_dir: str):
    cfg = get_system_config()

    device_name    = cfg["device"]
    frame_interval = cfg["frame_interval"]
    max_frames     = cfg["max_frames"]
    use_amp        = cfg["amp"]

    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    # ---------------------------------------------------------
    # Logging
    # ---------------------------------------------------------
    print("=" * 70)
    print("PI3 VIDEO → 3D (STABLE + PORTABLE)")
    print("=" * 70)
    print("Video          :", video_path)
    print("Output dir     :", output_dir)

    if device_name == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"Device         : CUDA ({props.name}, {props.total_memory/1024**3:.1f} GB)")
    else:
        print("Device         : CPU")

    print("Frame interval :", frame_interval)
    print("Max frames     :", max_frames)
    print("AMP enabled    :", use_amp)
    print("=" * 70)

    # ---------------------------------------------------------
    # Dependencies
    # ---------------------------------------------------------
    try:
        import pi3  # noqa
    except ImportError:
        pip_install(["git+https://github.com/yyfz/Pi3.git"])

    try:
        import plyfile  # noqa
    except ImportError:
        pip_install(["plyfile", "safetensors"])

    from pi3.utils.basic import load_images_as_tensor, write_ply
    from pi3.utils.geometry import depth_edge
    from pi3.models.pi3 import Pi3

    device = torch.device(device_name)

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------
    print("🔹 Loading Pi3 model...")
    model = Pi3.from_pretrained("yyfz233/Pi3").to(device)
    model.eval()
    torch.set_grad_enabled(False)

    # ---------------------------------------------------------
    # Load frames (CPU first)
    # ---------------------------------------------------------
    print("🔹 Loading frames from video...")
    imgs = load_images_as_tensor(
        str(video_path),
        interval=frame_interval,
    )

    # Evenly sample frames if too many
    if imgs.shape[0] > max_frames:
        idx = torch.linspace(
            0, imgs.shape[0] - 1, max_frames
        ).long()
        imgs = imgs[idx]

    # ---------------------------------------------------------
    # Patch-aligned resize
    # ---------------------------------------------------------
    _, _, H, W = imgs.shape
    new_h = snap_to_patch(H)
    new_w = snap_to_patch(W)

    if (new_h, new_w) != (H, W):
        print(f"🔧 Resizing {H}x{W} → {new_h}x{new_w}")
        imgs = F.interpolate(
            imgs,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )

    print(f"✅ Frames loaded: {imgs.shape[0]} @ {new_h}x{new_w}")

    # ---------------------------------------------------------
    # Move to device
    # ---------------------------------------------------------
    imgs = imgs.to(device, non_blocking=True)

    # ---------------------------------------------------------
    # AMP selection
    # ---------------------------------------------------------
    amp_dtype = None
    if device.type == "cuda" and use_amp:
        cc = torch.cuda.get_device_capability()
        amp_dtype = torch.bfloat16 if cc[0] >= 8 else torch.float16

    # ---------------------------------------------------------
    # Inference
    # ---------------------------------------------------------
    print("🚀 Running Pi3 inference...")
    start = time.time()

    if amp_dtype:
        with torch.cuda.amp.autocast(dtype=amp_dtype):
            res = model(imgs[None])
    else:
        res = model(imgs[None])

    print(f"✅ Inference time: {time.time() - start:.2f}s")

    # Free frame tensor ASAP
    del imgs
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Mask filtering
    # ---------------------------------------------------------
    print("🔹 Filtering point cloud...")
    masks = torch.sigmoid(res["conf"][..., 0]) > 0.1
    non_edge = ~depth_edge(res["local_points"][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0]

    # ---------------------------------------------------------
    # Save PLY
    # ---------------------------------------------------------
    output_ply = output_dir / f"result_{int(time.time())}.ply"

    write_ply(
        res["points"][0][masks].cpu(),
        res["colors"][0][masks].cpu(),
        output_ply,
    )

    print("🎉 Saved 3D model:", output_ply)
    return str(output_ply)


# ---------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------
# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Pi3 Video to 3D")
#     parser.add_argument("video", help="Path to input video (.mp4)")
#     parser.add_argument(
#         "-o",
#         "--output",
#         default="pi3_output",
#         help="Output directory",
#     )

#     args = parser.parse_args()

#     run_local_video_to_3d(args.video, args.output)
