from pathlib import Path
import subprocess
import imageio_ffmpeg


def ensure_mp4(video_path: str) -> str:
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    # Already MP4
    if video_path.suffix.lower() == ".mp4":
        print(f"[INFO] Already MP4: {video_path.name}")
        return str(video_path)

    # Only convert MOV
    if video_path.suffix.lower() != ".mov":
        raise ValueError("Only .mov or .mp4 supported")

    mp4_path = video_path.with_suffix(".mp4")

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    print(f"[INFO] Converting {video_path.name} → {mp4_path.name}")

    cmd = [
        ffmpeg_exe,
        "-y",
        "-loglevel", "error",   # ❗ only errors, no spam
        "-i", str(video_path),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(mp4_path)
    ]

    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,  # ❗ silence ffmpeg completely
        check=True
    )

    print(f"[SUCCESS] Saved: {mp4_path}")

    return str(mp4_path)
