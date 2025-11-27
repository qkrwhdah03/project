import os
import sys
import argparse
import subprocess
import shutil
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image


# ============================================================
# Utility: Disk space check
# ============================================================

def check_disk_space(required_gb=5):
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024 ** 3)
    print(f"💾 Available disk: {free_gb:.2f} GB")

    if free_gb < required_gb:
        print(f"⚠  WARNING: Less than {required_gb}GB space available.")
        return False
    return True


# ============================================================
# Utility: File download with progress bar
# ============================================================

def download_file(url, save_path):
    print(f"📥 Downloading: {url}")
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))

    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f, tqdm(
        total=total,
        unit='B',
        unit_scale=True,
        desc=save_path.name,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(1024 * 1024):
            f.write(chunk)
            bar.update(len(chunk))

    return save_path.exists()


# ============================================================
# Dummy Scene Creator (for testing without real ScanNet data)
# ============================================================

def create_dummy_scene(scene_dir, num_frames=500, width=640, height=480):
    print(f"🧪 Creating dummy scene at: {scene_dir}")
    scene_dir.mkdir(parents=True, exist_ok=True)

    (scene_dir / "color").mkdir(exist_ok=True)
    (scene_dir / "depth").mkdir(exist_ok=True)
    (scene_dir / "pose").mkdir(exist_ok=True)
    (scene_dir / "intrinsic").mkdir(exist_ok=True)

    # Intrinsic
    K = np.array([
        [577.87, 0, 320],
        [0, 577.87, 240],
        [0, 0, 1]
    ], dtype=np.float32)
    np.savetxt(scene_dir / "intrinsic" / "intrinsic_color.txt", K)

    # Frames
    for i in range(num_frames):
        # Color
        img = Image.new("RGB", (width, height),
                        color=(i * 5 % 255, i * 11 % 255, i * 17 % 255))
        img.save(scene_dir / "color" / f"{i}.jpg")

        # Depth (uint16)
        depth = np.random.randint(500, 4000, (height, width), dtype=np.uint16)
        Image.fromarray(depth, mode='I;16').save(scene_dir / "depth" / f"{i}.png")

        # Pose (4x4)
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = i * 0.1
        pose[1, 3] = i * 0.05
        pose[2, 3] = i * 0.02
        np.savetxt(scene_dir / "pose" / f"{i}.txt", pose)

    print(f"✅ Dummy scene generated ({num_frames} frames)")
    return True


# ============================================================
# Actual ScanNet Downloader
# (Must have download access + official script installed)
# ============================================================

def download_scannet_scene(scene_id, save_dir, official_script="download-scannet.py"):
    """
    실제 ScanNet 다운로드를 시도.
    이 함수는 사용자가 이미 ScanNet 계정과 라이선스 동의를 완료했다고 가정.
    """

    scene_dir = save_dir / scene_id

    if scene_dir.exists():
        print(f"✔ Scene already exists: {scene_dir}")
        return True

    print(f"📥 Attempting official ScanNet download: {scene_id}")

    official_cmd = [
        "python",
        official_script,
        "--id", scene_id,
        "--o", str(save_dir)
    ]

    print("➡ Running:", " ".join(official_cmd))
    try:
        subprocess.run(official_cmd, check=True)
    except Exception as e:
        print(f"❌ Error running official script: {e}")
        return False

    return scene_dir.exists()


# ============================================================
# Main Entrypoint
# ============================================================

def main():
    parser = argparse.ArgumentParser("ScanNet Downloader / Preparator")
    parser.add_argument("--scene", required=True, help="Scene ID (e.g., scene0000_00)")
    parser.add_argument("--save_dir", type=str, default="./scannet_data")
    parser.add_argument("--dummy", action="store_true",
                        help="Create dummy scene instead of real download")
    parser.add_argument("--required_space", type=float, default=5.0,
                        help="Minimum free disk space (GB)")
    parser.add_argument("--official_script", type=str,
                        default="download-scannet.py",
                        help="Path to official ScanNet download script")

    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    print("==============================================")
    print("        🏗 ScanNet Downloader Start")
    print("==============================================")

    # Disk space check
    if not check_disk_space(args.required_space):
        print("❌ Not enough disk space.")
        sys.exit(1)

    # Main logic
    if args.dummy:
        create_dummy_scene(save_dir / args.scene)
    else:
        print("⚠ Must comply with ScanNet Terms & official script requirements.")
        downloaded = download_scannet_scene(
            args.scene,
            save_dir,
            official_script=args.official_script
        )
        if not downloaded:
            print("❌ Failed to download via official script.")
            print("   → Tip: Use --dummy to create test data.")
            sys.exit(1)

    print(f"🎉 Completed. Scene ready at: {save_dir / args.scene}")


if __name__ == "__main__":
    main()
