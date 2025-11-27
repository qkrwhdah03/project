#!/usr/bin/env python3
"""
ScanNet 직접 다운로드 스크립트
ScanNet 웹사이트에서 sens 파일을 직접 다운로드합니다.
"""
import os
import sys
import requests
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse
import zipfile

# ScanNet 다운로드 베이스 URL (공식)
SCANNET_BASE_URL = "http://kaldir.vc.in.tum.de/scannet/v2/scans"

def check_disk_space(required_gb=10):
    """디스크 공간 확인"""
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024 ** 3)
    print(f"💾 Available disk space: {free_gb:.2f} GB")
    
    if free_gb < required_gb:
        print(f"⚠️  Warning: Less than {required_gb}GB available.")
        return False
    return True

def download_file(url, save_path, desc="Downloading"):
    """파일 다운로드 with progress bar"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Download failed: {e}")
        return False

def extract_sens_file(sens_path, output_dir):
    """sens 파일을 압축 해제하고 color, depth, pose, intrinsic 추출"""
    from SensReader import SensReader
    
    print(f"📦 Extracting sens file: {sens_path}")
    reader = SensReader(str(sens_path))
    
    scene_dir = Path(output_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)
    
    # 디렉토리 생성
    (scene_dir / "color").mkdir(exist_ok=True)
    (scene_dir / "depth").mkdir(exist_ok=True)
    (scene_dir / "pose").mkdir(exist_ok=True)
    (scene_dir / "intrinsic").mkdir(exist_ok=True)
    
    # 프레임 추출
    num_frames = reader.num_frames
    print(f"   Extracting {num_frames} frames...")
    
    for i in tqdm(range(num_frames), desc="Extracting frames"):
        # Color
        color = reader.get_color(i)
        if color is not None:
            from PIL import Image
            Image.fromarray(color).save(scene_dir / "color" / f"{i}.jpg")
        
        # Depth
        depth = reader.get_depth(i)
        if depth is not None:
            from PIL import Image
            Image.fromarray(depth).save(scene_dir / "depth" / f"{i}.png")
        
        # Pose
        pose = reader.get_pose(i)
        if pose is not None:
            import numpy as np
            np.savetxt(scene_dir / "pose" / f"{i}.txt", pose)
    
    # Intrinsic
    intrinsic = reader.get_intrinsic()
    if intrinsic is not None:
        import numpy as np
        np.savetxt(scene_dir / "intrinsic" / "intrinsic_color.txt", intrinsic)
    
    print(f"✅ Extraction completed")
    return True

def download_scene_simple(scene_id, save_dir, max_size_gb=10):
    """
    간단한 방법: sens 파일 다운로드 시도
    Note: ScanNet은 인증이 필요할 수 있습니다.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    scene_dir = save_path / scene_id
    if scene_dir.exists() and any((scene_dir / "color").iterdir()):
        print(f"✅ Scene {scene_id} already exists")
        return True
    
    # sens 파일 URL
    sens_url = f"{SCANNET_BASE_URL}/{scene_id}/{scene_id}_sens.zip"
    sens_file = save_path / f"{scene_id}_sens.zip"
    
    print(f"📥 Downloading {scene_id}...")
    print(f"   URL: {sens_url}")
    
    # 다운로드 시도
    if not sens_file.exists():
        success = download_file(sens_url, sens_file, desc=f"{scene_id}_sens.zip")
        if not success:
            print(f"❌ Failed to download {scene_id}")
            print(f"\n💡 Note: ScanNet requires:")
            print(f"   1. Account registration at http://www.scan-net.org/")
            print(f"   2. License agreement acceptance")
            print(f"   3. Direct download may require authentication")
            print(f"\n   Alternative: Use the official web interface to download")
            return False
    
    # 파일 크기 확인
    file_size_gb = sens_file.stat().st_size / (1024 ** 3)
    print(f"   Downloaded: {file_size_gb:.2f} GB")
    
    if file_size_gb > max_size_gb:
        print(f"⚠️  File size ({file_size_gb:.2f}GB) exceeds limit ({max_size_gb}GB)")
        sens_file.unlink()
        return False
    
    # 압축 해제
    print(f"📦 Extracting {sens_file.name}...")
    extract_dir = save_path / f"{scene_id}_extracted"
    extract_dir.mkdir(exist_ok=True)
    
    try:
        with zipfile.ZipFile(sens_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # sens 파일 찾기
        sens_files = list(extract_dir.rglob("*.sens"))
        if not sens_files:
            print(f"❌ No .sens file found in archive")
            return False
        
        sens_path = sens_files[0]
        
        # SensReader를 사용하여 추출 (또는 간단한 방법 사용)
        # 여기서는 SensReader가 없을 수 있으므로, 간단한 방법 사용
        print(f"✅ Extracted to {extract_dir}")
        print(f"   Sens file: {sens_path}")
        print(f"\n💡 Note: To extract frames, you may need SensReader")
        print(f"   Or use the extracted directory structure")
        
        # 임시로 sens 파일을 scene_dir로 이동
        if not scene_dir.exists():
            extract_dir.rename(scene_dir)
        
        # sens 파일 삭제 (공간 절약)
        sens_file.unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Download ScanNet scene data (direct download attempt)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--scene",
        required=True,
        help="Scene ID (e.g., scene0000_00)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./scannet_sample",
        help="Directory to save scene (default: ./scannet_sample)"
    )
    parser.add_argument(
        "--max_size_gb",
        type=float,
        default=10.0,
        help="Maximum download size in GB (default: 10.0)"
    )
    
    args = parser.parse_args()
    
    print("==============================================")
    print("   🏗 ScanNet Direct Downloader")
    print("==============================================")
    
    # 디스크 공간 확인
    if not check_disk_space(args.max_size_gb):
        print("❌ Insufficient disk space")
        sys.exit(1)
    
    # 다운로드
    success = download_scene_simple(
        args.scene,
        args.save_dir,
        args.max_size_gb
    )
    
    if success:
        print(f"\n🎉 Success! Scene ready at: {args.save_dir}/{args.scene}")
        print(f"\n📝 Next: Update DATA_DIR in train_dolly.py")
    else:
        print(f"\n❌ Download failed. Please check:")
        print(f"   1. Internet connection")
        print(f"   2. ScanNet account and license agreement")
        print(f"   3. Scene ID validity")
        sys.exit(1)

if __name__ == "__main__":
    main()

