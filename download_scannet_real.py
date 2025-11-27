#!/usr/bin/env python3
"""
ScanNet 실제 데이터 다운로드 스크립트
공식 ScanNet 다운로드 스크립트를 사용하여 실제 데이터를 다운로드합니다.
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path
import argparse

# ScanNet 공식 다운로드 스크립트 URL
SCANNET_REPO = "https://github.com/ScanNet/ScanNet.git"
SCANNET_SCRIPT = "download-scannet.py"

def check_disk_space(required_gb=10):
    """디스크 공간 확인"""
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024 ** 3)
    print(f"💾 Available disk space: {free_gb:.2f} GB")
    
    if free_gb < required_gb:
        print(f"⚠️  Warning: Less than {required_gb}GB available.")
        return False
    return True

def clone_scannet_repo(repo_dir="./ScanNet"):
    """ScanNet 저장소 클론 (스크립트만 필요)"""
    repo_path = Path(repo_dir)
    
    if repo_path.exists():
        print(f"✅ ScanNet repo already exists at {repo_path}")
        return repo_path
    
    print(f"📥 Cloning ScanNet repository...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", SCANNET_REPO, str(repo_path)],
            check=True,
            capture_output=True
        )
        print(f"✅ Cloned successfully")
        return repo_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to clone: {e}")
        return None

def download_scene(scene_id, save_dir, scannet_script_path, username=None, password=None):
    """
    ScanNet 장면 다운로드
    
    Args:
        scene_id: 장면 ID (예: scene0000_00)
        save_dir: 저장 디렉토리
        scannet_script_path: 공식 다운로드 스크립트 경로
        username: ScanNet 사용자명 (선택)
        password: ScanNet 비밀번호 (선택)
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    scene_dir = save_path / scene_id
    if scene_dir.exists():
        print(f"✅ Scene {scene_id} already exists")
        return True
    
    script_path = Path(scannet_script_path)
    if not script_path.exists():
        print(f"❌ ScanNet script not found at {script_path}")
        return False
    
    print(f"📥 Downloading scene {scene_id}...")
    print(f"   This may take a while (several GB)...")
    
    # 공식 스크립트 실행
    cmd = [
        "python", str(script_path),
        "-o", str(save_path),
        "--id", scene_id,
        "--type", "sens"  # sens 파일만 (color, depth, pose, intrinsic 포함)
    ]
    
    # 사용자명/비밀번호가 있으면 추가
    if username:
        cmd.extend(["--username", username])
    if password:
        cmd.extend(["--password", password])
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ Successfully downloaded {scene_id}")
        return scene_dir.exists()
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        print(f"\n💡 Note: ScanNet requires:")
        print(f"   1. Account registration at http://www.scan-net.org/")
        print(f"   2. License agreement acceptance")
        print(f"   3. Username/password (or use interactive prompt)")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Download real ScanNet scene data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download scene0000_00 (will prompt for credentials)
  python download_scannet_real.py --scene scene0000_00
  
  # Download with credentials
  python download_scannet_real.py --scene scene0000_00 --username USER --password PASS
  
  # Download multiple small scenes (within 10GB limit)
  python download_scannet_real.py --scene scene0000_00 scene0001_00 scene0002_00
        """
    )
    
    parser.add_argument(
        "--scene",
        nargs="+",
        required=True,
        help="Scene ID(s) to download (e.g., scene0000_00)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./scannet_sample",
        help="Directory to save scenes (default: ./scannet_sample)"
    )
    parser.add_argument(
        "--username",
        type=str,
        default=None,
        help="ScanNet username (optional, will prompt if not provided)"
    )
    parser.add_argument(
        "--password",
        type=str,
        default=None,
        help="ScanNet password (optional, will prompt if not provided)"
    )
    parser.add_argument(
        "--repo_dir",
        type=str,
        default="./ScanNet",
        help="Directory for ScanNet repository (default: ./ScanNet)"
    )
    parser.add_argument(
        "--max_size_gb",
        type=float,
        default=10.0,
        help="Maximum total download size in GB (default: 10.0)"
    )
    
    args = parser.parse_args()
    
    # 디스크 공간 확인
    if not check_disk_space(args.max_size_gb):
        print("❌ Insufficient disk space")
        sys.exit(1)
    
    # ScanNet 저장소 클론
    repo_path = clone_scannet_repo(args.repo_dir)
    if not repo_path:
        print("❌ Failed to get ScanNet repository")
        sys.exit(1)
    
    script_path = repo_path / SCANNET_SCRIPT
    if not script_path.exists():
        print(f"❌ Download script not found at {script_path}")
        print(f"   Please check the repository structure")
        sys.exit(1)
    
    print(f"\n🎯 Downloading {len(args.scene)} scene(s): {args.scene}")
    print(f"📁 Save directory: {args.save_dir}")
    print(f"💾 Max size: {args.max_size_gb} GB\n")
    
    # 각 장면 다운로드
    success_count = 0
    for scene_id in args.scene:
        try:
            success = download_scene(
                scene_id,
                args.save_dir,
                script_path,
                args.username,
                args.password
            )
            if success:
                success_count += 1
                print(f"✅ {scene_id} completed\n")
            else:
                print(f"❌ {scene_id} failed\n")
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error with {scene_id}: {e}\n")
    
    print(f"\n🎉 Completed: {success_count}/{len(args.scene)} scenes downloaded")
    
    if success_count > 0:
        print(f"\n📝 Next steps:")
        print(f"   1. Update DATA_DIR in train_dolly.py to: {args.save_dir}/{args.scene[0]}")
        print(f"   2. Run training: python train_dolly.py")

if __name__ == "__main__":
    main()

