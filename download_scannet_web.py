#!/usr/bin/env python3
"""
ScanNet 실제 데이터 다운로드 (웹 기반)
ScanNet 공식 웹사이트에서 데이터를 다운로드합니다.
인증이 필요한 경우 사용자에게 안내합니다.
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path
import argparse

def check_disk_space(required_gb=10):
    """디스크 공간 확인"""
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024 ** 3)
    print(f"💾 Available disk space: {free_gb:.2f} GB")
    return free_gb >= required_gb

def download_with_wget(scene_id, save_dir, username=None, password=None):
    """wget을 사용하여 ScanNet 데이터 다운로드"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # ScanNet sens 파일 URL
    base_url = "http://kaldir.vc.in.tum.de/scannet/v2/scans"
    sens_url = f"{base_url}/{scene_id}/{scene_id}_sens.zip"
    
    print(f"📥 Downloading {scene_id} from ScanNet...")
    print(f"   URL: {sens_url}")
    print(f"   Note: This requires ScanNet account authentication")
    
    # wget 명령어 구성
    cmd = ["wget", "-O", str(save_path / f"{scene_id}_sens.zip"), sens_url]
    
    if username and password:
        cmd.extend(["--user", username, "--password", password])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Download successful")
        return save_path / f"{scene_id}_sens.zip"
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed")
        print(f"   Error: {e.stderr}")
        if "401" in str(e.stderr) or "403" in str(e.stderr):
            print(f"\n💡 Authentication required!")
            print(f"   Please:")
            print(f"   1. Register at http://www.scan-net.org/")
            print(f"   2. Accept the Terms of Use")
            print(f"   3. Get your credentials")
            print(f"   4. Run with --username and --password")
        return None

def extract_and_organize(zip_path, scene_id, save_dir):
    """sens zip 파일을 압축 해제하고 구조화"""
    import zipfile
    
    scene_dir = Path(save_dir) / scene_id
    extract_dir = scene_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 Extracting {zip_path.name}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # sens 파일 찾기
        sens_files = list(extract_dir.rglob("*.sens"))
        if sens_files:
            print(f"✅ Found sens file: {sens_files[0]}")
            print(f"   To extract frames, use SensReader from ScanNet repository")
            return True
        else:
            print(f"⚠️  No .sens file found, but extraction completed")
            return True
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Download real ScanNet data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Try download (will prompt for auth if needed)
  python download_scannet_web.py --scene scene0000_00
  
  # With credentials
  python download_scannet_web.py --scene scene0000_00 --username USER --password PASS
  
Note: ScanNet requires:
  1. Registration at http://www.scan-net.org/
  2. Terms of Use acceptance
  3. Institutional email address
        """
    )
    
    parser.add_argument("--scene", required=True, help="Scene ID (e.g., scene0000_00)")
    parser.add_argument("--save_dir", default="./scannet_sample", help="Save directory")
    parser.add_argument("--username", help="ScanNet username")
    parser.add_argument("--password", help="ScanNet password")
    parser.add_argument("--max_size_gb", type=float, default=10.0, help="Max size in GB")
    
    args = parser.parse_args()
    
    print("==============================================")
    print("   🏗 ScanNet Real Data Downloader")
    print("==============================================")
    
    if not check_disk_space(args.max_size_gb):
        print("❌ Insufficient disk space")
        sys.exit(1)
    
    # 다운로드 시도
    zip_path = download_with_wget(
        args.scene,
        args.save_dir,
        args.username,
        args.password
    )
    
    if zip_path and zip_path.exists():
        # 압축 해제
        extract_and_organize(zip_path, args.scene, args.save_dir)
        
        # zip 파일 크기 확인
        size_gb = zip_path.stat().st_size / (1024 ** 3)
        print(f"\n📊 Downloaded: {size_gb:.2f} GB")
        print(f"✅ Scene ready at: {args.save_dir}/{args.scene}")
    else:
        print(f"\n❌ Download failed")
        print(f"\n💡 To download ScanNet data:")
        print(f"   1. Visit http://www.scan-net.org/")
        print(f"   2. Register and accept Terms of Use")
        print(f"   3. Download manually or use credentials with this script")
        sys.exit(1)

if __name__ == "__main__":
    main()

