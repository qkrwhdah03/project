#!/usr/bin/env python3
"""
TartanAir v2 Dataset Download Script

Downloads and processes TartanAir v2 dataset from HuggingFace Hub.
Designed for memory-efficient processing in constrained environments (20GB VRAM/Storage).

Process flow:
1. Download zip files one at a time
2. Extract contents
3. Keep only 1/5 of frames (every 5th frame)
4. Delete zip file immediately after extraction
5. Repeat for next zip file

This approach minimizes peak disk usage by avoiding simultaneous existence of
compressed and extracted files.
"""
import os
import subprocess
import sys
import zipfile
import glob
import requests
from huggingface_hub import snapshot_download
from tqdm import tqdm

# Configuration
TARTANAIR_ROOT = "/root/project/data/tartanair2"
REPO_ID = "theairlabcmu/tartanair2"
ENVIRONMENTS = [
    "AbandonedCable",
    "AmusementPark",
    "Downtown",
    "AncientTowns",
]


def get_directory_size(path):
    """
    Calculate directory size in GB.
    
    Args:
        path: Directory path to measure
        
    Returns:
        Size in GB as float, or 0.0 if path doesn't exist
    """
    if not os.path.exists(path):
        return 0.0
    
    result = subprocess.run(
        ['du', '-sb', path],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        size_bytes = int(result.stdout.split()[0])
        return size_bytes / (1024 ** 3)  # Convert to GB
    return 0.0


def get_zip_file_size(env_name):
    """
    Get total size of zip files for an environment from HuggingFace Hub.
    
    Uses HEAD requests to check file sizes without downloading.
    Returns sum of image_lcam_front.zip and depth_lcam_front.zip sizes.
    
    Args:
        env_name: Environment name (e.g., "Downtown")
        
    Returns:
        Total size in GB as float
    """
    total_size = 0.0
    
    zip_files = [
        f"{env_name}/Data_easy/image_lcam_front.zip",
        f"{env_name}/Data_easy/depth_lcam_front.zip",
    ]
    
    for zip_path in zip_files:
        try:
            url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{zip_path}"
            response = requests.head(url, allow_redirects=True, timeout=10)
            
            if response.status_code == 200:
                content_length = response.headers.get('Content-Length')
                if content_length:
                    size_bytes = int(content_length)
                    total_size += size_bytes / (1024 ** 3)  # Convert to GB
                else:
                    print(f"  WARNING: {zip_path}: No Content-Length header")
            else:
                print(f"  WARNING: {zip_path}: HTTP {response.status_code}")
        except Exception as e:
            print(f"  WARNING: {zip_path} size check failed: {e}")
    
    return total_size


def get_sorted_environments_by_size():
    """
    Sort environments by total zip file size (largest first).
    
    Returns:
        List of environment names sorted by size (descending)
    """
    print("\n" + "=" * 70)
    print("Checking zip file sizes for each environment...")
    print("=" * 70)
    
    env_sizes = []
    for env_name in ENVIRONMENTS:
        print(f"  Checking {env_name}...", end=" ", flush=True)
        total_size = get_zip_file_size(env_name)
        env_sizes.append((env_name, total_size))
        print(f"OK ({total_size:.2f} GB)")
    
    # Sort by size (largest first)
    env_sizes.sort(key=lambda x: x[1], reverse=True)
    
    print("\nProcessing order (largest first):")
    for i, (env_name, size) in enumerate(env_sizes, 1):
        print(f"   {i}. {env_name}: {size:.2f} GB")
    print("=" * 70)
    
    return [env_name for env_name, _ in env_sizes]


def resample_images_in_path(base_path, keep_every=5):
    """
    Keep only every Nth frame (default: every 5th frame).
    
    Filters files based on frame number in filename (e.g., "000123_lcam_front.png" -> 123).
    Keeps frames where frame_number % keep_every == 0 (0, 5, 10, 15, ...).
    
    Args:
        base_path: Base directory containing P00X trajectory folders
        keep_every: Keep every Nth frame (default: 5)
        
    Returns:
        Tuple of (kept_count, deleted_count)
    """
    traj_folders = []
    if not os.path.exists(base_path):
        return 0, 0
    
    # Find all P00X trajectory folders
    for item in os.listdir(base_path):
        traj_path = os.path.join(base_path, item)
        if os.path.isdir(traj_path) and item.startswith('P'):
            traj_folders.append(traj_path)
    
    if not traj_folders:
        return 0, 0
    
    total_deleted = 0
    total_kept = 0
    
    for traj_path in traj_folders:
        # Process both image and depth folders
        for folder_name in ['image_lcam_front', 'depth_lcam_front']:
            folder_path = os.path.join(traj_path, folder_name)
            if not os.path.exists(folder_path):
                continue
            
            # Collect files with frame numbers
            files = []
            for f in os.listdir(folder_path):
                if f.endswith(('.png', '.npy')):
                    # Extract frame number from filename (e.g., "000123_lcam_front.png" -> 123)
                    try:
                        num_str = f.split('_')[0]
                        num = int(num_str)
                        files.append((num, f))
                    except:
                        continue
            
            if not files:
                continue
            
            # Sort by frame number
            files.sort(key=lambda x: x[0])
            
            # Keep only frames where frame_number % keep_every == 0
            for num, filename in files:
                file_path = os.path.join(folder_path, filename)
                if num % keep_every == 0:
                    total_kept += 1
                else:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            total_deleted += 1
                    except Exception as e:
                        pass  # Ignore deletion failures
    
    return total_kept, total_deleted


def process_single_zip(env_name, zip_pattern, env_path):
    """
    Process a single zip file: download -> extract -> delete zip -> resample frames.
    
    Args:
        env_name: Environment name
        zip_pattern: Zip file pattern for HuggingFace Hub
        env_path: Local environment directory path
        
    Returns:
        True if successful, False otherwise
    """
    zip_filename = os.path.basename(zip_pattern)
    print(f"\n  Processing: {zip_filename}")
    
    try:
        # Step 1: Download zip file
        print(f"    [1/4] Downloading...")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=TARTANAIR_ROOT,
            allow_patterns=[zip_pattern],
        )
        
        # Find downloaded zip file path
        data_easy_path = os.path.join(env_path, "Data_easy")
        os.makedirs(data_easy_path, exist_ok=True)
        
        zip_file_path = os.path.join(data_easy_path, zip_filename)
        if not os.path.exists(zip_file_path):
            # Check nested path structure
            nested_path = os.path.join(data_easy_path, env_name, "Data_easy", zip_filename)
            if os.path.exists(nested_path):
                zip_file_path = nested_path
            else:
                print(f"    ERROR: Zip file not found: {zip_filename}")
                return False
        
        # Step 2: Extract zip file
        print(f"    [2/4] Extracting...")
        extract_path = data_easy_path
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            print(f"      Extracting {total_files} files...")
            
            for i, member in enumerate(zip_ref.namelist()):
                if (i + 1) % 1000 == 0 or (i + 1) == total_files:
                    print(f"      Progress: {i+1}/{total_files} ({100*(i+1)//total_files}%)")
                zip_ref.extract(member, extract_path)
        
        # Step 3: Delete zip file immediately after extraction
        print(f"    [3/4] Deleting zip file...")
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)
            print(f"      Deleted: {zip_filename}")
        
        # Step 4: Resample frames (keep 1/5)
        print(f"    [4/4] Resampling frames (keep 1/5)...")
        # Check for nested structure
        nested_path = os.path.join(data_easy_path, env_name, "Data_easy")
        if os.path.exists(nested_path):
            base_path = nested_path
        else:
            base_path = data_easy_path
        
        kept, deleted = resample_images_in_path(base_path, keep_every=5)
        print(f"      Kept: {kept}, Deleted: {deleted}")
        
        print(f"  Completed: {zip_filename}")
        return True
        
    except Exception as e:
        print(f"  ERROR: {zip_filename} processing failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to delete zip file even on failure
        try:
            if 'zip_file_path' in locals() and os.path.exists(zip_file_path):
                os.remove(zip_file_path)
                print(f"      Cleaned up failed zip file: {zip_filename}")
        except:
            pass
        return False


def download_environment(env_name):
    """
    Download and process all zip files for a single environment.
    
    Args:
        env_name: Environment name to process
        
    Returns:
        Final size in GB if successful, None otherwise
    """
    print(f"\n{'='*70}")
    print(f"Environment: {env_name}")
    print(f"{'='*70}")
    
    env_path = os.path.join(TARTANAIR_ROOT, env_name)
    os.makedirs(TARTANAIR_ROOT, exist_ok=True)
    
    size_before = get_directory_size(env_path)
    
    zip_patterns = [
        f"{env_name}/Data_easy/image_lcam_front.zip",
        f"{env_name}/Data_easy/depth_lcam_front.zip",
    ]
    
    print(f"Zip files to process:")
    for p in zip_patterns:
        print(f"  - {p}")
    print()
    
    success_count = 0
    for zip_pattern in zip_patterns:
        success = process_single_zip(env_name, zip_pattern, env_path)
        if success:
            success_count += 1
    
    size_after = get_directory_size(env_path)
    print(f"\n{env_name} completed ({success_count}/{len(zip_patterns)} successful)")
    print(f"Size: {size_after:.2f} GB (before: {size_before:.2f} GB)")
    
    return size_after if success_count == len(zip_patterns) else None


def main():
    """Main execution function."""
    print("=" * 70)
    print("TartanAir v2 Dataset Download Script")
    print("Memory-efficient processing for constrained environments")
    print("=" * 70)
    print(f"Output directory: {TARTANAIR_ROOT}")
    print(f"Environments: {len(ENVIRONMENTS)}")
    for i, env in enumerate(ENVIRONMENTS, 1):
        print(f"   {i}. {env}")
    print("=" * 70)
    print("Processing: Sequential zip file processing to minimize disk usage")
    print("Order: Largest zip files processed first")
    print("=" * 70)
    
    # Check sizes and sort by largest first
    sorted_environments = get_sorted_environments_by_size()
    
    results = {}
    total_size = 0.0
    
    for env_name in sorted_environments:
        size = download_environment(env_name)
        if size is not None:
            results[env_name] = size
            total_size += size
    
    # Final summary
    print("\n" + "=" * 70)
    print("Download Summary")
    print("=" * 70)
    print(f"{'Environment':<25} {'Size':>12}")
    print("-" * 70)
    
    for env_name in ENVIRONMENTS:
        if env_name in results:
            print(f"{env_name:<25} {results[env_name]:>11.2f} GB")
    
    print("-" * 70)
    print(f"{'Total':<25} {total_size:>11.2f} GB")
    print("=" * 70)
    
    print("\nAll downloads completed.")


if __name__ == "__main__":
    main()
