import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset_dolly import ScanNetDataset  # 방금 수정한 데이터셋 클래스 import

# 설정 (경로는 본인 환경에 맞게 수정)
DATA_DIR = "./scannet_sample/scene0000_00" 
SAVE_DIR = "./data_check_vis"
os.makedirs(SAVE_DIR, exist_ok=True)

def unnormalize(tensor):
    """
    [-1, 1] 범위의 텐서를 [0, 1] numpy로 변환 (시각화용)
    Input: (C, H, W) tensor
    Output: (H, W, C) numpy array
    """
    img = (tensor + 1.0) * 0.5
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).cpu().numpy()

def visualize_sample():
    print(f"📂 Loading dataset from: {DATA_DIR}")
    
    # 데이터셋 인스턴스 생성
    try:
        dataset = ScanNetDataset(DATA_DIR, img_size=(512, 512))
    except Exception as e:
        print(f"❌ 데이터셋 로드 실패: {e}")
        return

    print(f"✅ 데이터셋 로드 완료. 총 프레임 수(추정): {len(dataset.frames)}")
    print("🔍 샘플링 및 시각화 시작...")

    # 유효한 샘플 5개 찾을 때까지 반복
    count = 0
    max_samples = 5
    
    for i in range(100): # 최대 100번 시도
        if count >= max_samples:
            break
            
        # 랜덤 샘플 가져오기
        sample = dataset[i] # 인덱스는 내부적으로 무시되고 랜덤 샘플링됨
        
        # 빈 데이터(로드 실패)인 경우 패스
        if not sample or "pixel_values" not in sample:
            continue
        
        # 필수 키 확인
        required_keys = ["pixel_values", "warped_image", "mask", "depth", "delta"]
        if not all(key in sample for key in required_keys):
            print(f"  ⚠️ Sample {i} missing required keys. Available: {list(sample.keys())}")
            continue
            
        # 데이터 추출
        target_tensor = sample["pixel_values"]    # GT (Output)
        warped_tensor = sample["warped_image"]    # Input Condition
        mask_tensor = sample["mask"]              # Mask
        depth_tensor = sample["depth"]            # Depth
        delta = sample["delta"].item()
        
        # 텐서 범위 체크 (디버깅용)
        print(f"\n[Sample {count+1}] Delta: {delta:.4f}m")
        print(f"  - Target Range: min={target_tensor.min():.2f}, max={target_tensor.max():.2f}")
        print(f"  - Warped Range: min={warped_tensor.min():.2f}, max={warped_tensor.max():.2f}")

        # 시각화 준비
        target_img = unnormalize(target_tensor)
        warped_img = unnormalize(warped_tensor)
        
        # Mask & Depth 시각화용 변환
        mask_img = mask_tensor.squeeze().cpu().numpy()
        depth_img = depth_tensor.squeeze().cpu().numpy()
        
        # Plot 그리기
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"Sample {count+1} | Camera Move: {delta:.3f}m (Dolly-in)", fontsize=16)
        
        # 1. Input (Warped Image)
        axes[0].imshow(warped_img)
        axes[0].set_title("Input: Warped Image\n(Condition)", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # 2. Mask (Holes)
        axes[1].imshow(mask_img, cmap='gray')
        axes[1].set_title("Input: Mask\n(White=Valid, Black=Hole)", fontsize=12)
        axes[1].axis('off')
        
        # 3. Ground Truth (Target)
        axes[2].imshow(target_img)
        axes[2].set_title("Target: Real Image\n(Ground Truth)", fontsize=12, fontweight='bold')
        axes[2].axis('off')

        # 4. Depth (Reference)
        axes[3].imshow(depth_img, cmap='inferno')
        axes[3].set_title("Reference: Depth Map", fontsize=12)
        axes[3].axis('off')
        
        # 저장
        save_path = os.path.join(SAVE_DIR, f"check_sample_{count+1}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"  📸 Saved visualization to {save_path}")
        
        count += 1

    if count == 0:
        print("⚠️ 유효한 샘플을 하나도 찾지 못했습니다. 경로 설정이나 데이터셋 로직을 확인하세요.")
    else:
        print(f"\n✅ 검증 완료! '{SAVE_DIR}' 폴더를 확인하세요.")

if __name__ == "__main__":
    visualize_sample()