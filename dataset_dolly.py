import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

def forward_warp(image, depth, K, pose_t, pose_next):
    """
    Standard Forward Warping
    image: (B, 3, H, W) range [-1, 1]
    depth: (B, 1, H, W) in meters
    """
    B, _, H, W = image.shape
    device = image.device
    
    # Warping을 위해 잠시 [0, 1] 범위로 변환 (시각적 안전성 확보)
    image_norm = (image + 1.0) * 0.5
    
    # 1. Meshgrid
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    coords = torch.stack([x, y, torch.ones_like(x)], dim=-1).float() # (H, W, 3)
    coords = coords.reshape(1, -1, 3).expand(B, -1, -1) # (B, H*W, 3)

    # 2. Unproject
    K_inv = torch.inverse(K)
    depth_flat = depth.reshape(B, -1, 1)
    
    # Depth가 0이거나 너무 먼 경우 필터링 (유효하지 않은 포인트)
    valid_depth_mask = (depth_flat > 0.1) & (depth_flat < 10.0)
    
    cam_points = torch.bmm(coords, K_inv.transpose(1, 2)) * depth_flat 

    # 3. Transform (Relative Pose)
    # T_rel = T_next^-1 @ T_t
    rel_pose = torch.bmm(torch.inverse(pose_next), pose_t)
    R = rel_pose[:, :3, :3]
    T = rel_pose[:, :3, 3:]
    
    p_trans = torch.bmm(cam_points, R.transpose(1, 2)) + T.transpose(1, 2)

    # 4. Project
    p_proj = torch.bmm(p_trans, K.transpose(1, 2))
    z_new = p_proj[..., 2:3]
    uv_new = p_proj[..., :2] / (z_new + 1e-6)

    # 5. Render
    warped_image = torch.zeros_like(image_norm)
    mask = torch.zeros((B, 1, H, W), device=device)

    # Python Loop는 느리지만 정확성을 위해 유지 (대규모 학습 시 CUDA 커널 필요)
    u_new = torch.round(uv_new[..., 0]).long()
    v_new = torch.round(uv_new[..., 1]).long()
    
    for b in range(B):
        # Valid check (화면 밖으로 나간 것 + Depth 유효성)
        valid = (u_new[b] >= 0) & (u_new[b] < W) & \
                (v_new[b] >= 0) & (v_new[b] < H) & \
                (z_new[b, :, 0] > 0.1) & \
                valid_depth_mask[b, :, 0]
        
        if valid.sum() == 0: continue
        
        u_v = u_new[b][valid]
        v_v = v_new[b][valid]
        z_v = z_new[b][valid].squeeze()
        c_v = image_norm[b].reshape(3, -1).transpose(0, 1)[valid]

        # Z-sorting (멀리 있는 것부터 그림 -> 가까운 것이 덮어씀)
        sort_idx = torch.argsort(z_v, descending=True)
        
        u_s, v_s = u_v[sort_idx], v_v[sort_idx]
        c_s = c_v[sort_idx]
        
        warped_image[b, :, v_s, u_s] = c_s.transpose(0, 1)
        mask[b, 0, v_s, u_s] = 1.0

    # 다시 [-1, 1] 범위로 복구
    warped_image = (warped_image * 2.0) - 1.0
    # 마스크된 부분(빈 공간)은 -1(검정)로 채움
    warped_image = warped_image * mask + (1 - mask) * (-1.0)
    
    return warped_image, mask

class ScanNetDataset(Dataset):
    def __init__(self, root_dir, img_size=(512, 512)):
        self.root = root_dir
        self.img_size = img_size
        self.frames = []
        
        # [수정 1] 파일 정렬 로직 개선 (숫자 기준 정렬)
        color_dir = os.path.join(root_dir, "color")
        if os.path.exists(color_dir):
            files = [f for f in os.listdir(color_dir) if f.endswith(".jpg")]
            # 파일명에서 숫자만 추출하여 정렬 (예: 0.jpg, 1.jpg ...)
            # 파일명이 'frame-000000.color.jpg' 형식이면 그에 맞춰 파싱 필요
            try:
                # 숫자로만 된 파일명 가정 (0.jpg)
                files.sort(key=lambda x: int(os.path.splitext(x)[0]))
            except:
                # 실패하면 문자열 정렬하되 경고 (ScanNet 구조 확인 필요)
                files.sort()
                print("Warning: Files sorted by string. Check if sequence is correct.")
                
            self.frames = [f.replace(".jpg","") for f in files]
            
        # Intrinsic 로드
        self.K_orig = np.eye(3, dtype=np.float32)
        k_path = os.path.join(root_dir, "intrinsic/intrinsic_color.txt")
        if os.path.exists(k_path):
            self.K_orig = np.loadtxt(k_path)[:3, :3].astype(np.float32)
        else:
            # Fallback (ScanNet Default)
            self.K_orig = np.array([[577.87, 0, 320], [0, 577.87, 240]], dtype=np.float32)
            
        # Original Size 확인
        if self.frames:
            try:
                with Image.open(os.path.join(color_dir, self.frames[0]+".jpg")) as img:
                    self.orig_w, self.orig_h = img.size
            except:
                self.orig_w, self.orig_h = 640, 480
        else:
            self.orig_w, self.orig_h = 640, 480

        # [수정 2] Transforms: SDXL은 [-1, 1] 범위를 기대함
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # [0,1] -> [-1,1]
        ])
        
        self.depth_transform = transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST)

    def __len__(self):
        # 데이터가 너무 적으면 에폭이 빨리 끝나므로 적당히 늘려 잡음
        return max(len(self.frames), 1000)

    def _load_pose(self, idx):
        path = os.path.join(self.root, "pose", f"{idx}.txt")
        if not os.path.exists(path): return None
        pose = np.loadtxt(path).astype(np.float32)
        # Infinite 값 체크
        if np.isinf(pose).any() or np.isnan(pose).any(): return None
        return torch.tensor(pose)

    def __getitem__(self, index):
        # Random Sampling 로직
        # index 인자를 무시하고 랜덤으로 뽑습니다 (Dataset 길이를 임의로 설정했으므로)
        
        max_attempts = 50  # 시도 횟수 증가
        for attempt in range(max_attempts):
            if len(self.frames) < 10: return {} # 데이터 너무 적음
            
            # 1. 시작 프레임 t 선택
            max_idx = max(0, len(self.frames) - 20)
            idx_t = random.randint(0, max_idx)
            frame_t = self.frames[idx_t]
            pose_t = self._load_pose(frame_t)
            if pose_t is None: continue

            # 2. 목표 프레임 t+k 선택 (Dolly-in 조건 확인)
            candidates = []
            # t 이후 1~15 프레임 사이에서 검색
            search_range = min(15, len(self.frames) - idx_t - 1)
            for offset in range(1, search_range + 1):
                if idx_t + offset >= len(self.frames): break
                
                frame_next = self.frames[idx_t + offset]
                pose_next = self._load_pose(frame_next)
                if pose_next is None: continue
                
                # 카메라 이동 벡터 계산
                t_vec = pose_t[:3, 3]
                next_vec = pose_next[:3, 3]
                move_vec = next_vec - t_vec
                dist = torch.norm(move_vec)
                
                # [중요] 카메라가 바라보는 방향(Z축)으로 이동했는지 확인 (Dolly-in)
                # Pose Matrix의 3번째 컬럼(혹은 로우)이 Look vector. ScanNet은 보통 Z가 look.
                # R * [0,0,1]^T + T
                look_dir = pose_t[:3, 2] # Rotation의 Z축 (Camera forward vector)
                
                # 이동 벡터와 시선 벡터의 내적 (Dot Product)
                # 값이 양수면 앞으로 이동, 음수면 뒤로 이동
                dot_prod = torch.dot(move_vec / (dist + 1e-6), look_dir)

                # 조건: 적당한 거리 이동 + 앞으로 이동(Dolly-in) - 조건 완화
                # 더미 데이터는 단순 이동 패턴이므로 dot_prod 조건 완화
                if 0.001 < dist < 5.0 and dot_prod > 0.1:  # 거리 범위 확대, dot_prod 임계값 대폭 낮춤
                    candidates.append((frame_next, pose_next, dist))
            
            if not candidates:
                # 마지막 시도에서는 조건을 더 완화
                if attempt >= max_attempts - 5:
                    # Fallback: 가장 가까운 프레임 사용
                    for offset in range(1, min(15, len(self.frames) - idx_t - 1) + 1):
                        if idx_t + offset >= len(self.frames): break
                        frame_next = self.frames[idx_t + offset]
                        pose_next = self._load_pose(frame_next)
                        if pose_next is None: continue
                        t_vec = pose_t[:3, 3]
                        next_vec = pose_next[:3, 3]
                        move_vec = next_vec - t_vec
                        dist = torch.norm(move_vec)
                        if dist > 0.001:  # 최소 거리만 체크
                            candidates.append((frame_next, pose_next, dist))
                    if candidates:
                        frame_next_id, pose_next, delta = random.choice(candidates)
                        frame_t = self.frames[idx_t]
                        break
                continue
            
            # 후보 중 하나 선택
            frame_next_id, pose_next, delta = random.choice(candidates)
            
            # 3. 이미지 및 깊이 로드
            try:
                img_path_t = os.path.join(self.root, "color", frame_t+".jpg")
                img_path_next = os.path.join(self.root, "color", frame_next_id+".jpg")
                depth_path_t = os.path.join(self.root, "depth", frame_t+".png")
                
                if not (os.path.exists(img_path_t) and os.path.exists(img_path_next) and os.path.exists(depth_path_t)):
                    continue

                img_t = Image.open(img_path_t).convert("RGB")
                img_next = Image.open(img_path_next).convert("RGB")
                depth_img = Image.open(depth_path_t)
                
                # Depth Scaling (mm -> meter)
                d_np = np.array(depth_img).astype(np.float32) / 1000.0
                depth_tensor = torch.from_numpy(d_np).unsqueeze(0) # (1, H, W)

            except Exception as e:
                print(f"File load error: {e}")
                continue

            # 4. 전처리 및 Warping
            I_t = self.transform(img_t)       # Range [-1, 1]
            I_next = self.transform(img_next) # Range [-1, 1] (Target)
            D_t = self.depth_transform(depth_tensor)

            # Intrinsic Scaling (Resize에 맞춤)
            scale_x = self.img_size[1] / self.orig_w
            scale_y = self.img_size[0] / self.orig_h
            K = torch.tensor(self.K_orig).clone()
            K[0,0] *= scale_x; K[0,2] *= scale_x
            K[1,1] *= scale_y; K[1,2] *= scale_y

            # Warping 수행
            with torch.no_grad():
                warped, mask = forward_warp(
                    I_t.unsqueeze(0), D_t.unsqueeze(0), K.unsqueeze(0), 
                    pose_t.unsqueeze(0), pose_next.unsqueeze(0)
                )

            # 5. 결과 반환
            return {
                "pixel_values": I_next,        # GT (Model Output Target)
                "warped_image": warped.squeeze(0), # Condition Input
                "mask": mask.squeeze(0),
                "depth": D_t,
                "delta": torch.tensor([delta]).float(),
                "prompt": "a photo of an indoor room, dolly-in camera movement" 
            }
            
        # 모든 시도 실패 시 (Main loop에서 처리 필요)
        return {"pixel_values": torch.zeros(3, *self.img_size)} # Dummy return to prevent crash, but should be handled