import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import cv2


class FourierDeltaEmbedder(nn.Module):
    """Fourier-based delta embedding for distance encoding."""
    def __init__(self, freq_bands=8, out_channels=1):
        super().__init__()
        self.freq_bands = freq_bands
        input_dim = (freq_bands * 2) + 1
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SiLU(),
            nn.Linear(64, out_channels),
        )

    def forward(self, delta, target_h, target_w):
        if isinstance(delta, float) or (isinstance(delta, torch.Tensor) and delta.dim() == 0):
            delta = torch.tensor([delta], device=self.mlp[0].weight.device).float().view(1, 1)
        if isinstance(delta, torch.Tensor) and delta.dim() == 1:
            delta = delta.unsqueeze(1)
        B = delta.shape[0]
        embeds = [delta]
        for i in range(self.freq_bands):
            embeds.append(torch.sin(delta * (2 ** i) * torch.pi))
            embeds.append(torch.cos(delta * (2 ** i) * torch.pi))
        embed_cat = torch.cat(embeds, dim=-1)
        feat_vec = self.mlp(embed_cat)
        feat_map = feat_vec.unsqueeze(-1).unsqueeze(-1).expand(B, -1, target_h, target_w)
        return feat_map


def fill_holes_cv2(img_tensor, mask_tensor):
    """
    Fill holes in warped image using OpenCV inpainting.
    
    Args:
        img_tensor: (3, H, W) range [-1, 1]
        mask_tensor: (1, H, W) 1 for valid, 0 for invalid
        
    Returns:
        Filled image tensor and updated mask
    """
    # Convert to numpy
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 0.5 + 0.5) * 255.0
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    
    mask_np = mask_tensor.squeeze(0).cpu().numpy().astype(np.uint8) * 255
    
    # Dilation to fill small gaps
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(mask_np, kernel, iterations=1)
    
    # Inpainting
    hole_mask = cv2.bitwise_not(mask_np)
    filled_img = cv2.inpaint(img_np, hole_mask, 3, cv2.INPAINT_TELEA)
    
    # Convert back to tensor
    filled_img = filled_img.astype(np.float32) / 255.0
    filled_img = (filled_img - 0.5) / 0.5  # [0, 1] -> [-1, 1]
    
    out_tensor = torch.from_numpy(filled_img).permute(2, 0, 1).to(img_tensor.device)
    out_mask = torch.from_numpy(dilated_mask).float().unsqueeze(0).to(mask_tensor.device) / 255.0
    
    return out_tensor, out_mask


def forward_warp(image, depth, K, rel_pose):
    """
    Forward warping with hole filling.
    
    Args:
        image: (B, 3, H, W)
        depth: (B, 1, H, W)
        rel_pose: (B, 4, 4)
        K: (B, 3, 3) camera intrinsics
        
    Returns:
        Warped image and validity mask
    """
    B, _, H, W = image.shape
    device = image.device
    
    # Geometry projection
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    coords = torch.stack([x, y, torch.ones_like(x)], dim=-1).float()
    coords = coords.reshape(1, -1, 3).expand(B, -1, -1)

    K_inv = torch.inverse(K)
    depth_flat = depth.reshape(B, -1, 1)
    
    points_src = torch.bmm(coords, K_inv.transpose(1, 2)) * depth_flat
    
    R_rel = rel_pose[:, :3, :3]
    T_rel = rel_pose[:, :3, 3:]
    
    points_dst = torch.bmm(points_src, R_rel.transpose(1, 2)) + T_rel.transpose(1, 2)
    proj = torch.bmm(points_dst, K.transpose(1, 2))
    z = proj[..., 2:3] + 1e-6
    uv = proj[..., :2] / z
    
    u = torch.round(uv[..., 0]).long()
    v = torch.round(uv[..., 1]).long()
    
    # Initialize with background value
    warped_image = torch.zeros_like(image) - 1.0
    mask = torch.zeros((B, 1, H, W), device=device)

    # Rendering loop with z-buffer
    for b in range(B):
        valid = (u[b] >= 0) & (u[b] < W) & (v[b] >= 0) & (v[b] < H) & (z[b].squeeze() > 0.1)
        if not valid.any():
            continue
        
        u_v, v_v = u[b][valid], v[b][valid]
        z_v = z[b][valid].squeeze()
        if z_v.ndim == 0:
            z_v = z_v.unsqueeze(0)
            
        color_v_flat = image[b].reshape(3, -1).transpose(0, 1)
        color_v = color_v_flat[valid]
        if color_v.ndim == 1:
            color_v = color_v.unsqueeze(0)
        
        # Z-buffer: render far to near
        _, idx = torch.sort(z_v, descending=True)
        
        u_s, v_s = u_v[idx], v_v[idx]
        c_s = color_v[idx]
        
        flat_idx = v_s * W + u_s
        
        # Pixel assignment
        for c in range(3):
            plane = warped_image[b, c].view(-1).clone()
            if c_s.ndim == 2:
                val = c_s[:, c]
            else:
                val = c_s
            plane[flat_idx] = val
            warped_image[b, c] = plane.view(H, W)
            
        mask_plane = mask[b, 0].view(-1).clone()
        mask_plane[flat_idx] = 1.0
        mask[b, 0] = mask_plane.view(H, W)

    # Hole filling post-processing
    final_warped = []
    final_masks = []
    
    for b in range(B):
        if mask[b].sum() == 0:
            final_warped.append(warped_image[b])
            final_masks.append(mask[b])
            continue
            
        filled_img, filled_mask = fill_holes_cv2(warped_image[b], mask[b])
        final_warped.append(filled_img)
        final_masks.append(filled_mask)
        
    warped_image = torch.stack(final_warped)
    mask = torch.stack(final_masks)

    return warped_image, mask


class TartanAirForwardDataset(Dataset):
    """Dataset for forward warping with rectification."""
    def __init__(self, root_dir, img_size=(512, 512), min_dist=1.0, max_dist=10.0):
        self.root = root_dir
        self.img_size = img_size
        self.min_dist = min_dist
        self.max_dist = max_dist
        
        self.pairs = []
        self._scan_and_pair()
        
        self.delta_embedder = FourierDeltaEmbedder()
        
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])
        
        self.depth_transform = transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST)

        # NED to Camera coordinate conversion
        self.T_ned2cam = torch.tensor([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32)

    def _scan_and_pair(self):
        """Scan dataset and find valid forward motion pairs."""
        print(f"Scanning TartanAir at {self.root}...")
        traj_folders = []
        
        # TartanAir v2 structure: EnvName/Data_easy/EnvName/Data_easy/P00X
        for env_dir in os.listdir(self.root):
            env_path = os.path.join(self.root, env_dir)
            if not os.path.isdir(env_path):
                continue
            
            data_easy_path = os.path.join(env_path, "Data_easy")
            if not os.path.exists(data_easy_path):
                continue
            
            # Check nested structure
            nested_data_easy = os.path.join(data_easy_path, env_dir, "Data_easy")
            if os.path.exists(nested_data_easy):
                data_easy_path = nested_data_easy
            
            # Find P00X folders
            for item in os.listdir(data_easy_path):
                traj_path = os.path.join(data_easy_path, item)
                if os.path.isdir(traj_path) and item.startswith('P'):
                    img_dir = os.path.join(traj_path, "image_lcam_front")
                    depth_dir = os.path.join(traj_path, "depth_lcam_front")
                    pose_file = os.path.join(traj_path, "pose_lcam_front.txt")
                    
                    if os.path.exists(img_dir) and os.path.exists(depth_dir) and os.path.exists(pose_file):
                        traj_folders.append(traj_path)
        
        # Support legacy structure: image_left and pose_left.txt
        for root, dirs, files in os.walk(self.root):
            if "image_left" in dirs and "pose_left.txt" in files:
                if root not in traj_folders:
                    traj_folders.append(root)
        
        for traj in tqdm(traj_folders):
            # Determine file naming convention
            if os.path.exists(os.path.join(traj, "pose_lcam_front.txt")):
                pose_file = os.path.join(traj, "pose_lcam_front.txt")
                img_dir = os.path.join(traj, "image_lcam_front")
                depth_dir = os.path.join(traj, "depth_lcam_front")
                img_suffix = "_lcam_front.png"
                depth_suffix = "_lcam_front_depth.png"
            else:
                pose_file = os.path.join(traj, "pose_left.txt")
                img_dir = os.path.join(traj, "image_left")
                depth_dir = os.path.join(traj, "depth_left")
                img_suffix = "_left.png"
                depth_suffix = "_left_depth.npy"
            
            if not os.path.exists(pose_file):
                continue
                
            poses = np.loadtxt(pose_file)
            valid_pairs = self.find_forward_motion_pairs(poses, self.min_dist, self.max_dist)
            
            for (idx_t, idx_next) in valid_pairs:
                if (os.path.exists(os.path.join(img_dir, f"{idx_t:06d}{img_suffix}")) and
                    os.path.exists(os.path.join(img_dir, f"{idx_next:06d}{img_suffix}")) and
                    os.path.exists(os.path.join(depth_dir, f"{idx_t:06d}{depth_suffix}")) and
                    os.path.exists(os.path.join(depth_dir, f"{idx_next:06d}{depth_suffix}"))):
                    
                    self.pairs.append({
                        "folder": traj,
                        "t": idx_t,
                        "next": idx_next,
                        "pose_t": poses[idx_t],
                        "pose_next": poses[idx_next],
                        "img_suffix": img_suffix,
                        "depth_suffix": depth_suffix
                    })
        print(f"Total valid pairs found: {len(self.pairs)}")

    def find_forward_motion_pairs(self, poses, min_dist, max_dist):
        """Find frame pairs with forward motion."""
        valid_pairs = []
        num_frames = len(poses)
        positions = poses[:, :3]
        quaternions = poses[:, 3:]
        rotations = R.from_quat(quaternions)
        search_window = 50
        
        for i in range(num_frames - 1):
            pos_ref = positions[i]
            rot_ref = rotations[i]
            view_vec = rot_ref.apply(np.array([1, 0, 0]))
            
            end = min(num_frames, i + search_window)
            for j in range(i + 1, end):
                pos_tgt = positions[j]
                motion_vec = pos_tgt - pos_ref
                dist = np.linalg.norm(motion_vec)
                
                if dist < min_dist:
                    continue
                if dist > max_dist:
                    break
                
                motion_dir = motion_vec / (dist + 1e-8)
                cos_sim = np.dot(view_vec, motion_dir)
                if cos_sim < 0.5:
                    continue
                
                rot_tgt = rotations[j]
                view_vec_tgt = rot_tgt.apply(np.array([1, 0, 0]))
                view_sim = np.dot(view_vec, view_vec_tgt)
                if view_sim < 0.8:
                    continue
                
                valid_pairs.append((i, j))
                break
        return valid_pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        folder = item['folder']
        idx_t = item['t']
        idx_next = item['next']
        img_suffix = item.get('img_suffix', '_left.png')
        depth_suffix = item.get('depth_suffix', '_left_depth.npy')
        
        # Determine directory paths
        if 'lcam_front' in img_suffix:
            img_dir_name = "image_lcam_front"
            depth_dir_name = "depth_lcam_front"
        else:
            img_dir_name = "image_left"
            depth_dir_name = "depth_left"
        
        # Load images and depths
        try:
            I_t = Image.open(os.path.join(folder, img_dir_name, f"{idx_t:06d}{img_suffix}")).convert("RGB")
            I_next = Image.open(os.path.join(folder, img_dir_name, f"{idx_next:06d}{img_suffix}")).convert("RGB")
            
            depth_file_t = os.path.join(folder, depth_dir_name, f"{idx_t:06d}{depth_suffix}")
            depth_file_next = os.path.join(folder, depth_dir_name, f"{idx_next:06d}{depth_suffix}")
            
            if depth_suffix.endswith('.png'):
                # PNG format: TartanAir depth decode (RGBA -> float32)
                depth_rgba_t = cv2.imread(depth_file_t, cv2.IMREAD_UNCHANGED)
                depth_rgba_next = cv2.imread(depth_file_next, cv2.IMREAD_UNCHANGED)
                
                if depth_rgba_t is None:
                    raise FileNotFoundError(f"Failed to read depth file: {depth_file_t}")
                if depth_rgba_next is None:
                    raise FileNotFoundError(f"Failed to read depth file: {depth_file_next}")
                
                if depth_rgba_t.dtype != np.uint8:
                    depth_rgba_t = depth_rgba_t.astype(np.uint8)
                if depth_rgba_next.dtype != np.uint8:
                    depth_rgba_next = depth_rgba_next.astype(np.uint8)
                
                if depth_rgba_t.ndim != 3 or depth_rgba_t.shape[2] != 4:
                    raise ValueError(f"Depth PNG does not have 4 channels (RGBA): {depth_rgba_t.shape}")
                if depth_rgba_next.ndim != 3 or depth_rgba_next.shape[2] != 4:
                    raise ValueError(f"Depth PNG does not have 4 channels (RGBA): {depth_rgba_next.shape}")
                
                # Reinterpret RGBA bytes as float32
                depth_float_t = depth_rgba_t.view("<f4")
                depth_float_t = np.squeeze(depth_float_t, -1)
                
                depth_float_next = depth_rgba_next.view("<f4")
                depth_float_next = np.squeeze(depth_float_next, -1)
                
                # Clean NaN/inf
                depth_float_t = np.nan_to_num(depth_float_t, nan=0.0, posinf=0.0, neginf=0.0)
                depth_float_next = np.nan_to_num(depth_float_next, nan=0.0, posinf=0.0, neginf=0.0)
                
                D_t = depth_float_t
                D_next = depth_float_next
            else:
                # NPY format
                D_t = np.load(depth_file_t)
                D_next = np.load(depth_file_next)
        except:
            return self.__getitem__(random.randint(0, len(self.pairs)-1))
        
        # Transform
        I_t_tensor = self.transform(I_t) * 2.0 - 1.0  # [-1, 1]
        I_next_tensor = self.transform(I_next) * 2.0 - 1.0  # [-1, 1]
        
        D_t_tensor = self.depth_transform(torch.from_numpy(D_t).unsqueeze(0).float())
        D_next_tensor = self.depth_transform(torch.from_numpy(D_next).unsqueeze(0).float())
        
        # Camera intrinsics (approximate)
        orig_w, orig_h = 640, 480
        fx, fy, cx, cy = 320.0, 320.0, 320.0, 240.0
        scale_x = self.img_size[1] / orig_w
        scale_y = self.img_size[0] / orig_h
        K = torch.eye(3)
        K[0, 0] = fx * scale_x
        K[1, 1] = fy * scale_y
        K[0, 2] = cx * scale_x
        K[1, 2] = cy * scale_y

        # Pose calculation
        pose_t_ned = torch.from_numpy(self._quat_to_mat(item['pose_t'])).float()
        pose_next_ned = torch.from_numpy(self._quat_to_mat(item['pose_next'])).float()
        
        # Distance in NED frame
        dist = torch.norm(pose_t_ned[:3, 3] - pose_next_ned[:3, 3])
        
        # Virtual pure forward pose in camera frame
        rel_pose_virt_cam = torch.eye(4)
        rel_pose_virt_cam[2, 3] = -dist  # Dolly-in: points get closer
        
        # Warp source (t) -> virtual
        warped_cond, mask_cond = forward_warp(
            I_t_tensor.unsqueeze(0),
            D_t_tensor.unsqueeze(0),
            K.unsqueeze(0),
            rel_pose_virt_cam.unsqueeze(0)
        )
        
        # Warp target (next) -> virtual (rectification)
        T_ned2cam = self.T_ned2cam
        
        # Relative pose T->Next in NED
        rel_t_to_next_ned = torch.mm(torch.inverse(pose_next_ned), pose_t_ned)
        # Convert to camera frame
        rel_t_to_next_cam = torch.mm(torch.mm(T_ned2cam, rel_t_to_next_ned), torch.inverse(T_ned2cam))
        
        # Transformation: T_virt_rel @ T_next_rel^-1
        rel_next_to_virt_cam = torch.mm(rel_pose_virt_cam, torch.inverse(rel_t_to_next_cam))
        
        # Warp target image to align with virtual pure forward
        rectified_target, mask_target = forward_warp(
            I_next_tensor.unsqueeze(0),
            D_next_tensor.unsqueeze(0),
            K.unsqueeze(0),
            rel_next_to_virt_cam.unsqueeze(0)
        )
        
        # Remove batch dimension
        warped_cond = warped_cond.squeeze(0)
        mask_cond = mask_cond.squeeze(0)
        rectified_target = rectified_target.squeeze(0)
        mask_target = mask_target.squeeze(0)
        
        # Combine masks: valid where both condition and target are valid
        final_mask = mask_cond * mask_target
        
        # Delta embedding
        self.delta_embedder.eval()
        with torch.no_grad():
            delta_map = self.delta_embedder(dist, self.img_size[0], self.img_size[1])
            if delta_map.dim() == 4:
                delta_map = delta_map.squeeze(0)

        # Concatenate condition features
        D_norm = torch.clamp(D_t_tensor, 0, 50.0) / 50.0
        condition_concat = torch.cat([warped_cond, D_norm, final_mask, delta_map], dim=0)
        
        return {
            "pixel_values": rectified_target,
            "condition_image": condition_concat,
            "prompt": "outdoor scene viewed from the front, forward camera movement along the ground, dolly-in shot, perspective converging ahead, visible parallax in foreground objects, realistic and detailed",
            "mask": final_mask
        }

    def _quat_to_mat(self, pose_7d):
        """Convert 7D pose (translation + quaternion) to 4x4 matrix."""
        t = pose_7d[:3]
        q = pose_7d[3:]
        mat = np.eye(4)
        mat[:3, :3] = R.from_quat(q).as_matrix()
        mat[:3, 3] = t
        return mat


class TartanAirDepthDataset(Dataset):
    """Dataset for depth estimation training."""
    def __init__(self, root_dir, img_size=(512, 512)):
        self.root = root_dir
        self.img_size = img_size
        
        self.pairs = []
        self._scan_and_pair()
        
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),  # [0, 1]
        ])
        
        self.depth_transform = transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST)

    def _scan_and_pair(self):
        """Scan TartanAir v2 structure and collect all valid frames."""
        print(f"Scanning TartanAir v2 for depth training at {self.root}...")
        traj_folders = []
        
        # TartanAir v2 structure: EnvName/Data_easy/EnvName/Data_easy/P00X
        for env_dir in os.listdir(self.root):
            env_path = os.path.join(self.root, env_dir)
            if not os.path.isdir(env_path):
                continue
            
            data_easy_path = os.path.join(env_path, "Data_easy")
            if not os.path.exists(data_easy_path):
                continue
            
            # Check nested structure
            nested_data_easy = os.path.join(data_easy_path, env_dir, "Data_easy")
            if os.path.exists(nested_data_easy):
                data_easy_path = nested_data_easy
            
            # Find P00X folders
            for item in os.listdir(data_easy_path):
                traj_path = os.path.join(data_easy_path, item)
                if os.path.isdir(traj_path) and item.startswith('P'):
                    img_dir = os.path.join(traj_path, "image_lcam_front")
                    depth_dir = os.path.join(traj_path, "depth_lcam_front")
                    pose_file = os.path.join(traj_path, "pose_lcam_front.txt")
                    
                    if os.path.exists(img_dir) and os.path.exists(depth_dir) and os.path.exists(pose_file):
                        traj_folders.append(traj_path)
        
        print(f"Found {len(traj_folders)} trajectories. Collecting frames...")
        
        for traj in tqdm(traj_folders):
            pose_file = os.path.join(traj, "pose_lcam_front.txt")
            if not os.path.exists(pose_file):
                continue
                
            poses = np.loadtxt(pose_file)  # (N, 7)
            img_dir = os.path.join(traj, "image_lcam_front")
            depth_dir = os.path.join(traj, "depth_lcam_front")
            
            # Check each frame for RGB and depth existence
            for idx in range(len(poses)):
                img_file = os.path.join(img_dir, f"{idx:06d}_lcam_front.png")
                depth_file = os.path.join(depth_dir, f"{idx:06d}_lcam_front_depth.png")
                
                if os.path.exists(img_file) and os.path.exists(depth_file):
                    self.pairs.append({
                        "folder": traj,
                        "frame_idx": idx,
                        "img_file": img_file,
                        "depth_file": depth_file
                    })
        
        print(f"Total valid frames found: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        
        # Load RGB image
        I = Image.open(item['img_file']).convert("RGB")
        I_tensor = self.transform(I)  # (3, H, W), [0, 1]
        
        # Load depth (PNG format: RGBA -> float32)
        depth_file = item['depth_file']
        
        if depth_file.endswith('.png'):
            depth_rgba = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
            
            if depth_rgba is None:
                raise FileNotFoundError(f"Failed to read depth file: {depth_file}")
            
            if depth_rgba.dtype != np.uint8:
                depth_rgba = depth_rgba.astype(np.uint8)
            
            if depth_rgba.ndim != 3 or depth_rgba.shape[2] != 4:
                raise ValueError(f"Depth PNG does not have 4 channels (RGBA): {depth_rgba.shape}")
            
            # Reinterpret RGBA bytes as float32
            depth_float = depth_rgba.view("<f4")
            depth_float = np.squeeze(depth_float, -1)
            
            # Clean NaN/inf
            depth_float = np.nan_to_num(depth_float, nan=0.0, posinf=0.0, neginf=0.0)
            
            D = depth_float.astype(np.float32)
        else:
            # NPY format
            D = np.load(depth_file).astype(np.float32)
        
        # Depth transform
        D_tensor = torch.from_numpy(D).unsqueeze(0).float()  # (1, H_orig, W_orig)
        D_tensor = self.depth_transform(D_tensor)  # (1, H, W)
        
        return {
            "source_rgb": I_tensor,      # (3, H, W), [0, 1]
            "source_depth": D_tensor     # (1, H, W), meters
        }
