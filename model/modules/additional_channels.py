# model/modules/additional_channels.py
# Additional channels for CubeDiff (Stable Diffusion 2)

# NOTE: Cited from https://github.com/Juan5713/OpenCubeDiff/ (Open source)

from typing import List, Tuple
import torch
import torch.nn as nn
import numpy as np
import math

def calculate_coords(
    view: torch.Tensor,   # unit vector (3,)
    up: torch.Tensor,     # unit vector (3,)
    face_size: int = 64,
    fov: float = 95.0,
)-> Tuple[torch.Tensor, torch.Tensor]:
    
    theta = (fov / 2) * math.pi / 180.0
    scale = torch.tan(torch.tensor(theta, dtype=view.dtype, device=view.device))

    left = torch.cross(up, view, dim=0)  # (3,)
    left_top = view + (up + left) * scale

    v = -up * scale
    u = -left * scale
    n = torch.arange(1, 2 * face_size, 2) / face_size
    gx, gy = torch.meshgrid(n, n)  # (face_size, face_size)

    rays = (
        left_top[None, None, :] +
        gx[..., None] * u[None, None, :] +
        gy[..., None] * v[None, None, :]
    )

    x = rays[..., 0]
    y = rays[..., 1]
    z = rays[..., 2]

    theta = torch.atan2(y, x)
    phi = torch.atan2(z, torch.sqrt(x**2 + y **2))

    return theta, phi

def calculate_positional_encoding(
    batch_size: int,
    face_size: int= 64, 
    fov: float= 95.0,
    suffixes: List[str] = ["posx", "posy", "posz", "negx", "negy", "negz"]
)-> torch.Tensor: # [B, 6, 2, H, W]
    """
    Computes (u,v) positional encodings for all six cubemap faces
    using unit cube formulation and consistent global normalization.
    """

    encs = []
    for face in suffixes:
        if face == "posx":
            theta, phi = calculate_coords(torch.Tensor([1., 0., 0.]), torch.Tensor([0., 0., 1.]), face_size, fov)
        elif face == "negx":
            theta, phi = calculate_coords(torch.Tensor([-1., 0., 0.]), torch.Tensor([0., 0., 1.]), face_size, fov)
        elif face == "posy":
            theta, phi = calculate_coords(torch.Tensor([0., 1., 0.]), torch.Tensor([0., 0., 1.]), face_size, fov)
        elif face == "negy":
            theta, phi = calculate_coords(torch.Tensor([0., -1., 0.]), torch.Tensor([0., 0., 1.]), face_size, fov)
        elif face == "posz":
            theta, phi = calculate_coords(torch.Tensor([0., 0., 1.]), torch.Tensor([1., 0., 0.]), face_size, fov)
        elif face == "negz":
            theta, phi = calculate_coords(torch.Tensor([0., 0., -1.]), torch.Tensor([1., 0., 0.]), face_size, fov)
    
        # theta : [-pi, pi], phi: [-pi/2, pi/2]
        # Normalize to [0,1]
        theta = (theta / math.pi + 1) / 2.0
        phi = phi / math.pi + 1/2

        pos_enc = torch.stack([theta, phi], dim = 0) # [2, face_size, face_size]
        encs.append(pos_enc)

    encs = torch.stack(encs, dim = 0) # [6, 2, face_size, face_size]
    encs = encs.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1) # [B, 6, 2, face_size, face_size]
    return encs

def calculate_mask_tensors(
    batch_size: int, 
    drop_ids: torch.Tensor, 
    face_size: int, 
)-> torch.Tensor:
    mask = torch.ones((batch_size, 6, 1, face_size, face_size), dtype=torch.float32) # Shape: (B, T, 1, H, W)
    mask[:, drop_ids] = 0.0
    return mask # [B, T, 1, H, W]

def make_extra_channels_tensor(
    batch_size: int, 
    drop_ids: torch.Tensor,
    face_size: int = 64, 
    fov: float = 95.0,
)-> torch.Tensor:
    """
    Combine encoding tensors and mask tensors into a single (B, T, 3, H, W) tensor.
    Channel 0-1: u_enc, v_enc, Channel 2: binary mask
    """
    enc_tensor = calculate_positional_encoding(batch_size, face_size, fov) # [B, T, 2, face_size, face_size]
    mask_tensor = calculate_mask_tensors(batch_size, drop_ids, face_size)  # [B, T, 1, face_size, face_size]
    extra = torch.cat([enc_tensor, mask_tensor], dim=2) # [B, T, 3, face_size, face_size]
    
    B, T, C, H, W = extra.shape
    return extra.reshape(B * T, C, H, W) # [B*T, 3, face_size, face_size]