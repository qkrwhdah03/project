# model/modules/group_norm.py
# GroupNorm modules for CubeDiff (Stable Diffusion 2)

# NOTE: Cited from https://github.com/Juan5713/OpenCubeDiff/ (Open source)

import torch.nn as nn
from einops import rearrange

class CubeDiffGroupNorm(nn.Module):
    def __init__(self, original_norm: nn.GroupNorm, num_faces: int = 6, sync_enabled: bool = True):
        super().__init__()
        self.num_faces = num_faces
        self.sync_enabled = sync_enabled

        # Create internal GroupNorm with same config
        self.norm = nn.GroupNorm(
            num_groups=original_norm.num_groups,
            num_channels=original_norm.num_channels,
            eps=original_norm.eps,
            affine=original_norm.affine
        )

        # Copy original weights/bias
        if original_norm.affine:
            self.norm.weight.data.copy_(original_norm.weight.data)
            self.norm.bias.data.copy_(original_norm.bias.data)

    def forward(self, x):
        """
        x: (B*T, C, H, W) or (B*T, C, H*W) → Cube-aware reshape → Apply shared GroupNorm → reshape back
        """

        if not self.sync_enabled or x.shape[0] < self.num_faces:
            # fallback: standard groupnorm without reshaping
            return self.norm(x)

        if len(x.shape) == 4: #(BT, C, H, W)
            bt, c, h, w = x.shape
            T = self.num_faces
            B = bt // T
            assert bt == B * T, f"Input batch size {bt} is not divisible by num_faces {T}"

            # Reshape across cube faces
            x = rearrange(x, "(b t) c h w -> b c (t h w)", b=B, t=T)

            # Apply GroupNorm across combined spatial area
            x = self.norm(x)

            # Reshape back
            x = rearrange(x, "b c (t h w) -> (b t) c h w", t=T, h=h, w=w)

        elif len(x.shape) == 3: # (BT, C, HW)
            bt, c, hw = x.shape
            T = self.num_faces
            B = bt // T
            assert bt == B * T, f"Input batch size {bt} is not divisible by num_faces {T}"

            # Reshape across cube faces
            x = rearrange(x, "(b t) c hw -> b c (t hw)", b=B, t=T)
            # Apply GroupNorm across combined spatial area
            x = self.norm(x)
            # Reshape back
            x = rearrange(x, "b c (t hw) -> (b t) c hw", t=T, hw=hw)
        
        return x

def patch_groupnorm(root: nn.Module, num_faces: int = 6) -> None:
    """
    Recursively replace GroupNorm with CubeDiffGroupNorm (Synchronized GroupNorm).
    """
    for name, child in list(root.named_children()):
        patch_groupnorm(child, num_faces=num_faces)
        if isinstance(child, nn.GroupNorm) and not isinstance(child, CubeDiffGroupNorm):
            setattr(root, name, CubeDiffGroupNorm(child, num_faces=num_faces))
