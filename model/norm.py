# Code adapted from: https://github.com/Juan5713/OpenCubeDiff/

import torch.nn as nn

def patch_groupnorm(root: nn.Module, num_faces: int = 6) -> None:
    """Recursively replace GroupNorm with CubeDiffGroupNorm (in-place)."""
    for name, child in list(root.named_children()):
        patch_groupnorm(child, num_faces=num_faces)
        if isinstance(child, nn.GroupNorm) and not isinstance(child, CubeDiffGroupNorm):
            setattr(root, name, CubeDiffGroupNorm(child, num_faces=num_faces))


class CubeDiffGroupNorm(nn.Module):
    def __init__(self, original_norm: nn.GroupNorm, num_faces: int = 6):
        super().__init__()
        self.num_faces = num_faces

        # Make copy of original group normalization
        self.norm = nn.GroupNorm(
            num_groups=original_norm.num_groups,
            num_channels=original_norm.num_channels,
            eps=original_norm.eps,
            affine=original_norm.affine
        )

        if original_norm.affine:
            self.norm.weight.data.copy_(original_norm.weight.data)
            self.norm.bias.data.copy_(original_norm.bias.data)

    def forward(self, x):
        """
        x: torch.Tensor of shape (B*T, C, H, W) where T is number of faces
        """

        bt, C, H, W = x.shape
        T = self.num_faces
        B = bt // T
        assert bt == B * T, f"Input batch size {bt} is not divisible by num_faces {T}"

        # Reshape across cube faces
        x = x.reshape(B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4) # (B, C, T, H, W)
        x = x.reshape(B, C, T*H*W) # (B, C, T*H*W)

        # Apply GroupNorm across combined spatial area
        x = self.norm(x)

        # Reshape back
        x = x.reshape(B, C, T, H, W) # (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4) # (B, T, C, H, W)
        x = x.reshape(B*T, C, H, W) # (B*T, C, H, W)
        
        return x