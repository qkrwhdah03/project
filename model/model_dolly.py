import torch
import torch.nn as nn
from diffusers import ControlNetModel

class FourierDeltaEmbedder(nn.Module):
    def __init__(self, hidden_dim=64, out_channels=1):
        super().__init__()
        # Fourier Features: sin(2^k * PI * x), cos(...)
        self.freqs = [2**i for i in range(8)] 
        input_dim = len(self.freqs) * 2 + 1
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_channels)
        )

    def forward(self, delta, h, w):
        """
        delta: (B, 1) -> output: (B, out_channels, h, w)
        """
        B = delta.shape[0]
        feats = [delta]
        for freq in self.freqs:
            feats.append(torch.sin(delta * freq * torch.pi))
            feats.append(torch.cos(delta * freq * torch.pi))
        
        cat_feats = torch.cat(feats, dim=-1) # (B, input_dim)
        embed = self.mlp(cat_feats)          # (B, out_channels)
        
        # Spatial Broadcast
        return embed.unsqueeze(-1).unsqueeze(-1).expand(B, -1, h, w)

def make_custom_controlnet(base_model_id="diffusers/controlnet-depth-sdxl-1.0"):
    print(f"🔧 Loading & Modifying ControlNet from {base_model_id}...")
    controlnet = ControlNetModel.from_pretrained(base_model_id, torch_dtype=torch.float32)
    
    # Target: First Conv Layer
    # Standard path in Diffusers ControlNet
    conv_in = controlnet.controlnet_cond_embedding.conv_in
    
    old_weights = conv_in.weight.data
    old_bias = conv_in.bias.data
    
    # New Config: 3(RGB) + 1(Depth) + 1(Mask) + 1(Delta) = 6 Channels
    new_in_channels = 6
    new_conv = nn.Conv2d(
        new_in_channels, conv_in.out_channels,
        kernel_size=conv_in.kernel_size,
        stride=conv_in.stride,
        padding=conv_in.padding
    )
    
    # Weight Initialization (Zero-Convolution Strategy)
    # RGB 채널은 기존 가중치 유지, 나머지는 0으로 초기화 -> 학습 초기 안정성 확보
    with torch.no_grad():
        new_conv.weight.data[:, :3, :, :] = old_weights
        new_conv.weight.data[:, 3:, :, :] = torch.zeros_like(new_conv.weight.data[:, 3:, :, :])
        if old_bias is not None:
            new_conv.bias.data.copy_(old_bias)
        else:
            new_conv.bias = None
        
    controlnet.controlnet_cond_embedding.conv_in = new_conv
    print("✅ ControlNet is now 6-channel ready.")
    return controlnet