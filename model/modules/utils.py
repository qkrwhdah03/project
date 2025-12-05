# model/modules/utils.py
# Utility functions for CubeDiff (Stable Diffusion 2)

# NOTE: Cited from https://github.com/Juan5713/OpenCubeDiff/ (Open source)

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from model.modules.attention import swap_transformer_blocks

def freeze(module: nn.Module) -> None:
    """
    Freeze all parameters in a module so they do not require gradients
    """
    for p in module.parameters():
        p.requires_grad = False

def expand_input_conv(unet: UNet2DConditionModel, new_channels: int) -> None:
    """
    Grow `conv_in` to `new_channels`, copying the first 4 kernels
    """
    old = unet.conv_in
    if old.in_channels == new_channels:
        return
    
    new = nn.Conv2d(
        new_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    )
    
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, : old.in_channels] = old.weight
        if old.bias is not None:
            new.bias.copy_(old.bias)
    
    unet.conv_in = new

def patch_unet(unet: UNet2DConditionModel, in_channels: int = 7) -> UNet2DConditionModel:
    """
    Patch a base UNet to CubeDiff architecture
    """
    # Swap transformer blocks (Apply cross-view attention)
    swap_transformer_blocks(unet)
    # Expand input conv layer (Add extra channels for additional conditions)
    expand_input_conv(unet, in_channels)
    # Update config
    unet.register_to_config(in_channels=in_channels)
    return unet
