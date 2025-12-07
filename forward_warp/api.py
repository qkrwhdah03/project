"""
Public API for the depth → forward-warp → Img2Img refinement pipeline.

This module is designed to be imported from other projects or notebooks.
The main entrypoint is `forward_warp_image`, which takes an input image,
predicts its depth, warps it by the requested distance, optionally refines
the result with Stable Diffusion Img2Img, and returns a PIL image along
with useful intermediates.
"""
from __future__ import annotations

import os
import gc
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import (
    AutoPipelineForImage2Image,
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
)

from .dataset_tartanair2 import forward_warp as default_forward_warp

# Default configuration
DEFAULT_MODEL_ID = "Manojb/stable-diffusion-2-base"
DEFAULT_DEPTH_CHECKPOINT = "/root/project/forward_warp/depth_estimator/final_depth_estimator_ckpt.pth"
DEFAULT_IMG_SIZE: Tuple[int, int] = (512, 512)
DEFAULT_NEGATIVE_PROMPT = (
    "visible car hood, inside car POV, windshield frame, fisheye distortion, cartoon, "
    "anime, painting, render, distorted buildings, lowres, blurry, text overlays, "
    "deformed people, floating objects"
)
MAX_DEPTH = 50.0


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class SD2DepthEstimator(nn.Module):
    """Depth estimator adapted from Stable Diffusion 2 base."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        super().__init__()

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            load_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            load_dtype = torch.float16
        else:
            load_dtype = torch.float32

        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=load_dtype
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=load_dtype
        )
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

        self.vae.requires_grad_(False)
        self.vae.to(dtype=torch.bfloat16)
        self.unet.enable_gradient_checkpointing()

        old_conv = self.unet.conv_in
        new_conv = nn.Conv2d(
            8,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            padding=old_conv.padding,
        )

        with torch.no_grad():
            new_conv.weight[:, :4] = old_conv.weight
            new_conv.weight[:, 4:] = old_conv.weight * 0.1
            new_conv.bias = old_conv.bias
        self.unet.conv_in = new_conv

    @torch.no_grad()
    def encode_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb = rgb.to(dtype=self.vae.dtype)
        rgb = 2 * rgb - 1
        return self.vae.encode(rgb).latent_dist.mode() * 0.18215

    @torch.no_grad()
    def decode_depth(self, latent: torch.Tensor) -> torch.Tensor:
        latent = latent.to(dtype=self.vae.dtype)
        latent = latent / 0.18215
        out = self.vae.decode(latent).sample
        out = (out / 2 + 0.5).clamp(0, 1)
        return out.mean(1, keepdim=True) * MAX_DEPTH

    @torch.no_grad()
    def predict_depth(self, rgb_image: torch.Tensor) -> torch.Tensor:
        B = rgb_image.size(0)
        device = rgb_image.device

        rgb_latent = self.encode_rgb(rgb_image)
        latents = torch.randn_like(rgb_latent)

        scheduler = DDIMScheduler.from_pretrained(DEFAULT_MODEL_ID, subfolder="scheduler")
        scheduler.set_timesteps(50)

        encoder_hidden_states = torch.zeros(
            (B, 1, 1024),
            device=device,
            dtype=self.unet.dtype,
        )

        for t in scheduler.timesteps:
            x_in = torch.cat([latents, rgb_latent], dim=1)
            noise_pred = self.unet(
                x_in,
                t,
                encoder_hidden_states=encoder_hidden_states,
            ).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        depth = self.decode_depth(latents)
        return depth


def create_camera_intrinsics(img_size: Tuple[int, int] = DEFAULT_IMG_SIZE) -> torch.Tensor:
    """Create a simple pinhole intrinsics matrix scaled to the target size."""
    orig_w, orig_h = 640, 480
    fx, fy, cx, cy = 320.0, 320.0, 320.0, 240.0
    scale_x = img_size[1] / orig_w
    scale_y = img_size[0] / orig_h

    K = torch.eye(3)
    K[0, 0] = fx * scale_x
    K[1, 1] = fy * scale_y
    K[0, 2] = cx * scale_x
    K[1, 2] = cy * scale_y
    return K


def refine_image(
    pipe: AutoPipelineForImage2Image,
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    strength: float = 0.5,
) -> Image.Image:
    """Apply Img2Img refinement to a PIL Image."""
    result = pipe(
        prompt=prompt,
        image=image,
        negative_prompt=negative_prompt,
        strength=strength,
        guidance_scale=7.5,
        num_inference_steps=40,
    ).images[0]
    return result


_DEPTH_CACHE: Dict[str, SD2DepthEstimator] = {}
_IMG2IMG_CACHE: Dict[str, AutoPipelineForImage2Image] = {}


def _load_depth_model(
    device: str,
    checkpoint_path: str = DEFAULT_DEPTH_CHECKPOINT,
    model_id: str = DEFAULT_MODEL_ID,
) -> SD2DepthEstimator:
    depth_model = SD2DepthEstimator(model_id)
    depth_model = depth_model.to(device)
    depth_model.vae.to(device=device, dtype=torch.bfloat16)
    if torch.cuda.is_bf16_supported():
        depth_model.unet = depth_model.unet.to(dtype=torch.bfloat16)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["unet"] if isinstance(checkpoint, dict) and "unet" in checkpoint else checkpoint
    depth_model.unet.load_state_dict(state_dict, strict=False)
    depth_model.eval()
    return depth_model


def _load_img2img_pipe(device: str) -> AutoPipelineForImage2Image:
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    if device.startswith("cuda"):
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
    return pipe


def _get_depth_model(device: str, cache: bool = True) -> SD2DepthEstimator:
    if cache:
        if device not in _DEPTH_CACHE:
            _DEPTH_CACHE[device] = _load_depth_model(device)
        return _DEPTH_CACHE[device]
    # No-cache: load fresh
    return _load_depth_model(device)


def _get_img2img_pipe(device: str, cache: bool = True) -> AutoPipelineForImage2Image:
    if cache:
        if device not in _IMG2IMG_CACHE:
            _IMG2IMG_CACHE[device] = _load_img2img_pipe(device)
        return _IMG2IMG_CACHE[device]
    return _load_img2img_pipe(device)


def _load_image(
    target: Union[str, Image.Image, torch.Tensor],
    img_size: Tuple[int, int],
) -> Tuple[torch.Tensor, Image.Image]:
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ]
    )

    if isinstance(target, str):
        img = Image.open(target).convert("RGB")
    elif isinstance(target, Image.Image):
        img = target.convert("RGB")
    elif isinstance(target, torch.Tensor):
        if target.ndim == 3:
            return target.clamp(0, 1), transforms.ToPILImage()(target.clamp(0, 1))
        raise ValueError("Tensor target must have shape (3, H, W)")
    else:
        raise TypeError("target must be a path, PIL Image, or torch Tensor")

    img_tensor = transform(img)
    return img_tensor, img


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().float()
    tensor = tensor.permute(1, 2, 0)
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0, 1)
    return Image.fromarray((tensor.numpy() * 255).astype("uint8"))


def refine_depth_with_rgb(depth_tensor: torch.Tensor, rgb_tensor: torch.Tensor) -> torch.Tensor:
    """
    Guided Filter로 depth를 RGB 경계에 맞춰 선명하게 보정.
    depth_tensor: (1, H, W)
    rgb_tensor: (3, H, W), [-1, 1]
    cv2.ximgproc.guidedFilter가 없으면 원본을 그대로 반환.
    """
    try:
        import cv2  # type: ignore
    except Exception:
        return depth_tensor

    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "guidedFilter"):
        return depth_tensor

    device = depth_tensor.device
    depth_np = depth_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
    rgb_np = (rgb_tensor.permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5).astype(np.float32)

    scale = float(depth_np.max()) if depth_np.max() > 0 else 1.0
    depth_norm = depth_np / scale if scale > 0 else depth_np

    refined_norm = cv2.ximgproc.guidedFilter(guide=rgb_np, src=depth_norm, radius=4, eps=1e-6)
    refined = refined_norm * scale

    return torch.from_numpy(refined).unsqueeze(0).to(device=device, dtype=torch.float32)


def forward_warp_image(
    target: Union[str, Image.Image, torch.Tensor],
    warp_dist: float,
    prompt: str,
    *,
    negative_prompt: Optional[str] = None,
    strength: float = 0.15,
    device: Optional[str] = None,
    img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
    additional: Optional[Dict[str, Any]] = None,
    variant: str = "v2",
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    High-level helper that returns a refined warped image.

    Args:
        target: Path, PIL image, or tensor (3, H, W) in [0, 1].
        warp_dist: Forward dolly-in distance in meters (positive).
        prompt: Text prompt for Img2Img refinement.
        negative_prompt: Negative prompt for Img2Img (optional).
        strength: Img2Img strength (0-1). Default 0.3 for v1/v2.
        device: Torch device string; defaults to CUDA if available.
        img_size: Spatial resolution used for depth + warp.
        additional: Optional dict to override pieces of the pipeline:
            - "depth_model": preloaded depth model
            - "img2img_pipe": preloaded img2img pipeline
            - "forward_warp_fn": alternative forward_warp implementation
            - "camera_intrinsics": precomputed intrinsics (B,3,3)
            - "preprocess_fn": callable(img_tensor, img_pil, opts) -> (tensor, pil)
            - "postprocess_fn": callable(refined_pil, mask, opts) -> Image
            - any other keys are preserved in the returned metadata

    Returns:
        (refined_pil, meta) where meta contains depth, warp mask, and warped PIL.
    variant: "v1"은 기본 파이프라인,
             "v2"는 guided depth sharpening을 warp 전에 적용.
    """
    opts = additional or {}
    device = device or opts.get("device") or _default_device()
    negative_prompt = negative_prompt or opts.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT

    cache_models = opts.get("cache_models", False)
    cleanup = opts.get("cleanup", True)
    depth_model_provided = "depth_model" in opts
    depth_model: SD2DepthEstimator = opts.get("depth_model") or _get_depth_model(device, cache=cache_models)
    forward_warp_fn: Callable = opts.get("forward_warp_fn") or default_forward_warp

    img_tensor, img_pil = _load_image(target, img_size)
    if preprocess_fn := opts.get("preprocess_fn"):
        img_tensor, img_pil = preprocess_fn(img_tensor, img_pil, opts)

    img_batch = img_tensor.unsqueeze(0).to(device)
    depth_pred = depth_model.predict_depth(img_batch)

    if not depth_model_provided:
        # Sequential offload: free depth model before loading SDXL
        depth_model.to("cpu")
        del depth_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # v2: guided depth sharpening (or opts['guided_depth']=True)
    if variant == "v2" or opts.get("guided_depth"):
        depth_pred = refine_depth_with_rgb(depth_pred[0], (img_batch[0] * 0.5 + 0.5)).unsqueeze(0)

    rel_pose = torch.eye(4, device=device).unsqueeze(0)
    rel_pose[0, 2, 3] = -float(warp_dist)

    K = opts.get("camera_intrinsics")
    if K is None:
        K = create_camera_intrinsics(img_size).unsqueeze(0).to(device)
    elif K.dim() == 2:
        K = K.unsqueeze(0).to(device)
    else:
        K = K.to(device)

    warp_input = img_batch * 2.0 - 1.0
    warped_tensor, mask = forward_warp_fn(warp_input, depth_pred, K, rel_pose)
    warped_pil = _tensor_to_pil(warped_tensor.squeeze(0))

    img2img_pipe_provided = "img2img_pipe" in opts
    img2img_pipe: AutoPipelineForImage2Image = opts.get("img2img_pipe") or _get_img2img_pipe(device, cache=cache_models)
    if opts.get("offload_img2img") and hasattr(img2img_pipe, "enable_model_cpu_offload"):
        img2img_pipe.enable_model_cpu_offload()
    if device.startswith("cuda") and hasattr(img2img_pipe, "enable_vae_tiling"):
        img2img_pipe.enable_vae_tiling()

    refined_pil = refine_image(
        img2img_pipe,
        warped_pil,
        prompt,
        negative_prompt=negative_prompt,
        strength=strength,
    )

    if postprocess_fn := opts.get("postprocess_fn"):
        refined_pil = postprocess_fn(refined_pil, mask.squeeze(0), opts)

    meta = {
        "depth": depth_pred.detach().cpu(),
        "warp_mask": mask.detach().cpu(),
        "warped_pil": warped_pil,
        "device": device,
    }
    meta.update({k: v for k, v in opts.items() if k not in {"preprocess_fn", "postprocess_fn"}})

    # If no-cache or explicit cleanup was requested, drop models to free memory
    if cleanup or not cache_models:
        try:
            if not depth_model_provided:
                del depth_pred  # intermediate
        except Exception:
            pass
        try:
            if not img2img_pipe_provided:
                del img2img_pipe
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return refined_pil, meta

