#!/usr/bin/env python3
"""
Depth estimation (Marigold SD2.0) → guided depth sharpening → 5m forward warp → SDXL Img2Img refinement.
Single-pass (s=0.3/0.6) and multi-pass sequences, with grid export.
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusers import AutoPipelineForImage2Image, DDIMScheduler
from torchvision import transforms
from tqdm import tqdm

from dataset_tartanair2 import forward_warp

# ---------------------------------------------------------------------
# Imports for Marigold depth estimator (with fallbacks)
# ---------------------------------------------------------------------
try:
    from train_depth_marigold import SD2DepthEstimator  # type: ignore
except Exception:
    try:
        from train_depth_estimator import SD2DepthEstimator  # fallback
    except Exception:
        from api import SD2DepthEstimator  # last resort

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = {
    "0_posx": (
        "Remote small town in southern Utah or northern Arizona, cracked asphalt road with potholes stretching into the distance, "
        "red rock cliffs and mesas in the background, mobile homes and small single-story houses on the right, dusty dirt yards, "
        "a few green cottonwood trees, clear blue sky, harsh midday sunlight, dry desert landscape, realistic, photorealistic, "
        "Google Street View style, captured in the 2020s"
    ),
    "1_posx": (
        "Photorealistic view of a perfectly straight two-lane asphalt highway cutting through a barren desert landscape in northern Chile, "
        "high rocky mountains on both sides, extremely arid environment with no vegetation, bright blue cloudless sky, harsh midday sunlight, "
        "yellow center line and white edge lines, a single yellow road sign in the distance, power lines running parallel to the road on the left, "
        "Google Street View style"
    ),
    "2_posx": (
        "Photorealistic view of a perfectly straight two-lane of college campus road, a few students on both sides of the road, "
        "Google Street View style"
    ),
}

NEGATIVE_PROMPT = (
    "visible car hood, inside car POV, windshield frame, fisheye distortion, cartoon, anime, painting, render, "
    "distorted buildings, lowres, blurry, text overlays, deformed people, floating objects"
)

INPUT_IMAGES = [
    "/root/project/results/outputs/0_posx.png",
    "/root/project/results/outputs/1_posx.png",
    "/root/project/results/outputs/2_posx.png",
]
OUTPUT_DIR = "/root/project/forward_warp/depth_estimator/warped_results_guided"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_ID_SD2 = "Manojb/stable-diffusion-2-base"
DEPTH_CKPT = "/root/project/forward_warp/depth_estimator/final_depth_estimator_ckpt.pth"

WARP_DISTANCE = 5.0  # meters
IMG_SIZE = (512, 512)

SINGLE_STRENGTHS = [0.3, 0.6]
MULTI_SEQS = [
    [0.3, 0.3],
    [0.3, 0.5],
    [0.3, 0.7],
    [0.3, 0.3, 0.3],
    [0.3, 0.5, 0.7],
]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def refine_depth_with_rgb(depth_tensor: torch.Tensor, rgb_tensor: torch.Tensor) -> torch.Tensor:
    """
    Guided Filter로 depth를 RGB 경계에 맞춰 선명하게 보정.
    depth_tensor: (1, H, W)
    rgb_tensor: (3, H, W), [-1, 1]
    """
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


def load_depth_model() -> SD2DepthEstimator:
    model = SD2DepthEstimator(MODEL_ID_SD2)
    ckpt = torch.load(DEPTH_CKPT, map_location=DEVICE)
    if isinstance(ckpt, dict) and "unet" in ckpt:
        model.unet.load_state_dict(ckpt["unet"])
    elif isinstance(ckpt, dict) and "unet_state_dict" in ckpt:
        model.unet.load_state_dict(ckpt["unet_state_dict"])
    else:
        model.unet.load_state_dict(ckpt)
    model.vae.to(DEVICE)
    model.unet.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def estimate_depth(model: SD2DepthEstimator, image_pil: Image.Image) -> np.ndarray:
    transform = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor()])
    rgb = transform(image_pil).unsqueeze(0).to(DEVICE)

    scheduler = DDIMScheduler.from_pretrained(MODEL_ID_SD2, subfolder="scheduler")
    scheduler.set_timesteps(50)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    with torch.autocast(device_type="cuda", dtype=dtype) if DEVICE.startswith("cuda") else torch.no_grad():
        rgb_latent = model.encode_rgb(rgb)
        latents = torch.randn_like(rgb_latent)
        hs = torch.zeros((1, 1, 1024), device=DEVICE, dtype=rgb_latent.dtype)
        for t in scheduler.timesteps:
            x_in = torch.cat([latents, rgb_latent], dim=1)
            noise_pred = model.unet(x_in, t, encoder_hidden_states=hs).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        depth = model.decode_depth(latents)
        depth_np = depth.squeeze().float().cpu().numpy()
    return depth_np


def warp_image(image_pil: Image.Image, depth_map: np.ndarray, dist_z: float = WARP_DISTANCE):
    w, h = image_pil.size
    img_tensor = transforms.ToTensor()(image_pil).unsqueeze(0).to(DEVICE)
    depth_tensor = torch.from_numpy(depth_map).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    if depth_tensor.shape[2:] != (h, w):
        depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=(h, w), mode="nearest")

    # Guided filter depth sharpening
    depth_tensor = refine_depth_with_rgb(depth_tensor[0], img_tensor[0]).unsqueeze(0)

    # Camera intrinsics scaled to image size
    orig_w, orig_h = 640, 480
    fx, fy, cx, cy = 320.0, 320.0, 320.0, 240.0
    scale_x, scale_y = w / orig_w, h / orig_h
    K = torch.eye(3, device=DEVICE, dtype=torch.float32).unsqueeze(0)
    K[0, 0, 0], K[0, 1, 1] = fx * scale_x, fy * scale_y
    K[0, 0, 2], K[0, 1, 2] = cx * scale_x, cy * scale_y

    rel_pose = torch.eye(4, device=DEVICE, dtype=torch.float32).unsqueeze(0)
    rel_pose[0, 2, 3] = -dist_z

    img_norm = img_tensor * 2.0 - 1.0  # [-1,1]
    warped, mask = forward_warp(img_norm, depth_tensor, K, rel_pose)
    warped_01 = torch.clamp((warped + 1.0) / 2.0, 0.0, 1.0)
    warped_pil = transforms.ToPILImage()(warped_01.squeeze(0))
    return warped_pil, mask


def refine_single(pipe: AutoPipelineForImage2Image, image_pil: Image.Image, strength: float, prompt: str):
    return pipe(
        prompt=prompt,
        image=image_pil,
        negative_prompt=NEGATIVE_PROMPT,
        strength=strength,
        guidance_scale=7.5,
        num_inference_steps=40,
    ).images[0]


def refine_sequence(pipe: AutoPipelineForImage2Image, image_pil: Image.Image, seq, prompt: str):
    current = image_pil
    for s in seq:
        current = refine_single(pipe, current, s, prompt)
    return current


def make_grid(base_title, warped_img, single_results, multi_results, save_path):
    cols = 1 + len(single_results) + len(multi_results)
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
    axes = axes.flatten()

    axes[0].imshow(warped_img)
    axes[0].set_title(f"Warped {int(WARP_DISTANCE)}m\n{base_title}")
    axes[0].axis("off")

    col = 1
    for s in SINGLE_STRENGTHS:
        axes[col].imshow(single_results[s])
        axes[col].set_title(f"s={s}")
        axes[col].axis("off")
        col += 1

    for seq in MULTI_SEQS:
        label = "->".join([f"{x:.1f}" for x in seq])
        axes[col].imshow(multi_results[label])
        axes[col].set_title(label)
        axes[col].axis("off")
        col += 1

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved grid: {save_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("✨ Guided Depth + Warp + SDXL Img2Img (single & multi) [v2]")
    print("=" * 70)

    # Load Img2Img pipeline (SDXL)
    print("\n📥 Loading SDXL Image2Image pipeline...")
    dtype = torch.float16 if DEVICE.startswith("cuda") else torch.float32
    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=False)
    print("✅ SDXL Image2Image loaded")

    # Load depth estimator
    print("\n📥 Loading Marigold depth estimator (SD2.0)...")
    depth_model = load_depth_model()
    print("✅ Depth model loaded")

    for img_path in tqdm(INPUT_IMAGES, desc="Images"):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        prompt = PROMPTS.get(base_name, "")
        print(f"\n🖼️ Processing {base_name} (warp {WARP_DISTANCE}m)")

        image = Image.open(img_path).convert("RGB")

        # 1) Depth estimation
        depth_map = estimate_depth(depth_model, image)

        # 2) Warp (guided depth + hole filling)
        warped_img, _ = warp_image(image, depth_map, dist_z=WARP_DISTANCE)
        warped_path = os.path.join(OUTPUT_DIR, f"{base_name}_warped_{int(WARP_DISTANCE)}m.png")
        warped_img.save(warped_path)

        # 3) Single-pass refinements
        single_results = {}
        for s in SINGLE_STRENGTHS:
            out = refine_single(pipe, warped_img, s, prompt)
            save_path = os.path.join(OUTPUT_DIR, f"{base_name}_s{s:.1f}.png")
            out.save(save_path)
            single_results[s] = out
            print(f"  ✅ saved {os.path.basename(save_path)}")

        # 4) Multi-pass refinements
        multi_results = {}
        for seq in MULTI_SEQS:
            final_img = refine_sequence(pipe, warped_img, seq, prompt)
            label = "->".join([f"{x:.1f}" for x in seq])
            save_path = os.path.join(OUTPUT_DIR, f"{base_name}_seq_{label.replace('.', '')}.png")
            final_img.save(save_path)
            multi_results[label] = final_img
            print(f"  ✅ saved {os.path.basename(save_path)}")

        # 5) Grid export
        grid_path = os.path.join(OUTPUT_DIR, f"{base_name}_grid.png")
        make_grid(base_name, warped_img, single_results, multi_results, grid_path)

    print("\n✅ All done. Results saved to:")
    print(f"   {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

