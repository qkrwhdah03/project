#!/usr/bin/env python3
"""
Depth estimation, forward warp, and Img2Img refinement
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import csv
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from diffusers import (
    AutoPipelineForImage2Image,
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler
)
from tqdm import tqdm

from dataset_tartanair2 import forward_warp

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Manojb/stable-diffusion-2-base"
CHECKPOINT_PATH = "/root/project/forward_warp/depth_estimator/final_depth_estimator_ckpt.pth"
MAX_DEPTH = 50.0
IMG_SIZE = (512, 512)
WARP_DISTANCES = [5.0, 10.0]  # Test both 5m and 10m

# Image paths
IMAGE_PATHS = [
    "/root/project/results/outputs/0_posx.png",
    "/root/project/results/outputs/1_posx.png",
    "/root/project/results/outputs/2_posx.png",
]

# Output directory
OUTPUT_DIR = "/root/project/forward_warp/depth_estimator/warped_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Strength values for Img2Img
STRENGTH_VALUES = [0.3, 0.5, 0.7]

# Negative prompt (shared)
NEGATIVE_PROMPT = "visible car hood, inside car POV, windshield frame, fisheye distortion, cartoon, anime, painting, render, distorted buildings, lowres, blurry, text overlays, deformed people, floating objects"

# Load prompts from CSV
PROMPT_CSV = os.path.join(OUTPUT_DIR, "prompt.csv")
PROMPTS = {}
if os.path.exists(PROMPT_CSV):
    with open(PROMPT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            PROMPTS[row['filename']] = row['prompt']
    print(f"Loaded prompts from {PROMPT_CSV}")
else:
    print(f"Warning: {PROMPT_CSV} not found, using default prompt")
    # Default prompt if CSV not found
    default_prompt = "Realistic street view, photorealistic road ahead, clean front view"
    for img_path in IMAGE_PATHS:
        filename = os.path.basename(img_path)
        PROMPTS[filename] = default_prompt


class SD2DepthEstimator(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

        self.vae.requires_grad_(False)
        self.vae.to(dtype=torch.bfloat16)
        self.unet.enable_gradient_checkpointing()

        old_conv = self.unet.conv_in
        new_conv = nn.Conv2d(8, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             padding=old_conv.padding)

        with torch.no_grad():
            new_conv.weight[:, :4] = old_conv.weight
            new_conv.weight[:, 4:] = old_conv.weight * 0.1
            new_conv.bias = old_conv.bias
        self.unet.conv_in = new_conv

    @torch.no_grad()
    def encode_rgb(self, rgb):
        rgb = rgb.to(dtype=self.vae.dtype)
        rgb = 2 * rgb - 1
        return self.vae.encode(rgb).latent_dist.mode() * 0.18215

    @torch.no_grad()
    def decode_depth(self, latent):
        latent = latent.to(dtype=self.vae.dtype)
        latent = latent / 0.18215
        out = self.vae.decode(latent).sample
        out = (out / 2 + 0.5).clamp(0, 1)
        return out.mean(1, keepdim=True) * MAX_DEPTH

    @torch.no_grad()
    def predict_depth(self, rgb_image):
        B = rgb_image.size(0)
        device = rgb_image.device
        
        rgb_latent = self.encode_rgb(rgb_image)
        latents = torch.randn_like(rgb_latent)
        
        scheduler = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
        scheduler.set_timesteps(50)
        
        encoder_hidden_states = torch.zeros(
            (B, 1, 1024),
            device=device,
            dtype=self.unet.dtype
        )
        
        for t in scheduler.timesteps:
            x_in = torch.cat([latents, rgb_latent], dim=1)
            noise_pred = self.unet(
                x_in, t,
                encoder_hidden_states=encoder_hidden_states
            ).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        depth = self.decode_depth(latents)
        return depth


def load_image(image_path, img_size=(512, 512)):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img)
    return img_tensor, img


def create_camera_intrinsics(img_size=(512, 512)):
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


def refine_image(pipe, image, prompt, negative_prompt, strength=0.5):
    """Apply Img2Img refinement to PIL Image."""
    result = pipe(
        prompt=prompt,
        image=image,
        negative_prompt=negative_prompt,
        strength=strength,
        guidance_scale=7.5,
        num_inference_steps=40
    ).images[0]
    return result


def main():
    print("=" * 70)
    print("Depth Estimation, Forward Warp, and Img2Img Refinement")
    print("=" * 70)
    
    # Load prompts from CSV
    global PROMPTS
    PROMPT_CSV = os.path.join(OUTPUT_DIR, "prompt.csv")
    if os.path.exists(PROMPT_CSV):
        with open(PROMPT_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                PROMPTS[row['filename']] = row['prompt']
        print(f"\nLoaded prompts from {PROMPT_CSV}")
        for filename, prompt in PROMPTS.items():
            print(f"  {filename}: {prompt[:60]}...")
    else:
        print(f"\nWarning: {PROMPT_CSV} not found, using default prompt")
        default_prompt = "Realistic street view, photorealistic road ahead, clean front view"
        for img_path in IMAGE_PATHS:
            filename = os.path.basename(img_path)
            PROMPTS[filename] = default_prompt
    
    # 1. Load depth estimation model
    print("\n[1/4] Loading depth estimation model...")
    depth_model = SD2DepthEstimator(MODEL_ID)
    depth_model = depth_model.to(DEVICE)
    depth_model.vae.to(device=DEVICE, dtype=torch.bfloat16)
    if torch.cuda.is_bf16_supported():
        depth_model.unet = depth_model.unet.to(dtype=torch.bfloat16)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    if isinstance(checkpoint, dict) and 'unet' in checkpoint:
        state_dict = checkpoint['unet']
    else:
        state_dict = checkpoint
    depth_model.unet.load_state_dict(state_dict, strict=False)
    depth_model.eval()
    print("  Depth model loaded")
    
    # 2. Load Img2Img pipeline
    print("\n[2/4] Loading SDXL Image2Image pipeline...")
    img2img_pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to(DEVICE)
    img2img_pipe.set_progress_bar_config(disable=False)
    print("  Img2Img pipeline loaded")
    
    # 3. Process each image
    print("\n[3/4] Processing images (depth estimation + forward warp)...")
    K = create_camera_intrinsics(IMG_SIZE).unsqueeze(0).to(DEVICE)
    
    warped_images = {}
    
    for img_path in IMAGE_PATHS:
        if not os.path.exists(img_path):
            print(f"  Warning: {img_path} not found")
            continue
        
        img_name = os.path.basename(img_path).replace('.png', '')
        print(f"\n  Processing: {img_name}")
        
        # Load image
        img_tensor, img_pil = load_image(img_path, IMG_SIZE)
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        
        # Predict depth
        print("    Predicting depth...")
        depth_pred = depth_model.predict_depth(img_tensor)
        
        # Forward warp for each distance
        warped_by_distance = {}
        for warp_dist in WARP_DISTANCES:
            print(f"    Applying {int(warp_dist)}m forward warp...")
            rel_pose = torch.eye(4, device=DEVICE).unsqueeze(0)
            rel_pose[0, 2, 3] = -warp_dist
            
            img_warp_input = img_tensor * 2.0 - 1.0
            warped_img, mask = forward_warp(img_warp_input, depth_pred, K, rel_pose)
            warped_img = warped_img.squeeze(0)
            
            # Convert to PIL Image
            warped_np = (warped_img.permute(1, 2, 0).cpu().float().numpy() + 1.0) / 2.0
            warped_np = np.clip(warped_np, 0, 1)
            warped_pil = Image.fromarray((warped_np * 255).astype(np.uint8))
            
            warped_by_distance[warp_dist] = warped_pil
            
            # Save warped image
            warped_path = os.path.join(OUTPUT_DIR, f"{img_name}_warped_{int(warp_dist)}m.png")
            warped_pil.save(warped_path)
            print(f"      Saved: {os.path.basename(warped_path)}")
        
        warped_images[img_name] = {
            'original': img_pil,
            'warped': warped_by_distance,
            'depth': depth_pred.squeeze(0).cpu().float()
        }
    
    # 4. Apply Img2Img refinement
    print("\n[4/4] Applying Img2Img refinement with different strength values...")
    
    all_results = {}
    
    for img_name, images in warped_images.items():
        print(f"\n  Refining: {img_name}")
        
        # Get prompt for this image
        filename = f"{img_name}.png"
        prompt = PROMPTS.get(filename, "Realistic street view, photorealistic road ahead")
        print(f"    Using prompt: {prompt[:80]}...")
        
        refined_by_distance = {}
        
        # Refine each warped distance
        for warp_dist in WARP_DISTANCES:
            print(f"    Distance: {int(warp_dist)}m")
            refined_results = {}
            
            for strength in STRENGTH_VALUES:
                print(f"      Strength={strength}...")
                refined_img = refine_image(
                    img2img_pipe,
                    images['warped'][warp_dist],
                    prompt,
                    NEGATIVE_PROMPT,
                    strength=strength
                )
                
                refined_results[strength] = refined_img
                
                # Save refined image
                refined_path = os.path.join(OUTPUT_DIR, f"{img_name}_refined_{int(warp_dist)}m_s{strength:.1f}.png")
                refined_img.save(refined_path)
                print(f"        Saved: {os.path.basename(refined_path)}")
            
            refined_by_distance[warp_dist] = refined_results
        
        all_results[img_name] = {
            'original': images['original'],
            'warped': images['warped'],
            'depth': images['depth'],
            'refined': refined_by_distance
        }
    
    # 5. Create comparison visualizations
    print("\n[5/5] Creating comparison visualizations...")
    
    for img_name, results in all_results.items():
        # Create comparison for each distance
        for warp_dist in WARP_DISTANCES:
            # Original, Depth, Warped, Refined (3 strengths)
            fig, axes = plt.subplots(1, len(STRENGTH_VALUES) + 3, figsize=(5 * (len(STRENGTH_VALUES) + 3), 5))
            
            # Original
            axes[0].imshow(results['original'])
            axes[0].set_title("Original", fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            # Depth map
            depth_np = results['depth'].squeeze().numpy()
            if depth_np.ndim != 2:
                depth_np = depth_np[0] if depth_np.ndim == 3 else depth_np
            magma = plt.get_cmap('magma')
            depth_vis = magma(np.clip(depth_np / MAX_DEPTH, 0, 1))[:, :, :3]
            axes[1].imshow(depth_vis)
            axes[1].set_title(f"Depth\n[{depth_np.min():.1f}-{depth_np.max():.1f}m]", fontsize=12, fontweight='bold')
            axes[1].axis('off')
            
            # Warped
            axes[2].imshow(results['warped'][warp_dist])
            axes[2].set_title(f"Warped\n{int(warp_dist)}m", fontsize=12, fontweight='bold')
            axes[2].axis('off')
            
            # Refined images
            for i, strength in enumerate(STRENGTH_VALUES):
                axes[i+3].imshow(results['refined'][warp_dist][strength])
                axes[i+3].set_title(f"Refined\ns={strength}", fontsize=12, fontweight='bold')
                axes[i+3].axis('off')
            
            plt.tight_layout()
            comparison_path = os.path.join(OUTPUT_DIR, f"{img_name}_comparison_{int(warp_dist)}m.png")
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {img_name}_comparison_{int(warp_dist)}m.png")
        
        # Create side-by-side comparison of 5m vs 10m
        fig, axes = plt.subplots(2, len(STRENGTH_VALUES) + 2, figsize=(5 * (len(STRENGTH_VALUES) + 2), 10))
        
        # Depth map (shared)
        depth_np = results['depth'].squeeze().numpy()
        if depth_np.ndim != 2:
            depth_np = depth_np[0] if depth_np.ndim == 3 else depth_np
        magma = plt.get_cmap('magma')
        depth_vis = magma(np.clip(depth_np / MAX_DEPTH, 0, 1))[:, :, :3]
        
        for row, warp_dist in enumerate(WARP_DISTANCES):
            # Warped
            axes[row, 0].imshow(results['warped'][warp_dist])
            axes[row, 0].set_title(f"Warped {int(warp_dist)}m", fontsize=12, fontweight='bold')
            axes[row, 0].axis('off')
            
            # Depth (only show once)
            if row == 0:
                axes[row, 1].imshow(depth_vis)
                axes[row, 1].set_title(f"Depth\n[{depth_np.min():.1f}-{depth_np.max():.1f}m]", fontsize=12, fontweight='bold')
                axes[row, 1].axis('off')
            else:
                axes[row, 1].axis('off')
            
            # Refined images
            for col, strength in enumerate(STRENGTH_VALUES):
                axes[row, col+2].imshow(results['refined'][warp_dist][strength])
                axes[row, col+2].set_title(f"Refined s={strength}", fontsize=12, fontweight='bold')
                axes[row, col+2].axis('off')
        
        plt.tight_layout()
        side_by_side_path = os.path.join(OUTPUT_DIR, f"{img_name}_5m_vs_10m.png")
        plt.savefig(side_by_side_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {img_name}_5m_vs_10m.png")
    
    print(f"\n✅ All results saved to: {OUTPUT_DIR}")
    print(f"   - Warped images: *_warped_10m.png")
    print(f"   - Refined images: *_refined_sX.X.png")
    print(f"   - Comparisons: *_comparison.png")


if __name__ == "__main__":
    main()
