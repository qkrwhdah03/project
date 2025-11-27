import os
# 메모리 파편화 방지를 위한 환경 변수 설정 (import torch 이전에 설정하는 것이 좋습니다)
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from PIL import Image
import numpy as np
import sys
import bitsandbytes as bnb  # [변경] 8-bit optimizer를 위해 추가
import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_dolly import ScanNetDataset
from model.model_dolly import make_custom_controlnet, FourierDeltaEmbedder

# --- CONFIG ---
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DATA_DIR = "./scannet_sample/scene0000_00"
BATCH_SIZE = 3
LR = 5e-6  # Reduced from 1e-5 to prevent gradient explosion
EPOCHS = 40
ACCUMULATION_STEPS = 1
OUTPUT_DIR = "./output"
SAVE_IMAGES_EVERY = 0
MAX_GRAD_NORM = 1.0  # Gradient clipping to prevent explosion
MAX_STEPS = 5000  # Stop training after this many steps
SAVE_IMAGES_EVERY_N_STEPS = 1000  # Generate images every N steps

device = torch.device("cuda")

# --------------------------- 샘플 생성 함수 ---------------------------
def generate_image_sample(
    vae, unet, controlnet, delta_embedder,
    tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2,
    noise_scheduler, control_cond, prompt, device,
    num_inference_steps=10
):
    torch_dtype = torch.float16

    with torch.no_grad():
        # Text
        t1 = tokenizer_1([prompt], padding="max_length", max_length=77,
                         truncation=True, return_tensors="pt").to(device)

        t2 = tokenizer_2([prompt], padding="max_length", max_length=77,
                         truncation=True, return_tensors="pt").to(device)

        emb1 = text_encoder_1(t1.input_ids).last_hidden_state.to(torch_dtype)
        emb2_out = text_encoder_2(t2.input_ids)
        emb2 = emb2_out.last_hidden_state.to(torch_dtype)
        pooled_emb = emb2_out.text_embeds.to(torch_dtype)
        prompt_embeds = torch.cat([emb1, emb2], dim=-1).to(torch_dtype)

        add_time_ids = torch.tensor(
            [512., 512., 0., 0., 512., 512.], device=device, dtype=torch_dtype
        ).unsqueeze(0)

        latents = torch.randn((1, 4, 64, 64), device=device, dtype=torch_dtype)
        latents *= getattr(noise_scheduler, "init_noise_sigma", 1.0)

        timesteps = noise_scheduler.timesteps[:num_inference_steps]

        for t in timesteps:
            timestep = torch.tensor([t], device=device, dtype=torch_dtype)
            control_cond_fp16 = control_cond.to(torch_dtype)

            down, mid = controlnet(
                sample=latents,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=control_cond_fp16,
                added_cond_kwargs={"text_embeds": pooled_emb, "time_ids": add_time_ids},
                return_dict=False,
            )
            down = [d.to(torch_dtype) for d in down]
            mid = mid.to(torch_dtype)

            noise_pred = unet(
                latents,
                timestep,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down,
                mid_block_additional_residual=mid,
                added_cond_kwargs={"text_embeds": pooled_emb, "time_ids": add_time_ids},
            ).sample.to(torch_dtype)

            latents = noise_scheduler.step(
                noise_pred, t, latents
            ).prev_sample.to(torch_dtype)

        # VAE decode를 위해 float16으로 변환 (VAE가 float16이므로 입력도 float16)
        latents = latents / vae.config.scaling_factor
        latents = latents.to(torch.float16)
        
        # VAE decode
        image = vae.decode(latents).sample
        
        # 결과를 float32로 변환하여 후처리
        image = image.to(torch.float32)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
        return image[0]

# --------------------------- 이미지 저장 함수 ---------------------------
def save_comparison_image(generated, gt_pixels, warped_img, control_cond, epoch, step, suffix=""):
    """생성된 이미지, GT, 입력을 한눈에 볼 수 있게 저장"""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # GUI 없이 사용
    
    # 텐서를 numpy로 변환
    if isinstance(generated, torch.Tensor):
        generated = generated.cpu().numpy()
    if isinstance(gt_pixels, torch.Tensor):
        gt_pixels = gt_pixels.cpu().permute(1, 2, 0).numpy()
        gt_pixels = (gt_pixels * 255).astype(np.uint8)
    if isinstance(warped_img, torch.Tensor):
        warped_img = warped_img.cpu().permute(1, 2, 0).numpy()
        warped_img = (warped_img * 255).astype(np.uint8)
    
    # Control condition의 RGB 부분만 추출 (첫 3채널)
    if isinstance(control_cond, torch.Tensor):
        control_rgb = control_cond[:3].cpu().permute(1, 2, 0).numpy()
        control_rgb = (control_rgb * 255).astype(np.uint8)
        control_rgb = np.clip(control_rgb, 0, 255)
    else:
        control_rgb = warped_img
    
    # Figure 생성 (2x2 그리드)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f'Epoch {epoch} - Step {step} - {suffix}', fontsize=16, fontweight='bold')
    
    # 1. Generated Image (모델 출력)
    axes[0, 0].imshow(generated)
    axes[0, 0].set_title('Generated (Model Output)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. Ground Truth (목표 이미지)
    axes[0, 1].imshow(gt_pixels)
    axes[0, 1].set_title('Ground Truth (Target)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. Warped Image (입력 - warped)
    axes[1, 0].imshow(warped_img)
    axes[1, 0].set_title('Input: Warped Image', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 4. Control Condition RGB (입력 - control condition의 RGB 부분)
    axes[1, 1].imshow(control_rgb)
    axes[1, 1].set_title('Input: Control Condition (RGB)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # 저장
    save_path = os.path.join(OUTPUT_DIR, "samples", f"epoch_{epoch}_step_{step}_{suffix}_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 개별 이미지도 저장 (필요시)
    Image.fromarray(generated).save(os.path.join(OUTPUT_DIR, "samples", f"epoch_{epoch}_step_{step}_{suffix}_generated.png"))
    Image.fromarray(gt_pixels).save(os.path.join(OUTPUT_DIR, "samples", f"epoch_{epoch}_step_{step}_{suffix}_gt.png"))
    Image.fromarray(warped_img).save(os.path.join(OUTPUT_DIR, "samples", f"epoch_{epoch}_step_{step}_{suffix}_warped.png"))

# --------------------------- main ---------------------------
def print_memory(step_name):
    """메모리 사용량 출력"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        # max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  [{step_name}] GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    else:
        print(f"  [{step_name}] CUDA not available")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "samples"), exist_ok=True)

    print("=" * 60)
    print("🚀 Training Initialization - Memory Optimized Version")
    print("=" * 60)
    print_memory("Initial")

    # 1. VAE
    print("\n📥 Step 1: Loading VAE...")
    try:
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        ).to(device)
        vae.enable_slicing()
        vae.enable_tiling() # [권장] 메모리 추가 절약
        vae.requires_grad_(False)
        print("✅ VAE loaded successfully")
        print_memory("VAE Loaded")
    except Exception as e:
        print(f"❌ VAE loading failed: {e}")
        raise

    # 2. UNet
    print("\n📥 Step 2: Loading UNet...")
    try:
        unet = UNet2DConditionModel.from_pretrained(
            MODEL_ID, subfolder="unet", torch_dtype=torch.float16
        ).to(device)
        unet.requires_grad_(False)
        # UNet도 Gradient Checkpointing 켜면 좋지만, Freeze 상태라 불필요
        print("✅ UNet loaded successfully")
        print_memory("UNet Loaded")
    except Exception as e:
        print(f"❌ UNet loading failed: {e}")
        raise

    # 3. Text Encoders
    print("\n📥 Step 3: Loading tokenizers and text encoders...")
    try:
        tokenizer_1 = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer", use_fast=False)
        tokenizer_2 = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer_2", use_fast=False)
        
        text_encoder_1 = CLIPTextModel.from_pretrained(
            MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float16
        ).to(device)
        
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            MODEL_ID, subfolder="text_encoder_2", torch_dtype=torch.float16
        ).to(device)
        
        text_encoder_1.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        print("✅ Text encoders loaded successfully")
        print_memory("Text Encoders Loaded")
    except Exception as e:
        print(f"❌ Text encoder loading failed: {e}")
        raise

    # 4. ControlNet & DeltaEmbedder
    print("\n📥 Step 4: Loading ControlNet and DeltaEmbedder...")
    try:
        controlnet = make_custom_controlnet().to(device)
        
        # [핵심 변경 1] Gradient Checkpointing 활성화! (메모리 30~50% 절약)
        controlnet.enable_gradient_checkpointing()
        print("  --> ControlNet Gradient Checkpointing Enabled ✅")

        # FP16 변환 (Training 대상이므로 주의 필요하지만, 기존 코드 유지)
        controlnet = controlnet.to(torch.float16)
        controlnet.train()

        delta_embedder = FourierDeltaEmbedder(out_channels=1).to(device)
        delta_embedder.train()
        
        print("✅ ControlNet and DeltaEmbedder loaded successfully")
        print_memory("ControlNet Setup")
    except Exception as e:
        print(f"❌ ControlNet/DeltaEmbedder loading failed: {e}")
        raise

    # 5. Optimizer & Data
    print("\n📥 Step 5: Setting up optimizer and data...")
    try:
        # [핵심 변경 2] 8-bit AdamW 사용 (Optimizer State 메모리 70% 절약)
        optimizer = bnb.optim.AdamW8bit(
            list(controlnet.parameters()) + list(delta_embedder.parameters()),
            lr=LR
        )
        print("  --> 8-bit AdamW Optimizer Initialized ✅")
        
        dataset = ScanNetDataset(DATA_DIR, img_size=(512, 512))
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        
        noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
        print("✅ Setup complete")
        print_memory("Setup Final")
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        raise

    print("=" * 60)
    print("🚀 Training Start")
    print("=" * 60)

    global_step = 0
    
    for epoch in range(EPOCHS):
        print(f"\n📊 Epoch {epoch} starting...")
        print_memory(f"Epoch {epoch} start")
        
        # 에폭 시작 시 샘플 이미지 생성 (baseline)
        if epoch == 0:
            print("  Generating baseline sample...")
            try:
                with torch.no_grad():
                    # 첫 번째 배치로 샘플 생성
                    sample_batch = next(iter(dataloader))
                    sample_pixels = sample_batch["pixel_values"][:1].to(device)
                    sample_warped = sample_batch["warped_image"][:1].to(device)
                    sample_depth = sample_batch["depth"][:1].to(device)
                    sample_mask = sample_batch["mask"][:1].to(device)
                    sample_delta = sample_batch["delta"][:1].to(device)
                    sample_prompt = sample_batch["prompt"][0] if isinstance(sample_batch["prompt"], list) else sample_batch["prompt"]
                    
                    sample_delta_map = delta_embedder(sample_delta, 512, 512)
                    sample_control_cond = torch.cat([sample_warped, sample_depth, sample_mask, sample_delta_map], dim=1).to(torch.float16)
                    
                    generated = generate_image_sample(
                        vae, unet, controlnet, delta_embedder,
                        tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2,
                        noise_scheduler, sample_control_cond, sample_prompt, device,
                        num_inference_steps=20
                    )

                    # main 함수 내 이미지 저장 부분 (save_comparison_image 전)
                    # image는 [-1, 1] 범위이므로 [0, 1]로 변환
                    image_vis = (generated / 2 + 0.5).clamp(0, 1)                    
                    
                    # 이미지 저장
                    save_comparison_image(
                        image_vis, sample_pixels[0], sample_warped[0], 
                        sample_control_cond[0], epoch, 0, "baseline"
                    )
                    print("  ✅ Baseline sample saved")
            except Exception as e:
                print(f"  ⚠️ Baseline generation failed: {e}")
        
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            
            # --- Data Loading ---
            pixels = batch["pixel_values"].to(device)
            warped  = batch["warped_image"].to(device)
            mask    = batch["mask"].to(device)
            depth   = batch["depth"].to(device)
            delta   = batch["delta"].to(device)
            prompts = batch["prompt"] if isinstance(batch["prompt"], list) else [batch["prompt"]]
            
            # --- VAE Encode ---
            with torch.no_grad():
                latents = vae.encode(pixels.to(torch.float16)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # --- Add Noise ---
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (pixels.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # --- Text Encode ---
            with torch.no_grad():
                t1 = tokenizer_1(prompts, padding="max_length", max_length=77,
                                 truncation=True, return_tensors="pt").to(device)
                t2 = tokenizer_2(prompts, padding="max_length", max_length=77,
                                 truncation=True, return_tensors="pt").to(device)

                emb1 = text_encoder_1(t1.input_ids).last_hidden_state
                emb2_out = text_encoder_2(t2.input_ids)
                pooled_emb = emb2_out.text_embeds
                emb2 = emb2_out.last_hidden_state

                prompt_embeds = torch.cat([emb1, emb2], dim=-1)
                
                # Time IDs (SDXL specific)
                add_time_ids = torch.tensor(
                    [512., 512., 0., 0., 512., 512.], device=device
                ).repeat(pixels.shape[0], 1)

            # --- Control Condition ---
            delta_map = delta_embedder(delta, 512, 512)
            # Ensure dimensions match before concat if necessary
            control_cond = torch.cat([warped, depth, mask, delta_map], dim=1).to(torch.float16)
            
            # Require grad for gradient checkpointing to work properly
            noisy_latents.requires_grad_(True) 

            # --- ControlNet Forward ---
            down, mid = controlnet(
                sample=noisy_latents.to(torch.float16),
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds.to(torch.float16),
                controlnet_cond=control_cond,
                added_cond_kwargs={"text_embeds": pooled_emb.to(torch.float16),
                                   "time_ids": add_time_ids},
                return_dict=False,
            )

            # --- UNet Forward ---
            pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down,
                mid_block_additional_residual=mid,
                added_cond_kwargs={"text_embeds": pooled_emb, "time_ids": add_time_ids},
            ).sample

            # --- Loss Calculation ---
            loss = F.mse_loss(pred.float(), noise.float()) # Ensure float32 for stable loss
            loss = loss / ACCUMULATION_STEPS
            
            # Check for NaN/Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  WARNING: NaN/Inf loss detected at step {global_step}! Skipping this batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()

            # --- Optimizer Step (Gradient Accumulation) ---
            if (step + 1) % ACCUMULATION_STEPS == 0:
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(controlnet.parameters(), MAX_GRAD_NORM)
                
                optimizer.step()
                
                # [핵심 변경 3] set_to_none=True로 메모리 절약
                optimizer.zero_grad(set_to_none=True)
                
                # 캐시 정리를 통해 파편화 완화 (필요시 주석 처리 가능)
                torch.cuda.empty_cache()

            if global_step % 100 == 0:
                print(f"step [{global_step}] loss: {loss.item() * ACCUMULATION_STEPS:.6f}")
                print_memory("Step End")
            
            # 중간에 이미지 생성 (매 N 스텝마다)
            if global_step > 0 and global_step % SAVE_IMAGES_EVERY_N_STEPS == 0:
                print(f"\n📸 Generating sample images at step {global_step}...")
                try:
                    with torch.no_grad():
                        sample_batch = next(iter(dataloader))
                        if sample_batch:
                            sample_pixels = sample_batch["pixel_values"][:1].to(device)
                            sample_warped = sample_batch["warped_image"][:1].to(device)
                            sample_depth = sample_batch["depth"][:1].to(device)
                            sample_mask = sample_batch["mask"][:1].to(device)
                            sample_delta = sample_batch["delta"][:1].to(device)
                            sample_prompt = sample_batch["prompt"][0] if isinstance(sample_batch["prompt"], list) else sample_batch["prompt"]
                            
                            sample_delta_map = delta_embedder(sample_delta, 512, 512)
                            sample_control_cond = torch.cat([sample_warped, sample_depth, sample_mask, sample_delta_map], dim=1).to(torch.float16)
                            
                            generated = generate_image_sample(
                                vae, unet, controlnet, delta_embedder,
                                tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2,
                                noise_scheduler, sample_control_cond, sample_prompt, device,
                                num_inference_steps=10
                            )
                            
                            save_comparison_image(
                                generated, sample_pixels[0], sample_warped[0],
                                sample_control_cond[0], epoch, global_step, f"step_{global_step}"
                            )
                            print(f"  ✅ Step {global_step} sample saved")
                            print_memory("After step sample generation")
                except Exception as e:
                    print(f"  ⚠️ Step {global_step} sample generation failed: {e}")

            global_step += 1
            
            # 5000 스텝에 도달하면 중단
            if global_step >= MAX_STEPS:
                print(f"\n✅ Reached {MAX_STEPS} steps. Stopping training.")
                break
        
        # 에폭 끝에 샘플 이미지 생성 (5000 스텝에 도달하지 않은 경우에만)
        if global_step < MAX_STEPS:
            print(f"\n📸 Generating sample images for Epoch {epoch} end...")
            try:
                with torch.no_grad():
                    # 새로운 배치로 샘플 생성 (현재 배치는 이미 사용됨)
                    sample_batch = next(iter(dataloader))
                    if not sample_batch:
                        print("  ⚠️ Empty batch, skipping epoch end sample")
                    else:
                        sample_pixels = sample_batch["pixel_values"][:1].to(device)
                        sample_warped = sample_batch["warped_image"][:1].to(device)
                        sample_depth = sample_batch["depth"][:1].to(device)
                        sample_mask = sample_batch["mask"][:1].to(device)
                        sample_delta = sample_batch["delta"][:1].to(device)
                        sample_prompt = sample_batch["prompt"][0] if isinstance(sample_batch["prompt"], list) else sample_batch["prompt"]
                        
                        sample_delta_map = delta_embedder(sample_delta, 512, 512)
                        sample_control_cond = torch.cat([sample_warped, sample_depth, sample_mask, sample_delta_map], dim=1).to(torch.float16)
                        
                        generated = generate_image_sample(
                            vae, unet, controlnet, delta_embedder,
                            tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2,
                            noise_scheduler, sample_control_cond, sample_prompt, device,
                            num_inference_steps=10
                        )
                        
                        # 비교 이미지 저장
                        save_comparison_image(
                            generated, sample_pixels[0], sample_warped[0],
                            sample_control_cond[0], epoch, global_step, "epoch_end"
                        )
                        print(f"  ✅ Epoch {epoch} sample saved")
                        print_memory("After sample generation")
            except Exception as e:
                print(f"  ⚠️ Sample generation failed: {e}")
        
        # 5000 스텝에 도달했으면 루프 종료
        if global_step >= MAX_STEPS:
            break
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()