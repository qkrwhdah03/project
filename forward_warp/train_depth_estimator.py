import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler

from dataset_tartanair2 import TartanAirDepthDataset  # Adjust import name if needed

# Configuration
MODEL_ID = "Manojb/stable-diffusion-2-base"
TARTANAIR_DATA_ROOT = "/root/project/data/tartanair2"
OUTPUT_DIR = "/root/project/forward_warp/depth_estimator"
BATCH_SIZE = 4
LR = 1e-5
EPOCHS = 20
SAVE_EVERY_N_EPOCHS = 5
MAX_DEPTH = 50.0
IMG_SIZE = (512, 512)


class SD2DepthEstimator(nn.Module):
    def __init__(self, model_id):
        super().__init__()

        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

        # Freeze VAE and convert to bfloat16 for memory efficiency
        self.vae.requires_grad_(False)
        self.vae.to(dtype=torch.bfloat16)
        
        # Enable gradient checkpointing for UNet
        self.unet.enable_gradient_checkpointing()

        # Modify input channels: [Depth(4) + RGB(4)] = 8ch
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
        """Encode RGB image to latent space."""
        rgb = rgb.to(dtype=self.vae.dtype)
        rgb = 2 * rgb - 1
        return self.vae.encode(rgb).latent_dist.mode() * 0.18215

    @torch.no_grad()
    def encode_depth(self, depth):
        """Encode depth image to latent space."""
        depth = depth.to(dtype=self.vae.dtype)
        depth3 = depth.repeat(1, 3, 1, 1)
        norm = 2 * (depth3 / MAX_DEPTH) - 1
        return self.vae.encode(norm).latent_dist.mode() * 0.18215

    @torch.no_grad()
    def decode_depth(self, latent):
        """Decode latent to depth image."""
        latent = latent.to(dtype=self.vae.dtype)
        latent = latent / 0.18215
        out = self.vae.decode(latent).sample
        out = (out / 2 + 0.5).clamp(0, 1)
        return out.mean(1, keepdim=True) * MAX_DEPTH

    def forward(self, batch):
        rgb = batch["source_rgb"]
        depth_gt = batch["source_depth"]
        B = rgb.size(0)
        device = rgb.device
        
        # Encode inputs
        with torch.no_grad():
            rgb_latent = self.encode_rgb(rgb)
            depth_latent = self.encode_depth(depth_gt)

        # Add noise (skip first 10% timesteps)
        timesteps = torch.randint(
            int(0.1 * self.scheduler.config.num_train_timesteps),
            self.scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        noise = torch.randn_like(depth_latent)
        noisy_latent = self.scheduler.add_noise(depth_latent, noise, timesteps)

        # Concatenate depth and RGB latents
        unet_in = torch.cat([noisy_latent, rgb_latent], dim=1)

        # Null embeddings
        encoder_hidden_states = torch.zeros(
            (B, 1, 1024), 
            device=device, 
            dtype=self.unet.dtype
        )

        # Predict noise
        noise_pred = self.unet(
            unet_in, timesteps,
            encoder_hidden_states=encoder_hidden_states
        ).sample

        return noise_pred, noise, depth_gt


@torch.no_grad()
def run_validation(model, val_loader, device, epoch, output_dir):
    """Run validation and save visualization."""
    model.eval()
    scheduler = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    scheduler.set_timesteps(50)

    batch = next(iter(val_loader))
    rgb = batch['source_rgb'].to(device)
    depth_gt = batch['source_depth'].to(device)
    B = rgb.size(0)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    with torch.autocast(device_type="cuda", dtype=dtype):
        rgb_latent = model.encode_rgb(rgb)
        latents = torch.randn_like(rgb_latent)
        hs = torch.zeros((B, 1, 1024), device=device, dtype=rgb_latent.dtype)

        for t in tqdm(scheduler.timesteps, desc="Sampling", ncols=80):
            x_in = torch.cat([latents, rgb_latent], dim=1)
            noise_pred = model.unet(x_in, t, encoder_hidden_states=hs).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        depth_pred = model.decode_depth(latents)

    save_combined_validation(
        rgb.float(), 
        depth_gt.float(), 
        depth_pred.float(), 
        epoch, 
        output_dir
    )

    model.train()


def save_combined_validation(rgb, gt, pred, epoch, out_dir):
    """Save validation visualization."""
    os.makedirs(out_dir, exist_ok=True)
    import matplotlib.cm as cm
    magma = cm.get_cmap('magma')

    rows = rgb.size(0)
    W = rgb.shape[3]
    H = rgb.shape[2]
    cell_h = H + 30
    canvas = Image.new("RGB", (3*W, rows*cell_h), "black")
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()

    for i in range(rows):
        rgb_np = (rgb[i].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
        rgb_img = Image.fromarray(rgb_np)
        canvas.paste(rgb_img, (0, i*cell_h+30))
        draw.text((W//2, i*cell_h+10), "RGB", fill="white", anchor="mm", font=font)

        gt_np = gt[i].squeeze().detach().cpu().numpy()
        gt_vis = (magma(np.clip(gt_np/MAX_DEPTH, 0, 1))[:,:,:3] * 255).astype(np.uint8)
        gt_img = Image.fromarray(gt_vis)
        canvas.paste(gt_img, (W, i*cell_h+30))
        draw.text((W+W//2, i*cell_h+10), f"GT [{gt_np.min():.1f}-{gt_np.max():.1f}]", fill="white", anchor="mm", font=font)

        pd_np = pred[i].squeeze().detach().cpu().numpy()
        pd_vis = (magma(np.clip(pd_np/MAX_DEPTH, 0, 1))[:,:,:3] * 255).astype(np.uint8)
        pd_img = Image.fromarray(pd_vis)
        canvas.paste(pd_img, (2*W, i*cell_h+30))
        draw.text((2*W+W//2, i*cell_h+10), f"Pred [{pd_np.min():.1f}-{pd_np.max():.1f}]", fill="white", anchor="mm", font=font)

    canvas.save(f"{out_dir}/val_ep{epoch}_combined.png", quality=95)
    print("Validation saved!")


def plot_loss_graph(losses, output_dir, current_epoch):
    """Plot and save loss graph."""
    os.makedirs(output_dir, exist_ok=True)
    
    epochs = list(range(1, len(losses) + 1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training Loss (Up to Epoch {current_epoch})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    if len(losses) >= 5:
        recent_epochs = epochs[-5:]
        recent_losses = losses[-5:]
        plt.plot(recent_epochs, recent_losses, 'r-', linewidth=3, alpha=0.5, label='Last 5 Epochs')
        plt.legend(fontsize=11)
    
    save_path = os.path.join(output_dir, "loss_graph.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    loss_file = os.path.join(output_dir, "loss_history.txt")
    with open(loss_file, 'w') as f:
        f.write("Epoch\tLoss\n")
        for ep, loss in zip(epochs, losses):
            f.write(f"{ep}\t{loss:.6f}\n")
    
    print(f"Loss graph saved: {save_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    log_file = os.path.join(OUTPUT_DIR, "training.log")
    log_f = open(log_file, 'w', buffering=1)
    
    def log_print(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=log_f)
        log_f.flush()

    log_print("=" * 60)
    log_print("Depth Estimator Training Started")
    log_print("=" * 60)
    log_print(f"  Device: {device}")
    log_print(f"  Batch Size: {BATCH_SIZE}")
    if torch.cuda.is_available():
        log_print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()
        initial_mem = torch.cuda.memory_allocated() / 1024**3
        log_print(f"  Initial VRAM: {initial_mem:.2f} GB")
    log_print("=" * 60)

    dataset = TartanAirDepthDataset(TARTANAIR_DATA_ROOT, img_size=IMG_SIZE)
    n_val = max(20, int(len(dataset)*0.03))
    n_train = len(dataset)-n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False, num_workers=2
    )

    model = SD2DepthEstimator(MODEL_ID)
    model = model.to(device)
    model.vae.to(device=device, dtype=torch.bfloat16)
    # UNet should also use bfloat16 for memory efficiency
    if torch.cuda.is_bf16_supported():
        model.unet = model.unet.to(dtype=torch.bfloat16)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = get_scheduler(
        "cosine", optimizer=optimizer, num_warmup_steps=0, 
        num_training_steps=len(train_loader)*EPOCHS
    )

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=90)

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            noise_pred, noise, depth_gt = model(batch)

            # Loss calculation in float32 for precision
            loss = F.mse_loss(noise_pred.float(), noise.float())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            
            # VRAM monitoring
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                max_allocated = torch.cuda.max_memory_allocated() / 1024**3
                
                if batch_idx == 0 or batch_idx % 10 == 0:
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        vram_alloc=f"{allocated:.2f}GB",
                        vram_max=f"{max_allocated:.2f}GB"
                    )

        avg_loss = total_loss / len(train_loader)
        
        if torch.cuda.is_available():
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            max_reserved = torch.cuda.max_memory_reserved() / 1024**3
            current_allocated = torch.cuda.memory_allocated() / 1024**3
            current_reserved = torch.cuda.memory_reserved() / 1024**3
            
            log_print(f"\nEpoch {epoch}/{EPOCHS} Completed. Avg Loss: {avg_loss:.5f}")
            log_print(f"  VRAM Usage:")
            log_print(f"     Current Allocated: {current_allocated:.2f} GB")
            log_print(f"     Current Reserved:  {current_reserved:.2f} GB")
            log_print(f"     MAX Allocated:     {max_allocated:.2f} GB")
            log_print(f"     MAX Reserved:      {max_reserved:.2f} GB")
        else:
            log_print(f"\nEpoch {epoch}/{EPOCHS} Completed. Avg Loss: {avg_loss:.5f}")

        # Validation (optional)
        # run_validation(model, val_loader, device, epoch, OUTPUT_DIR)

        if epoch % SAVE_EVERY_N_EPOCHS == 0:
            ckpt = os.path.join(OUTPUT_DIR, f"checkpoint_ep{epoch}.pth")
            # Convert bfloat16 to float32 for checkpoint compatibility
            unet_state = {k: v.cpu().float() if v.dtype == torch.bfloat16 else v.cpu()
                          for k, v in model.unet.state_dict().items()}
            torch.save(unet_state, ckpt)
            log_print(f"Checkpoint saved: {ckpt}")
    
    log_print("\nTraining completed!")
    log_f.close()


if __name__ == "__main__":
    main()
