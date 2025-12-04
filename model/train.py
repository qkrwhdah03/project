# model/train.py
# Training code for CubeDiff (Stable Diffusion 2)

# NOTE: Cited from https://github.com/Juan5713/OpenCubeDiff/ (Open source)

import os
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from diffusers import DDIMScheduler
from tqdm import tqdm
from matplotlib import pyplot as plt

from modules.dataset import CubemapDataset
from pipeline import SD2CubeDiffPipeline
from modules.additional_channels import make_extra_channels_tensor

def plot_train_loss(train_losses, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Iterations")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return 

class Config:
    """
    Configuration class for training
    """
    def __init__(self):
        # Model and data paths
        self.model_id = "Manojb/stable-diffusion-2-base"
        self.data_dir = "/root/project/data/cubemap"
        
        timestamp = datetime.now().strftime("%m-%d-%H%M%S")
        self.results_dir = f"/root/project/results/{timestamp}"
        self.checkpoints_dir = self.results_dir
        
        self.resume_checkpoint = None

        # Training settings
        self.image_size = 512  # Reduced from 512 for faster training (latent: 32x32)
        self.fov = 95
        self.batch_size = 1   # Increased with smaller image size
        self.num_workers = 4
        self.epochs = 10
        self.learning_rate = 2e-4
        self.prediction_type = "v_prediction" # or "epsilon"
        self.seed = 42

        # Logging & saving
        self.log_interval = 10
        self.save_interval_epoch = 5

        # Etc.
        self.dtype = "float16"

    def to_dict(self):
        """
        Convert all config attributes to a serializable dictionary.
        """
        out = {}
        for k, v in self.__dict__.items():
            # torch dtype 같은 비-JSON 직렬화 타입을 처리
            if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                out[k] = str(v)
            else:
                out[k] = v
        return out

    def to_json(self, save_path: str) -> None:
        """
        Save configuration as a JSON file.
        """
        with open(save_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)


cfg = Config()

def main():
    print("======= Training the model... =======\n")

    # Setup
    print("Setting up the training environment...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    cfg.to_json(os.path.join(cfg.checkpoints_dir, "config.json"))

    # set_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load model
    print("Loading the pipeline...")

    pipeline = SD2CubeDiffPipeline.from_pretrained(
        model_name_or_path=cfg.model_id,
        dtype=torch.float32 if cfg.dtype == "float32" else torch.float16,
    )

    # Setup scheduler
    print("Setting up the scheduler...")

    if cfg.prediction_type == "v_prediction":
        pipeline.scheduler = DDIMScheduler.from_pretrained(
            "Manojb/stable-diffusion-2-base",
            subfolder="scheduler"
        )
        pipeline.scheduler.config.prediction_type = "v_prediction"
    else:
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.scheduler.config.prediction_type = "epsilon"
    
    # Move models to device
    pipeline.vae.to(device)
    pipeline.unet.to(device)
    pipeline.text_encoder.to(device)

    # Freeze and unfreeze parameters
    pipeline.vae.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    
    for name, param in pipeline.unet.conv_in.named_parameters():
        param.requires_grad_(True)
    for name, param in pipeline.unet.named_parameters():
        if "attn" in name:
            param.requires_grad_(True)
    
    # Dataset and dataloader
    print(f"Loading the dataset from {cfg.data_dir}...")
    
    # Transform with resize to cfg.image_size
    train_transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = CubemapDataset(
        dir_path=cfg.data_dir,
        transform=train_transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer
    params = [p for p in pipeline.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate)
    scaler = GradScaler(device="cuda" if torch.cuda.is_available() else "cpu")

    # Training loop
    global_step = 0
    start_epoch = 0

    if cfg.resume_checkpoint and os.path.exists(cfg.resume_checkpoint):
        print(f"Resuming from {cfg.resume_checkpoint}...")
        checkpoint = torch.load(cfg.resume_checkpoint, map_location=device)
        pipeline.unet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
    
    pipeline.unet.train()
    T = 6 # faces

    # Calculate total iterations
    steps_per_epoch = len(dataloader)
    total_iterations = cfg.epochs * steps_per_epoch
    start_iteration = start_epoch * steps_per_epoch

    # Training info
    print(f"\nStarting training for {total_iterations} iterations ({cfg.epochs} epochs)...")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Device: {device}")
    print(f"Save directory: {cfg.checkpoints_dir}\n")

    # Training losses for averaging
    train_losses = []
    
    # Best model tracking
    best_loss = float('inf')

    # Main training loop with tqdm
    pbar = tqdm(range(start_iteration, total_iterations), desc="Training", initial=start_iteration, total=total_iterations)
    
    data_iter = iter(dataloader)
    current_epoch = start_epoch

    for iteration in pbar:
        # Get batch (handle epoch boundaries)
        try:
            target = next(data_iter)
        except StopIteration:
            current_epoch += 1
            data_iter = iter(dataloader)
            target = next(data_iter)

        # target : [B, 6, C, H, W]
        
        # 1. Data Preparation
        target = target.to(device)

        if target.dim() == 5:
            b, t, c, h, w = target.shape
            target = target.view(b * t, c, h, w)
            # target : [B*6, C, H, W]
        
        # VAE Encoding (No Grad)
        with torch.no_grad():
            latents = pipeline.vae.encode(target).latent_dist.mean
            latents = latents * pipeline.vae.config.scaling_factor
            # latents : [B*6, 4, H/8, W/8]

        B_batch = latents.shape[0] // T
        _, _, H, W = latents.shape

        # 2. Text Embedding
        # Simple logic: using prompts directly (multitext support assumed default)
        '''
        text_inputs = pipeline.tokenizer(
            prompts, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(device)
        
        with torch.no_grad():
            encoder_hidden_states = pipeline.text_encoder(text_input_ids)[0]
        '''
        empty_inputs = pipeline.tokenizer(
            [""] * (B_batch * T),  # Must match batch dimension of latent_input
            padding="max_length",
            max_length=77,
            return_tensors="pt",
        )
        empty_input_ids = empty_inputs.input_ids.to(device)
        with torch.no_grad():
            encoder_hidden_states = pipeline.text_encoder(empty_input_ids)[0]

        # 3. Noise & Timestep Generation
        # Sample random timesteps
        timesteps = torch.randint(
            0, pipeline.scheduler.config.num_train_timesteps, (B_batch,), 
            device= device, dtype=torch.long
        )
        timesteps = timesteps.repeat_interleave(T)

        # Face Masking (Front face vs Others)
        face_ids = torch.arange(B_batch * T, device=device) % T  
        drop_ids = torch.tensor(
            np.random.choice(T, np.random.randint(1, T+1), replace=False),
            device=device,
            dtype=torch.long 
        )
        drop_mask = torch.isin(face_ids, drop_ids)
        
        noise = torch.randn_like(latents)
        noisy_latents = latents.clone()

        
        # Add noise only to non-front faces
        noisy_latents[drop_mask] = pipeline.scheduler.add_noise(
            latents[drop_mask], noise[drop_mask], timesteps[drop_mask]
        )
        
        # Extra channels setup (use actual latent spatial size H, not cfg.image_size//8)
        extra_channels = make_extra_channels_tensor(B_batch, drop_ids, H, cfg.fov).to(device, dtype=latents.dtype)
        latent_input = torch.cat([noisy_latents, extra_channels], dim=1)

        # 4. Forward Pass (with Mixed Precision)
        optimizer.zero_grad()
        
        with autocast(device_type=device.type):
            model_pred = pipeline.unet(
                latent_input,
                timesteps,
                encoder_hidden_states=encoder_hidden_states
            ).sample

            # 5. Loss Calculation
            if cfg.prediction_type == "v_prediction":
                v_target = pipeline.scheduler.get_velocity(
                    latents[drop_mask], noise[drop_mask], timesteps[drop_mask]
                )
                loss = nn.functional.mse_loss(model_pred[drop_mask], v_target)
            else: # epsilon
                loss = nn.functional.mse_loss(model_pred[drop_mask], noise[drop_mask])

        # 6. Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0) # Gradient Clipping
        scaler.step(optimizer)
        scaler.update()

        # Track loss
        loss_val = loss.item()
        train_losses.append(loss_val)
        global_step += 1

        # Update progress bar with current loss
        pbar.set_postfix({"Loss": f"{loss_val:.4f}"})

        # Logging at intervals
        if (iteration + 1) % cfg.log_interval == 0:
            avg_loss = sum(train_losses[-cfg.log_interval:]) / min(cfg.log_interval, len(train_losses))
            current_epoch_num = (iteration + 1) // steps_per_epoch + 1
            step_in_epoch = (iteration + 1) % steps_per_epoch
            tqdm.write(f"Iteration {iteration+1}/{total_iterations}, Epoch {current_epoch_num}, "
                      f"Step {step_in_epoch}/{steps_per_epoch}, Loss: {loss_val:.4f}, Avg Loss: {avg_loss:.4f}")
            
            # Plot & Save Loss Figure
            plot_train_loss(train_losses, os.path.join(cfg.checkpoints_dir, "loss.png"))

        # Save Checkpoint at epoch boundaries
        if (iteration + 1) % steps_per_epoch == 0:
            epoch_num = (iteration + 1) // steps_per_epoch
            epoch_losses = train_losses[-(steps_per_epoch):]
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            tqdm.write(f"\n  Epoch {epoch_num} Finished. Avg Loss: {avg_epoch_loss:.4f}")

            # Save best model (overwrite if loss improved)
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_path = os.path.join(cfg.checkpoints_dir, "best_model.pt")
                torch.save({
                    'epoch': epoch_num - 1,
                    'global_step': global_step,
                    'model_state_dict': pipeline.unet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_epoch_loss,
                }, best_path)
                tqdm.write(f"  ★ New best model saved! (Loss: {avg_epoch_loss:.4f})")

            # Save latest checkpoint (overwrite previous, every save_interval_epoch)
            if epoch_num % cfg.save_interval_epoch == 0:
                save_path = os.path.join(cfg.checkpoints_dir, "latest_checkpoint.pt")
                torch.save({
                    'epoch': epoch_num - 1,
                    'global_step': global_step,
                    'model_state_dict': pipeline.unet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_epoch_loss,
                }, save_path)
                tqdm.write(f"  Latest checkpoint saved (Epoch {epoch_num})\n")

    # Save Final Model
    final_path = os.path.join(cfg.checkpoints_dir, "final_cubediff.pt")
    torch.save({'model_state_dict': pipeline.unet.state_dict()}, final_path)
    
    print(f"\nTraining completed!")
    print(f"  Final model: {final_path}")
    print(f"  Best model:  {os.path.join(cfg.checkpoints_dir, 'best_model.pt')} (Loss: {best_loss:.4f})")
    print(f"  Results dir: {cfg.checkpoints_dir}")

    return 0 # Success


if __name__ == "__main__":
    main()
