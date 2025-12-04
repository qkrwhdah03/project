# model/pipeline.py
# Entire pipeline for CubeDiff (Stable Diffusion 2)

# NOTE: Cited from https://github.com/Juan5713/OpenCubeDiff/ (Open source)

import os
import json
import torch
import numpy as np
from typing import Union, List, Optional
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import BaseOutput
from dataclasses import dataclass

from modules.additional_channels import make_extra_channels_tensor
from modules.attention import swap_transformer_blocks
from modules.group_norm import patch_groupnorm
from modules.utils import patch_unet
from modules.postprocess import postprocess_outputs

# Stable diffusion 2
'''
model_id = "Manojb/stable-diffusion-2-base"

scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  
    
image.save("astronaut_rides_horse.png")
'''

@dataclass
class SD2CubeDiffPipelineOutput(BaseOutput):
    faces: np.ndarray
    faces_cropped: np.ndarray
    equirectangular: np.ndarray

class SD2CubeDiffPipeline(StableDiffusionPipeline):
    
    @classmethod
    def load_checkpoint(cls, checkpoint_path: str, model_name_or_path: str = "Manojb/stable-diffusion-2-base", **kwargs):
        pipeline = cls.from_pretrained(model_name_or_path, **kwargs)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        pipeline.unet.load_state_dict(ckpt["model_state_dict"])

        config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        pipeline.image_size = config["image_size"]
        pipeline.fov = config["fov"]
        pipeline.prediction_type = config["prediction_type"]
        pipeline.scheduler.config.prediction_type = pipeline.prediction_type
        return pipeline


    @classmethod
    def from_pretrained(cls, model_name_or_path: str = "Manojb/stable-diffusion-2-base", **kwargs):
        # Pretrained model or checkpoint
        pipeline = super().from_pretrained(model_name_or_path, **kwargs)
        # Patch UNet to CubeDiff architecture
        patch_unet(pipeline.unet, in_channels=7)
        # Apply synchronized GroupNorm
        patch_groupnorm(pipeline.vae)
        return pipeline

    @torch.no_grad()
    def __call__(
        self,
        conditioning_images: List[torch.Tensor],      # (C,H,W)
        conditioning_faces: List[str] = ["posx"],  # "posx", "posy", "posz", "negx", "negy", "negz"
        num_inference_steps: int = 50,
        cfg_scale: float = 3.5,
    ):
        assert len(conditioning_images) == len(conditioning_faces)

        device = self._execution_device
        
        # Unconditional guidance
        empty_inputs = self.tokenizer(
            [""] * 6,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt",
        )
        empty_embeddings = self.text_encoder(empty_inputs.input_ids.to(device))[0]

        # --- scheduler / latents -------------------------------------------
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        latents = torch.randn(
            (6, 4, self.image_size//8, self.image_size//8),
            device=device,
            dtype= self.unet.dtype,
        )
        latents *= self.scheduler.init_noise_sigma

        uncond_latents = latents.clone()

        # Conditioning image (Reference for the generation process)
        for i, conditioning_image in enumerate(conditioning_images):
            if conditioning_image.ndim == 3:
                conditioning_image = conditioning_image.unsqueeze(0)
            conditioning_image = conditioning_image.to(device, dtype=self.unet.dtype)
            conditioning_images[i] = conditioning_image

        # Reference latent for the conditioning face
        ref_lats = []
        for conditioning_image in conditioning_images:
            ref_lat = self.vae.encode(conditioning_image).latent_dist.mean[0]
            ref_lat *= self.vae.config.scaling_factor
            ref_lats.append(ref_lat)

        # Face to index mapping
        face_to_index = {"posx": 0,  "posy": 1, "posz": 2, "negx": 3, "negy":4, "negz": 5}
        conditioning_index = [face_to_index[face] for face in conditioning_faces ]
        drop_ids = torch.tensor(list(set(range(6)) - set(conditioning_index)))
        static_extra = make_extra_channels_tensor(
            batch_size=1,
            drop_ids=drop_ids,
            face_size= self.image_size//8,
        ).to(device, dtype=self.unet.dtype)

        for t in self.scheduler.timesteps:

            # Conditional Generation
            for index, ref_lat in zip(conditioning_index, ref_lats):
                latents[index] = ref_lat  # Keep the conditioning face fixed
            latents_scaled = self.scheduler.scale_model_input(latents, t)
            latents_input = torch.cat([latents_scaled, static_extra], dim=1)

            noise_pred = self.unet(
                latents_input,
                t,
                encoder_hidden_states=empty_embeddings
            ).sample
            
            # Unconditional Generation
            uncond_latents_scaled = self.scheduler.scale_model_input(uncond_latents, t)
            uncond_latents_input = torch.cat([uncond_latents_scaled, static_extra], dim=1)
            noise_pred_uncond = self.unet(
                uncond_latents_input,
                t,
                encoder_hidden_states=empty_embeddings
            ).sample

            # Classifier Free-Guidance
            combined = noise_pred_uncond + cfg_scale * (noise_pred - noise_pred_uncond)
            latents = self.scheduler.step(combined, t, latents).prev_sample

        # Put conditioned image again before decode
        for index, ref_lat in zip(conditioning_index, ref_lats):
                latents[index] = ref_lat  # Keep the conditioning face fixed
        
        # Decode predicted latent
        imgs = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1) # Rescale it to [0,1]
        
        # Postprocess
        equirec, uncropped, cropped = postprocess_outputs(imgs)
        return SD2CubeDiffPipelineOutput(
            faces=uncropped,
            faces_cropped=cropped,
            equirectangular=equirec,
        )
