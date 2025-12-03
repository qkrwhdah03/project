# model/pipeline.py
# Entire pipeline for CubeDiff (Stable Diffusion 2)

# NOTE: Cited from https://github.com/Juan5713/OpenCubeDiff/ (Open source)

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
        # prompts: Union[str, List[str]],        # Prompts for text conditioning
        *,
        conditioning_image: torch.Tensor,      # (C,H,W)
        conditioning_face: str = "front",  # "front", "back", "left", "right", "top", "bottom"
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        cfg_scale: float = 3.5,
    ):
        device = self._execution_device
        T = 6  # nubmer of faces

        # Classifier-free guidance
        # # Prepare prompts
        # if isinstance(prompts, str):
        #     prompts = [prompts] * T  # Copy the prompts for each face
        # if len(prompts) != T:
        #     raise ValueError(f"Expected 6 prompts, got {len(prompts)}")
        
        # # Encode text prompts
        # text_inputs = self.tokenizer(
        #     prompts,
        #     max_length=self.tokenizer.model_max_length,
        #     padding="max_length",
        #     return_tensors="pt",
        # )
        # encoder_hidden_states = self.text_encoder(text_inputs.input_ids.to(device))[0]

        # # Encode blank prompts for unconditional guidance
        # uncond_inputs = self.tokenizer(
        #     [""] * T,
        #     max_length=self.tokenizer.model_max_length,
        #     padding="max_length",
        #     return_tensors="pt",
        # )
        # uncond_embeddings = self.text_encoder(uncond_inputs.input_ids.to(device))[0]

        # Unconditional guidance
        empty_inputs = self.tokenizer(
            [""] * T,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt",
        )
        empty_embeddings = self.text_encoder(empty_inputs.input_ids.to(device))[0]

        # --- scheduler / latents -------------------------------------------
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        latents = torch.randn(
            (T, 4, self.unet.config.sample_size, self.unet.config.sample_size),
            generator=generator,
            device=device,
            dtype=self.unet.dtype,
        )
        latents *= self.scheduler.init_noise_sigma

        # Conditioning image (Reference for the generation process)
        if conditioning_image.ndim == 3:
            conditioning_image = conditioning_image.unsqueeze(0)  # Add batch dimension
        conditioning_image = conditioning_image.to(device, dtype=self.unet.dtype)

        # Reference latent for the conditioning face
        ref_lat = self.vae.encode(conditioning_image).latent_dist.mean[0]
        ref_lat *= self.vae.config.scaling_factor

        # Face to index mapping
        face_to_index = {"front": 0, "back": 1, "left": 2, "right": 3, "top": 4, "bottom": 5}
        conditioning_index = face_to_index[conditioning_face]

        # Extend tensor channels for extra conditions
        # drop_ids = faces to generate (mask=0), conditioning face should have mask=1
        all_faces = set(range(6))
        drop_ids = torch.tensor(list(all_faces - {conditioning_index}))
        static_extra = make_extra_channels_tensor(
            batch_size=1,
            drop_ids=drop_ids,
            face_size=self.unet.config.sample_size,
        ).to(device, dtype=self.unet.dtype)

        for t in self.scheduler.timesteps:
            latents[conditioning_index] = ref_lat  # Keep the conditioning face fixed
            latents_scaled = self.scheduler.scale_model_input(latents, t)
            latents_input = torch.cat([latents_scaled, static_extra], dim=1)

            # Classifier-free guidance
            # Predict noise twice to implement classifier-free guidance
            noise_pred = self.unet(
                latents_input,
                t,
                # encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states=empty_embeddings
            ).sample
            noise_pred_uncond = self.unet(
                latents_input,
                t,
                # encoder_hidden_states=uncond_embeddings,
                encoder_hidden_states=empty_embeddings,
                cross_attention_kwargs={"drop_face_index": conditioning_index},
            ).sample

            combined = noise_pred_uncond + cfg_scale * (noise_pred - noise_pred_uncond)
            # latents[1:] = self.scheduler.step(combined[1:], t, latents[1:]).prev_sample
            latents = self.scheduler.step(combined, t, latents).prev_sample

        # --- decode ---------------------------------------------------------
        imgs = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        equirec, uncropped, cropped = postprocess_outputs(imgs)

        return SD2CubeDiffPipelineOutput(
            faces=uncropped,
            faces_cropped=cropped,
            equirectangular=equirec,
        )
