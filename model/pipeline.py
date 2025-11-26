# Code adapted from: https://github.com/Juan5713/OpenCubeDiff/

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import BaseOutput
from dataclasses import dataclass
from norm import patch_groupnorm
from attn import swap_transformer_blocks
from utils import patch_unet

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
    pass 

class SD2CubeDiffPipeline(StableDiffusionPipeline):
    
    @classmethod
    def from_pretrained(cls, **kwargs):

        model_id = "Manojb/stable-diffusion-2-base"
        pipeline = super().from_pretrained(model_id, **kwargs)

        if pipeline.unet.config.in_channels != 7:
            # Is a base SD model, patch input conv as well
            patch_unet(pipeline.unet, in_channels=7)
        else:
            # Apply attention patches (swap BasicTransformerBlock -> CubeDiffTransformerBlock)
            swap_transformer_blocks(pipeline.unet)

        # Synchronized GroupNorm
        patch_groupnorm(pipeline.vae)

        return pipeline


    @torch.no_grad()
    def __call__(
        self, 
        num_inference_steps: int = 50
    )-> SD2CubeDiffPipelineOutput:
        pass