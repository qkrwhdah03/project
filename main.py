# main.py
# Main script for the project
import argparse
import gc
import sys
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms
from diffusers import StableDiffusionXLPipeline

from model.pipeline import SD2CubeDiffPipeline
from viewer.viewer import CubeMapViewer, CubeMapReader
from forward_warp import forward_warp_image

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

@torch.no_grad()
def main(args):
    output_dir = project_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    # Generate reference image
    # ref_image_prompt = input("Enter your prompt: ")
    ref_image_prompt = """Photorealistic view of a perfectly straight two-lane of college campus road, a few students on both sides of the road, Google Street View style image"""
    
    ref_image_pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        dtype=dtype, 
        use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None,
    ).to(device)

    ref_image = ref_image_pipe(ref_image_prompt).images[0]

    ref_image_path = output_dir / f"ref_image_0.png"
    ref_image.save(ref_image_path)

    # Free SDXL pipeline memory
    ref_image_pipe.to('cpu')
    del ref_image_pipe
    gc.collect()
    torch.cuda.empty_cache()

    warped_images = [ref_image]
    for i in range(1, 3):
        warped, _ = forward_warp_image(
            target=ref_image,
            warp_dist=args.step,
            strength= args.strength,
            prompt=ref_image_prompt,
        )
        ref_image = warped
        warped_save_path = output_dir / f"ref_image_{i}_warped.png"
        warped.save(warped_save_path)
        warped_images.append(warped)

    cubediff_pipeline = SD2CubeDiffPipeline.load_checkpoint(
        checkpoint_path=args.checkpoint_path,
        dtype= torch.float16 if args.dtype == "float16" else torch.float32
    )

    transform = transforms.Compose([
        transforms.Resize((cubediff_pipeline.image_size, cubediff_pipeline.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    for i, warped_img in enumerate(warped_images):
        cubediff_pipeline.to(device)
        
        conditioning_image = transform(warped_img)

        output = cubediff_pipeline(
            conditioning_images=[conditioning_image],
            num_inference_steps=50,
            cfg_scale=args.cfg_scale,
        )

        faces = ["posx", "posy", "posz", "negx", "negy", "negz"]
        for face_name, face_image in zip(faces, output.faces_cropped):
            face_path = output_dir / f"output_{i}_{face_name}.png"
            Image.fromarray(face_image).save(face_path)

        equirec_path = output_dir / f"output_{i}_equirec.png"
        Image.fromarray(output.equirectangular).save(equirec_path)

        cubediff_pipeline.to('cpu')
        torch.cuda.empty_cache()

    # Run viewer
    #viewer = CubeMapViewer(cube_render_x= 100, cube_render_y= 100, cube_render_z= 100)
    #cubemap = CubeMapReader(name="cubemap", dir_path=output_dir, prefix="output_0", ext=".png")
    #viewer.add_cubemap(cubemap)
    #viewer.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="results/12-04-164610/final_cubediff.pt",
                        help="Path to the checkpoint .pt file")
    parser.add_argument("--cfg_scale", type=float, help="Cfg scale", default=1.05)
    parser.add_argument("--step", type=float, help="Warp distance", default=5.0)
    parser.add_argument("--strength", type=float, help="Strength", default=0.15)
    parser.add_argument("--dtype", type=str, choices=["float16", "float32"], default="float16",
                        help="Data type to use for the pipeline")
    args = parser.parse_args()

    main(args)
