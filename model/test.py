# model/test.py
# Test the pipeline

import torch
from torchvision import transforms

import argparse
import sys
from pathlib import Path
from PIL import Image
from pipeline import SD2CubeDiffPipeline

def main(args):
    # Check device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("======= Testing the pipeline... =======\n")
    
    project_root = Path(__file__).parent.parent
    image_path = project_root / "data" / "cubemap" / f"{args.prefix}_posx.png"
    output_dir = project_root / "results" / "cubediff"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"Target image file not found: {image_path}")

    print(f"Image path: {image_path}")
    print(f"Output directory: {output_dir}\n")
    
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    print(f"Device: {device} with dtype {dtype}\n")
    
    # Load pipeline
    print("[1/4] Loading pipeline...\n")
    
    pipeline = SD2CubeDiffPipeline.load_checkpoint(
        checkpoint_path=args.checkpoint_path,
        dtype=dtype,
    ).to(device)

    print("\n[1/4] Pipeline loaded successfully!\n")

    # Load and preprocess target image
    print("[2/4] Loading target image...")

    transform = transforms.Compose([
        transforms.Resize((pipeline.image_size, pipeline.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    image = Image.open(image_path).convert("RGB")
    conditioning_image = transform(image)

    print("[2/4] Target image loaded successfully!\n")

    # Run inference
    print("[3/4] Running inference...")
    
    output = pipeline(
        conditioning_images= [conditioning_image],
        num_inference_steps=50,
        cfg_scale=args.cfg_scale
    )

    print("[3/4] Inference completed successfully!\n")

    # Save results
    print("[4/4] Saving results...")

    faces =  ["posx", "posy", "posz", "negx", "negy", "negz"]

    for face_name, face_image in zip(faces, output.faces_cropped):
        face_image = Image.fromarray(face_image)
        face_path = output_dir / f"{args.prefix}_{face_name}.png"
        face_image.save(face_path)
    
    equirec_img = Image.fromarray(output.equirectangular)
    equirec_path = output_dir / f"{args.prefix}_equirec.png"
    equirec_img.save(equirec_path)

    print("[4/4] Results saved successfully!")

    return 0  # Success


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, help="Prefix of the target image to test", required=True)
    parser.add_argument("--dtype", type=str, choices=["float16", "float32"], default="float16",
                        help="Data type to use for the pipeline")
    parser.add_argument("--checkpoint_path", type=str, help="Path to checkpoint pt file", required=True)
    parser.add_argument("--cfg_scale", type=float, help="Cfg scale", default=1.0)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None,
                        help="Device to run inference (default: auto-detect)")
    args = parser.parse_args()

    try:
        sys.exit(main(args))
    except Exception as e:
        print(f"\nError: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

