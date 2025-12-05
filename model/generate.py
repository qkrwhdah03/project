import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.generator import generate

if __name__ == "__main__":

    import argparse 
    parser = argparse.ArgumentParser(description="Generate the cube world based on a single conditioning image")
    
    parser.add_argument("--image_path", type=str, required=True, help="Conditioning image path")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Checkpoint to load pipeline")
    parser.add_argument("--save_path", type=str, default= "/root/project/results/world/", help= "Cube world save path")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Cfg scale")
    parser.add_argument("--world_level", type=int, default=1, help="World level to create (0 will be a single cube)")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--device", type=str, default='cuda', help="Device to load pipeline")
    
    args = parser.parse_args()

    generate(args)
