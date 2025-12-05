import sys
import os
import glob
from pathlib import Path
import json
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from viewer.viewer import CubeMapViewer, CubeMapReader


if __name__ == "__main__":

    import argparse 
    parser = argparse.ArgumentParser(description="Convert equirectangular StreetView images to cubemap faces.")
    parser.add_argument("--cube_x", type=int, default= 100,
                        help="Size of rendered cubemap in x-axis")
    parser.add_argument("--cube_y", type=int, default= 100,
                    help="Size of rendered cubemap in y-axis")
    parser.add_argument("--cube_z", type=int, default= 100,
                    help="Size of rendered cubemap in z-axis")
    parser.add_argument("--pad", type=int, default= 1, help= "Padding between two cubemap")
    parser.add_argument("--dir_path", type=str, default="/root/project/results/world/",
                        help="World data directory path")

    
    args = parser.parse_args()

    viewer = CubeMapViewer(cube_render_x= args.cube_x, cube_render_y= args.cube_y, cube_render_z= args.cube_z)
    
    # Open json file
    json_path = os.path.join(args.dir_path, "pos.json")
    with open(json_path, "r") as f:
        data = json.load(f)
    level = data["world_level"]
    size = 2 * level + 1
    posx_paths = glob.glob(os.path.join(args.dir_path, "*_posx.png"))

    for posx_path in posx_paths:
        filename = os.path.basename(posx_path)  
        x, y = filename.split("_")[:2]
        cubemap = CubeMapReader(name=f"Cube({x},{y})", dir_path= args.dir_path, prefix=f"{x}_{y}",)
        xx = (int(x) - level) * args.pad + (int(x) - level) * args.cube_x
        yy = (int(y) - level) * args.pad + (int(y) - level) * args.cube_y
        origin = np.array([xx, yy, 0.])
        viewer.add_cubemap(cubemap, origin)
    viewer.run()