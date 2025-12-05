import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
import json
from collections import deque
from model.utils import (
    read_image, 
    load_pipeline, 
    predict_pipeline, 
    project_cube,
    DIRS, 
    REVERSE, 
    DELTA
)

def generate(args):
    # Load pipeline and input image
    pipeline = load_pipeline(args.checkpoint_path, args.device)
    orig_image = read_image(args.image_path, pipeline.image_size).to(args.device) # torch.Tensor [C, H, W]

    # BFS manner generation
    level = args.world_level
    size = 2 * level + 1
    grid = [[None for _ in range(size)] for _ in range(size)] # level, level is center
    
    q = deque()
    curx = level
    cury = level
    for l in range(level + 1):
        if l == 0: # Initial Cube
            cube = predict_pipeline(
                pipeline= pipeline,
                conditioning_images= [orig_image],
                conditioning_faces= ["posx"],
                num_inference_steps= args.num_inference_steps,
                cfg_scale= args.cfg_scale,
                grid_x= curx,
                grid_y= cury
            )
            q.appendleft(cube)
            grid[curx][cury] = cube
        
        else: # Adjacent Cube
            cur_size = len(q) # 이전 level에 있는 cube 수 만큼
            for _ in range(cur_size):
                prev_cube = q.pop()
                for dir in DIRS:
                    dx, dy = DELTA[dir]
                    curx = prev_cube.x + dx
                    cury = prev_cube.y + dy
                    
                    if grid[curx][cury] == None:
                        
                        # Get conditioning images from adjacent cubes
                        conditioning_images = [] 
                        conditioning_faces = []
                        
                        for c_dir, (xx, yy) in DELTA.items():
                            adjx = curx + xx
                            adjy = cury + yy
                            if adjx < 0 or adjy < 0 or adjx >= size or adjy >= size:
                                continue
                            
                            adj_cube = grid[adjx][adjy]
                            if adj_cube is not None:
                                view = c_dir
                                projected_image = project_cube(adj_cube, view, pipeline.image_size)
                                conditioning_images.append(projected_image)
                                conditioning_faces.append(view)
                        
                        # Generate cube
                        cube = predict_pipeline(
                            pipeline= pipeline,
                            conditioning_images= conditioning_images,
                            conditioning_faces= conditioning_faces,
                            num_inference_steps= args.num_inference_steps,
                            cfg_scale= args.cfg_scale,
                            grid_x= curx,
                            grid_y= cury
                        )

                        q.appendleft(cube)
                        grid[curx][cury] = cube

    # Save results
    os.makedirs(args.save_path, exist_ok=True)
    for i in range(size):
        for j in range(size):
            cube = grid[i][j]
            if cube is not None:
                cube.save(args.save_path)
    
    with open(os.path.join(args.save_path, "pos.json"), "w") as f:
        data = {
            "world_level" : args.world_level
        }
        json.dump(data, f)    
    return