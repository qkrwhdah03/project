from typing import List
from utils import wait, read_image, get_face_quaternion

import os
import numpy as np
import viser


class CubeMap:
    def __init__(
        self,
        name: str,
        dir_path: str, 
        prefix: str,
        suffixes: List[str] = ["posx", "posy", "posz", "negx", "negy", "negz"],
        ext: str = ".png"
    )-> None:
        self.name = name
        
        paths = [os.path.join(dir_path, prefix + "_" + suffix + ext) for suffix in suffixes]
        self.images = {}
        for suffix, path in zip(suffixes, paths):
            self.images[suffix] = read_image(path) # list of [W, H, C] shaped uint8 numpy ndarray
        
        return
    
    def get(self, key: str)-> np.ndarray:
        return self.images[key]

class CubeMapViewer:
    def __init__(
        self,
        cube_render_x: int = 100,
        cube_render_y: int = 100,
        cube_render_z: int = 100,
        keys: List[str] = ["posx", "posy", "posz", "negx", "negy", "negz"]
    )-> None:
        assert len(keys) == 6
        self.server = viser.ViserServer(host="0.0.0.0", port=8080)
        self.cube_render_x = cube_render_x
        self.cube_render_y = cube_render_y
        self.cube_render_z = cube_render_z
        self.keys = keys

        # Note that the scene is reverted upside down.
        self.pos = {
            "posx": np.array([self.cube_render_x/2, 0, 0]),
            "posy": np.array([0, self.cube_render_y/2, 0]),
            "posz": np.array([0, 0, self.cube_render_z/2]),
            "negx": np.array([-self.cube_render_x/2, 0, 0]),
            "negy": np.array([0, -self.cube_render_y/2, 0]),
            "negz": np.array([0, 0, -self.cube_render_z/2]),
        }

        self.wxyz = {
            "posx": get_face_quaternion(np.array([1., 0., 0.]), np.array([0., 0., 1.])),
            "posy": get_face_quaternion(np.array([0., 1., 0.]), np.array([0., 0., 1.])),
            "posz": get_face_quaternion(np.array([0., 0., 1.]), np.array([1., 0., 0.])),
            "negx": get_face_quaternion(np.array([-1., 0., 0.]), np.array([0., 0., 1.])),
            "negy": get_face_quaternion(np.array([0., -1., 0.]), np.array([0., 0., 1.])),
            "negz": get_face_quaternion(np.array([0., 0., -1.]), np.array([1., 0., 0.])),
        }
        

        self.width = {
            "posx": self.cube_render_z,
            "posy": self.cube_render_x,
            "posz": self.cube_render_x,
            "negx": self.cube_render_z,
            "negy": self.cube_render_x,
            "negz": self.cube_render_x
        }
        self.height = {
            "posx": self.cube_render_y,
            "posy": self.cube_render_z,
            "posz": self.cube_render_y,
            "negx": self.cube_render_y,
            "negy": self.cube_render_z,
            "negz": self.cube_render_y
        }
        return
    
    def add_cubemap(self, cubemap: CubeMap)-> None:
        for key in self.keys:
            self.server.scene.add_image(
                name = f"{cubemap.name}_{key}",
                image = cubemap.get(key),
                render_width = self.width[key],
                render_height = self.height[key],
                wxyz = self.wxyz[key],
                position = self.pos[key],
                visible = True
            )
        return

    def run(self)-> None:
        print("Viewer running at http://localhost:8080")
        wait()