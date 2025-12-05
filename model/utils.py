import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import List
from PIL import Image
import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from model.pipeline import SD2CubeDiffPipeline
from PIL import Image

DIRS = ["posx", "posy", "negx", "negy"]

REVERSE = {
    "posx" : "negx",
    "posy" : "negy",
    "negx" : "posx",
    "negy" : "posy"
}

DELTA = {
    "posx": (1,0),
    "negx": (-1,0),
    "posy": (0,1),
    "negy": (0,-1)
}

FACES = { # view dir : front, left, right, up, down
    "posx": ["posx", "posy", "negy", "posz", "negz"],
    "negx": ["negx", "negy", "posy", "posz", "negz"],
    "posy": ["posy", "negx", "posx", "posz", "negz"],
    "negy": ["negy", "posx", "negx", "posz", "negz"] 
}

UP = { # Up vector direction for each face
    "posx" : np.array([0., 0., 1.]),
    "negx" : np.array([0., 0., 1.]),
    "posy" : np.array([0., 0., 1.]),
    "negy" : np.array([0., 0., 1.]),
    "posz" : np.array([1., 0., 0.]),
    "negz" : np.array([1., 0., 0.])
}

VIEW = { # View vector from origin to each face
    "posx" : np.array([1., 0., 0.]),
    "negx" : np.array([-1., 0., 0.]),
    "posy" : np.array([0., 1., 0.]),
    "negy" : np.array([0., -1., 0.]),
    "posz" : np.array([0., 0., 1.]),
    "negz" : np.array([0., 0., -1.])
}

ROTATE = { # Number of rotation to left required 
    "posx": 0,
    "negx": 2,
    "posy": 1,
    "negy": 3
}

ORIG_COORDS = {
    "front": np.array([[0., 0.], [1. ,0.], [1., 1.], [0., 1.]]),
    "left": np.array([[0., 0.], [1. ,0.], [1., 1.], [0., 1.]]),
    "right": np.array([[0., 0.], [1. ,0.], [1., 1.], [0., 1.]]),
    "up": np.array([[1., 1.], [0. , 1.], [0., 0.], [1., 0.]]),
    "down" : np.array([[0., 0.], [1. ,0.], [1., 1.], [0., 1.]])
}

TARGET_COORDS = { # 목표 좌표 위치 - 왼쪽 위 부터 반시계 방향
    "front": np.array([[0.25, 0.25], [0.75 ,0.25], [0.75, 0.75], [0.25, 0.75]]),
    "left": np.array([[0., 0.], [1. ,0.], [0.75, 0.25], [0.25, 0.25]]),
    "right": np.array([[0.25, 0.75], [0.75 ,0.75], [1., 1.], [0., 1.]]),
    "up": np.array([[0., 0.], [0.25 ,0.25], [0.25, 0.75], [0., 1.]]),
    "down" : np.array([[0.75, 0.25], [1. ,0.], [1., 1.], [0.75, 0.75]]),
}

class CubeMapData:
    def __init__(self, images, grid_x, grid_y):
        self.image = {}
        faces =  ["posx", "posy", "posz", "negx", "negy", "negz"]
        for face_name, face_image in zip(faces, images):
            self.image[face_name] = face_image

        self.x = grid_x
        self.y = grid_y
        return

    def save(self, save_path: str):
        assert self.x is not None and self.y is not None
        for face_name, face_image in self.image.items():
            face_image = Image.fromarray(face_image)
            face_path = os.path.join(save_path, f"{self.x}_{self.y}_{face_name}.png")
            face_image.save(face_path)
        return

def preprocess_image(
    image : Image,
    image_size: int
)-> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return transform(image)

def read_image(
    image_path: str,
    image_size: int
)-> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = preprocess_image(image, image_size)
    return image

def load_pipeline(
    checkpoint_path: str, # path to pt file 
    device: str,
)-> SD2CubeDiffPipeline:
    pipeline = SD2CubeDiffPipeline.load_checkpoint(
        checkpoint_path=checkpoint_path,
    ).to(device)
    return pipeline

def predict_pipeline(
    pipeline: SD2CubeDiffPipeline,
    conditioning_images : List[torch.Tensor],
    conditioning_faces : List[str],
    num_inference_steps: int,
    cfg_scale: float,
    grid_x: int,
    grid_y: int
)-> CubeMapData:
    output = pipeline(
        conditioning_images,
        conditioning_faces,
        num_inference_steps,
        cfg_scale
    ).faces_cropped  # numpy array
    
    output = CubeMapData(output, grid_x, grid_y)
    return output

def compute_homography(src, targets):
    A = []
    for (x, y), (X, Y) in zip(src, targets):
        A.append([-x, -y, -1, 0, 0, 0, x*X, y*X, X])
        A.append([0, 0, 0, -x, -y, -1, x*Y, y*Y, Y])
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3,3)
    return H / H[2,2] # (3, 3) np.array

def rotate_left(image: np.ndarray, times: int) -> np.ndarray:
    """
    image: (C, H, W)
    times: 왼쪽으로 90도 회전하는 횟수
    """
    times = times % 4
    img_hw_c = image.transpose(1, 2, 0)
    img_rot = np.rot90(img_hw_c, k=times)   # left-rotation
    return img_rot.transpose(2, 0, 1)   

def warp_with_homography(image: np.ndarray,
                         homography: np.ndarray,
                         out_size=None) -> np.ndarray:
    """
    image: (C, H, W)
    homography: (3, 3)
    out_size: (W, H)  # OpenCV 규칙
    """
    C, H, W = image.shape
    if out_size is None:
        out_size = (W, H)  # (W,H)

    img_hw_c = image.transpose(1, 2, 0)
    H_mat = homography.astype(np.float32)
    warped = cv2.warpPerspective(
        img_hw_c,
        H_mat,
        out_size,               
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    warped = warped.transpose(2, 0, 1)
    return warped

def project_cube(
    cube: CubeMapData,
    view: str, # [posx, posy, negx, negy]
    image_size: int
)-> torch.Tensor: # view 방향에 cube가 있을때 projection한 결과 이미지를 얻기

    projected_image = np.zeros((3, image_size, image_size), dtype=np.uint8)

    faces = FACES[view] # view dir : front, left, right, up, down
    directions = ['front', 'left', 'right', 'up', 'down']
    for face, direction in zip(faces, directions):
        image = cube.image[face].transpose(2, 0, 1) # [0, 255]

        if direction in ['up', 'down']:
            image = rotate_left(image, ROTATE[view])

        H, W = image.shape[1], image.shape[2]
        src = ORIG_COORDS[direction][:, ::-1] * np.array([W, H])
        dst = TARGET_COORDS[direction][:,::-1] * np.array([W, H])
        homography = compute_homography(src, dst)
        image = warp_with_homography(image, homography) 

        projected_image = np.maximum(projected_image, image)

    projected_image = Image.fromarray(projected_image.transpose(1, 2, 0))
    projected_image = preprocess_image(projected_image, image_size) # To [-1,1]
    return projected_image


'''
image_path = "/root/project/results/cubediff/0_negx.png"
save_path = "/root/project/results/cubediff/sample.png"

image = read_image(image_path, 512).numpy()  # [-1, 1]
image = (image + 1.0) * 0.5 * 255
print(image.shape, image.max(), image.min())

dir = "front"
H, W = image.shape[1], image.shape[2]
src = ORIG_COORDS[dir][:, ::-1] * np.array([W, H])
dst = TARGET_COORDS[dir][:,::-1] * np.array([W, H])
h = compute_homography(src, dst)
print(h)

result = warp_with_homography(image, h) 
print(result.shape, result.max(), result.min())

result_hwc = np.transpose(result, (1, 2, 0))
print(result_hwc.shape, result_hwc.max(), result_hwc.min())

img = Image.fromarray(result_hwc.astype(np.uint8))
img.save(save_path)
'''