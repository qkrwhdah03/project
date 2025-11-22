from typing import Dict
import os
import cv2
import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


# StreetView360Atoz from Huggingface
class StreetViewDataset:
    def __init__(
        self, 
        save_dir = "/root/project/data"
    )-> None: 
        os.makedirs(save_dir, exist_ok= True)
        self.dataset = load_dataset(
            "everettshen/StreetView360AtoZ",
            split= "train",
            cache_dir=save_dir
        )
        return

    def size(self)-> int:
        return len(self.dataset)
    
    def get(self, idx: int)-> Image.Image:
        assert 0 <= idx < self.size(), f"index {idx} out of range"
        sample = self.dataset[idx]['image'] # PIL Image
        return sample

    
def sample_from_equirect(
    image: np.array,
    theta: np.array,
    phi: np.array
)-> np.array:
    H, W, _ = image.shape
    x = (theta + np.pi) / (2 * np.pi) * W
    y = (phi + np.pi / 2) / np.pi * H
    sample = cv2.remap(
        image, 
        x.astype(np.float32), 
        y.astype(np.float32), 
        interpolation=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_WRAP
    )
    return sample
    

def get_face(
    image: np.array,
    face_size: int,
    view: np.array, # unit vector
    up: np.array # unit vector
)-> np.array: # (face_size, face_size, 3)
    
    left = np.cross(up, view) # unit vector
    left_top = view + up + left
    # right_top = view + up - left
    # left_bottom = view - up + left
    # right_bottom = view - up - left

    v = -up
    u = -left
    n = np.arange(1, 2 * face_size, 2) / face_size
    gx, gy = np.meshgrid(n, n)
    rays = ( 
        left_top[None, None, :] 
        + gx[..., None] * u[None, None, :] 
        + gy[..., None] * v[None, None, :]
    )

    x = rays[:,:,0] # (face_size, face_size)
    y = rays[:,:,1] # (face_size, face_size)
    z = rays[:,:,2] # (face_size, face_size)

    theta = np.arctan2(y, x)
    phi = np.arctan(z / np.sqrt(x**2 + y **2))

    face = sample_from_equirect(image, theta, phi)
    return face

def equirect_to_cubemap(
    image: np.array,
    face_size: int
)-> Dict[str, np.array]:
    faces = {}

    faces["posx"] = get_face(image, face_size, np.array([1,0,0]),  np.array([0,-1,0]))
    faces["negx"] = get_face(image, face_size, np.array([-1,0,0]), np.array([0,-1,0]))

    faces["posy"] = get_face(image, face_size, np.array([0,1,0]),  np.array([0,0,1]))
    faces["negy"] = get_face(image, face_size, np.array([0,-1,0]), np.array([0,0,-1]))

    faces["posz"] = get_face(image, face_size, np.array([0,0,1]),  np.array([0,-1,0]))
    faces["negz"] = get_face(image, face_size, np.array([0,0,-1]), np.array([0,-1,0]))

    return faces

def convert_cubemap(
    save_dir: str,
    face_size: int
)-> None:
    dataset = StreetViewDataset(save_dir)

    new_save_dir = os.path.join(save_dir, "cubemap")
    os.makedirs(new_save_dir, exist_ok= True)

    for i in tqdm(range(dataset.size()), desc="Converting to cubemap"):
        image = dataset.get(i)
        image = np.array(image) # (H, W, C), 0~255 uint8 

        faces = equirect_to_cubemap(image, face_size)

        for key, face in faces.items():
            save_path = os.path.join(new_save_dir, f"{i}_{key}.png")
            Image.fromarray(face).save(save_path)
    
    print(f"Generating cubemap done. Save images at {new_save_dir}")
    return


if __name__ == "__main__":

    import argparse 
    parser = argparse.ArgumentParser(description="Convert equirectangular StreetView images to cubemap faces.")
    parser.add_argument("--save_dir", type=str, default="/root/project/data",
                        help="Directory to save cubemap images and/or cache dataset")
    parser.add_argument("--face_size", type=int, default=512,
                        help="Resolution of each cubemap face (default: 512)")
    
    args = parser.parse_args()

    convert_cubemap(args.save_dir, args.face_size)