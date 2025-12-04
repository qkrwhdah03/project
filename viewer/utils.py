import time 
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

def postprocess_image(image, fov = 95)-> np.array:
    H, W = image.shape[:2]
    
    theta = fov / 2
    theta = theta * np.pi / 180
    ratio = 1 / np.tan(theta)

    new_w = int(W * ratio)
    new_h = int(H * ratio)

    x1 = (W - new_w) // 2
    y1 = (H - new_h) // 2
    x2 = x1 + new_w
    y2 = y1 + new_h

    cropped = image[y1:y2, x1:x2, :]
    resized = np.array(Image.fromarray(cropped).resize((W, H), Image.BICUBIC))

    return resized

def read_image(path: str)-> np.array:
    img = Image.open(path).convert("RGB")
    img = np.array(img) # (H, W, 3)
    # img = postprocess_image(img) # (H, W, 3)
    return img

def wait():
    while True:
        time.sleep(1.0)
    return 

def get_face_quaternion(
    target_view: np.array, # unit vector
    target_up: np.array, # unit vector
    base_view: np.array = np.array([0.0, 0.0, 1.0]),
    base_up: np.array = np.array([0.0, -1.0, 0.0])
)-> np.array:
    base_left = np.cross(base_up, base_view)
    target_left = np.cross(target_up, target_view)

    base = np.stack([base_up, base_view, base_left], axis=1)
    target = np.stack([target_up, target_view, target_left], axis=1)

    rot_mat = target @ base.T

    quat = R.from_matrix(rot_mat).as_quat()  # [x, y, z, w]
    return np.array([quat[3], quat[0], quat[1], quat[2]])  # wxyz