import time 
import numpy as np
from PIL import Image

def read_image(path: str)-> np.array:
    img = Image.open(path).convert("RGB")
    return np.array(img)

def wait():
    while True:
        time.sleep(1.0)
    return 