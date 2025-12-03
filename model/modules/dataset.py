from typing import List
from collections import Counter
import os

from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class CubemapDataset(Dataset):
    def __init__(
        self, 
        dir_path = "/root/project/data/cubemap",
        suffixes: List[str] = ["posx", "posy", "posz", "negx", "negy", "negz"],
        transform = None
    )-> None:
        self.dir_path = dir_path
        self.suffixes = suffixes
        self.transform = transform if transform is not None else T.ToTensor()

        files = os.listdir(dir_path)
        counter = Counter(f.split("_")[0] for f in files)
        self.indices = sorted([int(idx) for idx, cnt in counter.items() if cnt == len(suffixes)])

    def __len__(self)-> int:
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        imgs = [
            self.transform(Image.open(f"{self.dir_path}/{idx}_{s}.png").convert("RGB"))
            for s in self.suffixes
        ]
        
        imgs = torch.stack(imgs, dim=0) # [6, C, H, W] normalized to [0,1]
        return imgs