import os
from datasets import load_dataset
from PIL import Image

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