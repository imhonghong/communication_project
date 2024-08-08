import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class IsoDataset(Dataset):
    def __init__(self, input_dir1, input_dir2, output_dir, transform=None):
        self.input_dir1 = input_dir1
        self.input_dir2 = input_dir2
        self.output_dir = output_dir
        self.input_files1 = sorted(os.listdir(input_dir1))
        self.input_files2 = sorted(os.listdir(input_dir2))
        self.output_files = sorted(os.listdir(output_dir))
        self.transform = transform

    def __len__(self):
        return len(self.input_files1)

    def __getitem__(self, idx):
        input_path1 = os.path.join(self.input_dir1, self.input_files1[idx])
        input_path2 = os.path.join(self.input_dir2, self.input_files2[idx])
        output_path = os.path.join(self.output_dir, self.output_files[idx])
        
        input_image1 = np.load(input_path1).astype(np.float32)
        input_image2 = np.load(input_path2).astype(np.float32)
        output_image = np.load(output_path).astype(np.float32)
        
        input_image = np.stack((input_image1,input_image2), axis=-1)
        input_image = torch.from_numpy(input_image).permute(2, 0, 1)  # Change from (H, W, C) to (C, H, W)
        output_image = torch.from_numpy(output_image).unsqueeze(0)  # Add channel dimension

        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)
            
        return input_image, output_image