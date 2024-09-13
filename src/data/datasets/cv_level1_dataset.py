import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class CV_Level1_Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.img_dir, self.data.iloc[idx, 1]
        )  # image_path 열 사용
        image = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx, 2]  # target 열 사용

        if self.transform:
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


class CV_Level1_TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = [
            f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, self.image_files[idx]
