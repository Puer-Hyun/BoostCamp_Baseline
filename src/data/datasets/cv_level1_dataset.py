import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CV_Level1_Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 1])
        image = Image.open(img_name).convert("RGB")
        label = self.data.iloc[idx, 2]

        if self.transform:
            image = self.transform(image)

        return image, label
