import os
from typing import Optional
from pathlib import Path

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.data.base_datamodule import BaseDataModule
from src.data.collate_fns.image_classification.custom_collate_fn import (
    image_classification_collate_fn,
    predict_collate_fn,
)
from src.data.datasets.cv_level1_dataset import CV_Level1_Dataset, CV_Level1_TestDataset
from src.utils.data_utils import load_yaml_config
from src.utils.path_utils import load_env_vars

load_env_vars()


class ImageClassificationDataModule(BaseDataModule):
    def __init__(self, data_config_path: str, augmentation_config_path: str, seed: int):
        project_root = Path(os.environ["PROJECT_ROOT"])
        self.data_config = load_yaml_config(project_root / data_config_path)
        self.augmentation_config = load_yaml_config(
            project_root / augmentation_config_path
        )
        self.seed = seed
        super().__init__(self.data_config)

    def setup(self, stage: Optional[str] = None):
        # 시드 설정
        torch.manual_seed(self.seed)

        # ImageNet 스타일 데이터셋에 대한 일반적인 정규화 값
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.augmentation_config["augmentation"]["use_augmentation"]:
            train_transforms = self._get_augmentation_transforms(mean, std)
        else:
            train_transforms = self._get_base_transforms(mean, std)

        test_transforms = self._get_base_transforms(mean, std)

        # Load datasets
        project_root = Path(os.environ["PROJECT_ROOT"])
        csv_file = project_root / self.config["data"]["csv_file_path"]
        img_dir = project_root / self.config["data"]["image_dir_path"]

        full_dataset = CV_Level1_Dataset(
            csv_file=csv_file, img_dir=img_dir, transform=train_transforms
        )

        # Split dataset into train, validation, and test
        total_size = len(full_dataset)
        train_size = int(total_size * self.config["data"]["train_val_split"])
        val_size = int(total_size * self.config["data"]["val_test_split"])
        test_size = total_size - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = (
            torch.utils.data.random_split(
                full_dataset, [train_size, val_size, test_size]
            )
        )

        # Set transforms
        self.val_dataset.dataset.transform = test_transforms
        self.test_dataset.dataset.transform = test_transforms

        # Prediction dataset
        predict_img_dir = project_root / self.config["data"]["predict_image_dir_path"]
        self.predict_dataset = CV_Level1_TestDataset(
            img_dir=predict_img_dir, transform=test_transforms
        )

    def _get_base_transforms(self, mean, std):
        return A.Compose(
            [
                A.LongestMaxSize(max_size=224),
                A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=0),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    def _get_augmentation_transforms(self, mean, std):
        aug_list = [
            A.LongestMaxSize(max_size=224),
            A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=0),
        ]
        for transform_config in self.augmentation_config["augmentation"]["transforms"]:
            aug_class = getattr(A, transform_config["name"])
            aug_list.append(aug_class(**transform_config["params"]))

        aug_list.extend(
            [
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

        return A.Compose(aug_list)

    class PrintImageStats(A.BasicTransform):
        def __init__(self, always_apply=False, p=1.0):
            super().__init__(always_apply, p)

        def apply(self, img, **params):
            if isinstance(img, torch.Tensor):
                print(
                    f"Image stats: min={img.min().item():.4f}, "
                    f"max={img.max().item():.4f}, "
                    f"mean={img.mean().item():.4f}"
                )
            else:
                print(
                    f"Image stats: min={img.min():.4f}, "
                    f"max={img.max():.4f}, "
                    f"mean={img.mean():.4f}"
                )
            return img

        @property
        def targets(self):
            return {"image": self.apply}

        def get_transform_init_args_names(self):
            return ("always_apply", "p")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config["data"]["train"]["batch_size"],
            num_workers=self.config["data"]["train"]["num_workers"],
            shuffle=self.config["data"]["train"]["shuffle"],
            collate_fn=self.debug_collate_fn,  # 디버깅을 위한 커스텀 콜레이트 함수 사용
            persistent_workers=True,
            pin_memory=self.config["data"]["pin_memory"],
        )

    def debug_collate_fn(self, batch):
        images, labels = image_classification_collate_fn(batch)
        return images, labels

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config["data"]["val"]["batch_size"],
            num_workers=self.config["data"]["val"]["num_workers"],
            shuffle=self.config["data"]["val"]["shuffle"],
            collate_fn=image_classification_collate_fn,
            persistent_workers=True,
            pin_memory=self.config["data"]["pin_memory"],
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config["data"]["test"]["batch_size"],
            num_workers=self.config["data"]["test"]["num_workers"],
            shuffle=self.config["data"]["test"]["shuffle"],
            collate_fn=image_classification_collate_fn,
            persistent_workers=True,
            pin_memory=self.config["data"]["pin_memory"],
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.config["data"]["predict"]["batch_size"],
            num_workers=self.config["data"]["predict"]["num_workers"],
            shuffle=self.config["data"]["predict"]["shuffle"],
            collate_fn=predict_collate_fn,  # 별도의 collate_fn 사용
            persistent_workers=True,
            pin_memory=self.config["data"]["pin_memory"],
        )
