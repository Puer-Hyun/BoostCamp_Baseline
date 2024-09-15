import os
import sys
from PIL import Image
import pytest
import numpy as np
from dotenv import load_dotenv
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torchvision.utils as vutils

# .env 파일 로드
load_dotenv()

sys.path.append("../../src")
from src.data.datasets.cv_level1_dataset import (  # noqa : E402
    CV_Level1_Dataset,
    CV_Level1_TestDataset,
)

# PROJECT_ROOT 환경 변수 가져오기
PROJECT_ROOT = os.getenv("PROJECT_ROOT")


@pytest.fixture
def cv_level1_dataset():
    csv_file = os.path.join(PROJECT_ROOT, "data/boostcamp_task1/train.csv")
    img_dir = os.path.join(PROJECT_ROOT, "data/boostcamp_task1/train")
    return CV_Level1_Dataset(csv_file, img_dir)


class ScaleToRange01(A.BasicTransform):
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        return (img - img.min()) / (img.max() - img.min())

    @property
    def targets(self):
        return {"image": self.apply}


def test_dataset_and_transformations(cv_level1_dataset):
    # 저장 경로 설정
    save_dir = "/data/seonghyun/BoostCamp_Baseline/visualization_example"
    original_dir = os.path.join(save_dir, "original")
    transformed_dir = os.path.join(save_dir, "transformed")

    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(transformed_dir, exist_ok=True)

    # Albumentations 변환 설정
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=480),
            A.PadIfNeeded(min_height=480, min_width=480, border_mode=0, value=0),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
            ScaleToRange01(),
        ]
    )

    for i in range(10):
        # 원본 이미지와 라벨 가져오기
        image, label = cv_level1_dataset[i]

        # 원본 이미지 저장
        image.save(os.path.join(original_dir, f"image_{i}_label_{label}.png"))

        # 이미지를 numpy 배열로 변환
        np_image = np.array(image)

        # Albumentations 변환 적용
        transformed = transforms(image=np_image)
        transformed_image = transformed["image"]

        # -1에서 1 사이의 값을 0에서 1 사이로 조정
        transformed_image = (transformed_image + 1) / 2

        # 변환된 이미지 저장
        vutils.save_image(
            transformed_image,
            os.path.join(transformed_dir, f"transformed_image_{i}_label_{label}.png"),
        )

        # 변환 전후 이미지 통계 출력
        print(f"Image {i}:")
        print(
            f"  Original - min: {np_image.min()}, "
            f"max: {np_image.max()}, "
            f"mean: {np_image.mean():.2f}"
        )
        print(
            f"  Transformed - min: {transformed_image.min():.4f}, "
            f"max: {transformed_image.max():.4f}, "
            f"mean: {transformed_image.mean():.4f}"
        )

    print(f"Images saved in {original_dir} and {transformed_dir}")


# 기존 테스트 함수들은 그대로 유지
def test_dataset_length(cv_level1_dataset):
    assert len(cv_level1_dataset) > 0, "Dataset should not be empty"


def test_dataset_item(cv_level1_dataset):
    item = cv_level1_dataset[0]
    assert len(item) == 2, "Dataset item should contain image and label"
    assert isinstance(item[0], Image.Image), "First item should be an image"
    assert isinstance(
        item[1], (int, np.integer)
    ), "Second item should be an integer or numpy integer"


def test_image_paths(cv_level1_dataset):
    for i in range(len(cv_level1_dataset)):
        _ = cv_level1_dataset[i]
        img_path = os.path.join(
            cv_level1_dataset.img_dir, cv_level1_dataset.data.iloc[i, 1]
        )
        assert os.path.exists(img_path), f"이미지 파일이 존재하지 않습니다: {img_path}"
