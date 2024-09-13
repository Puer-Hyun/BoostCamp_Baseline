import os
import sys
from PIL import Image
import pytest
import numpy as np

sys.path.append("../../src")
from src.data.datasets.cv_level1_dataset import (  # noqa: E402
    CV_Level1_Dataset,
    CV_Level1_TestDataset,
)


@pytest.fixture
def cv_level1_dataset():
    csv_file = (
        "/Users/seonghyunpark/Desktop/BoostCamp_Baseline"
        "/data/boostcamp_task1/data/train.csv"
    )
    img_dir = (
        "/Users/seonghyunpark/Desktop/BoostCamp_Baseline"
        "/data/boostcamp_task1/data/train"
    )
    return CV_Level1_Dataset(csv_file, img_dir)


def test_dataset_length(cv_level1_dataset):
    assert len(cv_level1_dataset) > 0, "Dataset should not be empty"


def test_dataset_item(cv_level1_dataset):
    item = cv_level1_dataset[0]
    print(item)
    assert len(item) == 2, "Dataset item should contain image and label"
    assert isinstance(item[0], Image.Image), "First item should be an image"
    print(item[1])
    assert isinstance(
        item[1], (int, np.integer)
    ), "Second item should be an integer or numpy integer"


def test_image_paths(cv_level1_dataset):
    for i in range(len(cv_level1_dataset)):
        _ = cv_level1_dataset[i]
        img_path = os.path.join(
            cv_level1_dataset.img_dir, cv_level1_dataset.data.iloc[i, 1]
        )
        assert os.path.exists(img_path), f"Image file does not exist: {img_path}"
