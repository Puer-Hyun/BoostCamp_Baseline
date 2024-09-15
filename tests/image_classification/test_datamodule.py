# 프로젝트 루트를 Python 경로에 추가
import sys
import pytest
import torch
from omegaconf import OmegaConf
from pathlib import Path
import os
import torchvision.utils as vutils
import numpy as np

# 환경 변수 로드
from src.utils.path_utils import load_env_vars

load_env_vars()

sys.path.append(os.environ["PROJECT_ROOT"])

from src.data.custom_datamodules.image_classification.imageclassification_datamodule import (  # noqa : E402, E501
    ImageClassificationDataModule,
)


@pytest.fixture
def data_config():
    project_root = Path(os.environ["PROJECT_ROOT"])
    return OmegaConf.create(
        {
            "data": {
                "csv_file_path": str(project_root / "data/boostcamp_task1/train.csv"),
                "image_dir_path": str(project_root / "data/boostcamp_task1/train"),
                "predict_image_dir_path": str(
                    project_root / "data/boostcamp_task1/test"
                ),
                "train_val_split": 0.8,
                "val_test_split": 0.7,
                "pin_memory": True,
                "train": {"batch_size": 64, "num_workers": 11, "shuffle": True},
                "val": {"batch_size": 32, "num_workers": 2, "shuffle": False},
                "test": {"batch_size": 1, "num_workers": 2, "shuffle": False},
                "predict": {"batch_size": 1, "num_workers": 2, "shuffle": False},
            }
        }
    )


@pytest.fixture
def augmentation_config():
    return OmegaConf.create(
        {
            "augmentation": {
                "use_augmentation": True,
                "transforms": [
                    {"name": "HorizontalFlip", "params": {"p": 0.5}},
                    {"name": "Rotate", "params": {"limit": 15, "p": 0.5}},
                ],
            }
        }
    )


def test_image_classification_datamodule(data_config, augmentation_config):
    project_root = Path(os.environ["PROJECT_ROOT"])

    data_module = ImageClassificationDataModule(
        data_config_path=str(
            project_root / "configs/data_configs/image_classification_contest.yaml"
        ),
        augmentation_config_path=str(
            project_root / "configs/augmentation_configs/image_classification.yaml"
        ),
        seed=42,
    )

    # 설정 파일 대신 fixture를 사용
    data_module.data_config = data_config
    data_module.augmentation_config = augmentation_config

    # setup 메소드 호출
    data_module.setup()

    # 데이터셋 생성 확인
    assert data_module.train_dataset is not None
    assert data_module.val_dataset is not None
    assert data_module.test_dataset is not None
    assert data_module.predict_dataset is not None

    # 데이터로더 생성 및 확인
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    predict_loader = data_module.predict_dataloader()

    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)
    assert isinstance(test_loader, torch.utils.data.DataLoader)
    assert isinstance(predict_loader, torch.utils.data.DataLoader)

    # 배치 크기 확인
    assert train_loader.batch_size == data_config.data.train.batch_size
    assert val_loader.batch_size == data_config.data.val.batch_size
    assert test_loader.batch_size == data_config.data.test.batch_size
    assert predict_loader.batch_size == data_config.data.predict.batch_size

    # 데이터 샘플 확인
    batch = next(iter(train_loader))
    assert len(batch) == 2  # (images, labels)
    assert isinstance(batch[0], torch.Tensor)  # images
    assert isinstance(batch[1], torch.Tensor)  # labels

    # 이미지와 라벨 값 확인
    images, labels = batch
    assert images.shape == (
        data_config.data.train.batch_size,
        3,
        480,
        480,
    )  # 이미지 크기 확인 (채널, 높이, 너비)
    assert labels.shape == (data_config.data.train.batch_size,)  # 라벨 shape 확인
    assert (
        labels.min() >= 0 and labels.max() < 500
    )  # 라벨 값 범위 확인 (0~499, 500개 클래스 가정)

    # 이미지 저장
    save_path = Path("/data/seonghyun/BoostCamp_Baseline/visualization_example")
    save_path.mkdir(parents=True, exist_ok=True)

    # 배치의 이미지를 그리드로 만들어 저장
    grid = vutils.make_grid(images[:16], nrow=4, normalize=True, padding=2)
    vutils.save_image(grid, str(save_path / "batch_images.png"))

    # 개별 이미지 저장 (첫 5개만)
    for i in range(min(10, len(images))):
        img = images[i].cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
        img = (img * 255).astype(np.uint8)
        vutils.save_image(
            images[i], str(save_path / f"image_{i}_label_{labels[i].item()}.png")
        )

    print(f"이미지가 {save_path}에 저장되었습니다.")

    # predict 데이터셋 확인
    predict_batch = next(iter(predict_loader))
    assert len(predict_batch) == 2  # (images, file_names)
    assert isinstance(predict_batch[0], torch.Tensor)  # images
    assert isinstance(predict_batch[1], list)  # file_names
