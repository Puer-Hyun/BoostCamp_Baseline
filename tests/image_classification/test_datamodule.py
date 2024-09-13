import pytest
import torch
from omegaconf import OmegaConf
import sys

sys.path.append("/Users/seonghyunpark/Desktop/BoostCamp_Baseline")
from src.data.custom_datamodules.image_classification.imageclassification_datamodule import (  # noqa: E402, E501
    ImageClassificationDataModule,
)


@pytest.fixture
def data_config():
    return OmegaConf.create(
        {
            "data": {
                "csv_file_path": (
                    "/Users/seonghyunpark/Desktop/BoostCamp_Baseline/"
                    "data/boostcamp_task1/data/train.csv"
                ),
                "image_dir_path": (
                    "/Users/seonghyunpark/Desktop/BoostCamp_Baseline/"
                    "data/boostcamp_task1/data/train"
                ),
                "predict_dir_path": (
                    "/Users/seonghyunpark/Desktop/BoostCamp_Baseline/"
                    "data/boostcamp_task1/data/test"
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
    data_module = ImageClassificationDataModule(
        data_config_path=(
            "/Users/seonghyunpark/Desktop/BoostCamp_Baseline/"
            "configs/data_configs/image_classification_contest.yaml"
        ),
        augmentation_config_path=(
            "/Users/seonghyunpark/Desktop/BoostCamp_Baseline/"
            "configs/augmentation_configs/image_classification.yaml"
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

    # predict 데이터셋 확인
    predict_batch = next(iter(predict_loader))
    assert len(predict_batch) == 2  # (images, file_names)
    assert isinstance(predict_batch[0], torch.Tensor)  # images
    assert isinstance(predict_batch[1], list)  # file_names
