"""
train.py를 실행시키기만 하면 다 돌아가도록.
이때 옵션으로 모델, 데이터, 옵티마이저,
데이터 로더 등을 설정할 수 있도록 yaml 파일을 입력으로 받는다.
"""

import argparse

import pytorch_lightning as pl
import yaml
from omegaconf import OmegaConf


def main(config_path):
    # YAML 파일 로드
    config = OmegaConf.load(config_path)
    print(config)

    # 여기에 PyTorch Lightning을 사용한 훈련 로직을 추가
    # 예: 모델 정의, 데이터 모듈 설정, Trainer 설정 등


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with PyTorch Lightning")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()

    main(args.config)
