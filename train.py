import argparse
import importlib

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def main(config_path):
    # YAML 파일 로드
    config = OmegaConf.load(config_path)
    print(config)

    # 데이터 모듈 동적 임포트
    data_module_path, data_module_class = config.data_module.rsplit(".", 1)
    DataModuleClass = getattr(
        importlib.import_module(data_module_path), data_module_class
    )

    # 데이터 모듈 설정
    data_config_path = config.data_config_path
    augmentation_config_path = config.augmentation_config_path
    data_module = DataModuleClass(data_config_path, augmentation_config_path)
    data_module.setup()

    # 모델 모듈 동적 임포트
    model_module_path, model_module_class = config.model_module.rsplit(".", 1)
    ModelModuleClass = getattr(
        importlib.import_module(model_module_path), model_module_class
    )

    # 모델 설정
    model = ModelModuleClass(config)

    # 콜백 설정
    checkpoint_callback = ModelCheckpoint(
        monitor=config.callbacks.model_checkpoint.monitor,
        save_top_k=config.callbacks.model_checkpoint.save_top_k,
        mode=config.callbacks.model_checkpoint.mode,
    )
    early_stopping_callback = EarlyStopping(
        monitor=config.callbacks.early_stopping.monitor,
        patience=config.callbacks.early_stopping.patience,
        mode=config.callbacks.early_stopping.mode,
    )

    # 트레이너 설정
    trainer = pl.Trainer(
        **config.trainer, callbacks=[checkpoint_callback, early_stopping_callback]
    )

    # 훈련 시작
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with PyTorch Lightning")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()

    main(args.config)
