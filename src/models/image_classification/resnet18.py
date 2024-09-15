import torch
from torch import nn
import timm


class ResNet18Model(nn.Module):
    def __init__(self, config):
        super(ResNet18Model, self).__init__()
        self.config = config

        # 클래스 수를 설정에서 가져옴.
        num_classes = config.get("num_classes", 500)

        # timm을 사용하여 resnet18 모델을 불러오고, 출력 클래스 수를 설정함.
        self.model = timm.create_model(
            "resnet18", pretrained=True, num_classes=num_classes
        )

        # 가중치 초기화 메서드 호출 (필요한 경우 추가 설정 가능)
        self._initialize_weights()

    def forward(self, x):
        return self.model(x)

    def _initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
