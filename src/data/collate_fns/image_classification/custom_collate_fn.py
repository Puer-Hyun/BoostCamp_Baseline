import torch
import numpy as np


def image_classification_collate_fn(batch):
    images, labels = zip(*batch)

    # 이미지가 이미 텐서인 경우와 numpy 배열인 경우를 모두 처리
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images)
    else:
        images = torch.stack(
            [
                (
                    torch.from_numpy(img)
                    if isinstance(img, np.ndarray)
                    else torch.tensor(img)
                )
                for img in images
            ]
        )

    # 레이블을 텐서로 변환 (정수형이라고 가정)
    labels = torch.tensor(labels)

    return images, labels


def predict_collate_fn(batch):
    images, file_names = zip(*batch)

    # 이미지가 이미 텐서인 경우와 numpy 배열인 경우를 모두 처리
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images)
    else:
        images = torch.stack(
            [
                (
                    torch.from_numpy(img)
                    if isinstance(img, np.ndarray)
                    else torch.tensor(img)
                )
                for img in images
            ]
        )

    # 파일 이름은 리스트로 유지
    return images, list(file_names)
