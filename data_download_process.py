import os
import subprocess
import tarfile
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from PIL import Image


def download_cifar10(raw_dir):
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    response = requests.get(url)
    file = tarfile.open(fileobj=BytesIO(response.content))
    file.extractall(path=raw_dir)
    file.close()


def process_cifar10(raw_dir, processed_dir):
    cifar_dir = os.path.join(raw_dir, "cifar-10-batches-py")
    images_dir = os.path.join(raw_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    metadata = []

    for batch_file in [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    ]:
        with open(os.path.join(cifar_dir, batch_file), "rb") as f:
            batch_data = np.load(f, allow_pickle=True, encoding="bytes")

        for i in range(10000):
            img = batch_data[b"data"][i].reshape(3, 32, 32).transpose(1, 2, 0)
            img = Image.fromarray(img)
            img_name = f"{batch_file}_{i}.png"
            img.save(os.path.join(images_dir, img_name))

            metadata.append(
                {
                    "image_name": img_name,
                    "label": batch_data[b"labels"][i],
                    "batch_file": batch_file,
                }
            )

    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(raw_dir, "metadata.csv"), index=False)


def setup_dvc():
    subprocess.run(["dvc", "init"])
    subprocess.run(["dvc", "add", "data/raw"])
    subprocess.run(["git", "add", "data/.gitignore", "data/raw.dvc"])
    subprocess.run(["git", "commit", "-m", "Add raw data"])


if __name__ == "__main__":
    raw_dir = "data/raw"
    processed_dir = "data/processed"

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    print("Downloading CIFAR-10 dataset...")
    download_cifar10(raw_dir)

    print("Processing CIFAR-10 dataset...")
    process_cifar10(raw_dir, processed_dir)

    print("Setting up DVC...")
    setup_dvc()

    print("Data download and processing complete!")
