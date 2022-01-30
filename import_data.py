# import_data.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Sun 30 Jan 2022 14:45:29 GMT

import gzip
from pathlib import Path
import numpy as np


def training_images():
    return get_images(
        Path(__file__).parent / "mnist_data/train-images-idx3-ubyte.gz"
    )


def test_images():
    return get_images(
        Path(__file__).parent / "mnist_data/t10k-images-idx3-ubyte.gz"
    )


def get_content(path):
    with gzip.open(path, "rb") as fh:
        content = fh.read()
    # Strip away magic number, number of images, and image dims
    # Each is a 32-bit int, so 4 bytes each -> 16 bytes
    return content[16:]


def get_images(path):
    content = get_content(path)
    # Extract each image as a (784, 1) numpy array
    return [np.array([c for c in content[i * 784:(i + 1) * 784]]).reshape(784, 1)
            for i in range(len(content) // 784)]
