# import_data.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Sun 30 Jan 2022 14:49:43 GMT

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


def training_labels():
    return get_labels(
        Path(__file__).parent / "mnist_data/train-labels-idx1-ubyte.gz"
    )


def test_labels():
    return get_labels(
        Path(__file__).parent / "mnist_data/t10k-labels-idx1-ubyte.gz"
    )


def get_content(path):
    with gzip.open(path, "rb") as fh:
        content = fh.read()
    return content


def get_images(path):
    content = get_content(path)
    # Strip away magic number, number of images, and image dims
    # Each is a 32-bit int, so 4 bytes each -> 16 bytes
    content = content[16:]
    # Extract each image as a (784, 1) numpy array
    return [np.array([c for c in content[i * 784:(i + 1) * 784]]).reshape(784, 1)
            for i in range(len(content) // 784)]


def get_labels(path):
    content = get_content(path)
    # Strip away magic number, number of labels
    # Each is a 32-bit int, so 2 bytes each -> 8 bytes
    content = content[8:]
    return [c for c in content]
