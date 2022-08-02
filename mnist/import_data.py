# import_data.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 02 Aug 2022 23:49:39 BST

from __future__ import annotations
from dataclasses import dataclass
import gzip
from pathlib import Path
from typing import Iterable, Tuple
import random

import numpy as np


DATADIR = Path(__file__).parent / "data"


def training_images():
    return get_images(DATADIR / "train-images-idx3-ubyte.gz")


def test_images():
    return get_images(DATADIR / "t10k-images-idx3-ubyte.gz")


def training_labels():
    return get_labels(DATADIR / "train-labels-idx1-ubyte.gz")


def test_labels():
    return get_labels(DATADIR / "t10k-labels-idx1-ubyte.gz")


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
    return np.hstack([
        np.array([c for c in content[i * 784:(i + 1) * 784]]).reshape(784, 1) / 256
        for i in range(len(content) // 784)
    ])


def get_labels(path):
    content = get_content(path)
    # Strip away magic number, number of labels
    # Each is a 32-bit int, so 2 bytes each -> 8 bytes
    content = content[8:]
    labels = [c for c in content]
    targets = np.zeros((10, len(labels)))
    for i, label in enumerate(labels):
        targets[label, i] = 1
    return targets


@dataclass
class Dataset:
    data: np.ndarray
    labels: np.ndarray

    @property
    def size(self) -> int:
        return self.data.shape[1]

    @property
    def shape(self) -> int:
        return self.data.shape

    def shuffle(self) -> None:
        ordering = list(range(self.size))
        random.shuffle(ordering)
        idx = np.empty_like(ordering)
        idx[ordering] = np.arange(self.size)
        self.data = self.data[:, idx]
        self.labels = self.labels[:, idx]

    def make_batches(self, batch_size: int) -> Iterable[Dataset]:
        self.shuffle()
        return [
            Dataset(
                self.data[:, i : i + batch_size],
                self.labels[:, i : i + batch_size],
            )
            for i in range(0, self.size, batch_size)
        ]

    def get_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return Dataset(self.data[:, idx, None], self.labels[:, idx, None])


training_images = training_images()
training_labels = training_labels()

TRAINING_DATA = Dataset(
    training_images[:, :50000],
    training_labels[:, :50000],
)

VALIDATION_DATA = Dataset(
    training_images[:, 50000:],
    training_labels[:, 50000:],
)

TEST_DATA = Dataset(
    test_images(),
    test_labels(),
)
