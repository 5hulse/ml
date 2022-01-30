# import_data.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Sun 30 Jan 2022 14:37:22 GMT

import gzip
import numpy as np


def training_images():
    return get_images("train-images-idx3-ubyte.gz", 60000)


def test_images():
    return get_images("t10k-images-idx3-ubyte.gz", 10000)


def get_images(filename, n):
    with gzip.open(f"mnist_data/{filename}", "rb") as fh:
        content = fh.read()
    # Strip away magic number, number of images, and image dims
    # Each is a 32-bit int, so 4 bytes each -> 16 bytes
    content = content[16:]

    # Extract each image as a (784, 1) numpy array
    return [np.array([c for c in content[i * 784:(i + 1) * 784]]).reshape(784, 1)
            for i in range(n)]
