# generate_fids.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 15 Jun 2022 00:16:08 BST

from typing import Tuple

import nmrespy as ne
import numpy as np


def random_array(
    size: int,
    amp_range: Tuple[float, float],
    phase_range: Tuple[float, float],
    freq_range: Tuple[float, float],
    damp_range: Tuple[float, float],
) -> np.ndarray:
    arr = np.zeros((size, 4), dtype="float64")
    arr[:, 0] = np.random.uniform(low=amp_range[0], high=amp_range[1], size=size)
    arr[:, 1] = np.random.uniform(low=phase_range[0], high=phase_range[1], size=size)
    arr[:, 2] = np.random.uniform(low=freq_range[0], high=freq_range[1], size=size)
    arr[:, 3] = np.random.uniform(low=damp_range[0], high=damp_range[1], size=size)
    return arr


def generate_fids(
    expinfo: ne.ExpInfo,
    labels: np.ndarray,
) -> np.ndarray:
    size = labels.shape[0]
    fids = np.zeros((2 * expinfo.default_pts[0], size), dtype="float64")
    for i in range(size):
        if i % 1000 == 0:
            print(i)
        fid = expinfo.make_fid(np.expand_dims(labels[i], axis=0))
        fids[::2, i] = fid.real
        fids[1::2, i] = fid.imag
    return fids


expinfo = ne.ExpInfo(1, 5000., 2500., 500., "1H", 256)
train_size = 50000
test_size = 10000
a_range = (1., 10.)
phi_range = (-np.pi, np.pi)
f_range = (
    expinfo.sw()[0] - expinfo.offset()[0] / 2,
    expinfo.sw()[0] + expinfo.offset()[0] / 2,
)
eta_range = (8., 16.)

train_labels = random_array(train_size, a_range, phi_range, f_range, eta_range)
test_labels = random_array(test_size, a_range, phi_range, f_range, eta_range)
train_fids = generate_fids(expinfo, train_labels)
test_fids = generate_fids(expinfo, test_labels)

np.save("train_labels.npy", train_labels)
np.save("test_labels.npy", test_labels)
np.save("train_fids.npy", train_fids)
np.save("test_fids.npy", test_fids)
