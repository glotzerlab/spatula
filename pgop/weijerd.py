import collections.abc
import itertools
from pathlib import Path
import h5py
import numpy as np

from . import _pgop


class _WignerData(collections.abc.Mapping):
    def __init__(self):
        # Open the HDF5 file
        with h5py.File(Path(__file__).parent / "data.h5", "r") as f:
            # Get the data
            dataset = f["/data/matrices"]
            self._data = dataset[:]
            self._columns = dataset.attrs["point_groups"]

    def __getitem__(self, key):
        if key not in self._columns:
            raise KeyError(f"WignerD matrix for point group {key} not found.")
        idx = np.flatnonzero(self._columns == key)[0]
        return self._data[idx, :]

    def __len__(self):
        return len(self._columns)

    def __contains__(self, key):
        return key in self._columns

    def __iter__(self):
        yield from self._columns


def _parse_point_group(schonflies_symbol):
    family = schonflies_symbol[0]
    if len(schonflies_symbol) == 1:
        return (family, None, None)
    modifier = (
        schonflies_symbol[-1] if schonflies_symbol[-1].isalpha() else None
    )
    if len(schonflies_symbol) == 2 and modifier is not None:
        return (family, modifier, None)
    order = (
        int(schonflies_symbol[1:-1]) if modifier else int(schonflies_symbol[1:])
    )
    return (family, modifier, order)


class WeigerD:
    _MAX_L = 12

    def __init__(self, max_l):
        if max_l > self._MAX_L:
            raise ValueError(f"Maximum supported l value is {self._MAX_L}.")
        self._max_l = max_l
        self._index = slice(0, self._compute_last_index(max_l))
        self._data = _WignerData()
    @staticmethod
    def _compute_last_index(max_l):
        return sum((2 * l + 1) ** 2 for l in range(0, max_l + 1))

    def __getitem__(self, point_group):
        family, modifier, order = _parse_point_group(point_group)
        if family in "TOI":
            if modifier == "h":
                return _pgop.wignerD_semidirect_prod(
                    self._data[family][self._index],
                    self._data["Ci"][self._index],
                )
            elif modifier is not None:
                raise KeyError(f"{point_group} is not currently supported.")
            return self._data[family][self._index]
        if family in "CD":
            if family == "C" and modifier == "i":
                return self._data["Ci"]
            if family == "C" and modifier == "h":
                return _pgop.wignerD_semidirect_prod(
                    self._data[family + str(order)][self._index],
                    self._data["Ci"][self._index],
                )
            if modifier is not None:
                raise KeyError(f"{point_group} is not currently supported.")
            return self._data[family + str(order)][self._index]

    def _iter_sph_indices(self):
        for l in range(self._max_l + 1):
            ms = range(-l, l + 1)
            for mprime, m in itertools.product(ms, repeat=2):
                yield l, mprime, m
