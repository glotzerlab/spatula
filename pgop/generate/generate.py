"""Generates WignerD matrices for use in the main code base."""
import itertools
from pathlib import Path

import numpy as np
import h5py
import quaternionic
import spherical
import util
from rotations import PointGroupRotations

PG_ROTATION_CACHE = PointGroupRotations()


POINT_GROUPS = itertools.chain(
    map("".join, itertools.product("C", map(str, range(2, 13)))),
    map("".join, itertools.product("D", map(str, range(2, 13)))),
    ("T", "O", "I"),
    ("Ci",),
)


def inverse(max_l):
    """Return the WignerD matrix for Ci up to the given l."""

    def delta(a, b):
        return int(a == b)

    return (
        np.array(
            [
                delta(mprime, m) * delta(l % 2, 0)
                for l, mprime, m in util.iter_sph_indices(max_l)
            ],
            dtype=complex,
        )
        / 2
    )


def generate_wignerd(schonflies_symbol, max_l):
    """Generate a single WignerD matrix for the given point group."""
    if schonflies_symbol == "Ci":
        return inverse(max_l)
    symmetry_operations = PG_ROTATION_CACHE[schonflies_symbol]
    wigner = spherical.Wigner(max_l)
    quaternions = quaternionic.array(symmetry_operations)
    return wigner.D(quaternions).sum(axis=0) / len(symmetry_operations)


def generate_multiple(schonflies_symbols, max_l):
    """Generate a multiple WignerD matrices for the given point groups."""
    return {sym: generate_wignerd(sym, max_l) for sym in schonflies_symbols}


def save_wignerd(matrices, fn, mode, compression_level=5):
    """Save the WignerD matrices into an HDF5 file with compression."""
    with h5py.File(fn, mode) as f:
        for i, matrix in enumerate(matrices):
            # Creating a dataset for each matrix with gzip compression
            f.create_dataset(
                str(i),
                data=np.array(matrix),
                compression="gzip",
                compression_opts=compression_level,
            )


if __name__ == "__main__":
    src = Path(__file__).parents[1]
    save_wignerd(generate_multiple(POINT_GROUPS, 12), src / "data.h5", "w")
