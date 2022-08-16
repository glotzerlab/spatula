import functools
import itertools

import numpy as np

from . import util

_tetrahedral = np.array(
    [
        # l = 0
        1,
        # l = 1
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # l = 2, mprime = -2
        0,
        0,
        0,
        0,
        0,
        # mprime = -1
        0,
        0,
        0,
        0,
        0,
        # mprime = -0
        0,
        0,
        0,
        0,
        0,
        # mprime = 1
        0,
        0,
        0,
        0,
        0,
        # mprime = 2
        0,
        0,
        0,
        0,
        0,
        # l = 3, mprime = -3
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # mprime = -2
        0,
        0.5,
        0,
        0,
        0,
        -0.5,
        0,
        # mprime = -1
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # mprime = 0
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # mprime = 1
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # mprime = 2
        0,
        -0.5,
        0,
        0,
        0,
        0.5,
        0,
        # mprime = 3
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # l = 4, mprime = -4
        5 / 24,
        0,
        0,
        0,
        np.sqrt(70) / 24,
        0,
        0,
        0,
        5 / 24,
        # l = 4, mprime = -3
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # l = 4, mprime = -2
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # l = 4, mprime = -1
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # l = 4, mprime = -0
        0,
        0,
        0,
        0,
        7 / 12,
        0,
        0,
        0,
        0,
        # l = 4, mprime = 1
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # l = 4, mprime = 2
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # l = 4, mprime = 3
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # l = 4, mprime = 4
        5 / 24,
        0,
        0,
        np.sqrt(70) / 24,
        0,
        0,
        0,
        0,
        5 / 24,
        # l = 5, mprime = -5
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # l = 5, mprime = -4
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # l = 5, mprime = -3
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # l = 5, mprime = -2
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # l = 5, mprime = -1
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # l = 5, mprime = 0
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # l = 5, mprime = 1
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # l = 5, mprime = 2
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # l = 5, mprime = 3
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # l = 5, mprime = 4
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # l = 5, mprime = 5
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # l = 6, mprime = -6
        5 / 32,
        0,
        0,
        0,
        -np.sqrt(55) / 32,
        0,
        0,
        0,
        -np.sqrt(55) / 32,
        0,
        0,
        0,
        5 / 32,
        # l = 6, mprime = -5
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # l = 6, mprime = -4
        0,
        0,
        7 / 16,
        0,
        0,
        0,
        -np.sqrt(14) / 16,
        0,
        0,
        0,
        7 / 16,
        0,
        0,
        # l = 6, mprime = -3
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # l = 6, mprime = -2
        0,
        0,
        0,
        0,
        11 / 32,
        0,
        0,
        0,
        11 / 32,
        0,
        0,
        0,
        0,
        # l = 6, mprime = -1
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # l = 6, mprime = 0
        0,
        0,
        0,
        0,
        0,
        0,
        1 / 8,
        0,
        0,
        0,
        0,
        0,
        0,
        # l = 6, mprime = 1
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # l = 6, mprime = 2
        0,
        0,
        0,
        0,
        11 / 32,
        0,
        0,
        0,
        11 / 32,
        0,
        0,
        0,
        0,
        # l = 6, mprime = 3
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # l = 6, mprime = 4
        0,
        0,
        7 / 16,
        0,
        0,
        0,
        -np.sqrt(14) / 16,
        0,
        0,
        0,
        7 / 16,
        0,
        0,
        # l = 6, mprime = 5
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        # l = 6, mprime = 6
        5 / 32,
        0,
        0,
        0,
        -np.sqrt(55) / 32,
        0,
        0,
        0,
        -np.sqrt(55) / 32,
        0,
        0,
        0,
        5 / 32,
    ]
)


class WeigerD:
    def __init__(self, max_l):
        if max_l > 6:
            raise ValueError("max_l cannot be greater than 6.")
        self._max_l = max_l

    def _cyclic(self, order, inverse):
        dij = (
            np.array(
                [
                    util.delta(mprime, m) * util.delta(m % order, 0)
                    for l, mprime, m in self._iter_sph_indices()
                ]
            )
            / order
        )
        if inverse:
            return self._inverse() * dij
        return dij

    def _diherdral(self, order, inverse):
        d_mprime_m_l = []
        for l, mprime, m in self._iter_sph_indices():
            if m % order:
                d_mprime_m_l.append(0)
                continue
            sign = 1 if l % 2 == 0 else -1
            d_mprime_m_l.append(
                0.5 * (util.delta(mprime, m) + sign * util.delta(mprime, -m))
            )
        dij = np.array(d_mprime_m_l) / (2 * order)
        if inverse:
            return self._inverse() * dij
        return dij

    def _tetrahedral(self):
        elems = 0
        for l in range(self._max_l + 1):
            elems += (2 * l + 1) ** 2
        return _tetrahedral[:elems] / 12

    def _inverse(self):
        return (
            np.array(
                [
                    util.delta(mprime, m) * util.delta(l % 2, 0)
                    for l, mprime, m in self._iter_sph_indices()
                ]
            )
            / 2
        )

    def _iter_sph_indices(self):
        for l in range(self._max_l + 1):
            ms = range(-l, l + 1)
            for mprime, m in itertools.product(ms, repeat=2):
                yield l, mprime, m

    @functools.lru_cache(maxsize=6)
    def qlm_indices(self, l=None):
        ind = 0
        indices = []
        for l in range(self._max_l + 1):
            ms = range(-l, l + 1)
            for mprime in ms:
                for ind_m in range(len(ms)):
                    indices.append(ind + ind_m)
            ind += len(ms)
        return np.array(indices)

    def __getitem__(self, sym):
        if sym == "T":
            return self._tetrahedral()
        if sym == "i":
            return self._inverse()
        inverse = "i" in sym
        order = int(sym[slice(2, None) if inverse else slice(1, None)])
        if sym[0] == "C":
            return self._cyclic(order, inverse)
        if sym[0] == "D":
            return self._diherdral(order, inverse)
        else:
            raise KeyError(f"Point group {sym} is not supported or does not exist.")

    def group_cardinality(self, sym):
        if sym == "T":
            return 12
        if sym == "i":
            return 2
        inverse = "i" in sym
        order = int(sym[slice(2, None) if inverse else slice(1, None)])
        inv_factor = 2 if inverse else 1
        if sym[0] == "C":
            return order * inv_factor
        if sym[0] == "D":
            return 2 * order * inv_factor
        else:
            raise ValueError(f"Point group {sym} is not supported or does not exist.")


def _weijer_qlm_sum(Dij_dot_qlms, max_l):
    size = sum(2 * l + 1 for l in range(max_l + 1))
    sym_qlm = np.empty(size, dtype=complex)
    start = 0
    summed_ind = 0
    for l in range(max_l + 1):
        skip = 2 * l + 1
        for ind_m in range(skip):
            sym_qlm[summed_ind] = Dij_dot_qlms[start : start + skip].sum()
            summed_ind += 1
            start += skip
    return sym_qlm


def symmetrize_qlm(qlms, Dij, weijer):
    cols = len(Dij)
    rows = sum(2 * l + 1 for l in range(weijer._max_l + 1))
    sym_qlm = np.empty((len(qlms), cols, rows), dtype=complex)
    qshape = qlms.shape
    dshape = Dij.shape
    dij_dot_qlms = qlms.reshape((qshape[0], 1, qshape[1]))[
        ..., weijer.qlm_indices()
    ] * Dij.reshape((1, dshape[0], dshape[1]))
    start = 0
    summed_ind = 0
    for l in range(weijer._max_l + 1):
        skip = 2 * l + 1
        for ind_m in range(skip):
            sym_qlm[..., summed_ind] = dij_dot_qlms[..., start : start + skip].sum(
                axis=-1
            )
            summed_ind += 1
            start += skip
    return sym_qlm


def _particle_symmetrize_qlm(qlms, Dij, weijer):
    cols = len(Dij)
    rows = sum(2 * l + 1 for l in range(weijer._max_l + 1))
    sym_qlm = np.empty((cols, rows), dtype=complex)
    qshape = qlms.shape
    dshape = Dij.shape
    dij_dot_qlms = qlms.reshape((1, -1))[..., weijer.qlm_indices()] * Dij
    start = 0
    summed_ind = 0
    for l in range(weijer._max_l + 1):
        skip = 2 * l + 1
        for ind_m in range(skip):
            sym_qlm[:, summed_ind] = dij_dot_qlms[:, start : start + skip].sum(axis=-1)
            summed_ind += 1
            start += skip
    return sym_qlm


def symmetrize_qlm_compiled(qlms, Dij, max_l):
    cols = len(Dij)
    rows = sum(2 * l + 1 for l in range(weijer._max_l + 1))
    sym_qlm = np.empty((len(qlms), cols, rows), dtype=complex)
    for particle_i in range(len(sym_qlm)):
        for sym_i in range(cols):
            sym_qlm_i = 0
            qlm_i = 0
            dij_i = 0
            for l in range(max_l + 1):
                ms = range(2 * l + 1)
                for mprime in ms:
                    tmp_sym_qlm = 0
                    for m in ms:
                        tmp_sym_qlm += (
                            Dij[sym_i, dij_i + m] * qlms[particle_i, qlm_i + m]
                        )
                    dij_i += len(ms)
                qlm_i += len(ms)
                sym_qlm[particle_i, sym_i, sym_qlm_i]
                sym_qlm_i += 1
    return sym_qlm
