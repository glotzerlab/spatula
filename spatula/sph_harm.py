# Copyright (c) 2021-2026 The Regents of the University of Michigan
# Part of spatula, released under the BSD 3-Clause License.

"""Provide helper class for computing spherical harmonics."""

import numpy as np
import scipy as sp
import scipy.special


class SphHarm:
    r"""Compute spherical harmonics up to a given spherical harmonic number.

    The class always computes all values of the spherical harmonics (all
    :math:`m` and :math:`\ell`) upto the maximum :math:`\ell`.
    """

    def __init__(self, max_l):
        """Create a SphHarm object.

        Parameters
        ----------
        max_l : int
            The highest order spherical harmonics to calculate. Class does not
            support values higher than 12.

        """
        self._max_l = max_l
        self._l, self._m = self.harmonic_indices

    def __call__(self, theta, phi):
        r"""Compute all spherical harmonics for given polar and azimuthal angles.

        Note
        ----
        ``theta`` and ``phi`` are expected to have the same size and correspond
        to each other such that the first value in the returned array
        corresponds to the spherical harmonic from the first index of both
        arrays.

        Parameters
        ----------
        theta : :math:`(N,1)` numpy.ndarray of float
            The polar angles to compute.
        phi : :math:`(N,1)` numpy.ndarray of float
            The azimuthal angles to compute.

        Returns
        -------
        harmonics : :math:`(N_{harm}, N_{angle})` numpy.ndarray of float
            The spherical harmonics. The first dimension is in increasing
            :math:`\ell` and :math:`m`, and the second in the order of pased
            angles. Use `~.harmonic_indices` to examine the order of the first
            dimension further.

        """
        # Note the different convention in theta and phi between scipy and this
        sph_ind, angle_ind = np.mgrid[0 : len(self._m), 0 : len(theta)]
        return sp.special.sph_harm_y(
            self._l[sph_ind], self._m[sph_ind], theta[angle_ind], phi[angle_ind]
        )

    @property
    def harmonic_indices(self):
        """Returns harmonic indices.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]: indices for the values returned by this
        class.

        """
        l = []
        m = []
        prev_m_length = 0
        for i in range(self._max_l + 1):
            m.extend(j for j in range(-i, i + 1))
            l.extend(i for _ in range(0, len(m) - prev_m_length))
            prev_m_length = len(m)
        return np.array(l, dtype=int), np.array(m, dtype=int)
