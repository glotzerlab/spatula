import numpy as np
import scipy as sp
import scipy.special


class SphHarm:
    def __init__(self, max_l):
        self._max_l = max_l
        self._l, self._m = self.harmonic_indices(max_l)

    def __call__(self, theta, phi):
        # Note the different convention in theta and phi between scipy and this
        sph_ind, angle_ind = np.mgrid[0 : len(self._m), 0 : len(theta)]
        return sp.special.sph_harm(
            self._m[sph_ind], self._l[sph_ind], phi[angle_ind], theta[angle_ind]
        )

    @staticmethod
    def harmonic_indices(max_l):
        l = []
        m = []
        prev_m_length = 0
        for i in range(max_l + 1):
            m.extend(j for j in range(-i, i + 1))
            l.extend(i for _ in range(0, len(m) - prev_m_length))
            prev_m_length = len(m)
        return np.array(l, dtype=int), np.array(m, dtype=int)
