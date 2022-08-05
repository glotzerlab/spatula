import numpy as np
import scipy as sp
import scipy.stats

from . import util


class _Fisher:
    def __init__(self, kappa):
        self.kappa = kappa
        self.prefactor = kappa / (2 * np.pi * (np.exp(kappa) - np.exp(-kappa)))

    def __call__(self, angles):
        return self.prefactor * np.exp(self.kappa * np.cos(angles))


class BondOrderFisher:
    def __init__(self, theta, phi, kappa):
        theta = theta - np.pi / 2
        angle_shape = (1, -1)
        self._sin_theta = np.sin(theta).reshape(angle_shape)
        self._cos_theta = np.cos(theta).reshape(angle_shape)
        self._phi = phi.reshape(angle_shape)
        self._len = len(theta)
        self._dist = _Fisher(kappa)

    def __call__(self, theta, phi):
        angles = self._get_angles(theta, phi)
        return np.einsum("ij->i", self._dist(angles)) / self._len

    def _fast_call(self, sin_theta, cos_theta, phi):
        angles = self._fast_get_angles(sin_theta, cos_theta, phi)
        return np.einsum("ij->i", self._dist(angles)) / self._len

    def _fast_get_angles(self, sin_theta, cos_theta, phi):
        # assumes 1d arrays
        query_shape = (-1, 1)
        return util._central_angle_fast(
            sin_ta=self._sin_theta,
            sin_tb=sin_theta.reshape(query_shape),
            cos_ta=self._cos_theta,
            cos_tb=cos_theta.reshape(query_shape),
            phi_a=self._phi,
            phi_b=phi.reshape(query_shape)
        )
            
    def _get_angles(self, theta, phi):
        theta = theta - np.pi / 2
        return self._fast_get_angles(np.sin(theta), np.cos(theta), phi)


class BondOrderUniform:
    def __init__(self, theta, phi, max_theta):
        self._theta = theta
        self._phi = phi
        self._max_theta = max_theta
        self._len = len(theta)
        great_circle_area = 2 * np.pi * (1 - np.cos(max_theta))
        self._normalization = 1 / great_circle_area

    def __call__(self, theta, phi):
        i_query, i_points = np.mgrid[0 : len(theta), 0 : len(self._theta)]
        angles = util.central_angle(
            self._theta[i_points],
            self._phi[i_points],
            theta[i_query],
            phi[i_query],
        )
        return np.einsum("ij->i", self._dist(angles)) / self._len

    def _dist(self, theta):
        return np.where(theta <= self._max_theta, self._normalization, 0)
