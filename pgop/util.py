"""General utility functions for the package and users."""

import collections

import numpy as np
import operators

from . import _pgop

PI_2 = np.pi / 2
PI_4 = np.pi / 4


def sph_to_cart(theta, phi):
    r"""Convert spherical to Cartesian coordinates on the unit sphere.

    Parameter
    ---------
    theta: :math:`(N,)` numpy.ndarray of float
        The longitudinal (polar) angle from :math:`[0, \pi]`.
    phi : :math:`(N,)` numpy.ndarray of float
        The latitudinal (azimuthal) angle from :math:`[0, 2 \pi]`.

    Returns
    -------
    coords : :math:`(N, 3)` numpy.ndarray of float
        The Cartesian coordinates of the points.
    """
    x = np.empty(theta.shape + (3,))
    sin_theta = np.sin(theta)
    x[..., 0] = sin_theta * np.cos(phi)
    x[..., 1] = sin_theta * np.sin(phi)
    x[..., 2] = np.cos(theta)
    return x


def set_num_threads(num_threads):
    """Set the number of threads to use when computing PGOP.

    Parameters
    ----------
    num_threads : int
        The number of threads.
    """
    if num_threads < 1:
        raise ValueError("Must set to a positive number of threads.")
    try:
        num_threads = int(num_threads)
    except ValueError as err:
        raise ValueError("num_threads must be convertible to an int") from err
    _pgop.set_num_threads(num_threads)


def get_num_threads():
    """Get the number of threads used when computing PGOP.

    Returns
    -------
    num_threads : int
        The number of threads.
    """
    return _pgop.get_num_threads()


class _Cache:
    def __init__(self, max_size=None):
        self._data = {}
        self._key_counts = collections.Counter()
        self._recent_keys = collections.deque()
        self.max_size = max_size

    def __contains__(self, key):
        return key in self._data

    def __getitem__(self, key):
        data = self._data.get(key, None)
        if data is None:
            return data
        if self.max_size is not None:
            self._key_counts[key] += 1
            self._recent_keys.append(key)
            if self.max_size // 10 > len(self._recent_keys):
                self._recent_keys.popleft()
        return data

    def __setitem__(self, key, data):
        self._data[key] = data
        if self.max_size is None:
            return
        if len(self._data) > self.max_size:
            removal_order = sorted(
                self._key_counts, key=operators.itemgetter(1)
            )
            for k, _ in removal_order:
                if k not in self._recent_keys:
                    del self._data[k]
                    break
