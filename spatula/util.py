# Copyright (c) 2021-2025 The Regents of the University of Michigan
# Part of spatula, released under the BSD 3-Clause License.

"""General utility functions for the package and users."""

import collections
import operator

import numpy as np

from . import _spatula, freud

PI_2 = np.pi / 2
PI_4 = np.pi / 4


def sph_to_cart(theta, phi):
    r"""Convert spherical to Cartesian coordinates on the unit sphere.

    Parameters
    ----------
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
    _spatula.set_num_threads(num_threads)
    freud.parallel.set_num_threads(num_threads)


def get_num_threads():
    """Get the number of threads used when computing PGOP.

    Returns
    -------
    num_threads : int
        The number of threads.

    """
    return _spatula.get_num_threads()


class _Cache:
    """A simple cache that supports a maximum size.

    Size restraints by removing the least frequently used keys. The cache also
    supports preventing deleting the most recently used keys as well through a
    FIFO stack.
    """

    def __init__(self, max_size=None, keep_n_most_recent=1):
        """Construct a cache object.

        Parameters
        ----------
        max_size : int, optional
            The maximum size for the cache. Defaults to ``None`` which means no
            limit on cache size.
        keep_n_most_recent : int, optional
            The number of recent examples to not allow for removal regardless of
            popularity. Defaults to 1 key.

        """
        self._data = {}
        self._key_counts = collections.Counter()
        self._recent_keys = collections.deque(maxlen=keep_n_most_recent)
        self.max_size = max_size

    def __contains__(self, key):
        """Return whether the given key is in the cache."""
        return key in self._data

    def __getitem__(self, key):
        """Get the cached value for key and error if not present."""
        data = self._data.get(key, None)
        if data is None:
            return data
        if self.max_size is not None:
            self._recent_keys.append(key)
            self._key_counts[key] += 1
        return data

    def __setitem__(self, key, data):
        """Set the cached value for key overwriting if necessary."""
        self._data[key] = data
        if self.max_size is None:
            return
        if len(self._data) > self.max_size:
            removal_order = sorted(self._key_counts, key=operator.itemgetter(1))
            for k, _ in removal_order:
                if k not in self._recent_keys:
                    del self._data[k]
                    break
