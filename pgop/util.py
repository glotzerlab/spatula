"""General utility functions for the package and users."""

import numpy as np

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
