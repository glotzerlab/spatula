import numpy as np
import rowan
import scipy as sp
import scipy.special
import scipy.stats

from . import _pgop


def partial_surface_area(R, delta_theta):
    return 2 * np.pi * R * R * (1 - np.cos(delta_theta))


def central_angle(theta_a, phi_a, theta_b, phi_b):
    """Compute the central angles between sets of points.

    Assumes the first column is the polar (latitude) and the second is the
    azmuthal angle. The polar angle is assumed to go from 0 to pi.
    """
    ta = theta_a - (np.pi / 2)
    tb = theta_b - (np.pi / 2)
    return _pgop.central_angle(ta, phi_a, tb, phi_b)


def _central_angle_fast(sin_ta, sin_tb, cos_ta, cos_tb, phi_a, phi_b):
    return _pgop.fast_central_angle(sin_ta, cos_ta, phi_a, sin_tb, cos_tb, phi_b)


def sph_to_cart(theta, phi):
    x = np.empty(theta.shape + (3,))
    sin_theta = np.sin(theta)
    x[..., 0] = sin_theta * np.cos(phi)
    x[..., 1] = sin_theta * np.sin(phi)
    x[..., 2] = np.cos(theta)
    return x


def project_to_unit_sphere(x):
    return (
        np.arccos(x[:, 2] / np.sqrt((x * x).sum(axis=1))),
        np.arctan2(x[:, 1], x[:, 0]),
    )


def delta(x, y):
    return 1 if x == y else 0
