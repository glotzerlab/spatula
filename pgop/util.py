import numpy as np
import rowan
import scipy as sp
import scipy.special
import scipy.stats

from . import _pgop

PI_2 = np.pi / 2
PI_4 = np.pi / 4


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


def normalize(a, out=None):
    norms = np.linalg.norm(a, axis=-1)
    if out is None:
        return a / norms
    np.divide(a, norms[..., None], out=out)
    return None
