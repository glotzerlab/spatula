import functools

import numpy as np
import scipy as sp
import scipy.special


@functools.lru_cache
def gauss_legendre_quad_points(m, weights=False):

    legrende_nodes, legrende_weights = sp.special.roots_legendre(m)
    thetas = np.arccos(legrende_nodes)
    phis = np.linspace(0, 2 * np.pi, num=2 * m, endpoint=False)
    i, j = np.mgrid[0:m, 0 : 2 * m]
    i, j = i.ravel(), j.ravel()
    if weights:
        return (thetas[i], phis[j]), legrende_weights[i]
    return thetas[i], phis[j]


def gauss_legendre_quad(func, m):
    (theta, phi), weights = gauss_legendre_quad_points(m, True)
    return np.pi / m * np.sum(weights * func(i_theta.ravel(), i_phi.ravel()))
