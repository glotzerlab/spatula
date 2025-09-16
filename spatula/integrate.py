# Copyright (c) 2021-2025 The Regents of the University of Michigan
# Part of spatula, released under the BSD 3-Clause License.

"""Methods to help perform a Gauss Legendre quatrature."""

import functools

import numpy as np
import scipy as sp
import scipy.special

from . import util


@functools.lru_cache
def gauss_legendre_quad_points(m, weights=False, cartesian=False):
    """Return the given weights for a quadrature of degree ``m``.

    Parameters
    ----------
    m : int
        The degree of the Gauss Legendre quadrature.
    weights : bool, optional
        Whether to return weights with the points or not. Defaults to ``False``.
    cartesian : bool, optional
        Whether to return points in Cartesian coordinates. Defaults to ``False``
        which returns points in spherical coordinates.

    Returns
    -------
    thetas : :math:`(N,)` numpy.ndarray of float
        The polar angles of the points of the quadrature.
    phi : :math:`(N,)` numpy.ndarray of float
        The azimuthal angles of the points of the quadrature.
    x : :math:`(N, 3)` numpy.ndarray of float
        If ``cartesian is True`` then the coordinates are returned in a single
        array.
    weights : :math:`(N,)` numpy.ndarray of float
        If the argument ``weights is True``, then the weights for the quadrature
        are returned.

    """
    coordinates, legrende_weights = _gauss_legendre_quad_points(m)
    if cartesian:
        coordinates = util.sph_to_cart(*coordinates)
    if weights:
        return coordinates, legrende_weights
    return coordinates


def _gauss_legendre_quad_points(m):
    """Return quadrature points and weights."""
    legrende_nodes, legrende_weights = sp.special.roots_legendre(m)
    thetas = np.arccos(legrende_nodes)
    phis = np.linspace(0, 2 * np.pi, num=2 * m, endpoint=False)
    i, j = np.mgrid[0:m, 0 : 2 * m]
    i, j = i.ravel(), j.ravel()
    return (thetas[i], phis[j]), legrende_weights[i]


def gauss_legendre_quad(func, m):
    """Perform a Gauss Legendre quadrature on the unit sphere.

    Parameters
    ----------
    func : callable[[numpy.ndarray, numpy.ndarray], float]
        The function to integrate over the sphere. Expects to be provided theta
        and phi.
    m : int
        The degree of the Gauss Legendre quadrature.

    Returns
    -------
    float
        The integral of ``func`` over the sphere.

    """
    (theta, phi), weights = gauss_legendre_quad_points(m, True)
    return np.pi / m * np.sum(weights * func(theta.ravel(), phi.ravel()))
