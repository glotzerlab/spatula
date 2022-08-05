import functools
import itertools

import freud
import numpy as np

from . import bond_order, integrate, sph_harm, weijerd, util


class PGOP:
    def __init__(self, dist, max_l, symmetries, bo_kwargs):
        self._dist = dist 
        self._bo_kwargs = bo_kwargs
        self._sph_harm = sph_harm.SphHarm(max_l)
        self._max_l = max_l
        self._symmetries = symmetries
        self._pgop = None
        self._weijer = weijerd.WeigerD(max_l)
        self._Dij = self._precompute_weijer_d()

    def compute(self, system, neighbors, m=6):
        neigh_query, neighbors = self._get_neighbors(system, neighbors)
        theta, phi = self._project_to_unit_sphere(neigh_query, neighbors)
        qlms, bo = self._compute_qlms(theta, phi, neighbors, m)
        sym_qlms = weijerd.symmetrize_qlm(qlms, self._Dij, self._weijer)
        self._pgop = self._covar(qlms[:, 1:], sym_qlms[..., 1:])
        self._qlms = qlms
        self._bo = bo

    def _compute_qlms(self, theta, phi, nlist, m):
        theta_i = 0
        (quad_theta, quad_phi), wij = integrate.gauss_legendre_quad_points(
            m, True)
        quad_theta = quad_theta - np.pi / 2
        bo_values = np.empty((nlist.num_points, len(quad_theta)))
        sin_quad_theta, cos_quad_theta = np.sin(quad_theta), np.cos(quad_theta)
        for particle_i, num_neigh  in enumerate(nlist.neighbor_counts):
            j = slice(theta_i, theta_i + num_neigh)
            theta_i += num_neigh
            bo_values[particle_i, :] = self._compute_bond_order(
                theta[j], phi[j], sin_quad_theta, cos_quad_theta, quad_phi
            )
        # normalization = (pi / m) * (1 / 4 pi)
        normalization = 1 / (4 * m)
        qlms = np.einsum("ik,jk,k->ij", bo_values, self._ylms(m), wij)
        return normalization * qlms, bo_values

    def _get_neighbors(self, system, neighbors):
        query = freud.locality.AABBQuery.from_system(system)
        if isinstance(neighbors, freud.locality.NeighborList):
            return query, neighbors
        return query, query.query(query.points, neighbors).toNeighborList()

    def _project_to_unit_sphere(self, neigh_query, neighbors):
        pos = neigh_query.points 
        theta, phi = util.project_to_unit_sphere(
            neigh_query.box.wrap(pos[neighbors[:, 0]] - pos[neighbors[:, 1]]))
        return theta, phi

    def _compute_bond_order(self, theta, phi, sin_theta, cos_theta, compute_phi):
        bo_dist = self._get_bond_order(theta, phi)
        return bo_dist._fast_call(sin_theta, cos_theta, compute_phi)

    def _covar(self, qlms, sym_qlms):
        # Uses the covariance trick of spherical harmonic expansions and
        # symmetrized expansions.
        covar_plain = np.real(np.einsum("ij,ij->i", qlms, np.conj(qlms)))
        covar_sym = np.real(np.einsum("ijk,ijk->ij", sym_qlms, np.conj(sym_qlms)))
        covar_mixed = np.real(
            np.einsum("ijk,ik->ij", sym_qlms, np.conj(qlms)))
        return covar_mixed / np.sqrt(covar_sym * covar_plain[:, np.newaxis])

    @functools.lru_cache
    def _ylms(self, m):
        return self._sph_harm(*integrate.gauss_legendre_quad_points(m))

    @property
    def pgop(self):
        return self._pgop

    def _weijer_indices(self):
        for sym in self._symmetries:
            sym = sym.lower()
            inverse = "i" in sym
            if sym[0] == "c":
                yield self._weijer.cyclic(int(sym[-1]), inverse)
            if sym[0] == "d":
                yield self._weijer.diherdral(int(sym[-1]), inverse)
            if sym[0] == "t":
                yield self._weijer.tetrahedral()


    def _get_bond_order(self, theta, phi):
        if self._dist == "uniform":
            return bond_order.BondOrderUniform(theta, phi, **self._bo_kwargs)
        if self._dist == "fisher":
            return bond_order.BondOrderFisher(theta, phi, **self._bo_kwargs)
        else:
            raise ValueError("Distribution must be uniform or fisher.")

    def _precompute_weijer_d(self):
        matrices = []
        for matrix in self._weijer_indices():
            matrices.append(matrix)
        return np.stack(matrices, axis=0)
