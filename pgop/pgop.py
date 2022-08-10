import enum
import functools
import itertools

import freud
import numpy as np
import rowan

from . import bond_order, integrate, optimize, sph_harm, weijerd, util


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
            if sym == "T":
                yield self._weijer.tetrahedral()
            inverse = "i" in sym
            num_slice = slice(2) if inverse else slice(1)
            if sym[0] == "C":
                yield self._weijer.cyclic(int(sym[num_slice]), inverse)
            if sym[0] == "D":
                yield self._weijer.diherdral(int(sym[num_slice]), inverse)

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


class _QlmEval:
    def __init__(self, pgop_compute, m):
        self.m = m
        (quad_theta, quad_phi), wij = integrate.gauss_legendre_quad_points(
            m, True)
        self.theta = quad_theta
        self.phi = quad_phi
        # Need to adjust for use in util._central_angle_fast
        quad_theta = quad_theta - np.pi / 2
        self.sin_theta = np.sin(quad_theta)
        self.cos_theta = np.cos(quad_theta)

        normalization = 1 / (4 * m)
        self.weighted_ylms = (
            normalization * wij[None, :] * pgop_compute._ylms(m))

    def eval(self, bod):
        evaluated_bod = bod._fast_call(self.sin_theta, self.cos_theta, self.phi)
        return np.einsum("jk,k", self.weighted_ylms, evaluated_bod)


class OptimizedPGOP:
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
        qlm_eval = _QlmEval(self, m)
        neigh_query, neighbors = self._get_neighbors(system, neighbors)
        dist = self._compute_distances(neigh_query, neighbors)
        neigh_i = 0
        pgop = np.empty((neighbors.num_points, len(self._symmetries)))
        for particle_i, num_neighbors in enumerate(neighbors.neighbor_counts):
            pgop[particle_i] = self._compute_particle(
                dist[neigh_i : neigh_i + num_neighbors], qlm_eval)
            neigh_i += num_neighbors
        self._pgop = pgop

    def _compute_particle(self, dist, qlm_eval):
        pi_2 = np.pi / 2
        pi_4 = np.pi / 4
        rotations = np.array(
            [angles for angles in itertools.product(
                (-pi_2, pi_2), (-pi_4, pi_4), (-pi_2, pi_2))]
        )
        bounds = np.array([[-np.pi, np.pi], [0, np.pi], [-np.pi, np.pi]])
        brute_opt = optimize.BruteForce(rotations, bounds)
        for rotation in brute_opt:
            pgop = self._compute_pgop(dist, rotation, qlm_eval)
            brute_opt.report_objective(self._score(pgop))
        init_simplex = optimize.NelderMead.create_initial_simplex(
            brute_opt.optimum[0], delta=np.array([pi_4, np.pi / 8, pi_4]))
        simplex_opt = optimize.NelderMead(
            init_simplex, bounds, max_iter=50, dist_tol=1e-2, std_tol=1e-3)
        for rotation in simplex_opt:
            pgop = self._compute_pgop(dist, rotation, qlm_eval)
            simplex_opt.report_objective(self._score(pgop))
        return self._compute_pgop(dist, simplex_opt.optimum[0], qlm_eval)

    def _compute_distances(self, neigh_query, neighbors):
        pos, box = neigh_query.points, neigh_query.box
        return box.wrap(pos[neighbors[:, 0]] - pos[neighbors[:, 1]])

    def _get_neighbors(self, system, neighbors):
        query = freud.locality.AABBQuery.from_system(system)
        if isinstance(neighbors, freud.locality.NeighborList):
            return query, neighbors
        return query, query.query(query.points, neighbors).toNeighborList()

    def _covar(self, qlms, sym_qlms):
        # Uses the covariance trick of spherical harmonic expansions and
        # symmetrized expansions.
        covar_plain = np.real(np.einsum("j,j", qlms, np.conj(qlms)))
        covar_sym = np.real(np.einsum("jk,jk->j", sym_qlms, np.conj(sym_qlms)))
        covar_mixed = np.real(np.einsum("jk,k->j", sym_qlms, np.conj(qlms)))
        return covar_mixed / np.sqrt(covar_sym * covar_plain)

    def _ylms(self, m):
        return self._sph_harm(*integrate.gauss_legendre_quad_points(m))

    @property
    def pgop(self):
        return self._pgop

    def _weijer_indices(self):
        for sym in self._symmetries:
            if sym == "T":
                yield self._weijer.tetrahedral()
            inverse = "i" in sym
            num_slice = slice(2) if inverse else slice(1)
            if sym[0] == "C":
                yield self._weijer.cyclic(int(sym[num_slice]), inverse)
            if sym[0] == "D":
                yield self._weijer.diherdral(int(sym[num_slice]), inverse)

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

    def _compute_pgop(self, dist, rotation, qlm_eval):
        rotation = rowan.from_euler(*rotation)
        rotated_dist = rowan.rotate(rotation, dist)
        theta, phi = util.project_to_unit_sphere(rotated_dist)
        bod = self._get_bond_order(theta, phi)
        qlms = qlm_eval.eval(bod)
        sym_qlms = weijerd._particle_symmetrize_qlm(
            qlms, self._Dij, self._weijer)
        return self._covar(qlms[1:], sym_qlms[..., 1:])

    def _score(self, pgop):
        return -np.linalg.norm(pgop)
