import enum
import functools
import itertools

import freud
import numpy as np
import rowan

from . import bond_order, integrate, optimize, sph_harm, util, weijerd


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
        self._score_weights = np.array(
            [self._weijer.group_cardinality(pg) for pg in self._symmetries])

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

    def _get_bond_order(self, theta, phi):
        if self._dist == "uniform":
            return bond_order.BondOrderUniform(theta, phi, **self._bo_kwargs)
        if self._dist == "fisher":
            return bond_order.BondOrderFisher(theta, phi, **self._bo_kwargs)
        else:
            raise ValueError("Distribution must be uniform or fisher.")

    def _precompute_weijer_d(self):
        matrices = []
        for point_group in self._symmetries:
            matrices.append(self._weijer[point_group])
        return np.stack(matrices, axis=0)

    def _compute_pgop(self, dist, rotation, qlm_eval):
        rotated_dist = util.rotate(dist, *rotation)
        theta, phi = util.project_to_unit_sphere(rotated_dist)
        bod = self._get_bond_order(theta, phi)
        qlms = qlm_eval.eval(bod)
        sym_qlms = weijerd.particle_symmetrize_qlm(qlms, self._Dij, self._weijer)
        return self._covar(qlms[1:], sym_qlms[..., 1:])

    def _score(self, pgop):
        return -_weighted_minkowski(
            pgop, p=3, weights=self._score_weights)


def _weighted_minkowski(a, p=2, weights=None):
    if weights is None:
        weights = np.ones(len(a))
    pow = np.power(a, p)
    return np.dot(pow, weights) / np.sum(weights)
