import itertools

import freud
import numpy as np
from tqdm import tqdm

from . import _pgop, bond_order, integrate, optimize, sph_harm, util, weijerd


class _QlmEval:
    def __init__(self, pgop_compute, m, dist):
        (quad_theta, quad_phi), wij = integrate.gauss_legendre_quad_points(
            m, True
        )
        positions = util.sph_to_cart(quad_theta, quad_phi)
        self._cpp = _pgop.QlmEval(m, positions, wij, pgop_compute._ylms(m))
        if dist == "uniform":
            self.eval = self.uniform_eval
        elif dist == "fisher":
            self.eval = self.fisher_eval

    def uniform_eval(self, bod):
        return self._cpp.uniform_eval(bod._cpp)

    def fisher_eval(self, bod):
        return self._cpp.fisher_eval(bod._cpp)


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
        self._score_cls = _WeightedMinkowski(
            p=3,
            weights=[
                self._weijer.group_cardinality(pg) for pg in self._symmetries
            ],
        )

    def compute(self, system, neighbors, m=6):
        qlm_eval = _QlmEval(self, m, self._dist)
        neigh_query, neighbors = self._get_neighbors(system, neighbors)
        dist = self._compute_distances(neigh_query, neighbors)
        util.normalize(dist, out=dist)
        neigh_i = 0
        pgop = np.empty((neighbors.num_points, len(self._symmetries)))
        iterator = tqdm(
            enumerate(neighbors.neighbor_counts), total=neighbors.num_points
        )
        for particle_i, num_neighbors in iterator:
            ppgop = self._compute_particle(
                dist[neigh_i : neigh_i + num_neighbors], qlm_eval
            )
            pgop[particle_i] = ppgop
            neigh_i += num_neighbors
        self._pgop = pgop

    def _compute_particle(self, dist, qlm_eval):
        pi_2 = np.pi / 2
        pi_4 = np.pi / 4
        rotations = [
            angles
            for angles in itertools.product(
                (-pi_2, pi_2), (-pi_4, pi_4), (-pi_2, pi_2)
            )
        ]
        rotations.append([0, 0, 0])
        bounds = np.array([[-np.pi, np.pi], [0, np.pi], [-np.pi, np.pi]])
        brute_opt = optimize.BruteForce(np.array(rotations), bounds)
        for rotation in brute_opt:
            pgop = self._compute_pgop(dist, rotation, qlm_eval)
            brute_opt.report_objective(self._score(pgop))
        init_simplex = self._get_initial_simplex(brute_opt.optimum[0])
        simplex_opt = optimize.NelderMead(
            init_simplex, bounds, max_iter=150, dist_tol=1e-2, std_tol=1e-3
        )
        for rotation in simplex_opt:
            pgop = self._compute_pgop(dist, rotation, qlm_eval)
            simplex_opt.report_objective(self._score(pgop))
        return self._compute_pgop(dist, simplex_opt.optimum[0], qlm_eval)

    def _compute_distances(self, neigh_query, neighbors):
        pos, box = neigh_query.points, neigh_query.box
        return box.wrap(pos[neighbors[:, 0]] - pos[neighbors[:, 1]])

    def _get_initial_simplex(self, origin):
        """Get a tetrahedron with given origin and 1/6 the volume of SO(3).

        Here SO(3) denotes the space of Euler angles.
        """
        volume_fraction = 1 / 5
        s = np.pi * (2 ** (1.0 / 6.0)) * ((3 * volume_fraction) ** (1.0 / 3.0))
        b = 1 / np.sqrt(2)
        return origin + s * np.array(
            [[1, 0, -b], [-1, 0, -b], [0, 1, b], [0, -1, b]]
        )

    def _get_neighbors(self, system, neighbors):
        query = freud.locality.AABBQuery.from_system(system)
        if isinstance(neighbors, freud.locality.NeighborList):
            return query, neighbors
        return query, query.query(query.points, neighbors).toNeighborList()

    def _covar(self, qlms, sym_qlms):
        return _pgop.covariance_score(qlms, sym_qlms)

    def _ylms(self, m):
        return self._sph_harm(*integrate.gauss_legendre_quad_points(m))

    @property
    def pgop(self):
        return self._pgop

    def _get_bond_order(self, positions):
        if self._dist == "uniform":
            return bond_order.BondOrderUniform(positions, **self._bo_kwargs)
        if self._dist == "fisher":
            return bond_order.BondOrderFisher(positions, **self._bo_kwargs)
        else:
            raise ValueError("Distribution must be uniform or fisher.")

    def _precompute_weijer_d(self):
        matrices = []
        for point_group in self._symmetries:
            matrices.append(self._weijer[point_group])
        return np.stack(matrices, axis=0)

    def _compute_pgop(self, dist, rotation, qlm_eval):
        rotated_dist = util.rotate(dist, *rotation)
        bod = self._get_bond_order(rotated_dist)
        qlms = qlm_eval.eval(bod)
        sym_qlms = weijerd.particle_symmetrize_qlm(
            qlms, self._Dij, self._weijer
        )
        return self._covar(qlms[1:], sym_qlms[..., 1:])

    def _score(self, pgop):
        return -self._score_cls(pgop)


class _WeightedMinkowski:
    def __init__(self, p=2, weights=None):
        if weights is None:
            weights = []
        self._cpp = getattr(_pgop, f"Weighted{p}Norm")(weights)

    def __call__(self, a):
        return self._cpp(a)
