import itertools

import freud
import numpy as np
from tqdm import tqdm

import pgop._pgop

from . import integrate, sph_harm, util, weijerd


class PGOP:
    def __init__(self, dist, max_l, symmetries, bo_kwargs):
        self._symmetries = symmetries
        self._weijer = weijerd.WeigerD(max_l)
        dist_param = bo_kwargs.popitem()[1]
        D_ij = self._precompute_weijer_d()
        if dist == "uniform":
            self._cpp = pgop._pgop.PGOPUniform(max_l, D_ij, 3, dist_param)
        if dist == "fisher":
            self._cpp = pgop._pgop.PGOPFisher(max_l, D_ij, 3, dist_param)
        self._sph_harm = sph_harm.SphHarm(max_l)
        self._pgop = None

    def compute(self, system, neighbors, m=6):
        neigh_query, neighbors = self._get_neighbors(system, neighbors)
        dist = self._compute_distances(neigh_query, neighbors)
        util.normalize(dist, out=dist)
        quad_positions, quad_weights = self._get_quad(m)
        self._pgop = self._cpp.compute(
            dist,
            neighbors.neighbor_counts,
            m,
            self._ylms(m),
            quad_positions,
            quad_weights,
        )

    def _compute_distances(self, neigh_query, neighbors):
        pos, box = neigh_query.points, neigh_query.box
        return box.wrap(pos[neighbors[:, 0]] - pos[neighbors[:, 1]])

    def _get_neighbors(self, system, neighbors):
        query = freud.locality.AABBQuery.from_system(system)
        if isinstance(neighbors, freud.locality.NeighborList):
            return query, neighbors
        return query, query.query(query.points, neighbors).toNeighborList()

    def _ylms(self, m):
        return self._sph_harm(*integrate.gauss_legendre_quad_points(m))

    @property
    def pgop(self):
        return self._pgop

    def _precompute_weijer_d(self):
        matrices = []
        for point_group in self._symmetries:
            matrices.append(self._weijer[point_group])
        return np.stack(matrices, axis=0)

    @staticmethod
    def _get_quad(m):
        (quad_theta, quad_phi), wij = integrate.gauss_legendre_quad_points(
            m, True
        )
        return util.sph_to_cart(quad_theta, quad_phi), wij
