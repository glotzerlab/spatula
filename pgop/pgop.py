"""Python interface for the package.

Provides the `PGOP` class which computes the point group symmetry for a
particle's neighborhood.
"""

import freud
import numpy as np

import pgop._pgop

from . import integrate, sph_harm, util, weijerd


class PGOP:
    """Compute the degree of point group symmetry for specified point groups.

    This class detects the point group symmetry of the modified bond order
    diagram of a particles. Rather than treating the neighbor vectors as delta
    functions, this class treats these vectors as the mean of a distribution on
    the surface of the sphere (e.g. von-Mises-Fisher or uniform distributions).
    """

    def __init__(self, dist, max_l, symmetries, bo_kwargs):
        """Create a PGOP object.

        Parameters
        ----------
        dist : str
            The distribution to use. Either "fisher" for the von-Mises-Fisher
            distribution or "uniform" for a uniform distribution.
        max_l : int
            The maximum :math:`l` spherical harmonic to use. Supports l upto 6.
        symmetries : list[str]
            A list of point groups to test each particles' neighborhood. Uses
            Schoenflies notation and is case sensitive.
        bo_kwargs : dict[str, float]
            A dictionary to pass as keyword arguments to the distribution's
            constructor. The "fisher" distribution expects ``kappa``
            (concentration parameter) and "uniform" ``max_theta`` (angle after
            which distribution is zero).
        """
        self._symmetries = symmetries
        self._weijer = weijerd.WeigerD(max_l)
        dist_param = bo_kwargs.popitem()[1]
        p_norm_weights = [
            self._weijer.group_cardinality(s) for s in self._symmetries
        ]
        D_ij = self._precompute_weijer_d()  # noqa :D806
        if dist == "uniform":
            cls_ = pgop._pgop.PGOPUniform
        if dist == "fisher":
            cls_ = pgop._pgop.PGOPFisher
        self._cpp = cls_(max_l, D_ij, 2, p_norm_weights, dist_param)
        self._sph_harm = sph_harm.SphHarm(max_l)
        self._pgop = None

    def compute(self, system, neighbors, m=6):
        """Compute the point group symmetry for a given system and neighbor.

        Parameter
        ---------
        system :
             A ``freud`` system-like object. Common examples include a tuple of
             a `freud.box.Box` and a `numpy.ndarray` of positions and a
             `gsd.hoomd.Snapshot`.
        neighbors :
            A ``freud`` neighbor query object. Defines neighbors for the system.
            Weights provided by a neighbor list are currently unused.
        m : int, optional
            The number of points to use in the longitudinal direction for
            spherical Gauss-Legrende quadrature. More concentrated distributions
            require larger ``m`` to properly evaluate bond order functions. The
            number of points to evaluate scales as :math:`4 m^2`.
        """
        neigh_query, neighbors = self._get_neighbors(system, neighbors)
        dist = self._compute_distances(neigh_query, neighbors)
        quad_positions, quad_weights = self._get_cartesian_quad(m)
        self._pgop = self._cpp.compute(
            dist,
            neighbors.neighbor_counts,
            m,
            self._ylms(m),
            quad_positions,
            quad_weights,
        )

    def _compute_distances(self, neigh_query, neighbors):
        """Given a query and neighbors get wrapped distances to neighbors."""
        pos, box = neigh_query.points, neigh_query.box
        return box.wrap(pos[neighbors[:, 0]] - pos[neighbors[:, 1]])

    def _get_neighbors(self, system, neighbors):
        """Get a NeighborQuery and NeighborList object.

        Returns the query and neighbor list consistent with the system and
        neighbors passed to `~.compute`.
        """
        query = freud.locality.AABBQuery.from_system(system)
        if isinstance(neighbors, freud.locality.NeighborList):
            return query, neighbors
        return query, query.query(query.points, neighbors).toNeighborList()

    def _ylms(self, m):
        """Return the spherical harmonics at the Gauss-Legrende points.

        Returns all spherical harmonics upto ``self._max_l`` at the points of
        the Gauss-Legrende quadrature of the given ``m``.
        """
        return self._sph_harm(*integrate.gauss_legendre_quad_points(m))

    @property
    def pgop(self):
        """:math:`(N_p, N_{sym})` numpy.ndarray of float: The order parameter.

        The symmetry order is consistent with the order passed to `~.compute`.
        """
        return self._pgop

    def _precompute_weijer_d(self):
        """Return a NumPy array of WignerD matrices for given symmetries."""
        matrices = []
        for point_group in self._symmetries:
            matrices.append(self._weijer[point_group])
        return np.stack(matrices, axis=0)

    @staticmethod
    def _get_cartesian_quad(m):
        """Get the Cartesian coordinates for the Gauss-Legrende quadrature."""
        (quad_theta, quad_phi), wij = integrate.gauss_legendre_quad_points(
            m, True
        )
        return util.sph_to_cart(quad_theta, quad_phi), wij
