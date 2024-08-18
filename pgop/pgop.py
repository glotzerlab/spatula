"""Python interface for the package.

Provides the `PGOP` class which computes the point group symmetry for a
particle's neighborhood.
"""

import freud
import numpy as np

import pgop._pgop

from . import integrate, sph_harm, util, wignerd


class PGOP:
    """Compute the degree of point group symmetry for specified point groups.

    This class detects the point group symmetry of the modified bond order
    diagram of a particles. Rather than treating the neighbor vectors as delta
    functions, this class treats these vectors as the mean of a distribution on
    the surface of the sphere (e.g. von-Mises-Fisher or uniform distributions).
    """

    def __init__(
        self,
        dist: str,
        symmetries: list[str],
        optimizer: pgop.optimize.Optimizer,
        max_l: int = 10,
        kappa: float = 11.5,
        max_theta: float = 0.61,
    ):
        """Create a PGOP object.

        Note
        ----
            A ``max_l`` of at least 9 is needed to capture several higher order groups
            such as Cnh, Cnv and some D groups.

        Parameters
        ----------
        dist : str
            The distribution to use. Either "fisher" for the von-Mises-Fisher
            distribution or "uniform" for a uniform distribution.
        symmetries : list[str]
            A list of point groups to test each particles' neighborhood. Uses
            Schoenflies notation and is case sensitive.
        optimizer : pgop.optimize.Optimizer
            An optimizer to optimize the rotation of the particle's local
            neighborhoods.
        max_l : `int`, optional
            The maximum spherical harmonic l to use for computations. This number should
            be larger than the ``l`` and ``refine_l`` used in ``compute``. Defaults to
            10. 
        kappa : float
            The concentration parameter for the von-Mises-Fisher distribution.
            Only used when ``dist`` is "fisher". This number should be roughly equal to
            average number of neighbors. If neighborhood is more dense (has more
            neighbors) higher values are recommended. Should be larger than ``l`` for
            good accuracy. Defaults to 11.5.
        max_theta : float
            The maximum angle (in radians) that the uniform distribution
            extends. Only used when ``dist`` is uniform. Defauts to 0.61
            (roughly 35 degrees).
        """
        if isinstance(symmetries, str):
            raise ValueError("symmetries must be an iterable of str instances.")
        self._symmetries = symmetries
        # computing the PGOP
        self._optmizer = optimizer
        self._max_l = max_l
        if dist == "fisher":
            dist_param = kappa
        elif dist == "uniform":
            dist_param = max_theta
        try:
            cls_ = getattr(pgop._pgop, "PGOP" + dist.title())
        except AttributeError as err:
            raise ValueError(f"Distribution {dist} not supported.") from err
        matrices = []
        for point_group in self._symmetries:
            matrices.append(
                wignerd.WignerD(point_group, self._max_l).condensed_matrices
            )
        D_ij = np.stack(matrices, axis=0)  # noqa N806
        self._cpp = cls_(D_ij, optimizer._cpp, dist_param)
        self._pgop = None
        self._ylm_cache = util._Cache(5)

    def compute(
        self,
        system: tuple[freud.box.Box, np.ndarray],
        neighbors: freud.locality.NeighborList | freud.locality.NeighborQuery,
        query_points: np.ndarray = None,
        l: int = 10,
        m: int = 10,
        refine: bool = False,
        refine_l: int = 20,
        refine_m: int = 20,
    ):
        """Compute the point group symmetry for a given system and neighbor.

        Note
        ----
            Higher ``max_l`` requires higher ``m``. A rough equality is usually
            good enough to ensure accurate results for the given fidelity,
            though setting ``m`` to 1 to 2 higher often still improves results.

        Parameters
        ----------
        system: tuple[freud.box.Box, np.ndarray]
             A ``freud`` system-like object. Common examples include a tuple of
             a `freud.box.Box` and a `numpy.ndarray` of positions and a
             `gsd.hoomd.Frame`.
        neighbors: freud.locality.NeighborList | freud.locality.NeighborQuery
            A ``freud`` neighbor query object. Defines neighbors for the system.
            Weights provided by a neighbor list are currently unused.
        query_points : `numpy.ndarray`, optional
            The points to compute the PGOP for. Defaults to ``None`` which
            computes the PGOP for all points in the system. The shape should be
            ``(N_p, 3)`` where ``N_p`` is the number of points.
        l: `int`, optional
            The spherical harmonic l to use for the bond order functions calculation.
            Increasing ``l`` increases the accuracy of the bond order calculation at the
            cost of performance. The sweet spot number which is high enough for all
            point groups and gives reasonable accuracy for relatively high number of
            neighbors is 10. Point group O needs ``l`` of at least 9 and T needs at
            least 8. Lower values increase speed. Defaults to 10.
        m: `int`, optional
            The number of points to use in the longitudinal direction for
            spherical Gauss-Legrende quadrature. Defaults to 10. We recommend ``m`` to
            be equal or larger than l. More concentrated distributions require larger
            ``m`` to properly evaluate bond order functions. The number of points to
            evaluate scales as :math:`4 m^2`.
        refine: `bool`, optional
            Whether to recompute the PGOP after optimizing. Defaults to
            ``False``. This is used to enable a higher fidelity calculation
            after a lower fidelity optimization. If used the ``refine_l`` and
            ``refine_m`` should be set to a higher value than ``l`` and ``m``. Make sure
            ``max_l`` is higher or equal to ``refine_l``.
        refine_l : `int`, optional
            The maximum spherical harmonic l to use for refining. Defaults
            to 10.
        refine_m : `int`, optional
            The number of points to use in the longitudinal direction for
            spherical Gauss-Legrende quadrature in refining. Defaults to 10. More
            concentrated distributions require larger ``m`` to properly evaluate
            bond order functions. The number of points to evaluate scales as
            :math:`4 m^2`.
        """
        if l > self._max_l:
            raise ValueError("l must be less than or equal to max_l.")
        if refine:
            if refine_l > self._max_l:
                raise ValueError("refine_l must be less than or equal to max_l.")
            if refine_l <l or refine_m < m or (refine_l == l and refine_m == m):
                raise ValueError("refine_l and refine_m must be larger than l and m.")
        neigh_query, neighbors = self._get_neighbors(system, neighbors)
        dist = self._compute_distance_vectors(neigh_query, neighbors, query_points)
        quad_positions, quad_weights = integrate.gauss_legendre_quad_points(
            m=m, weights=True, cartesian=True
        )
        self._pgop, self._rotations = self._cpp.compute(
            dist,
            neighbors.weights,
            neighbors.neighbor_counts,
            m,
            np.conj(self._ylms(l, m)),
            quad_positions,
            quad_weights,
        )
        if refine:
            quad_positions, quad_weights = integrate.gauss_legendre_quad_points(
                m=refine_m, weights=True, cartesian=True
            )
            self._pgop = self._cpp.refine(
                dist,
                self._rotations,
                neighbors.weights,
                neighbors.neighbor_counts,
                refine_m,
                np.conj(self._ylms(refine_l, refine_m)),
                quad_positions,
                quad_weights,
            )

    def _compute_distance_vectors(self, neigh_query, neighbors, query_points):
        """Given a query and neighbors get wrapped distances to neighbors.

        .. todo::
            This should be unnecessary come freud 3.0 as distance vectors should
            be directly available.
        """
        pos, box = neigh_query.points, neigh_query.box
        if query_points is None:
            query_points = pos
        return box.wrap(
            query_points[neighbors.query_point_indices] - pos[neighbors.point_indices]
        )

    def _get_neighbors(self, system, neighbors):
        """Get a NeighborQuery and NeighborList object.

        Returns the query and neighbor list consistent with the system and
        neighbors passed to `PGOP.compute`.
        """
        query = freud.locality.AABBQuery.from_system(system)
        if isinstance(neighbors, freud.locality.NeighborList):
            return query, neighbors
        return query, query.query(query.points, neighbors).toNeighborList()

    def _ylms(self, l, m):
        """Return the spherical harmonics at the Gauss-Legrende points.

        Returns all spherical harmonics upto ``self._max_l`` at the points of
        the Gauss-Legrende quadrature of the given ``m``.
        """
        key = (l, m)
        if key not in self._ylm_cache:
            self._ylm_cache[key] = sph_harm.SphHarm(l)(
                *integrate.gauss_legendre_quad_points(m)
            )
        return self._ylm_cache[key]

    @property
    def pgop(self) -> np.ndarray:
        """:math:`(N_p, N_{sym})` numpy.ndarray of float: The order parameter.

        The symmetry order is consistent with the order passed to
        `PGOP.compute`.
        """
        return self._pgop

    @property
    def rotations(self) -> np.ndarray:
        """:math:`(N_p, N_{sym}, 4)` numpy.ndarray of float: Optimial rotations.

        The optimial rotations expressed as quaternions for each particles and
        each point group.
        """
        return self._rotations

    @property
    def max_l(self) -> int:
        """The maximum spherical harmonic l used in computations."""
        return self._max_l

    @property
    def symmetries(self) -> list[str]:
        """The point group symmetries tested."""
        return self._symmetries
