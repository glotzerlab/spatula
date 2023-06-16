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

    def __init__(self, dist, symmetries, optimizer, kappa=11.5, max_theta=0.61):
        """Create a PGOP object.

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
        kappa : double
            The concentration parameter for the von-Mises-Fisher distribution.
            Only used when ``dist`` is "fisher". Defaults to 11.5.
        max_theta : double
            The maximum angle (in radians) that the uniform distribution
            extends. Only used when ``dist`` is uniform. Defauts to 0.61
            (roughly 35 degrees).
        """
        if isinstance(symmetries, str):
            raise ValueError("symmetries must be an iterable of str instances.")
        self._symmetries = symmetries
        # Always use maximum l and let compute decide the ls to use for
        # computing the PGOP
        self._weijer = weijerd.WeigerD(12)
        self._optmizer = optimizer
        D_ij = self._precompute_weijer_d()  # noqa :D806
        if dist == "fisher":
            dist_param = kappa
        elif dist == "uniform":
            dist_param = max_theta
        try:
            cls_ = getattr(pgop._pgop, "PGOP" + dist.title())
        except AttributeError as err:
            raise ValueError(f"Distribution {dist} not supported.") from err
        self._cpp = cls_(D_ij, optimizer._cpp, dist_param)
        self._pgop = None
        self._ylm_cache = util._Cache(100)

    def compute(
        self,
        system,
        neighbors,
        query_points=None,
        max_l=6,
        m=5,
        refine=True,
        refine_l=9,
        refine_m=10,
    ):
        """Compute the point group symmetry for a given system and neighbor.

        Note:
        ----
            A ``max_l`` of at least 6 is needed to caputure icosahedral ordering
            and a max of 4 is needed for octahedral.

        Note:
        ----
            Higher ``max_l`` requires higher ``m``. A rough equality is usually
            good enough to ensure accurate results for the given fidelity.

        Parameter
        ---------
        system :
             A ``freud`` system-like object. Common examples include a tuple of
             a `freud.box.Box` and a `numpy.ndarray` of positions and a
             `gsd.hoomd.Snapshot`.
        neighbors :
            A ``freud`` neighbor query object. Defines neighbors for the system.
            Weights provided by a neighbor list are currently unused.
        max_l : int, optional
            The maximum spherical harmonic l to use for computations. Defaults
            to 6. Can go up to 12.
        m : int, optional
            The number of points to use in the longitudinal direction for
            spherical Gauss-Legrende quadrature. Defaults to 5. More
            concentrated distributions require larger ``m`` to properly evaluate
            bond order functions. The number of points to evaluate scales as
            :math:`4 m^2`.
        refine: bool, optional
            Whether to recompute the PGOP after optimizing. Defaults to
            ``True``. This is used to enable a higher fidelity calculation
            after a lower fidelity optimization.
        refine_l : int, optional
            The maximum spherical harmonic l to use for refining. Defaults
            to 9. Can go up to 12.
        refine_m : int, optional
            The number of points to use in the longitudinal direction for
            spherical Gauss-Legrende quadrature in refining. Defaults to 9. More
            concentrated distributions require larger ``m`` to properly evaluate
            bond order functions. The number of points to evaluate scales as
            :math:`4 m^2`.
        """
        neigh_query, neighbors = self._get_neighbors(system, neighbors)
        dist = self._compute_distances(neigh_query, neighbors, query_points)
        quad_positions, quad_weights = self._get_cartesian_quad(m)
        self._pgop, self._rotations = self._cpp.compute(
            dist,
            neighbors.weights,
            neighbors.neighbor_counts,
            m,
            np.conj(self._ylms(max_l, m)),
            quad_positions,
            quad_weights,
        )
        if refine:
            quad_positions, quad_weights = self._get_cartesian_quad(refine_m)
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

    def _compute_distances(self, neigh_query, neighbors, query_points):
        """Given a query and neighbors get wrapped distances to neighbors."""
        pos, box = neigh_query.points, neigh_query.box
        if query_points is None:
            query_points = pos
        return box.wrap(
            query_points[neighbors.query_point_indices]
            - pos[neighbors.point_indices]
        )

    def _get_neighbors(self, system, neighbors):
        """Get a NeighborQuery and NeighborList object.

        Returns the query and neighbor list consistent with the system and
        neighbors passed to `~.compute`.
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
    def pgop(self):
        """:math:`(N_p, N_{sym})` numpy.ndarray of float: The order parameter.

        The symmetry order is consistent with the order passed to `~.compute`.
        """
        return self._pgop

    @property
    def rotations(self):
        """:math:`(N_p, N_{sym}, 4)` numpy.ndarray of float: Optimial rotations.

        The optimial rotations expressed as quaternions for each particles and
        each point group.
        """
        return self._rotations

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
