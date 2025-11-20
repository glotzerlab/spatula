# Copyright (c) 2021-2025 The Regents of the University of Michigan
# Part of spatula, released under the BSD 3-Clause License.

"""Python interface for the package.

Provides the `PGOP` and `BOOSOP` class which computes the point group symmetry for a
particle's neighborhood or its local bond orientation order diagram.
"""

import warnings

import numpy as np

import spatula._spatula

from . import freud, integrate, representations, sph_harm, util


def _get_neighbors(
    system: tuple[freud.box.Box, np.ndarray],
    neighbors: freud.locality.NeighborList | freud.locality.NeighborQuery,
    query_points: np.ndarray | None,
) -> tuple[np.ndarray, freud.locality.NeighborList]:
    """Get a NeighborQuery and NeighborList object.

    Returns the query and neighbor list consistent with the system and
    neighbors passed.
    """
    query = freud.locality.AABBQuery.from_system(system)
    if isinstance(neighbors, freud.locality.NeighborList):
        if query_points is not None:
            warnings.warn(
                "query_points are ignored when a NeighborList is passed.",
                UserWarning,
                stacklevel=2,
            )
    else:
        query_points = query_points if query_points is not None else query.points
        neighbors = query.query(query_points, neighbors).toNeighborList()
    neighbors.filter(neighbors.distances > 0)
    return query.box.wrap(neighbors.vectors), neighbors


class BOOSOP:
    """Compute the point group symmetry order for bond orientational order diagram.

    This class implements the method described in :cite:`butler2024development`. It
    detects the point group symmetry of the modified bond order diagram of a particles.
    Rather than treating the neighbor vectors as delta functions, this class treats
    these vectors as the mean of a distribution on the surface of the sphere (e.g.
    von-Mises-Fisher or uniform distributions).
    """

    def __init__(
        self,
        dist: str,
        symmetries: list[str],
        optimizer: spatula.optimize.Optimizer,
        max_l: int = 10,
        kappa: float = 10,
        max_theta: float = 0.61,
    ):
        """Create a BOOSOP object.

        This class implements the method described in :cite:`butler2024development`. All
        point groups of finite order are supported.

        Note
        ----
            A ``max_l`` of at least 9 is needed to capture several higher order groups
            such as :math:`C_{nh}`, :math:`C_{nv}` and some :math:`D` groups.

        Parameters
        ----------
        dist : str
            The distribution to use. Either "fisher" for the von-Mises-Fisher
            distribution or "uniform" for a uniform distribution.
        symmetries : list[str]
            A list of point groups to test each particles' neighborhood. Uses
            Schoenflies notation and is case sensitive. Options are
            :math:`C_i`, :math:`C_s`, :math:`C_n`, :math:`C_{nh}`, :math:`C_{nv}`,
            :math:`S_n`, :math:`D_n`, :math:`D_{nh}`, :math:`D_{nd}`, :math:`T`,
            :math:`T_h`, :math:`T_d`, :math:`O`, :math:`O_h`, :math:`I`, :math:`I_h`.
            Replace :math:`n` with an integer, and pass them as strings, e.g.,
            ``["C3", "D6h"]``.
        optimizer : spatula.optimize.Optimizer
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
            extends. Only used when ``dist`` is uniform. Defaults to 0.61
            (roughly 35 degrees).

        """
        if isinstance(symmetries, str):
            raise ValueError("symmetries must be an iterable of str instances.")
        self._symmetries = symmetries
        # computing the BOOSOP
        self._optmizer = optimizer
        self._max_l = max_l
        if dist == "fisher":
            dist_param = kappa
        elif dist == "uniform":
            dist_param = max_theta
        try:
            cls_ = getattr(spatula._spatula, "BOOSOP" + dist.title())
        except AttributeError as err:
            raise ValueError(f"Distribution {dist} not supported.") from err
        matrices = []
        for point_group in self._symmetries:
            matrices.append(
                representations.WignerD(point_group, self._max_l).condensed_matrices
            )
        D_ij = np.stack(matrices, axis=0)  # noqa N806
        self._cpp = cls_(D_ij, optimizer._cpp, dist_param)
        self._order = None
        self._rotations = None
        self._ylm_cache = util._Cache(5)

    def compute(
        self,
        system: tuple[freud.box.Box, np.ndarray],
        neighbors: freud.locality.NeighborList | freud.locality.NeighborQuery,
        query_points: np.ndarray | None = None,
        l: int = 10,
        m: int = 10,
        refine: bool = False,
        refine_l: int = 20,
        refine_m: int = 20,
    ):
        r"""Compute the point group symmetry for a given system and neighbor.

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
        query_points: np.ndarray | None, optional
            The points to compute the BOOSOP for. Defaults to ``None`` which
            computes the BOOSOP for all points in the system. The shape should be
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
            Whether to recompute the BOOSOP after optimizing. Defaults to
            ``False``. This is used to enable a higher fidelity calculation
            after a lower fidelity optimization. If used the ``refine_l`` and
            ``refine_m`` should be set to a higher value than ``l`` and ``m``. Make sure
            ``max_l`` is higher or equal to ``refine_l``.
        refine_l: `int`, optional
            The maximum spherical harmonic l to use for refining. Defaults
            to 10.
        refine_m: `int`, optional
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
            if refine_l < l or refine_m < m or (refine_l == l and refine_m == m):
                raise ValueError("refine_l and refine_m must be larger than l and m.")
        dist, neighbors = _get_neighbors(system, neighbors, query_points)
        quad_positions, quad_weights = integrate.gauss_legendre_quad_points(
            m=m, weights=True, cartesian=True
        )
        self._order, self._rotations = self._cpp.compute(
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
            self._order = self._cpp.refine(
                dist,
                self._rotations,
                neighbors.weights,
                neighbors.neighbor_counts,
                refine_m,
                np.conj(self._ylms(refine_l, refine_m)),
                quad_positions,
                quad_weights,
            )

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
    def order(self) -> np.ndarray:
        """:math:`(N_p, N_{sym})` numpy.ndarray of float: The order parameter is [0,1].

        The symmetry order is consistent with the order passed to
        `BOOSOP.compute`.
        """
        if self._order is None:
            raise ValueError("BOOSOP not computed, call compute first.")
        return self._order

    @property
    def rotations(self) -> np.ndarray:
        """:math:`(N_p, N_{sym}, 4)` numpy.ndarray of float: Optimal rotations.

        The optimal rotations of local neighborhoods that maximize the value of PGOP for
        each query particle and each point group. Rotations are expressed as
        quaternions. Note that these use different convention to scipy! The convention
        used here is [w,x,y,z]. The scipy convention is [x,y,z,w].
        """
        if self._rotations is None:
            raise ValueError("BOOSOP not computed, call compute first.")
        return self._rotations

    @property
    def max_l(self) -> int:
        """The maximum spherical harmonic l used in computations."""
        return self._max_l

    @property
    def symmetries(self) -> list[str]:
        """The point group symmetries tested."""
        return self._symmetries


class PGOP:
    r"""Compute the point group symmetry order for a given point cloud.

    This class implements the algorithm highlighted in :cite:`fijan2025quantifying`. It
    detects the point group symmetry of a point in space based on the surrounding
    points. Rather than treating the neighbor vectors as delta functions, this class
    treats these vectors as a Gaussian distribution. This enables continuous evaluation
    of the point group symmetry.
    """

    def __init__(
        self,
        symmetries: list[str],
        optimizer: spatula.optimize.Optimizer,
        mode: str = "full",
        compute_per_operator_values_for_final_orientation: bool = False,
    ):
        r"""Create a PGOP object.

        This class implements the algorithm highlighted in :cite:`fijan2025quantifying`.
        All point groups of finite order are supported.


        Parameters
        ----------
        symmetries : list[str]
            A list of point groups to test each particles' neighborhood. Uses
            Schoenflies notation and is case sensitive. Options are :math:`C_i`,
            :math:`C_s`, :math:`C_n`, :math:`C_{nh}`, :math:`C_{nv}`, :math:`S_n`,
            :math:`D_n`, :math:`D_{nh}`, :math:`D_{nd}`, :math:`T`, :math:`T_h`,
            :math:`T_d`, :math:`O`, :math:`O_h`, :math:`I`, :math:`I_h`, where :math:`n`
            where n should be replaced with group order (an integer) and passed as a
            list of strings.
        optimizer : spatula.optimize.Optimizer
            An optimizer to optimize the rotation of the particle's local
            neighborhoods.
        mode : str, optional
            The mode to use for the computation. Either "full" or "boo". Defaults to
            "full". "full" computes the full point group symmetry order parameter (PGOP)
            while "boo" computes the point group symmetry order of the bond
            orientational order diagram (PGOP-BOOD).
        compute_per_operator_values_for_final_orientation : bool, optional
            Whether to compute the PGOP values for all subgroups of point group
            symmetries of interest, at same orientation as the point group of interest
            PGOP value. Defaults to False. `order` values are in order point group
            symmetry, order for symmetry operators of this point group in order given by
            the representations.matrices, order for second point group symmetry, etc.

        """
        if isinstance(symmetries, str):
            raise ValueError("symmetries must be an iterable of str instances.")
        self._symmetries = symmetries
        # computing the PGOP
        self._optmizer = optimizer
        matrices = []
        for point_group in self._symmetries:
            pg = representations.CartesianRepMatrix(point_group)
            # skips E operator if group is not C1
            if point_group == "C1":
                matrices.append(pg.condensed_matrices)
            else:
                matrices.append(pg.condensed_matrices[9:])
        if mode == "full":
            m_mode = 0
        elif mode == "boo":
            m_mode = 1
        else:
            msg = f"Mode '{mode}' is not valid (valid params: {{'full', 'boo'}})"
            raise ValueError(msg)
        self._mode = mode
        self._cpp = spatula._spatula_nb.PGOP(
            matrices,
            optimizer._cpp,
            m_mode,
            compute_per_operator_values_for_final_orientation,
        )
        self._order = None
        self._rotations = None

    def compute(
        self,
        system: tuple[freud.box.Box, np.ndarray],
        sigmas: np.ndarray | float | None,
        neighbors: freud.locality.NeighborList | freud.locality.NeighborQuery,
        query_points: np.ndarray | None = None,
    ):
        """Compute the point group symmetry for a given system and neighbor.

        Parameters
        ----------
        system: tuple[freud.box.Box, np.ndarray]
            A ``freud`` system-like object. Common examples include a tuple of
            a `freud.box.Box` and a `numpy.ndarray` of positions and a
            `gsd.hoomd.Frame`.
        sigmas: np.ndarray | float
            The standard deviation of the Gaussian distribution for each particle for
            mode "full". If mode is "boo", the kappa parameter for the von-Mises-Fisher.
            Note that for gaussian distribution, smaller sigma values are more
            concentrated and larger sigma values are more spread out. For von-Mises-
            Fisher distribution, smaller kappa values are more spread out and larger
            kappa values are more concentrated.
            If a float is passed, the same value is used for all particles. If `None` is
            passed, sigma is determined as the value at which the gaussian function
            value evaluated at half of the smallest bond distance has 25% height of max
            gaussian height for the same sigma for the "full" mode. In the "boo" mode,
            the default value is 15.0.
        neighbors: freud.locality.NeighborList | freud.locality.NeighborQuery | dict
            Neighbors used for the computation. If a `freud.locality.NeighborList` is
            passed, the neighbors are used directly (in this case ``query_points``
            should not be given as they are ignored). If a
            `freud.locality.NeighborQuery` is passed, the neighbors are computed using
            the query (working in conjunction with query points). If a dictionary is
            used it should be used as freud's neighbor query dictionary (can also be
            used in conjunction with ``query_points``).
        query_points : np.ndarray | None, optional
            The points to compute the PGOP for. Defaults to ``None`` which
            computes the PGOP for all points in the system. The shape should be
            ``(N_p, 3)`` where ``N_p`` is the number of points.

        """
        dist, neighbors = _get_neighbors(system, neighbors, query_points)
        if isinstance(sigmas, (float, int)):
            sigmas = np.full(
                neighbors.num_points * neighbors.num_query_points,
                sigmas,
                dtype=np.float32,
            )
        elif isinstance(sigmas, (np.ndarray, list)):
            if len(sigmas) != neighbors.num_points:
                raise ValueError(
                    "sigmas must be a float, a list of floats or an array of floats "
                    "with the same length as the number of points in the system."
                )
            sigmas = np.array(
                [sigmas[i] for i in neighbors.point_indices], dtype=np.float32
            )
        elif sigmas is None:
            distances = np.linalg.norm(dist, axis=1)
            # filter distances that are smaller then 0.001 of mean distance
            filter = distances > 0.001 * np.mean(distances)
            if self.mode == "full":
                distances_filtered = distances[filter]
                # find gaussian width sigma at which the value of the gaussian function
                # at half of the smallest bond distance has 25% height of max gaussian
                # height for the same sigma
                sigma = np.min(distances_filtered) * 0.5 / (np.sqrt(-2 * np.log(0.25)))
            elif self.mode == "boo":
                bond_vectors = dist[filter] / np.linalg.norm(
                    dist[filter], axis=1, keepdims=True
                )
                # Get segments to split bond vectors into neighborhoods
                min_angular_dists = []
                # Iterate over neighborhoods
                for iseg in range(len(neighbors.segments)):
                    cseg = neighbors.segments[iseg]
                    nseg = (
                        neighbors.segments[iseg + 1]
                        if iseg + 1 < len(neighbors.segments)
                        else len(neighbors.point_indices)
                    )
                    neighborhood_vectors = bond_vectors[cseg:nseg]
                    # Compute pairwise angular distances within the neighborhood
                    angular_dists = np.arccos(
                        np.clip(
                            np.dot(neighborhood_vectors, neighborhood_vectors.T),
                            -1.0,
                            1.0,
                        )
                    )
                    # Mask diagonal to ignore self-distances
                    np.fill_diagonal(angular_dists, np.inf)
                    # flatten angular dists and remove zeros if present
                    angular_dists = angular_dists.flatten()
                    angular_dists = angular_dists[angular_dists > 0]
                    min_angular_dists.append(np.min(angular_dists))
                # Find the global minimum angular distance across all neighborhoods
                min_angular_dist = np.min(min_angular_dists)
                # this is the minimum resolution that will work for optimization
                if min_angular_dist < 0.33:
                    min_angular_dist = 0.33
                # again 25% height of max fisher distribution height
                sigma = np.log(0.25) / (np.cos(min_angular_dist * 0.5) - 1)
            sigmas = np.full(
                neighbors.num_points * neighbors.num_query_points,
                sigma,
                dtype=np.float32,
            )
        else:
            raise ValueError(
                "sigmas must be a float, a list of floats or an array of floats "
                "with the same length as the number of points in the system."
            )
        self._sigmas = sigmas
        self._order, self._rotations = self._cpp.compute(
            dist, neighbors.weights, neighbors.neighbor_counts, sigmas
        )

    @property
    def order(self) -> np.ndarray:
        """:math:`(N_p, N_{sym})` numpy.ndarray of float: The order parameter is [0,1].

        The symmetry order is consistent with the order passed to
        `PGOP.compute`.
        """
        if self._order is None:
            raise ValueError("PGOP not computed, call compute first.")
        return self._order

    @property
    def rotations(self) -> np.ndarray:
        """:math:`(N_p, N_{sym}, 4)` numpy.ndarray of float: Optimal rotations.

        The optimal rotations of local neighborhoods that maximize the value of PGOP for
        each query particle and each point group. Rotations are expressed as
        quaternions. Note that these use different convention to scipy! The convention
        used here is [w,x,y,z]. The scipy convention is [x,y,z,w].
        """
        if self._rotations is None:
            raise ValueError("PGOP not computed, call compute first.")
        return self._rotations

    @property
    def sigmas(self) -> np.ndarray:
        """The standard deviation of the Gaussian distribution for each particle."""
        if self._sigmas is None:
            raise ValueError("PGOP not computed, call compute first.")
        return self._sigmas

    @property
    def symmetries(self) -> list[str]:
        """The point group symmetries tested."""
        return self._symmetries

    @property
    def mode(self) -> str:
        """The mode used for the computation."""
        return self._mode
