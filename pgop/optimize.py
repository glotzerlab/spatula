"""Optimization schemes for PGOP."""

import abc
import itertools

import numpy as np

import pgop._pgop

HYPERSPHERE_BOUNDS = np.array([[0, np.pi], [0, np.pi], [0, np.pi]])


class Optimizer(abc.ABC):
    """Base class for optimization schemes.

    Defines the interface that optimizers must impliment.

    The interface in Python is an state based iterator. The optimizer is
    iterated over and at each iteration `~.record_objective` must be called with
    the objective of the point provided in the last iteration. Iteration stops
    when the `~.terminate` is ``True``.

    Warning:
        User created `Optimizer` subclasses are not currently supported.
    """

    # User Interface
    @property
    def optimum(self):
        """list[float]: The current best point."""
        return self._cpp.optimum

    @property
    def terminated(self):
        """bool: Whether the optimization should be terminated."""
        return self._cpp.terminate

    def __iter__(self):
        """Iterate over the points to test for optimization.

        This iteration is unique in Python in that it requires
        `~.report_objective` to be called inbetween iterations.

        Yields
        ------
        points : list[float]
            The next point to test.
        """
        while not self.terminated:
            yield self._cpp.next_point()

    def report_objective(self, objective):
        """Record the objective for the last iterated point."""
        self._cpp.record_objective(objective)

    @staticmethod
    def _default_bounds(bounds, dim):
        if bounds is not None:
            return np.asarray(bounds)
        return np.tile([-np.inf, np.inf], (dim, 1))


class BruteForce(Optimizer):
    """Find the optimum among specified points.

    While not an optimizer in the traditional sense, `BruteForce` will determine
    the optimum from the set of points provided in its constructor.
    """

    def __init__(self, points, bounds=None, max_iter=None):
        self._points = np.copy(points)
        if self._points.ndim != 2:
            raise ValueError("points must have shape (npoints, ndim)")
        bounds = Optimizer._default_bounds(bounds, self._points.shape[1])
        self._bounds = bounds
        self._cpp = pgop._pgop.BruteForce(
            self._points.tolist(), bounds[:, 0].tolist(), bounds[:, 1].tolist()
        )

    @classmethod
    def from_mesh(cls, num_splits, bounds=None, use_ends=False):
        """Create a mesh of points for constructing a `BruteForce` object.

        Parameters
        ----------
        num_split : int or tuple[int]
        bounds : array_like of shape :math:`(N_{dim}, 2)` of float, optional
        use_ends : bool or list[bool]

        Returns
        -------
        BruteForce
        """
        if bounds is None:
            bounds = HYPERSPHERE_BOUNDS
        if use_ends is None:
            use_ends = True
        if isinstance(use_ends, bool):
            use_ends = itertools.repeat(use_ends)
        if not hasattr(num_splits, "__iter__"):
            num_splits = itertools.repeat(num_splits)
        coords = []
        for bound, n, use_end in zip(bounds, num_splits, use_ends):
            if use_end:
                coords.append(np.linspace(bound[0], bound[1], n))
            else:
                coords.append(np.linspace(bound[0], bound[1], n + 2)[1:-1])
        points = [p for p in itertools.product(*coords)]
        return cls(points, bounds)

    @property
    def points(self):
        """:math:`(N, N_{dim})` numpy.ndarray of float: The points to query."""
        return self._points


class NelderMead(Optimizer):
    def __init__(
        self,
        initial_simplex,
        bounds=None,
        alpha=1,
        gamma=2,
        rho=0.5,
        sigma=0.5,
        max_iter=None,
        dist_tol=1e-2,
        std_tol=1e-3,
    ):
        bounds = Optimizer._default_bounds(bounds, initial_simplex.shape[1])

        if max_iter is None:
            max_iter = 2**16 - 1
        self._cpp = pgop._pgop.NelderMead(
            pgop._pgop.NelderMeadParams(alpha, gamma, rho, sigma),
            initial_simplex.tolist(),
            bounds[:, 0].tolist(),
            bounds[:, 1].tolist(),
            max_iter,
            dist_tol,
            std_tol,
        )

    @staticmethod
    def create_initial_simplex(point, delta):
        simplex = np.tile(point, (len(point) + 1, 1))
        index = np.diag_indices(len(point))
        simplex[1:][index] += delta
        return simplex


class Union(Optimizer):
    @classmethod
    def with_nelder(
        cls,
        optimizer,
        delta,
        alpha=1,
        gamma=2,
        rho=0.5,
        sigma=0.5,
        max_iter=None,
        dist_tol=1e-2,
        std_tol=1e-3,
    ):
        instance = cls()
        if max_iter is None:
            max_iter = 2**16 - 1
        instance._cpp = pgop._pgop.Union.with_nelder_mead(
            optimizer._cpp,
            pgop._pgop.NelderMeadParams(alpha, gamma, rho, sigma),
            max_iter,
            dist_tol,
            std_tol,
            delta,
        )
        return instance

    @classmethod
    def with_mc(
        cls, optimizer, kT=0.1, max_move_size=0.05, seed=0, max_iter=150
    ):
        instance = cls()
        instance._cpp = pgop._pgop.Union.with_mc(
            optimizer._cpp,
            kT,
            max_move_size,
            seed,
            max_iter,
        )
        return instance


class MonteCarlo(Optimizer):
    """Find the optimum using a random MC annealing."""

    def __init__(
        self,
        initial_point,
        kT,
        max_move_size,
        bounds=None,
        seed=0,
        max_iter=150,
    ):
        bounds = Optimizer._default_bounds(bounds, len(initial_point[0]))
        self._cpp = pgop._pgop.MonteCarlo(
            bounds[:, 0].tolist(),
            bounds[:, 1].tolist(),
            initial_point,
            kT,
            max_move_size,
            seed,
            max_iter,
        )
