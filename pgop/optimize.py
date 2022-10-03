import abc
import enum
import itertools

import numpy as np

import pgop._pgop

HYPERSPHERE_BOUNDS = np.array([[0, 2 * np.pi], [np.pi / 2, np.pi], [0, np.pi]])


class Optimizer(abc.ABC):
    # User Interface
    @property
    def optimum(self):
        return self._cpp.optimum

    @property
    def terminated(self):
        return self._cpp.terminate

    def __iter__(self):
        while not self.terminated:
            yield self._cpp.next_point()

    def report_objective(self, objective):
        self._cpp.record_objective(objective)

    @staticmethod
    def _default_bounds(bounds, dim):
        if bounds is not None:
            return np.asarray(bounds)
        return np.tile([-np.inf, np.inf], (dim, 1))


class BruteForce(Optimizer):
    def __init__(self, points, bounds=None, max_iter=None):
        self._points = np.copy(points)
        if self._points.ndim != 2:
            raise ValueError("points must have shape (npoints, ndim)")
        bounds = Optimizer._default_bounds(bounds, self._points.shape[1])
        self._cpp = pgop._pgop.BruteForce(
            self._points.tolist(), bounds[:, 0].tolist(), bounds[:, 1].tolist()
        )

    @classmethod
    def from_mesh(cls, num_splits, bounds=None, use_end=None):
        if bounds is None:
            bounds = HYPERSPHERE_BOUNDS
            use_end = (False, True, True)
        if use_end is None:
            use_end = True
        if isinstance(use_end, bool):
            use_end = itertools.repeat(use_end)
        points = [
            p
            for p in itertools.product(
                *[
                    np.linspace(bound[0], bound[1], n, endpoint=end)
                    for bound, n, end in zip(bounds, num_splits, use_end)
                ]
            )
        ]
        return cls(points, bounds)

    @property
    def points(self):
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
