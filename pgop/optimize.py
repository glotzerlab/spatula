import abc
import enum

import numpy as np

import pgop._pgop


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
        points = np.asarray(points)
        if points.ndim != 2:
            raise ValueError("points must have shape (npoints, ndim)")
        bounds = Optimizer._default_bounds(bounds, points.shape[1])
        self._cpp = pgop._pgop.BruteForce(
            points.tolist(), bounds[:, 0].tolist(), bounds[:, 1].tolist()
        )


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
