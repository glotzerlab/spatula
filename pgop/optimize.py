import abc
import enum

import numpy as np

from . import _pgop


class Optimizer(abc.ABC):
    def __init__(
        self,
        bounds,
        max_iter=None,
    ):
        self._max_iter = max_iter

        self._bounds = np.asarray(bounds).reshape((-1, 2))

        # Internal state variables
        self._cnt = 0
        self._trial_point = None
        self._trial_objective = np.inf
        self._require_eval = False

    # User Interface
    def __iter__(self):
        while not (self._reached_max_iter() or self.terminated):
            self._trial_point = self._clip(self._next_point())
            self._require_eval = True
            self._cnt += 1
            yield self._trial_point
            if self._require_eval:
                raise RuntimeError(
                    "Must provide objective function value for last point "
                    "before getting the next."
                )

    def report_objective(self, objective):
        if not self._require_eval:
            raise RuntimeError(
                "Must query new point to evaluate before reporting objective "
                "function value."
            )
        self._require_eval = False
        self._trial_objective = objective

    @property
    @abc.abstractmethod
    def optimum(self):
        pass

    @property
    @abc.abstractmethod
    def terminated(self):
        pass

    @abc.abstractmethod
    def _next_point(self):
        pass

    def _reached_max_iter(self):
        return self._max_iter is not None and self._cnt >= self._max_iter

    def _clip(self, value):
        return np.clip(value, self._bounds[:, 0], self._bounds[:, 1])


def _default_bounds(bounds, dim):
    if bounds is not None:
        return bounds
    return np.tile([-np.inf, np.inf], (dim, 1))


class BruteForce(Optimizer):
    def __init__(self, points, bounds=None, max_iter=None):
        points = np.asarray(points)
        if points.ndim != 2:
            raise ValueError("points must have shape (npoints, ndim)")
        self._points = points
        self._dim = points.shape[1]
        self._best_objective = np.inf
        self._best_point = None

        super().__init__(_default_bounds(bounds, self._dim), max_iter)

    def _next_point(self):
        if self._trial_objective is None:
            return self._points[self._cnt]
        if self._best_objective > self._trial_objective:
            self._best_point = self._trial_point
            self._best_objective = self._trial_objective
        if self._cnt == len(self._points):
            return None
        return self._points[self._cnt]

    @property
    def optimum(self):
        return self._best_point, self._best_objective

    @property
    def terminated(self):
        return self._cnt >= len(self._points)


class Histogram(Optimizer):
    def __init__(self, nbins, bounds, max_iter, n_recompute=10, size=100):
        self._nbins = nbins
        self._bin_skip = np.flip(np.cumprod([1] + np.flip(nbins[1:]).tolist()))
        self._size = size
        self._bin_counts = np.zeros(size)
        self._dim = nbins.shape[0]
        super().__init__(bounds, max_iter)
        self._bin_delta = (
            self._bounds[:, 1] - self._bounds[:, 0]
        ) / self._nbins
        self.reset()
        self._n_recompute = n_recompute
        self._n_points = 0
        self._best_points = np.array([])

    def add_point(self, point):
        index = ((point - self._bounds[:, 0]) / self._bin_delta).astype(
            dtype=int
        )
        self._bin_counts[self._n_points % self._size] = np.sum(
            index * self._bin_skip
        )
        self._n_points += 1
        if self._n_points % self._n_recompute == 0:
            n_points = min(self._max_iter, self._n_points)
            flattened_indices, counts = np.unique(
                self._bin_counts[: self._n_points], return_counts=True
            )
            order = np.argsort(counts)[-n_points:]
            flattened_indices = flattened_indices[order]
            self._best_points = np.empty((len(order), self._dim))
            for i in range(self._dim):
                self._best_points[:, i] = flattened_indices // self._bin_skip[i]
                np.remainder(
                    flattened_indices, self._bin_skip[i], out=flattened_indices
                )

    def reset(self):
        self._cnt = 0
        self._best_objective = np.inf
        self._best_point = None
        self._require_eval = False

    def _next_point(self):
        if self._cnt >= self._best_points.shape[0]:
            return None
        if self._best_objective > self._trial_objective:
            self._best_point = self._trial_point
            self._best_objective = self._trial_objective
        new_point = self._best_points[self._cnt] * (0.5 + self._bin_delta)
        return new_point

    @property
    def optimum(self):
        return self._best_point, self._best_objective

    @property
    def terminated(self):
        return self._best_points.shape[0] >= self._cnt


# todo: This may add an extra iteration currently since __iter__ checks the
# termination condition, but we only check the current simplex which may change
# in self._next_point not the trial point.
class NelderMead:
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
        if max_iter is None:
            max_iter = 2**16 - 1
        self._cpp = _pgop.NelderMead(
            _pgop.NelderMeadParams(alpha, gamma, rho, sigma),
            initial_simplex.tolist(),
            bounds[:, 0].tolist(),
            bounds[:, 1].tolist(),
            max_iter,
            dist_tol,
            std_tol,
        )

    @property
    def optimum(self):
        return self._cpp.optimum

    @property
    def terminated(self):
        return self._cpp.terminate

    @staticmethod
    def create_initial_simplex(point, delta):
        simplex = np.tile(point, (len(point) + 1, 1))
        index = np.diag_indices(len(point))
        simplex[1:][index] += delta
        return simplex

    # User Interface
    def __iter__(self):
        while not self.terminated:
            yield self._cpp.next_point()

    def report_objective(self, objective):
        self._cpp.record_objective(objective)
