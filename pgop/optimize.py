import abc
import enum

import numpy as np


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
        self._trial_objective = None
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


class _OrderedSimplex:
    def __init__(self, dim):
        self._point = np.zeros((dim + 1, dim))
        self._objective = np.full(dim + 1, np.inf)

    def add(self, point, objective):
        index = np.sum(objective > self._objective)
        if index == len(self):
            return
        self._objective[index + 1 :] = self._objective[index:-1]
        self._point[index + 1 :] = self._point[index:-1]
        self._point[index] = point
        self._objective[index] = objective

    def __getitem__(self, index):
        return self._point[index]

    def __len__(self):
        return len(self._objective)

    @property
    def obj(self):
        return self._objective


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


# todo: This may add an extra iteration currently since __iter__ checks the
# termination condition, but we only check the current simplex which may change
# in self._next_point not the trial point.
class NelderMead(Optimizer):
    class Stages(enum.IntEnum):
        NEW_SIMPLEX = 0
        REFLECT = 1
        EXPAND = 2
        OUTSIDE_CONTRACT = 3
        INSIDE_CONTRACT = 4
        TERMINATE = 6

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
        initial_simplex = np.asarray(initial_simplex)
        if initial_simplex.ndim != 2:
            raise ValueError("initial_simplex must have shape (ndim + 1, ndim)")
        self._new_simplex = initial_simplex
        self._dim = self._new_simplex.shape[1]
        self._current_simplex = _OrderedSimplex(self._dim)
        super().__init__(_default_bounds(bounds, self._dim), max_iter)

        # Algorithm dof
        self._alpha = alpha
        self._gamma = gamma
        self._rho = rho
        self._sigma = sigma

        # Tolerances
        self._dist_tol = dist_tol
        self._std_tol = std_tol

        # Internal state variables
        self._state = self.Stages.NEW_SIMPLEX

        # stage specific attributes
        self._reflect_data = None
        self._new_simplex_index = 0

    @property
    def optimum(self):
        return self._current_simplex[0], self._current_simplex.obj[0]

    @property
    def terminated(self):
        if self._state == self.Stages.NEW_SIMPLEX:
            return False
        return self._points_too_close() or self._obj_too_close()

    @staticmethod
    def create_initial_simplex(point, delta):
        simplex = np.tile(point, (len(point) + 1, 1))
        index = np.diag_indices(len(point))
        simplex[1:][index] += delta
        return simplex

    def _next_point(self):
        if self._state == self.Stages.NEW_SIMPLEX:
            if self._new_simplex_index != 0:
                self._current_simplex.add(self._trial_point, self._trial_objective)
            if self._new_simplex_index == self._dim + 1:
                return self._reflect()
            else:
                self._new_simplex_index += 1
                point = self._new_simplex[self._new_simplex_index - 1]
                return point
        if self._state == self.Stages.REFLECT:
            if self._trial_objective < self._current_simplex.obj[0]:
                return self._expand()
            if self._trial_objective < self._current_simplex.obj[-2]:
                self._current_simplex.add(self._trial_point, self._trial_objective)
                return self._reflect()
            if self._trial_objective < self._current_simplex.obj[-1]:
                return self._outside_contract()
            return self._inside_contract()
        if self._state == self.Stages.EXPAND:
            reflect_pnt, reflect_obj = self._reflect_data
            if self._trial_objective < reflect_obj:
                self._current_simplex.add(self._trial_point, self._trial_objective)
            else:
                self._current_simplex.add(reflect_pnt, reflect_obj)
            return self._reflect()
        if self._state == self.Stages.OUTSIDE_CONTRACT:
            reflect_pnt, reflect_obj = self._reflect_data
            if self._trial_objective < reflect_obj:
                self._current_simplex.add(self._trial_point, self._trial_objective)
                return self._reflect()
            return self._shrink()
        if self._state == self.Stages.INSIDE_CONTRACT:
            if self._trial_objective < self._current_simplex.obj[-1]:
                self._current_simplex.add(self._trial_point, self._trial_objective)
                return self._reflect()
            return self._shrink()

    def _reflect(self):
        self._state = self.Stages.REFLECT
        centroid = self._centroid
        delta = self._alpha * (centroid - self._current_simplex[-1])
        return centroid + delta

    def _expand(self):
        self._state = self.Stages.EXPAND
        self._reflect_data = (self._trial_point, self._trial_objective)
        centroid = self._centroid
        delta = self._gamma * (self._trial_point - centroid)
        return centroid + delta

    def _outside_contract(self):
        self._state = self.Stages.OUTSIDE_CONTRACT
        self._reflect_data = (self._trial_point, self._trial_objective)
        centroid = self._centroid
        delta = self._rho * (self._trial_point - centroid)
        return centroid + delta

    def _inside_contract(self):
        self._state = self.Stages.INSIDE_CONTRACT
        centroid = self._centroid
        delta = self._rho * (self._current_simplex[-1] - centroid)
        return centroid + delta

    def _shrink(self):
        best_point, obj = self._current_simplex[0], self._current_simplex.obj[0]
        replace_pnts = self._current_simplex[1:]
        delta = self._sigma * (replace_pnts - best_point)
        self._new_simplex[0] = best_point
        self._new_simplex[1:] = replace_pnts + delta
        self._new_simplex_index = 1
        # Reset simplex
        self._current_simplex = _OrderedSimplex(self._dim)
        self._current_simplex.add(best_point, obj)
        self._state = self.Stages.NEW_SIMPLEX
        return self._new_simplex[1]

    @property
    def _centroid(self):
        return np.mean(self._current_simplex[:-1], axis=0)

    def _points_too_close(self):
        dist = self._current_simplex[None, ...] - self._current_simplex[:, None, :]
        dist = np.linalg.norm(dist.reshape((-1, self._dim)), axis=1)
        return dist.max() < self._dist_tol

    def _obj_too_close(self):
        if self._dim > 1:
            diff = self._current_simplex.obj.std()
        else:
            diff = self._current_simplex.obj[1] - self._current_simplex.obj[0]
        return diff < self._std_tol
