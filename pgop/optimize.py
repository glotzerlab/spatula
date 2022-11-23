import numpy as np

from . import _pgop


class RandomSearch:
    def __init__(self, max_iter=200, seed=42):
        self._cpp = _pgop.RandomSearch(max_iter, seed)


class LocalSearch:
    def __init__(self, initial_point=(1.0, 0.0, 0.0, 0.0), max_iter=200):
        self._cpp = _pgop.LocalFIRE(_pgop.Quaternion(initial_point), max_iter)


class Mesh:
    def __init__(self, points):
        self._cpp = _pgop.Mesh([_pgop.Quaternion(p) for p in points])


class LocalMonteCarlo:
    def __init__(
        self,
        initial_point=None,
        kT=1.0,
        max_theta=0.017,
        seed=42,
        iterations=200,
    ):
        if initial_point is None:
            initial_point = ((1.0, 0.0, 0.0, 0.0), np.inf)
        self._cpp = _pgop.QMonteCarlo(
            (_pgop.Quaternion(initial_point[0]), initial_point[1]),
            kT,
            max_theta,
            seed,
            iterations,
        )


class Union:
    @classmethod
    def with_fire(cls, optimizer, max_iter=200):
        instance = cls()
        instance._cpp = _pgop.QUnion.with_fire(optimizer._cpp, max_iter)
        return instance

    @classmethod
    def with_mc(
        cls, optimizer, kT=1.0, max_theta=0.017, seed=42, iterations=200
    ):
        instance = cls()
        instance._cpp = _pgop.QUnion.with_mc(
            optimizer._cpp, kT, max_theta, seed, iterations
        )
        return instance
