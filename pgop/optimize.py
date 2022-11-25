import numpy as np
import rowan
import scipy as sp

from . import _pgop


class RandomSearch:
    def __init__(self, max_iter=200, seed=42):
        self._cpp = _pgop.RandomSearch(max_iter, seed)


class LocalSearch:
    def __init__(self, initial_point=(1.0, 0.0, 0.0, 0.0), max_iter=200):
        self._cpp = _pgop.LocalFIRE(_pgop.Quaternion(initial_point), max_iter)


class Mesh:
    _dodehedron_vertices = np.array(
        [
            [-0.69813, 0.0, 0.13333],
            [0.69813, 0.0, -0.13333],
            [-0.21573, -0.66396, 0.13333],
            [-0.21573, 0.66396, 0.13333],
            [0.5648, -0.41035, 0.13333],
            [0.5648, 0.41035, 0.13333],
            [-0.13333, -0.41035, 0.5648],
            [-0.13333, 0.41035, 0.5648],
            [-0.34907, -0.25361, -0.5648],
            [-0.34907, 0.25361, -0.5648],
            [0.34907, -0.25361, 0.5648],
            [0.34907, 0.25361, 0.5648],
            [0.43147, 0.0, -0.5648],
            [-0.5648, -0.41035, -0.13333],
            [-0.5648, 0.41035, -0.13333],
            [-0.43147, 0.0, 0.5648],
            [0.13333, -0.41035, -0.5648],
            [0.13333, 0.41035, -0.5648],
            [0.21573, -0.66396, -0.13333],
            [0.21573, 0.66396, -0.13333],
        ]
    )

    def __init__(self, points):
        self._cpp = _pgop.Mesh([_pgop.Quaternion(p) for p in points])

    @classmethod
    def from_grid(cls, n_angles=6):
        points = np.empty(
            (n_angles * len(cls._dodehedron_vertices) + 1, 4), dtype=float
        )
        points[0] = np.array([1.0, 0.0, 0.0, 0.0])
        angles = cls.sample_angles(n_angles)
        points[1:] = rowan.from_axis_angle(
            cls._dodehedron_vertices[:, None], angles[None, :]
        ).reshape((-1, 4))
        return cls(points)

    @staticmethod
    def sample_angles(n_angles):
        def cdf(theta, root):
            ipi = 1 / np.pi
            sinx = np.sin(theta)
            return (
                ipi * (theta - sinx) - root,
                ipi * (1 - np.cos(theta)),
                ipi * sinx,
            )

        roots = np.linspace(0, 1, n_angles + 1)[1:]
        angles = []
        for r in roots:
            opt = sp.optimize.root_scalar(
                cdf,
                args=(r,),
                method="halley",
                bracket=[0, np.pi],
                x0=r * np.pi,
                fprime=True,
                fprime2=True,
            )
            angles.append(opt.root)
        return np.array(angles)


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
