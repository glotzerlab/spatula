from pathlib import Path

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


def _expand_shape(shape):
    verts = np.empty((len(shape.vertices) + len(shape.faces), 3))
    verts[: len(shape.vertices)] = shape.vertices
    for i, f in enumerate(shape.faces, start=len(shape.vertices)):
        center = np.mean(shape.vertices[f], axis=0)
        verts[i] = center / np.linalg.norm(center)
    return coxeter.shapes.ConvexPolyhedron(verts)


def _load_sphere_codes():
    fn = Path(__file__).parent / "sphere-codes.npz"
    with np.load(str(fn)) as data:
        return [arr for arr in data.values()]


class Mesh:

    _sphere_codes = _load_sphere_codes()

    def __init__(self, points):
        self._cpp = _pgop.Mesh([_pgop.Quaternion(p) for p in points])

    @classmethod
    def from_grid(cls, n_axes=40, n_angles=5):
        if n_axes < 1 or n_axes > 250:
            raise ValueError("Can only chose [1, 250] for n_axes.")
        axes = cls._sphere_codes[n_axes - 1].reshape((-1, 1, 3))
        points = np.empty((n_angles * n_axes + 1, 4), dtype=float)
        points[0] = np.array([1.0, 0.0, 0.0, 0.0])
        angles = cls.sample_angles(n_angles).reshape((1, -1, 1))
        points[1:] = rowan.from_axis_angle(
            axes[:, None, :], angles[None, :, None]
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
