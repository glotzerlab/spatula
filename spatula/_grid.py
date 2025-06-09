# Copyright (c) 2010-2025 The Regents of the University of Michigan
# Part of spatula, released under the BSD 3-Clause License.

import numpy as np

"""The Golden ratio, 1.61803398875..."""
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2

"""The angle that divides 2π radians into the golden ratio.

This takes the longer of the two segments π(√5 - 1), where π[(√5 - 1) + (3-√5)] = 2π
"""
GOLDEN_ANGLE = np.pi * (np.sqrt(5) - 1)


"""Ratio of the circumference of a circle to its radius (2π)."""
TAU = 2 * np.pi


def _spherical_to_cartesian(theta, phi):
    """Convert an (array of) angle pairs (θ,φ) to unit vectors."""
    return np.vstack(
        [
            np.cos(theta) * np.sin(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(phi),
        ]
    ).T


def fib_sphere_mod_cn(n: int, cn: int = 1, eps: float = 0.0):
    """Generate a grid of `n` low-discrepancy points on the fundemental domain of Cn.

    See source [here](https://stackoverflow.com/a/26127012/21897583), which provides an
    algorithm with fewer trig operations than most.
    """
    # NOTE: I think it should be possible to use a lower-density sequence than ℤ for t.
    #       However, determining what that sequence should be is not obvious to me
    n *= cn
    t = np.arange(n)

    z = 1 - (t / (n - 1)) * 2
    radius = np.sqrt(1 - z**2)
    theta = (GOLDEN_ANGLE * t) % (2 * np.pi)

    mask = theta <= 2 * np.pi / cn

    x = np.cos(theta) * radius
    y = np.sin(theta) * radius

    return np.column_stack([x, y, z])[mask].T


def _tri_intersect_cramer_origin_zero(ray, v0, v1, v2):
    """Check if a ray intersects a triangle defined by three vertices."""
    v0 = np.asarray(v0)
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)

    # https://facultyweb.cs.wwu.edu/~wehrwes/courses/csci480_20w/lectures/L10/L10.pdf
    denom = 1.0 / np.linalg.det([v1 - v0, v2 - v0, -ray])
    γ = denom * np.linalg.det([v2 - v0, v0, -ray])
    β = denom * np.linalg.det([v1 - v0, -v0, -ray])
    t = denom * np.linalg.det([v1 - v0, v2 - v0, -v0])

    if γ < 0.0 or β < 0.0 or (γ + β) > 1.0:
        t = -1.0

    return t > 0


def fib_sphere_within_domain(n: int, pointgroup: str = "e"):
    """Produce a low-discrepancy grid of `n` axes on a point group's fundemental domain.

    Based on https://openaccess.thecvf.com/content/CVPR2022/papers/Alexa_Super-Fibonacci_Spirals_Fast_Low-Discrepancy_Sampling_of_SO3_CVPR_2022_paper.pdf

    We use Fibonacci spirals to rapidly generate low-discrepancy samples on 𝕊2 (axes)
    or SO(3) (quaternions).
    We can then mask the sampled region to a fundemental domain of a given point group,
    using a ray-triangle intersection or hyperray-tetrahedron intersection.
    """  # noqa: RUF002
    scale = 4  # TODO: should be based on spherical triangle area relative to sphere
    t = np.arange(n * scale)
    n *= scale
    theta = TAU * t / GOLDEN_RATIO  # % (2 * np.pi)
    phi = np.arccos(1 - 2 * t / (n - 1))

    xyz = _spherical_to_cartesian(theta, phi)
    xyz = np.array(
        [
            pt
            for pt in xyz
            if _tri_intersect_cramer_origin_zero(pt, [0, 0, 1], [0, -1, 0], [1, 0, 0])
        ]
    )
    return xyz.T
