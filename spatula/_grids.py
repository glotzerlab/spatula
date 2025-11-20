# Copyright (c) 2021-2025 The Regents of the University of Michigan
# Part of spatula, released under the BSD 3-Clause License.

import itertools

import numpy as np

"""The Golden ratio, 1.61803398875..."""
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2


"""The angle that divides 2π radians into the golden ratio.
This takes the longer of the two segments π(√5 - 1), where π[(√5 - 1) + (3-√5)] = 2π
"""
GOLDEN_ANGLE = np.pi * (np.sqrt(5) - 1)


def _spherical_fibonacci_lattice(n):
    """Generate a Fibonacci lattice of `n` points on the 2-sphere.

    n : `int`
        n (int): The number of points in the lattice.
    """
    t = np.arange(n)

    z = 1 - (t / (n - 1)) * 2
    radius = np.sqrt(1 - z**2)
    theta = (GOLDEN_ANGLE * t) % (2 * np.pi)

    x = np.cos(theta) * radius
    y = np.sin(theta) * radius

    return np.column_stack([x, y, z]).T


def _quaternion_fibonacci_lattice(n):
    """Generate a near-uniform grid of `n` quaternions.

    This is equivalent to a Fibonacci lattice on the 3-sphere. See
    `this paper <https://ieeexplore.ieee.org/document/9878746>`_ for a derivation.
    """
    psi = 1.533751168755204288118041413  # Solution to ψ**4 = ψ + 4
    s = np.arange(n) + 1 / 2
    t = s / n
    d = 2 * np.pi * s
    r0, r1 = (np.sqrt(t), np.sqrt(1 - t))
    α, β = (d / np.sqrt(2), d / psi)

    # Allocate as rows and then transpose, rather than stacking columns
    result = np.empty((4, n))
    result[...] = r0 * np.sin(α), r0 * np.cos(α), r1 * np.sin(β), r1 * np.cos(β)
    return result.T


def _quaternion_outer_product(qi, qj):
    """Compute the outer product of two sets of quaternions."""
    qi = np.asarray(qi)[:, None, :]
    qj = np.asarray(qj)[None, :, :]

    # Determine broadcast shape (N, M, 4)
    output = np.empty(np.broadcast(qi, qj).shape)

    # Scalar part: w1*w2 - dot(v1, v2)
    output[..., 0] = qi[..., 0] * qj[..., 0] - np.sum(
        qi[..., 1:] * qj[..., 1:],
        axis=-1,
    )

    # Vector part: w1*v2 + w2*v1 + cross(v1, v2)
    # Note: axis=-1 is default for cross; slicing preserves the last dim as 3
    output[..., 1:] = (
        qi[..., 0, np.newaxis] * qj[..., 1:]
        + qj[..., 0, np.newaxis] * qi[..., 1:]
        + np.cross(qi[..., 1:], qj[..., 1:])
    )

    # Flatten (N, M, 4) -> (N*M, 4)
    return output.reshape(-1, 4)


def _hyperdodecahedron():
    """Create a Mesh optimizer from the 600 vertices of the hyperdodecahedron."""
    # Construction based on https://www.qfbox.info/4d/120-cell
    φ = GOLDEN_RATIO
    base_coords = [
        [2, 2, 0, 0],
        [np.sqrt(5), 1, 1, 1],
        [φ, φ, φ, φ**-2],
        [φ**2, φ**-1, φ**-1, φ**-1],
    ]

    even_coords = [
        [φ**2, φ**-2, 1, 0],
        [np.sqrt(5), φ**-1, φ, 0],
        [2, 1, φ, φ**-1],
    ]

    # Generate all vertices from base_coords
    vertices = [
        tuple(s * sign for s, sign in zip(perm, signs))
        for coords in base_coords
        for perm in itertools.permutations(coords)
        for signs in itertools.product([-1, 1], repeat=len(perm))
    ]

    # Generate all vertices from even_coords with even permutations only
    perms = np.array([*itertools.permutations([0, 1, 2, 3])])
    even_perms = [p for p in perms if np.linalg.det(np.eye(4, dtype=int)[p]) > 0]

    vertices += [
        tuple(coord * sign for coord, sign in zip(np.array(coords)[list(p)], signs))
        for coords in even_coords
        for p in even_perms
        for signs in itertools.product([-1, 1], repeat=len(coords))
    ]

    return np.unique(vertices, axis=0) / np.sqrt(8)
