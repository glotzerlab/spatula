# Copyright (c) 2010-2025 The Regents of the University of Michigan
# Part of spatula, released under the BSD 3-Clause License.

import numpy as np

from spatula._grid import _spherical_to_cartesian


def test_spherical_to_cartesian():
    theta = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    phi = np.array([np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2])
    expected = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])

    result = _spherical_to_cartesian(theta, phi)
    np.testing.assert_array_almost_equal(result, expected)
    np.testing.assert_array_almost_equal(_spherical_to_cartesian(0, 0), [[0, 0, 1]])
    np.testing.assert_array_almost_equal(
        _spherical_to_cartesian(0, np.pi), [[0, 0, -1]]
    )


# TODO: can test based on measured area of spherical caps, as in https://arxiv.org/pdf/0912.4540
