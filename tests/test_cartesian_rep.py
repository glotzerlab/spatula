import numpy as np
import pytest

from pgop.cartrep import (
    CartesianRepMatrix,
    identity,
    rotation_from_euler_angles,
)

order_range_to_test = range(2, 13)
order_range_to_test_odd = range(3, 13, 2)


def test_cartesian_init():
    cart = CartesianRepMatrix("C2")
    assert cart.point_group == "C2"


def test_cartesian_valid_point_group():
    cart = CartesianRepMatrix("C2")
    assert isinstance(cart.condensed_matrices, np.ndarray)


def test_cartesian_valid_point_group2():
    cart = CartesianRepMatrix("C1")
    assert isinstance(cart.condensed_matrices, np.ndarray)


def test_cartesian_invalid_point_group():
    with pytest.raises(KeyError):
        _ = CartesianRepMatrix("J")


def test_identity():
    assert np.allclose(identity(), rotation_from_euler_angles(0, 0, 0))


#
#
# @pytest.mark.parametrize("n", order_range_to_test_odd)
# def test_Sn_odd_equivalent_to_Cnh(n):
#    """https://en.wikipedia.org/wiki/Schoenflies_notation#Point_groups"""
#    assert np.allclose(
#        compute_condensed_wignerD_matrix_for_a_given_point_group("S" + str(n), maxl),
#        compute_condensed_wignerD_matrix_for_a_given_point_group(
#            "C" + str(n) + "h", maxl
#        ),
#    )


# def test_C2x_from_operations():
#    """According to paper by Engel (https://arxiv.org/pdf/2106.14846)
#    Note: Engel's paper has values for 2x and 2y swapped!
#    """
#    assert np.allclose(
#        np.array(
#            [
#                delta(mprime, -m) * ((-1) ** (l + m))
#                for l, mprime, m in iter_sph_indices(maxl)
#            ],
#            dtype=complex,
#        ),
#        dot_product(inversion(maxl), reflection_from_normal(maxl, [1, 0, 0])),
#    )
#
#
# def test_C2y_from_operations():
#    """According to paper by Engel (https://arxiv.org/pdf/2106.14846)
#    Note: Engel's paper has values for 2x and 2y swapped!
#    """
#    assert np.allclose(
#        np.array(
#            [
#                delta(mprime, -m) * ((-1) ** (l))
#                for l, mprime, m in iter_sph_indices(maxl)
#            ],
#            dtype=complex,
#        ),
#        dot_product(inversion(maxl), reflection_from_normal(maxl, [0, 1, 0])),
#    )
#
#
# def test_C2y_rotation_from_euler_angles():
#    """According to paper by Engel (https://arxiv.org/pdf/2106.14846)
#    Note: Engel's paper has values for 2x and 2y swapped!
#    """
#    assert np.allclose(
#        np.array(
#            [
#                delta(mprime, -m) * ((-1) ** (l))
#                for l, mprime, m in iter_sph_indices(maxl)
#            ],
#            dtype=complex,
#        ),
#        rotation_from_axis_order(maxl, [0, 1, 0], 2),
#    )
#
#
# def test_C2x_rotation_from_euler_angles():
#    """According to paper by Engel (https://arxiv.org/pdf/2106.14846)
#    Note: Engel's paper has values for 2x and 2y swapped!
#    """
#    assert np.allclose(
#        np.array(
#            [
#                delta(mprime, -m) * ((-1) ** (l + m))
#                for l, mprime, m in iter_sph_indices(maxl)
#            ],
#            dtype=complex,
#        ),
#        rotation_from_axis_order(maxl, [1, 0, 0], 2),
#    )
#
#
# @pytest.mark.parametrize("n", order_range_to_test)
# def test_Cz_rotation_from_euler_angles(n):
#    """According to Engel's paper (https://arxiv.org/pdf/2106.14846).
#    Note: Engel's paper has a minus in the exponent which is not present in altmann.
#    """
#    assert np.allclose(
#        np.array(
#            [
#                delta(mprime, m) * np.exp(2 * np.pi * m * 1j / n)
#                for _, mprime, m in iter_sph_indices(maxl)
#            ],
#            dtype=complex,
#        ),
#        rotation_from_axis_order(maxl, [0, 0, 1], n),
#    )
#
#
# @pytest.mark.parametrize("n", order_range_to_test)
# def test_generalized_rotation_self_correct(n):
#    assert np.allclose(
#        rotation_from_euler_angles(maxl, 2 * np.pi / n, 0, 0),
#        rotation_from_euler_angles(maxl, 0, 0, 2 * np.pi / n),
#    )
#
#
# @pytest.mark.parametrize("n", order_range_to_test)
# def test_generalized_rotation_against_axis_angle(n):
#    assert np.allclose(
#        rotation_from_euler_angles(maxl, 2 * np.pi / n, 0, 0),
#        rotation_from_axis_angle(maxl, [0, 0, 1], 2 * np.pi / n),
#    )
#
#
# @pytest.mark.parametrize("n", order_range_to_test)
# def test_generalized_rotation_against_axis_order(n):
#    assert np.allclose(
#        rotation_from_euler_angles(maxl, 2 * np.pi / n, 0, 0),
#        rotation_from_axis_order(maxl, [0, 0, 1], n),
#    )
#
#
# def test_inversion():
#    """According to Engel's paper (https://arxiv.org/pdf/2106.14846)"""
#    m_inv = np.array(
#        [delta(mprime, m) * ((-1) ** l) for l, mprime, m in iter_sph_indices(maxl)],
#        dtype=complex,
#    )
#    assert np.allclose(m_inv, inversion(maxl))
#
#
# def test_inversion_as_reflections():
#    """According to Engel's paper (https://arxiv.org/pdf/2106.14846) :"""
#    assert np.allclose(
#        inversion(maxl),
#        dot_product(
#            dot_product(
#                reflection_from_normal(maxl, [1, 0, 0]),
#                reflection_from_normal(maxl, [0, 1, 0]),
#            ),
#            reflection_from_normal(maxl, [0, 0, 1]),
#        ),
#    )
#
#
# def test_sigmaxy():
#    """According to Engel's paper (https://arxiv.org/pdf/2106.14846)"""
#    assert np.allclose(
#        reflection_from_normal(maxl, [0, 0, 1]),
#        np.array(
#            [
#                delta(mprime, m) * ((-1) ** (m + l))
#                for l, mprime, m in iter_sph_indices(maxl)
#            ],
#            dtype=complex,
#        ),
#    )
#
#
# def test_sigmaxz():
#    """According to Engel's paper (https://arxiv.org/pdf/2106.14846)
#    Note: Engel's paper has values for sigma_xz and sigma_yz swapped!"""
#    assert np.allclose(
#        reflection_from_normal(maxl, [0, 1, 0]),
#        np.array(
#            [delta(mprime, -m) for _, mprime, m in iter_sph_indices(maxl)],
#            dtype=complex,
#        ),
#    )
#
#
# def test_sigmayz():
#    """According to Engel's paper (https://arxiv.org/pdf/2106.14846).
#    Note: Engel's paper has values for sigma_xz and sigma_yz swapped!"""
#    assert np.allclose(
#        reflection_from_normal(maxl, [1, 0, 0]),
#        np.array(
#            [
#                delta(mprime, -m) * ((-1) ** m)
#                for _, mprime, m in iter_sph_indices(maxl)
#            ],
#            dtype=complex,
#        ),
#    )
#
#
# @pytest.mark.parametrize("n", order_range_to_test)
# def test_general_rotoreflection_self_correct(n):
#    assert np.allclose(
#        rotoreflection_from_euler_angles(maxl, 2 * np.pi / n, 0, 0),
#        rotoreflection_from_euler_angles(maxl, 0, 0, 2 * np.pi / n),
#    )
#
#
# @pytest.mark.parametrize("n", order_range_to_test)
# def test_general_rotoreflection_against_from_axis_angle(n):
#    assert np.allclose(
#        rotoreflection_from_euler_angles(maxl, 2 * np.pi / n, 0, 0),
#        rotoreflection_from_axis_angle(maxl, [0, 0, 1], 2 * np.pi / n),
#    )
#
#
# @pytest.mark.parametrize("n", order_range_to_test)
# def test_general_rotoreflection_against_axis_order(n):
#    assert np.allclose(
#        rotoreflection_from_euler_angles(maxl, 2 * np.pi / n, 0, 0),
#        rotoreflection_from_axis_order(maxl, [0, 0, 1], n),
#    )
#
#
# def test_Ci():
#    """According to Engel's paper (https://arxiv.org/pdf/2106.14846) :"""
#    assert np.isclose(
#        np.array(
#            [
#                delta(mprime, m) * delta(l % 2, 0)
#                for l, mprime, m in iter_sph_indices(maxl)
#            ],
#            dtype=complex,
#        ),
#        compute_condensed_wignerD_matrix_for_a_given_point_group("Ci", maxl),
#    ).all()
#
#
# @pytest.mark.parametrize("n", order_range_to_test)
# def test_Cn(n):
#    """According to Engel's paper (https://arxiv.org/pdf/2106.14846) :"""
#    w = np.array(
#       [delta(mprime, m) * delta(m % n, 0) for _, mprime, m in iter_sph_indices(maxl)],
#        dtype=complex,
#    )
#    assert np.isclose(
#        w, compute_condensed_wignerD_matrix_for_a_given_point_group("C" + str(n), maxl)
#    ).all()
#
#
# @pytest.mark.parametrize("n", order_range_to_test)
# def test_Cn_against_scipy_rotations_euler(n):
#    operations = []
#    for i in scipy.spatial.transform.Rotation.create_group("C" + str(n)):
#        operations.append(rotation_from_euler_angles(maxl, *i.as_euler("zyz")))
#    assert np.allclose(
#        compute_condensed_wignerD_matrix_for_a_given_point_group("C" + str(n), maxl),
#        condensed_wignerD_from_operations(operations),
#    )
#
#
# @pytest.mark.parametrize("n", order_range_to_test)
# def test_Cn_against_scipy_rotations_rotvec(n):
#    operations = []
#    for i in scipy.spatial.transform.Rotation.create_group("C" + str(n)):
#        # identity has no axis, so we have to check for it otherwise errors
#        if np.isclose(i.as_rotvec(), [0, 0, 0]).all():
#            operations.append(identity(maxl))
#        else:
#            axis = i.as_rotvec()
#            angle = np.linalg.norm(axis)
#            axis = axis / angle
#            operations.append(rotation_from_axis_angle(maxl, axis, angle))
#    assert np.allclose(
#        compute_condensed_wignerD_matrix_for_a_given_point_group("C" + str(n), maxl),
#        condensed_wignerD_from_operations(operations),
#    )
#
#
# @pytest.mark.parametrize("n", order_range_to_test)
# def test_Dn_against_scipy_rotations_euler(n):
#    operations = []
#    for i in scipy.spatial.transform.Rotation.create_group("D" + str(n)):
#        euler_angles = np.asarray(i.as_euler("zyz"))
#       # Fix for scipy starting from C2' that aligns with x axis, while I start with y.
#        if n % 2 == 1 and np.isclose(euler_angles[1], np.pi):
#            euler_angles[0] = euler_angles[0] + np.pi
#        operations.append(rotation_from_euler_angles(maxl, *euler_angles))
#    assert np.allclose(
#        compute_condensed_wignerD_matrix_for_a_given_point_group("D" + str(n), maxl),
#        condensed_wignerD_from_operations(operations),
#    )
#
#
# @pytest.mark.parametrize("n", order_range_to_test)
# def test_Dn_against_scipy_rotations_rotvec(n):
#    operations = []
#    for i in scipy.spatial.transform.Rotation.create_group("D" + str(n)):
#        # identity has no axis, so we have to check for it otherwise errors
#        if np.isclose(i.as_rotvec(), [0, 0, 0]).all():
#            operations.append(identity(maxl))
#        else:
#            axis = i.as_rotvec()
#            angle = np.linalg.norm(axis)
#            axis = axis / angle
#            # Fix for scipy starting from C2' that aligns with x axis, I start with y.
#            if n % 2 == 1:
#                axis = (
#                    scipy.spatial.transform.Rotation.from_euler(
#                        "zyz", [np.pi / 2, 0, 0]
#                    ).as_matrix()
#                    @ axis
#                )
#            operations.append(rotation_from_axis_angle(maxl, axis, angle))
#    assert np.allclose(
#        compute_condensed_wignerD_matrix_for_a_given_point_group("D" + str(n), maxl),
#        condensed_wignerD_from_operations(operations),
#    )
#
#
# @pytest.mark.parametrize("n", order_range_to_test)
# def test_Dn(n):
#    """According to Engel's paper (https://arxiv.org/pdf/2106.14846)"""
#    assert np.allclose(
#        np.array(
#            [
#                (delta(mprime, m) + delta(mprime, -m) * (-1) ** l) * delta(m % n, 0)
#                for l, mprime, m in iter_sph_indices(maxl)
#            ],
#            dtype=complex,
#        )
#        / 2,
#        compute_condensed_wignerD_matrix_for_a_given_point_group("D" + str(n), maxl),
#    )
