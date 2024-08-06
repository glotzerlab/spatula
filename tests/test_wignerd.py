import h5py
import numpy as np
import pytest

import pgop
from pgop.wignerd import (
    Ch,
    Ci,
    Cn,
    Cnh,
    Cnv,
    Cs,
    Dn,
    Dnd,
    Dnh,
    Sn,
    WignerD,
    _parse_point_group,
    base_rotations,
    compute_condensed_wignerD_matrix_for_a_given_point_group,
    condensed_wignerD_from_operations,
    delta,
    dot_product,
    generalized_rotation,
    generalized_rotation_from_axis_angle,
    generalized_rotation_from_axis_order,
    generalized_rotoreflection,
    generalized_rotoreflection_from_axis_angle,
    generalized_rotoreflection_from_axis_order,
    generalized_sigma,
    identity,
    inversion,
    iter_sph_indices,
    n_z,
    rotoreflection_z,
    semidirect_product,
    sigma_xy,
    sigma_xz,
    sigma_yz,
    two_x,
    two_y,
)


def test_parse_point_group():
    assert _parse_point_group("T") == ("T", None, None)
    assert _parse_point_group("Th") == ("T", "h", None)
    assert _parse_point_group("C4") == ("C", None, 4)


def test_WignerD_init():
    wig = WignerD("C2", 10)
    assert wig.max_l == 10


def test_WignerD_valid_point_group():
    wig = WignerD("C2", 10)
    assert isinstance(wig.condensed_matrices, np.ndarray)


def test_WignerD_invalid_point_group():
    with pytest.raises(KeyError):
        _ = WignerD("J", 10)


def test_WignerD_iter_sph_indices():
    indices = list(pgop.wignerd.iter_sph_indices(2))
    expected_indices = [
        (0, 0, 0),
        (1, -1, -1),
        (1, -1, 0),
        (1, -1, 1),
        (1, 0, -1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, -1),
        (1, 1, 0),
        (1, 1, 1),
        (2, -2, -2),
        (2, -2, -1),
        (2, -2, 0),
        (2, -2, 1),
        (2, -2, 2),
        (2, -1, -2),
        (2, -1, -1),
        (2, -1, 0),
        (2, -1, 1),
        (2, -1, 2),
        (2, 0, -2),
        (2, 0, -1),
        (2, 0, 0),
        (2, 0, 1),
        (2, 0, 2),
        (2, 1, -2),
        (2, 1, -1),
        (2, 1, 0),
        (2, 1, 1),
        (2, 1, 2),
        (2, 2, -2),
        (2, 2, -1),
        (2, 2, 0),
        (2, 2, 1),
        (2, 2, 2),
    ]
    assert indices == expected_indices


maxl = 12


# test operations themselves


def test_identity():
    assert np.allclose(identity(maxl), generalized_rotation(maxl, 0, 0, 0))


def test_two_x_from_operations():
    """According to paper by Michael (https://arxiv.org/pdf/2106.14846)"""
    assert np.allclose(two_x(maxl), dot_product(inversion(maxl), sigma_yz(maxl)))


def test_two_y_from_operations():
    """According to paper by Michael (https://arxiv.org/pdf/2106.14846)"""
    assert np.allclose(two_y(maxl), dot_product(inversion(maxl), sigma_xz(maxl)))


@pytest.mark.parametrize("n", range(2, 12))
def test_generalized_rotation_self_correct(n):
    assert np.allclose(
        generalized_rotation(maxl, 2 * np.pi / n, 0, 0),
        generalized_rotation(maxl, 0, 0, 2 * np.pi / n),
    )


@pytest.mark.parametrize("n", range(2, 12))
def test_generalized_rotation_against_axis_angle(n):
    assert np.allclose(
        generalized_rotation(maxl, 2 * np.pi / n, 0, 0),
        generalized_rotation_from_axis_angle(maxl, [0, 0, 1], 2 * np.pi / n),
    )


@pytest.mark.parametrize("n", range(2, 12))
def test_generalized_rotation_against_axis_order(n):
    assert np.allclose(
        generalized_rotation(maxl, 2 * np.pi / n, 0, 0),
        generalized_rotation_from_axis_order(maxl, [0, 0, 1], n),
    )


def test_two_x_from_generalized_rotation():
    """According to 7S. L. Altmann, “On the symmetries of spherical harmonics,”
    Mathematical Proceedings of the Cambridge Philosophical Society, page 347"""
    assert np.allclose(two_x(maxl), generalized_rotation(maxl, np.pi, np.pi, 0))


def test_two_y_from_generalized_rotation():
    """According to 7S. L. Altmann, “On the symmetries of spherical harmonics,”
    Mathematical Proceedings of the Cambridge Philosophical Society, page 347"""
    assert np.allclose(two_y(maxl), generalized_rotation(maxl, 0, np.pi, 0))


@pytest.mark.parametrize("n", range(2, 12))
def test_n_z_from_generalized_rotation(n):
    """According to 7S. L. Altmann, “On the symmetries of spherical harmonics,”
    Mathematical Proceedings of the Cambridge Philosophical Society, page 347"""
    for i, j in zip(n_z(maxl, n), generalized_rotation(maxl, 2 * np.pi / n, 0, 0)):
        if not np.allclose(i, j):
            print(i, j)
    print(n, n_z(maxl, n), generalized_rotation(maxl, 2 * np.pi / n, 0, 0))
    assert np.allclose(n_z(maxl, n), generalized_rotation(maxl, 2 * np.pi / n, 0, 0))


def test_inversion():
    """According to Michael's paper (https://arxiv.org/pdf/2106.14846) :"""
    m_inv = np.array(
        [delta(mprime, m) * ((-1) ** l) for l, mprime, m in iter_sph_indices(maxl)],
        dtype=complex,
    )
    assert np.allclose(m_inv, inversion(maxl))


def test_inversion_as_reflections():
    """According to paper by Michael (https://arxiv.org/pdf/2106.14846)"""
    assert np.allclose(
        inversion(maxl),
        dot_product(dot_product(sigma_yz(maxl), sigma_xz(maxl)), sigma_xy(maxl)),
    )


def test_sigmaxy():
    """According to 7S. L. Altmann, “On the symmetries of spherical harmonics,”
    Mathematical Proceedings of the Cambridge Philosophical Society, page 347"""
    sxy = dot_product(inversion(maxl), n_z(maxl, 2))
    assert np.allclose(sigma_xy(maxl), sxy)


def test_sigmaxz():
    """According to 7S. L. Altmann, “On the symmetries of spherical harmonics,”
    Mathematical Proceedings of the Cambridge Philosophical Society, page 347"""
    sxz = dot_product(inversion(maxl), two_y(maxl))
    assert np.allclose(sigma_xz(maxl), sxz)


def test_sigmayz():
    """According to 7S. L. Altmann, “On the symmetries of spherical harmonics,”
    Mathematical Proceedings of the Cambridge Philosophical Society, page 347"""
    syz = dot_product(inversion(maxl), two_x(maxl))
    assert np.allclose(sigma_yz(maxl), syz)


def test_sigma_yz_from_generalized_reflection():
    assert np.allclose(sigma_yz(maxl), generalized_sigma(maxl, [1, 0, 0]))


def test_sigma_xz_from_generalized_reflection():
    assert np.allclose(sigma_xz(maxl), generalized_sigma(maxl, [0, 1, 0]))


def test_sigma_xy_from_generalized_reflection():
    assert np.allclose(sigma_xy(maxl), generalized_sigma(maxl, [0, 0, 1]))


@pytest.mark.parametrize("n", range(2, 12))
def test_general_rotoreflection_self_correct(n):
    assert np.allclose(
        generalized_rotoreflection(maxl, 2 * np.pi / n, 0, 0),
        generalized_rotoreflection(maxl, 0, 0, 2 * np.pi / n),
    )


@pytest.mark.parametrize("n", range(2, 12))
def test_general_rotoreflection_agains_from_axis_angle(n):
    assert np.allclose(
        generalized_rotoreflection(maxl, 2 * np.pi / n, 0, 0),
        generalized_rotoreflection_from_axis_angle(maxl, [0, 0, 1], 2 * np.pi / n),
    )


@pytest.mark.parametrize("n", range(2, 12))
def test_general_rotoreflection_against_axis_order(n):
    assert np.allclose(
        generalized_rotoreflection(maxl, 2 * np.pi / n, 0, 0),
        generalized_rotoreflection_from_axis_order(maxl, [0, 0, 1], n),
    )


@pytest.mark.parametrize("n", range(2, 12))
def test_rotoreflection_z_against_general_rotoreflection(n):
    assert np.allclose(
        rotoreflection_z(maxl, n), generalized_rotoreflection(maxl, 0, 0, 2 * np.pi / n)
    )


@pytest.mark.parametrize("n", range(2, 12, 2))
def test_rotoreflection_z_from_operations(n):
    """According to https://en.wikipedia.org/wiki/Point_groups_in_three_dimensions a
    rotoreflection is equivalent to Cn followed by sigma h for even n"""
    assert np.allclose(
        rotoreflection_z(maxl, n), dot_product(n_z(maxl, n), sigma_xy(maxl))
    )


def test_Ci_from_operations():
    # Ci={E, i}
    assert np.isclose(
        condensed_wignerD_from_operations([identity(maxl), inversion(maxl)]), Ci(maxl)
    ).all()


# tests point groups from operations
@pytest.mark.parametrize("n", range(2, 13))
def test_Cn_from_operations(n):
    """Ezra p.188 or p.191
    Cn = {E, n_z, n_z^2, ..., n_z^(n-1)}
    direct product = matrix multiplication here?
    """
    operations = [identity(maxl), n_z(maxl, n)]
    for i in range(2, n):
        new_op = n_z(maxl, n)
        for _ in range(1, i):
            new_op = dot_product(new_op, n_z(maxl, n))
        operations.append(new_op)
    assert np.isclose(condensed_wignerD_from_operations(operations), Cn(maxl, n)).all()


# tests point groups from operations
@pytest.mark.parametrize("n", range(2, 13))
def test_Cn_from_operations2(n):
    """Ezra p.188 or p.191
    Cn = {E, n_z, n_z^2, ..., n_z^(n-1)}
    direct product = matrix multiplication here?
    """
    operations = [identity(maxl), n_z(maxl, n)]
    for i in range(2, n):
        new_op = generalized_rotation(maxl, 2 * np.pi / n, 0, 0)
        for _ in range(1, i):
            new_op = dot_product(new_op, n_z(maxl, n))
        operations.append(new_op)
    assert np.isclose(condensed_wignerD_from_operations(operations), Cn(maxl, n)).all()


# NEW TESTS
@pytest.mark.parametrize("n", range(2, 13))
def test_Cnv_from_product(n):
    assert np.allclose(Cnv(maxl, n), semidirect_product(Cn(maxl, n), Cs(maxl)))


@pytest.mark.parametrize("n", range(3, 13, 2))
def test_Cnh_from_product_odd(n):
    assert np.allclose(Cnh(maxl, n), semidirect_product(Cn(maxl, n), Ch(maxl)))


def test_base_polyhedral_rotations():
    max_l = maxl
    operations = [identity(max_l)]
    # add 3 C2
    operations.append(two_x(max_l))
    operations.append(two_y(max_l))
    operations.append(n_z(max_l, 2))
    # add 8 C3
    # 120 degrees around 0.5774, 0.5774, 0.5774
    operations.append(generalized_rotation(max_l, 0, np.pi / 2, np.pi / 2))
    # 120 degrees around -0.5774, -0.5774, 0.5774
    operations.append(generalized_rotation(max_l, 1, np.pi / 2, -np.pi / 2))
    # 120 degrees around 0.5774, -0.5774, -0.5774
    operations.append(generalized_rotation(max_l, -1, np.pi / 2, np.pi / 2))
    # 120 degrees around -0.5774, 0.5774, -0.5774
    operations.append(generalized_rotation(max_l, 0, np.pi / 2, -np.pi / 2))
    # 120 degrees around -0.5774, -0.5774, -0.5774 or 240 degrees around 0.5774, 0.5774, 0.5774
    operations.append(generalized_rotation(max_l, np.pi / 2, np.pi / 2, 1))
    # 120 degrees around 0.5774, 0.5774, -0.5774 or 240 degrees around -0.5774, -0.5774, 0.5774
    operations.append(generalized_rotation(max_l, -np.pi / 2, np.pi, 0))
    # 120 degrees around -0.5774, 0.5774, 0.5774 or 240 degrees around 0.5774, -0.5774, -0.5774
    operations.append(generalized_rotation(max_l, np.pi / 2, np.pi / 2, 0))
    # 120 degrees around 0.5774, -0.5774, 0.5774 or 240 degrees around -0.5774, 0.5774, -0.5774
    operations.append(generalized_rotation(max_l, -np.pi / 2, np.pi / 2, -1))
    # actual test
    # for rot, man in zip(base_rotations, operations):
    #    assert np.isclose(man, generalized_rotation(maxl, *rot)).all()
    # debuging
    import scipy.spatial

    for rot, man in zip(base_rotations, operations):
        if not np.isclose(man, generalized_rotation(maxl, *rot)).all():
            print(
                "False",
                scipy.spatial.transform.Rotation.from_euler("ZYZ", rot).as_rotvec()
                / np.pi,
                np.linalg.norm(
                    scipy.spatial.transform.Rotation.from_euler("YZY", rot).as_rotvec()
                )
                / np.pi,
                np.sum(np.isclose(man, generalized_rotation(maxl, *rot))),
                "/",
                len(man),
            )
            # for ii,i in enumerate(operations):
            #    print(ii, np.isclose(i, generalized_rotation(maxl, *rot)).all(),np.sum(np.isclose(i, generalized_rotation(maxl, *rot))),"/", len(i))
        else:
            print(
                np.isclose(man, generalized_rotation(maxl, *rot)).all(),
                scipy.spatial.transform.Rotation.from_euler("ZYZ", rot).as_rotvec()
                / np.pi,
                np.linalg.norm(
                    scipy.spatial.transform.Rotation.from_euler("YZY", rot).as_rotvec()
                )
                / np.pi,
                np.sum(np.isclose(man, generalized_rotation(maxl, *rot))),
                "/",
                len(man),
            )
    assert False


def test_D2_from_operations():
    # D2={E, 2_z, 2_y, 2_x}
    print(
        condensed_wignerD_from_operations(
            [identity(maxl), n_z(maxl, 2), two_y(maxl), two_x(maxl)]
        )
    )
    print(Dn(maxl, 2))
    print(
        np.testing.assert_allclose(
            condensed_wignerD_from_operations(
                [identity(maxl), n_z(maxl, 2), two_y(maxl), two_x(maxl)]
            ),
            Dn(maxl, 2),
        )
    )
    assert np.isclose(
        condensed_wignerD_from_operations(
            [identity(maxl), n_z(maxl, 2), two_y(maxl), two_x(maxl)]
        ),
        Dn(maxl, 2),
    ).all()


# test Dn against Michael's formula
@pytest.mark.parametrize("n", range(2, 12))
def test_Dn(n):
    dn_array = (
        np.array(
            [
                (delta(mprime, m) + delta(mprime, -m) * (-1) ** l) * delta(m % n, 0)
                for l, mprime, m in iter_sph_indices(maxl)
            ],
            dtype=complex,
        )
        / 2
    )
    assert np.allclose(Dn(maxl, n), dn_array)


# Semidirect product tests
@pytest.mark.parametrize("n", range(2, 12))
def test_Dn_from_product(n):
    """According to Altman Dn=Cn x| C2' Table 2 (p. 222)"""
    # C2'={E, 2_x}
    wignerd_c2prime = condensed_wignerD_from_operations([identity(maxl), two_x(maxl)])
    # Dn=Cn x| C2'
    assert np.isclose(
        semidirect_product(Cn(maxl, n), wignerd_c2prime), Dn(maxl, n)
    ).all()


@pytest.mark.parametrize("n", range(3, 13, 2))
def test_Dnh_from_product_odd_n(n):
    assert np.allclose(Dnh(maxl, n), semidirect_product(Dn(maxl, n), Ch(maxl)))


@pytest.mark.parametrize("n", range(3, 13, 2))
def test_Dnd_from_product_odd_n(n):
    assert np.allclose(Dnd(maxl, n), semidirect_product(Dn(maxl, n), Ci(maxl)))


@pytest.mark.parametrize("n", range(3, 12, 2))
def test_Cni_equivalence_to_Sn(n):
    """https://en.wikipedia.org/wiki/Schoenflies_notation#Point_groups"""
    cn_group = Cn(maxl, n)
    inversion_group = Ci(maxl)
    cni_group = semidirect_product(cn_group, inversion_group)
    assert np.allclose(cni_group, Sn(maxl, 2 * n))


# direct product tests
@pytest.mark.parametrize("n", range(2, 12, 4))
def test_rotoinversion_even_from_odd_direct_product(n):
    """Ezra p. 189:
    S2n = Cn x Ci for odd n"""
    cn_half = Cn(maxl, n // 2)
    assert np.allclose(Sn(maxl, n), semidirect_product(cn_half, Ci(maxl)))


def test_Th_from_product():
    assert np.allclose(
        semidirect_product(
            compute_condensed_wignerD_matrix_for_a_given_point_group("T", maxl),
            Ci(maxl),
        ),
        compute_condensed_wignerD_matrix_for_a_given_point_group("Th", maxl),
    )


def test_Oh_from_product():
    assert np.allclose(
        semidirect_product(
            compute_condensed_wignerD_matrix_for_a_given_point_group("O", maxl),
            Ci(maxl),
        ),
        compute_condensed_wignerD_matrix_for_a_given_point_group("Oh", maxl),
    )


def test_Ih_from_product():
    assert np.allclose(
        semidirect_product(
            compute_condensed_wignerD_matrix_for_a_given_point_group("I", maxl),
            Ci(maxl),
        ),
        compute_condensed_wignerD_matrix_for_a_given_point_group("Ih", maxl),
    )


file = h5py.File("data/data.h5", "r")
point_groups = file["data"]["matrices"].attrs["point_groups"]
matrices = file["data"]["matrices"]


@pytest.mark.parametrize("point_group, matrix", zip(point_groups, matrices))
def test_against_old_data(point_group, matrix):
    if point_group == "Ci":
        assert np.isclose(WignerD(point_group, 12).condensed_matrices / 2, matrix).all()
    else:
        assert np.isclose(
            WignerD(point_group, 12).condensed_matrices, matrix, atol=1e-2
        ).all()
