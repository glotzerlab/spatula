import h5py
import numpy as np
import pytest

import pgop
from pgop.wignerd import (
    Ci,
    Cn,
    Dn,
    Sn,
    WignerD,
    _parse_point_group,
    condensed_wignerD_from_operations,
    delta,
    direct_product,
    dot_product,
    generalized_rotation,
    identity,
    inversion,
    iter_sph_indices,
    n_z,
    rotoreflection,
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
def test_inversion():
    """According to Michaels paper (https://arxiv.org/pdf/2106.14846) :"""
    m_inv = np.array(
        [delta(mprime, m) * ((-1) ** l) for l, mprime, m in iter_sph_indices(maxl)],
        dtype=complex,
    )
    assert np.allclose(m_inv, inversion(maxl))


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


def test_identity():
    assert np.allclose(identity(maxl), generalized_rotation(maxl, 0, 0, 0))


def test_two_x():
    """According to paper by Michael (https://arxiv.org/pdf/2106.14846)"""
    assert np.allclose(two_x(maxl), dot_product(inversion(maxl), sigma_yz(maxl)))


def test_two_y():
    """According to paper by Michael (https://arxiv.org/pdf/2106.14846)"""
    assert np.allclose(two_y(maxl), dot_product(inversion(maxl), sigma_xz(maxl)))


def test_2x():
    """According to 7S. L. Altmann, “On the symmetries of spherical harmonics,”
    Mathematical Proceedings of the Cambridge Philosophical Society, page 347"""
    assert np.allclose(two_x(maxl), generalized_rotation(maxl, np.pi, np.pi, 0))


def test_2y():
    """According to 7S. L. Altmann, “On the symmetries of spherical harmonics,”
    Mathematical Proceedings of the Cambridge Philosophical Society, page 347"""
    assert np.allclose(two_y(maxl), generalized_rotation(maxl, 0, np.pi, 0))


def test_2z():
    """According to 7S. L. Altmann, “On the symmetries of spherical harmonics,”
    Mathematical Proceedings of the Cambridge Philosophical Society, page 347"""
    assert np.allclose(n_z(maxl, 2), generalized_rotation(maxl, np.pi, 0, 0))


def test_inversion_as_reflections():
    """According to paper by Michael (https://arxiv.org/pdf/2106.14846)"""
    assert np.allclose(
        inversion(maxl),
        dot_product(dot_product(sigma_yz(maxl), sigma_xz(maxl)), sigma_xy(maxl)),
    )


@pytest.mark.parametrize("n", range(2, 12, 2))
def test_rotoreflection(n):
    """According to https://en.wikipedia.org/wiki/Point_groups_in_three_dimensions a
    rotoreflection is equivalent to Cn followed by sigma h for even n"""
    assert np.allclose(
        rotoreflection(maxl, n), dot_product(n_z(maxl, n), sigma_xy(maxl))
    )


def test_Ci_from_operations():
    # Ci={E, i}
    assert np.isclose(
        condensed_wignerD_from_operations([identity(maxl), inversion(maxl)]), Ci(maxl)
    ).all()


def test_D2_from_operations():
    # D2={E, 2_z, 2_y, 2_x}
    assert np.isclose(
        condensed_wignerD_from_operations(
            [identity(maxl), n_z(maxl, 2), two_y(maxl), two_x(maxl)]
        ),
        Dn(maxl, 2),
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


@pytest.mark.parametrize("n", range(2, 12, 2))
def test_rotoinversion_even_from_operations(n):
    """https://en.wikipedia.org/wiki/Point_groups_in_three_dimensions"""
    operations = [identity(maxl)]
    cnz = n_z(maxl, n)
    for i in range(1, n, 2):
        new_operation = identity(maxl)
        for _ in range(1, i):
            new_operation = dot_product(new_operation, cnz)
        new_operation = dot_product(new_operation, sigma_xy(maxl))
        operations.append(new_operation)
    assert np.allclose(condensed_wignerD_from_operations(operations), Sn(maxl, n))


@pytest.mark.parametrize("n", range(2, 13))
def test_Cnh_from_operations(n):
    """https://en.wikipedia.org/wiki/Point_groups_in_three_dimensions"""
    operations = [identity(maxl), sigma_xy(maxl)]
    cnz = n_z(maxl, n)
    for i in range(1, n):
        new_operation = identity(maxl)
        for _ in range(1, i):
            new_operation = dot_product(new_operation, cnz)
        new_operation = dot_product(new_operation, sigma_xy(maxl))
        operations.append(new_operation)
    assert np.allclose(
        condensed_wignerD_from_operations(operations),
        WignerD("C" + str(n) + "h", maxl).condensed_matrices,
    )


# @pytest.mark.parametrize("n", range(3, 12, 2))
# def test_Dn_even_from_operations(n):
#    """Ezra p193.
#    D={E, n_z, n_z^2, ..., n_z^(n/2), 2_y, 2_y x n_z}
#    """
# # all is wrong here
#    operations = [identity(maxl), n_z(maxl, n)]
#    for i in range(2, (n - 1) // 2):
#        new_op = n_z(maxl, n)
#        for _ in range(1, i):
#            new_op = dot_product(new_op, n_z(maxl, n))
#        operations.append(new_op)
#    operations.append(n_z(maxl, 2))
#    operations.append(two_y(maxl))
#    operations.append(direct_product(two_y(maxl), n_z(maxl, n)))
#    assert np.isclose(
#        condensed_wignerD_from_operations(operations), Dn(maxl, n)
#    ).all()


@pytest.mark.parametrize("n", range(3, 12, 2))
def test_Dn_odd_from_operations(n):
    """Ezra p192.
    D={E, n_z, n_z^2, ..., n_z^((n-1)/2), 2_y}
    mult ={1,2,2,...2,n}"""
    operations = [identity(maxl), n_z(maxl, n)]
    for i in range(2, (n - 1) // 2 + 1):
        new_op = n_z(maxl, n)
        for _ in range(1, i):
            new_op = dot_product(new_op, n_z(maxl, n))
        operations.append(new_op)
    operations.append(two_y(maxl))
    assert np.isclose(condensed_wignerD_from_operations(operations), Dn(maxl, n)).all()


@pytest.mark.parametrize("n", range(3, 12))
def test_Cni_equivalence_to_Sn(n):
    """https://en.wikipedia.org/wiki/Schoenflies_notation#Point_groups"""
    assert np.allclose(
        WignerD("C" + str(n) + "i", maxl).condensed_matrices, Sn(maxl, n)
    )


# direct product tests
@pytest.mark.parametrize("n", range(2, 12, 4))
def test_rotoinversion_even_from_odd_direct_product(n):
    """Ezra p. 189:
    S2n = Cn x Ci for odd n"""
    cn_half = Cn(maxl, n // 2)
    assert np.allclose(Sn(maxl, n), direct_product(cn_half, Ci(maxl)))


def test_D2_direct_product():
    # C2'={E, 2_y}
    wignerd_c2prime = condensed_wignerD_from_operations([identity(maxl), two_y(maxl)])
    c2prime = WignerD.from_condensed_matrix_maxl_point_group(
        wignerd_c2prime, maxl, "C2'"
    )
    # D2=C2 x C2'
    assert np.isclose(
        direct_product(WignerD("C2", maxl).matrices, c2prime.matrices), Dn(maxl, 2)
    ).all()


@pytest.mark.parametrize("n", [x * 2 - 1 for x in range(1, 6)])
def test_Dnh_odd_n_direct_product(n):
    """Ezra, p. 189"""
    assert np.isclose(
        WignerD("D" + str(n) + "h", maxl).condensed_matrices,
        direct_product(
            WignerD("C" + str(n) + "v", maxl).matrices, WignerD("Ch", maxl).matrices
        ),
    ).all()


# Semidirect product tests
@pytest.mark.parametrize("n", range(2, 12))
def test_Dn_semidirect_product(n):
    """According to Altman Dn=Cn x| C2' Table 2 (p. 222)"""
    # C2'={E, 2_y}
    wignerd_c2prime = condensed_wignerD_from_operations([identity(maxl), two_y(maxl)])
    # Dn=Cn x| C2'
    assert np.isclose(
        semidirect_product(Cn(maxl, n), wignerd_c2prime), Dn(maxl, n)
    ).all()


file = h5py.File("data/data.h5", "r")
point_groups = file["data"]["matrices"].attrs["point_groups"]
matrices = file["data"]["matrices"]


@pytest.mark.parametrize("point_group, matrix", zip(point_groups, matrices))
def test_against_old_data(point_group, matrix):
    # TODO remove when OI gets implemented
    if point_group in "OI":
        return
    if point_group == "Ci":
        assert np.isclose(WignerD(point_group, 12).condensed_matrices / 2, matrix).all()
    else:
        assert np.isclose(WignerD(point_group, 12).condensed_matrices, matrix).all()
