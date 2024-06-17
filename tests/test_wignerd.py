import h5py
import numpy as np
import pytest

import pgop
from pgop.wignerd import (
    Ci,
    Cn,
    Dn,
    WignerD,
    _parse_point_group,
    condensed_wignerD_from_operations,
    direct_product,
    identity,
    inversion,
    n_z,
    semidirect_product,
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
            new_op = direct_product(new_op, n_z(maxl, n))
        operations.append(new_op)
    assert np.isclose(condensed_wignerD_from_operations(operations), Cn(maxl, n)).all()



@pytest.mark.parametrize("n", range(2, 12))
def test_Dn_semidirect_product(n):
    """According to Altman Dn=Cn x| C2' Table 2 (p. 222)"""
    # C2'={E, 2_y}
    wignerd_c2prime = condensed_wignerD_from_operations([identity(maxl), two_y(maxl)])
    # Dn=Cn x| C2'
    assert np.isclose(
        semidirect_product(Cn(maxl, n), wignerd_c2prime), Dn(maxl, n)
    ).all()


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
    assert np.isclose(
        WignerD("D" + str(n) + "h", maxl).condensed_matrices,
        direct_product(
            WignerD("C" + str(n) + "v", maxl).matrices, WignerD("Ch", maxl).matrices
        ),
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
