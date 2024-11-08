import numpy as np
import pytest
import scipy.spatial

import pgop
from pgop.wignerd import (
    WignerD,
    _parse_point_group,
    compute_condensed_wignerD_matrix_for_a_given_point_group,
    condensed_wignerD_from_operations,
    convert_hermann_mauguin_to_schonflies,
    delta,
    dot_product,
    identity,
    inversion,
    iter_sph_indices,
    reflection_from_normal,
    rotation_from_axis_angle,
    rotation_from_axis_order,
    rotation_from_euler_angles,
    rotoreflection_from_axis_angle,
    rotoreflection_from_axis_order,
    rotoreflection_from_euler_angles,
)


def semidirect_product(D_a: np.ndarray, D_b: np.ndarray) -> np.ndarray:  # noqa N802
    """Compute the semidirect product of two WignerD matrices.

    Note: This version doesn't take into account that duplicate operations produced by
    the procedures shouldn't be double counted!

    Parameters
    ----------
    D_a : np.ndarray
        The first WignerD matrix.
    D_b : np.ndarray
        The second WignerD matrix.

    Returns
    -------
    np.ndarray
        The semidirect product of the two WignerD matrices.
    """

    def collapse_to_zero(num, tol=1e-7):
        """Collapse a number to zero if it is less than a tolerance.

        Parameters
        ----------
        num : float
            The number to collapse to zero.
        tol : float
            The tolerance to collapse to zero.

        Returns
        -------
        float
            The number or zero if it is less than the tolerance.
        """
        if np.abs(num) < tol:
            return 0
        return num

    u_D_a = np.asarray(D_a).flatten()  # noqa N806
    u_D_b = np.asarray(D_b).flatten()  # noqa N806

    max_l = 0
    cnt = 0
    while cnt < u_D_a.size:
        max_l += 1
        cnt += (2 * max_l + 1) * (2 * max_l + 1)

    l_skip = 0
    D_ab = np.zeros_like(u_D_a, dtype=np.complex128)  # noqa N806

    for l in range(max_l):
        max_m = 2 * l + 1
        for m_prime in range(max_m):
            start_lmprime_i = l_skip + m_prime * max_m
            for m in range(max_m):
                sum_val = 0 + 0j
                for m_prime_2 in range(max_m):
                    sum_val += (
                        u_D_a[start_lmprime_i + m_prime_2]
                        * u_D_b[l_skip + m_prime_2 * max_m + m]
                    )
                D_ab[start_lmprime_i + m] = collapse_to_zero(sum_val, 1e-7)
        l_skip += max_m * max_m

    return D_ab.reshape(D_a.shape)


order_range_to_test = range(2, 13)
order_range_to_test_odd = range(3, 13, 2)


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
    with pytest.raises(ValueError):
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


def test_identity():
    assert np.allclose(identity(maxl), rotation_from_euler_angles(maxl, 0, 0, 0))


def test_C2x_from_operations():
    """According to paper by Engel (https://arxiv.org/pdf/2106.14846)
    Note: Engel's paper has values for 2x and 2y swapped!
    """
    assert np.allclose(
        np.array(
            [
                delta(mprime, -m) * ((-1) ** (l + m))
                for l, mprime, m in iter_sph_indices(maxl)
            ],
            dtype=complex,
        ),
        dot_product(inversion(maxl), reflection_from_normal(maxl, [1, 0, 0])),
    )


def test_C2y_from_operations():
    """According to paper by Engel (https://arxiv.org/pdf/2106.14846)
    Note: Engel's paper has values for 2x and 2y swapped!
    """
    assert np.allclose(
        np.array(
            [
                delta(mprime, -m) * ((-1) ** (l))
                for l, mprime, m in iter_sph_indices(maxl)
            ],
            dtype=complex,
        ),
        dot_product(inversion(maxl), reflection_from_normal(maxl, [0, 1, 0])),
    )


def test_C2y_rotation_from_euler_angles():
    """According to paper by Engel (https://arxiv.org/pdf/2106.14846)
    Note: Engel's paper has values for 2x and 2y swapped!
    """
    assert np.allclose(
        np.array(
            [
                delta(mprime, -m) * ((-1) ** (l))
                for l, mprime, m in iter_sph_indices(maxl)
            ],
            dtype=complex,
        ),
        rotation_from_axis_order(maxl, [0, 1, 0], 2),
    )


def test_C2x_rotation_from_euler_angles():
    """According to paper by Engel (https://arxiv.org/pdf/2106.14846)
    Note: Engel's paper has values for 2x and 2y swapped!
    """
    assert np.allclose(
        np.array(
            [
                delta(mprime, -m) * ((-1) ** (l + m))
                for l, mprime, m in iter_sph_indices(maxl)
            ],
            dtype=complex,
        ),
        rotation_from_axis_order(maxl, [1, 0, 0], 2),
    )


@pytest.mark.parametrize("n", order_range_to_test)
def test_Cz_rotation_from_euler_angles(n):
    """According to Engel's paper (https://arxiv.org/pdf/2106.14846).
    Note: Engel's paper has a minus in the exponent which is not present in altmann.
    """
    assert np.allclose(
        np.array(
            [
                delta(mprime, m) * np.exp(2 * np.pi * m * 1j / n)
                for _, mprime, m in iter_sph_indices(maxl)
            ],
            dtype=complex,
        ),
        rotation_from_axis_order(maxl, [0, 0, 1], n),
    )


@pytest.mark.parametrize("n", order_range_to_test)
def test_generalized_rotation_self_correct(n):
    assert np.allclose(
        rotation_from_euler_angles(maxl, 2 * np.pi / n, 0, 0),
        rotation_from_euler_angles(maxl, 0, 0, 2 * np.pi / n),
    )


@pytest.mark.parametrize("n", order_range_to_test)
def test_generalized_rotation_against_axis_angle(n):
    assert np.allclose(
        rotation_from_euler_angles(maxl, 2 * np.pi / n, 0, 0),
        rotation_from_axis_angle(maxl, [0, 0, 1], 2 * np.pi / n),
    )


@pytest.mark.parametrize("n", order_range_to_test)
def test_generalized_rotation_against_axis_order(n):
    assert np.allclose(
        rotation_from_euler_angles(maxl, 2 * np.pi / n, 0, 0),
        rotation_from_axis_order(maxl, [0, 0, 1], n),
    )


def test_inversion():
    """According to Engel's paper (https://arxiv.org/pdf/2106.14846)"""
    m_inv = np.array(
        [delta(mprime, m) * ((-1) ** l) for l, mprime, m in iter_sph_indices(maxl)],
        dtype=complex,
    )
    assert np.allclose(m_inv, inversion(maxl))


def test_inversion_as_reflections():
    """According to Engel's paper (https://arxiv.org/pdf/2106.14846) :"""
    assert np.allclose(
        inversion(maxl),
        dot_product(
            dot_product(
                reflection_from_normal(maxl, [1, 0, 0]),
                reflection_from_normal(maxl, [0, 1, 0]),
            ),
            reflection_from_normal(maxl, [0, 0, 1]),
        ),
    )


def test_sigmaxy():
    """According to Engel's paper (https://arxiv.org/pdf/2106.14846)"""
    assert np.allclose(
        reflection_from_normal(maxl, [0, 0, 1]),
        np.array(
            [
                delta(mprime, m) * ((-1) ** (m + l))
                for l, mprime, m in iter_sph_indices(maxl)
            ],
            dtype=complex,
        ),
    )


def test_sigmaxz():
    """According to Engel's paper (https://arxiv.org/pdf/2106.14846)
    Note: Engel's paper has values for sigma_xz and sigma_yz swapped!"""
    assert np.allclose(
        reflection_from_normal(maxl, [0, 1, 0]),
        np.array(
            [delta(mprime, -m) for _, mprime, m in iter_sph_indices(maxl)],
            dtype=complex,
        ),
    )


def test_sigmayz():
    """According to Engel's paper (https://arxiv.org/pdf/2106.14846).
    Note: Engel's paper has values for sigma_xz and sigma_yz swapped!"""
    assert np.allclose(
        reflection_from_normal(maxl, [1, 0, 0]),
        np.array(
            [
                delta(mprime, -m) * ((-1) ** m)
                for _, mprime, m in iter_sph_indices(maxl)
            ],
            dtype=complex,
        ),
    )


@pytest.mark.parametrize("n", order_range_to_test)
def test_general_rotoreflection_self_correct(n):
    assert np.allclose(
        rotoreflection_from_euler_angles(maxl, 2 * np.pi / n, 0, 0),
        rotoreflection_from_euler_angles(maxl, 0, 0, 2 * np.pi / n),
    )


@pytest.mark.parametrize("n", order_range_to_test)
def test_general_rotoreflection_against_from_axis_angle(n):
    assert np.allclose(
        rotoreflection_from_euler_angles(maxl, 2 * np.pi / n, 0, 0),
        rotoreflection_from_axis_angle(maxl, [0, 0, 1], 2 * np.pi / n),
    )


@pytest.mark.parametrize("n", order_range_to_test)
def test_general_rotoreflection_against_axis_order(n):
    assert np.allclose(
        rotoreflection_from_euler_angles(maxl, 2 * np.pi / n, 0, 0),
        rotoreflection_from_axis_order(maxl, [0, 0, 1], n),
    )


def test_Ci():
    """According to Engel's paper (https://arxiv.org/pdf/2106.14846) :"""
    assert np.isclose(
        np.array(
            [
                delta(mprime, m) * delta(l % 2, 0)
                for l, mprime, m in iter_sph_indices(maxl)
            ],
            dtype=complex,
        ),
        compute_condensed_wignerD_matrix_for_a_given_point_group("Ci", maxl),
    ).all()


@pytest.mark.parametrize("n", order_range_to_test)
def test_Cn(n):
    """According to Engel's paper (https://arxiv.org/pdf/2106.14846) :"""
    w = np.array(
        [delta(mprime, m) * delta(m % n, 0) for _, mprime, m in iter_sph_indices(maxl)],
        dtype=complex,
    )
    assert np.isclose(
        w, compute_condensed_wignerD_matrix_for_a_given_point_group("C" + str(n), maxl)
    ).all()


@pytest.mark.parametrize("n", order_range_to_test)
def test_Cn_against_scipy_rotations_euler(n):
    operations = []
    for i in scipy.spatial.transform.Rotation.create_group("C" + str(n)):
        operations.append(rotation_from_euler_angles(maxl, *i.as_euler("zyz")))
    assert np.allclose(
        compute_condensed_wignerD_matrix_for_a_given_point_group("C" + str(n), maxl),
        condensed_wignerD_from_operations(operations),
    )


@pytest.mark.parametrize("n", order_range_to_test)
def test_Cn_against_scipy_rotations_rotvec(n):
    operations = []
    for i in scipy.spatial.transform.Rotation.create_group("C" + str(n)):
        # identity has no axis, so we have to check for it otherwise errors
        if np.isclose(i.as_rotvec(), [0, 0, 0]).all():
            operations.append(identity(maxl))
        else:
            axis = i.as_rotvec()
            angle = np.linalg.norm(axis)
            axis = axis / angle
            operations.append(rotation_from_axis_angle(maxl, axis, angle))
    assert np.allclose(
        compute_condensed_wignerD_matrix_for_a_given_point_group("C" + str(n), maxl),
        condensed_wignerD_from_operations(operations),
    )


@pytest.mark.parametrize("n", order_range_to_test)
def test_Dn_against_scipy_rotations_euler(n):
    operations = []
    for i in scipy.spatial.transform.Rotation.create_group("D" + str(n)):
        euler_angles = np.asarray(i.as_euler("zyz"))
        # Fix for scipy starting from C2' that aligns with x axis, while I start with y.
        if n % 2 == 1 and np.isclose(euler_angles[1], np.pi):
            euler_angles[0] = euler_angles[0] + np.pi
        operations.append(rotation_from_euler_angles(maxl, *euler_angles))
    assert np.allclose(
        compute_condensed_wignerD_matrix_for_a_given_point_group("D" + str(n), maxl),
        condensed_wignerD_from_operations(operations),
    )


@pytest.mark.parametrize("n", order_range_to_test)
def test_Dn_against_scipy_rotations_rotvec(n):
    operations = []
    for i in scipy.spatial.transform.Rotation.create_group("D" + str(n)):
        # identity has no axis, so we have to check for it otherwise errors
        if np.isclose(i.as_rotvec(), [0, 0, 0]).all():
            operations.append(identity(maxl))
        else:
            axis = i.as_rotvec()
            angle = np.linalg.norm(axis)
            axis = axis / angle
            # Fix for scipy starting from C2' that aligns with x axis, I start with y.
            if n % 2 == 1:
                axis = (
                    scipy.spatial.transform.Rotation.from_euler(
                        "zyz", [np.pi / 2, 0, 0]
                    ).as_matrix()
                    @ axis
                )
            operations.append(rotation_from_axis_angle(maxl, axis, angle))
    assert np.allclose(
        compute_condensed_wignerD_matrix_for_a_given_point_group("D" + str(n), maxl),
        condensed_wignerD_from_operations(operations),
    )


@pytest.mark.parametrize("n", order_range_to_test)
def test_Dn(n):
    """According to Engel's paper (https://arxiv.org/pdf/2106.14846)"""
    assert np.allclose(
        np.array(
            [
                (delta(mprime, m) + delta(mprime, -m) * (-1) ** l) * delta(m % n, 0)
                for l, mprime, m in iter_sph_indices(maxl)
            ],
            dtype=complex,
        )
        / 2,
        compute_condensed_wignerD_matrix_for_a_given_point_group("D" + str(n), maxl),
    )


@pytest.mark.parametrize("n", order_range_to_test)
def test_Cnv_from_product(n):
    """Ezra p. 189:
    Cnv = Cn x Cs for odd n"""
    assert np.allclose(
        compute_condensed_wignerD_matrix_for_a_given_point_group(
            "C" + str(n) + "v", maxl
        ),
        semidirect_product(
            compute_condensed_wignerD_matrix_for_a_given_point_group(
                "C" + str(n), maxl
            ),
            compute_condensed_wignerD_matrix_for_a_given_point_group("Cs", maxl),
        ),
    )


@pytest.mark.parametrize("n", order_range_to_test_odd)
def test_Cnh_from_product(n):
    """Ezra p. 189:
    Cnv = Cn x Cs for all n.
    We cannot use this for even n's here because the
    semiproduct implementation does not take into account duplicate elements which makes
    the method fail if the produced group has duplicate elements, which shouldn't be
    counted for group action (group has no notion of duplicate elements)."""
    assert np.allclose(
        compute_condensed_wignerD_matrix_for_a_given_point_group(
            "C" + str(n) + "h", maxl
        ),
        semidirect_product(
            compute_condensed_wignerD_matrix_for_a_given_point_group(
                "C" + str(n), maxl
            ),
            compute_condensed_wignerD_matrix_for_a_given_point_group("Ch", maxl),
        ),
    )


@pytest.mark.parametrize("n", order_range_to_test_odd)
def test_Sn_odd_equivalent_to_Cnh(n):
    """https://en.wikipedia.org/wiki/Schoenflies_notation#Point_groups"""
    assert np.allclose(
        compute_condensed_wignerD_matrix_for_a_given_point_group("S" + str(n), maxl),
        compute_condensed_wignerD_matrix_for_a_given_point_group(
            "C" + str(n) + "h", maxl
        ),
    )


@pytest.mark.parametrize("n", order_range_to_test)
def test_Dn_from_product(n):
    """According to Altman Dn=Cn x| C2' Table 2 (p. 222)
    It can be either C2' = {E, 2_x} or C2''= {E, 2_y}. The group action matrix will be
    slightly different. But for PgOP this doesn't matter because we have to optimize
    the rotation/orientation of the BOOD anyways. There is just a free parameter which
    is the starting orientation of the C2's orthogonal to the principal axis.
    to match this implementation we use C2'={E, 2_y}"""
    c2prime = condensed_wignerD_from_operations(
        [identity(maxl), rotation_from_axis_order(maxl, [0, 1, 0], 2)]
    )
    dn_from_product = semidirect_product(
        compute_condensed_wignerD_matrix_for_a_given_point_group("C" + str(n), maxl),
        c2prime,
    )
    assert np.isclose(
        dn_from_product,
        compute_condensed_wignerD_matrix_for_a_given_point_group("D" + str(n), maxl),
    ).all()


@pytest.mark.parametrize("n", order_range_to_test_odd)
def test_Dnd_from_product_odd_n(n):
    """Ezra p. 189:
    Dnd = Dn x Ci for odd n"""
    assert np.allclose(
        compute_condensed_wignerD_matrix_for_a_given_point_group(
            "D" + str(n) + "d", maxl
        ),
        semidirect_product(
            compute_condensed_wignerD_matrix_for_a_given_point_group(
                "D" + str(n), maxl
            ),
            compute_condensed_wignerD_matrix_for_a_given_point_group("Ci", maxl),
        ),
    )


@pytest.mark.parametrize("n", order_range_to_test_odd)
def test_Cni_equivalence_to_Sn(n):
    """https://en.wikipedia.org/wiki/Schoenflies_notation#Point_groups"""
    cn_group = compute_condensed_wignerD_matrix_for_a_given_point_group(
        "C" + str(n), maxl
    )
    inversion_group = compute_condensed_wignerD_matrix_for_a_given_point_group(
        "Ci", maxl
    )
    cni_group = semidirect_product(cn_group, inversion_group)
    assert np.allclose(
        cni_group,
        compute_condensed_wignerD_matrix_for_a_given_point_group(
            "S" + str(2 * n), maxl
        ),
    )


@pytest.mark.parametrize("n", range(2, 12, 4))
def test_Sn_even_from_odd_direct_product(n):
    """Ezra p. 189:
    S2n = Cn x Ci for odd n"""
    cn_half = compute_condensed_wignerD_matrix_for_a_given_point_group(
        "C" + str(n // 2), maxl
    )
    assert np.allclose(
        compute_condensed_wignerD_matrix_for_a_given_point_group("S" + str(n), maxl),
        semidirect_product(
            cn_half,
            compute_condensed_wignerD_matrix_for_a_given_point_group("Ci", maxl),
        ),
    )


def test_Th_from_product():
    assert np.allclose(
        semidirect_product(
            compute_condensed_wignerD_matrix_for_a_given_point_group("T", maxl),
            compute_condensed_wignerD_matrix_for_a_given_point_group("Ci", maxl),
        ),
        compute_condensed_wignerD_matrix_for_a_given_point_group("Th", maxl),
    )


def test_Oh_from_product():
    assert np.allclose(
        semidirect_product(
            compute_condensed_wignerD_matrix_for_a_given_point_group("O", maxl),
            compute_condensed_wignerD_matrix_for_a_given_point_group("Ci", maxl),
        ),
        compute_condensed_wignerD_matrix_for_a_given_point_group("Oh", maxl),
    )


def test_Ih_from_product():
    assert np.allclose(
        semidirect_product(
            compute_condensed_wignerD_matrix_for_a_given_point_group("I", maxl),
            compute_condensed_wignerD_matrix_for_a_given_point_group("Ci", maxl),
        ),
        compute_condensed_wignerD_matrix_for_a_given_point_group("Ih", maxl),
    )


point_group_mapping = {
    "1": "C1",
    "-1": "Ci",
    "2": "C2",
    "m": "Cs",
    "2/m": "C2h",
    "222": "D2",
    "mm2": "C2v",
    "mmm": "D2h",
    "2/m2/m2/m": "D2h",
    "4": "C4",
    "-4": "S4",
    "4/m": "C4h",
    "422": "D4",
    "4mm": "C4v",
    "-42m": "D2d",
    "4/mmm": "D4h",
    "4/m2/m2/m": "D4h",
    "3": "C3",
    "-3": "S6",
    "32": "D3",
    "3m": "C3v",
    "-3m": "D3d",
    "-32/m": "D3d",
    "6": "C6",
    "-6": "C3h",
    "6/m": "C6h",
    "622": "D6",
    "6mm": "C6v",
    "-6m2": "D3h",
    "-62m": "D3h",
    "6/mmm": "D6h",
    "6/m2/m2/m": "D6h",
    "23": "T",
    "m-3": "Th",
    "2/m-3": "Th",
    "432": "O",
    "-43m": "Td",
    "m-3m": "Oh",
    "4/m-32/m": "Oh",
}


# use point group mapping to test all point groups
@pytest.mark.parametrize("point_group", point_group_mapping.keys())
def test_notations(point_group):
    assert (
        convert_hermann_mauguin_to_schonflies(point_group)
        == point_group_mapping[point_group]
    )
