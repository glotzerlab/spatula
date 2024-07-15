import itertools
from typing import Generator

import numpy as np
import scipy.special

golden_mean = (1 + np.sqrt(5)) / 2

base_rotations = (
    np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [1.0, 0.5, -0.5],
            [1.0, 0.5, 0.5],
            [0.0, 0.5, -0.5],
            [0.5, 0.5, 1.0],
            [-0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 1.0],
        ]
    )
    * np.pi
)

octahedral_rotations = (
    np.array(
        [
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, 0.5],
            [0.5, 1.0, 0.0],
            [-0.5, 1.0, 0.0],
            [1.0, 0.5, 0.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 0.0],
            [1.0, 0.5, 1.0],
            [0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, -0.5],
        ]
    )
    * np.pi
)

icosahedral_rotations = (
    np.array(
        [  # note to self np.arctan2(1/2,-golden_mean/2)/np.pi = 0.8237918088252166
            # note to self np.arctan2(1/2,golden_mean/2)/np.pi = 0.17620819117478337
            [0.82379181, 0.4, 0.17620819],
            [0.17620819, 0.4, 0.82379181],
            [-0.82379181, 0.4, -0.17620819],
            [-0.17620819, 0.4, -0.82379181],
            [0.82379181, 0.2, -0.17620819],
            [-0.82379181, 0.2, 0.17620819],
            [-0.17620819, 0.2, 0.82379181],
            [0.17620819, 0.2, -0.82379181],
            [1 - 1 / golden_mean, 2 / 3, 1 / golden_mean - 1],
            [-1 / golden_mean, 2 / 3, 1 / golden_mean],
            [1 / golden_mean - 1, 2 / 3, 1 - 1 / golden_mean],
            [1 / golden_mean, 2 / 3, -1 / golden_mean],
            [1 - 1 / golden_mean, 1 / 3, 1 - 1 / golden_mean],
            [1 / golden_mean, 1 / 3, 1 / golden_mean],
            [-1 / golden_mean, 1 / 3, -1 / golden_mean],
            [1 / golden_mean - 1, 1 / 3, 1 / golden_mean - 1],
            [0.82379181, 0.6, -0.17620819],
            [-0.82379181, 0.6, 0.17620819],
            [-0.17620819, 0.6, 0.82379181],
            [0.17620819, 0.6, -0.82379181],
            [0.17620819, 0.4, -0.17620819],
            [-0.82379181, 0.4, 0.82379181],
            [-0.17620819, 0.4, 0.17620819],
            [0.82379181, 0.4, -0.82379181],
            [0.17620819, 0.2, 0.17620819],
            [0.82379181, 0.2, 0.82379181],
            [-0.82379181, 0.2, -0.82379181],
            [-0.17620819, 0.2, -0.17620819],
            [0.17620819, 0.6, 0.17620819],
            [0.82379181, 0.6, 0.82379181],
            [-0.82379181, 0.6, -0.82379181],
            [-0.17620819, 0.6, -0.17620819],
            [0.17620819, 0.8, -0.17620819],
            [-0.82379181, 0.8, 0.82379181],
            [-0.17620819, 0.8, 0.17620819],
            [0.82379181, 0.8, -0.82379181],
            [1 / golden_mean, 1 / 3, 1 / golden_mean - 1],
            [-1 / golden_mean, 1 / 3, 1 - 1 / golden_mean],
            [1 / golden_mean - 1, 1 / 3, 1 / golden_mean],
            [1 - 1 / golden_mean, 1 / 3, -1 / golden_mean],
            [1 / golden_mean, 2 / 3, 1 - 1 / golden_mean],
            [1 - 1 / golden_mean, 2 / 3, 1 / golden_mean],
            [-1 / golden_mean, 2 / 3, 1 / golden_mean - 1],
            [1 / golden_mean - 1, 2 / 3, -1 / golden_mean],
            [0.82379181, 0.8, 0.17620819],
            [0.17620819, 0.8, 0.82379181],
            [-0.82379181, 0.8, -0.17620819],
            [-0.17620819, 0.8, -0.82379181],
        ]
    )
    * np.pi
)


def inversion(max_l: int) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for inversion up to the given l.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.

    Returns
    -------
    np.ndarray
        The WignerD matrix for inversion up to the given l.
    """
    return np.concatenate(
        [
            np.eye(2 * l + 1, 2 * l + 1, dtype=complex).flatten() * (-1) ** l
            for l in range(0, max_l + 1)
        ],
        dtype=complex,
    )


def rotoreflection(max_l: int, n: int) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for rotoreflection up to the given l.

    Implementation according to Altman, Mathematical Proceedings of the Cambridge
    Philosophical Society , Volume 53 , Issue 2 , April 1957
    (https://doi.org/10.1017/S0305004100032370), p.347 (bottom of the page).

    Parameters
    ----------
    max_l : int
        The maximum l value to include.
    n : int
        The order of the rotation group.

    Returns
    -------
    np.ndarray
        The WignerD matrix for rotoreflection up to the given l.
    """
    reflection_matrices = sigma_xy(max_l)
    rotation_matrices = n_z(max_l, n)
    return dot_product(reflection_matrices, rotation_matrices)


def generalized_rotation(
    max_l: int, alpha: float, beta: float, gamma: float
) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for a generalized rotation up to the given l.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.
    alpha : float
        The angle of rotation around the z-axis. Must be between 0 and 2*pi.
    beta : float
        The angle of rotation around the y-axis. Must be between 0 and pi.
    gamma : float
        The angle of rotation around the z-axis. Must be between 0 and 2*pi.

    Returns
    -------
    np.ndarray
        The WignerD matrix for the generalized rotation up to the given l.
    """

    def S_sum(l, m, mprime):  # noqa: N802
        return np.sum(
            [
                (
                    ((-1) ** k)
                    / (
                        scipy.special.factorial(l - mprime - k)
                        * scipy.special.factorial(l + m - k)
                        * scipy.special.factorial(k)
                        * scipy.special.factorial(k - m + mprime)
                    )
                )
                * np.cos(beta * 0.5) ** (2 * l + m - mprime - 2 * k)
                * np.sin(beta * 0.5) ** (-m + mprime + 2 * k)
                for k in range(max(0, m - mprime), min(l - mprime, l + m) + 1)
            ],
            dtype=complex,
        )

    def S_sum_when_beta_half_pi(l, m, mprime):  # noqa: N802
        return (2 ** (-l)) * np.sum(
            [
                (
                    ((-1) ** k)
                    / (
                        scipy.special.factorial(l - mprime - k)
                        * scipy.special.factorial(l + m - k)
                        * scipy.special.factorial(k)
                        * scipy.special.factorial(k - m + mprime)
                    )
                )
                for k in range(max(0, m - mprime), min(l - mprime, l + m) + 1)
            ],
            dtype=complex,
        )

    def Cmprimem(m, mprime, l):  # noqa: N802
        return (1j ** (np.abs(mprime) + mprime)) * 1j ** (np.abs(m) + m)

    def root_factorial(m, mprime, l):
        return np.sqrt(
            scipy.special.factorial(l + m)
            * scipy.special.factorial(l - m)
            * scipy.special.factorial(l + mprime)
            * scipy.special.factorial(l - mprime)
        )

    if np.isclose(beta, 0):
        return np.array(
            [
                np.exp(1j * m * alpha) * np.exp(1j * m * gamma) * delta(mprime, m)
                for _, mprime, m in iter_sph_indices(max_l)
            ],
            dtype=complex,
        )
    elif np.isclose(beta, np.pi / 2):
        return np.array(
            [
                S_sum_when_beta_half_pi(l, m, mprime)
                * Cmprimem(m, mprime, l)
                * root_factorial(m, mprime, l)
                * np.exp(1j * m * alpha)
                * np.exp(1j * mprime * gamma)
                for l, mprime, m in iter_sph_indices(max_l)
            ],
            dtype=complex,
        )
    elif np.isclose(beta, np.pi):
        return np.array(
            [
                ((-1) ** l)
                * np.exp(1j * m * alpha)
                * np.exp(-1j * m * gamma)
                * delta(mprime, -m)
                for l, mprime, m in iter_sph_indices(max_l)
            ],
            dtype=complex,
        )
    else:
        return np.array(
            [
                Cmprimem(m, mprime, l)
                * root_factorial(m, mprime, l)
                * S_sum(l, m, mprime)
                * np.exp(1j * m * alpha)
                * np.exp(1j * mprime * gamma)
                for l, mprime, m in iter_sph_indices(max_l)
            ],
            dtype=complex,
        )


def sigma_yz(max_l: int) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for sigma_xy up to the given l.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.

    Returns
    -------
    np.ndarray
        The WignerD matrix for sigma_xy up to the given l.
    """
    return np.array(
        [delta(mprime, -m) * ((-1) ** m) for _, mprime, m in iter_sph_indices(max_l)],
        dtype=complex,
    )


def sigma_xz(max_l: int) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for sigma_xz up to the given l.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.

    Returns
    -------
    np.ndarray
        The WignerD matrix for sigma_xz up to the given l.
    """
    return np.array(
        [delta(mprime, -m) for _, mprime, m in iter_sph_indices(max_l)],
        dtype=complex,
    )


def sigma_xy(max_l: int) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for sigma_xy up to the given l.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.

    Returns
    -------
    np.ndarray
        The WignerD matrix for sigma_xy up to the given l.
    """
    return np.array(
        [
            delta(mprime, m) * ((-1) ** (m + l))
            for l, mprime, m in iter_sph_indices(max_l)
        ],
        dtype=complex,
    )


def two_x(max_l: int) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for rotation for 180 around x up to the given l.

    Note: Michaels paper has values for 2x and 2y swapped!

    Parameters
    ----------
    max_l : int
        The maximum l value to include.

    Returns
    -------
    np.ndarray
        The WignerD matrix for rotation for 180 around x up to the given l.
    """
    return np.array(
        [
            delta(mprime, -m) * ((-1) ** (l + m))
            for l, mprime, m in iter_sph_indices(max_l)
        ],
        dtype=complex,
    )


def two_y(max_l: int) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for rotation for 180 around y up to the given l.

    Note: Michaels paper has values for 2x and 2y swapped!

    Parameters
    ----------
    max_l : int
        The maximum l value to include.

    Returns
    -------
    np.ndarray
        The WignerD matrix for rotation for 180 around y up to the given l.
    """
    return np.array(
        [delta(mprime, -m) * ((-1) ** (l)) for l, mprime, m in iter_sph_indices(max_l)],
        dtype=complex,
    )


def n_z(max_l: int, n: int) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for n-fold rotation around z up to the given l.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.
    n : int
        The order of the rotation.
    Returns
    -------
    np.ndarray
        The WignerD matrix for n-fold rotation around z up to the given l.
    """
    return np.array(
        [
            delta(mprime, m) * np.exp(-2 * np.pi * m * 1j / n)
            for _, mprime, m in iter_sph_indices(max_l)
        ],
        dtype=complex,
    )


def identity(max_l: int) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for E (identity) up to the given l.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.

    Returns
    -------
    np.ndarray
        The WignerD matrix for E up to the given l.
    """
    return np.array(
        [delta(mprime, m) for _, mprime, m in iter_sph_indices(max_l)],
        dtype=complex,
    )


def Cs(max_l: int) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for Cs group up to the given l.

    Cs={E, sigma_yz}

    Parameters
    ----------
    max_l : int
        The maximum l value to include.

    Returns
    -------
    np.ndarray
        The WignerD matrix for Cs up to the given l.
    """
    return condensed_wignerD_from_operations([identity(max_l), sigma_yz(max_l)])


def Ch(max_l: int) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for Ch group up to the given l.

    Ch={E, sigma_xy}

    Parameters
    ----------
    max_l : int
        The maximum l value to include.

    Returns
    -------
    np.ndarray
        The WignerD matrix for Ch up to the given l.
    """
    return condensed_wignerD_from_operations([identity(max_l), sigma_xy(max_l)])


def Ci(max_l: int) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for Ci up to the given l.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.

    Returns
    -------
    np.ndarray
        The WignerD matrix for Ci up to the given l.
    """

    return np.array(
        [
            delta(mprime, m) * delta(l % 2, 0)
            for l, mprime, m in iter_sph_indices(max_l)
        ],
        dtype=complex,
    )


def Cn(max_l: int, n: int) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for Cn (Cyclic group) up to the given l.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.
    n : int
        The order of the cyclic group.

    Returns
    -------
    np.ndarray
        The WignerD matrix for Cn up to the given l.
    """
    return np.array(
        [
            delta(mprime, m) * delta(m % n, 0)
            for _, mprime, m in iter_sph_indices(max_l)
        ],
        dtype=complex,
    )


def Sn(max_l: int, n: int) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for Sn (rotoreflection group) up to the given l.

    Elements of Sn are given by Sn = {I, Sn, Cn**2, Sn**3, Cn**4, Sn**5, ... until n-1
    (last can be C or S depending if n is odd or even)} according to
    https://en.wikipedia.org/wiki/Point_groups_in_three_dimensions and character tables.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.
    n : int
        The order of the rotation group.

    Returns
    -------
    np.ndarray
        The WignerD matrix for Sn up to the given l.
    """
    id_operation = identity(max_l)
    operations = [id_operation]
    rr_operation = rotoreflection(max_l, n)
    if n % 2 == 0:
        for i in range(1, n):
            new_op = id_operation
            for _ in range(1, i):
                new_op = dot_product(rr_operation, new_op)
        operations.append(new_op)
    else:
        for i in range(1, 2 * n):
            new_op = id_operation
            for _ in range(1, i):
                new_op = dot_product(new_op, rotoreflection(max_l, n))
    return condensed_wignerD_from_operations(operations)


def Cnh(max_l: int, n: int) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for Cnh (Cyclic group with reflection) up to the given
    l.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.
    n : int
        The order of the cyclic group.

    Returns
    -------
    np.ndarray
        The WignerD matrix for Cnh up to the given l.
    """
    if n % 2 == 1:
        return Sn(max_l, n)
    else:
        id_operation = identity(max_l)
        sigma_h_operation = sigma_xy(max_l)
        operations = [id_operation, sigma_h_operation]
        rr_operation = rotoreflection(max_l, n)
        rotation_operation = n_z(max_l, n)
        for i in range(1, n):
            rotation_op = id_operation
            rotoref_op = id_operation
            for _ in range(1, i):
                rotoref_op = dot_product(rotoref_op, rr_operation)
                rotation_op = dot_product(rotation_op, rotation_operation)
            operations.append(rotation_op)
            operations.append(rotoref_op)
        return condensed_wignerD_from_operations(operations)


def Cnv(max_l: int, n: int) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for Cnv (Cyclic group with reflection) up to the given
    l.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.
    n : int
        The order of the cyclic group.

    Returns
    -------
    np.ndarray
        The WignerD matrix for Cnv up to the given l.
    """
    id_operation = identity(max_l)
    sigma_v_operation = sigma_yz(max_l)
    rotation_operation = n_z(max_l, n)
    operations = [id_operation, sigma_v_operation]
    for i in range(1, n):
        rotation_op = id_operation
        for _ in range(1, i):
            rotation_op = dot_product(rotation_op, rotation_operation)
        operations.append(dot_product(sigma_v_operation, rotation_op))
        operations.append(rotation_op)
    return condensed_wignerD_from_operations(operations)


def Dn(max_l: int, n: int) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for Dn (Dihedral group) up to the given l.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.
    n : int
        The order of the dihedral group.

    Returns
    -------
    np.ndarray
        The WignerD matrix for Dn up to the given l.
    """
    id_operation = identity(max_l)
    c2x_operation = two_x(max_l)
    rotation_operation = n_z(max_l, n)
    operations = [id_operation, c2x_operation]
    for i in range(1, n):
        rotation_op = id_operation
        for _ in range(1, i):
            rotation_op = dot_product(rotation_op, rotation_operation)
        operations.append(rotation_op)
        operations.append(dot_product(c2x_operation, rotation_op))
    return condensed_wignerD_from_operations(operations)
    # TODO check if equivalent
    # return (
    #    np.array(
    #        [
    #            (delta(mprime, m) + delta(mprime, -m) * (-1) ** l) * delta(m % n, 0)
    #            for l, mprime, m in iter_sph_indices(max_l)
    #        ],
    #        dtype=complex,
    #    )
    #    / 2
    # )


def Dnh(max_l: int, n: int) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for Dnh (Dihedral group with reflection) up to the
    given l.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.
    n : int
        The order of the dihedral group.

    Returns
    -------
    np.ndarray
        The WignerD matrix for Dnh up to the given l.
    """
    id_operation = identity(max_l)
    c2x_operation = two_x(max_l)
    sigma_h_operation = sigma_xy(max_l)
    rotation_operation = n_z(max_l, n)
    rr_operation = rotoreflection(max_l, n)
    operations = [id_operation, c2x_operation, sigma_h_operation]
    for i in range(1, n):
        rotation_op = id_operation
        for _ in range(1, i):
            rotation_op = dot_product(rotation_op, rotation_operation)
        # Cn
        operations.append(rotation_op)
        # C2'
        c2prime_op = dot_product(c2x_operation, rotation_op)
        operations.append(c2prime_op)
        # sigma_h x C2 x Cn
        operations.append(dot_product(sigma_h_operation, c2prime_op))
    for i in range(1, n, 2):
        rotoref_op = id_operation
        for _ in range(1, i):
            rotoref_op = dot_product(rotoref_op, rr_operation)
        operations.append(rotoref_op)
    return condensed_wignerD_from_operations(operations)


def Dnd(max_l: int, n: int) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for Dnd (Dihedral group with n-fold rotation) up to
    the given l.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.
    n : int
        The order of the dihedral group.

    Returns
    -------
    np.ndarray
        The WignerD matrix for Dnd up to the given l.
    """
    id_operation = identity(max_l)
    inv_operation = inversion(max_l)
    c2x_operation = two_x(max_l)
    rotation_operation = n_z(max_l, n)
    rr_operation = rotoreflection(max_l, n)
    operations = [id_operation, c2x_operation]
    for i in range(1, n):
        rotation_op = id_operation
        for _ in range(1, i):
            rotation_op = dot_product(rotation_op, rotation_operation)
        # Cn
        operations.append(rotation_op)
        # C2'
        c2prime_op = dot_product(c2x_operation, rotation_op)
        operations.append(c2prime_op)
        # sigma_h x C2 x Cn
        operations.append(dot_product(inv_operation, c2prime_op))
    for i in range(1, 2 * n, 2):
        rotoref_op = id_operation
        for _ in range(1, i):
            rotoref_op = dot_product(rotoref_op, rr_operation)
        operations.append(rotoref_op)
    return condensed_wignerD_from_operations(operations)


def iter_sph_indices(max_l: int) -> Generator[int, int, int]:
    """Yield the l, m', and m values in order of appearance in matrices.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.

    Yields
    ------
    int
        The l value.
    int
        The mprime value.
    int
        The m value.
    """
    for l in range(max_l + 1):
        ms = range(-l, l + 1)
        for mprime, m in itertools.product(ms, repeat=2):
            yield l, mprime, m


def delta(a: int, b: int) -> int:
    """Kronecker delta function.

    Parameters
    ----------
    a : int
        The first index.
    b : int
        The second index.

    Returns
    -------
    int
        1 if a == b, 0 otherwise."""
    return int(a == b)


def convert_to_list_of_matrices(D: np.ndarray) -> list[np.ndarray]:  # noqa N802
    """Convert a condensed WignerD matrix for all l's to a list of matrices per l.

    Parameters
    ----------
    D : np.ndarray
        The condensed WignerD matrix for all l's. This should be a 1D array.
    max_l : int
        The maximum l value used to generate the D.

    Returns
    -------
    list[np.ndarray]
        A list of matrices per l.
    """
    matrices = []
    current = 0
    i = 0
    while current < D.size:
        dimz = i * 2 + 1
        matrix = D[current : current + dimz**2]
        current += dimz**2
        matrices.append(matrix.reshape(dimz, dimz))
        i += 1
    return matrices


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


def semidirect_product(D_a: np.ndarray, D_b: np.ndarray) -> np.ndarray:  # noqa N802
    """Compute the semidirect product of two WignerD matrices.

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


def dot_product(
    condensed_operation_a: np.ndarray, condensed_operation_b: np.ndarray
) -> np.ndarray:  # noqa N802
    """Direct product of two sets of WignerD matrices.

    Parameters
    ----------
    condensed_operation_a : np.ndarray
        The first set of WignerD matrices.
    condensed_operation_b : np.ndarray
        The second set of WignerD matrices.

    Returns
    -------
    np.ndarray
        The direct product of the two sets of WignerD matrices.
    """
    result = []
    for matrix1, matrix2 in zip(
        convert_to_list_of_matrices(condensed_operation_a),
        convert_to_list_of_matrices(condensed_operation_b),
    ):
        result.append(np.dot(matrix1, matrix2).flatten())
    return np.concatenate(result)


def _parse_point_group(schonflies_symbol):
    """Parse a Schönflies symbol into its component parts.

    For example, Ih would have the family "I" and modifier "h" with no order
    while C6 would have the family "C" and order "6" with no modifier, and the
    point group C6h would have famile "C", modifier "h", and order "6".
    """
    # once we figure out how to do Sn systematically stop hard coding.
    if schonflies_symbol == "S2":
        schonflies_symbol = "Ci"
    elif schonflies_symbol == "S6":
        schonflies_symbol = "C3i"
    elif schonflies_symbol == "V":
        schonflies_symbol = "D2"
    family = schonflies_symbol[0]
    if len(schonflies_symbol) == 1:
        return (family, None, None)
    modifier = schonflies_symbol[-1] if schonflies_symbol[-1].isalpha() else None
    if len(schonflies_symbol) == 2 and modifier is not None:
        return (family, modifier, None)
    order = int(schonflies_symbol[1:-1]) if modifier else int(schonflies_symbol[1:])
    if family == "V":
        family = "D"
        order = 2
    return (family, modifier, order)


def condensed_wignerD_from_operations(  # noqa N802
    operations: list[np.ndarray],
):
    """Compute the condensed WignerD matrix for a given set of operations.

    Parameters
    ----------
    operations : list[callable]
        A list of operations that return a WignerD matrix.

    Returns
    -------
    np.ndarray
        The condensed WignerD matrix for the point group.
    """
    return np.sum(operations, axis=0) / len(operations)


def compute_condensed_wignerD_for_C_family(  # noqa N802
    max_l: int, modifier: str, order: int
) -> np.ndarray:
    """
    Compute the condensed WignerD matrix for a given C family.

    Parameters
    ----------
    max_l : int
        The maximum l value to include in the WignerD matrices.
    modifier : str
        The modifier for the point group.
    order : int
        The order of the cyclic group.

    Returns
    -------
    np.ndarray
        The condensed WignerD matrix for the point group.
    """
    if modifier == "i" and order is not None:
        return semidirect_product(Cn(max_l, order), Ci(max_l))
    elif modifier == "h" and order is not None:
        return Cnh(max_l, order)
    elif modifier == "v" and order is not None:
        return Cnv(max_l, order)
    elif modifier is None and order is not None:
        return Cn(max_l, order)
    elif modifier == "i" and order is None:
        return Ci(max_l)
    elif modifier == "h" and order is None:
        return Ch(max_l)
    elif modifier == "s" and order is None:
        return Cs(max_l)
    else:
        return None


def compute_condensed_wignerD_for_D_family(  # noqa N802
    max_l: int, modifier: str, order: int
) -> np.ndarray:
    """
    Compute the condensed WignerD matrix for a given D family.

    Parameters
    ----------
    max_l : int
        The maximum l value to include in the WignerD matrices.
    modifier : str
        The modifier for the point group.
    order : int
        The order of the dihedral group.

    Returns
    -------
    np.ndarray
        The condensed WignerD matrix for the point group.
    """
    if modifier == "d" and order is not None:
        return Dnd(max_l, order)
    elif modifier == "h" and order is not None:
        return Dnh(max_l, order)
    elif modifier is None and order is not None:
        return Dn(max_l, order)
    else:
        return None


def compute_condensed_wignerD_for_S_family(  # noqa N802
    max_l: int, modifier: str, order: int
) -> np.ndarray:
    """
    Compute the condensed WignerD matrix for a given S family.

    Parameters
    ----------
    max_l : int
        The maximum l value to include in the WignerD matrices.
    modifier : str
        The modifier for the point group.
    order : int
        The order of the group.

    Returns
    -------
    np.ndarray
        The condensed WignerD matrix for the point group.
    """
    if modifier is None and order is not None:
        return Sn(max_l, order)
    else:
        return None


def compute_condensed_wignerD_for_tetrahedral_family(  # noqa N802
    max_l: int, modifier: str
) -> np.ndarray:
    """
    Compute the condensed WignerD matrix for a given terahedral family.

    Parameters
    ----------
    max_l : int
        The maximum l value to include in the WignerD matrices.
    modifier : str
        The modifier for the point group.

    Returns
    -------
    np.ndarray
        The condensed WignerD matrix for the point group.
    """
    if modifier == "d":
        return semidirect_product(Dn(max_l, 2), Cnv(max_l, 3))
    else:
        operations = []
        for rot in base_rotations:
            operations.append(generalized_rotation(max_l, *rot))
        T = condensed_wignerD_from_operations(operations)  # noqa N806
        if modifier == "h":
            return semidirect_product(T, Ci(max_l))
        elif modifier is None:
            return T
        else:
            return None


def compute_condensed_wignerD_for_octahedral_family(  # noqa N802
    max_l: int, modifier: str
) -> np.ndarray:
    """
    Compute the condensed WignerD matrix for a given octahedral family.

    Parameters
    ----------
    max_l : int
        The maximum l value to include in the WignerD matrices.
    modifier : str
        The modifier for the point group.

    Returns
    -------
    np.ndarray
        The condensed WignerD matrix for the point group.
    """
    operations = []
    for rot in np.concatenate((base_rotations, octahedral_rotations)):
        operations.append(generalized_rotation(max_l, *rot))
    O = condensed_wignerD_from_operations(operations)  # noqa N806
    if modifier == "h":
        return semidirect_product(O, Ci(max_l))
    elif modifier is None:
        return O
    else:
        return None


def compute_condensed_wignerD_for_icosahedral_family(  # noqa N802
    max_l: int, modifier: str
) -> np.ndarray:
    """
    Compute the condensed WignerD matrix for a given icosahedral family.

    Parameters
    ----------
    max_l : int
        The maximum l value to include in the WignerD matrices.
    modifier : str
        The modifier for the point group.

    Returns
    -------
    np.ndarray
        The condensed WignerD matrix for the point group.
    """
    operations = []
    for rot in np.concatenate((base_rotations, icosahedral_rotations)):
        operations.append(generalized_rotation(max_l, *rot))
    I = condensed_wignerD_from_operations(operations)  # noqa N806
    if modifier == "h":
        return semidirect_product(I, Ci(max_l))
    elif modifier is None:
        return I
    else:
        return None


def compute_condensed_wignerD_matrix_for_a_given_point_group(  # noqa N802
    point_group: str, max_l: int
) -> np.ndarray:
    """
    Compute the condensed WignerD matrix for a given point group.

    Parameters
    ----------
    point_group : str
        The point group in Schoenflies notation.
    max_l : int
        The maximum l value to include in the WignerD matrices.

    Returns
    -------
    np.ndarray
        The condensed WignerD matrix for the point group.
    """
    family, modifier, order = _parse_point_group(point_group)
    if family == "T":
        matrix = compute_condensed_wignerD_for_tetrahedral_family(max_l, modifier)
    elif family == "O":
        matrix = compute_condensed_wignerD_for_octahedral_family(max_l, modifier)
    elif family == "I":
        matrix = compute_condensed_wignerD_for_icosahedral_family(max_l, modifier)
    elif family == "C":
        matrix = compute_condensed_wignerD_for_C_family(max_l, modifier, order)
    elif family == "D":
        matrix = compute_condensed_wignerD_for_D_family(max_l, modifier, order)
    elif family == "S":
        matrix = compute_condensed_wignerD_for_S_family(max_l, modifier, order)
    else:
        matrix = None
    if matrix is not None:
        return matrix
    else:
        raise KeyError(f"{point_group} is not currently supported.")


class WignerD:
    def __init__(self, point_group: str, max_l: int) -> None:
        """Create a WignerD object.

        Parameters
        ----------
        point_group : str
            The point group in Schoenflies notation.
        max_l : int
            The highest spherical harmonic to include in the WignerD matrices.
        """
        self._max_l = max_l
        self._point_group = point_group
        self._matrix = compute_condensed_wignerD_matrix_for_a_given_point_group(
            point_group, max_l
        )

    @property
    def max_l(self) -> int:
        """The maximum l value used for constructing the matrix

        Returns
        -------
        int
            The maximum l value."""
        return self._max_l

    @property
    def point_group(self) -> str:
        """The point group in Schoenflies notation to return.

        Returns
        -------
        str
            The point group in Schoenflies notation.
        """
        return self._point_group

    @property
    def matrices(self) -> list[np.ndarray]:
        """The WignerD matrices for the point group up to the given l.

        Returns
        -------
        list[np.ndarray]
            The WignerD matrices for the point group up to the given l.
        """
        return convert_to_list_of_matrices(self._matrix)

    @property
    def condensed_matrices(self) -> np.ndarray:
        """The condensed WignerD matrices for the point group up to the given l.

        The matrices are flattend to give a 1D array

        Returns
        -------
        np.ndarray
            The condensed WignerD matrices for the point group up to the given l.
        """
        return self._matrix

    @property
    def matrix(self, l: int) -> np.ndarray:
        """The WignerD matrix for the point group for a given l.

        Parameters
        ----------
        l : int
            The l value for which to return the matrix

        Returns
        -------
        np.ndarray
            The WignerD matrix for the point group for a given l.
        """
        return self.matrices[l]

    @classmethod
    def from_condensed_matrix_maxl_point_group(
        cls, matrix: np.ndarray, max_l: int, point_group: str
    ):
        """Create a WignerD object from a condensed matrix.

        Parameters
        ----------
        matrix : np.ndarray
            The condensed matrix.
        max_l : int
            The maximum l value used to generate the matrix.
        point_group : str
            The point group in Schoenflies notation.

        Returns
        -------
        WignerDmatrix
            The WignerD object.
        """
        wigner = cls.__new__(cls)
        wigner._matrix = matrix
        wigner._max_l = max_l
        wigner._point_group = point_group
        return wigner
