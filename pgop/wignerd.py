import itertools
from typing import Generator

import numpy as np
import scipy.spatial
import scipy.special


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


def inversion(max_l: int) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for inversion up to the given l.

    Implementation according to Altman, Mathematical Proceedings of the Cambridge
    Philosophical Society , Volume 53 , Issue 2 , April 1957
    (https://doi.org/10.1017/S0305004100032370), p.347 (bottom of the page).

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


def rotation_from_euler_angles(
    max_l: int, alpha: float, beta: float, gamma: float
) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for a generalized rotation up to the given l.

    Implementation according to altmann. It uses zyz extrinsic standard.

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


def rotation_from_axis_angle(max_l: int, axis: np.ndarray, angle: float) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for a generalized rotation up to the given l.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.
    axis : np.ndarray
        The axis of rotation.
    angle : float
        The angle of rotation in radians.

    Returns
    -------
    np.ndarray
        The WignerD matrix for the generalized rotation up to the given l.
    """
    # normalize the axis of rotation
    rotation_axis = axis / np.linalg.norm(axis)
    rotation_euler = scipy.spatial.transform.Rotation.from_rotvec(
        rotation_axis * angle
    ).as_euler("zyz")
    # compute wigner D matrix
    return rotation_from_euler_angles(max_l, *rotation_euler)


def rotation_from_axis_order(max_l: int, axis: np.ndarray, order: int) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for a generalized rotation up to the given l.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.
    axis : np.ndarray
        The axis of rotation.
    order : int
        The order of the rotation.

    Returns
    -------
    np.ndarray
        The WignerD matrix for the generalized rotation up to the given l.
    """
    return rotation_from_axis_angle(max_l, axis, 2 * np.pi / order)


def rotoreflection_from_euler_angles(
    max_l: int, alpha: float, beta: float, gamma: float
) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for a generalized rotoreflection up to the given l.

    Implementation according to Altman, Mathematical Proceedings of the Cambridge
    Philosophical Society , Volume 53 , Issue 2 , April 1957
    (https://doi.org/10.1017/S0305004100032370), p.347 (bottom of the page).
    Implementation according to altmann. It uses zyz extrinsic standard.

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
        The WignerD matrix for the generalized rotoreflection up to the given l.
    """
    rotation_operator = rotation_from_euler_angles(max_l, alpha, beta, gamma)
    # find axis of rotation
    rotation_axis = scipy.spatial.transform.Rotation.from_euler(
        "zyz", [alpha, beta, gamma]
    ).as_rotvec()
    # normalize rotation axis
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    rotoreflection_operator = reflection_from_normal(max_l, rotation_axis)
    return dot_product(rotoreflection_operator, rotation_operator)


def rotoreflection_from_axis_angle(
    max_l: int, axis: np.ndarray, angle: float
) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for a generalized rotoreflection up to the given l.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.
    axis : np.ndarray
        The axis of rotation.
    angle : float
        The angle of rotation.

    Returns
    -------
    np.ndarray
        The WignerD matrix for the generalized rotoreflection up to the given l.
    """
    # normalize the axis of rotation
    rotation_axis = axis / np.linalg.norm(axis)
    rotoreflection_euler = scipy.spatial.transform.Rotation.from_rotvec(
        rotation_axis * angle
    ).as_euler("zyz")
    # find the rotoreflection matrix
    return rotoreflection_from_euler_angles(max_l, *rotoreflection_euler)


def rotoreflection_from_axis_order(
    max_l: int, axis: np.ndarray, order: int
) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for a generalized rotoreflection up to the given l.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.
    axis : np.ndarray
        The axis of rotation.
    order : int
        The order of the rotation.

    Returns
    -------
    np.ndarray
        The WignerD matrix for the generalized rotoreflection up to the given l.
    """
    return rotoreflection_from_axis_angle(max_l, axis, 2 * np.pi / order)


def reflection_from_normal(max_l: int, normal: np.ndarray):
    """Return the WignerD matrix for a generalized reflection up to the given l.

    A general reflection can be represented as a rotation by 180 degrees around a given
    axis, followed by an inversion. Implementation according to altmann.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.
    normal : np.ndarray
        The normal vector of the plane of reflection.

    Returns
    -------
    np.ndarray
        The WignerD matrix for the generalized sigma up to the given l.
    """
    inversion_operator = inversion(max_l)
    rotation_operator = rotation_from_axis_angle(max_l, normal, np.pi)
    return dot_product(inversion_operator, rotation_operator)


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
    return condensed_wignerD_from_operations(
        [identity(max_l), reflection_from_normal(max_l, [1, 0, 0])]
    )


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
    return condensed_wignerD_from_operations(
        [identity(max_l), reflection_from_normal(max_l, [0, 0, 1])]
    )


def Ci(max_l: int) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for Ci up to the given l.

    Ci={E, i}

    Parameters
    ----------
    max_l : int
        The maximum l value to include.

    Returns
    -------
    np.ndarray
        The WignerD matrix for Ci up to the given l.
    """
    return condensed_wignerD_from_operations([identity(max_l), inversion(max_l)])


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
    identity_operation = identity(max_l)
    operations = [identity_operation]
    rotation_operation = rotation_from_euler_angles(max_l, 2 * np.pi / n, 0, 0)
    for _ in range(1, n):
        operations.append(dot_product(operations[-1], rotation_operation))
    return condensed_wignerD_from_operations(operations)


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
    rr_operation = rotoreflection_from_euler_angles(max_l, 0, 0, 2 * np.pi / n)
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
                new_op = dot_product(new_op, rr_operation)
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
        sigma_h_operation = reflection_from_normal(max_l, [0, 0, 1])
        operations = [id_operation, sigma_h_operation]
        rr_operation = rotoreflection_from_euler_angles(max_l, 0, 0, 2 * np.pi / n)
        rotation_operation = rotation_from_euler_angles(max_l, 2 * np.pi / n, 0, 0)
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
    sigma_v_operation = reflection_from_normal(max_l, [1, 0, 0])
    rotation_operation = rotation_from_euler_angles(max_l, 2 * np.pi / n, 0, 0)
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
    c2x_operation = rotation_from_axis_order(max_l, np.array([1, 0, 0]), 2)
    rotation_operation = rotation_from_euler_angles(max_l, 2 * np.pi / n, 0, 0)
    operations = [id_operation, c2x_operation]
    for i in range(1, n):
        rotation_op = id_operation
        for _ in range(1, i):
            rotation_op = dot_product(rotation_op, rotation_operation)
        operations.append(rotation_op)
        #operations.append(rotation_from_euler_angles(max_l, 2 * np.pi / i, np.pi, 0))
        operations.append(dot_product(c2x_operation, rotation_op))
    print(len(operations))
    return condensed_wignerD_from_operations(operations)


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
    c2x_operation = rotation_from_axis_order(max_l, np.array([1, 0, 0]), 2)
    sigma_h_operation = reflection_from_normal(max_l, [0, 0, 1])
    rotation_operation = rotation_from_euler_angles(max_l, 2 * np.pi / n, 0, 0)
    rr_operation = rotoreflection_from_euler_angles(max_l, 0, 0, 2 * np.pi / n)
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
    c2x_operation = rotation_from_axis_order(max_l, np.array([1, 0, 0]), 2)
    rotation_operation = rotation_from_euler_angles(max_l, 2 * np.pi / n, 0, 0)
    rr_operation = rotoreflection_from_euler_angles(max_l, 0, 0, 2 * np.pi / n)
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
    if modifier == "h" and order is not None:
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
    operations = []
    for i in scipy.spatial.transform.Rotation.create_group("T"):
        rot = i.as_euler("zyz")
        operations.append(rotation_from_euler_angles(max_l, *rot))
    if modifier == "d":
        # 6 S4
        # 90 degrees around 1, 0, 0
        first_S4 = rotoreflection_from_axis_order(max_l, [0, 0, 1], 4)  # noqa N806
        # 90 degrees around 0, 1, 0
        second_S4 = rotoreflection_from_axis_order(max_l, [0, 1, 0], 4)  # noqa N806
        # 90 degrees around 0, 0, 1
        third_S4 = rotoreflection_from_axis_order(max_l, [1, 0, 0], 4)  # noqa N806
        # 270 degrees around 1, 0, 0
        # 270 degrees around 0, 1, 0
        # 270 degrees around 0, 0, 1
        fourth_S4 = identity(max_l)  # noqa N806
        fifth_S4 = identity(max_l)  # noqa N806
        sixth_S4 = identity(max_l)  # noqa N806
        for _ in range(3):
            fourth_S4 = dot_product(fourth_S4, first_S4)  # noqa N806
            fifth_S4 = dot_product(fifth_S4, second_S4)  # noqa N806
            sixth_S4 = dot_product(sixth_S4, third_S4)  # noqa N806
        operations.append(first_S4)
        operations.append(second_S4)
        operations.append(third_S4)
        operations.append(fourth_S4)
        operations.append(fifth_S4)
        operations.append(sixth_S4)
        # 6 sigma v
        operations.append(
            reflection_from_normal(max_l, [0, -np.sqrt(2) / 2, -np.sqrt(2) / 2])
        )
        operations.append(
            reflection_from_normal(max_l, [0, np.sqrt(2) / 2, -np.sqrt(2) / 2])
        )
        operations.append(
            reflection_from_normal(max_l, [-np.sqrt(2) / 2, 0, -np.sqrt(2) / 2])
        )
        operations.append(
            reflection_from_normal(max_l, [np.sqrt(2) / 2, 0, -np.sqrt(2) / 2])
        )
        operations.append(
            reflection_from_normal(max_l, [-np.sqrt(2) / 2, -np.sqrt(2) / 2, 0])
        )
        operations.append(
            reflection_from_normal(max_l, [np.sqrt(2) / 2, -np.sqrt(2) / 2, 0])
        )
        return condensed_wignerD_from_operations(operations)
    elif modifier == "h":
        inversion_operation = inversion(max_l)
        new_operations = operations.copy()
        for op in operations:
            new_operations.append(dot_product(op, inversion_operation))
        return condensed_wignerD_from_operations(new_operations)
    elif modifier is None:
        return condensed_wignerD_from_operations(operations)
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
    for i in scipy.spatial.transform.Rotation.create_group("O"):
        rot = i.as_euler("zyz")
        operations.append(rotation_from_euler_angles(max_l, *rot))
    if modifier == "h":
        inversion_operation = inversion(max_l)
        new_operations = operations.copy()
        for op in operations:
            new_operations.append(dot_product(op, inversion_operation))
        return condensed_wignerD_from_operations(new_operations)
    elif modifier is None:
        return condensed_wignerD_from_operations(operations)
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
    for i in scipy.spatial.transform.Rotation.create_group("I"):
        rot = i.as_euler("zyz")
        operations.append(rotation_from_euler_angles(max_l, *rot))
    if modifier == "h":
        inversion_operation = inversion(max_l)
        new_operations = operations.copy()
        for op in operations:
            new_operations.append(dot_product(op, inversion_operation))
        return condensed_wignerD_from_operations(new_operations)
    elif modifier is None:
        return condensed_wignerD_from_operations(operations)
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
