import numpy as np
import scipy.spatial
import scipy.special
from .wignerd import _parse_point_group


def identity() -> np.ndarray:  # noqa: N802
    """Return the Cartesian Representation matrix for E (identity).

    Returns
    -------
    np.ndarray
        The  Cartesian Representation matrix for E.
    """
    return np.eye(3)


def inversion() -> np.ndarray:  # noqa: N802
    """Return the  Cartesian Representation matrix for inversion.

    Returns
    -------
    np.ndarray
        The Cartesian Representation matrix for inversion .
    """
    return -1 * np.eye(3)


def rotation_from_euler_angles(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Return the  Cartesian Representation matrix for a generalized rotation.

    Parameters
    ----------
    alpha : float
        The angle of rotation around the z-axis. Must be between 0 and 2*pi.
    beta : float
        The angle of rotation around the y-axis. Must be between 0 and pi.
    gamma : float
        The angle of rotation around the z-axis. Must be between 0 and 2*pi.

    Returns
    -------
    np.ndarray
        The Cartesian Representation matrix for the generalized rotation .
    """
    return scipy.spatial.transform.Rotation.from_euler(
        "zyz", [alpha, beta, gamma]
    ).as_matrix()


def rotation_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return the Cartesian Representation matrix for a generalized rotation.

    Parameters
    ----------
    axis : np.ndarray
        The axis of rotation.
    angle : float
        The angle of rotation in radians.

    Returns
    -------
    np.ndarray
        The Cartesian Representation matrix for the generalized rotation.
    """
    # normalize the axis of rotation
    rotation_axis = axis / np.linalg.norm(axis)
    return scipy.spatial.transform.Rotation.from_rotvec(
        rotation_axis * angle
    ).as_matrix()


def rotation_from_axis_order(axis: np.ndarray, order: int) -> np.ndarray:
    """Return the Cartesian Representation matrix for a generalized rotation.

    Parameters
    ----------
    axis : np.ndarray
        The axis of rotation.
    order : int
        The order of the rotation.

    Returns
    -------
    np.ndarray
        The Cartesian Representation matrix for the generalized rotation.
    """
    return rotation_from_axis_angle(axis, 2 * np.pi / order)


def rotoreflection_from_euler_angles(
    alpha: float, beta: float, gamma: float
) -> np.ndarray:
    """Return the Cartesian Representation matrix for a generalized rotoreflection.

    Parameters
    ----------
    alpha : float
        The angle of rotation around the z-axis. Must be between 0 and 2*pi.
    beta : float
        The angle of rotation around the y-axis. Must be between 0 and pi.
    gamma : float
        The angle of rotation around the z-axis. Must be between 0 and 2*pi.

    Returns
    -------
    np.ndarray
        The Cartesian Representation matrix for the generalized rotoreflection.
    """
    rotation_operator = rotation_from_euler_angles(alpha, beta, gamma)
    # find axis of rotation
    rotation_axis = scipy.spatial.transform.Rotation.from_euler(
        "zyz", [alpha, beta, gamma]
    ).as_rotvec()
    # normalize rotation axis
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    rotoreflection_operator = reflection_from_normal(rotation_axis)
    return np.dot(rotoreflection_operator, rotation_operator)


def rotoreflection_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return the Cartesian Representation matrix for a generalized rotoreflection.

    Parameters
    ----------
    axis : np.ndarray
        The axis of rotation.
    angle : float
        The angle of rotation.

    Returns
    -------
    np.ndarray
        The Cartesian Representation matrix for the generalized rotoreflection.
    """
    # normalize the axis of rotation
    rotation_axis = axis / np.linalg.norm(axis)
    rotoreflection_euler = scipy.spatial.transform.Rotation.from_rotvec(
        rotation_axis * angle
    ).as_euler("zyz")
    # find the rotoreflection matrix
    return rotoreflection_from_euler_angles(*rotoreflection_euler)


def rotoreflection_from_axis_order(axis: np.ndarray, order: int) -> np.ndarray:
    """Return the Cartesian Representation matrix for a generalized rotoreflection.

    Parameters
    ----------
    axis : np.ndarray
        The axis of rotation.
    order : int
        The order of the rotation.

    Returns
    -------
    np.ndarray
        The Cartesian Representation matrix for the generalized rotoreflection.
    """
    return rotoreflection_from_axis_angle(axis, 2 * np.pi / order)


def reflection_from_normal(normal: np.ndarray):
    """Return the Cartesian Representation matrix for a generalized reflection.

    A general reflection can be represented as a rotation by 180 degrees around a given
    axis, followed by an inversion.

    Parameters
    ----------
    normal : np.ndarray
        The normal vector of the plane of reflection.

    Returns
    -------
    np.ndarray
        The Cartesian Representation matrix for the generalized sigma.
    """
    inversion_operator = inversion()
    rotation_operator = rotation_from_axis_angle(normal, np.pi)
    return np.dot(inversion_operator, rotation_operator)


def _cs_operations() -> np.ndarray:
    """Return the Cartesian Representation matrix for Cs group.

    Cs={E, sigma_yz}

    Returns
    -------
    list[np.ndarray]
        The operations list for Cs.
    """
    return [identity(), reflection_from_normal([1, 0, 0])]


def _ch_operations() -> np.ndarray:
    """Return the operations list for Ch group.

    Ch={E, sigma_xy}

    Returns
    -------
    list[np.ndarray]
        The operations list for Ch.
    """
    return [identity(), reflection_from_normal([0, 0, 1])]


def _ci_operations() -> np.ndarray:
    """Return the operations list for Ci.

    Ci={E, i}

    Returns
    -------
    list[np.ndarray]
        The operations list for Ci.
    """
    return [identity(), inversion()]


def _cn_operations(n: int) -> np.ndarray:
    """Return the operations list for Cn (Cyclic group).

    Elements of Cn are given by Cn = {E, Cn, Cn**2, Cn**3, ... Cn**(n-1)}.

    Parameters
    ----------
    n : int
        The order of the cyclic group.

    Returns
    -------
    list[np.ndarray]
        The operations list for Cn.
    """
    operations = [identity()]
    rotation_operation = rotation_from_euler_angles(2 * np.pi / n, 0, 0)
    for _ in range(1, n):
        operations.append(np.dot(operations[-1], rotation_operation))
    return operations


def _sn_operations(n: int) -> np.ndarray:
    """Return the operations list for Sn (rotoreflection group).

    Elements of Sn are given by Sn = {E, Sn, Sn**2, Sn**3, ... Sn**(n-1)} for even n and
    Sn = {E, Sn, Sn**3, Sn**5, ... Sn**(2n-1), Cn, Cn**2, ... Cn**(n-1)} for odd n. Sn
    is a rotation by 2*pi/n followed by a reflection.

    Parameters
    ----------
    n : int
        The order of the rotation group.

    Returns
    -------
    list[np.ndarray]
        The operations list for Sn.
    """
    identity_operation = identity()
    operations = [identity_operation]
    rotoreflection_operation = rotoreflection_from_euler_angles(0, 0, 2 * np.pi / n)
    if n % 2 == 0:
        for _ in range(1, n):
            operations.append(np.dot(operations[-1], rotoreflection_operation))
    else:
        for _ in range(1, 2 * n):
            operations.append(np.dot(operations[-1], rotoreflection_operation))
    return operations


def _cnh_operations(n: int) -> np.ndarray:
    """Return the operations list for Cnh group .

    Parameters
    ----------
    n : int
        The order of the cyclic group.

    Returns
    -------
    list[np.ndarray]
        The operations list for Cnh .
    """
    operations = _cn_operations(n)
    sigma_h_operation = reflection_from_normal([0, 0, 1])
    for operation in operations.copy():
        operations.append(np.dot(operation, sigma_h_operation))
    return operations


def _cnv_operations(n: int) -> np.ndarray:
    """Return the operations list for Cnv group .

    Parameters
    ----------
    n : int
        The order of the cyclic group.

    Returns
    -------
    list[np.ndarray]
        The operations list for Cnv .
    """
    operations = _cn_operations(n)
    sigma_v_operation = reflection_from_normal([1, 0, 0])
    for operation in operations.copy():
        operations.append(np.dot(operation, sigma_v_operation))
    return operations


def _dn_operations(n: int) -> np.ndarray:
    """Return the operations list for Dn (Dihedral group) .

    Parameters
    ----------
    n : int
        The order of the dihedral group.

    Returns
    -------
    list[np.ndarray]
        The operations list for Dn .
    """
    operations = _cn_operations(n)
    xy_rotation = np.dot(
        reflection_from_normal([0, 0, 1]),
        reflection_from_normal([1, 0, 0]),
    )
    for operation in operations.copy():
        operations.append(np.dot(operation, xy_rotation))
    return operations


def _dnh_operations(n: int) -> np.ndarray:
    """Return the operations list for Dnh group .

    Parameters
    ----------
    n : int
        The order of the dihedral group.

    Returns
    -------
    list[np.ndarray]
        The operations list for Dnh .
    """
    operations = _dn_operations(n)
    identity_operation = identity()
    sigma_h_operation = reflection_from_normal([0, 0, 1])
    rotoreflection_operation = rotoreflection_from_euler_angles(0, 0, 2 * np.pi / n)
    for i in range(n, 2 * n):
        c2prime_operation = operations[i]
        operations.append(np.dot(sigma_h_operation, c2prime_operation))
    operations.append(sigma_h_operation)
    for i in range(1, n, 2):
        final_operation = identity_operation
        for _ in range(0, i):
            final_operation = np.dot(final_operation, rotoreflection_operation)
        operations.append(final_operation)
    return operations


def _dnd_operations(n: int) -> np.ndarray:
    """Return the operations list for Dnd group .

    Parameters
    ----------
    n : int
        The order of the dihedral group.

    Returns
    -------
    list[np.ndarray]
        The operations list for Dnd .
    """
    operations = _sn_operations(2 * n)
    vertical_reflection = reflection_from_normal([0, 1, 0])
    for operation in operations.copy():
        operations.append(np.dot(operation, vertical_reflection))
    return operations


def _rotation_operations_for_polyhedral_point_groups(point_group: str) -> np.ndarray:
    """Return the operations list for a given polyhedral point group.

    Parameters
    ----------
    point_group : str
        The point group for which the operations are to be computed.

    Returns
    -------
    list[np.ndarray]
        The operations list for the given polyhedral point group .
    """
    operations = []
    for i in scipy.spatial.transform.Rotation.create_group(point_group):
        rot = i.as_euler("zyz")
        operations.append(rotation_from_euler_angles(*rot))
    return operations


def compute_Cartesian_Representation_for_C_family(  # noqa N802
    modifier: str, order: int
) -> list[np.ndarray]:
    """
    Compute the Cartesian Representation matrices for a given C family.

    Parameters
    ----------
    modifier : str
        The modifier for the point group.
    order : int
        The order of the cyclic group.

    Returns
    -------
    list[np.ndarray]
        The Cartesian Representation matrices for the point group.
    """
    if modifier == "h" and order is not None:
        return _cnh_operations(order)
    elif modifier == "v" and order is not None:
        return _cnv_operations(order)
    elif modifier is None and order is not None:
        return _cn_operations(order)
    elif modifier == "i" and order is None:
        return _ci_operations()
    elif modifier == "h" and order is None:
        return _ch_operations()
    elif modifier == "s" and order is None:
        return _cs_operations()
    else:
        return None


def compute_Cartesian_Representation_for_D_family(  # noqa N802
    modifier: str, order: int
) -> list[np.ndarray]:
    """
    Compute the Cartesian Representation matrices for a given D family.

    Parameters
    ----------
    modifier : str
        The modifier for the point group.
    order : int
        The order of the dihedral group.

    Returns
    -------
    list[np.ndarray]
        The Cartesian Representation matrices for the point group.
    """
    if modifier == "d" and order is not None:
        return _dnd_operations(order)
    elif modifier == "h" and order is not None:
        return _dnh_operations(order)
    elif modifier is None and order is not None:
        return _dn_operations(order)
    else:
        return None


def compute_Cartesian_Representation_for_S_family(  # noqa N802
    modifier: str, order: int
) -> list[np.ndarray]:
    """
    Compute the Cartesian Representation matrices for a given S family.

    Parameters
    ----------
    modifier : str
        The modifier for the point group.
    order : int
        The order of the group.

    Returns
    -------
    list[np.ndarray]
        The Cartesian Representation matrices for the point group.
    """
    if modifier is None and order is not None:
        return _sn_operations(order)
    else:
        return None


def compute_Cartesian_Representation_for_tetrahedral_family(  # noqa N802
    modifier: str,
) -> list[np.ndarray]:
    """
    Compute the Cartesian Representation matrices for a given terahedral family.

    Parameters
    ----------
    modifier : str
        The modifier for the point group.

    Returns
    -------
    np.ndarray
        The Cartesian Representation matrices for the point group.
    """
    operations = _rotation_operations_for_polyhedral_point_groups("T")
    if modifier == "d":
        # 6 S4
        # 90 degrees around 1, 0, 0
        first_S4 = rotoreflection_from_axis_order([0, 0, 1], 4)  # noqa N806
        # 90 degrees around 0, 1, 0
        second_S4 = rotoreflection_from_axis_order([0, 1, 0], 4)  # noqa N806
        # 90 degrees around 0, 0, 1
        third_S4 = rotoreflection_from_axis_order([1, 0, 0], 4)  # noqa N806
        # 270 degrees around 1, 0, 0
        # 270 degrees around 0, 1, 0
        # 270 degrees around 0, 0, 1
        fourth_S4 = identity()  # noqa N806
        fifth_S4 = identity()  # noqa N806
        sixth_S4 = identity()  # noqa N806
        for _ in range(3):
            fourth_S4 = np.dot(fourth_S4, first_S4)  # noqa N806
            fifth_S4 = np.dot(fifth_S4, second_S4)  # noqa N806
            sixth_S4 = np.dot(sixth_S4, third_S4)  # noqa N806
        operations.append(first_S4)
        operations.append(second_S4)
        operations.append(third_S4)
        operations.append(fourth_S4)
        operations.append(fifth_S4)
        operations.append(sixth_S4)
        # 6 sigma v
        operations.append(reflection_from_normal([0, -np.sqrt(2) / 2, -np.sqrt(2) / 2]))
        operations.append(reflection_from_normal([0, np.sqrt(2) / 2, -np.sqrt(2) / 2]))
        operations.append(reflection_from_normal([-np.sqrt(2) / 2, 0, -np.sqrt(2) / 2]))
        operations.append(reflection_from_normal([np.sqrt(2) / 2, 0, -np.sqrt(2) / 2]))
        operations.append(reflection_from_normal([-np.sqrt(2) / 2, -np.sqrt(2) / 2, 0]))
        operations.append(reflection_from_normal([np.sqrt(2) / 2, -np.sqrt(2) / 2, 0]))
        return operations
    elif modifier == "h":
        inversion_operation = inversion()
        for operation in operations.copy():
            operations.append(np.dot(operation, inversion_operation))
        return operations
    elif modifier is None:
        return operations
    else:
        return None


def compute_Cartesian_Representation_for_octahedral_family(  # noqa N802
    modifier: str,
) -> list[np.ndarray]:
    """
    Compute the Cartesian Representation matrices for a given octahedral family.

    Parameters
    ----------
    modifier : str
        The modifier for the point group.

    Returns
    -------
    list[np.ndarray]
        The Cartesian Representation matrices for the point group.
    """
    operations = _rotation_operations_for_polyhedral_point_groups("O")
    if modifier == "h":
        inversion_operation = inversion()
        for operation in operations.copy():
            operations.append(np.dot(operation, inversion_operation))
        return operations
    elif modifier is None:
        return operations
    else:
        return None


def compute_Cartesian_Representation_for_icosahedral_family(  # noqa N802
    modifier: str,
) -> list[np.ndarray]:
    """
    Compute the Cartesian Representation matrix for a given icosahedral family.

    Parameters
    ----------
    modifier : str
        The modifier for the point group.

    Returns
    -------
    list[np.ndarray]
        The Cartesian Representation matrices for the point group.
    """
    operations = _rotation_operations_for_polyhedral_point_groups("I")
    if modifier == "h":
        inversion_operation = inversion()
        for operation in operations.copy():
            operations.append(np.dot(operation, inversion_operation))
        return operations
    elif modifier is None:
        return operations
    else:
        return None


def compute_Cartesian_Representation_matrix_for_a_given_point_group(  # noqa N802
    point_group: str,
) -> np.ndarray:
    """
    Compute the Cartesian Representation matrices for a given point group.

    Parameters
    ----------
    point_group : str
        The point group in Schoenflies notation.

    Returns
    -------
    np.ndarray
        The Cartesian Representation matrices for the point group.
    """
    family, modifier, order = _parse_point_group(point_group)
    if family == "T":
        matrix = compute_Cartesian_Representation_for_tetrahedral_family(modifier)
    elif family == "O":
        matrix = compute_Cartesian_Representation_for_octahedral_family(modifier)
    elif family == "I":
        matrix = compute_Cartesian_Representation_for_icosahedral_family(modifier)
    elif family == "C":
        matrix = compute_Cartesian_Representation_for_C_family(modifier, order)
    elif family == "D":
        matrix = compute_Cartesian_Representation_for_D_family(modifier, order)
    elif family == "S":
        matrix = compute_Cartesian_Representation_for_S_family(modifier, order)
    else:
        matrix = None
    if matrix is not None:
        return matrix
    else:
        raise KeyError(f"{point_group} is not currently supported.")


class CartesianRepMatrix:
    def __init__(self, point_group: str):
        """Create a Cartesian Representation of a Point Group.

        Parameters
        ----------
        point_group : str
            The point group in Schoenflies notation.
        """
        self._point_group = point_group
        self._matrices = (
            compute_Cartesian_Representation_matrix_for_a_given_point_group(point_group)
        )

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
        """The Cartesian Representation matrices for the point group .

        Returns
        -------
        list[np.ndarray]
            The Cartesian representation matrices for the operations in the point group.
        """
        return self._matrices

    @property
    def condensed_matrices(self) -> np.ndarray:
        """The condensed Cartesian Representation matrices for the point group .

        The matrices are flattend to give a 1D array

        Returns
        -------
        np.ndarray
            The condensed Cartesian Representation matrices for the point group .
        """
        return np.concatenate([matrix.flatten() for matrix in self._matrices])
