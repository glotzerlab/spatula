# Copyright (c) 2021-2026 The Regents of the University of Michigan
# Part of spatula, released under the BSD 3-Clause License.

"""Point group and symmetry operation representations."""

import itertools
from collections.abc import Generator

import numpy as np
import scipy.spatial
import scipy.special


def extract_first_element_of_hermann_mauguin_notation(s: str) -> tuple[str, int]:
    """Extract the elements according to Hermann-Mauguin notation.

    Extract the first element of a Hermann-Mauguin notation string, returning
    both the canonical form (e.g. "-10", "12/m", "m") AND the number of
    characters physically consumed from the start of `s`.

    This resolves issues with multi-digit axes and parentheses such as
    '(-10)', '(-12)', '(12)', etc.

    Returns
    -------
    (canonical_element, consumed_length)
        canonical_element : str
            The parsed chunk in canonical form (e.g. "-10", "12/m").
        consumed_length : int
            How many characters of `s` were used up (including parentheses).

    """
    if not s:
        raise ValueError("Invalid HM input: empty string")

    i, negative = 0, False

    if s[i] == "m":
        return ("m", 1)

    if s[i] == "-":
        negative, i = True, i + 1
        if i >= len(s):
            raise ValueError(f"Invalid HM input {s}, '-' with nothing after.")

    if i < len(s) and s[i] == "(":
        canonical, i = _parse_parentheses_block(s, i, negative)
    elif i < len(s) and s[i].isdigit():
        canonical, i = _parse_digits(s, i, negative)
    else:
        msg = "expected digits" if negative else "expected digits or '('"
        raise ValueError(f"Invalid HM input {s}, {msg} at start.")

    if np.abs(int(canonical)) > 21:
        canonical, i = _truncate_large_number(canonical)

    if i + 1 < len(s) and s[i] == "/" and s[i + 1] == "m":
        canonical += "/m"
        i += 2

    return (canonical, i)


def _parse_parentheses_block(s: str, i: int, negative: bool) -> tuple[str, int]:
    i += 1
    if s[i] == "-":
        negative, i = True, i + 1
    start_paren = i
    while i < len(s) and s[i] != ")":
        i += 1
    if i >= len(s):
        raise ValueError(f"Invalid HM input {s}, missing closing parenthesis.")
    inside = s[start_paren:i]
    i += 1
    if not inside.isdigit():
        raise ValueError(f"Invalid HM input {s}, parentheses must contain digits.")
    canonical = ("-" if negative else "") + inside
    return canonical, i


def _parse_digits(s: str, i: int, negative: bool) -> tuple[str, int]:
    start_digits = i
    while i < len(s) and s[i].isdigit():
        i += 1
    digit_str = s[start_digits:i]
    canonical = ("-" if negative else "") + digit_str
    return canonical, i


def _truncate_large_number(canonical: str) -> tuple[str, int]:
    if canonical[0] == "-":
        new_canonical = canonical[:2]
        i = 2
    else:
        new_canonical = canonical[:1]
        i = 1
    return new_canonical, i


def extract_elements_from_of_hermann_mauguin_notation(
    point_group: str,
) -> tuple[str, str | None, str | None]:
    """Extract the elements of a Hermann-Mauguin notation string.

    Short and full symbols are supported.

    Parameters
    ----------
    point_group : str
        The Hermann-Mauguin notation string.

    Returns
    -------
    list[str, str|None, str|None]
        The extracted three elements from the point group string in
        Hermann-Mauguin notation.

    """
    first, used1 = extract_first_element_of_hermann_mauguin_notation(point_group)
    remainder = point_group[used1:]
    second = None
    third = None
    if remainder:
        second, used2 = extract_first_element_of_hermann_mauguin_notation(remainder)
        remainder = remainder[used2:]
        if remainder:
            third, used3 = extract_first_element_of_hermann_mauguin_notation(remainder)
            if used3 != len(remainder):
                raise ValueError(
                    f"Invalid HM input {point_group}, {first}, {second}, {third}"
                )
    return first, second, third


def convert_hermann_mauguin_to_schonflies(point_group: str) -> str:  # noqa N802
    """Convert a Hermann-Mauguin notation to a Schönflies notation.

    Both short and full Hermann-Mauguin notation is supported. Supported point
    groups are of finite order, including polyhedral groups.

    Parameters
    ----------
    point_group : str
        The point group in Hermann-Mauguin notation.

    Returns
    -------
    str
        The point group in Schönflies notation.

    """
    # convert from Hermann-Mauguin notation to schonflies
    # remove dots and spaces:
    point_group = point_group.replace(".", "")
    point_group = point_group.replace(" ", "")

    first, second, third = extract_elements_from_of_hermann_mauguin_notation(
        point_group
    )
    # Cn is if second and third are none and first does not contain m, and first
    # is a positive number larger than zero
    if second is None and third is None:
        if first.isnumeric():
            if int(first) > 0:
                point_group = "C" + first
        elif first[0] == "-" and first[1:].isnumeric():
            if int(first[1:]) > 2:
                if int(first[1:]) % 4 == 2:
                    point_group = f"C{int(first[1:]) // 2}h"
                elif int(first[1:]) % 2 == 1:
                    point_group = f"S{2 * int(first[1:])}"
                elif int(first[1:]) % 2 == 0:
                    point_group = "S" + first[1:]
            elif int(first[1:]) == 1:
                point_group = "Ci"
            else:
                raise ValueError(
                    f"Invalid HM input {point_group}; {first} {second} {third}"
                )
        elif first.endswith("/m"):
            if int(first[:-2]) > 1:
                point_group = "C" + first[:-2] + "h"
            elif int(first[:-2]) == 1:
                point_group = "Cs"
            else:
                raise ValueError(
                    f"Invalid HM input {point_group}; {first} {second} {third}"
                )
        elif first == "m":
            point_group = "Cs"
        else:
            raise ValueError(
                f"Invalid HM input {point_group}; {first} {second} {third}"
            )

    # if first element is a positive number followed by a m or mm its Cnv
    elif first.isnumeric() and second == "m" and (third is None or third == "m"):
        if int(first) > 1:
            point_group = "C" + first + "v"
        else:
            raise ValueError(
                f"Invalid HM input {point_group}; {first} {second} {third}"
            )
    # Dn is nmm
    elif first.isnumeric() and second == "2" and (third is None or third == "2"):
        if int(first) > 1:
            point_group = "D" + first
        else:
            raise ValueError(
                f"Invalid HM input {point_group}; {first} {second} {third}"
            )
    # D2d is also -4m2
    elif first == "-4" and second == "m" and third == "2":
        point_group = "D2d"
    # D2 is also mm2
    elif first == "m" and second == "m" and third == "2":
        point_group = "C2v"
    # Dn/2h is -nm2 for n=6,10,14.. this is Dnh for odd n
    elif second == "m" and third == "2" and first[0] == "-" and first[1:].isnumeric():
        if int(first[1:]) > 2 and int(first[1:]) % 4 == 2:
            point_group = f"D{int(first[1:]) // 2}h"
        else:
            raise ValueError(
                f"Invalid HM input {point_group}; {first} {second} {third}"
            )
    # Dnh n/m2/m2/m which is eq to n/mmm
    elif (
        first.endswith("/m")
        and first[0].isdigit()
        and ((second == "m" or second == "2/m") and (third == "m" or third == "2/m"))
    ):
        if int(first[:-2]) > 1:
            point_group = "D" + first[:-2] + "h"
        else:
            raise ValueError(
                f"Invalid HM input {point_group}; {first} {second} {third}"
            )
    # D2h is mmm also
    elif first == "m" and second == "m" and third == "m":
        point_group = "D2h"
    # D3h is also -62m or -6m2
    elif (
        first == "-6"
        and second == "m"
        and third == "2"
        or first == "-6"
        and second == "2"
        and third == "m"
    ):
        point_group = "D3h"
    # Dn/2d is -n2m for n=4,8,12,...
    elif second == "2" and third == "m" and first[0] == "-" and int(first[1:]) % 4 == 0:
        if int(first[1:]) > 2:
            point_group = f"D{int(first[1:]) // 2}d"
        else:
            raise ValueError(
                f"Invalid HM input {point_group}; {first} {second} {third}"
            )
    # Dnd is -n2/m or -nm for n = 3,5,7,9
    elif (
        (second == "2/m" or second == "m") and first[0] == "-" and first[1:].isnumeric()
    ):
        if int(first[1:]) > 1 and int(first[1:]) % 2 == 1:
            point_group = "D" + first[1:] + "d"
        else:
            raise ValueError(
                f"Invalid HM input {point_group}; {first} {second} {third}"
            )
    # group T 23
    elif first == "2" and second == "3" and third is None:
        point_group = "T"
    # group Td -43m
    elif first == "-4" and second == "3" and third == "m":
        point_group = "Td"
    # group Th 2/m-3 or m-3 or m3
    elif (
        second == "-3"
        and (first == "m" or first == "2/m")
        and third is None
        or first == "m"
        and second == "3"
        and third is None
    ):
        point_group = "Th"
    # group O 432
    elif first == "4" and second == "3" and third == "2":
        point_group = "O"
    # group Oh 4/m-32/m or m-3m or m3m
    elif (
        (
            second == "-3"
            and (first == "m" or first == "4/m")
            and (third == "m" or third == "2/m")
        )
        or first == "m"
        and second == "3"
        and third == "m"
    ):
        point_group = "Oh"
    # group I  is 235 or 25 or 532 or 53
    elif (
        first == "2"
        and ((second == "3" and third == "5") or (second == "5" and third is None))
        or (first == "5" and second == "3" and (third is None or third == "2"))
    ):
        point_group = "I"
    # group Ih is m-3-5 or m -5 or m-5m or -5-3m
    elif (
        first == "m"
        and (
            (second == "-5" and (third == "m" or third is None))
            or (second == "-3" and third == "-5")
        )
    ) or (first == "-5" and second == "-3" and third == "m"):
        point_group = "Ih"
    else:
        raise ValueError(f"Invalid HM input {point_group}, {first}, {second}, {third}")
    return point_group


def identity_cart() -> np.ndarray:  # noqa: N802
    """Return the Cartesian Representation matrix for E (identity).

    Returns
    -------
    np.ndarray
        The  Cartesian Representation matrix for E.

    """
    return np.eye(3)


def inversion_cart() -> np.ndarray:  # noqa: N802
    """Return the  Cartesian Representation matrix for inversion.

    Returns
    -------
    np.ndarray
        The Cartesian Representation matrix for inversion .

    """
    return -1 * np.eye(3)


def rotation_from_euler_angles_cart(
    alpha: float, beta: float, gamma: float
) -> np.ndarray:
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


def rotation_from_axis_angle_cart(axis: np.ndarray, angle: float) -> np.ndarray:
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


def rotation_from_axis_order_cart(axis: np.ndarray, order: int) -> np.ndarray:
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
    return rotation_from_axis_angle_cart(axis, 2 * np.pi / order)


def rotoreflection_from_euler_angles_cart(
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
    rotation_operator = rotation_from_euler_angles_cart(alpha, beta, gamma)
    # find axis of rotation
    rotation_axis = scipy.spatial.transform.Rotation.from_euler(
        "zyz", [alpha, beta, gamma]
    ).as_rotvec()
    # normalize rotation axis
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    rotoreflection_operator = reflection_from_normal_cart(rotation_axis)
    return np.dot(rotoreflection_operator, rotation_operator)


def rotoreflection_from_axis_angle_cart(axis: np.ndarray, angle: float) -> np.ndarray:
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
    return rotoreflection_from_euler_angles_cart(*rotoreflection_euler)


def rotoreflection_from_axis_order_cart(axis: np.ndarray, order: int) -> np.ndarray:
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
    return rotoreflection_from_axis_angle_cart(axis, 2 * np.pi / order)


def reflection_from_normal_cart(normal: np.ndarray):
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
    inversion_operator = inversion_cart()
    rotation_operator = rotation_from_axis_angle_cart(normal, np.pi)
    return np.dot(inversion_operator, rotation_operator)


def _cs_operations_cart() -> np.ndarray:
    """Return the Cartesian Representation matrix for Cs group.

    Cs={E, sigma_yz}

    Returns
    -------
    list[np.ndarray]
        The operations list for Cs.

    """
    return [identity_cart(), reflection_from_normal_cart([1, 0, 0])]


def _ch_operations_cart() -> np.ndarray:
    """Return the operations list for Ch group.

    Ch={E, sigma_xy}

    Returns
    -------
    list[np.ndarray]
        The operations list for Ch.

    """
    return [identity_cart(), reflection_from_normal_cart([0, 0, 1])]


def _ci_operations_cart() -> np.ndarray:
    """Return the operations list for Ci.

    Ci={E, i}

    Returns
    -------
    list[np.ndarray]
        The operations list for Ci.

    """
    return [identity_cart(), inversion_cart()]


def _cn_operations_cart(n: int) -> np.ndarray:
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
    operations = [identity_cart()]
    rotation_operation = rotation_from_euler_angles_cart(2 * np.pi / n, 0, 0)
    for _ in range(1, n):
        operations.append(np.dot(operations[-1], rotation_operation))
    return operations


def _sn_operations_cart(n: int) -> np.ndarray:
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
    identity_operation = identity_cart()
    operations = [identity_operation]
    rotoreflection_operation = rotoreflection_from_euler_angles_cart(
        0, 0, 2 * np.pi / n
    )
    if n % 2 == 0:
        for _ in range(1, n):
            operations.append(np.dot(operations[-1], rotoreflection_operation))
    else:
        for _ in range(1, 2 * n):
            operations.append(np.dot(operations[-1], rotoreflection_operation))
    return operations


def _cnh_operations_cart(n: int) -> np.ndarray:
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
    operations = _cn_operations_cart(n)
    sigma_h_operation = reflection_from_normal_cart([0, 0, 1])
    for operation in operations.copy():
        operations.append(np.dot(operation, sigma_h_operation))
    return operations


def _cnv_operations_cart(n: int) -> np.ndarray:
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
    operations = _cn_operations_cart(n)
    sigma_v_operation = reflection_from_normal_cart([1, 0, 0])
    for operation in operations.copy():
        operations.append(np.dot(operation, sigma_v_operation))
    return operations


def _dn_operations_cart(n: int) -> np.ndarray:
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
    operations = _cn_operations_cart(n)
    xy_rotation = np.dot(
        reflection_from_normal_cart([0, 0, 1]),
        reflection_from_normal_cart([1, 0, 0]),
    )
    for operation in operations.copy():
        operations.append(np.dot(operation, xy_rotation))
    return operations


def _dnh_operations_cart(n: int) -> np.ndarray:
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
    operations = _dn_operations_cart(n)
    identity_operation = identity_cart()
    sigma_h_operation = reflection_from_normal_cart([0, 0, 1])
    rotation_operation = rotation_from_euler_angles_cart(0, 0, 2 * np.pi / n)
    for i in range(n, 2 * n):
        c2prime_operation = operations[i]
        operations.append(np.dot(sigma_h_operation, c2prime_operation))
    operations.append(sigma_h_operation)
    for i in range(1, n):
        final_operation = identity_operation
        for _ in range(0, i):
            final_operation = np.dot(final_operation, rotation_operation)
        final_operation = np.dot(final_operation, sigma_h_operation)
        operations.append(final_operation)
    return operations


def _dnd_operations_cart(n: int) -> np.ndarray:
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
    operations = _sn_operations_cart(2 * n)
    vertical_reflection = reflection_from_normal_cart([0, 1, 0])
    for operation in operations.copy():
        operations.append(np.dot(operation, vertical_reflection))
    return operations


def _rotation_operations_for_polyhedral_point_groups_cart(
    point_group: str,
) -> np.ndarray:
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
        operations.append(rotation_from_euler_angles_cart(*rot))
    # swap first and 3rd element so that identity is first
    operations[0], operations[3] = operations[3], operations[0]
    return operations


def compute_Cartesian_Representation_for_C_family(  # noqa N802
    modifier: str, order: int
) -> list[np.ndarray]:
    """Compute the Cartesian Representation matrices for a given C family.

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
        return _cnh_operations_cart(order)
    elif modifier == "v" and order is not None:
        return _cnv_operations_cart(order)
    elif modifier is None and order is not None:
        return _cn_operations_cart(order)
    elif modifier == "i" and order is None:
        return _ci_operations_cart()
    elif modifier == "h" and order is None:
        return _ch_operations_cart()
    elif modifier == "s" and order is None:
        return _cs_operations_cart()
    else:
        return None


def compute_Cartesian_Representation_for_D_family(  # noqa N802
    modifier: str, order: int
) -> list[np.ndarray]:
    """Compute the Cartesian Representation matrices for a given D family.

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
        return _dnd_operations_cart(order)
    elif modifier == "h" and order is not None:
        return _dnh_operations_cart(order)
    elif modifier is None and order is not None:
        return _dn_operations_cart(order)
    else:
        return None


def compute_Cartesian_Representation_for_S_family(  # noqa N802
    modifier: str, order: int
) -> list[np.ndarray]:
    """Compute the Cartesian Representation matrices for a given S family.

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
        return _sn_operations_cart(order)
    else:
        return None


def compute_Cartesian_Representation_for_tetrahedral_family(  # noqa N802
    modifier: str,
) -> list[np.ndarray]:
    """Compute the Cartesian Representation matrices for a given terahedral family.

    Parameters
    ----------
    modifier : str
        The modifier for the point group.

    Returns
    -------
    np.ndarray
        The Cartesian Representation matrices for the point group.

    """
    operations = _rotation_operations_for_polyhedral_point_groups_cart("T")
    if modifier == "d":
        # 6 S4
        # 90 degrees around 1, 0, 0
        first_S4 = rotoreflection_from_axis_order_cart([0, 0, 1], 4)  # noqa N806
        # 90 degrees around 0, 1, 0
        second_S4 = rotoreflection_from_axis_order_cart([0, 1, 0], 4)  # noqa N806
        # 90 degrees around 0, 0, 1
        third_S4 = rotoreflection_from_axis_order_cart([1, 0, 0], 4)  # noqa N806
        # 270 degrees around 1, 0, 0
        # 270 degrees around 0, 1, 0
        # 270 degrees around 0, 0, 1
        fourth_S4 = identity_cart()  # noqa N806
        fifth_S4 = identity_cart()  # noqa N806
        sixth_S4 = identity_cart()  # noqa N806
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
        operations.append(
            reflection_from_normal_cart([0, -np.sqrt(2) / 2, -np.sqrt(2) / 2])
        )
        operations.append(
            reflection_from_normal_cart([0, np.sqrt(2) / 2, -np.sqrt(2) / 2])
        )
        operations.append(
            reflection_from_normal_cart([-np.sqrt(2) / 2, 0, -np.sqrt(2) / 2])
        )
        operations.append(
            reflection_from_normal_cart([np.sqrt(2) / 2, 0, -np.sqrt(2) / 2])
        )
        operations.append(
            reflection_from_normal_cart([-np.sqrt(2) / 2, -np.sqrt(2) / 2, 0])
        )
        operations.append(
            reflection_from_normal_cart([np.sqrt(2) / 2, -np.sqrt(2) / 2, 0])
        )
        return operations
    elif modifier == "h":
        inversion_operation = inversion_cart()
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
    """Compute the Cartesian Representation matrices for a given octahedral family.

    Parameters
    ----------
    modifier : str
        The modifier for the point group.

    Returns
    -------
    list[np.ndarray]
        The Cartesian Representation matrices for the point group.

    """
    operations = _rotation_operations_for_polyhedral_point_groups_cart("O")

    if modifier == "h":
        inversion_operation = inversion_cart()
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
    """Compute the Cartesian Representation matrix for a given icosahedral family.

    Parameters
    ----------
    modifier : str
        The modifier for the point group.

    Returns
    -------
    list[np.ndarray]
        The Cartesian Representation matrices for the point group.

    """
    operations = _rotation_operations_for_polyhedral_point_groups_cart("I")
    if modifier == "h":
        inversion_operation = inversion_cart()
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
    """Compute the Cartesian Representation matrices for a given point group.

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


def identity_sph(max_l: int) -> np.ndarray:  # noqa: N802
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


def inversion_sph(max_l: int) -> np.ndarray:  # noqa: N802
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


def rotation_from_euler_angles_sph(
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


def rotation_from_axis_angle_sph(
    max_l: int, axis: np.ndarray, angle: float
) -> np.ndarray:  # noqa: N802
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
    return rotation_from_euler_angles_sph(max_l, *rotation_euler)


def rotation_from_axis_order_sph(
    max_l: int, axis: np.ndarray, order: int
) -> np.ndarray:  # noqa: N802
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
    return rotation_from_axis_angle_sph(max_l, axis, 2 * np.pi / order)


def rotoreflection_from_euler_angles_sph(
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
    rotation_operator = rotation_from_euler_angles_sph(max_l, alpha, beta, gamma)
    # find axis of rotation
    rotation_axis = scipy.spatial.transform.Rotation.from_euler(
        "zyz", [alpha, beta, gamma]
    ).as_rotvec()
    # normalize rotation axis
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    rotoreflection_operator = reflection_from_normal_sph(max_l, rotation_axis)
    return dot_product(rotoreflection_operator, rotation_operator)


def rotoreflection_from_axis_angle_sph(
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
    return rotoreflection_from_euler_angles_sph(max_l, *rotoreflection_euler)


def rotoreflection_from_axis_order_sph(
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
    return rotoreflection_from_axis_angle_sph(max_l, axis, 2 * np.pi / order)


def reflection_from_normal_sph(max_l: int, normal: np.ndarray):
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
    inversion_operator = inversion_sph(max_l)
    rotation_operator = rotation_from_axis_angle_sph(max_l, normal, np.pi)
    return dot_product(inversion_operator, rotation_operator)


def _cs_operations_sph(max_l: int) -> np.ndarray:
    """Return the WignerD matrix for Cs group up to the given l.

    Cs={E, sigma_yz}

    Parameters
    ----------
    max_l : int
        The maximum l value to include.

    Returns
    -------
    list[np.ndarray]
        The operations list for Cs up to the given l.

    """
    return [identity_sph(max_l), reflection_from_normal_sph(max_l, [1, 0, 0])]


def _ch_operations_sph(max_l: int) -> np.ndarray:
    """Return the operations list for Ch group up to the given l.

    Ch={E, sigma_xy}

    Parameters
    ----------
    max_l : int
        The maximum l value to include.

    Returns
    -------
    list[np.ndarray]
        The operations list for Ch up to the given l.

    """
    return [identity_sph(max_l), reflection_from_normal_sph(max_l, [0, 0, 1])]


def _ci_operations_sph(max_l: int) -> np.ndarray:
    """Return the operations list for Ci up to the given l.

    Ci={E, i}

    Parameters
    ----------
    max_l : int
        The maximum l value to include.

    Returns
    -------
    list[np.ndarray]
        The operations list for Ci up to the given l.

    """
    return [identity_sph(max_l), inversion_sph(max_l)]


def _cn_operations_sph(max_l: int, n: int) -> np.ndarray:
    """Return the operations list for Cn (Cyclic group) up to the given l.

    Elements of Cn are given by Cn = {E, Cn, Cn**2, Cn**3, ... Cn**(n-1)}.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.
    n : int
        The order of the cyclic group.

    Returns
    -------
    list[np.ndarray]
        The operations list for Cn up to the given l.

    """
    operations = [identity_sph(max_l)]
    rotation_operation = rotation_from_euler_angles_sph(max_l, 2 * np.pi / n, 0, 0)
    for _ in range(1, n):
        operations.append(dot_product(operations[-1], rotation_operation))
    return operations


def _sn_operations_sph(max_l: int, n: int) -> np.ndarray:
    """Return the operations list for Sn (rotoreflection group) up to the given l.

    Elements of Sn are given by Sn = {E, Sn, Sn**2, Sn**3, ... Sn**(n-1)} for even n and
    Sn = {E, Sn, Sn**3, Sn**5, ... Sn**(2n-1), Cn, Cn**2, ... Cn**(n-1)} for odd n. Sn
    is a rotation by 2*pi/n followed by a reflection.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.
    n : int
        The order of the rotation group.

    Returns
    -------
    list[np.ndarray]
        The operations list for Sn up to the given l.

    """
    identity_operation = identity_sph(max_l)
    operations = [identity_operation]
    rotoreflection_operation = rotoreflection_from_euler_angles_sph(
        max_l, 0, 0, 2 * np.pi / n
    )
    if n % 2 == 0:
        for _ in range(1, n):
            operations.append(dot_product(operations[-1], rotoreflection_operation))
    else:
        for _ in range(1, 2 * n):
            operations.append(dot_product(operations[-1], rotoreflection_operation))
    return operations


def _cnh_operations_sph(max_l: int, n: int) -> np.ndarray:
    """Return the operations list for Cnh group up to the given l.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.
    n : int
        The order of the cyclic group.

    Returns
    -------
    list[np.ndarray]
        The operations list for Cnh up to the given l.

    """
    operations = _cn_operations_sph(max_l, n)
    sigma_h_operation = reflection_from_normal_sph(max_l, [0, 0, 1])
    for operation in operations.copy():
        operations.append(dot_product(operation, sigma_h_operation))
    return operations


def _cnv_operations_sph(max_l: int, n: int) -> np.ndarray:
    """Return the operations list for Cnv group up to the given l.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.
    n : int
        The order of the cyclic group.

    Returns
    -------
    list[np.ndarray]
        The operations list for Cnv up to the given l.

    """
    operations = _cn_operations_sph(max_l, n)
    sigma_v_operation = reflection_from_normal_sph(max_l, [1, 0, 0])
    for operation in operations.copy():
        operations.append(dot_product(operation, sigma_v_operation))
    return operations


def _dn_operations_sph(max_l: int, n: int) -> np.ndarray:
    """Return the operations list for Dn (Dihedral group) up to the given l.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.
    n : int
        The order of the dihedral group.

    Returns
    -------
    list[np.ndarray]
        The operations list for Dn up to the given l.

    """
    operations = _cn_operations_sph(max_l, n)
    xy_rotation = dot_product(
        reflection_from_normal_sph(max_l, [0, 0, 1]),
        reflection_from_normal_sph(max_l, [1, 0, 0]),
    )
    for operation in operations.copy():
        operations.append(dot_product(operation, xy_rotation))
    return operations


def _dnh_operations_sph(max_l: int, n: int) -> np.ndarray:
    """Return the operations list for Dnh group up to the given l.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.
    n : int
        The order of the dihedral group.

    Returns
    -------
    list[np.ndarray]
        The operations list for Dnh up to the given l.

    """
    operations = _dn_operations_sph(max_l, n)
    identity_operation = identity_sph(max_l)
    sigma_h_operation = reflection_from_normal_sph(max_l, [0, 0, 1])
    rotation_operation = rotation_from_euler_angles_sph(max_l, 0, 0, 2 * np.pi / n)
    for i in range(n, 2 * n):
        c2prime_operation = operations[i]
        operations.append(dot_product(sigma_h_operation, c2prime_operation))
    operations.append(sigma_h_operation)
    for i in range(1, n):
        final_operation = identity_operation
        for _ in range(0, i):
            final_operation = dot_product(final_operation, rotation_operation)
        final_operation = dot_product(final_operation, sigma_h_operation)
        operations.append(final_operation)
    return operations


def _dnd_operations_sph(max_l: int, n: int) -> np.ndarray:
    """Return the operations list for Dnd group up to the given l.

    Parameters
    ----------
    max_l : int
        The maximum l value to include.
    n : int
        The order of the dihedral group.

    Returns
    -------
    list[np.ndarray]
        The operations list for Dnd up to the given l.

    """
    operations = _sn_operations_sph(max_l, 2 * n)
    vertical_reflection = reflection_from_normal_sph(max_l, [0, 1, 0])
    for operation in operations.copy():
        operations.append(dot_product(operation, vertical_reflection))
    return operations


def _rotation_operations_for_polyhedral_point_groups_sph(
    point_group: str, max_l: int
) -> np.ndarray:
    """Return the operations list for a given polyhedral point group.

    Parameters
    ----------
    point_group : str
        The point group for which the operations are to be computed.
    max_l : int
        The maximum l value to include.

    Returns
    -------
    list[np.ndarray]
        The operations list for the given polyhedral point group up to the given l.

    """
    operations = []
    for i in scipy.spatial.transform.Rotation.create_group(point_group):
        rot = i.as_euler("zyz")
        operations.append(rotation_from_euler_angles_sph(max_l, *rot))
    return operations


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
    1 if a == b, 0 otherwise.

    """
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
    """Compute the condensed WignerD matrix for a given C family.

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
        return condensed_wignerD_from_operations(_cnh_operations_sph(max_l, order))
    elif modifier == "v" and order is not None:
        return condensed_wignerD_from_operations(_cnv_operations_sph(max_l, order))
    elif modifier is None and order is not None:
        return condensed_wignerD_from_operations(_cn_operations_sph(max_l, order))
    elif modifier == "i" and order is None:
        return condensed_wignerD_from_operations(_ci_operations_sph(max_l))
    elif modifier == "h" and order is None:
        return condensed_wignerD_from_operations(_ch_operations_sph(max_l))
    elif modifier == "s" and order is None:
        return condensed_wignerD_from_operations(_cs_operations_sph(max_l))
    else:
        return None


def compute_condensed_wignerD_for_D_family(  # noqa N802
    max_l: int, modifier: str, order: int
) -> np.ndarray:
    """Compute the condensed WignerD matrix for a given D family.

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
        return condensed_wignerD_from_operations(_dnd_operations_sph(max_l, order))
    elif modifier == "h" and order is not None:
        return condensed_wignerD_from_operations(_dnh_operations_sph(max_l, order))
    elif modifier is None and order is not None:
        return condensed_wignerD_from_operations(_dn_operations_sph(max_l, order))
    else:
        return None


def compute_condensed_wignerD_for_S_family(  # noqa N802
    max_l: int, modifier: str, order: int
) -> np.ndarray:
    """Compute the condensed WignerD matrix for a given S family.

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
        return condensed_wignerD_from_operations(_sn_operations_sph(max_l, order))
    else:
        return None


def compute_condensed_wignerD_for_tetrahedral_family(  # noqa N802
    max_l: int, modifier: str
) -> np.ndarray:
    """Compute the condensed WignerD matrix for a given terahedral family.

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
    operations = _rotation_operations_for_polyhedral_point_groups_sph("T", max_l)
    if modifier == "d":
        # 6 S4
        # 90 degrees around 1, 0, 0
        first_S4 = rotoreflection_from_axis_order_sph(max_l, [0, 0, 1], 4)  # noqa N806
        # 90 degrees around 0, 1, 0
        second_S4 = rotoreflection_from_axis_order_sph(max_l, [0, 1, 0], 4)  # noqa N806
        # 90 degrees around 0, 0, 1
        third_S4 = rotoreflection_from_axis_order_sph(max_l, [1, 0, 0], 4)  # noqa N806
        # 270 degrees around 1, 0, 0
        # 270 degrees around 0, 1, 0
        # 270 degrees around 0, 0, 1
        fourth_S4 = identity_sph(max_l)  # noqa N806
        fifth_S4 = identity_sph(max_l)  # noqa N806
        sixth_S4 = identity_sph(max_l)  # noqa N806
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
            reflection_from_normal_sph(max_l, [0, -np.sqrt(2) / 2, -np.sqrt(2) / 2])
        )
        operations.append(
            reflection_from_normal_sph(max_l, [0, np.sqrt(2) / 2, -np.sqrt(2) / 2])
        )
        operations.append(
            reflection_from_normal_sph(max_l, [-np.sqrt(2) / 2, 0, -np.sqrt(2) / 2])
        )
        operations.append(
            reflection_from_normal_sph(max_l, [np.sqrt(2) / 2, 0, -np.sqrt(2) / 2])
        )
        operations.append(
            reflection_from_normal_sph(max_l, [-np.sqrt(2) / 2, -np.sqrt(2) / 2, 0])
        )
        operations.append(
            reflection_from_normal_sph(max_l, [np.sqrt(2) / 2, -np.sqrt(2) / 2, 0])
        )
        return condensed_wignerD_from_operations(operations)
    elif modifier == "h":
        inversion_operation = inversion_sph(max_l)
        for operation in operations.copy():
            operations.append(dot_product(operation, inversion_operation))
        return condensed_wignerD_from_operations(operations)
    elif modifier is None:
        return condensed_wignerD_from_operations(operations)
    else:
        return None


def compute_condensed_wignerD_for_octahedral_family(  # noqa N802
    max_l: int, modifier: str
) -> np.ndarray:
    """Compute the condensed WignerD matrix for a given octahedral family.

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
    operations = _rotation_operations_for_polyhedral_point_groups_sph("O", max_l)
    if modifier == "h":
        inversion_operation = inversion_sph(max_l)
        for operation in operations.copy():
            operations.append(dot_product(operation, inversion_operation))
        return condensed_wignerD_from_operations(operations)
    elif modifier is None:
        return condensed_wignerD_from_operations(operations)
    else:
        return None


def compute_condensed_wignerD_for_icosahedral_family(  # noqa N802
    max_l: int, modifier: str
) -> np.ndarray:
    """Compute the condensed WignerD matrix for a given icosahedral family.

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
    operations = _rotation_operations_for_polyhedral_point_groups_sph("I", max_l)
    if modifier == "h":
        inversion_operation = inversion_sph(max_l)
        for operation in operations.copy():
            operations.append(dot_product(operation, inversion_operation))
        return condensed_wignerD_from_operations(operations)
    elif modifier is None:
        return condensed_wignerD_from_operations(operations)
    else:
        return None


def compute_condensed_wignerD_matrix_for_a_given_point_group(  # noqa N802
    point_group: str, max_l: int
) -> np.ndarray:
    """Compute the condensed WignerD matrix for a given point group.

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


class CartesianRepMatrix:
    """A class to represent a point group operations in Cartesian coordinates."""

    def __init__(self, point_group: str):
        """Create a Cartesian Representation of a Point Group.

        Parameters
        ----------
        point_group : str
            The point group in Schoenflies notation.

        """
        if (
            "C" in point_group
            or "D" in point_group
            or "S" in point_group
            or "T" in point_group
            or "O" in point_group
            or "I" in point_group
        ):
            pass
        elif any(char.isdigit() or char in {"m", "-", "/"} for char in point_group):
            point_group = convert_hermann_mauguin_to_schonflies(point_group)
        else:
            raise KeyError(f"Unknown point group {point_group}.")
        self._point_group = point_group
        self._matrices = (
            compute_Cartesian_Representation_matrix_for_a_given_point_group(point_group)
        )

    def __repr__(self):
        """Represent the matrices in string form."""
        return f"{np.array(self.matrices).round(8)}\nShape: {np.shape(self.matrices)}"

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
        return np.asarray([matrix.flatten() for matrix in self._matrices]).flatten()


class WignerD:
    """A class to represent a WignerD matrix for a given point group.

    WignerD matrices are repsentations of symmetry groups in spherical harmonics.
    """

    def __init__(self, point_group: str, max_l: int) -> None:
        """Create a WignerD object.

        Parameters
        ----------
        point_group : str
            The point group in Schoenflies notation or Hermann-Mauguin notation
            (both short and full symbols are supported). Strings are case sensitive.
            Schoenflies notation must contain an uppercase letter (C, D, S, T, O, I);
            any subsequent letters must be lowercase (h, d, i). Hermann-Mauguin
            notation must be entirely lowercase.

        max_l : int
            The highest spherical harmonic to include in the WignerD matrices.

        """
        self._max_l = max_l
        if (
            "C" in point_group
            or "D" in point_group
            or "S" in point_group
            or "T" in point_group
            or "O" in point_group
            or "I" in point_group
        ):
            pass
        elif any(char.isdigit() or char in {"m", "-", "/"} for char in point_group):
            point_group = convert_hermann_mauguin_to_schonflies(point_group)
        else:
            raise KeyError(f"Unknown point group {point_group}.")

        self._point_group = point_group
        self._matrix = compute_condensed_wignerD_matrix_for_a_given_point_group(
            point_group, max_l
        )

    @property
    def max_l(self) -> int:
        """The maximum l value used for constructing the matrix.

        Returns
        -------
        int
        The maximum l value.

        """
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
