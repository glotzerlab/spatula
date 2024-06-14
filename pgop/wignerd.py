import itertools
from typing import Generator

import numpy as np


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
    """Return the WignerD matrix for Cn up to the given l.

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


def Dn(max_l: int, n: int) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for Dn up to the given l.

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
    return (
        np.array(
            [
                (delta(mprime, m) + delta(mprime, -m) * (-1) ** l) * delta(m % n, 0)
                for l, mprime, m in iter_sph_indices(max_l)
            ],
            dtype=complex,
        )
        / 2
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
    return np.array(
        [delta(mprime, m) * ((-1) ** l) for l, mprime, m in iter_sph_indices(max_l)],
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
        [delta(mprime, -m) for _, mprime, m in iter_sph_indices(max_l)],
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
        [delta(mprime, -m) * ((-1) ** m) for _, mprime, m in iter_sph_indices(max_l)],
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
        [delta(mprime, -m) * ((-1) ** l) for l, mprime, m in iter_sph_indices(max_l)],
        dtype=complex,
    )


def two_y(max_l: int) -> np.ndarray:  # noqa: N802
    """Return the WignerD matrix for rotation for 180 around y up to the given l.

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
        [
            delta(mprime, -m) * ((-1) ** (m + l))
            for l, mprime, m in iter_sph_indices(max_l)
        ],
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


def convert_to_list_of_matrices(D: np.ndarray, max_l: int) -> list[np.ndarray]:  # noqa N802
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
    for i in range(max_l + 1):
        dimz = i * 2 + 1
        matrix = D[current : current + dimz**2]
        current += dimz**2
        matrices.append(matrix.reshape(dimz, dimz))
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


def direct_product(matrices_a: np.ndarray, matrices_b: np.ndarray) -> np.ndarray:  # noqa N802
    """Direct product of two sets of WignerD matrices.

    Parameters
    ----------
    matrices_a : np.ndarray
        The first set of WignerD matrices.
    matrices_b : np.ndarray
        The second set of WignerD matrices.

    Returns
    -------
    np.ndarray
        The direct product of the two sets of WignerD matrices.
    """
    result = []
    for matrix1, matrix2 in zip(matrices_a, matrices_b):
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


class WignerD:
    def __init__(self, point_group: str, max_l: int):
        """Create a WignerD object.

        Parameters
        ----------
        max_l : int
            The highest spherical harmonic to include in the WignerD matrices.
        """
        self._max_l = max_l
        self._point_group = point_group
        family, modifier, order = _parse_point_group(point_group)
        if family in "TOI":
            raise KeyError(f"{point_group} is not currently supported.")
            # if modifier == "h":
            #    self._matrix = semidirect_prod(????, Ci(max_l))
            # elif modifier is not None:
            #    raise KeyError(f"{point_group} is not currently supported.")
        if point_group == "Ci":
            self._matrix = Ci(max_l)
        if family == "C":
            if (modifier == "i" and order is not None) or (
                modifier == "h" and order % 2 == 0
            ):
                self._matrix = direct_product(Cn(max_l, order), Ci(max_l))
            elif modifier is None:
                self._matrix = Cn(max_l, order)
            elif modifier == "i" and order is None:
                self._matrix = Ci(max_l)
            else:
                raise KeyError(f"{point_group} is not currently supported.")
        if family == "D":
            if (modifier == "d" and order % 2 == 1) or (
                modifier == "h" and order % 2 == 0
            ):
                self._matrix = direct_product(Dn(max_l, order), Ci(max_l))
            elif modifier is None:
                self._matrix = Dn(max_l, order)
            else:
                raise KeyError(f"{point_group} is not currently supported.")

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
        return convert_to_list_of_matrices(self._matrix, self._max_l)

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
