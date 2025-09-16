# Copyright (c) 2021-2025 The Regents of the University of Michigan
# Part of spatula, released under the BSD 3-Clause License.

"""Simply bond order diagram classes."""

import spatula._spatula


class BondOrder:
    """Base class for all bond order diagram classes.

    Provides a ``__call__`` method for computing the value of the BOD at a given
    position.
    """

    def __call__(self, positions):
        """Return the BOD for the provided positions.

        Parameters
        ----------
        positions : :math:`(N, 3)` numpy.ndarray of float
            The positions to compute the BOD for in Cartesian coordinates.

        """
        return self._cpp(positions)


class BondOrderFisher(BondOrder):
    """BOD where neighbors are marked by von-Mises Fisher distributions."""

    def __init__(self, positions, kappa):
        """Create a BondOrderFisher object.

        Parameters
        ----------
        positions : :math:`(N, 3)` numpy.ndarray of float
            The neighbor positions for the BOD.
        kappa : float
            The concentration parameter for the von-Mises Fisher distribution.

        """
        self._cpp = spatula._spatula.FisherBondOrder(
            spatula._spatula.FisherDistribution(kappa), positions
        )


class BondOrderUniform(BondOrder):
    """BOD where neighbors are marked by uniform distributions."""

    def __init__(self, positions, max_theta):
        """Create a BondOrderUniform object.

        Parameters
        ----------
        positions : :math:`(N, 3)` numpy.ndarray of float
            The neighbor positions for the BOD.
        max_theta : float
            The distance from the distribution center where the density is
            non-zero for the uniform distribution.

        """
        self._cpp = spatula._spatula.UniformBondOrder(
            spatula._spatula.UniformDistribution(max_theta), positions
        )
