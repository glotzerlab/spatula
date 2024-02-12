"""Simply bond order diagram classes."""

import pgop._pgop


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
        self._cpp = pgop._pgop.FisherBondOrder(
            pgop._pgop.FisherDistribution(kappa), positions
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
        self._cpp = pgop._pgop.UniformBondOrder(
            pgop._pgop.UniformDistribution(max_theta), positions
        )
