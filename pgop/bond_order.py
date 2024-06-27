"""Simply bond order diagram classes."""

# TODO: We should have a constructor that takes a distribution, and allows for
# later calls to pass in the positions and weights.

import numpy as np

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

    def __init__(self, positions, kappa, weights=1.0):
        """Create a BondOrderFisher object.

        Parameters
        ----------
        positions : :math:`(N, 3)` numpy.ndarray of float
            The neighbor positions for the BOD.
        kappa : float
            The concentration parameter for the von-Mises Fisher distribution.
        weights : float or np.ndarray[float]
            Weights for each point. If a scalar is passed, it is expanded to
            len(positions). Default value = 1.
        """
        if not hasattr(weights, "__len__"):
            weights = np.full(shape=len(positions), fill_value=weights)

        self._cpp = pgop._pgop.BondOrderFisher(
            pgop._pgop.FisherDistribution(kappa),
            # TODO: this is dumb.
            # Correct code should be np.asarray(positions,dtype=np.float64)
            [pgop._pgop.Vec3(*row) for row in positions],
            weights,
        )


class BondOrderUniform(BondOrder):
    """BOD where neighbors are marked by uniform distributions."""

    def __init__(self, positions, max_theta, weights=1.0):
        """Create a BondOrderUniform object.

        Parameters
        ----------
        positions : :math:`(N, 3)` numpy.ndarray of float
            The neighbor positions for the BOD.
        max_theta : float
            The distance from the distribution center where the density is
            non-zero for the uniform distribution.
        weights : float or np.ndarray[float]
            Weights for each point. If a scalar is passed, it is expanded to
            len(positions). Default value = 1.
        """
        if not hasattr(weights, "__len__"):
            weights = np.full(shape=len(positions), fill_value=weights)

        self._cpp = pgop._pgop.BondOrderUniform(
            pgop._pgop.UniformDistribution(max_theta),
            [pgop._pgop.Vec3(*row) for row in positions],
            weights,
        )
