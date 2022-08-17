import numpy as np
import scipy as sp
import scipy.stats

from . import _pgop, util


class BondOrder:
    def __call__(self, positions):
        return self._cpp(positions)


class BondOrderFisher(BondOrder):
    def __init__(self, positions, kappa):
        self._cpp = _pgop.FisherBondOrder(
            _pgop.FisherDistribution(kappa), positions
        )


class BondOrderUniform(BondOrder):
    def __init__(self, positions, max_theta):
        self._cpp = _pgop.UniformBondOrder(
            _pgop.UniformDistribution(max_theta), positions
        )
