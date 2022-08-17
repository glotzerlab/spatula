import numpy as np
import scipy as sp
import scipy.stats

from . import _pgop, util


class BondOrder:
    def __call__(self, theta, phi):
        return self._cpp(theta, phi)


class BondOrderFisher(BondOrder):
    def __init__(self, theta, phi, kappa):
        self._cpp = _pgop.FisherBondOrder(
            _pgop.FisherDistribution(kappa), theta, phi
        )


class BondOrderUniform(BondOrder):
    def __init__(self, theta, phi, max_theta):
        self._cpp = _pgop.UniformBondOrder(
            _pgop.UniformDistribution(max_theta), theta, phi
        )
