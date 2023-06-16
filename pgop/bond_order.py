import pgop._pgop


class BondOrder:
    def __call__(self, positions):
        return self._cpp(positions)


class BondOrderFisher(BondOrder):
    def __init__(self, positions, kappa):
        self._cpp = pgop._pgop.FisherBondOrder(
            pgop._pgop.FisherDistribution(kappa), positions
        )


class BondOrderUniform(BondOrder):
    def __init__(self, positions, max_theta):
        self._cpp = pgop._pgop.UniformBondOrder(
            pgop._pgop.UniformDistribution(max_theta), positions
        )
