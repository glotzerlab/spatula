# ruff: noqa: F401
import pytest

import pgop.bond_order


@pytest.mark.parametrize(
    "module", [m for m in dir(pgop.bond_order) if m[:4] == "Bond"]
)
def test_constructor(module):
    bond_order = getattr(pgop.bond_order, module)
    distribution = module[9:]

    positions = [[0, 1, 0]]

    bond_order() if distribution == "" else bond_order(positions, 0.1)
