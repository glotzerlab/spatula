# ruff: noqa: F401
import pytest

import pgop.bond_order


@pytest.mark.parametrize(
    "module", [m for m in dir(pgop.bond_order) if m[:4] == "Bond"]
)
def test_imports(module):
    __import__("pgop." + module)
