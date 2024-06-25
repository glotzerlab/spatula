import numpy as np
import pytest

from pgop.wignerd import WignerD, _parse_point_group, _WignerData


def test_init():
    data = _WignerData()
    assert len(data._columns) > 0
    assert len(data._data) > 0


def test_getitem():
    data = _WignerData()
    key = data._columns[0]
    item = data[key]
    assert isinstance(item, np.ndarray)


def test_len():
    data = _WignerData()
    assert len(data) == len(data._columns)


def test_contains():
    data = _WignerData()
    key = data._columns[0]
    assert key in data


def test_iter():
    data = _WignerData()
    for key in data:
        assert key in data._columns


def test_getitem_invalid_key():
    data = _WignerData()
    with pytest.raises(KeyError):
        data["invalid_key"]


def test_parse_point_group():
    assert _parse_point_group("T") == ("T", None, None)
    assert _parse_point_group("Th") == ("T", "h", None)
    assert _parse_point_group("C4") == ("C", None, 4)


def test_WignerD_init():
    wig = WignerD(10)
    assert wig._max_l == 10
    assert isinstance(wig._data, _WignerData)


def test_WignerD_init_fail():
    with pytest.raises(ValueError):
        WignerD(13)  # more than the max l


def test_WignerD_getitem():
    wig = WignerD(10)
    item = wig["T"]
    assert isinstance(item, np.ndarray)


def test_WignerD_getitem_invalid():
    wig = WignerD(10)
    with pytest.raises(KeyError):
        wig["TT"]  # not supported


def test_WignerD_iter_sph_indices():
    wig = WignerD(2)
    indices = list(wig.iter_sph_indices())
    expected_indices = [
        (0, 0, 0),
        (1, -1, -1),
        (1, -1, 0),
        (1, -1, 1),
        (1, 0, -1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, -1),
        (1, 1, 0),
        (1, 1, 1),
        (2, -2, -2),
        (2, -2, -1),
        (2, -2, 0),
        (2, -2, 1),
        (2, -2, 2),
        (2, -1, -2),
        (2, -1, -1),
        (2, -1, 0),
        (2, -1, 1),
        (2, -1, 2),
        (2, 0, -2),
        (2, 0, -1),
        (2, 0, 0),
        (2, 0, 1),
        (2, 0, 2),
        (2, 1, -2),
        (2, 1, -1),
        (2, 1, 0),
        (2, 1, 1),
        (2, 1, 2),
        (2, 2, -2),
        (2, 2, -1),
        (2, 2, 0),
        (2, 2, 1),
        (2, 2, 2),
    ]
    assert indices == expected_indices
