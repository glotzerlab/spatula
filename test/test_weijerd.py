import pytest
import numpy as np
from pgop.weijerd import _WignerData, _parse_point_group, WeigerD


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
        item = data["invalid_key"]


def test_parse_point_group():
    assert _parse_point_group("T") == ("T", None, None)
    assert _parse_point_group("Th") == ("T", "h", None)
    assert _parse_point_group("C4") == ("C", None, 4)


def test_weigerD_init():
    wig = WeigerD(10)
    assert wig._max_l == 10
    assert isinstance(wig._data, _WignerData)


def test_weigerD_init_fail():
    with pytest.raises(ValueError):
        wig = WeigerD(13)  # more than the max l


def test_weigerD_getitem():
    wig = WeigerD(10)
    item = wig["T"]
    assert isinstance(item, np.ndarray)


def test_weigerD_getitem_invalid():
    wig = WeigerD(10)
    with pytest.raises(KeyError):
        item = wig["TT"]  # not supported


def test_weigerD_iter_sph_indices():
    wig = WeigerD(2)
    indices = list(wig._iter_sph_indices())
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
