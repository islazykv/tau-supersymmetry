from __future__ import annotations

import pytest

from src.processing.processor import _ntau_from_channel


def test_ntau_channel_1():
    assert _ntau_from_channel("1") == "1"


def test_ntau_channel_1had0lep():
    assert _ntau_from_channel("1had0lep") == "1"


def test_ntau_channel_1had1lep():
    assert _ntau_from_channel("1had1lep") == "1"


def test_ntau_channel_2():
    assert _ntau_from_channel("2") == "2"


def test_ntau_channel_0():
    assert _ntau_from_channel("0") == "0"


def test_ntau_invalid_channel():
    with pytest.raises(ValueError, match="Unsupported channel"):
        _ntau_from_channel("invalid")
