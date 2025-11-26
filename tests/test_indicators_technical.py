import math

import pytest

from indicators.technical import (
    ema,
    sma,
    rsi,
    atr,
    volume_moving_average,
    ema_crossed,
    detect_support_levels,
    detect_resistance_levels,
)


def test_sma_and_ema() -> None:
    series = [1, 2, 3, 4, 5]
    assert sma(series, 5) == pytest.approx(3.0)
    assert ema(series, 3) > 3.5


def test_rsi_bounds() -> None:
    series = [50, 51, 52, 55, 60, 58, 57, 59, 61, 63, 64, 63, 62, 61, 60]
    value = rsi(series, 14)
    assert 0 <= value <= 100


def test_atr_calculation() -> None:
    highs = [10, 11, 12, 11.5, 12.2, 12.4, 13]
    lows = [9.5, 10, 10.2, 10.5, 10.8, 11.1, 11.7]
    closes = [9.7, 10.5, 11.0, 11.2, 11.9, 12.0, 12.5]
    value = atr(highs, lows, closes, period=5)
    assert value > 0


def test_volume_ma() -> None:
    volumes = [100, 120, 130, 140, 110]
    assert volume_moving_average(volumes, 5) == pytest.approx(sum(volumes) / 5)


def test_ema_cross_detection() -> None:
    fast = [1.0, 2.0, 3.0, 3.4, 4.2]
    slow = [1.5, 2.4, 3.2, 3.5, 4.0]
    assert ema_crossed(fast, slow)


def test_support_resistance_detection() -> None:
    prices = [10, 9, 8, 9, 10, 11, 12, 11, 10, 9, 8, 9, 10, 11, 12]
    supports = detect_support_levels(prices, lookback=2)
    resistances = detect_resistance_levels(prices, lookback=2)
    assert supports
    assert resistances
