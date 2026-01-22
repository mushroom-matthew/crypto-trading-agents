import numpy as np
import pandas as pd
import pytest

from metrics import compute_metrics


def _constant_df(value: float = 100.0, periods: int = 15) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=periods, freq="h", tz="UTC")
    data = {
        "timestamp": ts,
        "open": np.full(periods, value, dtype="float64"),
        "high": np.full(periods, value, dtype="float64"),
        "low": np.full(periods, value, dtype="float64"),
        "close": np.full(periods, value, dtype="float64"),
        "volume": np.ones(periods, dtype="float64") * 100,
    }
    return pd.DataFrame(data)


def test_rsi_handles_flat_prices():
    df = _constant_df()
    result = compute_metrics(df, features=["RSI"], params={"RSI": {"period": 5}}, output="wide")
    rsi = result["RSI_5"]

    # First window should be NaN, steady state should be zero (no gains)
    assert rsi.isna().sum() >= 4
    assert rsi.iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_rsi_handles_strict_up_trend():
    ts = pd.date_range("2024-01-01", periods=15, freq="h", tz="UTC")
    close = np.linspace(100, 120, num=15)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.ones_like(close) * 50,
        }
    )
    result = compute_metrics(df, features=["RSI"], params={"RSI": {"period": 5}}, output="wide")
    rsi = result["RSI_5"].dropna()
    assert rsi.iloc[-1] == pytest.approx(100.0, abs=1e-6)


def test_obv_ignores_zero_volume():
    df = _constant_df()
    df.loc[5:7, "volume"] = 0.0
    df.loc[5:7, "close"] = [101, 99, 100]
    result = compute_metrics(df, features=["OBV"], output="wide")
    obv = result["OBV"]
    # OBV should not change when volume is zero
    assert obv.iloc[7] == obv.iloc[4]


def test_vwap_session_resets():
    df = _constant_df(periods=6)
    df["session_id"] = [1, 1, 1, 2, 2, 2]
    df.loc[:, "close"] = [100, 101, 102, 103, 104, 105]
    df.loc[:, "high"] = df["close"] * 1.001
    df.loc[:, "low"] = df["close"] * 0.999
    df.loc[:, "open"] = df["close"]

    result = compute_metrics(df, features=["VWAP"], output="wide")
    vwap = result["VWAP_session"]

    first_session = vwap.iloc[2]
    second_session_start = vwap.iloc[3]
    assert second_session_start != pytest.approx(first_session)
    assert not np.isnan(second_session_start)


def test_warmup_nan_behavior():
    df = _constant_df()
    result = compute_metrics(
        df,
        features=["SMA", "WMA", "MACD", "BollingerBands", "ATR", "ADX"],
        params={"BollingerBands": {"period": 5, "mult": 2.0}},
        output="wide",
    )

    # SMA/WMA should have NaNs until window fills
    expected_warmup = min(len(df), 20 - 1)
    assert result["SMA_20"].isna().sum() >= expected_warmup
    assert result["WMA_20"].isna().sum() >= expected_warmup

    # Bollinger components share warm-up behaviour
    assert result["BBANDS_5_2.0_basis"].isna().sum() >= 4

    # ATR/ADX should start with NaNs due to Wilder smoothing warm-up
    assert result["ATR_14"].isna().sum() >= 13
    assert result["ADX_14"].isna().sum() >= 13
