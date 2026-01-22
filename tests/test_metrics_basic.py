import numpy as np
import pandas as pd
import pytest

from metrics import compute_metrics, list_metrics


def load_sample() -> pd.DataFrame:
    return pd.read_csv("tests/data/btc_ohlcv_sample.csv", parse_dates=["timestamp"])


def test_list_metrics_contains_all_expected():
    expected = {
        "SMA",
        "EMA",
        "WMA",
        "MACD",
        "RSI",
        "BollingerBands",
        "ATR",
        "ADX",
        "ROC",
        "OBV",
        "VWAP",
    }
    assert set(list_metrics()) == expected


def test_compute_metrics_wide_adds_columns():
    df = load_sample()
    features = list_metrics()
    params = {"RSI": {"period": 14}, "BollingerBands": {"period": 20, "mult": 2.0}}
    result = compute_metrics(df, features=features, params=params, output="wide")

    assert len(result) == len(df)
    # Original columns preserved
    for col in ["timestamp", "open", "high", "low", "close", "volume"]:
        assert col in result.columns

    expected_columns = [
        "SMA_20",
        "EMA_20",
        "WMA_20",
        "MACD_12_26_9",
        "MACD_12_26_9_signal",
        "MACD_12_26_9_hist",
        "RSI_14",
        "BBANDS_20_2.0_basis",
        "BBANDS_20_2.0_upper",
        "BBANDS_20_2.0_lower",
        "BBANDS_20_2.0_bandwidth",
        "BBANDS_20_2.0_pctB",
        "ATR_14",
        "ADX_14_plus_di",
        "ADX_14_minus_di",
        "ADX_14",
        "ROC_12",
        "OBV",
        "VWAP",
    ]
    for col in expected_columns:
        assert col in result.columns


def test_compute_metrics_long_format():
    df = load_sample()
    result = compute_metrics(df, features=["MACD", "BollingerBands"], output="long")

    assert set(result.columns) == {"timestamp", "feature", "key", "value"}
    assert set(result["feature"].unique()) == {"MACD", "BollingerBands"}

    macd_keys = set(result.loc[result["feature"] == "MACD", "key"].unique())
    bb_keys = set(result.loc[result["feature"] == "BollingerBands", "key"].unique())

    assert macd_keys == {"value", "signal", "hist"}
    assert bb_keys == {"basis", "upper", "lower", "bandwidth", "pctB"}


def test_compute_metrics_deterministic():
    df = load_sample()
    result_1 = compute_metrics(df, features=["RSI", "MACD"], output="wide")
    result_2 = compute_metrics(df, features=["RSI", "MACD"], output="wide")
    pd.testing.assert_frame_equal(result_1, result_2)
