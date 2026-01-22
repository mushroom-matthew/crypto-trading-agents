import pandas as pd
from pandas.testing import assert_frame_equal

from data_loader.factors import build_factor_df, build_factors_from_csvs, load_cached_factors, resample_factors, save_factors


def test_build_factor_df_and_resample(tmp_path):
    idx = pd.date_range("2024-01-01", periods=5, freq="H")
    btc = pd.DataFrame({"close": [100, 101, 102, 103, 104], "market_cap": [1, 1.1, 1.2, 1.3, 1.4]}, index=idx)
    eth = pd.DataFrame({"close": [10, 10.5, 10.4, 10.6, 10.7], "market_cap": [0.2, 0.21, 0.22, 0.23, 0.24]}, index=idx)
    factors = build_factor_df(btc, eth)
    assert set(factors.columns) == {"market", "dominance", "eth_beta"}
    resampled = resample_factors(factors, timeframe="2H")
    assert not resampled.empty
    path = tmp_path / "factors.parquet"
    save_factors(resampled, path)
    loaded = load_cached_factors(path)
    assert_frame_equal(resampled, loaded, check_names=False, check_freq=False)


def test_build_factors_from_csvs_filters_window(tmp_path):
    idx = pd.date_range("2024-01-01", periods=5, freq="H", tz="UTC")
    btc = pd.DataFrame({"timestamp": idx, "close": [100, 101, 102, 103, 104], "market_cap": [1, 1.1, 1.2, 1.3, 1.4]})
    eth = pd.DataFrame({"timestamp": idx, "close": [10, 10.5, 10.4, 10.6, 10.7], "market_cap": [0.2, 0.21, 0.22, 0.23, 0.24]})
    btc_path = tmp_path / "btc.csv"
    eth_path = tmp_path / "eth.csv"
    btc.to_csv(btc_path, index=False)
    eth.to_csv(eth_path, index=False)
    factors = build_factors_from_csvs(btc_path, eth_path, timeframe="2H", start=idx[1], end=idx[-2])
    assert not factors.empty
    assert factors.index.min() >= idx[1]
    assert factors.index.max() <= idx[-2]
