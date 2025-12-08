from datetime import timezone

import numpy as np
import pandas as pd

from agents.analytics.factors import compute_factor_loadings, example_crypto_factors


def _price_series(base: float, returns: list[float]) -> pd.DataFrame:
    prices = [base]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    timestamps = pd.date_range("2024-01-01", periods=len(prices), freq="D", tz=timezone.utc)
    return pd.DataFrame({"timestamp": timestamps, "close": prices}).set_index("timestamp")


def test_compute_factor_loadings_matches_known_beta():
    rng = np.random.default_rng(42)
    # random factor returns with variance
    factor_returns = rng.normal(0.0, 0.01, size=120).tolist()
    market_prices = _price_series(100.0, factor_returns)
    dominance = market_prices["close"]
    eth_ratio = market_prices["close"] * 0 + 1.0
    factors = example_crypto_factors(dominance, eth_ratio)

    # asset with beta ~1.5 vs market factor plus noise
    asset_returns = [1.5 * r + rng.normal(0, 0.002) for r in factor_returns]
    asset_prices = _price_series(50.0, asset_returns)

    exposures = compute_factor_loadings({"TEST": asset_prices}, factors, lookback=60)
    exposure = exposures["TEST"]
    assert "market" in exposure.betas
    assert 1.3 <= exposure.betas["market"] <= 1.7
    assert exposure.idiosyncratic_vol < 0.02
    assert exposure.window >= 60


def test_example_crypto_factors_alignment():
    dom = pd.Series([60, 61, 62], index=pd.date_range("2024-01-01", periods=3, freq="D", tz=timezone.utc))
    eth = pd.Series([0.05, 0.051, 0.052], index=dom.index)
    df = example_crypto_factors(dom, eth)
    assert "market" in df.columns and "eth_beta" in df.columns
    assert not df.empty
