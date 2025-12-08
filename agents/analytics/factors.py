"""Factor exposure estimation for crypto portfolios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorExposure:
    """Rolling factor beta estimate and idiosyncratic volatility."""

    symbol: str
    betas: dict[str, float]
    idiosyncratic_vol: float
    r2: float | None = None
    window: int | None = None

    def to_dict(self) -> dict[str, float | str | None]:
        payload: dict[str, float | str | None] = {
            "symbol": self.symbol,
            "idiosyncratic_vol": self.idiosyncratic_vol,
            "r2": self.r2,
            "window": self.window,
        }
        for name, beta in self.betas.items():
            payload[f"beta_{name}"] = beta
        return payload


def _returns_from_prices(df: pd.DataFrame) -> pd.Series:
    if "close" not in df.columns:
        raise ValueError("price dataframe must contain a 'close' column")
    return df["close"].pct_change().dropna()


def _align_returns(
    returns: pd.Series,
    factors: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    joined = factors.join(returns.rename("asset_ret"), how="inner")
    joined = joined.dropna()
    y = joined["asset_ret"]
    x = joined.drop(columns=["asset_ret"])
    return y, x


def _ols_beta(y: pd.Series, x: pd.DataFrame) -> tuple[np.ndarray, float | None]:
    if y.empty or x.empty:
        return np.array([]), None
    x_mat = x.to_numpy()
    y_vec = y.to_numpy()
    # add intercept
    x_design = np.c_[np.ones(len(x_mat)), x_mat]
    coeffs, *_ = np.linalg.lstsq(x_design, y_vec, rcond=None)
    intercept = coeffs[0]
    betas = coeffs[1:]
    fitted = x_design @ coeffs
    residuals = y_vec - fitted
    if len(y_vec) == 0:
        r2 = None
    else:
        ss_tot = ((y_vec - y_vec.mean()) ** 2).sum()
        ss_res = (residuals**2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot else None
    return betas, r2


def compute_factor_loadings(
    frames_by_symbol: Mapping[str, pd.DataFrame],
    factors: pd.DataFrame,
    lookback: int = 90,
) -> Dict[str, FactorExposure]:
    """
    Estimate factor betas per symbol using rolling regression.

    Args:
        frames_by_symbol: map of symbol -> OHLCV dataframe (must have 'close' indexed by datetime).
        factors: dataframe indexed by datetime with factor columns (e.g., market, dominance, eth_beta).
        lookback: minimum observations for regression.
    """

    exposures: Dict[str, FactorExposure] = {}
    for symbol, df in frames_by_symbol.items():
        returns = _returns_from_prices(df)
        returns = returns.tail(lookback * 2)  # keep recent history only
        y, x = _align_returns(returns, factors)
        if len(y) < lookback or x.shape[0] < lookback:
            exposures[symbol] = FactorExposure(symbol=symbol, betas={}, idiosyncratic_vol=float("nan"), r2=None, window=len(y))
            continue
        y_window = y.tail(lookback)
        x_window = x.tail(lookback)
        betas_vec, r2 = _ols_beta(y_window, x_window)
        betas: dict[str, float] = {}
        for name, beta in zip(x_window.columns, betas_vec):
            betas[name] = float(beta)
        # residuals for idio vol
        x_design = np.c_[np.ones(len(x_window)), x_window.to_numpy()]
        fitted = x_design @ np.r_[0.0, betas_vec]
        residuals = y_window.to_numpy() - fitted
        idio_vol = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float("nan")
        exposures[symbol] = FactorExposure(
            symbol=symbol,
            betas=betas,
            idiosyncratic_vol=idio_vol,
            r2=r2,
            window=len(y_window),
        )
    return exposures


def example_crypto_factors(dominance: pd.Series, eth_btc: pd.Series) -> pd.DataFrame:
    """
    Build a minimal factor dataframe from crypto proxies.

    Args:
        dominance: BTC dominance percentage series indexed by datetime.
        eth_btc: ETH/BTC ratio series indexed by datetime.
    """

    df = pd.DataFrame({"market": dominance.pct_change(), "eth_beta": eth_btc.pct_change()})
    return df.dropna()
