# Factor Exposure & Hedging (Draft)

## Purpose
Quantify each symbol’s beta to crypto-wide factors (market, ETH/BTC, dominance) and surface idiosyncratic volatility so we can hedge beta risk or size positions accordingly.

## API
- Module: `agents/analytics/factors.py`
- Entry point: `compute_factor_loadings(frames_by_symbol, factors_df, lookback=90) -> Dict[str, FactorExposure]`
  - Inputs: OHLCV frames with `close`, factor dataframe indexed by datetime (columns: `market`, `eth_beta`, etc.)
  - Outputs: per-symbol betas, idiosyncratic volatility, R², window length.
- Helpers: `example_crypto_factors(dominance, eth_btc)` builds minimal factor dataframe from BTC dominance and ETH/BTC ratio series.
- Serialization: `FactorExposure.to_dict()` flattens betas to `beta_<name>` keys for telemetry.

## Telemetry Plan
- Add `factor_exposures` to daily reports and `run_summary.json`:
  ```json
  "factor_exposures": {
    "BTC-USD": {"beta_market": 0.9, "beta_eth_beta": 0.2, "idiosyncratic_vol": 0.012, "r2": 0.45, "window": 90}
  }
  ```
- LLM context: include in `global_context.factor_exposures` so strategist/judge can down-weight naked beta or add hedges.
- Judge policy: optional auto-hedge mode targets market beta ≈ 0 by sizing offsets; log target vs. achieved beta.

## Fetch & Wiring
- Fetch CoinGecko factors (requires `COINGECKO_API_KEY` if 401s occur):
  ```
  ./scripts/fetch_factors.py --days 180 --interval hourly --timeframe 1h --out data/factors.parquet
  ```
- Offline/local option: pass your own OHLCV CSVs (timestamp, close[,market_cap]) for BTC/ETH (and optional total mcap):
  ```
  ./scripts/fetch_factors.py --btc-csv data/btc.csv --eth-csv data/eth.csv --total-csv data/total.csv --timeframe 1h --out data/factors.csv
  ```
- Pass to LLM backtests via CLI:
  ```
  uv run python -m backtesting.cli --llm-strategist enabled ... --factor-data data/factors.parquet --auto-hedge-market
  # or auto-build per run:
  uv run python -m backtesting.cli --llm-strategist enabled ... --factor-auto-fetch --factor-btc-csv data/btc.csv --factor-eth-csv data/eth.csv --factor-start 2024-01-01 --factor-end 2024-02-01
  ```
- Backtester also builds simple proxies from price data when no factor file is provided.

## Next Steps
- Wire factor loading computation into backtester (per-symbol, per-timeframe) using cached factor proxies.
- Add CLI flag `--auto-hedge market` to enable judge-driven neutralization.
- Extend prompt text to explain beta fields and hedge expectations.
