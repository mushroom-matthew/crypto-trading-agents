# Metrics Service (Tier I)

The metrics package delivers production-ready OHLCV technical indicators for durable agents and analytics pipelines. It exposes a compact API that validates market data, feeds deterministic indicators to decision makers, and provides tidy outputs for LLM ingestion.

## Installation & Layout

The package ships with the repository—no extra install step beyond the project requirements (`pandas`, `numpy`). Relevant files:

```
metrics/
  __init__.py
  base.py
  technical.py
  registry.py
  typing.py
  market_context.py   # Tier II scaffolding
  sentiment.py        # Tier III scaffolding
```

See `examples/metrics_quickstart.ipynb` for a runnable walkthrough.

## Data Contract

`compute_metrics` expects a DataFrame with **float64** columns:

| Column     | Notes                                   |
|------------|-----------------------------------------|
| timestamp  | Parseable datetime, UTC preferred       |
| open       | Non-negative float                      |
| high       | ≥ low                                   |
| low        | ≤ high                                  |
| close      | Float                                   |
| volume     | Non-negative float                      |

Rows must be in chronological order; any mild disorder is sorted internally. The service tolerates NaNs and zero volume, emitting partial outputs while avoiding division-by-zero.

## Usage

```python
from pathlib import Path
import sys

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()
    if not (ROOT / "metrics").exists() and (ROOT.parent / "metrics").exists():
        ROOT = ROOT.parent

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
from metrics import compute_metrics, list_metrics

df = pd.read_csv("tests/data/btc_ohlcv_sample.csv", parse_dates=["timestamp"])

signals = compute_metrics(
    df,
    features=["RSI", "MACD", "BollingerBands", "ATR", "VWAP", "ADX", "OBV", "ROC", "EMA", "SMA"],
    params={"RSI": {"period": 14}, "BollingerBands": {"period": 20, "mult": 2.0}, "EMA": {"period": 21}},
    output="wide",
)

tidy = compute_metrics(df, features=["MACD", "BollingerBands"], output="long")
```

`list_metrics()` returns all available indicators (`SMA`, `EMA`, `WMA`, `MACD`, `RSI`, `BollingerBands`, `ATR`, `ADX`, `ROC`, `OBV`, `VWAP`).

### MCP Tools

The MCP server now exposes convenience tools for agents and automation:

| Tool name                | Description                                                            |
|--------------------------|------------------------------------------------------------------------|
| `list_technical_metrics` | Enumerate registered Tier I indicators.                               |
| `update_market_cache`    | Fetch Coinbase Exchange OHLCV candles and persist them to `data/`.    |
| `compute_technical_metrics` | Run the metrics pipeline over cached/live data (wide or tidy output). |

Example payload:

```python
import asyncio
from mcp_server.app import compute_technical_metrics

result = asyncio.run(
    compute_technical_metrics(
        symbol="BTC/USD",
        timeframe="1h",
        limit=500,
        features=["RSI", "MACD", "BollingerBands"],
        params={"BollingerBands": {"period": 20, "mult": 2.0}},
        tail=10,
        use_cache=True,
        fetch_if_missing=False,
        data_path="tests/data/btc_ohlcv_sample.csv",
    )
)
```

## Indicator Glossary

| Indicator          | Description                                                       |
|--------------------|-------------------------------------------------------------------|
| SMA, EMA, WMA      | Rolling moving averages (simple, exponential, weighted)           |
| MACD               | Fast/slow EMAs, signal line, histogram                            |
| RSI                | Wilder RSI with gain/loss smoothing                               |
| Bollinger Bands    | Basis, upper, lower, bandwidth, %B                                |
| ATR                | Wilder Average True Range                                         |
| ADX                | +DI, −DI, ADX                                                      |
| ROC                | Rate of change (`close/close.shift(n) - 1`)                        |
| OBV                | On-Balance Volume                                                 |
| VWAP               | Session-aware VWAP (fallback rolling/cumulative)                  |

## Extensibility Roadmap

* `metrics/market_context.py` will grow Tier II metrics (Sharpe, Sortino, beta, dominance ratios).
* `metrics/sentiment.py` will host Tier III sentiment analytics once external feeds are wired in.
* Feature flags for warm-up behaviour and session inference are earmarked for stretch milestones.
