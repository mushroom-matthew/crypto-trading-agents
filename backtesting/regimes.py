"""Market regime definitions for backtesting.

Regime periods are defined as (start_date, end_date) tuples.
These can be used by both the CLI and API.
"""

from __future__ import annotations

from typing import Dict, Tuple

# Market regime definitions with dates
REGIMES: Dict[str, Tuple[str, str]] = {
    "bull_2020_2021": ("2020-10-01", "2021-04-14"),
    "late_bull_2021": ("2021-07-20", "2021-11-10"),
    "bear_2018": ("2018-01-01", "2018-12-15"),
    "covid_crash": ("2020-02-20", "2020-03-15"),
    "bear_2022": ("2022-04-01", "2022-11-10"),
    "range_2019": ("2019-01-01", "2019-10-01"),
    "mid_2021_consolidation": ("2021-05-01", "2021-07-20"),
    "late_2023_consolidation": ("2023-06-01", "2023-10-01"),
    "vol_china_elon": ("2021-05-01", "2021-06-01"),
    "ftx_collapse": ("2022-11-01", "2022-11-20"),
}

# Human-readable metadata for each regime
REGIME_METADATA: Dict[str, Dict[str, str]] = {
    "bull_2020_2021": {
        "name": "Bull Market 2020-2021",
        "description": "Post-COVID recovery rally through early 2021 ATH",
        "character": "bull",
    },
    "late_bull_2021": {
        "name": "Late Bull 2021",
        "description": "Summer rally to November ATH",
        "character": "bull",
    },
    "bear_2018": {
        "name": "Bear Market 2018",
        "description": "Extended drawdown following 2017 peak",
        "character": "bear",
    },
    "covid_crash": {
        "name": "COVID Crash",
        "description": "March 2020 pandemic panic (extreme volatility)",
        "character": "volatile",
    },
    "bear_2022": {
        "name": "Bear Market 2022",
        "description": "Extended drawdown following Nov 2021 peak",
        "character": "bear",
    },
    "range_2019": {
        "name": "Range 2019",
        "description": "Sideways consolidation through 2019",
        "character": "ranging",
    },
    "mid_2021_consolidation": {
        "name": "Mid-2021 Consolidation",
        "description": "May-July 2021 correction and consolidation",
        "character": "ranging",
    },
    "late_2023_consolidation": {
        "name": "Late 2023 Consolidation",
        "description": "Summer 2023 low-volatility period",
        "character": "ranging",
    },
    "vol_china_elon": {
        "name": "China/Elon Volatility",
        "description": "May 2021 China ban + Elon tweet volatility",
        "character": "volatile",
    },
    "ftx_collapse": {
        "name": "FTX Collapse",
        "description": "November 2022 FTX implosion (extreme volatility)",
        "character": "volatile",
    },
}
