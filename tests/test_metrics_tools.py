import asyncio
from pathlib import Path

from mcp_server.app import compute_technical_metrics, list_technical_metrics


def test_list_technical_metrics_tool():
    result = asyncio.run(list_technical_metrics())
    assert "metrics" in result
    assert "RSI" in result["metrics"]


def test_compute_technical_metrics_from_sample_csv():
    data_path = Path("tests/data/btc_ohlcv_sample.csv")
    result = asyncio.run(
        compute_technical_metrics(
            symbol="BTC/USD",
            timeframe="1h",
            limit=300,
            features=["RSI", "SMA"],
            params={"RSI": {"period": 14}},
            output="wide",
            tail=3,
            use_cache=False,
            fetch_if_missing=False,
            data_path=str(data_path),
        )
    )

    assert result["rows"] == 300
    assert len(result["preview"]) == 3
    preview_keys = result["preview"][0].keys()
    assert "RSI_14" in preview_keys
    assert "SMA_20" in preview_keys
