from __future__ import annotations

import logging

from backtesting.logging_config import setup_backtest_logging


def test_setup_backtest_logging_creates_file(tmp_path):
    log_path = tmp_path / "backtest.log"
    setup_backtest_logging(level="DEBUG", log_file=str(log_path), json_logs=False)
    logger = logging.getLogger("backtesting.test")
    logger.debug("debug entry")
    logger.info("info entry")
    assert log_path.exists()
    text = log_path.read_text(encoding="utf-8")
    assert "info entry" in text
