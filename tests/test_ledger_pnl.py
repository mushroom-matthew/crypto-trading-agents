import pytest
from agents.workflows import ExecutionLedgerWorkflow
from decimal import Decimal


def test_pnl_uses_initial_cash_and_unrealized_value():
    wf = ExecutionLedgerWorkflow()
    assert wf.get_pnl() == 0.0
    wf.record_fill({
        "side": "BUY",
        "symbol": "BTC/USD",
        "qty": 10,
        "fill_price": 1000,
        "cost": 10000,
    })
    assert wf.get_pnl() == 0.0
    wf.last_price["BTC/USD"] = Decimal("1100")
    assert wf.get_pnl() == pytest.approx(1000.0)


def test_entry_price_tracking():
    wf = ExecutionLedgerWorkflow()
    wf.record_fill({
        "side": "BUY",
        "symbol": "ETH/USD",
        "qty": 2,
        "fill_price": 100,
        "cost": 200,
    })
    assert wf.get_entry_prices()["ETH/USD"] == pytest.approx(100.0)
    wf.record_fill({
        "side": "BUY",
        "symbol": "ETH/USD",
        "qty": 2,
        "fill_price": 120,
        "cost": 240,
    })
    assert wf.get_entry_prices()["ETH/USD"] == pytest.approx(110.0)
    wf.record_fill({
        "side": "SELL",
        "symbol": "ETH/USD",
        "qty": 1,
        "fill_price": 130,
        "cost": 130,
    })
    assert wf.get_entry_prices()["ETH/USD"] == pytest.approx(110.0)
    wf.record_fill({
        "side": "SELL",
        "symbol": "ETH/USD",
        "qty": 3,
        "fill_price": 115,
        "cost": 345,
    })
    assert "ETH/USD" not in wf.get_entry_prices()


def test_short_entry_updates_signed_position_and_equity():
    wf = ExecutionLedgerWorkflow()
    initial_cash = wf.get_cash()

    wf.record_fill({
        "side": "SELL",
        "symbol": "BTC/USD",
        "qty": 2,
        "fill_price": 100,
        "cost": 200,
        "intent": "entry",
    })

    assert wf.get_positions()["BTC/USD"] == pytest.approx(-2.0)
    assert wf.get_entry_prices()["BTC/USD"] == pytest.approx(100.0)
    assert wf.get_cash() == pytest.approx(initial_cash + 200.0)

    wf.last_price["BTC/USD"] = Decimal("90")
    assert wf.get_unrealized_pnl() == pytest.approx(20.0)
    assert wf.get_pnl() == pytest.approx(20.0)


def test_buy_exit_covers_short_and_realizes_pnl():
    wf = ExecutionLedgerWorkflow()
    initial_cash = wf.get_cash()

    wf.record_fill({
        "side": "SELL",
        "symbol": "BTC/USD",
        "qty": 2,
        "fill_price": 100,
        "cost": 200,
        "intent": "entry",
    })
    wf.record_fill({
        "side": "BUY",
        "symbol": "BTC/USD",
        "qty": 2,
        "fill_price": 90,
        "cost": 180,
        "intent": "exit",
    })

    assert wf.get_positions() == {}
    assert wf.get_entry_prices() == {}
    assert wf.get_realized_pnl() == pytest.approx(20.0)
    assert wf.get_cash() == pytest.approx(initial_cash + 20.0)


def test_long_exit_realized_pnl_and_cash_include_fees():
    wf = ExecutionLedgerWorkflow()
    initial_cash = wf.get_cash()

    wf.record_fill({
        "side": "BUY",
        "symbol": "BTC/USD",
        "qty": 1,
        "fill_price": 100,
        "cost": 100,
        "fee": 1,
        "intent": "entry",
    })
    wf.record_fill({
        "side": "SELL",
        "symbol": "BTC/USD",
        "qty": 1,
        "fill_price": 110,
        "cost": 110,
        "fee": 1,
        "intent": "exit",
    })

    # Net trade P&L is +8 after entry/exit fees. Profit scraping removes 20% of that
    # from free cash but should not change realized P&L itself.
    assert wf.get_realized_pnl() == pytest.approx(8.0)
    assert wf.get_cash() == pytest.approx(initial_cash + 8.0 - 1.6)


def test_short_cover_realized_pnl_and_cash_include_fees():
    wf = ExecutionLedgerWorkflow()
    initial_cash = wf.get_cash()

    wf.record_fill({
        "side": "SELL",
        "symbol": "BTC/USD",
        "qty": 2,
        "fill_price": 100,
        "cost": 200,
        "fee": 2,
        "intent": "entry",
    })
    wf.record_fill({
        "side": "BUY",
        "symbol": "BTC/USD",
        "qty": 2,
        "fill_price": 90,
        "cost": 180,
        "fee": 2,
        "intent": "exit",
    })

    assert wf.get_positions() == {}
    assert wf.get_entry_prices() == {}
    assert wf.get_realized_pnl() == pytest.approx(16.0)
    assert wf.get_cash() == pytest.approx(initial_cash + 16.0)


def test_flat_exit_sell_is_ignored_and_not_logged():
    wf = ExecutionLedgerWorkflow()

    wf.record_fill({
        "side": "SELL",
        "symbol": "BTC/USD",
        "qty": 1,
        "fill_price": 100,
        "cost": 100,
        "intent": "exit",
    })

    assert wf.get_positions() == {}
    assert wf.transaction_history == []
