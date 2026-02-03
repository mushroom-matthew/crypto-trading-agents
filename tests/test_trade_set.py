"""Tests for TradeSet and TradeLeg position lifecycle accounting."""

from datetime import datetime, timezone
from uuid import UUID

import pytest

from schemas.trade_set import TradeLeg, TradeSet, TradeSetBuilder


class TestTradeLeg:
    """Tests for TradeLeg model."""

    def test_leg_id_auto_generated(self):
        """leg_id should be auto-generated as valid UUID."""
        leg = TradeLeg(
            side="buy",
            qty=1.0,
            price=100.0,
            timestamp=datetime.now(timezone.utc),
            is_entry=True,
        )
        # Verify it's a valid UUID
        assert UUID(leg.leg_id)

    def test_leg_id_unique_across_legs(self):
        """Each leg should get a unique ID."""
        ts = datetime.now(timezone.utc)
        legs = [
            TradeLeg(side="buy", qty=1.0, price=100.0, timestamp=ts, is_entry=True)
            for _ in range(10)
        ]
        leg_ids = [leg.leg_id for leg in legs]
        assert len(set(leg_ids)) == 10  # All unique

    def test_leg_with_all_fields(self):
        """Leg should accept all optional fields."""
        leg = TradeLeg(
            side="sell",
            qty=0.5,
            price=105.0,
            fees=0.25,
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            trigger_id="momentum_long_exit",
            category="risk_reduce",
            reason="Taking partial profits",
            is_entry=False,
            exit_fraction=0.5,
            wac_at_fill=100.0,
            realized_pnl=2.25,
            position_after=0.5,
            learning_book=True,
            experiment_id="exp-001",
            experiment_variant="aggressive",
        )
        assert leg.side == "sell"
        assert leg.exit_fraction == 0.5
        assert leg.realized_pnl == 2.25
        assert leg.learning_book is True

    def test_leg_qty_must_be_positive(self):
        """qty must be > 0."""
        with pytest.raises(ValueError):
            TradeLeg(
                side="buy",
                qty=0,
                price=100.0,
                timestamp=datetime.now(timezone.utc),
                is_entry=True,
            )

    def test_leg_price_must_be_positive(self):
        """price must be > 0."""
        with pytest.raises(ValueError):
            TradeLeg(
                side="buy",
                qty=1.0,
                price=-100.0,
                timestamp=datetime.now(timezone.utc),
                is_entry=True,
            )


class TestTradeSet:
    """Tests for TradeSet model."""

    def test_trade_set_computed_fields(self):
        """Computed fields should calculate correctly."""
        ts_open = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        ts_close = datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)

        trade_set = TradeSet(
            symbol="BTC-USD",
            timeframe="1h",
            opened_at=ts_open,
            closed_at=ts_close,
            legs=[
                TradeLeg(
                    side="buy",
                    qty=1.0,
                    price=100.0,
                    fees=0.5,
                    timestamp=ts_open,
                    is_entry=True,
                    position_after=1.0,
                ),
                TradeLeg(
                    side="sell",
                    qty=1.0,
                    price=110.0,
                    fees=0.5,
                    timestamp=ts_close,
                    is_entry=False,
                    realized_pnl=9.0,
                    position_after=0.0,
                ),
            ],
            pnl_realized_total=9.0,
            fees_total=1.0,
            entry_side="long",
        )

        assert trade_set.num_legs == 2
        assert trade_set.num_entries == 1
        assert trade_set.num_exits == 1
        assert trade_set.is_closed is True
        assert trade_set.hold_duration_hours == 4.0
        assert trade_set.avg_entry_price == 100.0
        assert trade_set.avg_exit_price == 110.0
        assert trade_set.total_entry_qty == 1.0
        assert trade_set.total_exit_qty == 1.0
        assert trade_set.max_exposure == 1.0

    def test_trade_set_multi_leg(self):
        """TradeSet with multiple entries and exits."""
        ts1 = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc)
        ts3 = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        ts4 = datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)

        trade_set = TradeSet(
            symbol="BTC-USD",
            opened_at=ts1,
            closed_at=ts4,
            legs=[
                # Buy 1.0 @ 100
                TradeLeg(
                    side="buy", qty=1.0, price=100.0, timestamp=ts1,
                    is_entry=True, position_after=1.0,
                ),
                # Partial exit: sell 0.3 @ 105
                TradeLeg(
                    side="sell", qty=0.3, price=105.0, timestamp=ts2,
                    is_entry=False, exit_fraction=0.3, position_after=0.7,
                    realized_pnl=1.5,
                ),
                # Scale back in: buy 0.2 @ 103
                TradeLeg(
                    side="buy", qty=0.2, price=103.0, timestamp=ts3,
                    is_entry=True, position_after=0.9,
                ),
                # Final close: sell 0.9 @ 108
                TradeLeg(
                    side="sell", qty=0.9, price=108.0, timestamp=ts4,
                    is_entry=False, position_after=0.0,
                    realized_pnl=6.0,
                ),
            ],
            pnl_realized_total=7.5,
            fees_total=0.0,
            entry_side="long",
        )

        assert trade_set.num_legs == 4
        assert trade_set.num_entries == 2
        assert trade_set.num_exits == 2
        assert trade_set.total_entry_qty == 1.2
        assert trade_set.total_exit_qty == 1.2
        assert trade_set.max_exposure == 1.0  # Peak was 1.0 after first buy

    def test_trade_set_to_paired_trade_dict(self):
        """Conversion to legacy PairedTrade format."""
        ts_open = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        ts_close = datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)

        trade_set = TradeSet(
            symbol="ETH-USD",
            timeframe="5m",
            opened_at=ts_open,
            closed_at=ts_close,
            legs=[
                TradeLeg(
                    side="buy", qty=2.0, price=2000.0, fees=1.0,
                    timestamp=ts_open, is_entry=True, trigger_id="trend_entry",
                ),
                TradeLeg(
                    side="sell", qty=2.0, price=2100.0, fees=1.0,
                    timestamp=ts_close, is_entry=False, trigger_id="trend_exit",
                    realized_pnl=198.0,
                ),
            ],
            pnl_realized_total=198.0,
            fees_total=2.0,
            entry_side="long",
        )

        paired = trade_set.to_paired_trade_dict()

        assert paired["symbol"] == "ETH-USD"
        assert paired["side"] == "buy"
        assert paired["entry_price"] == 2000.0
        assert paired["exit_price"] == 2100.0
        assert paired["entry_trigger"] == "trend_entry"
        assert paired["exit_trigger"] == "trend_exit"
        assert paired["qty"] == 2.0
        assert paired["pnl"] == 198.0
        assert paired["fees"] == 2.0
        assert paired["hold_duration_hours"] == 4.0


class TestTradeSetBuilder:
    """Tests for TradeSetBuilder which constructs TradeSets from fills."""

    def test_simple_long_round_trip(self):
        """Simple buy then sell creates one TradeSet."""
        builder = TradeSetBuilder()

        ts1 = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        # Open long
        leg1 = builder.process_fill(
            symbol="BTC-USD",
            side="buy",
            qty=1.0,
            price=40000.0,
            timestamp=ts1,
            fees=20.0,
            trigger_id="momentum_long",
        )

        assert leg1.is_entry is True
        assert builder.get_position("BTC-USD") == 1.0
        assert builder.get_wac("BTC-USD") == 40000.0
        assert len(builder.open_sets) == 1
        assert len(builder.closed_sets) == 0

        # Close long
        leg2 = builder.process_fill(
            symbol="BTC-USD",
            side="sell",
            qty=1.0,
            price=41000.0,
            timestamp=ts2,
            fees=20.0,
            trigger_id="momentum_long_exit",
        )

        assert leg2.is_entry is False
        assert leg2.realized_pnl == 1000.0 - 20.0  # (41000 - 40000) * 1 - fees
        assert builder.get_position("BTC-USD") == 0.0
        assert len(builder.open_sets) == 0
        assert len(builder.closed_sets) == 1

        # Verify TradeSet
        trade_set = builder.closed_sets[0]
        assert trade_set.symbol == "BTC-USD"
        assert trade_set.num_legs == 2
        assert trade_set.is_closed is True
        assert trade_set.pnl_realized_total == 980.0
        assert trade_set.fees_total == 40.0

    def test_partial_exit_wac_accounting(self):
        """Partial exits should use WAC for P&L calculation."""
        builder = TradeSetBuilder()

        ts1 = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc)
        ts3 = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        # Open with 1.0 @ 100
        builder.process_fill(
            symbol="TEST",
            side="buy",
            qty=1.0,
            price=100.0,
            timestamp=ts1,
        )

        assert builder.get_wac("TEST") == 100.0

        # Partial exit 0.5 @ 110
        leg2 = builder.process_fill(
            symbol="TEST",
            side="sell",
            qty=0.5,
            price=110.0,
            timestamp=ts2,
            exit_fraction=0.5,
        )

        assert leg2.is_entry is False
        assert leg2.realized_pnl == 0.5 * (110.0 - 100.0)  # 5.0
        assert builder.get_position("TEST") == 0.5
        assert builder.get_wac("TEST") == 100.0  # WAC unchanged on sell
        assert len(builder.open_sets) == 1

        # Close remaining 0.5 @ 105
        leg3 = builder.process_fill(
            symbol="TEST",
            side="sell",
            qty=0.5,
            price=105.0,
            timestamp=ts3,
        )

        assert leg3.realized_pnl == 0.5 * (105.0 - 100.0)  # 2.5
        assert builder.get_position("TEST") == 0.0
        assert len(builder.closed_sets) == 1

        trade_set = builder.closed_sets[0]
        assert trade_set.pnl_realized_total == 7.5  # 5.0 + 2.5

    def test_scale_in_wac_update(self):
        """Scaling into position should update WAC."""
        builder = TradeSetBuilder()

        ts1 = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc)
        ts3 = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        # Buy 1.0 @ 100
        builder.process_fill(symbol="TEST", side="buy", qty=1.0, price=100.0, timestamp=ts1)
        assert builder.get_wac("TEST") == 100.0

        # Scale in: buy 1.0 @ 110
        builder.process_fill(symbol="TEST", side="buy", qty=1.0, price=110.0, timestamp=ts2)

        # WAC should be (1*100 + 1*110) / 2 = 105
        assert builder.get_wac("TEST") == 105.0
        assert builder.get_position("TEST") == 2.0

        # Close all @ 108
        leg3 = builder.process_fill(symbol="TEST", side="sell", qty=2.0, price=108.0, timestamp=ts3)

        # P&L = 2.0 * (108 - 105) = 6.0
        assert leg3.realized_pnl == 6.0
        assert builder.get_position("TEST") == 0.0

    def test_multiple_symbols_separate_sets(self):
        """Each symbol should have its own TradeSets."""
        builder = TradeSetBuilder()
        ts = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        builder.process_fill(symbol="BTC-USD", side="buy", qty=1.0, price=40000.0, timestamp=ts)
        builder.process_fill(symbol="ETH-USD", side="buy", qty=10.0, price=2000.0, timestamp=ts)

        assert len(builder.open_sets) == 2
        assert "BTC-USD" in builder.open_sets
        assert "ETH-USD" in builder.open_sets
        assert builder.get_position("BTC-USD") == 1.0
        assert builder.get_position("ETH-USD") == 10.0

    def test_short_position(self):
        """Short positions should be tracked correctly."""
        builder = TradeSetBuilder()

        ts1 = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        # Open short: sell 1.0 @ 100
        leg1 = builder.process_fill(symbol="TEST", side="sell", qty=1.0, price=100.0, timestamp=ts1)

        assert leg1.is_entry is True
        assert builder.get_position("TEST") == -1.0
        assert builder.get_wac("TEST") == 100.0

        # Close short: buy 1.0 @ 95 (profit)
        leg2 = builder.process_fill(symbol="TEST", side="buy", qty=1.0, price=95.0, timestamp=ts2)

        assert leg2.is_entry is False
        # Short P&L: qty * (entry - exit) = 1.0 * (100 - 95) = 5.0
        assert leg2.realized_pnl == 5.0
        assert builder.get_position("TEST") == 0.0

        trade_set = builder.closed_sets[0]
        assert trade_set.entry_side == "short"
        assert trade_set.pnl_realized_total == 5.0

    def test_unique_leg_ids_same_timestamp(self):
        """Multiple fills at same timestamp should have unique leg_ids."""
        builder = TradeSetBuilder()
        ts = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        # Multiple fills at exact same timestamp (common in crypto)
        leg1 = builder.process_fill(symbol="BTC", side="buy", qty=0.5, price=100.0, timestamp=ts)
        leg2 = builder.process_fill(symbol="BTC", side="buy", qty=0.5, price=100.0, timestamp=ts)

        assert leg1.leg_id != leg2.leg_id
        assert UUID(leg1.leg_id)
        assert UUID(leg2.leg_id)

    def test_same_bar_flatten_reopen(self):
        """Flatten and reopen on same bar creates two separate TradeSets."""
        builder = TradeSetBuilder()
        ts = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        # First trade set: buy then sell
        builder.process_fill(symbol="TEST", side="buy", qty=1.0, price=100.0, timestamp=ts)
        builder.process_fill(symbol="TEST", side="sell", qty=1.0, price=101.0, timestamp=ts)

        # Second trade set: buy again on same timestamp
        builder.process_fill(symbol="TEST", side="buy", qty=1.0, price=101.0, timestamp=ts)

        assert len(builder.closed_sets) == 1
        assert len(builder.open_sets) == 1

        # All sets
        all_sets = builder.all_sets()
        assert len(all_sets) == 2

    def test_learning_book_tags_propagate(self):
        """Learning book tags should propagate to TradeSet."""
        builder = TradeSetBuilder()
        ts = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        builder.process_fill(
            symbol="TEST",
            side="buy",
            qty=1.0,
            price=100.0,
            timestamp=ts,
            learning_book=True,
            experiment_id="exp-001",
        )

        trade_set = builder.open_sets["TEST"]
        assert trade_set.learning_book is True
        assert trade_set.experiment_id == "exp-001"

    def test_all_sets_sorted_by_opened_at(self):
        """all_sets() should return sets sorted by opened_at."""
        builder = TradeSetBuilder()

        ts1 = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc)
        ts3 = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        # Create trades in non-chronological order (by symbol)
        builder.process_fill(symbol="ZZZ", side="buy", qty=1.0, price=100.0, timestamp=ts3)
        builder.process_fill(symbol="AAA", side="buy", qty=1.0, price=100.0, timestamp=ts1)
        builder.process_fill(symbol="MMM", side="buy", qty=1.0, price=100.0, timestamp=ts2)

        all_sets = builder.all_sets()

        assert all_sets[0].symbol == "AAA"
        assert all_sets[1].symbol == "MMM"
        assert all_sets[2].symbol == "ZZZ"
