"""Lightweight market-structure utilities (support/resistance/tests/reclaims)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Literal, Sequence

import pandas as pd

TrendLabel = Literal["uptrend", "downtrend", "range", "unclear"]
LevelSide = Literal["support", "resistance"]
TestResult = Literal["successful_test", "failed_test", "reclaim", "breakout", "breakdown"]


@dataclass(slots=True)
class SwingPoint:
    """Local pivot used to seed support/resistance levels."""

    timestamp: datetime
    price: float
    kind: LevelSide


@dataclass(slots=True)
class SupportLevel:
    """Support zone derived from swing lows."""

    price: float
    strength: int = 1
    last_touched: datetime | None = None
    width: float | None = None
    source: str = "swing_low"

    def contains(self, price: float) -> bool:
        buffer = self.width or 0.0
        return self.price - buffer <= price <= self.price + buffer


@dataclass(slots=True)
class ResistanceLevel:
    """Resistance zone derived from swing highs."""

    price: float
    strength: int = 1
    last_touched: datetime | None = None
    width: float | None = None
    source: str = "swing_high"

    def contains(self, price: float) -> bool:
        buffer = self.width or 0.0
        return self.price - buffer <= price <= self.price + buffer


@dataclass(slots=True)
class LevelTestEvent:
    """Single interaction with a level, annotated with outcome."""

    timestamp: datetime
    level: float
    side: LevelSide
    result: TestResult
    attempts: int = 1
    price_close: float | None = None
    window: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "side": self.side,
            "result": self.result,
            "attempts": self.attempts,
            "price_close": self.price_close,
            "window": self.window,
        }


@dataclass(slots=True)
class MarketStructureState:
    """Summary of swing relationships used for regime tagging."""

    trend: TrendLabel
    last_swing_high: float | None = None
    last_swing_low: float | None = None
    higher_high: bool | None = None
    higher_low: bool | None = None
    lower_high: bool | None = None
    lower_low: bool | None = None


@dataclass(slots=True)
class MarketStructureTelemetry:
    """Serializable snapshot for downstream telemetry and prompts."""

    timestamp: datetime
    symbol: str
    timeframe: str
    trend: TrendLabel
    last_swing_high: float | None
    last_swing_low: float | None
    nearest_support: float | None
    nearest_resistance: float | None
    distance_to_support_pct: float | None
    distance_to_resistance_pct: float | None
    support_levels: list[float]
    resistance_levels: list[float]
    recent_tests: list[LevelTestEvent]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "trend": self.trend,
            "last_swing_high": self.last_swing_high,
            "last_swing_low": self.last_swing_low,
            "nearest_support": self.nearest_support,
            "nearest_resistance": self.nearest_resistance,
            "distance_to_support_pct": self.distance_to_support_pct,
            "distance_to_resistance_pct": self.distance_to_resistance_pct,
            "support_levels": self.support_levels,
            "resistance_levels": self.resistance_levels,
            "recent_tests": [event.to_dict() for event in self.recent_tests],
        }


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            working = df.copy()
            working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
            working = working.set_index("timestamp")
        elif "time" in df.columns:
            working = df.copy()
            working["time"] = pd.to_datetime(working["time"], utc=True)
            working = working.set_index("time")
        else:
            raise ValueError("Dataframe must have a datetime index or a time column")
    else:
        working = df.copy()
    return working.sort_index()


def _atr(df: pd.DataFrame, period: int) -> float | None:
    """Minimal ATR calculation used for level buffers."""

    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1)
    atr_series = tr.rolling(window=period, min_periods=period).mean()
    if atr_series.empty or pd.isna(atr_series.iloc[-1]):
        return None
    return float(atr_series.iloc[-1])


def find_swing_points(df: pd.DataFrame, left: int = 2, right: int = 2, lookback: int | None = None) -> list[SwingPoint]:
    """Return swing highs/lows using local window comparisons."""

    working = _prepare_df(df)
    if lookback:
        working = working.tail(lookback)
    swings: list[SwingPoint] = []
    highs = working["high"]
    lows = working["low"]
    for idx in range(left, len(working) - right):
        window = slice(idx - left, idx + right + 1)
        ts = working.index[idx].to_pydatetime()
        high_val = highs.iloc[idx]
        low_val = lows.iloc[idx]
        if high_val == highs.iloc[window].max():
            swings.append(SwingPoint(timestamp=ts, price=float(high_val), kind="resistance"))
        if low_val == lows.iloc[window].min():
            swings.append(SwingPoint(timestamp=ts, price=float(low_val), kind="support"))
    return sorted(swings, key=lambda sp: sp.timestamp)


def _merge_level(
    levels: list[SupportLevel] | list[ResistanceLevel],
    price: float,
    ts: datetime,
    width: float,
    side: LevelSide,
) -> None:
    for level in levels:
        if abs(level.price - price) <= width:
            level.strength += 1
            level.last_touched = ts
            return
    if side == "resistance":
        levels.append(ResistanceLevel(price=price, width=width, last_touched=ts))
    else:
        levels.append(SupportLevel(price=price, width=width, last_touched=ts))


def compute_support_resistance_levels(
    df: pd.DataFrame,
    lookback: int = 150,
    swing_window: int = 2,
    atr_period: int = 14,
    tolerance_mult: float = 0.5,
    max_levels: int = 8,
) -> tuple[list[SupportLevel], list[ResistanceLevel], float | None]:
    """Detect support/resistance levels from swings with ATR-sized buffers."""

    working = _prepare_df(df).tail(lookback)
    if working.empty:
        return [], [], None
    atr_val = _atr(working, atr_period)
    last_price = float(working["close"].iloc[-1])
    base_width = (atr_val or last_price * 0.005) * tolerance_mult
    swings = find_swing_points(working, left=swing_window, right=swing_window)
    supports: list[SupportLevel] = []
    resistances: list[ResistanceLevel] = []
    for swing in swings:
        if swing.kind == "support":
            _merge_level(supports, swing.price, swing.timestamp, base_width, side="support")
        else:
            _merge_level(resistances, swing.price, swing.timestamp, base_width, side="resistance")
    supports = sorted(supports, key=lambda lvl: lvl.price, reverse=True)[:max_levels]
    resistances = sorted(resistances, key=lambda lvl: lvl.price)[:max_levels]
    return supports, resistances, atr_val


def infer_market_structure_state(swings: Sequence[SwingPoint]) -> MarketStructureState:
    """Infer HH/HL vs. LH/LL structure from ordered swings."""

    highs = [sp for sp in swings if sp.kind == "resistance"]
    lows = [sp for sp in swings if sp.kind == "support"]
    last_highs = highs[-2:]
    last_lows = lows[-2:]
    higher_high = None
    higher_low = None
    lower_high = None
    lower_low = None
    if len(last_highs) == 2:
        higher_high = last_highs[1].price > last_highs[0].price
        lower_high = last_highs[1].price < last_highs[0].price
    if len(last_lows) == 2:
        higher_low = last_lows[1].price > last_lows[0].price
        lower_low = last_lows[1].price < last_lows[0].price
    trend: TrendLabel
    if higher_high and higher_low:
        trend = "uptrend"
    elif lower_high and lower_low:
        trend = "downtrend"
    elif higher_high or lower_low:
        trend = "range"
    else:
        trend = "unclear"
    return MarketStructureState(
        trend=trend,
        last_swing_high=last_highs[-1].price if last_highs else None,
        last_swing_low=last_lows[-1].price if last_lows else None,
        higher_high=higher_high,
        higher_low=higher_low,
        lower_high=lower_high,
        lower_low=lower_low,
    )


def _detect_tests_for_level(
    df: pd.DataFrame,
    level_price: float,
    side: LevelSide,
    base_buffer: float,
    attempts: int,
) -> tuple[list[LevelTestEvent], int]:
    events: list[LevelTestEvent] = []
    closes = df["close"]
    highs = df["high"]
    lows = df["low"]
    for ts, high, low, close in zip(df.index, highs, lows, closes):
        if low <= level_price + base_buffer and high >= level_price - base_buffer:
            attempts += 1
            prev_close = closes.shift(1).get(ts)
            if side == "support":
                result: TestResult
                if prev_close is not None and prev_close < level_price <= close:
                    result = "reclaim"
                elif close < level_price:
                    result = "breakdown"
                else:
                    result = "successful_test"
            else:
                if prev_close is not None and prev_close > level_price >= close:
                    result = "reclaim"
                elif close > level_price:
                    result = "breakout"
                else:
                    result = "successful_test"
            event_result: TestResult
            if result == "breakdown":
                event_result = "breakdown"
            else:
                event_result = result
            events.append(
                LevelTestEvent(
                    timestamp=ts.to_pydatetime(),
                    level=float(level_price),
                    side=side,
                    result=event_result,
                    attempts=attempts,
                    price_close=float(close),
                    window=f"{len(df)} bars",
                )
            )
    return events, attempts


def detect_level_tests(
    df: pd.DataFrame,
    supports: Sequence[SupportLevel],
    resistances: Sequence[ResistanceLevel],
    base_buffer: float,
    lookback: int = 50,
) -> list[LevelTestEvent]:
    """Scan recent bars for tests/retests/reclaims at known levels."""

    working = _prepare_df(df).tail(lookback)
    events: list[LevelTestEvent] = []
    attempt_counts: dict[tuple[LevelSide, float], int] = {}
    for support in supports:
        key = ("support", support.price)
        attempts = attempt_counts.get(key, 0)
        detected, attempts = _detect_tests_for_level(working, support.price, "support", support.width or base_buffer, attempts)
        attempt_counts[key] = attempts
        if detected:
            support.last_touched = detected[-1].timestamp
        events.extend(detected)
    for resistance in resistances:
        key = ("resistance", resistance.price)
        attempts = attempt_counts.get(key, 0)
        detected, attempts = _detect_tests_for_level(working, resistance.price, "resistance", resistance.width or base_buffer, attempts)
        attempt_counts[key] = attempts
        if detected:
            resistance.last_touched = detected[-1].timestamp
        events.extend(detected)
    return sorted(events, key=lambda ev: ev.timestamp)


def _nearest_level(price: float, levels: Sequence[SupportLevel] | Sequence[ResistanceLevel], prefer: str) -> float | None:
    if not levels:
        return None
    if prefer == "below":
        below = [lvl.price for lvl in levels if lvl.price <= price]
        return max(below) if below else min(levels, key=lambda lvl: abs(lvl.price - price)).price
    above = [lvl.price for lvl in levels if lvl.price >= price]
    return min(above) if above else min(levels, key=lambda lvl: abs(lvl.price - price)).price


def _distance_pct(price: float, level: float | None) -> float | None:
    if level is None or price == 0:
        return None
    return ((price - level) / price) * 100.0


def build_market_structure_snapshot(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    lookback: int = 200,
    swing_window: int = 2,
    atr_period: int = 14,
    tolerance_mult: float = 0.5,
    max_levels: int = 8,
    test_lookback: int = 50,
) -> MarketStructureTelemetry | None:
    """Bundle structure/level/test telemetry for a single symbol/timeframe."""

    working = _prepare_df(df)
    if working.empty:
        return None
    supports, resistances, atr_val = compute_support_resistance_levels(
        working,
        lookback=lookback,
        swing_window=swing_window,
        atr_period=atr_period,
        tolerance_mult=tolerance_mult,
        max_levels=max_levels,
    )
    swings = find_swing_points(working, left=swing_window, right=swing_window, lookback=lookback)
    structure = infer_market_structure_state(swings)
    last_price = float(working["close"].iloc[-1])
    base_buffer = (atr_val or last_price * 0.005) * tolerance_mult
    tests = detect_level_tests(working, supports, resistances, base_buffer=base_buffer, lookback=test_lookback)
    nearest_support = _nearest_level(last_price, supports, prefer="below")
    nearest_resistance = _nearest_level(last_price, resistances, prefer="above")
    return MarketStructureTelemetry(
        timestamp=working.index[-1].to_pydatetime(),
        symbol=symbol,
        timeframe=timeframe,
        trend=structure.trend,
        last_swing_high=structure.last_swing_high,
        last_swing_low=structure.last_swing_low,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        distance_to_support_pct=_distance_pct(last_price, nearest_support),
        distance_to_resistance_pct=_distance_pct(last_price, nearest_resistance),
        support_levels=[lvl.price for lvl in supports],
        resistance_levels=[lvl.price for lvl in resistances],
        recent_tests=tests[-5:],
    )
