"""Universe screener scoring service and file-backed latest-state store."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import pandas as pd

from metrics import base as metrics_base
from metrics import technical as tech
from agents.llm.client_factory import get_llm_client
from agents.llm.model_utils import output_token_args, reasoning_args, temperature_args
from schemas.screener import (
    InstrumentRecommendation,
    InstrumentRecommendationBatch,
    InstrumentRecommendationGroup,
    InstrumentRecommendationItem,
    ScreenerResult,
    ScreenerSessionPreflight,
    SymbolAnomalyScore,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


class ScreenerStateStore:
    """Persist latest screener result/recommendation for ops API reads."""

    def __init__(
        self,
        result_path: str | Path | None = None,
        recommendation_path: str | Path | None = None,
    ) -> None:
        self.result_path = Path(
            result_path or os.environ.get("SCREENER_RESULT_PATH", "data/screener/latest_result.json")
        )
        self.recommendation_path = Path(
            recommendation_path
            or os.environ.get("SCREENER_RECOMMENDATION_PATH", "data/screener/latest_recommendation.json")
        )

    def save_result(self, result: ScreenerResult) -> None:
        self._write(self.result_path, result.to_json(indent=2))

    def load_result(self) -> ScreenerResult | None:
        raw = self._read(self.result_path)
        if raw is None:
            return None
        return ScreenerResult.model_validate_json(raw)

    def save_recommendation(self, recommendation: InstrumentRecommendation) -> None:
        self._write(self.recommendation_path, recommendation.to_json(indent=2))

    def load_recommendation(self) -> InstrumentRecommendation | None:
        raw = self._read(self.recommendation_path)
        if raw is None:
            return None
        return InstrumentRecommendation.model_validate_json(raw)

    @staticmethod
    def _write(path: Path, payload: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")

    @staticmethod
    def _read(path: Path) -> str | None:
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")


class UniverseScreenerService:
    """Compute anomaly scores over a configurable symbol universe."""

    SUPPORTED_HYPOTHESES = [
        "compression_breakout",
        "volatile_breakout",
        "bull_trending",
        "bear_defensive",
        "range_mean_revert",
        "uncertain_wait",
    ]

    DEFAULT_UNIVERSE = [
        "BTC-USD",
        "ETH-USD",
        "SOL-USD",
        "BNB-USD",
        "XRP-USD",
        "DOGE-USD",
        "AVAX-USD",
        "LINK-USD",
        "DOT-USD",
        "MATIC-USD",
    ]

    def __init__(
        self,
        universe: list[str] | None = None,
        *,
        ohlcv_fetcher: Callable[[str, str, int], Any] | None = None,
        store: ScreenerStateStore | None = None,
        batch_annotation_transport: Callable[[str, str], str] | None = None,
    ) -> None:
        self.universe = universe or self._load_universe_from_env() or list(self.DEFAULT_UNIVERSE)
        self.ohlcv_fetcher = ohlcv_fetcher
        self.store = store or ScreenerStateStore()
        self.batch_annotation_transport = batch_annotation_transport

    @staticmethod
    def _weights() -> dict[str, float]:
        compression = _safe_float(os.environ.get("SCREENER_COMPRESSION_WEIGHT", "0.50"), 0.50)
        expansion = _safe_float(os.environ.get("SCREENER_EXPANSION_WEIGHT", "0.50"), 0.50)
        total = compression + expansion
        if total <= 0:
            return {"compression": 0.5, "expansion": 0.5}
        return {"compression": compression / total, "expansion": expansion / total}

    @staticmethod
    def _top_n() -> int:
        try:
            return max(1, int(os.environ.get("SCREENER_TOP_N", "8")))
        except ValueError:
            return 8

    @staticmethod
    def _load_universe_from_env() -> list[str] | None:
        path_str = os.environ.get("SCREENER_UNIVERSE_FILE")
        if not path_str:
            return None
        path = Path(path_str)
        if not path.exists():
            logger.warning("Screener universe file not found: %s", path)
            return None
        try:
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                symbols = payload.get("symbols")
            else:
                symbols = payload
            if not isinstance(symbols, list):
                raise ValueError("Universe file must be a list[str] or {'symbols': [...]}")
            cleaned = [str(sym).strip() for sym in symbols if str(sym).strip()]
            return cleaned or None
        except Exception as exc:
            logger.warning("Failed to load screener universe file %s: %s", path, exc)
            return None

    async def screen(self, timeframe: str = "1h", lookback_bars: int = 50) -> ScreenerResult:
        """Score all symbols and return a ranked candidate list."""
        scores: list[SymbolAnomalyScore] = []
        for symbol in self.universe:
            try:
                scores.append(await self._score_symbol(symbol, timeframe, lookback_bars))
            except Exception as exc:
                logger.warning("Screener failed to score %s: %s", symbol, exc)
        scores.sort(key=lambda item: item.composite_score, reverse=True)
        weights = self._weights()
        top_n = self._top_n()
        return ScreenerResult(
            run_id=str(uuid4()),
            as_of=_utcnow(),
            universe_size=len(scores),
            top_candidates=scores[:top_n],
            screener_config={
                "weights": weights,
                "top_n": top_n,
                "timeframe": timeframe,
                "lookback_bars": lookback_bars,
                "template_thresholds": {
                    "compression_score_gt": 0.60,
                    "expansion_score_gt": 0.55,
                    "low_anomaly_composite_lt": 0.30,
                },
            },
        )

    def recommend_from_result(self, result: ScreenerResult, timeframe: str | None = None) -> InstrumentRecommendation:
        """Build a deterministic recommendation fallback from screener candidates.

        This ships the runbook contract without requiring live LLM availability in tests.
        The prompt in ``prompts/instrument_recommendation.txt`` can be used by a later
        online recommender step without changing the data contracts.
        """
        if not result.top_candidates:
            raise ValueError("Cannot recommend instrument from empty screener result")

        selected = self._pick_candidate(result.top_candidates)
        template_id = self._candidate_template_id(selected)
        strategy_type = template_id or "volatile_breakout"
        prior_high = self._reconstruct_level(selected.close, selected.dist_to_prior_high_pct)
        prior_low = self._reconstruct_level(selected.close, selected.dist_to_prior_low_pct)
        key_levels = {
            "support": round(prior_low, 8),
            "resistance": round(prior_high, 8),
            "pivot": round((prior_high + prior_low) / 2.0, 8),
        }

        reasons = {
            c.symbol: self._disqualify_reason(c)
            for c in result.top_candidates
            if c.symbol != selected.symbol
        }
        confidence = "high" if selected.composite_score >= 0.7 else "medium" if selected.composite_score >= 0.45 else "low"
        hold_tf = timeframe or str(result.screener_config.get("timeframe") or "1h")

        thesis = (
            f"{selected.symbol} ranks highest with composite={selected.composite_score:.2f} "
            f"(compression={_safe_float(selected.score_components.get('compression_score')):.2f}, "
            f"expansion={_safe_float(selected.score_components.get('expansion_score')):.2f}). "
            f"Trend={selected.trend_state}, vol={selected.vol_state}; prioritize a {strategy_type} setup."
        )
        return InstrumentRecommendation(
            selected_symbol=selected.symbol,
            thesis=thesis,
            strategy_type=strategy_type,
            template_id=template_id,
            regime_view=f"{selected.trend_state}/{selected.vol_state}",
            key_levels=key_levels,
            expected_hold_timeframe=hold_tf,
            confidence=confidence,  # type: ignore[arg-type]
            disqualified_symbols=sorted(reasons.keys()),
            disqualification_reasons=reasons,
        )

    def build_recommendation_batch(
        self,
        result: ScreenerResult,
        *,
        max_per_group: int | None = None,
        annotate_with_llm: bool = False,
    ) -> InstrumentRecommendationBatch:
        """Group shortlist candidates by supported strategy hypothesis + timeframe."""
        limit = max_per_group or self._max_recommendations_per_group()
        ranked = sorted(result.top_candidates, key=lambda c: c.composite_score, reverse=True)
        grouped: dict[tuple[str, str], list[InstrumentRecommendationItem]] = {}
        group_order: list[tuple[str, str]] = []

        for idx, candidate in enumerate(ranked, start=1):
            hypothesis = self._candidate_hypothesis(candidate)
            timeframe = self._candidate_hold_timeframe(candidate, hypothesis)
            key = (hypothesis, timeframe)
            if key not in grouped:
                grouped[key] = []
                group_order.append(key)
            if len(grouped[key]) >= limit:
                continue
            item = InstrumentRecommendationItem(
                symbol=candidate.symbol,
                hypothesis=hypothesis,
                template_id=self._candidate_template_id(candidate),
                expected_hold_timeframe=timeframe,
                thesis=self._candidate_thesis(candidate, hypothesis, timeframe),
                confidence=self._candidate_confidence(candidate),  # type: ignore[arg-type]
                composite_score=float(candidate.composite_score),
                key_levels=self._candidate_key_levels(candidate),
                rank_global=idx,
                rank_in_group=len(grouped[key]) + 1,
                score_components=dict(candidate.score_components or {}),
            )
            grouped[key].append(item)

        groups: list[InstrumentRecommendationGroup] = []
        for hypothesis, timeframe in group_order:
            items = grouped[(hypothesis, timeframe)]
            if not items:
                continue
            groups.append(
                InstrumentRecommendationGroup(
                    hypothesis=hypothesis,
                    timeframe=timeframe,
                    template_id=items[0].template_id or hypothesis,
                    label=self._group_label(hypothesis, timeframe),
                    rationale=self._group_rationale(hypothesis, timeframe, items),
                    recommendations=items,
                )
            )

        # Stable UI ordering: actionability first, then watchlist.
        groups.sort(
            key=lambda g: (
                self._hypothesis_priority(g.hypothesis),
                self._timeframe_priority(g.timeframe),
                -(g.recommendations[0].composite_score if g.recommendations else 0.0),
            )
        )

        batch = InstrumentRecommendationBatch(
            run_id=result.run_id,
            as_of=result.as_of,
            supported_hypotheses=list(self.SUPPORTED_HYPOTHESES),
            max_per_group=limit,
            total_candidates_considered=len(ranked),
            groups=groups,
        )
        if annotate_with_llm or self._screener_llm_annotation_enabled():
            return self._annotate_batch_with_llm(batch, result)
        return batch

    def build_session_preflight(self, result: ScreenerResult, *, mode: str = "paper") -> ScreenerSessionPreflight:
        """Compose a session-start shortlist payload for UI preflight."""
        batch = self.build_recommendation_batch(result)
        top_item = next((item for group in batch.groups for item in group.recommendations), None)
        notes = [
            "Shortlist is grouped by supported strategy hypotheses and expected hold timeframe.",
            "Use screener weights to bias which hypotheses surface more frequently.",
        ]
        if mode == "live":
            notes.append("Live mode: require explicit operator confirmation before selecting a symbol/template.")
        return ScreenerSessionPreflight(
            mode="live" if str(mode).lower() == "live" else "paper",
            as_of=result.as_of,
            screener_run_id=result.run_id,
            shortlist=batch,
            suggested_default_symbol=(top_item.symbol if top_item else None),
            suggested_default_template_id=(top_item.template_id if top_item else None),
            notes=notes,
        )

    def build_session_preflight_with_options(
        self,
        result: ScreenerResult,
        *,
        mode: str = "paper",
        annotate_with_llm: bool = False,
    ) -> ScreenerSessionPreflight:
        """Compose session preflight with optional LLM annotation/re-ranking."""
        batch = self.build_recommendation_batch(result, annotate_with_llm=annotate_with_llm)
        preflight = self.build_session_preflight(result, mode=mode)
        preflight.shortlist = batch
        preflight.suggested_default_symbol = next(
            (item.symbol for group in batch.groups for item in group.recommendations),
            preflight.suggested_default_symbol,
        )
        preflight.suggested_default_template_id = next(
            (item.template_id for group in batch.groups for item in group.recommendations if item.template_id),
            preflight.suggested_default_template_id,
        )
        if batch.annotation_meta and batch.annotation_meta.get("applied"):
            preflight.notes = [
                "LLM annotation/re-ranking applied on top of deterministic shortlist.",
                *preflight.notes,
            ]
        return preflight

    def persist_latest(
        self,
        result: ScreenerResult,
        recommendation: InstrumentRecommendation | None = None,
    ) -> None:
        self.store.save_result(result)
        if recommendation is not None:
            self.store.save_recommendation(recommendation)

    async def _score_symbol(self, symbol: str, timeframe: str, lookback_bars: int) -> SymbolAnomalyScore:
        df = await self._fetch_ohlcv_df(symbol, timeframe, lookback_bars)
        if len(df) < 30:
            raise ValueError(f"Insufficient data for {symbol}: {len(df)} bars")
        df = metrics_base.prepare_ohlcv_df(df)

        vol = df["volume"].fillna(0.0)
        bar_range = (df["high"] - df["low"]).abs().fillna(0.0)

        volume_z = self._zscore_last(vol, 20, clamp_abs=5.0)
        range_expansion_z = self._zscore_last(bar_range, 20, clamp_abs=5.0)

        atr_series = tech.atr(df, 14).series_list[0].series.fillna(0.0)
        atr_short = _safe_float(atr_series.rolling(5, min_periods=1).mean().iloc[-1], 0.0)
        atr_long = _safe_float(atr_series.rolling(20, min_periods=1).mean().iloc[-1], 0.0)
        atr_expansion = ((atr_short / max(atr_long, 1e-9)) - 1.0) if atr_long > 0 else 0.0
        atr_expansion = _clamp(_safe_float(atr_expansion), -3.0, 3.0)

        bb_result = tech.bollinger_bands(df, 20, 2.0)
        bandwidth = next(series.series for series in bb_result.series_list if series.key == "bandwidth").fillna(0.0)
        bb_rank = _safe_float(bandwidth.rank(pct=True, method="average").iloc[-1], 0.5)
        bb_rank = _clamp(bb_rank, 0.0, 1.0)

        compression_score = _clamp(1.0 - bb_rank, 0.0, 1.0)
        expansion_score = (
            0.45 * _clamp(max(volume_z, 0.0) / 3.0, 0.0, 1.0)
            + 0.35 * _clamp(max(atr_expansion, 0.0) / 1.0, 0.0, 1.0)
            + 0.20 * _clamp(max(range_expansion_z, 0.0) / 3.0, 0.0, 1.0)
        )

        weights = self._weights()
        composite_score = (
            weights["compression"] * compression_score + weights["expansion"] * expansion_score
        )
        composite_score = _clamp(_safe_float(composite_score), 0.0, 1.0)

        prior_high = _safe_float(df["high"].iloc[-2], _safe_float(df["high"].iloc[-1], 0.0))
        prior_low = _safe_float(df["low"].iloc[-2], _safe_float(df["low"].iloc[-1], 0.0))
        close = _safe_float(df["close"].iloc[-1], 0.0)
        dist_to_prior_high_pct = self._pct_distance(close, prior_high)
        dist_to_prior_low_pct = self._pct_distance(close, prior_low)

        trend_state = self._classify_trend(df)
        vol_state = self._classify_vol(atr_series, close)
        template_id_suggestion = self._template_id_from_scores(
            compression_score=compression_score,
            expansion_score=expansion_score,
            trend_state=trend_state,
            composite_score=composite_score,
        )

        return SymbolAnomalyScore(
            symbol=symbol,
            as_of=_utcnow(),
            volume_z=round(volume_z, 6),
            atr_expansion=round(atr_expansion, 6),
            range_expansion_z=round(range_expansion_z, 6),
            bb_bandwidth_pct_rank=round(bb_rank, 6),
            close=close,
            trend_state=trend_state,
            vol_state=vol_state,
            dist_to_prior_high_pct=round(dist_to_prior_high_pct, 6),
            dist_to_prior_low_pct=round(dist_to_prior_low_pct, 6),
            composite_score=round(composite_score, 6),
            score_components={
                "compression_score": round(compression_score, 6),
                "expansion_score": round(expansion_score, 6),
                "volume_component": round(_clamp(max(volume_z, 0.0) / 3.0, 0.0, 1.0), 6),
                "atr_component": round(_clamp(max(atr_expansion, 0.0), 0.0, 1.0), 6),
                "range_component": round(_clamp(max(range_expansion_z, 0.0) / 3.0, 0.0, 1.0), 6),
                "weights": weights,
                "template_id_suggestion": template_id_suggestion,
            },
        )

    async def _fetch_ohlcv_df(self, symbol: str, timeframe: str, lookback_bars: int) -> pd.DataFrame:
        if self.ohlcv_fetcher is not None:
            payload = self.ohlcv_fetcher(symbol, timeframe, lookback_bars)
            if inspect.isawaitable(payload):
                payload = await payload
            return self._coerce_to_dataframe(payload)

        lookback_days = self._bars_to_lookback_days(timeframe, lookback_bars)
        from services.market_data_worker import fetch_ohlcv_history

        candles = await asyncio.to_thread(fetch_ohlcv_history, symbol, timeframe, lookback_days)
        return self._coerce_to_dataframe(candles)

    @staticmethod
    def _bars_to_lookback_days(timeframe: str, lookback_bars: int) -> int:
        tf = str(timeframe).strip().lower()
        if tf.endswith("m"):
            minutes = max(1, int(tf[:-1]))
            total_minutes = minutes * max(lookback_bars, 1)
            return max(1, math.ceil(total_minutes / (24 * 60)))
        if tf.endswith("h"):
            hours = max(1, int(tf[:-1]))
            total_hours = hours * max(lookback_bars, 1)
            return max(2, math.ceil(total_hours / 24))
        if tf.endswith("d"):
            days = max(1, int(tf[:-1]))
            return max(days * max(lookback_bars, 1), 2)
        return 7

    @staticmethod
    def _coerce_to_dataframe(payload: Any) -> pd.DataFrame:
        if isinstance(payload, pd.DataFrame):
            return payload.copy()
        if payload is None:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        rows: list[dict[str, Any]] = []
        for item in payload:
            if isinstance(item, dict):
                rows.append(dict(item))
            elif is_dataclass(item):
                rows.append(asdict(item))
            else:
                rows.append(
                    {
                        "timestamp": getattr(item, "timestamp", None),
                        "open": getattr(item, "open", None),
                        "high": getattr(item, "high", None),
                        "low": getattr(item, "low", None),
                        "close": getattr(item, "close", None),
                        "volume": getattr(item, "volume", None),
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _zscore_last(series: pd.Series, window: int, *, clamp_abs: float | None = None) -> float:
        values = pd.to_numeric(series, errors="coerce").fillna(0.0)
        roll_mean = values.rolling(window, min_periods=window).mean()
        roll_std = values.rolling(window, min_periods=window).std(ddof=0)
        std = _safe_float(roll_std.iloc[-1], 0.0)
        if std <= 0:
            z = 0.0
        else:
            z = (_safe_float(values.iloc[-1]) - _safe_float(roll_mean.iloc[-1])) / std
        if clamp_abs is not None:
            z = _clamp(z, -clamp_abs, clamp_abs)
        return _safe_float(z, 0.0)

    @staticmethod
    def _pct_distance(price: float, anchor: float) -> float:
        if not anchor:
            return 0.0
        return ((price - anchor) / anchor) * 100.0

    @staticmethod
    def _classify_trend(df: pd.DataFrame) -> str:
        close = df["close"].astype("float64")
        ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        last = _safe_float(close.iloc[-1], 0.0)
        if last <= 0:
            return "unclear"
        spread_pct = abs(ema20 - ema50) / last
        if spread_pct < 0.002:
            return "range"
        if last > ema20 > ema50:
            return "uptrend"
        if last < ema20 < ema50:
            return "downtrend"
        return "unclear"

    @staticmethod
    def _classify_vol(atr_series: pd.Series, close: float) -> str:
        if close <= 0:
            return "normal"
        atr_pct = pd.to_numeric(atr_series, errors="coerce").fillna(0.0) / max(close, 1e-9)
        current = _safe_float(atr_pct.iloc[-1], 0.0)
        baseline = _safe_float(atr_pct.rolling(20, min_periods=5).median().iloc[-1], 0.0)
        if baseline <= 0:
            baseline = max(current, 1e-6)
        ratio = current / baseline if baseline else 1.0
        if ratio < 0.8:
            return "low"
        if ratio < 1.25:
            return "normal"
        if ratio < 1.8:
            return "high"
        return "extreme"

    @classmethod
    def _template_id_from_scores(
        cls,
        *,
        compression_score: float,
        expansion_score: float,
        trend_state: str,
        composite_score: float,
    ) -> str | None:
        if compression_score > 0.60:
            return "compression_breakout"
        if expansion_score > 0.55 and trend_state in {"uptrend", "unclear"}:
            return "bull_trending" if trend_state == "uptrend" else "volatile_breakout"
        if expansion_score > 0.55 and trend_state == "downtrend":
            return "bear_defensive"
        if composite_score < 0.30:
            return "uncertain_wait"
        return None

    def _candidate_template_id(self, candidate: SymbolAnomalyScore) -> str | None:
        hint = candidate.score_components.get("template_id_suggestion")
        return str(hint) if hint else None

    def _candidate_hypothesis(self, candidate: SymbolAnomalyScore) -> str:
        template_id = self._candidate_template_id(candidate)
        if template_id in self.SUPPORTED_HYPOTHESES:
            return template_id
        # Route non-template suggestions into supported groups for UI consistency.
        if candidate.trend_state == "range":
            return "range_mean_revert"
        if candidate.vol_state in {"high", "extreme"} or candidate.atr_expansion > 0.25:
            return "volatile_breakout"
        if candidate.trend_state == "uptrend":
            return "bull_trending"
        if candidate.trend_state == "downtrend":
            return "bear_defensive"
        return "uncertain_wait"

    def _candidate_hold_timeframe(self, candidate: SymbolAnomalyScore, hypothesis: str) -> str:
        if hypothesis == "compression_breakout":
            return "1h" if candidate.vol_state in {"normal", "high"} else "4h"
        if hypothesis == "volatile_breakout":
            return "15m" if candidate.atr_expansion > 0.4 or candidate.vol_state == "extreme" else "1h"
        if hypothesis == "range_mean_revert":
            return "15m" if abs(candidate.volume_z) < 1.5 else "1h"
        if hypothesis in {"bull_trending", "bear_defensive"}:
            return "4h" if candidate.trend_state in {"uptrend", "downtrend"} else "1h"
        return "1h"

    def _candidate_confidence(self, candidate: SymbolAnomalyScore) -> str:
        if candidate.composite_score >= 0.75:
            return "high"
        if candidate.composite_score >= 0.45:
            return "medium"
        return "low"

    def _candidate_key_levels(self, candidate: SymbolAnomalyScore) -> dict[str, float]:
        prior_high = self._reconstruct_level(candidate.close, candidate.dist_to_prior_high_pct)
        prior_low = self._reconstruct_level(candidate.close, candidate.dist_to_prior_low_pct)
        return {
            "support": round(prior_low, 8),
            "resistance": round(prior_high, 8),
            "pivot": round((prior_low + prior_high) / 2.0, 8),
        }

    def _candidate_thesis(self, candidate: SymbolAnomalyScore, hypothesis: str, timeframe: str) -> str:
        return (
            f"{candidate.symbol}: {hypothesis} candidate for {timeframe} hold window. "
            f"Composite={candidate.composite_score:.2f}, trend={candidate.trend_state}, vol={candidate.vol_state}, "
            f"compression={_safe_float(candidate.score_components.get('compression_score')):.2f}, "
            f"expansion={_safe_float(candidate.score_components.get('expansion_score')):.2f}."
        )

    @classmethod
    def _group_label(cls, hypothesis: str, timeframe: str) -> str:
        label_map = {
            "compression_breakout": "Compression Breakout",
            "volatile_breakout": "Volatile Breakout",
            "bull_trending": "Bull Trending",
            "bear_defensive": "Bear Defensive",
            "range_mean_revert": "Range Mean Revert",
            "uncertain_wait": "Uncertain / Wait",
        }
        return f"{label_map.get(hypothesis, hypothesis)} ({timeframe})"

    @classmethod
    def _group_rationale(
        cls,
        hypothesis: str,
        timeframe: str,
        items: list[InstrumentRecommendationItem],
    ) -> str:
        best = items[0]
        return (
            f"{len(items)} candidates routed to {hypothesis} on {timeframe}; "
            f"top candidate {best.symbol} scored {best.composite_score:.2f}."
        )

    @staticmethod
    def _hypothesis_priority(hypothesis: str) -> int:
        order = {
            "compression_breakout": 0,
            "volatile_breakout": 1,
            "bull_trending": 2,
            "range_mean_revert": 3,
            "bear_defensive": 4,
            "uncertain_wait": 5,
        }
        return order.get(hypothesis, 99)

    @staticmethod
    def _timeframe_priority(timeframe: str) -> int:
        order = {"15m": 0, "1h": 1, "4h": 2}
        return order.get(timeframe, 99)

    @staticmethod
    def _max_recommendations_per_group() -> int:
        try:
            return max(1, min(10, int(os.environ.get("SCREENER_MAX_RECOMMENDATIONS_PER_GROUP", "10"))))
        except ValueError:
            return 10

    @staticmethod
    def _screener_llm_annotation_enabled() -> bool:
        return os.environ.get("SCREENER_LLM_ANNOTATION_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}

    def _annotate_batch_with_llm(
        self,
        batch: InstrumentRecommendationBatch,
        result: ScreenerResult,
    ) -> InstrumentRecommendationBatch:
        """Optionally annotate/rerank a deterministic batch using an LLM, with strict fallback."""
        prompt = self._load_instrument_recommendation_prompt()
        user_payload = {
            "task": "Annotate and optionally rerank grouped screener shortlist without adding new symbols.",
            "constraints": {
                "keep_supported_hypotheses_only": True,
                "max_per_group": batch.max_per_group,
                "allowed_symbols": sorted({item.symbol for group in batch.groups for item in group.recommendations}),
                "allowed_hypotheses": batch.supported_hypotheses,
                "allowed_timeframes": sorted({item.expected_hold_timeframe for group in batch.groups for item in group.recommendations}),
            },
            "screener_result": result.model_dump(mode="json"),
            "deterministic_batch": batch.model_dump(mode="json"),
            "output_contract": "Return valid JSON matching InstrumentRecommendationBatch exactly.",
        }
        payload_json = json.dumps(user_payload, ensure_ascii=True)
        try:
            raw = self._run_screener_annotation_llm(prompt, payload_json)
            candidate = self._parse_batch_annotation_json(raw)
            validated = self._validate_llm_annotated_batch(candidate, baseline=batch)
            validated.annotation_meta = {
                "applied": True,
                "mode": "llm_annotation_rerank",
                "model": os.environ.get("SCREENER_LLM_MODEL") or os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
            }
            validated.source = "llm_annotated_screener_grouping"
            return validated
        except Exception as exc:
            logger.warning("Screener LLM annotation failed; using deterministic batch: %s", exc)
            fallback = batch.model_copy(deep=True)
            fallback.annotation_meta = {
                "applied": False,
                "mode": "deterministic_fallback",
                "error": str(exc),
            }
            return fallback

    def _run_screener_annotation_llm(self, system_prompt: str, user_payload_json: str) -> str:
        if self.batch_annotation_transport is not None:
            return self.batch_annotation_transport(system_prompt, user_payload_json)

        model = os.environ.get("SCREENER_LLM_MODEL") or os.environ.get("OPENAI_MODEL", "gpt-5-mini")
        client = get_llm_client()
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload_json},
            ],
            **output_token_args(model, 5000),
            **temperature_args(model, 0.1),
            **reasoning_args(model, effort="low"),
        )
        content = getattr(response, "output_text", None) or ""
        if not content:
            raise ValueError("LLM annotation returned empty output")
        return content

    @staticmethod
    def _load_instrument_recommendation_prompt() -> str:
        path = Path(__file__).resolve().parents[1] / "prompts" / "instrument_recommendation.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing prompt: {path}")
        return path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _parse_batch_annotation_json(raw: str) -> InstrumentRecommendationBatch:
        try:
            return InstrumentRecommendationBatch.model_validate_json(raw)
        except Exception:
            pass
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
        if fenced:
            return InstrumentRecommendationBatch.model_validate_json(fenced.group(1))
        first = raw.find("{")
        last = raw.rfind("}")
        if first >= 0 and last > first:
            return InstrumentRecommendationBatch.model_validate_json(raw[first:last + 1])
        raise ValueError("No JSON object found in LLM annotation output")

    @staticmethod
    def _validate_llm_annotated_batch(
        candidate: InstrumentRecommendationBatch,
        *,
        baseline: InstrumentRecommendationBatch,
    ) -> InstrumentRecommendationBatch:
        allowed_symbols = {item.symbol for group in baseline.groups for item in group.recommendations}
        allowed_hypotheses = set(baseline.supported_hypotheses)
        allowed_pairs = {
            (item.symbol, item.hypothesis, item.expected_hold_timeframe)
            for group in baseline.groups
            for item in group.recommendations
        }
        seen_per_group: dict[tuple[str, str], set[str]] = {}
        for group in candidate.groups:
            if group.hypothesis not in allowed_hypotheses:
                raise ValueError(f"Unsupported hypothesis from LLM: {group.hypothesis}")
            if len(group.recommendations) > baseline.max_per_group:
                raise ValueError(f"LLM exceeded max_per_group for {group.hypothesis}/{group.timeframe}")
            key = (group.hypothesis, group.timeframe)
            seen_per_group.setdefault(key, set())
            for idx, item in enumerate(group.recommendations, start=1):
                if item.symbol not in allowed_symbols:
                    raise ValueError(f"LLM introduced unknown symbol: {item.symbol}")
                if item.hypothesis != group.hypothesis or item.expected_hold_timeframe != group.timeframe:
                    raise ValueError("LLM item/group mismatch on hypothesis/timeframe")
                if (item.symbol, item.hypothesis, item.expected_hold_timeframe) not in allowed_pairs:
                    raise ValueError(f"LLM introduced unsupported symbol-group assignment: {item.symbol}")
                if item.symbol in seen_per_group[key]:
                    raise ValueError(f"LLM duplicated symbol in group: {item.symbol}")
                seen_per_group[key].add(item.symbol)
                item.rank_in_group = idx
        candidate.max_per_group = baseline.max_per_group
        candidate.supported_hypotheses = list(baseline.supported_hypotheses)
        candidate.total_candidates_considered = baseline.total_candidates_considered
        candidate.run_id = baseline.run_id
        candidate.as_of = baseline.as_of
        return candidate

    def _pick_candidate(self, candidates: list[SymbolAnomalyScore]) -> SymbolAnomalyScore:
        preferred = [c for c in candidates if c.vol_state != "extreme"]
        pool = preferred or candidates
        return sorted(pool, key=lambda c: c.composite_score, reverse=True)[0]

    def _disqualify_reason(self, candidate: SymbolAnomalyScore) -> str:
        if candidate.vol_state == "extreme":
            return "Volatility is extreme; skip unless explicitly trading a volatility thesis."
        if candidate.bb_bandwidth_pct_rank > 0.8 and candidate.atr_expansion <= 0:
            return "Already expanded without fresh ATR confirmation."
        if candidate.volume_z < 0.5 and candidate.range_expansion_z < 0.5:
            return "Anomaly score is mostly structural compression without immediate participation."
        return "Lower composite score than selected symbol."

    @staticmethod
    def _reconstruct_level(close: float, dist_pct: float) -> float:
        denom = 1.0 + (dist_pct / 100.0)
        if abs(denom) < 1e-9:
            return close
        return close / denom
