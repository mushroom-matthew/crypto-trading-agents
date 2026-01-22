# Scalper Mode Implementation Plan

This document outlines the implementation plan for "Scalper Mode" - a comprehensive set of features enabling aggressive short-term trading with configurable parameters, leverage comparison, market-driven triggers, and walk-away thresholds.

## Feature Overview

| Feature | Priority | Complexity | Status |
|---------|----------|------------|--------|
| 1. Aggressive Trading Parameters UI | P0 | Medium | Planned |
| 2. Short-Term Timeframe Support (5m/15m) | P0 | Low | Planned |
| 3. Whipsaw Removal Controls | P0 | Medium | Planned |
| 4. Leverage Comparison Backtests | P1 | High | Planned |
| 5. Walk-Away Threshold | P1 | Medium | Planned |
| 6. Market-Driven Strategy Triggers | P2 | High | Planned |
| 7. Real Leverage Trading Roadmap | P2 | High | Planned |

---

## 1. Aggressive Trading Parameters UI

### Goal
Expose all aggressive trading parameters in both BacktestControl and PaperTradingControl UI components.

### New Parameters to Expose

#### Risk Engine Parameters
```typescript
interface AggressiveRiskConfig {
  // Position sizing
  max_position_risk_pct: number       // 1-10%, default 2%
  max_symbol_exposure_pct: number     // 10-100%, default 25%
  max_portfolio_exposure_pct: number  // 50-500%, default 80%
  max_daily_loss_pct: number          // 1-20%, default 3%
  max_daily_risk_budget_pct: number   // 5-30%, default 10%

  // Trade frequency
  max_trades_per_day: number          // 1-100, default 10
  max_triggers_per_symbol_per_day: number  // 1-20, default 5
}
```

#### Trigger Engine Parameters
```typescript
interface WhipsawConfig {
  min_hold_bars: number               // 0-10, default 4
  trade_cooldown_bars: number         // 0-5, default 2
  confidence_override_threshold: 'A' | 'B' | 'C' | null  // default 'A'
}
```

#### Execution Gating Parameters
```typescript
interface ExecutionGatingConfig {
  min_price_move_pct: number          // 0.01-2%, default 0.5%
  max_staleness_seconds: number       // 60-7200, default 1800
  max_calls_per_hour_per_symbol: number  // 10-200, default 60
}
```

### API Changes

#### Backend: `ops_api/routers/backtests.py`

Add to `BacktestConfig`:
```python
class BacktestConfig(BaseModel):
    # ... existing fields ...

    # Risk Engine
    max_position_risk_pct: Optional[float] = None
    max_symbol_exposure_pct: Optional[float] = None
    max_portfolio_exposure_pct: Optional[float] = None
    max_daily_loss_pct: Optional[float] = None
    max_daily_risk_budget_pct: Optional[float] = None
    max_trades_per_day: Optional[int] = None
    max_triggers_per_symbol_per_day: Optional[int] = None

    # Whipsaw Controls
    min_hold_bars: Optional[int] = None
    trade_cooldown_bars: Optional[int] = None
    confidence_override_threshold: Optional[str] = None

    # Execution Gating
    min_price_move_pct: Optional[float] = None
    max_staleness_seconds: Optional[int] = None
```

#### Backend: `ops_api/routers/paper_trading.py`

Add same fields to `PaperTradingSessionConfig`.

### UI Changes

#### `ui/src/components/BacktestControl.tsx`

Add collapsible "Advanced Settings" panel:

```tsx
<CollapsibleSection title="Aggressive Trading Settings" defaultOpen={false}>
  <div className="grid grid-cols-2 gap-4">
    {/* Risk Parameters */}
    <ParameterSlider
      label="Position Risk %"
      value={config.max_position_risk_pct}
      min={1} max={10} step={0.5}
      onChange={(v) => setConfig({...config, max_position_risk_pct: v})}
    />
    <ParameterSlider
      label="Portfolio Exposure %"
      value={config.max_portfolio_exposure_pct}
      min={50} max={500} step={10}
      onChange={(v) => setConfig({...config, max_portfolio_exposure_pct: v})}
    />

    {/* Whipsaw Controls */}
    <ParameterSlider
      label="Min Hold Bars"
      value={config.min_hold_bars}
      min={0} max={10} step={1}
      onChange={(v) => setConfig({...config, min_hold_bars: v})}
      tooltip="Set to 0 to disable minimum hold period (allows rapid exits)"
    />
    <ParameterSlider
      label="Trade Cooldown Bars"
      value={config.trade_cooldown_bars}
      min={0} max={5} step={1}
      onChange={(v) => setConfig({...config, trade_cooldown_bars: v})}
      tooltip="Set to 0 to disable cooldown (allows immediate re-entry)"
    />

    {/* ... more parameters ... */}
  </div>
</CollapsibleSection>
```

### Files to Modify

| File | Changes |
|------|---------|
| `ops_api/routers/backtests.py` | Add new config fields, pass to simulator |
| `ops_api/routers/paper_trading.py` | Add new config fields, pass to workflow |
| `backtesting/llm_strategist_runner.py` | Accept whipsaw/gating params |
| `agents/strategies/trigger_engine.py` | Accept configurable defaults |
| `ui/src/components/BacktestControl.tsx` | Add advanced settings panel |
| `ui/src/components/PaperTradingControl.tsx` | Add advanced settings panel |
| `ui/src/lib/api.ts` | Update TypeScript interfaces |

---

## 2. Short-Term Timeframe Support

### Goal
Enable 5-minute and 15-minute candles for high-frequency trading with appropriate indicator calculations.

### Current State
- Timeframes already supported: `1m`, `5m`, `15m`, `1h`, `4h`, `1d`
- UI dropdown already includes 5m and 15m options
- Indicator calculations work on any timeframe

### Required Changes

#### Multi-Timeframe Indicator Enhancement

For short-term trading, we need faster indicator periods:

```python
# Current (designed for 1h+)
INDICATOR_PERIODS = {
    'sma_short': 20,
    'sma_medium': 50,
    'rsi_period': 14,
    'atr_period': 14,
}

# Short-term mode (for 5m/15m)
SHORT_TERM_INDICATOR_PERIODS = {
    'sma_short': 8,       # ~40 minutes on 5m
    'sma_medium': 21,     # ~1.75 hours on 5m
    'rsi_period': 7,      # Faster RSI
    'atr_period': 7,      # Faster ATR
}
```

#### New: Timeframe-Aware Indicator Config

Add to `BacktestConfig`:
```python
class BacktestConfig(BaseModel):
    # ... existing ...
    indicator_mode: Literal['standard', 'short_term', 'custom'] = 'standard'
    custom_indicator_periods: Optional[Dict[str, int]] = None
```

#### UI: Timeframe Presets

Add "Scalper Mode" preset to `ui/src/lib/presets.ts`:

```typescript
export const PRESETS: Record<string, BacktestPreset> = {
  // ... existing presets ...

  'scalper-5m': {
    id: 'scalper-5m',
    name: 'Scalper Mode (5m)',
    description: 'High-frequency 5-minute candles for intraday scalping',
    config: {
      symbols: ['BTC-USD'],
      timeframe: '5m',
      start_date: getDateDaysAgo(3),  // 3 days of 5m data
      end_date: getTodayDate(),
      initial_cash: 10000,
      strategy: 'llm_strategist',
      strategy_id: 'aggressive_active',
      // Aggressive settings
      max_position_risk_pct: 4.0,
      max_portfolio_exposure_pct: 100,
      min_hold_bars: 1,
      trade_cooldown_bars: 0,
      min_price_move_pct: 0.1,
      indicator_mode: 'short_term',
    }
  },

  'scalper-15m': {
    id: 'scalper-15m',
    name: 'Scalper Mode (15m)',
    description: '15-minute candles for short-term momentum trades',
    config: {
      symbols: ['BTC-USD', 'ETH-USD'],
      timeframe: '15m',
      start_date: getDateDaysAgo(7),
      end_date: getTodayDate(),
      initial_cash: 10000,
      strategy: 'llm_strategist',
      strategy_id: 'aggressive_active',
      max_position_risk_pct: 3.0,
      max_portfolio_exposure_pct: 100,
      min_hold_bars: 2,
      trade_cooldown_bars: 1,
      min_price_move_pct: 0.2,
      indicator_mode: 'short_term',
    }
  },
}
```

### Files to Modify

| File | Changes |
|------|---------|
| `backtesting/llm_strategist_runner.py` | Add indicator_mode parameter |
| `ops_api/routers/backtests.py` | Add indicator_mode to config |
| `ui/src/lib/presets.ts` | Add scalper presets |
| `ui/src/components/BacktestControl.tsx` | Show indicator mode selector |

---

## 3. Whipsaw Removal Controls

### Goal
Make anti-flip-flop settings fully configurable via UI and expose them for both backtests and paper trading.

### Implementation

#### New Preset: "No Whipsaw Protection"

```python
WHIPSAW_PRESETS = {
    'conservative': {
        'min_hold_bars': 4,
        'trade_cooldown_bars': 2,
        'confidence_override_threshold': 'A',
    },
    'moderate': {
        'min_hold_bars': 2,
        'trade_cooldown_bars': 1,
        'confidence_override_threshold': 'B',
    },
    'aggressive': {
        'min_hold_bars': 1,
        'trade_cooldown_bars': 0,
        'confidence_override_threshold': 'C',
    },
    'disabled': {  # Full scalper mode
        'min_hold_bars': 0,
        'trade_cooldown_bars': 0,
        'confidence_override_threshold': None,
    },
}
```

#### UI: Whipsaw Preset Dropdown

```tsx
<Select
  label="Trade Protection Level"
  value={whipsawPreset}
  onChange={(preset) => applyWhipsawPreset(preset)}
>
  <option value="conservative">Conservative (default)</option>
  <option value="moderate">Moderate</option>
  <option value="aggressive">Aggressive</option>
  <option value="disabled">Disabled (Scalper Mode)</option>
  <option value="custom">Custom...</option>
</Select>

{whipsawPreset === 'custom' && (
  <CustomWhipsawControls
    minHoldBars={config.min_hold_bars}
    cooldownBars={config.trade_cooldown_bars}
    confidenceThreshold={config.confidence_override_threshold}
    onChange={handleWhipsawChange}
  />
)}
```

### Files to Modify

| File | Changes |
|------|---------|
| `ui/src/components/BacktestControl.tsx` | Add whipsaw preset selector |
| `ui/src/components/PaperTradingControl.tsx` | Add whipsaw preset selector |
| `agents/strategies/trigger_engine.py` | Ensure 0 values work correctly |

---

## 4. Leverage Comparison Backtests

### Goal
Run parallel backtests with different leverage settings and compare results side-by-side.

### Approach: Comparison Mode

Instead of modifying the backtest engine, create a new "comparison" workflow that:
1. Takes a base configuration
2. Runs multiple backtests with varied parameters
3. Aggregates and compares results

#### New API Endpoint

`POST /backtests/compare`

```python
class ComparisonConfig(BaseModel):
    base_config: BacktestConfig
    variations: List[Dict[str, Any]]  # Parameter overrides for each variant
    labels: List[str]  # Human-readable labels

# Example request:
{
    "base_config": {
        "symbols": ["BTC-USD"],
        "timeframe": "15m",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "initial_cash": 10000,
        "strategy": "llm_strategist",
        "strategy_id": "aggressive_active"
    },
    "variations": [
        {"max_portfolio_exposure_pct": 100},   # 1x (no leverage)
        {"max_portfolio_exposure_pct": 200},   # 2x leverage
        {"max_portfolio_exposure_pct": 300},   # 3x leverage
        {"max_portfolio_exposure_pct": 500}    # 5x leverage
    ],
    "labels": ["No Leverage", "2x Leverage", "3x Leverage", "5x Leverage"]
}
```

#### Response: Comparison Results

```python
class ComparisonResults(BaseModel):
    comparison_id: str
    base_run_id: str
    variant_run_ids: List[str]
    summary: Dict[str, List[float]]  # metric -> [value per variant]
    # Example:
    # {
    #   "equity_return_pct": [15.2, 28.4, 38.1, -12.5],
    #   "max_drawdown_pct": [8.1, 16.2, 24.3, 100.0],  # 5x got liquidated
    #   "sharpe_ratio": [1.2, 1.1, 0.9, -0.5],
    #   "total_trades": [45, 45, 45, 32],  # 5x stopped early
    #   "liquidation_events": [0, 0, 0, 1]
    # }
```

#### UI: Comparison View

New component `ui/src/components/LeverageComparison.tsx`:

```tsx
<LeverageComparisonChart
  variants={results.labels}
  metrics={{
    returns: results.summary.equity_return_pct,
    drawdown: results.summary.max_drawdown_pct,
    sharpe: results.summary.sharpe_ratio,
  }}
/>

<ComparisonTable
  headers={['Metric', ...results.labels]}
  rows={[
    ['Return %', ...results.summary.equity_return_pct.map(formatPct)],
    ['Max Drawdown %', ...results.summary.max_drawdown_pct.map(formatPct)],
    ['Sharpe Ratio', ...results.summary.sharpe_ratio.map(formatNum)],
    ['Trades', ...results.summary.total_trades],
    ['Liquidations', ...results.summary.liquidation_events],
  ]}
  highlightBest={true}
/>
```

#### Liquidation Detection

Add to `ExecutionLedgerWorkflow`:

```python
def check_liquidation(self, leverage: float) -> bool:
    """Check if current drawdown would trigger liquidation."""
    if leverage <= 1.0:
        return False

    max_adverse_move = 100.0 / leverage  # e.g., 20% for 5x
    current_drawdown = self.get_current_drawdown_pct()

    if current_drawdown >= max_adverse_move:
        self.liquidation_triggered = True
        self.liquidation_timestamp = datetime.utcnow()
        return True
    return False
```

### Files to Create/Modify

| File | Changes |
|------|---------|
| `ops_api/routers/backtests.py` | Add `/backtests/compare` endpoint |
| `backtesting/comparison.py` | New: Comparison orchestration logic |
| `agents/workflows/execution_ledger_workflow.py` | Add liquidation detection |
| `ui/src/components/LeverageComparison.tsx` | New: Comparison UI |
| `ui/src/components/BacktestControl.tsx` | Add "Compare Leverage" button |

---

## 5. Walk-Away Threshold

### Goal
Stop trading for the day after reaching a profit target (e.g., +25% return).

### Implementation

#### New Config Parameter

```python
class WalkAwayConfig(BaseModel):
    enabled: bool = False
    profit_target_pct: float = 25.0      # Stop after +25%
    loss_limit_pct: float = 10.0         # Also stop after -10% (existing max_daily_loss_pct)
    resume_next_session: bool = True     # Auto-resume next day
```

#### Trigger Engine Integration

In `trigger_engine.py`:

```python
def on_bar(self, bar: Bar, ...) -> tuple[List[Order], List[dict]]:
    # Check walk-away threshold FIRST
    if self.walk_away_config.enabled:
        daily_return = self._calculate_daily_return()

        if daily_return >= self.walk_away_config.profit_target_pct:
            self._record_walk_away('profit_target', daily_return)
            return [], []  # No orders - we've hit our target

        if daily_return <= -self.walk_away_config.loss_limit_pct:
            self._record_walk_away('loss_limit', daily_return)
            return [], []  # No orders - stop losses

    # ... rest of on_bar logic ...
```

#### Walk-Away Event Tracking

```python
@dataclass
class WalkAwayEvent:
    timestamp: datetime
    trigger: Literal['profit_target', 'loss_limit']
    daily_return_pct: float
    trades_today: int
    equity_at_trigger: float
```

#### UI: Walk-Away Configuration

```tsx
<ToggleSwitch
  label="Walk-Away Mode"
  checked={config.walk_away_enabled}
  onChange={(v) => setConfig({...config, walk_away_enabled: v})}
/>

{config.walk_away_enabled && (
  <div className="pl-4 space-y-2">
    <ParameterSlider
      label="Profit Target %"
      value={config.walk_away_profit_target_pct}
      min={5} max={100} step={5}
      onChange={(v) => setConfig({...config, walk_away_profit_target_pct: v})}
    />
    <InfoBox>
      Trading will stop for the day after reaching {config.walk_away_profit_target_pct}% profit.
      This locks in gains and prevents giving back profits.
    </InfoBox>
  </div>
)}
```

### Files to Modify

| File | Changes |
|------|---------|
| `agents/strategies/trigger_engine.py` | Add walk-away check in on_bar |
| `schemas/llm_strategist.py` | Add WalkAwayConfig schema |
| `ops_api/routers/backtests.py` | Add walk-away config to BacktestConfig |
| `ops_api/routers/paper_trading.py` | Add walk-away config |
| `ui/src/components/BacktestControl.tsx` | Add walk-away controls |
| `backtesting/llm_strategist_runner.py` | Track walk-away events in results |

---

## 6. Market-Driven Strategy Triggers

### Goal
Allow market conditions to trigger strategy reassessment beyond the scheduled interval.

### Trigger Types

#### 1. Volatility Spike Trigger
```python
class VolatilitySpikeTrigger:
    """Trigger replan when volatility exceeds threshold."""
    atr_spike_multiplier: float = 2.0  # Trigger if ATR > 2x recent average
    lookback_bars: int = 20
```

#### 2. Drawdown Trigger
```python
class DrawdownTrigger:
    """Trigger replan when equity drops significantly."""
    drawdown_threshold_pct: float = 5.0  # Trigger if down 5% from peak
```

#### 3. Price Level Trigger
```python
class PriceLevelTrigger:
    """Trigger replan when price breaks key levels."""
    support_break: bool = True
    resistance_break: bool = True
```

#### 4. Time-Based Triggers (Enhanced)
```python
class TimeBasedTrigger:
    """Trigger replan at specific times."""
    scheduled_hours: List[int] = [0, 8, 16]  # UTC hours
    min_interval_minutes: int = 60  # Minimum time between replans
```

### Implementation

#### New: Market Condition Monitor

Create `agents/strategies/market_monitor.py`:

```python
class MarketConditionMonitor:
    """Monitors market conditions and triggers strategy reassessment."""

    def __init__(
        self,
        volatility_trigger: VolatilitySpikeTrigger | None = None,
        drawdown_trigger: DrawdownTrigger | None = None,
        price_level_trigger: PriceLevelTrigger | None = None,
        time_trigger: TimeBasedTrigger | None = None,
        max_daily_replans: int = 5,  # Budget for reactive replans
    ):
        self.triggers = []
        if volatility_trigger:
            self.triggers.append(volatility_trigger)
        # ... etc

        self.daily_replan_count = 0
        self.last_replan_time = None

    def check_triggers(
        self,
        bar: Bar,
        indicators: IndicatorSnapshot,
        portfolio: PortfolioState,
    ) -> ReplanDecision | None:
        """Check all triggers and return replan decision if triggered."""

        if self.daily_replan_count >= self.max_daily_replans:
            return None  # Budget exhausted

        for trigger in self.triggers:
            decision = trigger.evaluate(bar, indicators, portfolio)
            if decision.should_replan:
                self.daily_replan_count += 1
                self.last_replan_time = bar.timestamp
                return decision

        return None
```

#### Integration with LLM Strategist

In `backtesting/llm_strategist_runner.py`:

```python
def _process_bar(self, bar: Bar, indicators: IndicatorSnapshot):
    # Check scheduled replan
    should_replan_scheduled = self._is_replan_due(bar.timestamp)

    # Check market-driven triggers
    replan_decision = self.market_monitor.check_triggers(
        bar, indicators, self.portfolio_state
    )

    if should_replan_scheduled or replan_decision:
        reason = replan_decision.reason if replan_decision else 'scheduled'
        self._generate_new_plan(bar.timestamp, reason)

    # ... rest of bar processing ...
```

#### API Configuration

```python
class MarketTriggerConfig(BaseModel):
    # Volatility triggers
    volatility_spike_enabled: bool = False
    volatility_spike_multiplier: float = 2.0

    # Drawdown triggers
    drawdown_replan_enabled: bool = False
    drawdown_threshold_pct: float = 5.0

    # Budget
    max_reactive_replans_per_day: int = 3
    min_replan_interval_minutes: int = 60
```

### Files to Create/Modify

| File | Changes |
|------|---------|
| `agents/strategies/market_monitor.py` | New: Market condition monitoring |
| `schemas/llm_strategist.py` | Add MarketTriggerConfig |
| `backtesting/llm_strategist_runner.py` | Integrate market monitor |
| `tools/paper_trading.py` | Integrate market monitor for live |
| `ops_api/routers/backtests.py` | Add market trigger config |
| `ui/src/components/BacktestControl.tsx` | Add market trigger settings |

---

## 7. Real Leverage Trading Roadmap

### Overview

Real leverage trading requires significant infrastructure beyond the mock ledger. This roadmap outlines the path from simulated leverage to actual leveraged positions.

### Phase 1: Perpetual Futures API Integration

**Timeline: Foundation work**

#### 1.1 Coinbase Perpetuals Client

Create `app/coinbase/perpetuals_client.py`:

```python
class CoinbasePerpetualClient:
    """Client for Coinbase perpetual futures trading."""

    async def get_perpetual_products(self) -> List[PerpetualProduct]:
        """List available perpetual contracts."""

    async def get_funding_rate(self, product_id: str) -> FundingRate:
        """Get current funding rate."""

    async def place_perpetual_order(
        self,
        product_id: str,
        side: Literal['buy', 'sell'],
        size: Decimal,
        leverage: int,
        order_type: Literal['market', 'limit'],
        limit_price: Optional[Decimal] = None,
    ) -> PerpetualOrder:
        """Place a perpetual futures order."""

    async def get_position(self, product_id: str) -> PerpetualPosition:
        """Get current position for a perpetual."""

    async def close_position(self, product_id: str) -> PerpetualOrder:
        """Close entire position."""

    async def set_leverage(self, product_id: str, leverage: int) -> None:
        """Set leverage for a product."""
```

#### 1.2 Funding Rate Tracking

Track funding payments for accurate P&L:

```python
@dataclass
class FundingPayment:
    timestamp: datetime
    product_id: str
    rate: Decimal
    payment: Decimal  # Positive = received, negative = paid
    position_size: Decimal
```

### Phase 2: Margin Management

**Timeline: After Phase 1**

#### 2.1 Margin Calculator

```python
class MarginCalculator:
    """Calculate margin requirements and liquidation prices."""

    def initial_margin(
        self,
        notional: Decimal,
        leverage: int,
    ) -> Decimal:
        """Calculate initial margin required."""
        return notional / leverage

    def maintenance_margin(
        self,
        notional: Decimal,
        leverage: int,
    ) -> Decimal:
        """Calculate maintenance margin (typically 50% of initial)."""
        return self.initial_margin(notional, leverage) * Decimal('0.5')

    def liquidation_price(
        self,
        entry_price: Decimal,
        leverage: int,
        side: Literal['long', 'short'],
    ) -> Decimal:
        """Calculate liquidation price."""
        margin_pct = Decimal('1') / leverage
        if side == 'long':
            return entry_price * (1 - margin_pct + Decimal('0.005'))  # 0.5% buffer
        else:
            return entry_price * (1 + margin_pct - Decimal('0.005'))
```

#### 2.2 Position Monitor

```python
class PositionMonitor:
    """Monitor positions for margin calls and liquidation risk."""

    async def check_margin_health(self, position: PerpetualPosition) -> MarginHealth:
        """Check if position is healthy, warning, or critical."""

    async def auto_reduce_position(
        self,
        position: PerpetualPosition,
        target_margin_ratio: Decimal,
    ) -> Optional[PerpetualOrder]:
        """Automatically reduce position to improve margin health."""
```

### Phase 3: Risk Controls

**Timeline: After Phase 2**

#### 3.1 Leverage Guard Rails

```python
class LeverageGuardRails:
    """Safety controls for leveraged trading."""

    max_leverage: int = 5  # Hard cap
    max_position_value: Decimal  # In USD
    max_daily_funding_cost: Decimal

    def validate_order(self, order: PerpetualOrder) -> ValidationResult:
        """Validate order against guard rails."""

    def check_funding_budget(self, expected_cost: Decimal) -> bool:
        """Check if funding cost is within budget."""
```

#### 3.2 Automatic Deleveraging

```python
class AutoDeleverager:
    """Automatically reduce leverage when risk thresholds are breached."""

    async def monitor_and_deleverage(
        self,
        position: PerpetualPosition,
        risk_threshold: Decimal,
    ) -> List[PerpetualOrder]:
        """Monitor position and deleverage if needed."""
```

### Phase 4: Integration with Execution Agent

**Timeline: After Phase 3**

#### 4.1 Unified Order Router

```python
class OrderRouter:
    """Route orders to appropriate venue (spot vs perpetual)."""

    def route_order(
        self,
        order: Order,
        preferences: UserPreferences,
    ) -> RoutedOrder:
        """Determine best venue for order execution."""

        if preferences.enable_leverage and order.leverage > 1:
            return self._route_to_perpetuals(order)
        else:
            return self._route_to_spot(order)
```

#### 4.2 Leverage Mode Toggle

Add to user preferences:

```python
class LeveragePreferences(BaseModel):
    enable_perpetuals: bool = False
    max_leverage: int = 3
    auto_deleverage_threshold: float = 0.8  # 80% of liquidation
    funding_rate_limit: float = 0.1  # Max 0.1% per 8h
```

### Phase 5: UI for Real Leverage

**Timeline: After Phase 4**

#### 5.1 Leverage Dashboard

New component showing:
- Current leverage per position
- Margin health (visual gauge)
- Liquidation prices
- Funding rate costs (projected daily/monthly)
- P&L including funding

#### 5.2 Risk Warnings

Mandatory acknowledgments before enabling leverage:
- "Leverage can result in losses exceeding your deposit"
- "Funding rates can significantly impact returns"
- "Positions may be liquidated during volatile markets"

### Implementation Priority

| Phase | Components | Dependencies |
|-------|-----------|--------------|
| 1 | Perpetuals Client | None |
| 2 | Margin Calculator, Position Monitor | Phase 1 |
| 3 | Guard Rails, Auto Deleverager | Phase 2 |
| 4 | Order Router, Preference Toggle | Phase 3 |
| 5 | Dashboard, Risk Warnings | Phase 4 |

### Files to Create

| File | Purpose |
|------|---------|
| `app/coinbase/perpetuals_client.py` | Perpetuals API client |
| `app/leverage/margin.py` | Margin calculations |
| `app/leverage/position_monitor.py` | Position health monitoring |
| `app/leverage/guard_rails.py` | Safety controls |
| `app/leverage/auto_deleverager.py` | Automatic risk reduction |
| `agents/order_router.py` | Order routing logic |
| `ui/src/components/LeverageDashboard.tsx` | Leverage monitoring UI |

---

## 8. UI Component Summary

### New Components to Create

| Component | Purpose |
|-----------|---------|
| `AggressiveSettingsPanel.tsx` | Collapsible panel for risk/whipsaw settings |
| `LeverageComparison.tsx` | Side-by-side leverage comparison view |
| `WalkAwayConfig.tsx` | Walk-away threshold configuration |
| `MarketTriggersConfig.tsx` | Market-driven replan trigger settings |
| `LeverageDashboard.tsx` | Real leverage position monitoring |
| `ParameterSlider.tsx` | Reusable slider with tooltip |
| `PresetSelector.tsx` | Quick preset selection dropdown |

### Modified Components

| Component | Changes |
|-----------|---------|
| `BacktestControl.tsx` | Add advanced settings, presets, comparison |
| `PaperTradingControl.tsx` | Add advanced settings, walk-away |
| `BacktestPlaybackViewer.tsx` | Show walk-away events, replan triggers |

---

## 9. Database Schema Changes

### New Tables (if persisting to DB)

```sql
-- Walk-away events
CREATE TABLE walk_away_events (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    triggered_at TIMESTAMP NOT NULL,
    trigger_type VARCHAR(20) NOT NULL,  -- 'profit_target' or 'loss_limit'
    daily_return_pct DECIMAL(10,4) NOT NULL,
    trades_today INTEGER NOT NULL,
    equity_at_trigger DECIMAL(20,8) NOT NULL
);

-- Market-driven replans
CREATE TABLE replan_events (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    triggered_at TIMESTAMP NOT NULL,
    trigger_type VARCHAR(30) NOT NULL,  -- 'scheduled', 'volatility_spike', 'drawdown', etc.
    trigger_details JSONB,
    plan_hash VARCHAR(64)  -- Hash of generated plan for deduplication
);

-- Leverage comparison runs
CREATE TABLE comparison_runs (
    id VARCHAR(64) PRIMARY KEY,
    base_run_id VARCHAR(64) NOT NULL,
    variant_run_ids VARCHAR(64)[] NOT NULL,
    labels TEXT[] NOT NULL,
    created_at TIMESTAMP NOT NULL,
    summary JSONB
);
```

---

## 10. Testing Strategy

### Unit Tests

| Test File | Coverage |
|-----------|----------|
| `tests/test_walk_away.py` | Walk-away threshold logic |
| `tests/test_market_monitor.py` | Market trigger conditions |
| `tests/test_leverage_comparison.py` | Comparison aggregation |
| `tests/test_liquidation.py` | Liquidation detection |

### Integration Tests

| Test | Description |
|------|-------------|
| Backtest with walk-away | Verify trading stops at target |
| Backtest with market triggers | Verify replans fire correctly |
| Leverage comparison | Verify parallel runs complete |
| Paper trading with new params | Verify all params flow through |

### E2E Tests

| Test | Description |
|------|-------------|
| UI preset selection | Select scalper preset, verify params |
| Run comparison from UI | Start comparison, view results |
| Walk-away in paper trading | Verify UI shows walk-away status |

---

## 11. Implementation Order

### Sprint 1: Foundation (P0 Features)
1. Add aggressive parameters to BacktestConfig
2. Add whipsaw controls to BacktestConfig
3. Update UI with advanced settings panel
4. Add scalper presets
5. Test end-to-end

### Sprint 2: Walk-Away & Comparison (P1 Features)
1. Implement walk-away threshold in trigger engine
2. Add walk-away config to UI
3. Create comparison endpoint
4. Build comparison UI
5. Add liquidation detection

### Sprint 3: Market Triggers (P2 Features)
1. Build market condition monitor
2. Implement trigger types
3. Integrate with LLM strategist
4. Add trigger config to UI
5. Test reactive replanning

### Sprint 4: Real Leverage Roadmap (P2 Features)
1. Perpetuals client (Phase 1)
2. Margin calculator (Phase 2)
3. Guard rails (Phase 3)
4. Documentation and warnings

---

## 12. Success Metrics

| Metric | Target |
|--------|--------|
| Backtest parameter coverage | 100% of parameters exposed in UI |
| Comparison run time | < 2x single backtest time |
| Walk-away accuracy | 100% triggers at correct threshold |
| Market trigger latency | < 1 bar delay from condition to replan |
| UI responsiveness | < 500ms for parameter changes |

---

## Appendix: Quick Reference

### Scalper Mode Defaults

```python
SCALPER_DEFAULTS = {
    # Timeframe
    'timeframe': '5m',
    'indicator_mode': 'short_term',

    # Risk
    'max_position_risk_pct': 4.0,
    'max_portfolio_exposure_pct': 100.0,
    'max_daily_loss_pct': 6.0,

    # Whipsaw (disabled)
    'min_hold_bars': 0,
    'trade_cooldown_bars': 0,
    'confidence_override_threshold': None,

    # Execution
    'min_price_move_pct': 0.1,
    'max_trades_per_day': 50,

    # Walk-away
    'walk_away_enabled': True,
    'walk_away_profit_target_pct': 25.0,

    # Market triggers
    'volatility_spike_enabled': True,
    'drawdown_replan_enabled': True,
    'max_reactive_replans_per_day': 3,
}
```

### CLI Quick Start

```bash
# Run scalper backtest
uv run python -m backtesting.cli \
  --llm-strategist enabled \
  --llm-prompt prompts/strategies/aggressive_active.txt \
  --pair BTC-USD \
  --start 2024-01-01 \
  --end 2024-01-07 \
  --initial-cash 10000 \
  --timeframes 5m 15m 1h \
  --max-position-risk-pct 4.0 \
  --max-portfolio-exposure-pct 100 \
  --max-daily-loss-pct 6.0 \
  --llm-run-id scalper-test
```
