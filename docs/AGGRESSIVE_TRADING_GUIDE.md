# Aggressive Short-Term Trading Guide

This guide documents how to configure and use the crypto trading agents for aggressive short-term trading strategies aimed at maximizing returns through frequent intraday positions.

## Executive Summary

Your potential user's vision:
- **Target**: 25% daily returns (ambitious, but illustrative of the use case)
- **Style**: Short-term trades on hourly candles
- **Expectation**: Multiple positions opened/closed throughout the day
- **Key insight**: Returns should be measured against the **full portfolio**, not just the tradeable fraction

This system CAN support this use case, but requires specific configuration changes from the defaults (which are conservative).

---

## Table of Contents

1. [Understanding the Current Architecture](#understanding-the-current-architecture)
2. [Key Configuration Parameters](#key-configuration-parameters)
3. [The Flip-Flop Prevention Problem](#the-flip-flop-prevention-problem)
4. [Risk Budget & Tradeable Fraction](#risk-budget--tradeable-fraction)
5. [Returns Calculation: Full Portfolio vs Tradeable Amount](#returns-calculation-full-portfolio-vs-tradeable-amount)
6. [Using Leverage](#using-leverage)
7. [Strategy Templates for Aggressive Trading](#strategy-templates-for-aggressive-trading)
8. [Running Backtests](#running-backtests)
9. [Tax Implications of Short-Term Trading](#tax-implications-of-short-term-trading)
10. [Recommended Testing Approach](#recommended-testing-approach)

---

## Understanding the Current Architecture

### How Trades Are Generated

1. **LLM Strategist** generates a `StrategyPlan` containing:
   - Multiple `TriggerCondition` objects (entry/exit rules)
   - Risk constraints (position limits, exposure caps)
   - Sizing rules per symbol

2. **Trigger Engine** (`agents/strategies/trigger_engine.py`) evaluates these triggers on each price bar:
   - Checks entry rules against technical indicators
   - Checks exit rules for open positions
   - Applies anti-whipsaw protections (cooldowns, hold periods)

3. **Risk Engine** (`agents/strategies/risk_engine.py`) sizes positions:
   - Applies portfolio-level constraints
   - Calculates quantity based on risk budget
   - Enforces daily loss limits

4. **Execution Ledger** (`agents/workflows/execution_ledger_workflow.py`) tracks:
   - Cash, positions, P&L
   - Profit scraping (locking away gains)
   - Transaction history

### Key Files for Configuration

| File | Purpose |
|------|---------|
| `agents/strategies/trigger_engine.py:47-80` | Anti-flip-flop settings |
| `agents/strategies/risk_engine.py` | Position sizing logic |
| `agents/execution_config.py` | Minimum price move thresholds |
| `prompts/strategies/aggressive_active.txt` | Aggressive strategy prompt |
| `schemas/llm_strategist.py:111-119` | Risk constraint schema |

---

## Key Configuration Parameters

### Environment Variables

```bash
# Lower this to allow trading on smaller price moves (default: 0.5%)
EXECUTION_MIN_PRICE_MOVE_PCT=0.1

# Increase staleness tolerance if needed (default: 1800 seconds)
EXECUTION_MAX_STALENESS_SECONDS=3600

# Starting portfolio balance for mock ledger
INITIAL_PORTFOLIO_BALANCE=10000

# Allow more LLM calls per hour for responsive trading
EXECUTION_MAX_CALLS_PER_HOUR_PER_SYMBOL=120
```

### Risk Constraint Parameters

These are set in the `StrategyPlan` by the LLM or via CLI overrides:

| Parameter | Conservative | Aggressive | Description |
|-----------|-------------|------------|-------------|
| `max_position_risk_pct` | 1-2% | 3-5% | Risk per trade as % of equity |
| `max_symbol_exposure_pct` | 25% | 40-60% | Max notional for single symbol |
| `max_portfolio_exposure_pct` | 80% | 100-150% | Total portfolio exposure |
| `max_daily_loss_pct` | 3% | 5-8% | Daily stop-loss threshold |
| `max_daily_risk_budget_pct` | 5% | 10-15% | Cumulative daily risk allocation |
| `max_trades_per_day` | 5-10 | 20-50 | Hard cap on daily trades |
| `max_triggers_per_symbol_per_day` | 3-5 | 8-15 | Per-symbol trade limit |

### Trigger Engine Anti-Flip-Flop Settings

Located in `agents/strategies/trigger_engine.py:47-72`:

```python
# Current defaults (CONSERVATIVE - reduce for active trading)
min_hold_bars: int = 4        # Minimum bars before exit allowed
trade_cooldown_bars: int = 2  # Minimum bars between trades

# For aggressive trading, consider:
min_hold_bars: int = 1        # Allow exit after 1 bar
trade_cooldown_bars: int = 0  # No cooldown between trades
```

**Important**: These settings prevent the "flip-flop" behavior your potential user WANTS. For aggressive intraday trading, you should REDUCE or DISABLE these protections.

---

## The Flip-Flop Prevention Problem

### What Was Implemented

The system has multiple layers of anti-whipsaw protection:

1. **Minimum Hold Period** (`min_hold_bars=4`):
   - Prevents exiting a position within 4 bars of entry
   - On 1-hour candles, this means holding for at least 4 hours
   - **Impact**: Reduces trade frequency significantly

2. **Trade Cooldown** (`trade_cooldown_bars=2`):
   - After any trade, the symbol is "locked" for 2 bars
   - Prevents rapid re-entry after stop-out
   - **Impact**: At most 12 trades per day on 1h candles

3. **Exit Priority with Confidence Override** (`trigger_engine.py:333-406`):
   - By default, exit signals override entry signals
   - Only high-confidence (A-grade) entries can override exits
   - **Impact**: Favors closing positions over opening new ones

4. **Minimum Price Move** (`EXECUTION_MIN_PRICE_MOVE_PCT=0.5%`):
   - LLM execution agent won't even consider trading unless price moved 0.5%
   - **Impact**: Filters out many potential trade opportunities

### Why This May Be Wrong for Your User

Your potential user expects:
- Opening and closing positions **several times per day**
- Capturing **small moves** (0.5-1% profit targets)
- **Quick scalping** rather than position holding

The current defaults are designed for a more conservative "let winners run" approach, NOT for active day trading.

### How to Adjust for Aggressive Trading

**Option 1: Modify Trigger Engine Initialization**

When creating the `TriggerEngine`, pass different parameters:

```python
engine = TriggerEngine(
    plan=strategy_plan,
    risk_engine=risk_engine,
    confidence_override_threshold="C",  # Allow all entries to override exits
    min_hold_bars=1,                     # Exit after 1 bar minimum
    trade_cooldown_bars=0,               # No cooldown between trades
)
```

**Option 2: Use the Aggressive Strategy Prompt**

The `prompts/strategies/aggressive_active.txt` template is designed for active trading:

```bash
# Run backtest with aggressive prompt
uv run python -m backtesting.cli \
  --llm-strategist enabled \
  --llm-prompt prompts/strategies/aggressive_active.txt \
  --pair BTC-USD \
  --start 2024-01-01 \
  --end 2024-01-31 \
  --initial-cash 10000 \
  --max-position-risk-pct 3.0 \
  --max-daily-loss-pct 5.0
```

**Option 3: Environment Variable Tuning**

```bash
# Allow trading on smaller moves
export EXECUTION_MIN_PRICE_MOVE_PCT=0.1

# More LLM calls for faster response
export EXECUTION_MAX_CALLS_PER_HOUR_PER_SYMBOL=120
```

---

## Risk Budget & Tradeable Fraction

### Understanding the Dual System

There are TWO separate concepts:

1. **Tradeable Fraction** (Production Ledger only):
   - Set via CLI: `wallet set-tradeable-fraction <wallet_id> 0.20`
   - Only applies to the **production ledger** (real Coinbase trading)
   - Determines what percentage of your wallet balance can be deployed
   - Example: $10,000 wallet with 20% tradeable = $2,000 available for trades

2. **Portfolio Exposure Limits** (Risk Engine):
   - `max_portfolio_exposure_pct`: Total notional that can be deployed
   - `max_symbol_exposure_pct`: Max per-symbol allocation
   - These are calculated against **full portfolio equity**

### The Critical Insight

**Returns are calculated on FULL PORTFOLIO EQUITY, not just the tradeable amount.**

From `execution_ledger_workflow.py:477-478`:

```python
current_portfolio_value = float(self.initial_cash) + float(self.get_pnl())
total_pnl = current_portfolio_value - float(self.initial_cash)
```

This means:
- If you have $10,000 portfolio with 20% tradeable ($2,000)
- And you make $500 profit (25% return on tradeable amount)
- Your **actual portfolio return** is only 5% ($500 / $10,000)

### Implications for 25% Daily Returns

To achieve 25% returns on the FULL portfolio, you need either:

1. **Higher tradeable fraction** (e.g., 80-100%)
2. **Higher position risk per trade** (e.g., 5-10% per trade)
3. **More frequent trades** (capturing many small gains)
4. **Leverage** (the system supports >100% exposure)

### Recommended Configuration for Aggressive Returns

```python
# Risk constraints for aggressive daily returns
risk_constraints = {
    "max_position_risk_pct": 5.0,        # 5% risk per trade
    "max_symbol_exposure_pct": 50.0,     # Up to 50% in one symbol
    "max_portfolio_exposure_pct": 120.0, # Allow 1.2x leverage
    "max_daily_loss_pct": 8.0,           # 8% daily stop-loss
    "max_daily_risk_budget_pct": 15.0,   # 15% cumulative daily risk
}

# Plan limits
plan_limits = {
    "max_trades_per_day": 30,
    "max_triggers_per_symbol_per_day": 10,
}
```

---

## Returns Calculation: Full Portfolio vs Tradeable Amount

### Current Implementation

The mock ledger tracks:

```python
# Entry price tracking (weighted average)
self.entry_price[symbol] = (existing_value + new_value) / new_total_qty

# Realized P&L on exit
position_realized_pnl = (exit_price - entry_price) * quantity_sold
self.realized_pnl += position_realized_pnl

# Unrealized P&L for open positions
unrealized_pnl = sum((current_price - entry_price) * qty for symbol, qty in positions)

# Total P&L
total_pnl = realized_pnl + unrealized_pnl

# Portfolio value
portfolio_value = initial_cash + total_pnl
```

### Return Percentage Calculation

```python
# As implemented (full portfolio basis)
return_pct = (portfolio_value - initial_cash) / initial_cash * 100

# Example:
# Initial: $10,000
# Final: $12,500
# Return: 25%
```

### What This Means for Your User

If the user wants 25% daily returns on $10,000:
- They need to generate $2,500 in profit per day
- With 5% risk per trade and 50% win rate, they'd need ~20 winning trades
- Or fewer trades with higher conviction and better win rate

The math for achievability:
```
Target daily profit: $2,500
Avg profit per winning trade (1% move on $5,000 position): $50
Required winning trades: 50 (at 50% win rate = 100 total trades)

Alternative with larger positions:
Avg profit per winning trade (2% move on $8,000 position): $160
Required winning trades: ~16 (at 60% win rate = ~27 total trades)
```

---

## Using Leverage

Leverage amplifies both gains AND losses, allowing you to control larger positions than your capital would normally permit. This section explains how leverage works in this system and the critical risks involved.

### How Leverage Works in This System

#### The `max_portfolio_exposure_pct` Parameter

The system supports leverage through the `max_portfolio_exposure_pct` parameter in risk constraints:

```python
# From risk_engine.py:96-100
def _available_portfolio_capacity(self, portfolio: PortfolioState) -> float:
    max_portfolio = portfolio.equity * self._scaled_fraction(
        self.constraints.max_portfolio_exposure_pct, None
    )
    return max(0.0, max_portfolio - self._portfolio_exposure(portfolio))
```

| Setting | Meaning | Effective Leverage |
|---------|---------|-------------------|
| `max_portfolio_exposure_pct: 80` | Conservative (default) | 0.8x |
| `max_portfolio_exposure_pct: 100` | Fully invested | 1.0x |
| `max_portfolio_exposure_pct: 150` | 1.5x leverage | 1.5x |
| `max_portfolio_exposure_pct: 200` | 2x leverage | 2.0x |
| `max_portfolio_exposure_pct: 500` | 5x leverage | 5.0x |

#### Example: 2x Leverage Configuration

```python
risk_constraints = {
    "max_position_risk_pct": 5.0,
    "max_symbol_exposure_pct": 100.0,      # Allow full equity in one symbol
    "max_portfolio_exposure_pct": 200.0,   # 2x leverage
    "max_daily_loss_pct": 10.0,            # Higher daily loss tolerance
}
```

With $10,000 equity and 2x leverage:
- Maximum position size: $20,000 notional
- A 5% price move = $1,000 profit/loss (10% of equity)
- A 10% adverse move = $2,000 loss (20% of equity)

### Coinbase Leverage Options

#### Spot Trading (Current Default)
- **No native leverage** on Coinbase spot markets
- The system's "leverage" is **simulated** in the mock ledger
- For real trading, you'd need to implement margin borrowing elsewhere

#### Coinbase Advanced Trade (Perpetual Futures)
Coinbase offers perpetual futures with up to **20x leverage** on select pairs:

| Product | Max Leverage | Funding Rate |
|---------|-------------|--------------|
| BTC-PERP | 20x | Variable (8h) |
| ETH-PERP | 20x | Variable (8h) |
| Other alts | 5-10x | Variable (8h) |

**To use perpetuals**, you would need to:
1. Integrate with Coinbase's perpetuals API (different from spot)
2. Handle funding rate payments (can be positive or negative)
3. Manage margin requirements and liquidation thresholds

#### Mock Ledger Leverage (Backtesting/Paper Trading)

The mock ledger (`execution_ledger_workflow.py`) allows leverage by:
- Permitting positions larger than available cash
- Tracking negative cash balances (margin debt)
- Calculating P&L on full position size

```python
# The mock ledger allows cash to go negative (margin)
# From execution_ledger_workflow.py - fill handling
self.cash -= (price * qty)  # Can result in negative cash = margin debt
```

### Leverage Math: Amplification of Returns

#### Basic Leverage Calculation

```
Leveraged Return = Unleveraged Return × Leverage Factor

Example:
- Underlying move: +2%
- Leverage: 3x
- Your return: +6% (minus fees/funding)
```

#### The Double-Edged Sword

| Scenario | No Leverage | 2x Leverage | 5x Leverage |
|----------|-------------|-------------|-------------|
| +5% move | +5% | +10% | +25% |
| +2% move | +2% | +4% | +10% |
| -2% move | -2% | -4% | -10% |
| -5% move | -5% | -10% | -25% |
| -20% move | -20% | -40% | **-100% (LIQUIDATION)** |

#### Liquidation Risk

At high leverage, small adverse moves can wipe out your entire position:

```
Liquidation Price Movement = 100% / Leverage

Examples:
- 2x leverage: Liquidated at -50% move
- 5x leverage: Liquidated at -20% move
- 10x leverage: Liquidated at -10% move
- 20x leverage: Liquidated at -5% move
```

### Configuring Leverage for Backtests

#### CLI Options for Leveraged Backtesting

```bash
uv run python -m backtesting.cli \
  --llm-strategist enabled \
  --llm-prompt prompts/strategies/aggressive_active.txt \
  --pair BTC-USD \
  --start 2024-01-01 \
  --end 2024-03-31 \
  --initial-cash 10000 \
  --max-portfolio-exposure-pct 200 \
  --max-symbol-exposure-pct 150 \
  --max-position-risk-pct 5.0 \
  --max-daily-loss-pct 15.0 \
  --llm-run-id leveraged-2x-test
```

#### Risk Config File for Leverage

Create `configs/leverage_2x.json`:

```json
{
  "risk_limits": {
    "max_position_risk_pct": 5.0,
    "max_symbol_exposure_pct": 150.0,
    "max_portfolio_exposure_pct": 200.0,
    "max_daily_loss_pct": 15.0,
    "max_daily_risk_budget_pct": 20.0
  }
}
```

Then use:
```bash
uv run python -m backtesting.cli \
  --risk-config configs/leverage_2x.json \
  ...
```

### Leverage Strategies by Risk Tolerance

#### Conservative Leverage (1.2-1.5x)
- Suitable for experienced traders
- Modest return amplification
- Survivable drawdowns

```python
risk_constraints = {
    "max_portfolio_exposure_pct": 150.0,
    "max_daily_loss_pct": 8.0,
    "max_position_risk_pct": 3.0,
}
```

#### Moderate Leverage (2-3x)
- Significant amplification
- Requires strict stop-losses
- Daily monitoring essential

```python
risk_constraints = {
    "max_portfolio_exposure_pct": 250.0,
    "max_daily_loss_pct": 12.0,
    "max_position_risk_pct": 4.0,
}
```

#### Aggressive Leverage (5x+)
- **HIGH RISK - NOT RECOMMENDED**
- Suitable only for very short-term scalping
- Requires automated stop-losses
- Can lose entire account in minutes

```python
# DANGER ZONE - Use at your own risk
risk_constraints = {
    "max_portfolio_exposure_pct": 500.0,
    "max_daily_loss_pct": 20.0,
    "max_position_risk_pct": 2.0,  # Lower per-trade risk!
}
```

### Leverage Risk Management

#### Required Stop-Loss Discipline

With leverage, stop-losses are **MANDATORY**, not optional:

| Leverage | Maximum Stop-Loss | Reason |
|----------|------------------|--------|
| 2x | 10% | Avoid >20% equity loss |
| 3x | 6% | Avoid >18% equity loss |
| 5x | 4% | Avoid >20% equity loss |
| 10x | 2% | Avoid >20% equity loss |

**Configure tight stops in triggers:**

```python
{
    "id": "btc_leveraged_scalp",
    "symbol": "BTC-USD",
    "direction": "long",
    "entry_rule": "rsi_14 < 35 and close > sma_medium",
    "exit_rule": "unrealized_pnl_pct > 1.0",  # Quick profit target
    "stop_loss_pct": 0.5  # TIGHT stop for leveraged position
}
```

#### Position Sizing with Leverage

When using leverage, **reduce position risk percentage**:

```
Adjusted Risk = Base Risk / Leverage

Example:
- Base risk tolerance: 5% per trade
- Using 3x leverage
- Adjusted risk: 5% / 3 = 1.67% per trade
```

This keeps your actual dollar risk constant despite leverage.

#### Daily Loss Circuit Breakers

Configure aggressive daily loss limits to prevent catastrophic drawdowns:

```python
# For 3x leverage, use tighter daily loss limit
risk_constraints = {
    "max_portfolio_exposure_pct": 300.0,
    "max_daily_loss_pct": 5.0,  # Stop trading after 5% daily loss
}
```

### Leverage Costs and Considerations

#### Funding Rates (Perpetual Futures)

If using perpetual futures:
- **Funding rate** paid/received every 8 hours
- Can be positive (longs pay shorts) or negative (shorts pay longs)
- Typical range: -0.1% to +0.1% per 8 hours
- Annualized: Can exceed 100% in trending markets!

```
Daily Funding Cost (worst case) = Position Size × 0.3%
Monthly Funding Cost = Position Size × 9%

Example with $20,000 position:
- Daily: $60 in funding fees
- Monthly: $1,800 in funding fees
```

#### Margin Interest (Spot Margin)

If borrowing for spot margin:
- Interest rates: 5-15% APR typical
- Calculated hourly, compounded
- Reduces net returns significantly

```
Monthly Interest Cost = Borrowed Amount × (APR / 12)

Example: Borrowing $10,000 at 10% APR
- Monthly cost: $83.33
- Annual cost: $1,000
```

#### Liquidation Fees

Most exchanges charge liquidation fees:
- Coinbase: ~1% of position
- Other exchanges: 0.5-2%

These fees are deducted BEFORE returning remaining equity.

### Leverage Backtest Example

Compare unleveraged vs leveraged results:

```bash
# Baseline: No leverage
uv run python -m backtesting.cli \
  --pair BTC-USD \
  --start 2024-01-01 \
  --end 2024-03-31 \
  --initial-cash 10000 \
  --max-portfolio-exposure-pct 100 \
  --llm-run-id no-leverage-baseline

# Test: 2x leverage
uv run python -m backtesting.cli \
  --pair BTC-USD \
  --start 2024-01-01 \
  --end 2024-03-31 \
  --initial-cash 10000 \
  --max-portfolio-exposure-pct 200 \
  --max-daily-loss-pct 15 \
  --llm-run-id leverage-2x-test

# Compare results in .cache/strategy_plans/
```

### Leverage Warnings

#### The Leverage Trap

Many traders are attracted to leverage because it can turn small accounts into large profits quickly. However:

1. **Most leveraged traders lose money** - Studies show 70-80% of retail traders using leverage lose their accounts
2. **Leverage magnifies mistakes** - A single bad trade can wipe out weeks of gains
3. **Psychology becomes harder** - Larger swings create emotional decision-making
4. **Costs compound** - Funding rates and interest eat into returns

#### When NOT to Use Leverage

- You're new to trading
- You don't have strict stop-loss discipline
- You can't monitor positions continuously
- You're trading with money you can't afford to lose
- Market conditions are highly volatile or uncertain
- You don't fully understand liquidation mechanics

#### Safe Leverage Guidelines

1. **Start with paper trading** - Test leveraged strategies in the mock ledger first
2. **Begin with 1.5x or less** - Even small leverage amplifies results significantly
3. **Always use stop-losses** - No exceptions with leveraged positions
4. **Size positions appropriately** - Reduce position risk % when using leverage
5. **Monitor funding costs** - Factor these into your return calculations
6. **Have a maximum drawdown rule** - Stop trading if you hit X% daily loss

### Leverage + Aggressive Trading: A Dangerous Combination

Combining high leverage with aggressive trading frequency multiplies risks:

| Factor | Impact |
|--------|--------|
| More trades | More opportunities for errors |
| Larger positions | Bigger impact per error |
| Faster execution | Less time to catch mistakes |
| Funding costs | Accumulate with position duration |
| Emotional pressure | Compounds with each trade |

**Recommendation**: If pursuing aggressive short-term trading, start with NO leverage until you've validated your strategy over multiple market conditions. Only then consider modest leverage (1.5-2x) with proven, disciplined execution.

---

## Strategy Templates for Aggressive Trading

### Aggressive Active Template

From `prompts/strategies/aggressive_active.txt`:

**Key characteristics:**
- **Multi-category triggers**: Exploit all market conditions
- **High trade frequency**: 6-10 triggers per symbol per day
- **Quick profit targets**: 0.5-1% exits
- **Higher risk tolerance**: 2.5-3% per trade, 5% daily loss limit
- **Both directions**: Long AND short triggers

**Example trigger from the template:**

```python
{
    "id": "btc_quick_momentum_long",
    "symbol": "BTC-USD",
    "category": "trend_continuation",
    "confidence_grade": "B",
    "direction": "long",
    "timeframe": "1h",
    "entry_rule": "rsi_14 > 60 and rsi_14[-1] < 55 and close > sma_short and macd_hist > 0",
    "exit_rule": "unrealized_pnl_pct > 0.75 or close < sma_short",
    "stop_loss_pct": 1.0
}
```

### Custom Aggressive Configuration

For your user's use case, consider this configuration:

```json
{
    "regime": "mixed",
    "allowed_directions": ["long", "short"],
    "max_trades_per_day": 40,
    "max_triggers_per_symbol_per_day": 15,
    "risk_constraints": {
        "max_position_risk_pct": 4.0,
        "max_symbol_exposure_pct": 50.0,
        "max_portfolio_exposure_pct": 100.0,
        "max_daily_loss_pct": 6.0,
        "max_daily_risk_budget_pct": 12.0
    },
    "sizing_rules": {
        "BTC-USD": {
            "sizing_mode": "fixed_fraction",
            "target_risk_pct": 4.0
        }
    },
    "triggers": [
        // Multiple triggers for each market condition
        // See aggressive_active.txt for patterns
    ]
}
```

---

## Running Backtests

### Basic Backtest Command

```bash
uv run python -m backtesting.cli \
  --pair BTC-USD \
  --start 2024-01-01 \
  --end 2024-01-31 \
  --initial-cash 10000 \
  --fee-rate 0.001
```

### LLM Strategist Backtest (Recommended for Testing Strategies)

```bash
uv run python -m backtesting.cli \
  --llm-strategist enabled \
  --llm-prompt prompts/strategies/aggressive_active.txt \
  --pair BTC-USD \
  --start 2024-01-01 \
  --end 2024-01-31 \
  --initial-cash 10000 \
  --fee-rate 0.001 \
  --llm-calls-per-day 2 \
  --llm-run-id aggressive-test-001 \
  --max-position-risk-pct 4.0 \
  --max-symbol-exposure-pct 50.0 \
  --max-daily-loss-pct 6.0 \
  --max-daily-risk-budget-pct 12.0 \
  --timeframes 1h 4h \
  --debug-limits verbose
```

### Multi-Asset Portfolio Backtest

```bash
uv run python -m backtesting.cli \
  --pairs BTC-USD ETH-USD SOL-USD \
  --weights 0.5 0.3 0.2 \
  --llm-strategist enabled \
  --llm-prompt prompts/strategies/aggressive_active.txt \
  --start 2024-01-01 \
  --end 2024-01-31 \
  --initial-cash 10000
```

### Key Backtest Options

| Flag | Description |
|------|-------------|
| `--llm-strategist enabled` | Use LLM to generate strategy plans |
| `--llm-prompt <path>` | Custom strategy prompt template |
| `--llm-calls-per-day N` | LLM budget (more = more adaptive) |
| `--max-position-risk-pct N` | Override per-trade risk |
| `--max-daily-loss-pct N` | Override daily stop-loss |
| `--flatten-daily` | Close all positions at day end |
| `--debug-limits verbose` | Detailed trade blocking analysis |
| `--timeframes 1h 4h` | Indicator calculation timeframes |

### Analyzing Backtest Results

Results are written to `.cache/strategy_plans/<run_id>/`:
- `run_summary.json`: Overall performance metrics
- `daily_reports/YYYY-MM-DD.json`: Daily breakdown with trigger quality

Key metrics to evaluate:
- `return_pct`: Total return over period
- `sharpe_ratio`: Risk-adjusted returns
- `max_drawdown`: Largest peak-to-trough decline
- `win_rate`: Percentage of profitable trades
- `trades_per_day`: Average daily trade count

---

## Tax Implications of Short-Term Trading

### Critical Tax Considerations

**WARNING**: Aggressive short-term crypto trading has significant tax implications that MUST be considered.

### U.S. Tax Treatment

#### Short-Term vs Long-Term Capital Gains

| Holding Period | Tax Treatment | Typical Rate |
|----------------|---------------|--------------|
| < 1 year | Short-term (ordinary income) | 10-37% |
| > 1 year | Long-term capital gains | 0-20% |

**Key insight**: EVERY trade in an aggressive day-trading strategy generates **short-term capital gains**, taxed at your ordinary income rate.

#### Wash Sale Rules (Currently Unclear for Crypto)

As of 2024, the IRS has not definitively applied wash sale rules to cryptocurrency. However:
- Traditional securities have a 30-day wash sale rule
- Proposed legislation may extend this to crypto
- **Conservative approach**: Assume wash sale rules may apply

#### Tax Implications by Strategy

| Strategy | Daily Trades | Annual Trades | Tax Impact |
|----------|-------------|---------------|------------|
| Buy & Hold | 0 | ~2 | Long-term rates if held >1 year |
| Conservative Bot | 1-3 | 250-750 | All short-term |
| Aggressive Bot | 10-40 | 2,500-10,000 | ALL short-term, massive paperwork |

### Example Tax Scenario

**Aggressive trading scenario:**
- Starting capital: $10,000
- Daily return target: 25%
- Trades per day: 30
- Trading days: 250

**If successful:**
- Gross profit: $250,000+ (theoretically)
- Tax liability (32% bracket): $80,000+
- Net after tax: $170,000

**If breakeven (50% win rate):**
- Gross trades: 7,500+
- Each trade generates a taxable event
- Accounting/software costs: $500-2,000/year minimum
- Tax prep complexity: Significant

### Record-Keeping Requirements

For each trade, you must track:
1. Date and time of acquisition
2. Date and time of disposal
3. Cost basis (purchase price + fees)
4. Proceeds (sale price - fees)
5. Gain or loss

**The system tracks this** in `execution_ledger_workflow.py`:
- Entry prices (weighted average)
- Exit prices and quantities
- Realized P&L per trade
- Transaction history with timestamps

### Tax-Efficient Strategies

1. **Specific Identification Method**
   - Choose which lots to sell (highest cost first = lower gains)
   - Requires detailed record-keeping

2. **Tax-Loss Harvesting**
   - Realize losses to offset gains
   - Be cautious of potential wash sale rules

3. **Consider Trading in Tax-Advantaged Accounts**
   - Some platforms allow crypto in IRAs
   - Deferred or tax-free gains

4. **Set Aside Tax Reserves**
   - Recommendation: 30-40% of gross profits
   - Avoid surprises at tax time

### Recommended Actions

1. **Consult a Tax Professional** familiar with crypto
2. **Use Crypto Tax Software** (CoinTracker, Koinly, TaxBit)
3. **Export Transaction History** regularly
4. **Estimate Quarterly Taxes** to avoid penalties
5. **Document Your Strategy** for potential audit defense

---

## Recommended Testing Approach

### Phase 1: Backtest Validation (Week 1)

1. **Run baseline backtest** with current conservative settings
2. **Run aggressive backtest** with modified parameters
3. **Compare metrics**: Returns, drawdown, trade frequency, Sharpe ratio

```bash
# Conservative baseline
uv run python -m backtesting.cli \
  --llm-strategist enabled \
  --llm-prompt prompts/strategies/conservative_defensive.txt \
  --pair BTC-USD \
  --start 2024-01-01 \
  --end 2024-03-31 \
  --initial-cash 10000 \
  --llm-run-id conservative-baseline

# Aggressive test
uv run python -m backtesting.cli \
  --llm-strategist enabled \
  --llm-prompt prompts/strategies/aggressive_active.txt \
  --pair BTC-USD \
  --start 2024-01-01 \
  --end 2024-03-31 \
  --initial-cash 10000 \
  --max-position-risk-pct 4.0 \
  --max-daily-loss-pct 6.0 \
  --llm-run-id aggressive-test
```

### Phase 2: Parameter Sensitivity Analysis (Week 2)

Test variations of key parameters:

| Test | `max_position_risk_pct` | `min_hold_bars` | `trade_cooldown_bars` |
|------|------------------------|-----------------|----------------------|
| A | 2.0 | 4 | 2 |
| B | 3.0 | 2 | 1 |
| C | 4.0 | 1 | 0 |
| D | 5.0 | 0 | 0 |

### Phase 3: Paper Trading Validation (Weeks 3-4)

1. Deploy with mock ledger (no real money)
2. Monitor for 2+ weeks across different market conditions
3. Track: Actual trades vs backtest expectations, slippage, execution timing

### Phase 4: Small Capital Live Test (Weeks 5-8)

1. Start with minimal real capital ($100-500)
2. Use conservative tradeable fraction (10-20%)
3. Gradually increase based on performance
4. Monitor tax implications in real-time

### Success Criteria

Before scaling up, validate:

- [ ] Backtest shows positive Sharpe ratio > 1.0
- [ ] Maximum drawdown < 20%
- [ ] Trade frequency meets expectations (10-30/day)
- [ ] Paper trading matches backtest within 20%
- [ ] Tax tracking is automated and accurate
- [ ] Risk controls trigger appropriately

---

## Summary: Configuration Checklist for Aggressive Trading

### Code Changes Required

1. **Reduce anti-flip-flop protections** (`trigger_engine.py`):
   ```python
   min_hold_bars=1
   trade_cooldown_bars=0
   confidence_override_threshold="C"
   ```

2. **Lower minimum price move** (environment):
   ```bash
   EXECUTION_MIN_PRICE_MOVE_PCT=0.1
   ```

3. **Increase trade frequency limits** (strategy plan):
   ```json
   "max_trades_per_day": 30,
   "max_triggers_per_symbol_per_day": 10
   ```

4. **Increase risk parameters** (CLI or plan):
   ```bash
   --max-position-risk-pct 4.0
   --max-daily-loss-pct 6.0
   --max-portfolio-exposure-pct 100.0
   ```

### Realistic Expectations

| Metric | Conservative | Aggressive | Theoretical Max |
|--------|-------------|------------|-----------------|
| Daily trades | 1-3 | 10-30 | 50+ |
| Per-trade profit target | 2-5% | 0.5-1% | 0.25% |
| Daily return (good day) | 1-2% | 3-8% | 15%+ |
| Daily return (bad day) | -1% | -4% | -8% |
| Win rate needed | 40% | 55% | 60%+ |
| Tax complexity | Low | High | Extreme |

### Final Note

Achieving consistent 25% daily returns is extremely ambitious and would require:
- Very high win rate (>60%)
- Multiple successful trades per day
- Minimal drawdowns
- Perfect timing

More realistic aggressive targets:
- **Daily**: 1-3% average (with high variance)
- **Monthly**: 15-30% (compounded)
- **Annual**: 200-500% (with significant drawdown risk)

The system CAN support aggressive trading, but the defaults are conservative by design. Test thoroughly before deploying with real capital.
