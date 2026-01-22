# Live Trading Safety Guards

**Status**: ✅ Fully Implemented and Tested
**Date**: 2026-01-02

## Overview

This document describes the multi-layer safety system that prevents accidental live trading without explicit acknowledgment. The system implements **defense-in-depth** with multiple checkpoints from startup through execution.

## The Problem

Trading systems can cause real financial loss if:
- Commands execute real trades without user awareness
- Configuration mistakes switch to live mode unintentionally
- Development/testing code accidentally runs against production APIs

## The Solution: Multi-Layer Protection

```
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 1: Startup                         │
│  agents/runtime_mode.py - Block app start if misconfigured │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                   LAYER 2: API Gateway                      │
│     ops_api/app.py - Block destructive HTTP endpoints      │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                 LAYER 3: Order Execution                    │
│  mcp_server/app.py - Block before workflow submission      │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                 LAYER 4: Exchange API                       │
│ app/coinbase/advanced_trade.py - Block before API call     │
└─────────────────────────────────────────────────────────────┘
```

## Layer 1: Startup Validation

**File**: `agents/runtime_mode.py`
**Checkpoint**: Application startup

### How It Works

```python
runtime = get_runtime_mode()  # Called at app startup
runtime.ensure_valid()         # Raises RuntimeError if unsafe
```

### Protection

- ✅ Reads `TRADING_MODE` environment variable (default: "paper")
- ✅ Reads `LIVE_TRADING_ACK` environment variable (default: false)
- ✅ **BLOCKS STARTUP** if `TRADING_MODE=live` without `LIVE_TRADING_ACK=true`

### Error Message

```
RuntimeError: Live trading requested but LIVE_TRADING_ACK=true not set; refusing to start.
```

### Fail-Safe Behavior

- **Default mode**: PAPER (safe)
- **Live mode**: EXPLICITLY OPT-IN only

## Layer 2: API Middleware

**File**: `ops_api/app.py`
**Checkpoint**: HTTP request handling

### How It Works

```python
@app.middleware("http")
async def live_trading_safety_check(request: Request, call_next):
    # Check if destructive operation in live mode
    if is_destructive and runtime.is_live and not runtime.live_trading_ack:
        return 403 Forbidden
```

### Protected Endpoints

- `POST /backtests` - Could trigger backtest with live execution
- `POST /wallets/reconcile` - Could trigger corrective trades
- All `PUT` and `DELETE` operations on these paths

### Response (403 Forbidden)

```json
{
  "error": "LIVE_TRADING_NOT_ACKNOWLEDGED",
  "message": "Cannot execute destructive operations in live mode without explicit LIVE_TRADING_ACK=true environment variable. This endpoint could trigger real trades with real money. Set LIVE_TRADING_ACK=true to acknowledge.",
  "endpoint": "/backtests",
  "method": "POST",
  "runtime_mode": "live",
  "live_trading_ack": false
}
```

### Audit Logging

```
CRITICAL: LIVE MODE DESTRUCTIVE API CALL: POST /backtests (LIVE_TRADING_ACK=True)
INFO: PAPER MODE DESTRUCTIVE API CALL: POST /backtests
ERROR: BLOCKED API CALL: POST /backtests in live mode without LIVE_TRADING_ACK
```

## Layer 3: Order Execution Guard

**File**: `mcp_server/app.py`
**Function**: `place_mock_order()`
**Checkpoint**: Before Temporal workflow submission

### How It Works

```python
async def place_mock_order(orders: List[OrderIntent]) -> Dict[str, Any]:
    # SAFETY CHECK at function entry
    runtime = get_runtime_mode()

    if runtime.is_live and not runtime.live_trading_ack:
        return {
            "error": "LIVE_TRADING_NOT_ACKNOWLEDGED",
            "orders_blocked": len(orders),
            ...
        }
```

### Protection

- ✅ Checks runtime mode before workflow execution
- ✅ Returns error response (doesn't raise exception)
- ✅ Prevents ANY orders from reaching Temporal workflows

### Error Response

```json
{
  "error": "LIVE_TRADING_NOT_ACKNOWLEDGED",
  "message": "LIVE TRADING BLOCKED: Cannot execute real trades without explicit LIVE_TRADING_ACK=true environment variable. Set LIVE_TRADING_ACK=true to acknowledge you understand this will place real orders with real money.",
  "orders_blocked": 3,
  "runtime_mode": "live",
  "live_trading_ack": false
}
```

### Audit Logging

```
CRITICAL: LIVE TRADING ORDER REQUESTED: 3 orders, mode=live, ack=True
CRITICAL: LIVE TRADING ACKNOWLEDGED: Proceeding with 3 real orders (LIVE_TRADING_ACK=true)
ERROR: LIVE TRADING BLOCKED (from error response)
INFO: Processing 3 order(s) in paper mode
```

## Layer 4: Exchange API Guard

**File**: `app/coinbase/advanced_trade.py`
**Function**: `place_order()`
**Checkpoint**: Before HTTP call to Coinbase

### How It Works

```python
async def place_order(...) -> OrderResponse:
    # SAFETY CHECK before API call
    runtime = get_runtime_mode()

    if runtime.is_live and not runtime.live_trading_ack:
        raise RuntimeError("COINBASE ORDER BLOCKED: ...")
```

### Protection

- ✅ Final checkpoint before real money moves
- ✅ Raises RuntimeError (cannot be ignored)
- ✅ Warns if called in paper mode (should never happen)

### Error (RuntimeError)

```
RuntimeError: COINBASE ORDER BLOCKED: Cannot place real Coinbase order without explicit LIVE_TRADING_ACK=true environment variable. This would execute a real trade with real money. Set LIVE_TRADING_ACK=true to acknowledge and proceed.
```

### Audit Logging

```
CRITICAL: COINBASE LIVE ORDER: BUY BTC-USD 0.00100000 MARKET at MARKET, ack=True
CRITICAL: COINBASE ORDER ACKNOWLEDGED: Proceeding with REAL order (LIVE_TRADING_ACK=true)
WARNING: COINBASE API CALL IN PAPER MODE: BUY BTC-USD 0.00100000 MARKET - This should not happen in paper trading!
ERROR: (from RuntimeError)
```

## How to Enable Live Trading

### Step 1: Set Environment Variables

```bash
export TRADING_MODE=live
export LIVE_TRADING_ACK=true
```

### Step 2: Understand the Risks

By setting `LIVE_TRADING_ACK=true`, you acknowledge:
- ✅ Real orders will be placed with real money
- ✅ Real trades can result in financial loss
- ✅ You have tested thoroughly in paper mode
- ✅ You understand the system behavior

### Step 3: Verify Configuration

```bash
# Check runtime mode
curl http://localhost:8081/status

# Should show:
{
  "stack": "agent",
  "mode": "live",
  "live_trading_ack": true,
  "ui_unlock": false
}
```

### Step 4: Monitor Audit Logs

All live trading operations log at **CRITICAL** level:

```bash
grep "LIVE TRADING" logs/app.log
grep "COINBASE" logs/app.log
```

## Testing the Safety Guards

Run the validation script:

```bash
uv run python scripts/test_safety_guards.py
```

Expected output:
```
✅ Paper mode works without LIVE_TRADING_ACK
✅ Live mode without ACK correctly blocked
✅ Live mode with ACK works correctly
✅ place_mock_order safety guards implemented
✅ Coinbase place_order safety guards implemented
✅ API middleware safety guards implemented
🎉 All safety tests passed!
```

## Fail-Safe Principles

1. **Explicit Opt-In**: Live trading requires explicit acknowledgment
2. **Defense-in-Depth**: Multiple independent checkpoints
3. **Fail Closed**: Block by default, allow by exception
4. **Complete Audit Trail**: All live operations logged at CRITICAL level
5. **Early Detection**: Catch mistakes at startup before damage
6. **Clear Error Messages**: Tell user exactly what's wrong and how to fix

## Files Modified

| File | Change | Purpose |
|------|--------|---------|
| `agents/runtime_mode.py` | ✅ Already had checks | Startup validation |
| `mcp_server/app.py` | ✅ Added guard | Order execution |
| `app/coinbase/advanced_trade.py` | ✅ Added guard | Exchange API |
| `ops_api/app.py` | ✅ Added middleware | HTTP endpoints |
| `scripts/test_safety_guards.py` | ✅ Created | Validation |

## Summary

✅ **4 layers of protection** prevent accidental live trading
✅ **Complete audit trail** for all live operations
✅ **Fail-safe defaults** (paper mode, no acknowledgment)
✅ **Clear error messages** guide users to safe configuration
✅ **Tested and validated** with automated test suite

**No command can execute real trades without explicit LIVE_TRADING_ACK=true acknowledgment.**
