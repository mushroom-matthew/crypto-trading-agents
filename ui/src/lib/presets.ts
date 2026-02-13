import type { BacktestConfig } from './api';

export interface BacktestPreset {
  id: string;
  name: string;
  description: string;
  config: BacktestConfig;
}

export const BACKTEST_PRESETS: BacktestPreset[] = [
  {
    id: 'btc-quick',
    name: 'BTC Quick Test (1 week)',
    description: 'Fast 1-week BTC test with 15-minute candles',
    config: {
      symbols: ['BTC-USD'],
      timeframe: '15m',
      start_date: '2024-01-01',
      end_date: '2024-01-07',
      initial_cash: 10000,
      strategy: 'baseline',
    },
  },
  {
    id: 'btc-monthly',
    name: 'BTC Monthly Analysis',
    description: 'Full month BTC backtest with hourly candles',
    config: {
      symbols: ['BTC-USD'],
      timeframe: '1h',
      start_date: '2024-01-01',
      end_date: '2024-01-31',
      initial_cash: 10000,
      strategy: 'baseline',
    },
  },
  {
    id: 'multi-crypto',
    name: 'Multi-Crypto Portfolio',
    description: 'BTC + ETH portfolio test over 2 weeks',
    config: {
      symbols: ['BTC-USD', 'ETH-USD'],
      timeframe: '1h',
      start_date: '2024-01-01',
      end_date: '2024-01-14',
      initial_cash: 20000,
      strategy: 'baseline',
    },
  },
  {
    id: 'llm-strategist',
    name: 'LLM Strategist Test',
    description: 'Test LLM-based strategy on BTC with 4-hour candles',
    config: {
      symbols: ['BTC-USD'],
      timeframe: '4h',
      start_date: '2024-01-01',
      end_date: '2024-01-31',
      initial_cash: 10000,
      strategy: 'llm_strategist',
    },
  },
  {
    id: 'high-frequency',
    name: 'High Frequency (5min)',
    description: 'Short-term trading with 5-minute candles',
    config: {
      symbols: ['BTC-USD'],
      timeframe: '5m',
      start_date: '2024-01-15',
      end_date: '2024-01-16',
      initial_cash: 5000,
      strategy: 'baseline',
    },
  },
  // ============================================================================
  // Scalper Mode Presets - Aggressive short-term trading
  // ============================================================================
  {
    id: 'scalper-5m',
    name: 'Scalper Mode (5min)',
    description: 'High-frequency scalping with disabled whipsaw protection and 25% walk-away target',
    config: {
      symbols: ['BTC-USD'],
      timeframe: '5m',
      start_date: '2024-01-15',
      end_date: '2024-01-18',
      initial_cash: 10000,
      strategy: 'llm_strategist',
      strategy_id: 'aggressive_active',
      // Aggressive risk settings
      max_position_risk_pct: 4.0,
      max_symbol_exposure_pct: 50.0,
      max_portfolio_exposure_pct: 100.0,
      max_daily_loss_pct: 6.0,
      max_daily_risk_budget_pct: 15.0,
      // High trade frequency
      max_trades_per_day: 50,
      max_triggers_per_symbol_per_day: 15,
      judge_cadence_hours: 4,
      // Whipsaw protection DISABLED for scalping
      min_hold_hours: 0,
      min_flat_hours: 0,
      confidence_override_threshold: null,
      // Low price move threshold
      min_price_move_pct: 0.1,
      // Walk-away after hitting profit target
      walk_away_enabled: true,
      walk_away_profit_target_pct: 25.0,
    },
  },
  {
    id: 'scalper-15m',
    name: 'Scalper Mode (15min)',
    description: 'Short-term trading with 15-min candles and aggressive settings',
    config: {
      symbols: ['BTC-USD', 'ETH-USD'],
      timeframe: '15m',
      start_date: '2024-01-15',
      end_date: '2024-01-22',
      initial_cash: 10000,
      strategy: 'llm_strategist',
      strategy_id: 'aggressive_active',
      // Moderate-aggressive risk settings
      max_position_risk_pct: 3.0,
      max_symbol_exposure_pct: 40.0,
      max_portfolio_exposure_pct: 100.0,
      max_daily_loss_pct: 5.0,
      max_daily_risk_budget_pct: 10.0,
      // High trade frequency
      max_trades_per_day: 30,
      max_triggers_per_symbol_per_day: 10,
      judge_cadence_hours: 6,
      // Minimal whipsaw protection
      min_hold_hours: 0.25,  // 15 minutes
      min_flat_hours: 0.25,
      confidence_override_threshold: 'C',
      // Low price move threshold
      min_price_move_pct: 0.15,
      // Walk-away after hitting profit target
      walk_away_enabled: true,
      walk_away_profit_target_pct: 20.0,
    },
  },
  {
    id: 'scalper-leverage-2x',
    name: 'Scalper with 2x Leverage',
    description: '2x leverage scalping - higher risk/reward',
    config: {
      symbols: ['BTC-USD'],
      timeframe: '15m',
      start_date: '2024-01-15',
      end_date: '2024-01-22',
      initial_cash: 10000,
      strategy: 'llm_strategist',
      strategy_id: 'aggressive_active',
      // 2x leverage via portfolio exposure
      max_position_risk_pct: 3.0,
      max_symbol_exposure_pct: 100.0,
      max_portfolio_exposure_pct: 200.0,  // 2x leverage
      max_daily_loss_pct: 10.0,
      max_daily_risk_budget_pct: 20.0,
      // High trade frequency
      max_trades_per_day: 30,
      max_triggers_per_symbol_per_day: 10,
      judge_cadence_hours: 6,
      // Minimal whipsaw protection
      min_hold_hours: 0.25,
      min_flat_hours: 0.25,
      confidence_override_threshold: 'B',
      // Low price move threshold
      min_price_move_pct: 0.1,
      // Walk-away - lower target due to leverage amplification
      walk_away_enabled: true,
      walk_away_profit_target_pct: 15.0,
    },
  },
  {
    id: 'conservative-daily',
    name: 'Conservative Daily',
    description: 'Standard whipsaw protection with moderate settings',
    config: {
      symbols: ['BTC-USD'],
      timeframe: '1h',
      start_date: '2024-01-01',
      end_date: '2024-01-31',
      initial_cash: 10000,
      strategy: 'llm_strategist',
      strategy_id: 'conservative_defensive',
      // Conservative risk settings
      max_position_risk_pct: 2.0,
      max_symbol_exposure_pct: 25.0,
      max_portfolio_exposure_pct: 80.0,
      max_daily_loss_pct: 3.0,
      max_daily_risk_budget_pct: 6.0,
      // Low trade frequency
      max_trades_per_day: 10,
      max_triggers_per_symbol_per_day: 5,
      judge_cadence_hours: 12,
      // Full whipsaw protection
      min_hold_hours: 2.0,
      min_flat_hours: 2.0,
      confidence_override_threshold: 'A',
      // Standard price move threshold
      min_price_move_pct: 0.5,
      // No walk-away (let positions run)
      walk_away_enabled: false,
    },
  },
];
