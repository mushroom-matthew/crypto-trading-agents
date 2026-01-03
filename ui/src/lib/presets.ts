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
];
