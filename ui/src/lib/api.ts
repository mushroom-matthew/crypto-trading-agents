import axios from 'axios';

// API Base URL - in development, use proxy; in production, use same host
const API_BASE_URL = import.meta.env.VITE_API_URL || (
  import.meta.env.DEV ? '' : 'http://localhost:8081'
);

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// ============================================================================
// Type Definitions
// ============================================================================

export interface BacktestConfig {
  symbols: string[];
  timeframe: string;
  start_date: string;
  end_date: string;
  initial_cash: number;
  strategy?: string;
}

export interface BacktestCreateResponse {
  run_id: string;
  status: string;
  message: string;
}

export interface BacktestStatus {
  run_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'not_implemented';
  progress: number;
  started_at?: string;
  completed_at?: string;
  candles_total?: number;
  candles_processed?: number;
  error?: string;
}

export interface BacktestResults {
  run_id: string;
  status: string;
  final_equity?: number;
  equity_return_pct?: number;
  sharpe_ratio?: number;
  max_drawdown_pct?: number;
  win_rate?: number;
  total_trades?: number;
  avg_win?: number;
  avg_loss?: number;
  profit_factor?: number;
}

export interface EquityCurvePoint {
  timestamp: string;
  equity: number;
}

export interface BacktestTrade {
  timestamp: string;
  symbol: string;
  side: string;
  qty: number;
  price: number;
  fee?: number;
  pnl?: number;
  trigger_id?: string;
}

// ============================================================================
// API Functions
// ============================================================================

export const backtestAPI = {
  // Start a new backtest
  create: async (config: BacktestConfig): Promise<BacktestCreateResponse> => {
    const response = await api.post('/backtests', config);
    return response.data;
  },

  // Get backtest status
  getStatus: async (runId: string): Promise<BacktestStatus> => {
    const response = await api.get(`/backtests/${runId}`);
    return response.data;
  },

  // Get backtest results
  getResults: async (runId: string): Promise<BacktestResults> => {
    const response = await api.get(`/backtests/${runId}/results`);
    return response.data;
  },

  // Get equity curve data
  getEquityCurve: async (runId: string): Promise<EquityCurvePoint[]> => {
    const response = await api.get(`/backtests/${runId}/equity`);
    return response.data;
  },

  // Get trade log
  getTrades: async (runId: string, limit = 100, offset = 0): Promise<BacktestTrade[]> => {
    const response = await api.get(`/backtests/${runId}/trades`, {
      params: { limit, offset },
    });
    return response.data;
  },

  // List all backtests
  list: async (status?: string, limit = 50): Promise<BacktestStatus[]> => {
    const response = await api.get('/backtests', {
      params: { status, limit },
    });
    return response.data;
  },
};

export default api;
