import axios from 'axios';

// API Base URL Configuration
// - Development: Empty string uses Vite proxy (configure in vite.config.ts)
// - Production: Port 8081 for Ops API (backtesting, monitoring, analytics)
// - Port 8080 is MCP Server (agent tools - NOT used by this UI)
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

export interface MarketTick {
  symbol: string;
  price: number;
  volume?: number;
  timestamp: string;
  source: string;
}

export interface AgentEvent {
  event_id: string;
  timestamp: string;
  source: string;
  type: string;
  payload: Record<string, any>;
  run_id?: string;
  correlation_id?: string;
}

// Playback types for interactive time-series navigation
export interface CandleWithIndicators {
  timestamp: string;
  symbol: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  sma_20?: number;
  sma_50?: number;
  ema_20?: number;
  rsi_14?: number;
  macd?: number;
  macd_signal?: number;
  macd_hist?: number;
  atr_14?: number;
  bb_upper?: number;
  bb_middle?: number;
  bb_lower?: number;
}

export interface PlaybackEvent {
  timestamp: string;
  event_type: string;
  symbol: string;
  data: Record<string, any>;
}

export interface PortfolioStateSnapshot {
  timestamp: string;
  cash: number;
  positions: Record<string, number>;
  equity: number;
  pnl: number;
  return_pct: number;
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

  // Playback endpoints for interactive time-series navigation
  getPlaybackCandles: async (
    runId: string,
    symbol: string,
    offset = 0,
    limit = 2000
  ): Promise<CandleWithIndicators[]> => {
    const response = await api.get(`/backtests/${runId}/playback/candles`, {
      params: { symbol, offset, limit },
    });
    return response.data;
  },

  getPlaybackEvents: async (
    runId: string,
    eventType?: string,
    symbol?: string,
    limit = 1000
  ): Promise<PlaybackEvent[]> => {
    const response = await api.get(`/backtests/${runId}/playback/events`, {
      params: { event_type: eventType, symbol, limit },
    });
    return response.data;
  },

  getStateSnapshot: async (
    runId: string,
    timestamp: string
  ): Promise<PortfolioStateSnapshot> => {
    const response = await api.get(`/backtests/${runId}/playback/state/${timestamp}`);
    return response.data;
  },

  getLLMInsights: async (runId: string) => {
    const response = await api.get(`/backtests/${runId}/llm-insights`);
    return response;
  },

  getProgress: async (runId: string) => {
    const response = await api.get(`/backtests/${runId}/progress`);
    return response;
  },

  listBacktests: async (status?: string, limit = 50) => {
    const response = await api.get('/backtests', {
      params: { status, limit },
    });
    return response.data;
  },
};

export const marketAPI = {
  // Get recent market ticks
  getTicks: async (symbol?: string, limit = 100): Promise<MarketTick[]> => {
    const response = await api.get('/market/ticks', {
      params: { symbol, limit },
    });
    return response.data;
  },

  // Get active symbols
  getSymbols: async (): Promise<string[]> => {
    const response = await api.get('/market/symbols');
    return response.data;
  },
};

export const agentAPI = {
  // Get agent events with optional filtering
  getEvents: async (params?: {
    type?: string;
    source?: string;
    run_id?: string;
    correlation_id?: string;
    since?: string;
    limit?: number;
  }): Promise<AgentEvent[]> => {
    const response = await api.get('/agents/events', { params });
    return response.data;
  },

  // Get event chain by correlation ID
  getEventChain: async (correlationId: string): Promise<AgentEvent[]> => {
    const response = await api.get(`/agents/events/correlation/${correlationId}`);
    return response.data;
  },
};

export const walletsAPI = {
  // List all wallets
  list: async () => {
    const response = await api.get('/wallets');
    return response.data;
  },

  // Trigger reconciliation
  reconcile: async (params: { threshold?: number }) => {
    const response = await api.post('/wallets/reconcile', params);
    return response.data;
  },

  // Get reconciliation history
  getHistory: async (limit = 50) => {
    const response = await api.get('/wallets/reconcile/history', {
      params: { limit },
    });
    return response.data;
  },
};

export default api;
