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

// Error interceptor to extract FastAPI error details
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Extract detail message from FastAPI error response
    if (error.response?.data?.detail) {
      error.message = error.response.data.detail;
    } else if (error.response?.data?.message) {
      error.message = error.response.data.message;
    }
    return Promise.reject(error);
  }
);

// ============================================================================
// Type Definitions
// ============================================================================

export interface BacktestConfig {
  symbols: string[];
  timeframe: string;
  start_date: string;
  end_date: string;
  initial_cash: number;
  initial_allocations?: Record<string, number>;
  strategy?: string;
  strategy_id?: string;
  strategy_prompt?: string;
  use_llm_shim?: boolean;
  use_judge_shim?: boolean;

  // Risk Engine Parameters
  max_position_risk_pct?: number;
  max_symbol_exposure_pct?: number;
  max_portfolio_exposure_pct?: number;
  max_daily_loss_pct?: number;
  max_daily_risk_budget_pct?: number;

  // Trade Frequency Parameters
  max_trades_per_day?: number;
  max_triggers_per_symbol_per_day?: number;
  judge_cadence_hours?: number;
  judge_check_after_trades?: number;
  replan_on_day_boundary?: boolean;

  // Whipsaw / Anti-Flip-Flop Controls
  min_hold_hours?: number;
  min_flat_hours?: number;
  confidence_override_threshold?: string | null;
  exit_binding_mode?: 'none' | 'category';
  conflicting_signal_policy?: 'ignore' | 'exit' | 'reverse' | 'defer';

  // Execution Gating
  min_price_move_pct?: number;

  // Walk-Away Threshold
  walk_away_enabled?: boolean;
  walk_away_profit_target_pct?: number;

  // Flattening Options
  flatten_positions_daily?: boolean;

  // Debug/Diagnostic Options
  debug_trigger_sample_rate?: number;
  debug_trigger_max_samples?: number;
  indicator_debug_mode?: string;
  indicator_debug_keys?: string[];

  // Learning Book
  learning_book_enabled?: boolean;
  learning_daily_risk_budget_pct?: number;
  learning_max_trades_per_day?: number;
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

export interface BacktestListItem {
  run_id: string;
  status: string;
  progress: number;
  started_at?: string;
  completed_at?: string;
  // Configuration metadata
  symbols: string[];
  strategy?: string;
  strategy_id?: string;
  timeframe?: string;
  start_date?: string;
  end_date?: string;
  initial_cash?: number;
  // Performance metrics (for completed backtests)
  return_pct?: number;
  final_equity?: number;
  total_trades?: number;
  sharpe_ratio?: number;
  max_drawdown_pct?: number;
  win_rate?: number;
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
  // Risk stats (Phase 6 trade-level visibility)
  risk_used_abs?: number;
  actual_risk_at_stop?: number;
  stop_distance?: number;
  allocated_risk_abs?: number;
  profile_multiplier?: number;
  r_multiple?: number;
}

export interface PairedTrade {
  symbol: string;
  side: string;
  entry_timestamp: string;
  exit_timestamp: string;
  entry_price?: number;
  exit_price?: number;
  entry_trigger?: string;
  exit_trigger?: string;
  entry_timeframe?: string;
  qty?: number;
  pnl?: number;
  fees?: number;
  hold_duration_hours?: number;
  risk_used_abs?: number;
  actual_risk_at_stop?: number;
  r_multiple?: number;
}

export interface TradeLeg {
  leg_id: string;
  side: string;
  qty: number;
  price: number;
  fees: number;
  timestamp: string;
  trigger_id?: string;
  category?: string;
  reason?: string;
  is_entry: boolean;
  exit_fraction?: number;
  wac_at_fill?: number;
  realized_pnl?: number;
  position_after?: number;
  learning_book: boolean;
  experiment_id?: string;
}

export interface TradeSet {
  set_id: string;
  symbol: string;
  timeframe?: string;
  opened_at: string;
  closed_at?: string;
  legs: TradeLeg[];
  pnl_realized_total: number;
  fees_total: number;
  entry_side: string;
  // Computed fields
  num_legs: number;
  num_entries: number;
  num_exits: number;
  is_closed: boolean;
  hold_duration_hours?: number;
  avg_entry_price?: number;
  avg_exit_price?: number;
  total_entry_qty: number;
  total_exit_qty: number;
  max_exposure: number;
  entry_trigger?: string;
  exit_trigger?: string;
  learning_book: boolean;
  experiment_id?: string;
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
  emitted_at?: string;
  source: string;
  type: string;
  payload: Record<string, any>;
  run_id?: string;
  correlation_id?: string;
}

export interface WorkflowSummary {
  run_id: string;
  latest_plan_id?: string;
  latest_judge_id?: string;
  status: 'running' | 'paused' | 'stopped';
  last_updated: string;
  mode: 'paper' | 'live';
}

export interface LLMTelemetry {
  run_id?: string;
  plan_id?: string;
  prompt_hash?: string;
  model: string;
  tokens_in: number;
  tokens_out: number;
  cost_estimate: number;
  duration_ms: number;
  ts: string;
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

  // Get paired (round-trip) trades (legacy 1:1 format)
  getPairedTrades: async (runId: string): Promise<PairedTrade[]> => {
    const response = await api.get(`/backtests/${runId}/paired_trades`);
    return response.data;
  },

  // Get trade sets (position lifecycle format with multiple legs)
  getTradeSets: async (runId: string): Promise<TradeSet[]> => {
    const response = await api.get(`/backtests/${runId}/trade_sets`);
    return response.data;
  },

  // List all backtests with rich metadata
  list: async (status?: string, limit = 50): Promise<BacktestListItem[]> => {
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

  listBacktests: async (status?: string, limit = 50): Promise<BacktestListItem[]> => {
    const response = await api.get('/backtests', {
      params: { status, limit },
    });
    return response.data;
  },

  // Signal diagnostics endpoints
  getTriggerAnalytics: async (runId: string) => {
    const response = await api.get(`/backtests/${runId}/trigger-analytics`);
    return response.data;
  },

  getTriggerSamples: async (runId: string, params?: {
    trigger_id?: string;
    symbol?: string;
    result?: boolean;
    limit?: number;
    offset?: number;
  }) => {
    const response = await api.get(`/backtests/${runId}/trigger-samples`, { params });
    return response.data;
  },

  getBlockAnalysis: async (runId: string) => {
    const response = await api.get(`/backtests/${runId}/block-analysis`);
    return response.data;
  },

  getJudgeHistory: async (runId: string) => {
    const response = await api.get(`/backtests/${runId}/judge-history`);
    return response.data;
  },

  getBarDecisions: async (runId: string, params?: {
    date?: string;
    has_orders?: boolean;
    symbol?: string;
    limit?: number;
    offset?: number;
  }) => {
    const response = await api.get(`/backtests/${runId}/bar-decisions`, { params });
    return response.data;
  },

  getDailyDiagnostics: async (runId: string, date?: string) => {
    const response = await api.get(`/backtests/${runId}/daily-diagnostics`, {
      params: date ? { date } : undefined,
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

  // Get LLM telemetry entries
  getLLMTelemetry: async (params?: {
    run_id?: string;
    limit?: number;
    since?: string;
  }): Promise<LLMTelemetry[]> => {
    const response = await api.get('/llm/telemetry', { params });
    return response.data;
  },

  // List workflow/run status for agents
  listWorkflows: async (): Promise<WorkflowSummary[]> => {
    const response = await api.get('/workflows');
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

// Prompt management types
export interface PromptInfo {
  name: string;
  content: string;
  file_path: string;
}

export interface PromptListResponse {
  prompts: string[];
}

export interface PromptVersion {
  version_id: string;
  timestamp: string;
  file_path: string;
  size_bytes: number;
}

export interface PromptVersionsResponse {
  name: string;
  versions: PromptVersion[];
}

export interface StrategyInfo {
  id: string;
  name: string;
  description: string;
  file_path: string;
}

export interface StrategiesListResponse {
  strategies: StrategyInfo[];
}

// Regime types for market period selection
export interface RegimeInfo {
  id: string;
  name: string;
  description: string;
  character: 'bull' | 'bear' | 'volatile' | 'ranging' | 'unknown';
  start_date: string;
  end_date: string;
}

export interface RegimesListResponse {
  regimes: RegimeInfo[];
}

export const promptsAPI = {
  // List available prompts
  list: async (): Promise<PromptListResponse> => {
    const response = await api.get('/prompts/');
    return response.data;
  },

  // Get a specific prompt by name
  get: async (name: string): Promise<PromptInfo> => {
    const response = await api.get(`/prompts/${name}`);
    return response.data;
  },

  // Update a prompt (creates a version backup first)
  update: async (name: string, content: string): Promise<PromptInfo> => {
    const response = await api.put(`/prompts/${name}`, { content });
    return response.data;
  },

  // Reset a prompt to default
  reset: async (name: string): Promise<PromptInfo> => {
    const response = await api.post(`/prompts/${name}/reset`);
    return response.data;
  },

  // List all versions of a prompt
  listVersions: async (name: string): Promise<PromptVersionsResponse> => {
    const response = await api.get(`/prompts/${name}/versions`);
    return response.data;
  },

  // Restore a specific version
  restoreVersion: async (name: string, versionId: string): Promise<PromptInfo> => {
    const response = await api.post(`/prompts/${name}/versions/${versionId}/restore`);
    return response.data;
  },

  // List all available strategy templates
  listStrategies: async (): Promise<StrategiesListResponse> => {
    const response = await api.get('/prompts/strategies/');
    return response.data;
  },

  // Get a specific strategy template
  getStrategy: async (strategyId: string): Promise<PromptInfo> => {
    const response = await api.get(`/prompts/strategies/${strategyId}`);
    return response.data;
  },
};

export const regimesAPI = {
  // List available market regimes
  list: async (): Promise<RegimesListResponse> => {
    const response = await api.get('/regimes/');
    return response.data;
  },
};

// ============================================================================
// Paper Trading Types
// ============================================================================

export interface PaperTradingSessionConfig {
  symbols: string[];
  initial_cash?: number;
  initial_allocations?: Record<string, number>;
  strategy_prompt?: string;
  strategy_id?: string;
  plan_interval_hours?: number;
  replan_on_day_boundary?: boolean;
  enable_symbol_discovery?: boolean;
  min_volume_24h?: number;
  llm_model?: string;

  // Risk Engine Parameters
  max_position_risk_pct?: number;
  max_symbol_exposure_pct?: number;
  max_portfolio_exposure_pct?: number;
  max_daily_loss_pct?: number;
  max_daily_risk_budget_pct?: number;

  // Trade Frequency Parameters
  max_trades_per_day?: number;
  max_triggers_per_symbol_per_day?: number;
  judge_cadence_hours?: number;
  judge_check_after_trades?: number;

  // Whipsaw / Anti-Flip-Flop Controls
  min_hold_hours?: number;
  min_flat_hours?: number;
  confidence_override_threshold?: string | null;
  exit_binding_mode?: 'none' | 'category';
  conflicting_signal_policy?: 'ignore' | 'exit' | 'reverse' | 'defer';

  // Execution Gating
  min_price_move_pct?: number;

  // Walk-Away Threshold
  walk_away_enabled?: boolean;
  walk_away_profit_target_pct?: number;

  // Flattening Options
  flatten_positions_daily?: boolean;

  // Debug/Diagnostic Options
  debug_trigger_sample_rate?: number;
  debug_trigger_max_samples?: number;
  indicator_debug_mode?: string;
  indicator_debug_keys?: string[];
}

export interface PaperTradingSession {
  session_id: string;
  status: string;
  symbols: string[];
  cycle_count: number;
  has_plan: boolean;
  last_plan_time: string | null;
  plan_interval_hours: number;
}

export interface PositionMeta {
  entry_trigger_id: string | null;
  entry_category: string | null;
  entry_side: string | null;
  opened_at: string | null;
  stop_price_abs: number | null;
  target_price_abs: number | null;
}

export interface PaperTradingPortfolio {
  cash: number;
  positions: Record<string, number>;
  entry_prices: Record<string, number>;
  last_prices: Record<string, number>;
  total_equity: number;
  unrealized_pnl: number;
  realized_pnl: number;
  position_meta: Record<string, PositionMeta>;
}

export interface PaperTradingTrigger {
  id: string;
  symbol: string;
  category: string;
  direction: string;
  timeframe: string;
  confidence: string | null;
  entry_rule: string | null;
}

export interface PaperTradingPlan {
  generated_at: string | null;
  valid_until: string | null;
  trigger_count: number;
  allowed_symbols: string[];
  max_trades_per_day: number | null;
  global_view: string | null;
  regime: string | null;
  triggers: PaperTradingTrigger[];
}

export interface PaperTradingActivityEvent {
  event_id: string;
  type: string;
  ts: string;
  payload: Record<string, any>;
  source: string;
}

export interface PaperTradingTrade {
  timestamp: string;
  symbol: string;
  side: string;
  qty: number;
  price: number;
  fee: number | null;
  pnl: number | null;
}

export interface SessionListItem {
  session_id: string;
  status: string;
  start_time: string | null;
  close_time: string | null;
}

export interface PaperTradingPlanRecord {
  plan_index: number;
  generated_at: string;
  trigger_count: number;
  max_trades_per_day: number | null;
  market_regime: string | null;
  symbols: string[];
  valid_until: string | null;
  triggers: Array<{
    id: string;
    symbol: string;
    direction: string;
    timeframe?: string;
  }>;
}

export interface PaperTradingPlanHistory {
  session_id: string;
  total_plans: number;
  plans: PaperTradingPlanRecord[];
}

export interface EquitySnapshot {
  timestamp: string;
  cash: number;
  total_equity: number;
  positions: Record<string, number>;
  unrealized_pnl: number;
  realized_pnl: number;
}

export interface PaperTradingEquityCurve {
  session_id: string;
  total_snapshots: number;
  equity_curve: EquitySnapshot[];
}

export const paperTradingAPI = {
  // Start a new paper trading session
  startSession: async (config: PaperTradingSessionConfig): Promise<{ session_id: string; status: string; message: string }> => {
    const response = await api.post('/paper-trading/sessions', config);
    return response.data;
  },

  // Get session status
  getSession: async (sessionId: string): Promise<PaperTradingSession> => {
    const response = await api.get(`/paper-trading/sessions/${sessionId}`);
    return response.data;
  },

  // Stop a session
  stopSession: async (sessionId: string): Promise<{ session_id: string; status: string; message: string }> => {
    const response = await api.post(`/paper-trading/sessions/${sessionId}/stop`);
    return response.data;
  },

  // Get portfolio status
  getPortfolio: async (sessionId: string): Promise<PaperTradingPortfolio> => {
    const response = await api.get(`/paper-trading/sessions/${sessionId}/portfolio`);
    return response.data;
  },

  // Get current strategy plan
  getPlan: async (sessionId: string): Promise<PaperTradingPlan> => {
    const response = await api.get(`/paper-trading/sessions/${sessionId}/plan`);
    return response.data;
  },

  // Force regeneration of strategy plan
  forceReplan: async (sessionId: string): Promise<{ session_id: string; status: string; message: string }> => {
    const response = await api.post(`/paper-trading/sessions/${sessionId}/replan`);
    return response.data;
  },

  // Get trade history
  getTrades: async (sessionId: string, limit = 100): Promise<PaperTradingTrade[]> => {
    const response = await api.get(`/paper-trading/sessions/${sessionId}/trades`, {
      params: { limit },
    });
    return response.data;
  },

  // Update symbols
  updateSymbols: async (sessionId: string, symbols: string[]): Promise<{ session_id: string; symbols: string[]; message: string }> => {
    const response = await api.post(`/paper-trading/sessions/${sessionId}/symbols`, symbols);
    return response.data;
  },

  // List all sessions
  listSessions: async (status?: string, limit = 20): Promise<{ sessions: SessionListItem[]; count: number }> => {
    const response = await api.get('/paper-trading/sessions', {
      params: { status, limit },
    });
    return response.data;
  },

  // Get plan history for a session (LLM insights)
  getPlanHistory: async (sessionId: string, limit = 50): Promise<PaperTradingPlanHistory> => {
    const response = await api.get(`/paper-trading/sessions/${sessionId}/plans`, {
      params: { limit },
    });
    return response.data;
  },

  // Get equity curve for a session
  getEquityCurve: async (sessionId: string, limit = 500): Promise<PaperTradingEquityCurve> => {
    const response = await api.get(`/paper-trading/sessions/${sessionId}/equity`, {
      params: { limit },
    });
    return response.data;
  },

  // Update strategy prompt for a running session
  updateStrategy: async (sessionId: string, strategyPrompt: string): Promise<{ session_id: string; status: string; message: string }> => {
    const response = await api.put(`/paper-trading/sessions/${sessionId}/strategy`, {
      strategy_prompt: strategyPrompt,
    });
    return response.data;
  },

  // Get recent activity events for a session (ticks, trigger_fired, trade_blocked, order_executed)
  getActivity: async (sessionId: string, limit = 40): Promise<{ session_id: string; events: PaperTradingActivityEvent[] }> => {
    const response = await api.get(`/paper-trading/sessions/${sessionId}/activity`, {
      params: { limit },
    });
    return response.data;
  },
};

export default api;
