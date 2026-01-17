import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from 'recharts';
import { PlayCircle, Loader2, TrendingUp, TrendingDown, Info } from 'lucide-react';
import { backtestAPI, promptsAPI, type BacktestConfig } from '../lib/api';
import { cn, formatCurrency, formatPercent, formatDateTime } from '../lib/utils';
import { BACKTEST_PRESETS } from '../lib/presets';
import { MarketTicker } from './MarketTicker';
import { EventTimeline } from './EventTimeline';
import { CandlestickChart } from './CandlestickChart';
import { BacktestPlaybackViewer } from './BacktestPlaybackViewer';
import { LLMInsights } from './LLMInsights';
import { LiveProgressMonitor } from './LiveProgressMonitor';
import { BacktestHistoryPanel } from './BacktestHistoryPanel';
import { PromptEditor } from './PromptEditor';
import { AggressiveSettingsPanel } from './AggressiveSettingsPanel';
import { PlanningSettingsPanel } from './PlanningSettingsPanel';

export function BacktestControl() {
  const defaultConfig: BacktestConfig = {
    symbols: ['BTC-USD'],
    timeframe: '15m',
    start_date: '2024-01-01',
    end_date: '2024-01-31',
    initial_cash: 10000,
    strategy: 'baseline',
  };
  const [config, setConfig] = useState<BacktestConfig>(defaultConfig);
  const [symbolsInput, setSymbolsInput] = useState(defaultConfig.symbols.join(', '));
  const [symbolsError, setSymbolsError] = useState<string | null>(null);
  const [allocationInput, setAllocationInput] = useState('');
  const [allocationError, setAllocationError] = useState<string | null>(null);
  const [selectedStrategyId, setSelectedStrategyId] = useState<string>('default');

  const [selectedRun, setSelectedRun] = useState<string | null>(() => {
    // Restore from localStorage on mount
    return localStorage.getItem('selectedBacktestRunId');
  });

  // Fetch available strategies
  const { data: strategiesData } = useQuery({
    queryKey: ['strategies'],
    queryFn: () => promptsAPI.listStrategies(),
  });
  const strategies = strategiesData?.strategies || [];

  // Load preset configuration
  const loadPreset = (presetId: string) => {
    const preset = BACKTEST_PRESETS.find((p) => p.id === presetId);
    if (preset) {
      setConfig(preset.config);
      setSymbolsInput(preset.config.symbols.join(', '));
      setSymbolsError(null);
      setAllocationInput('');
      setAllocationError(null);
    }
  };

  // Mutation to start backtest
  const startBacktest = useMutation({
    mutationFn: (config: BacktestConfig) => backtestAPI.create(config),
    onSuccess: (data) => {
      setSelectedRun(data.run_id);
      localStorage.setItem('selectedBacktestRunId', data.run_id);
    },
  });

  // Query backtest status (poll every 2s when running)
  const { data: backtest } = useQuery({
    queryKey: ['backtest', selectedRun],
    queryFn: () => backtestAPI.getStatus(selectedRun!),
    enabled: !!selectedRun,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === 'running' || status === 'queued' ? 2000 : false;
    },
  });

  // Query equity curve when complete
  const { data: equity } = useQuery({
    queryKey: ['equity', selectedRun],
    queryFn: () => backtestAPI.getEquityCurve(selectedRun!),
    enabled: backtest?.status === 'completed',
  });

  // Query results when complete
  const { data: results } = useQuery({
    queryKey: ['results', selectedRun],
    queryFn: () => backtestAPI.getResults(selectedRun!),
    enabled: backtest?.status === 'completed',
  });

  const { data: trades = [] } = useQuery({
    queryKey: ['trades', selectedRun],
    queryFn: () => backtestAPI.getTrades(selectedRun!, 100),
    enabled: backtest?.status === 'completed',
  });

  const { data: candles = [] } = useQuery({
    queryKey: ['candles', selectedRun, config.symbols[0]],
    queryFn: () => backtestAPI.getPlaybackCandles(selectedRun!, config.symbols[0], 0, 2000),
    enabled: backtest?.status === 'completed' && !!config.symbols[0],
  });

  const parseAllocations = (
    value: string,
    symbols: string[],
  ): Record<string, number> | undefined => {
    const trimmed = value.trim();
    if (!trimmed) {
      return undefined;
    }
    const allocations: Record<string, number> = {};
    const baseMap = symbols.reduce<Record<string, string[]>>((acc, symbol) => {
      const base = symbol.split('-')[0];
      if (!acc[base]) {
        acc[base] = [];
      }
      acc[base].push(symbol);
      return acc;
    }, {});
    const entries = trimmed.split(',').map((entry) => entry.trim()).filter(Boolean);
    for (const entry of entries) {
      const [rawKey, rawValue] = entry.split(/[:=]/, 2).map((part) => part.trim());
      if (!rawKey || !rawValue) {
        throw new Error('Allocations must use key:value pairs (e.g., cash:2000, BTC:4000)');
      }
      const amount = Number(rawValue);
      if (!Number.isFinite(amount) || amount < 0) {
        throw new Error(`Allocation for ${rawKey} must be a non-negative number`);
      }
      const key = rawKey.toUpperCase();
      if (rawKey.toLowerCase() === 'cash') {
        allocations.cash = (allocations.cash || 0) + amount;
        continue;
      }
      if (symbols.includes(key)) {
        allocations[key] = (allocations[key] || 0) + amount;
        continue;
      }
      const baseMatches = baseMap[key] || [];
      if (baseMatches.length === 1) {
        const mapped = baseMatches[0];
        allocations[mapped] = (allocations[mapped] || 0) + amount;
        continue;
      }
      if (baseMatches.length > 1) {
        throw new Error(`Allocation symbol ${rawKey} is ambiguous (matches ${baseMatches.join(', ')})`);
      }
      throw new Error(`Allocation symbol ${rawKey} is not in the Trading Pairs list`);
    }
    return allocations;
  };

  const parseSymbols = (value: string): string[] => {
    const symbols = value
      .split(/[,\s]+/)
      .map((symbol) => symbol.trim().toUpperCase())
      .filter(Boolean);
    if (!symbols.length) {
      throw new Error('Enter at least one symbol');
    }
    return symbols;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    let symbols: string[];
    let allocations: Record<string, number> | undefined;
    try {
      symbols = parseSymbols(symbolsInput);
      setSymbolsError(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Invalid symbol format';
      setSymbolsError(message);
      return;
    }
    try {
      allocations = parseAllocations(allocationInput, symbols);
      setAllocationError(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Invalid allocation format';
      setAllocationError(message);
      return;
    }

    // Fetch strategy prompt if using LLM strategist
    let strategyPrompt: string | undefined;
    if (config.strategy === 'llm_strategist' && selectedStrategyId) {
      try {
        const strategyData = await promptsAPI.getStrategy(selectedStrategyId);
        strategyPrompt = strategyData.content;
      } catch (err) {
        console.error('Failed to fetch strategy prompt:', err);
        // Continue with default prompt if fetch fails
      }
    }

    const payload = {
      ...config,
      symbols,
      initial_allocations: allocations,
      strategy_id: selectedStrategyId,
      strategy_prompt: strategyPrompt,
    };
    setConfig(payload);
    startBacktest.mutate(payload);
  };

  const isRunning = backtest?.status === 'running' || backtest?.status === 'queued';
  const isComplete = backtest?.status === 'completed';

  return (
    <div className="space-y-6">
      {/* Market Ticker */}
      <MarketTicker />

      <div className="p-6 space-y-6 max-w-7xl mx-auto">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Backtest Control</h1>
            <p className="text-gray-500 dark:text-gray-400 mt-1">
              Configure and run backtests to evaluate trading strategies
            </p>
          </div>
        </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Configuration Form */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Configuration</h2>

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Preset Selector */}
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
              <label className="block text-sm font-medium mb-2">
                Load Preset Configuration
              </label>
              <select
                onChange={(e) => e.target.value && loadPreset(e.target.value)}
                className="w-full px-3 py-2 border border-blue-300 dark:border-blue-600 rounded-lg bg-white dark:bg-gray-700 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={isRunning}
                defaultValue=""
              >
                <option value="">-- Select a preset --</option>
                {BACKTEST_PRESETS.map((preset) => (
                  <option key={preset.id} value={preset.id}>
                    {preset.name}
                  </option>
                ))}
              </select>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                Choose a predefined configuration or customize below
              </p>
            </div>

            {/* Backtest History Panel */}
            <BacktestHistoryPanel
              selectedRunId={selectedRun}
              onSelect={(runId) => {
                setSelectedRun(runId);
                localStorage.setItem('selectedBacktestRunId', runId);
              }}
              maxItems={15}
            />

            {/* Symbols */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Trading Pairs
              </label>
              <input
                type="text"
                value={symbolsInput}
                onChange={(e) => {
                  setSymbolsInput(e.target.value);
                  setSymbolsError(null);
                }}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 focus:ring-2 focus:ring-blue-500 focus:border-transparent font-mono text-sm"
                placeholder="BTC-USD, ETH-USD"
                disabled={isRunning}
              />
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Enter one or more symbols separated by commas (e.g., BTC-USD, ETH-USD)
              </p>
              {symbolsError && (
                <p className="text-xs text-red-600 dark:text-red-400 mt-1">{symbolsError}</p>
              )}
            </div>

            {/* Timeframe */}
            <div>
              <div className="flex items-center gap-2 mb-2">
                <label className="block text-sm font-medium">
                  Candlestick Interval (Timeframe)
                </label>
                <div className="group relative">
                  <Info className="w-4 h-4 text-gray-400 cursor-help" />
                  <div className="invisible group-hover:visible absolute left-0 top-6 z-10 w-64 p-2 bg-gray-800 text-white text-xs rounded shadow-lg">
                    The time period for each candlestick. Smaller intervals (1m, 5m) for short-term trading, larger intervals (1h, 4h) for longer-term strategies.
                  </div>
                </div>
              </div>
              <select
                value={config.timeframe}
                onChange={(e) => setConfig({ ...config, timeframe: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={isRunning}
              >
                <option value="1m">1 minute (high frequency)</option>
                <option value="5m">5 minutes</option>
                <option value="15m">15 minutes</option>
                <option value="1h">1 hour</option>
                <option value="4h">4 hours (swing trading)</option>
              </select>
            </div>

            {/* Date Range */}
            <div>
              <label className="block text-sm font-medium mb-2">Date Range</label>
              <div className="flex gap-2">
                <input
                  type="date"
                  value={config.start_date}
                  onChange={(e) => setConfig({ ...config, start_date: e.target.value })}
                  className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  disabled={isRunning}
                />
                <span className="self-center text-gray-500">to</span>
                <input
                  type="date"
                  value={config.end_date}
                  onChange={(e) => setConfig({ ...config, end_date: e.target.value })}
                  className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  disabled={isRunning}
                />
              </div>
            </div>

            {/* Initial Cash */}
            <div>
              <label className="block text-sm font-medium mb-2">Initial Cash</label>
              <input
                type="number"
                value={config.initial_cash}
                onChange={(e) =>
                  setConfig({ ...config, initial_cash: parseFloat(e.target.value) })
                }
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                min="100"
                step="100"
                disabled={isRunning}
              />
            </div>

            {/* Starting Allocations */}
            <div>
              <label className="block text-sm font-medium mb-2">Starting Allocation (USD)</label>
              <input
                type="text"
                value={allocationInput}
                onChange={(e) => setAllocationInput(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 focus:ring-2 focus:ring-blue-500 focus:border-transparent font-mono text-sm"
                placeholder="cash:2000, BTC:4000, ETH:4000"
                disabled={isRunning}
              />
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Optional. Uses symbols from Trading Pairs (full or base ticker) and overrides Initial Cash.
              </p>
              {allocationError && (
                <p className="text-xs text-red-600 dark:text-red-400 mt-1">{allocationError}</p>
              )}
            </div>

            {/* Strategy */}
            <div>
              <label className="block text-sm font-medium mb-2">Strategy</label>
              <select
                value={config.strategy || 'baseline'}
                onChange={(e) => setConfig({ ...config, strategy: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={isRunning}
              >
                <option value="baseline">Baseline</option>
                <option value="llm_strategist">LLM Strategist</option>
              </select>
            </div>

            {/* Strategy Template (only for LLM Strategist) */}
            {config.strategy === 'llm_strategist' && (
              <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
                <label className="block text-sm font-medium mb-2 text-purple-800 dark:text-purple-300">
                  Strategy Template
                </label>
                <select
                  value={selectedStrategyId}
                  onChange={(e) => setSelectedStrategyId(e.target.value)}
                  className="w-full px-3 py-2 border border-purple-300 dark:border-purple-600 rounded-lg bg-white dark:bg-gray-700 focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  disabled={isRunning}
                >
                  {strategies.map((strategy) => (
                    <option key={strategy.id} value={strategy.id}>
                      {strategy.name}
                    </option>
                  ))}
                </select>
                {strategies.find(s => s.id === selectedStrategyId)?.description && (
                  <p className="text-xs text-purple-600 dark:text-purple-400 mt-2">
                    {strategies.find(s => s.id === selectedStrategyId)?.description}
                  </p>
                )}
              </div>
            )}

            {/* Advanced Trading Settings (Scalper Mode, Leverage, Walk-Away) */}
            <AggressiveSettingsPanel
              config={config}
              onChange={setConfig}
              disabled={isRunning}
            />
            {config.strategy === 'llm_strategist' && (
              <PlanningSettingsPanel
                config={config}
                onChange={setConfig}
                disabled={isRunning}
              />
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isRunning || startBacktest.isPending}
              className={cn(
                'w-full py-3 px-4 rounded-lg font-semibold flex items-center justify-center gap-2 transition-colors',
                isRunning || startBacktest.isPending
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
              )}
            >
              {isRunning || startBacktest.isPending ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <PlayCircle className="w-5 h-5" />
                  Start Backtest
                </>
              )}
            </button>
          </form>

          {/* Error Display */}
          {startBacktest.isError && (
            <div className="mt-4 p-4 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 rounded-lg">
              <p className="font-semibold">Error starting backtest</p>
              <p className="text-sm mt-1">{(startBacktest.error as Error).message}</p>
            </div>
          )}
        </div>

        {/* Status Panel */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Status</h2>

          {!selectedRun && (
            <div className="text-center py-12 text-gray-500">
              <PlayCircle className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p>Configure and start a backtest to see results</p>
            </div>
          )}

          {selectedRun && backtest && (
            <div className="space-y-4">
              {/* Run ID */}
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">Run ID</p>
                <p className="font-mono text-sm">{backtest.run_id}</p>
              </div>

              {/* Status Badge */}
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">Status</p>
                <span
                  className={cn(
                    'inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold',
                    backtest.status === 'completed' && 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
                    backtest.status === 'running' && 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400',
                    backtest.status === 'queued' && 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400',
                    backtest.status === 'failed' && 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400'
                  )}
                >
                  {backtest.status}
                </span>
              </div>

              {/* Progress Bar */}
              {(backtest.status === 'running' || backtest.status === 'queued') && (
                <div>
                  <div className="flex justify-between text-sm text-gray-500 mb-2">
                    <span>Progress</span>
                    <span>{backtest.progress.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${backtest.progress}%` }}
                    />
                  </div>
                  {backtest.candles_total && (
                    <p className="text-sm text-gray-500 mt-2">
                      {backtest.candles_processed?.toLocaleString()} /{' '}
                      {backtest.candles_total.toLocaleString()} candles
                    </p>
                  )}
                </div>
              )}

              {/* Timestamps */}
              {backtest.started_at && (
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Started</p>
                  <p className="text-sm">{formatDateTime(backtest.started_at)}</p>
                </div>
              )}

              {backtest.completed_at && (
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Completed</p>
                  <p className="text-sm">{formatDateTime(backtest.completed_at)}</p>
                </div>
              )}

              {/* Error */}
              {backtest.error && (
                <div className="p-4 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 rounded-lg">
                  <p className="font-semibold">Error</p>
                  <p className="text-sm mt-1">{backtest.error}</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Prompt Editor (collapsible) */}
      <PromptEditor />

      {/* Live Progress Monitor (shown while running) */}
      {selectedRun && backtest && (
        <LiveProgressMonitor runId={selectedRun} status={backtest.status} />
      )}

      {/* Results Section */}
      {isComplete && results && equity && (
        <div className="space-y-6">
          {/* Performance Metrics */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Performance Metrics</h2>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {results.final_equity !== undefined && (
                <MetricCard
                  label="Final Equity"
                  value={formatCurrency(results.final_equity)}
                  trend={results.final_equity >= config.initial_cash}
                />
              )}

              {results.equity_return_pct !== undefined && (
                <MetricCard
                  label="Return"
                  value={formatPercent(results.equity_return_pct)}
                  trend={results.equity_return_pct > 0}
                />
              )}

              {results.sharpe_ratio !== undefined && results.sharpe_ratio !== null && (
                <MetricCard
                  label="Sharpe Ratio"
                  value={results.sharpe_ratio.toFixed(2)}
                  trend={results.sharpe_ratio > 1}
                />
              )}

              {results.max_drawdown_pct !== undefined && (
                <MetricCard
                  label="Max Drawdown"
                  value={formatPercent(results.max_drawdown_pct)}
                  trend={results.max_drawdown_pct < 20}
                  inverse
                />
              )}

              {results.win_rate !== undefined && (
                <MetricCard
                  label="Win Rate"
                  value={formatPercent(results.win_rate)}
                  trend={results.win_rate > 50}
                />
              )}

              {results.total_trades !== undefined && results.total_trades !== null && (
                <MetricCard label="Total Trades" value={results.total_trades.toString()} />
              )}

              {results.profit_factor !== undefined && (
                <MetricCard
                  label="Profit Factor"
                  value={results.profit_factor.toFixed(2)}
                  trend={results.profit_factor > 1}
                />
              )}
            </div>
          </div>

          {/* Equity Curve */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Equity Curve</h2>

            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={equity}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={(value) => new Date(value).toLocaleDateString()}
                  stroke="#9CA3AF"
                />
                <YAxis tickFormatter={(value) => `$${value.toLocaleString()}`} stroke="#9CA3AF" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1F2937',
                    border: '1px solid #374151',
                    borderRadius: '0.5rem',
                  }}
                  labelFormatter={(value) => formatDateTime(value)}
                  formatter={(value?: number) => [formatCurrency(value ?? 0), 'Equity']}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="equity"
                  stroke="#3B82F6"
                  strokeWidth={2}
                  dot={false}
                  name="Equity"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Price Chart with Market Data */}
          {candles.length > 0 && (
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-semibold mb-4">Market Price Action</h2>
              <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                Candlestick chart showing price movements, technical indicators, and trade executions
              </p>
              <CandlestickChart
                candles={candles}
                trades={trades}
                currentIndex={candles.length - 1}
              />
            </div>
          )}
        </div>
      )}

        {/* Recent Trades */}
        {isComplete && trades && trades.length > 0 && (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Recent Trades</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-gray-500 border-b border-gray-200 dark:border-gray-700">
                    <th className="pb-3">Time</th>
                    <th className="pb-3">Symbol</th>
                    <th className="pb-3">Side</th>
                    <th className="pb-3">Trigger</th>
                    <th className="pb-3 text-right">Qty</th>
                    <th className="pb-3 text-right">Price</th>
                    <th className="pb-3 text-right">P&L</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  {trades.slice(0, 20).map((trade, idx) => (
                    <tr key={idx}>
                      <td className="py-2 text-gray-500">{formatDateTime(trade.timestamp)}</td>
                      <td className="py-2 font-mono">{trade.symbol}</td>
                      <td className="py-2">
                        <span className={cn(
                          'px-2 py-0.5 rounded text-xs font-semibold',
                          trade.side?.toLowerCase() === 'buy' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400'
                        )}>
                          {trade.side?.toUpperCase()}
                        </span>
                      </td>
                      <td className="py-2 text-xs font-mono text-gray-600 dark:text-gray-400 max-w-32 truncate" title={trade.trigger_id}>
                        {trade.trigger_id || '-'}
                      </td>
                      <td className="py-2 text-right">{trade.qty?.toFixed(6)}</td>
                      <td className="py-2 text-right">{formatCurrency(trade.price)}</td>
                      <td className={cn(
                        'py-2 text-right font-semibold',
                        trade.pnl && trade.pnl > 0 ? 'text-green-500' : trade.pnl && trade.pnl < 0 ? 'text-red-500' : 'text-gray-500'
                      )}>
                        {trade.pnl ? formatCurrency(trade.pnl) : '-'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* LLM Strategy Insights (only shown for LLM strategist) */}
        {isComplete && selectedRun && (
          <div className="mt-6">
            <LLMInsights runId={selectedRun} />
          </div>
        )}

        {/* Interactive Playback Viewer */}
        {isComplete && selectedRun && (
          <div className="mt-6">
            <h2 className="text-2xl font-semibold mb-4 text-gray-900 dark:text-gray-100">
              Interactive Playback
            </h2>
            <BacktestPlaybackViewer
              runId={selectedRun}
              symbol={config.symbols[0]}
            />
          </div>
        )}

        {/* Event Timeline */}
        <EventTimeline limit={30} runId={selectedRun || undefined} />
      </div>
    </div>
  );
}

// Helper component for metric cards
interface MetricCardProps {
  label: string;
  value: string;
  trend?: boolean;
  inverse?: boolean;
}

function MetricCard({ label, value, trend, inverse = false }: MetricCardProps) {
  const showTrend = trend !== undefined;
  const isPositive = inverse ? !trend : trend;

  return (
    <div className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
      <p className="text-sm text-gray-500 dark:text-gray-400">{label}</p>
      <div className="flex items-center gap-2 mt-1">
        <p className="text-2xl font-bold">{value}</p>
        {showTrend && (
          <>
            {isPositive ? (
              <TrendingUp className="w-5 h-5 text-green-500" />
            ) : (
              <TrendingDown className="w-5 h-5 text-red-500" />
            )}
          </>
        )}
      </div>
    </div>
  );
}
