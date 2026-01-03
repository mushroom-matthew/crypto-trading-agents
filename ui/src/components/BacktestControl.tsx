import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from 'recharts';
import { PlayCircle, Loader2, TrendingUp, TrendingDown, Info } from 'lucide-react';
import { backtestAPI, type BacktestConfig } from '../lib/api';
import { cn, formatCurrency, formatPercent, formatDateTime } from '../lib/utils';
import { BACKTEST_PRESETS } from '../lib/presets';

export function BacktestControl() {
  const [config, setConfig] = useState<BacktestConfig>({
    symbols: ['BTC-USD'],
    timeframe: '15m',
    start_date: '2024-01-01',
    end_date: '2024-01-31',
    initial_cash: 10000,
    strategy: 'baseline',
  });

  const [selectedRun, setSelectedRun] = useState<string | null>(null);

  // Load preset configuration
  const loadPreset = (presetId: string) => {
    const preset = BACKTEST_PRESETS.find((p) => p.id === presetId);
    if (preset) {
      setConfig(preset.config);
    }
  };

  // Mutation to start backtest
  const startBacktest = useMutation({
    mutationFn: (config: BacktestConfig) => backtestAPI.create(config),
    onSuccess: (data) => {
      setSelectedRun(data.run_id);
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

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    startBacktest.mutate(config);
  };

  const isRunning = backtest?.status === 'running' || backtest?.status === 'queued';
  const isComplete = backtest?.status === 'completed';

  return (
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

            {/* Symbols */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Trading Pairs
              </label>
              <input
                type="text"
                value={config.symbols.join(', ')}
                onChange={(e) => {
                  const value = e.target.value;
                  setConfig({
                    ...config,
                    symbols: value
                      .split(',')
                      .map((s) => s.trim())
                      .filter(Boolean),
                  });
                }}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 focus:ring-2 focus:ring-blue-500 focus:border-transparent font-mono text-sm"
                placeholder="BTC-USD, ETH-USD"
                disabled={isRunning}
              />
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Enter one or more symbols separated by commas (e.g., BTC-USD, ETH-USD)
              </p>
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

              {results.sharpe_ratio !== undefined && (
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

              {results.total_trades !== undefined && (
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
                  formatter={(value: number) => [formatCurrency(value), 'Equity']}
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
        </div>
      )}
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
