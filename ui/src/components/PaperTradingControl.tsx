import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { PlayCircle, StopCircle, Loader2, RefreshCw, Activity, Zap } from 'lucide-react';
import { paperTradingAPI, promptsAPI, type PaperTradingSessionConfig } from '../lib/api';
import { cn, formatCurrency, formatDateTime } from '../lib/utils';
import { PromptEditor } from './PromptEditor';
import { EventTimeline } from './EventTimeline';
import { AggressiveSettingsPanel, type AggressiveSettings } from './AggressiveSettingsPanel';
import { PlanningSettingsPanel, type PlanningSettings } from './PlanningSettingsPanel';

export function PaperTradingControl() {
  const queryClient = useQueryClient();

  // Form state
  const [symbolsInput, setSymbolsInput] = useState('BTC-USD, ETH-USD');
  const [initialCash, setInitialCash] = useState(10000);
  const [allocationInput, setAllocationInput] = useState('');
  const [selectedStrategyId, setSelectedStrategyId] = useState('default');
  const [planIntervalHours, setPlanIntervalHours] = useState(4);
  const [enableDiscovery, setEnableDiscovery] = useState(false);

  // Aggressive settings state (defaults match conservative settings)
  const [aggressiveSettings, setAggressiveSettings] = useState<AggressiveSettings>({
    max_position_risk_pct: 2.0,
    max_symbol_exposure_pct: 25.0,
    max_portfolio_exposure_pct: 80.0,
    max_daily_loss_pct: 3.0,
    min_hold_hours: 2.0,
    min_flat_hours: 2.0,
    confidence_override_threshold: 'A',
    min_price_move_pct: 0.5,
    walk_away_enabled: false,
    walk_away_profit_target_pct: 25.0,
    flatten_positions_daily: false,
  });
  const [planningSettings, setPlanningSettings] = useState<PlanningSettings>({
    max_trades_per_day: 10,
    max_triggers_per_symbol_per_day: 5,
  });

  // Selected session
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(() => {
    return localStorage.getItem('selectedPaperTradingSessionId');
  });

  // Fetch strategies
  const { data: strategiesData } = useQuery({
    queryKey: ['strategies'],
    queryFn: () => promptsAPI.listStrategies(),
  });
  const strategies = strategiesData?.strategies || [];

  // Fetch session list
  const { data: sessionsData } = useQuery({
    queryKey: ['paper-trading-sessions'],
    queryFn: () => paperTradingAPI.listSessions(undefined, 20),
    refetchInterval: 10000,
  });

  // Fetch selected session status
  const { data: session } = useQuery({
    queryKey: ['paper-trading-session', selectedSessionId],
    queryFn: () => paperTradingAPI.getSession(selectedSessionId!),
    enabled: !!selectedSessionId,
    refetchInterval: (query) => {
      return query.state.data?.status === 'running' ? 5000 : false;
    },
  });

  // Fetch portfolio for selected session
  const { data: portfolio } = useQuery({
    queryKey: ['paper-trading-portfolio', selectedSessionId],
    queryFn: () => paperTradingAPI.getPortfolio(selectedSessionId!),
    enabled: !!selectedSessionId && session?.status === 'running',
    refetchInterval: 10000,
  });

  // Fetch current plan
  const { data: plan } = useQuery({
    queryKey: ['paper-trading-plan', selectedSessionId],
    queryFn: () => paperTradingAPI.getPlan(selectedSessionId!),
    enabled: !!selectedSessionId && session?.has_plan,
  });

  // Fetch trades
  const { data: trades = [] } = useQuery({
    queryKey: ['paper-trading-trades', selectedSessionId],
    queryFn: () => paperTradingAPI.getTrades(selectedSessionId!, 50),
    enabled: !!selectedSessionId,
    refetchInterval: 15000,
  });

  // Start session mutation
  const startSession = useMutation({
    mutationFn: async () => {
      // Parse symbols
      const symbols = symbolsInput
        .split(',')
        .map((s) => s.trim().toUpperCase())
        .filter(Boolean);

      if (symbols.length === 0) {
        throw new Error('Enter at least one symbol');
      }

      // Parse allocations
      let initialAllocations: Record<string, number> | undefined;
      if (allocationInput.trim()) {
        initialAllocations = {};
        const pairs = allocationInput.split(',');
        for (const pair of pairs) {
          const [key, valueStr] = pair.split(':').map((s) => s.trim());
          const value = parseFloat(valueStr);
          if (key && !isNaN(value)) {
            initialAllocations[key.toUpperCase()] = value;
          }
        }
      }

      // Fetch strategy prompt if not default
      let strategyPrompt: string | undefined;
      if (selectedStrategyId && selectedStrategyId !== 'default') {
        const strategyData = await promptsAPI.getStrategy(selectedStrategyId);
        strategyPrompt = strategyData.content;
      }

      const config: PaperTradingSessionConfig = {
        symbols,
        initial_cash: initialCash,
        initial_allocations: initialAllocations,
        strategy_id: selectedStrategyId,
        strategy_prompt: strategyPrompt,
        plan_interval_hours: planIntervalHours,
        enable_symbol_discovery: enableDiscovery,
        // Aggressive trading settings
        ...aggressiveSettings,
        ...planningSettings,
      };

      return paperTradingAPI.startSession(config);
    },
    onSuccess: (data) => {
      setSelectedSessionId(data.session_id);
      localStorage.setItem('selectedPaperTradingSessionId', data.session_id);
      queryClient.invalidateQueries({ queryKey: ['paper-trading-sessions'] });
    },
  });

  // Stop session mutation
  const stopSession = useMutation({
    mutationFn: () => paperTradingAPI.stopSession(selectedSessionId!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['paper-trading-session', selectedSessionId] });
      queryClient.invalidateQueries({ queryKey: ['paper-trading-sessions'] });
    },
  });

  // Force replan mutation
  const forceReplan = useMutation({
    mutationFn: () => paperTradingAPI.forceReplan(selectedSessionId!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['paper-trading-plan', selectedSessionId] });
    },
  });

  const isRunning = session?.status === 'running';
  const isStarting = startSession.isPending;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Activity className="w-6 h-6 text-green-500" />
            Paper Trading
          </h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            Run LLM strategies against live market data without risking real capital
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Configuration Form */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Session Configuration</h2>

          <div className="space-y-4">
            {/* Session Selector */}
            <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
              <label className="block text-sm font-medium mb-2">Active Sessions</label>
              <select
                value={selectedSessionId || ''}
                onChange={(e) => {
                  const id = e.target.value || null;
                  setSelectedSessionId(id);
                  if (id) localStorage.setItem('selectedPaperTradingSessionId', id);
                }}
                className="w-full px-3 py-2 border border-green-300 dark:border-green-600 rounded-lg bg-white dark:bg-gray-700 focus:ring-2 focus:ring-green-500 focus:border-transparent"
              >
                <option value="">-- Start a new session --</option>
                {sessionsData?.sessions.map((s) => (
                  <option key={s.session_id} value={s.session_id}>
                    {s.session_id} ({s.status})
                  </option>
                ))}
              </select>
            </div>

            {/* Symbols */}
            <div>
              <label className="block text-sm font-medium mb-2">Trading Pairs</label>
              <input
                type="text"
                value={symbolsInput}
                onChange={(e) => setSymbolsInput(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 focus:ring-2 focus:ring-green-500 focus:border-transparent font-mono text-sm"
                placeholder="BTC-USD, ETH-USD"
                disabled={isRunning}
              />
            </div>

            {/* Initial Cash */}
            <div>
              <label className="block text-sm font-medium mb-2">Initial Cash (USD)</label>
              <input
                type="number"
                value={initialCash}
                onChange={(e) => setInitialCash(parseFloat(e.target.value) || 0)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 focus:ring-2 focus:ring-green-500 focus:border-transparent"
                disabled={isRunning}
              />
            </div>

            {/* Initial Allocations */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Initial Allocations (Optional)
              </label>
              <input
                type="text"
                value={allocationInput}
                onChange={(e) => setAllocationInput(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 focus:ring-2 focus:ring-green-500 focus:border-transparent font-mono text-sm"
                placeholder="cash: 5000, BTC-USD: 3000, ETH-USD: 2000"
                disabled={isRunning}
              />
              <p className="text-xs text-gray-500 mt-1">
                Pre-allocate capital to specific assets (notional USD)
              </p>
            </div>

            {/* Strategy Selector */}
            <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
              <label className="block text-sm font-medium mb-2">Strategy Template</label>
              <select
                value={selectedStrategyId}
                onChange={(e) => setSelectedStrategyId(e.target.value)}
                className="w-full px-3 py-2 border border-purple-300 dark:border-purple-600 rounded-lg bg-white dark:bg-gray-700 focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                disabled={isRunning}
              >
                {strategies.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Plan Interval */}
            <div>
              <label className="block text-sm font-medium mb-2">Plan Interval (hours)</label>
              <select
                value={planIntervalHours}
                onChange={(e) => setPlanIntervalHours(parseFloat(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 focus:ring-2 focus:ring-green-500 focus:border-transparent"
                disabled={isRunning}
              >
                <option value={1}>Every 1 hour</option>
                <option value={2}>Every 2 hours</option>
                <option value={4}>Every 4 hours</option>
                <option value={8}>Every 8 hours</option>
                <option value={24}>Every 24 hours</option>
              </select>
            </div>

            {/* Symbol Discovery Toggle */}
            <div className="flex items-center gap-3">
              <input
                type="checkbox"
                id="enableDiscovery"
                checked={enableDiscovery}
                onChange={(e) => setEnableDiscovery(e.target.checked)}
                className="w-4 h-4 text-green-600 rounded"
                disabled={isRunning}
              />
              <label htmlFor="enableDiscovery" className="text-sm">
                Enable daily symbol discovery (add high-volume pairs automatically)
              </label>
            </div>

            {/* Aggressive Trading Settings */}
            <AggressiveSettingsPanel
              config={aggressiveSettings}
              onChange={setAggressiveSettings}
              disabled={isRunning}
            />
            <PlanningSettingsPanel
              config={planningSettings}
              onChange={setPlanningSettings}
              disabled={isRunning}
            />

            {/* Action Buttons */}
            <div className="flex gap-3 pt-4">
              {!isRunning ? (
                <button
                  onClick={() => startSession.mutate()}
                  disabled={isStarting}
                  className={cn(
                    'flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-lg font-medium transition-colors',
                    isStarting
                      ? 'bg-gray-400 cursor-not-allowed'
                      : 'bg-green-600 hover:bg-green-700 text-white'
                  )}
                >
                  {isStarting ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <PlayCircle className="w-5 h-5" />
                  )}
                  {isStarting ? 'Starting...' : 'Start Paper Trading'}
                </button>
              ) : (
                <>
                  <button
                    onClick={() => stopSession.mutate()}
                    disabled={stopSession.isPending}
                    className="flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-lg font-medium bg-red-600 hover:bg-red-700 text-white transition-colors"
                  >
                    <StopCircle className="w-5 h-5" />
                    Stop Session
                  </button>
                  <button
                    onClick={() => forceReplan.mutate()}
                    disabled={forceReplan.isPending}
                    className="flex items-center justify-center gap-2 px-4 py-3 rounded-lg font-medium bg-blue-600 hover:bg-blue-700 text-white transition-colors"
                  >
                    <RefreshCw className={cn('w-5 h-5', forceReplan.isPending && 'animate-spin')} />
                    Replan
                  </button>
                </>
              )}
            </div>

            {startSession.error && (
              <p className="text-red-500 text-sm">{(startSession.error as Error).message}</p>
            )}
          </div>
        </div>

        {/* Status & Portfolio */}
        <div className="space-y-6">
          {/* Session Status */}
          {session && (
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Zap className={cn('w-5 h-5', isRunning ? 'text-green-500' : 'text-gray-400')} />
                Session Status
              </h2>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-500">Status</p>
                  <p className={cn('font-semibold', isRunning ? 'text-green-600' : 'text-gray-600')}>
                    {session.status.toUpperCase()}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Cycle Count</p>
                  <p className="font-semibold">{session.cycle_count}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Symbols</p>
                  <p className="font-mono text-sm">{session.symbols.join(', ')}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Last Plan</p>
                  <p className="text-sm">
                    {session.last_plan_time ? formatDateTime(session.last_plan_time) : 'None'}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Portfolio Status */}
          {portfolio && (
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-semibold mb-4">Portfolio</h2>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-500">Cash</p>
                  <p className="font-semibold text-lg">{formatCurrency(portfolio.cash)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Total Equity</p>
                  <p className="font-semibold text-lg">{formatCurrency(portfolio.total_equity)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Unrealized P&L</p>
                  <p className={cn('font-semibold', portfolio.unrealized_pnl >= 0 ? 'text-green-600' : 'text-red-600')}>
                    {formatCurrency(portfolio.unrealized_pnl)}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Realized P&L</p>
                  <p className={cn('font-semibold', portfolio.realized_pnl >= 0 ? 'text-green-600' : 'text-red-600')}>
                    {formatCurrency(portfolio.realized_pnl)}
                  </p>
                </div>
              </div>

              {/* Positions */}
              {Object.keys(portfolio.positions).length > 0 && (
                <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                  <p className="text-sm font-medium mb-2">Positions</p>
                  <div className="space-y-2">
                    {Object.entries(portfolio.positions).map(([symbol, qty]) => (
                      <div key={symbol} className="flex justify-between items-center">
                        <span className="font-mono">{symbol}</span>
                        <div className="text-right">
                          <span className="font-semibold">{qty.toFixed(6)}</span>
                          <span className="text-gray-500 text-sm ml-2">
                            @ {formatCurrency(portfolio.entry_prices[symbol] || 0)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Strategy Plan */}
          {plan && (
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-semibold mb-4">Current Strategy Plan</h2>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-500">Triggers</p>
                  <p className="font-semibold">{plan.trigger_count}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Max Trades/Day</p>
                  <p className="font-semibold">{plan.max_trades_per_day || 'N/A'}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Valid Until</p>
                  <p className="text-sm">{plan.valid_until ? formatDateTime(plan.valid_until) : 'N/A'}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Symbols</p>
                  <p className="font-mono text-sm">{plan.allowed_symbols.join(', ')}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Trade History */}
      {trades.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Recent Trades</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-2 px-3">Time</th>
                  <th className="text-left py-2 px-3">Symbol</th>
                  <th className="text-left py-2 px-3">Side</th>
                  <th className="text-right py-2 px-3">Qty</th>
                  <th className="text-right py-2 px-3">Price</th>
                  <th className="text-right py-2 px-3">P&L</th>
                </tr>
              </thead>
              <tbody>
                {trades.map((trade, idx) => (
                  <tr key={idx} className="border-b border-gray-100 dark:border-gray-700/50">
                    <td className="py-2 px-3 text-gray-500">{formatDateTime(trade.timestamp)}</td>
                    <td className="py-2 px-3 font-mono">{trade.symbol}</td>
                    <td className={cn('py-2 px-3 font-semibold', trade.side === 'buy' ? 'text-green-600' : 'text-red-600')}>
                      {trade.side.toUpperCase()}
                    </td>
                    <td className="py-2 px-3 text-right">{trade.qty.toFixed(6)}</td>
                    <td className="py-2 px-3 text-right">{formatCurrency(trade.price)}</td>
                    <td className={cn('py-2 px-3 text-right', (trade.pnl ?? 0) >= 0 ? 'text-green-600' : 'text-red-600')}>
                      {trade.pnl != null ? formatCurrency(trade.pnl) : '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Prompt Editor (collapsible) - Same prompts used for backtesting and paper trading */}
      <PromptEditor />

      {/* Event Timeline - Shows paper trading events */}
      {selectedSessionId ? (
        <EventTimeline limit={30} runId={selectedSessionId} />
      ) : (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 text-sm text-gray-500 dark:text-gray-400">
          Select a paper trading session to view its event timeline.
        </div>
      )}
    </div>
  );
}
