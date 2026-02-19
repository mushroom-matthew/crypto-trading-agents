import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  PlayCircle, StopCircle, Loader2, RefreshCw, Activity, Zap,
  TrendingUp, TrendingDown, Minus, ChevronDown, ChevronUp,
  ArrowUpRight, Ban, CheckCircle2, Radio, BarChart3,
} from 'lucide-react';
import { paperTradingAPI, promptsAPI, type PaperTradingSessionConfig } from '../lib/api';
import { cn, formatCurrency, formatDateTime } from '../lib/utils';
import { PromptEditor } from './PromptEditor';
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
    conflicting_signal_policy: 'reverse',
    min_price_move_pct: 0.5,
    walk_away_enabled: false,
    walk_away_profit_target_pct: 25.0,
    flatten_positions_daily: false,
  });
  const [planningSettings, setPlanningSettings] = useState<PlanningSettings>({
    max_trades_per_day: 10,
    max_triggers_per_symbol_per_day: 5,
    judge_check_after_trades: 3,
    replan_on_day_boundary: true,
    indicator_debug_mode: 'off',
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

  // Fetch current plan (refresh every 30s while running to catch replans)
  const { data: plan } = useQuery({
    queryKey: ['paper-trading-plan', selectedSessionId],
    queryFn: () => paperTradingAPI.getPlan(selectedSessionId!),
    enabled: !!selectedSessionId && session?.has_plan,
    refetchInterval: session?.status === 'running' ? 30000 : false,
  });

  // Fetch live activity feed
  const { data: activityData } = useQuery({
    queryKey: ['paper-trading-activity', selectedSessionId],
    queryFn: () => paperTradingAPI.getActivity(selectedSessionId!, 40),
    enabled: !!selectedSessionId,
    refetchInterval: session?.status === 'running' ? 4000 : 15000,
  });
  const activityEvents = activityData?.events ?? [];

  // Track expanded trigger list
  const [showTriggers, setShowTriggers] = useState(false);

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
        exit_binding_mode: 'category',
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
              showDayBoundaryReplan
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
          {/* Session Status + Live Ticker */}
          {session && (
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Zap className={cn('w-5 h-5', isRunning ? 'text-green-500 animate-pulse' : 'text-gray-400')} />
                Session
                <span className={cn(
                  'ml-auto text-xs font-mono px-2 py-0.5 rounded-full',
                  isRunning ? 'bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-400' : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400'
                )}>
                  {session.status.toUpperCase()} · cycle {session.cycle_count}
                </span>
              </h2>

              {/* Live Price Ticker */}
              {portfolio && Object.keys(portfolio.last_prices).length > 0 && (
                <div className="mb-4 flex flex-wrap gap-2">
                  {Object.entries(portfolio.last_prices).map(([symbol, price]) => (
                    <div key={symbol} className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-50 dark:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-600">
                      <BarChart3 className="w-3.5 h-3.5 text-gray-500" />
                      <span className="font-mono text-xs font-medium text-gray-600 dark:text-gray-400">{symbol.replace('-USD', '')}</span>
                      <span className="font-mono text-sm font-bold">{formatCurrency(price)}</span>
                    </div>
                  ))}
                </div>
              )}

              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <p className="text-gray-500">Symbols</p>
                  <p className="font-mono">{session.symbols.join(', ')}</p>
                </div>
                <div>
                  <p className="text-gray-500">Last Plan</p>
                  <p>{session.last_plan_time ? formatDateTime(session.last_plan_time) : 'Pending...'}</p>
                </div>
              </div>
            </div>
          )}

          {/* Portfolio */}
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
              {Object.keys(portfolio.positions).length > 0 && (
                <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                  <p className="text-sm font-medium mb-2">Open Positions</p>
                  <div className="space-y-2">
                    {Object.entries(portfolio.positions).map(([symbol, qty]) => {
                      const entryPx = portfolio.entry_prices[symbol] || 0;
                      const lastPx = portfolio.last_prices[symbol] || entryPx;
                      const pnlPct = entryPx > 0 ? ((lastPx - entryPx) / entryPx) * 100 : 0;
                      return (
                        <div key={symbol} className="flex justify-between items-center text-sm">
                          <span className="font-mono font-medium">{symbol}</span>
                          <div className="text-right">
                            <span className="font-semibold">{qty.toFixed(6)}</span>
                            <span className="text-gray-500 ml-2">@ {formatCurrency(entryPx)}</span>
                            <span className={cn('ml-2 font-medium', pnlPct >= 0 ? 'text-green-600' : 'text-red-600')}>
                              {pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(2)}%
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Active Strategy Plan */}
          {plan && (
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
              <div className="flex items-start justify-between mb-3">
                <h2 className="text-xl font-semibold">Strategy Plan</h2>
                <div className="flex gap-2">
                  {plan.regime && (
                    <span className={cn(
                      'text-xs font-medium px-2 py-1 rounded-full',
                      plan.regime === 'trend' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-400' :
                      plan.regime === 'range' ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/40 dark:text-yellow-400' :
                      plan.regime === 'volatile' ? 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-400' :
                      'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400'
                    )}>
                      {plan.regime.toUpperCase()}
                    </span>
                  )}
                  <span className="text-xs font-medium px-2 py-1 rounded-full bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-400">
                    {plan.trigger_count} triggers
                  </span>
                </div>
              </div>

              {/* LLM market assessment */}
              {plan.global_view && (
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-3 italic leading-relaxed">
                  "{plan.global_view}"
                </p>
              )}

              <div className="grid grid-cols-2 gap-3 text-sm mb-3">
                <div>
                  <p className="text-gray-500">Max Trades/Day</p>
                  <p className="font-semibold">{plan.max_trades_per_day ?? 'N/A'}</p>
                </div>
                <div>
                  <p className="text-gray-500">Valid Until</p>
                  <p>{plan.valid_until ? formatDateTime(plan.valid_until) : 'N/A'}</p>
                </div>
              </div>

              {/* Expandable trigger list */}
              {plan.triggers.length > 0 && (
                <div>
                  <button
                    onClick={() => setShowTriggers(!showTriggers)}
                    className="flex items-center gap-1 text-sm font-medium text-blue-600 dark:text-blue-400 hover:underline"
                  >
                    {showTriggers ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    {showTriggers ? 'Hide' : 'Show'} trigger conditions
                  </button>
                  {showTriggers && (
                    <div className="mt-3 space-y-2 max-h-64 overflow-y-auto">
                      {plan.triggers.map((t) => (
                        <div key={t.id} className="p-2 bg-gray-50 dark:bg-gray-700 rounded text-xs border-l-2 border-l-blue-400">
                          <div className="flex items-center gap-2 mb-1">
                            {t.direction === 'long' ? (
                              <TrendingUp className="w-3 h-3 text-green-500 flex-shrink-0" />
                            ) : t.direction === 'short' ? (
                              <TrendingDown className="w-3 h-3 text-red-500 flex-shrink-0" />
                            ) : (
                              <Minus className="w-3 h-3 text-gray-400 flex-shrink-0" />
                            )}
                            <span className="font-mono font-semibold">{t.id}</span>
                            <span className="text-gray-500">{t.symbol} · {t.timeframe}</span>
                            <span className={cn(
                              'ml-auto px-1.5 py-0.5 rounded text-xs font-medium',
                              t.category === 'emergency_exit' ? 'bg-red-100 text-red-700' :
                              t.category === 'risk_off' ? 'bg-orange-100 text-orange-700' :
                              t.category === 'volatility_breakout' ? 'bg-purple-100 text-purple-700' :
                              t.category === 'trend_continuation' ? 'bg-blue-100 text-blue-700' :
                              'bg-gray-100 text-gray-600'
                            )}>
                              {t.category}
                            </span>
                            {t.confidence && (
                              <span className="px-1.5 py-0.5 bg-yellow-100 text-yellow-700 rounded text-xs font-bold">
                                {t.confidence}
                              </span>
                            )}
                          </div>
                          {t.entry_rule && (
                            <p className="font-mono text-gray-600 dark:text-gray-300 pl-5 truncate" title={t.entry_rule}>
                              {t.entry_rule}
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Live Activity Feed */}
      {selectedSessionId && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg">
          <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
            <h2 className="text-xl font-semibold flex items-center gap-2">
              <Radio className={cn('w-4 h-4', isRunning ? 'text-green-500 animate-pulse' : 'text-gray-400')} />
              Live Activity
            </h2>
            <span className="text-xs text-gray-500">{isRunning ? 'Refreshing every 4s' : 'Session stopped'}</span>
          </div>
          <div className="max-h-80 overflow-y-auto divide-y divide-gray-100 dark:divide-gray-700">
            {activityEvents.length === 0 ? (
              <div className="p-6 text-center text-gray-500 text-sm">
                No activity yet — waiting for the first evaluation cycle.
              </div>
            ) : (
              activityEvents.map((ev) => <ActivityRow key={ev.event_id} ev={ev} />)
            )}
          </div>
        </div>
      )}

      {/* Trade History */}
      {trades.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Trade History</h2>
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

      {/* Prompt Editor */}
      <PromptEditor />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Activity Feed Row
// ─────────────────────────────────────────────────────────────────────────────

function ActivityRow({ ev }: { ev: { event_id: string; type: string; ts: string; payload: Record<string, any> } }) {
  const [expanded, setExpanded] = useState(false);

  const config = {
    tick: { icon: BarChart3, color: 'text-gray-400', bg: 'bg-gray-50 dark:bg-gray-700', label: 'Prices' },
    trigger_fired: { icon: ArrowUpRight, color: 'text-blue-500', bg: 'bg-blue-50 dark:bg-blue-900/20', label: 'Trigger Fired' },
    trade_blocked: { icon: Ban, color: 'text-orange-500', bg: 'bg-orange-50 dark:bg-orange-900/20', label: 'Blocked' },
    order_executed: { icon: CheckCircle2, color: 'text-green-500', bg: 'bg-green-50 dark:bg-green-900/20', label: 'Executed' },
    plan_generated: { icon: Zap, color: 'text-yellow-500', bg: 'bg-yellow-50 dark:bg-yellow-900/20', label: 'New Plan' },
    session_started: { icon: PlayCircle, color: 'text-green-500', bg: 'bg-green-50 dark:bg-green-900/20', label: 'Started' },
    session_stopped: { icon: StopCircle, color: 'text-red-500', bg: 'bg-red-50 dark:bg-red-900/20', label: 'Stopped' },
  } as Record<string, { icon: React.ComponentType<any>; color: string; bg: string; label: string }>;

  const cfg = config[ev.type] ?? { icon: Activity, color: 'text-gray-400', bg: 'bg-gray-50 dark:bg-gray-700', label: ev.type };
  const Icon = cfg.icon;

  function summary() {
    const p = ev.payload;
    switch (ev.type) {
      case 'tick': {
        const parts = Object.entries(p.prices ?? {})
          .map(([s, v]) => `${s.replace('-USD', '')} $${Number(v).toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`)
          .join(' · ');
        return parts || 'Prices updated';
      }
      case 'trigger_fired':
        return `${p.symbol} ${p.side?.toUpperCase() ?? ''} — ${p.trigger_id ?? p.category ?? ''} @ ${formatCurrency(p.price ?? 0)}`;
      case 'trade_blocked':
        return `${p.symbol} ${p.trigger_id ?? ''} — ${p.reason ?? 'blocked'}${p.detail ? ': ' + String(p.detail).substring(0, 60) : ''}`;
      case 'order_executed':
        return `${p.side?.toUpperCase()} ${Number(p.quantity ?? 0).toFixed(6)} ${p.symbol} @ ${formatCurrency(p.price ?? 0)}`;
      case 'plan_generated':
        return `${p.trigger_count ?? '?'} triggers · plan #${p.plan_index ?? '?'}`;
      default:
        return JSON.stringify(p).substring(0, 80);
    }
  }

  return (
    <div
      className={cn('px-4 py-2.5 hover:opacity-90 cursor-pointer transition-opacity', cfg.bg)}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-center gap-2.5">
        <Icon className={cn('w-4 h-4 flex-shrink-0', cfg.color)} />
        <span className={cn('text-xs font-semibold w-24 flex-shrink-0', cfg.color)}>{cfg.label}</span>
        <span className="text-xs text-gray-700 dark:text-gray-300 flex-1 truncate">{summary()}</span>
        <span className="text-xs text-gray-400 flex-shrink-0">
          {new Date(ev.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
        </span>
        {expanded ? <ChevronUp className="w-3 h-3 text-gray-400" /> : <ChevronDown className="w-3 h-3 text-gray-400" />}
      </div>
      {expanded && (
        <pre className="mt-2 ml-6 text-xs font-mono text-gray-600 dark:text-gray-400 overflow-x-auto whitespace-pre-wrap">
          {JSON.stringify(ev.payload, null, 2)}
        </pre>
      )}
    </div>
  );
}
