import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  ComposedChart, Bar, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Cell,
} from 'recharts';
import {
  PlayCircle, StopCircle, Loader2, RefreshCw, Activity, Zap,
  TrendingUp, TrendingDown, Minus, ChevronDown, ChevronUp,
  ArrowUpRight, Ban, CheckCircle2, Radio, BarChart3,
} from 'lucide-react';
import {
  paperTradingAPI,
  promptsAPI,
  screenerAPI,
  type PaperTradingSessionConfig,
  type CandleBar,
  type PaperTradingTradeSet,
  type ScreenerRecommendationItem,
} from '../lib/api';
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
  const [annotateScreenerShortlist, setAnnotateScreenerShortlist] = useState(false);

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

  // Screener preflight for session-start UX (grouped by hypothesis + timeframe)
  const screenerPreflightQuery = useQuery({
    queryKey: ['screener-session-preflight', 'paper', annotateScreenerShortlist],
    queryFn: () => screenerAPI.getSessionPreflight('paper', { annotate: annotateScreenerShortlist }),
    refetchInterval: 60000,
    retry: false,
  });
  const screenerPreflight = screenerPreflightQuery.data;

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
  const [expandedTriggers, setExpandedTriggers] = useState<Record<string, boolean>>({});

  // Fetch trades
  const { data: trades = [] } = useQuery({
    queryKey: ['paper-trading-trades', selectedSessionId],
    queryFn: () => paperTradingAPI.getTrades(selectedSessionId!, 50),
    enabled: !!selectedSessionId,
    refetchInterval: 15000,
  });

  // Fetch trade sets (paired round-trip trades)
  const { data: tradeSetsData } = useQuery({
    queryKey: ['paper-trading-trade-sets', selectedSessionId],
    queryFn: () => paperTradingAPI.getTradeSets(selectedSessionId!, 50),
    enabled: !!selectedSessionId,
    refetchInterval: 15000,
  });
  const tradeSets = tradeSetsData?.trade_sets ?? [];

  // Chart state
  const [chartSymbol, setChartSymbol] = useState('BTC-USD');
  const [chartTimeframe, setChartTimeframe] = useState('1m');
  const { data: candlesData } = useQuery({
    queryKey: ['paper-trading-candles', selectedSessionId, chartSymbol, chartTimeframe],
    queryFn: () => paperTradingAPI.getCandles(selectedSessionId!, chartSymbol, chartTimeframe, 120),
    enabled: !!selectedSessionId,
    refetchInterval: session?.status === 'running' ? 30000 : false,
    staleTime: 25000,
  });
  const { data: structureData } = useQuery({
    queryKey: ['paper-trading-structure', selectedSessionId],
    queryFn: () => paperTradingAPI.getStructure(selectedSessionId!),
    enabled: !!selectedSessionId,
    refetchInterval: session?.status === 'running' ? 30000 : false,
    staleTime: 25000,
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
  const isLiteralFalseRule = (rule?: string | null) => (rule || '').trim().toLowerCase() === 'false';
  const screenerErrorStatus = (screenerPreflightQuery.error as any)?.response?.status as number | undefined;
  const applyScreenerCandidateToForm = (item: ScreenerRecommendationItem) => {
    setSymbolsInput((prev) => mergePrimarySymbol(prev, item.symbol));
    const strategyId = mapScreenerHypothesisToStrategyId(item.template_id || item.hypothesis, strategies);
    if (strategyId) {
      setSelectedStrategyId(strategyId);
    }
    const interval = recommendedPlanIntervalHours(item.expected_hold_timeframe);
    if (interval) {
      setPlanIntervalHours(interval);
    }
  };
  const selectedPositionMeta = portfolio?.position_meta?.[chartSymbol];
  const selectedPositionQty = portfolio?.positions?.[chartSymbol] ?? 0;
  const selectedEntryPrice = portfolio?.entry_prices?.[chartSymbol] ?? null;
  const selectedStructure = structureData?.indicators?.[chartSymbol] ?? null;
  const selectedStructureRows = buildStructureRows(selectedStructure);
  const chartExecutions = activityEvents
    .filter((ev) => ev.type === 'order_executed' && ev.payload?.symbol === chartSymbol)
    .map((ev) => ({
      ts: ev.ts,
      side: String(ev.payload?.side ?? '').toLowerCase(),
      intent: String(ev.payload?.intent ?? ''),
      price: Number(ev.payload?.price ?? 0),
      triggerId: String(ev.payload?.trigger_id ?? ev.payload?.reason ?? ''),
    }))
    .sort((a, b) => new Date(a.ts).getTime() - new Date(b.ts).getTime());

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

            {/* Screener preflight shortlist (session-start UX) */}
            <div className="rounded-lg border border-sky-200 dark:border-sky-800 bg-sky-50/80 dark:bg-sky-900/20 p-4">
              <div className="flex items-start justify-between gap-3 mb-3">
                <div>
                  <label className="block text-sm font-medium text-sky-900 dark:text-sky-200">
                    Screener Preflight (Paper)
                  </label>
                  <p className="text-xs text-sky-700/90 dark:text-sky-300/80 mt-0.5">
                    Grouped shortlist by supported strategy hypotheses and expected hold timeframe.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => screenerPreflightQuery.refetch()}
                  disabled={screenerPreflightQuery.isFetching}
                  className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs border border-sky-300 dark:border-sky-700 hover:bg-sky-100/80 dark:hover:bg-sky-900/40 disabled:opacity-60"
                >
                  <RefreshCw className={cn('w-3.5 h-3.5', screenerPreflightQuery.isFetching && 'animate-spin')} />
                  Refresh
                </button>
              </div>

              <div className="mb-3 flex items-center gap-2">
                <input
                  id="annotateScreenerShortlist"
                  type="checkbox"
                  checked={annotateScreenerShortlist}
                  onChange={(e) => setAnnotateScreenerShortlist(e.target.checked)}
                  className="w-4 h-4 text-sky-600 rounded"
                />
                <label htmlFor="annotateScreenerShortlist" className="text-xs text-sky-800 dark:text-sky-200">
                  Optional LLM annotate / re-rank (deterministic fallback on failure)
                </label>
              </div>

              {screenerPreflightQuery.isLoading && (
                <div className="text-xs text-sky-800 dark:text-sky-200 flex items-center gap-2">
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  Loading screener preflight…
                </div>
              )}

              {!screenerPreflightQuery.isLoading && screenerErrorStatus === 404 && (
                <p className="text-xs text-sky-800 dark:text-sky-200">
                  No screener data yet. Start the screener workflow to populate grouped recommendations.
                </p>
              )}

              {!screenerPreflightQuery.isLoading && screenerPreflightQuery.error && screenerErrorStatus !== 404 && (
                <p className="text-xs text-red-600 dark:text-red-400">
                  Failed to load screener preflight: {(screenerPreflightQuery.error as Error).message}
                </p>
              )}

              {screenerPreflight && screenerPreflight.shortlist.groups.length > 0 && (
                <div className="space-y-3">
                  <div className="flex flex-wrap items-center gap-2 text-xs">
                    <span className="px-2 py-1 rounded-full bg-white/90 dark:bg-gray-800 border border-sky-200 dark:border-sky-700 font-mono">
                      {formatDateTime(screenerPreflight.as_of)}
                    </span>
                    <span className="px-2 py-1 rounded-full bg-white/90 dark:bg-gray-800 border border-sky-200 dark:border-sky-700">
                      {screenerPreflight.shortlist.groups.length} groups
                    </span>
                    <span className="px-2 py-1 rounded-full bg-white/90 dark:bg-gray-800 border border-sky-200 dark:border-sky-700">
                      up to {screenerPreflight.shortlist.max_per_group}/group
                    </span>
                    <span className={cn(
                      'px-2 py-1 rounded-full border text-[11px]',
                      screenerPreflight.shortlist.annotation_meta?.applied
                        ? 'bg-violet-100 text-violet-700 border-violet-200 dark:bg-violet-900/30 dark:text-violet-300 dark:border-violet-800'
                        : 'bg-white/90 dark:bg-gray-800 border-sky-200 dark:border-sky-700 text-gray-600 dark:text-gray-300'
                    )}>
                      {screenerPreflight.shortlist.annotation_meta?.applied ? 'LLM annotated' : 'Deterministic'}
                    </span>
                    {screenerPreflight.suggested_default_symbol && (
                      <span className="px-2 py-1 rounded-full bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800">
                        Suggested: {screenerPreflight.suggested_default_symbol}
                      </span>
                    )}
                  </div>

                  <div className="grid grid-cols-1 gap-3 max-h-96 overflow-y-auto pr-1">
                    {screenerPreflight.shortlist.groups.map((group) => (
                      <div
                        key={`${group.hypothesis}:${group.timeframe}`}
                        className="rounded-lg border border-sky-200/80 dark:border-sky-800 p-3 bg-white/70 dark:bg-gray-900/30"
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <div className="flex items-center flex-wrap gap-2">
                              <span className="text-sm font-semibold">{group.label}</span>
                              <span className="text-[11px] px-1.5 py-0.5 rounded bg-sky-100 text-sky-700 dark:bg-sky-900/40 dark:text-sky-300 font-mono">
                                {group.hypothesis}
                              </span>
                              <span className="text-[11px] px-1.5 py-0.5 rounded bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300 font-mono">
                                {group.timeframe}
                              </span>
                            </div>
                            <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                              {group.rationale}
                            </p>
                          </div>
                        </div>

                        <div className="mt-2 space-y-2">
                          {group.recommendations.map((item) => (
                            <div
                              key={`${group.hypothesis}:${group.timeframe}:${item.symbol}:${item.rank_global}`}
                              className="rounded-md border border-gray-200 dark:border-gray-700 p-2.5"
                            >
                              <div className="flex items-start justify-between gap-3">
                                <div className="min-w-0">
                                  <div className="flex items-center gap-2 flex-wrap">
                                    <span className="font-mono font-semibold">{item.symbol}</span>
                                    <span className={cn(
                                      'text-[11px] px-1.5 py-0.5 rounded font-semibold',
                                      item.confidence === 'high'
                                        ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300'
                                        : item.confidence === 'medium'
                                          ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300'
                                          : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300'
                                    )}>
                                      {item.confidence}
                                    </span>
                                    <span className="text-[11px] text-gray-500 font-mono">
                                      score {item.composite_score.toFixed(2)} · #{item.rank_global}
                                    </span>
                                  </div>
                                  <p className="text-xs text-gray-600 dark:text-gray-400 mt-1 line-clamp-2">
                                    {item.thesis}
                                  </p>
                                  <div className="mt-1 flex flex-wrap gap-2 text-[11px] text-gray-500">
                                    <span>Hold: <span className="font-mono">{item.expected_hold_timeframe}</span></span>
                                    {item.template_id && (
                                      <span>Template: <span className="font-mono">{item.template_id}</span></span>
                                    )}
                                    {item.key_levels?.support != null && item.key_levels?.resistance != null && (
                                      <span>
                                        Levels: <span className="font-mono">S {Number(item.key_levels.support).toFixed(2)} / R {Number(item.key_levels.resistance).toFixed(2)}</span>
                                      </span>
                                    )}
                                  </div>
                                </div>
                                <button
                                  type="button"
                                  onClick={() => applyScreenerCandidateToForm(item)}
                                  disabled={isRunning}
                                  className={cn(
                                    'shrink-0 inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium border',
                                    isRunning
                                      ? 'opacity-50 cursor-not-allowed border-gray-300'
                                      : 'border-sky-300 dark:border-sky-700 hover:bg-sky-100/80 dark:hover:bg-sky-900/40'
                                  )}
                                  title="Apply symbol/template/timeframe hint to the session form"
                                >
                                  <CheckCircle2 className="w-3.5 h-3.5" />
                                  Use
                                </button>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>

                  {screenerPreflight.notes?.length > 0 && (
                    <div className="pt-1 space-y-1">
                      {screenerPreflight.notes.slice(0, 2).map((note, idx) => (
                        <p key={idx} className="text-[11px] text-sky-800/90 dark:text-sky-200/90">
                          {note}
                        </p>
                      ))}
                    </div>
                  )}
                </div>
              )}
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
                Pre-allocate from initial cash to specific assets (notional USD). If cash is omitted, remaining cash is auto-computed.
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
              {Object.keys(portfolio.positions).length > 0 && (() => {
                const posEntries = Object.entries(portfolio.positions);
                let totalExposure = 0;
                let totalAtRisk = 0;
                let positionsWithoutStop = 0;

                // Pre-compute per-position values for aggregate panel
                const posData = posEntries.map(([symbol, qty]) => {
                  const entryPx = portfolio.entry_prices[symbol] || 0;
                  const lastPx = portfolio.last_prices[symbol] || entryPx;
                  const meta = portfolio.position_meta?.[symbol];
                  const stopPx = meta?.stop_price_abs ?? null;
                  const targetPx = meta?.target_price_abs ?? null;
                  const side = meta?.entry_side ?? 'long';
                  const category = meta?.entry_category ?? null;
                  const absQty = Math.abs(qty);
                  const notional = entryPx * absQty;
                  const pnlAbs = (lastPx - entryPx) * absQty * (side === 'short' ? -1 : 1);
                  const pnlPct = entryPx > 0 ? (pnlAbs / notional) * 100 : 0;
                  const riskAbs = stopPx !== null ? Math.abs(entryPx - stopPx) * absQty : null;
                  const stopDistPct = stopPx !== null && lastPx > 0 ? Math.abs(lastPx - stopPx) / lastPx * 100 : null;
                  const tgtDistPct = targetPx !== null && lastPx > 0 ? Math.abs(targetPx - lastPx) / lastPx * 100 : null;
                  const stopSpan = stopPx !== null ? Math.abs(entryPx - stopPx) : null;
                  const currentR = stopSpan && stopSpan > 0
                    ? (side === 'short' ? (entryPx - lastPx) : (lastPx - entryPx)) / stopSpan
                    : null;
                  const rrRatio = stopSpan && stopSpan > 0 && targetPx !== null
                    ? Math.abs(targetPx - entryPx) / stopSpan
                    : null;
                  const riskPct = riskAbs !== null && portfolio.total_equity > 0
                    ? riskAbs / portfolio.total_equity * 100
                    : null;

                  totalExposure += notional;
                  if (riskAbs !== null) totalAtRisk += riskAbs;
                  if (stopPx === null) positionsWithoutStop++;

                  return { symbol, qty: absQty, entryPx, lastPx, meta, stopPx, targetPx, side, category, notional, pnlAbs, pnlPct, riskAbs, riskPct, stopDistPct, tgtDistPct, currentR, rrRatio };
                });

                const exposurePct = portfolio.total_equity > 0 ? totalExposure / portfolio.total_equity * 100 : 0;
                const totalRiskPct = portfolio.total_equity > 0 ? totalAtRisk / portfolio.total_equity * 100 : 0;

                return (
                  <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                    <p className="text-sm font-medium mb-3">Open Positions</p>
                    <div className="space-y-3">
                      {posData.map(({ symbol, qty, entryPx, lastPx, stopPx, targetPx, side, category, notional, pnlAbs, pnlPct, riskAbs, riskPct, stopDistPct, tgtDistPct, currentR, rrRatio }) => (
                        <div key={symbol} className="text-sm rounded-md border border-gray-200 dark:border-gray-700 p-2.5 space-y-1.5">
                          {/* Header row: symbol + badges */}
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="font-mono font-semibold">{symbol}</span>
                            <span className={cn(
                              'text-xs font-bold px-1.5 py-0.5 rounded',
                              side === 'short'
                                ? 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-400'
                                : 'bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-400'
                            )}>
                              {side.toUpperCase()}
                            </span>
                            {category && (
                              <span className="text-xs px-1.5 py-0.5 rounded bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400">
                                {category}
                              </span>
                            )}
                          </div>
                          {/* Size + current price + P&L */}
                          <div className="flex justify-between items-baseline text-xs text-gray-600 dark:text-gray-400">
                            <span className="font-mono">
                              {qty.toFixed(6)} @ {formatCurrency(entryPx)}
                              <span className="ml-2 text-gray-400">Notional: {formatCurrency(notional)}</span>
                            </span>
                            <span className={cn('font-semibold', pnlAbs >= 0 ? 'text-green-600' : 'text-red-600')}>
                              {pnlAbs >= 0 ? '+' : ''}{formatCurrency(pnlAbs)} ({pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(2)}%)
                            </span>
                          </div>
                          <div className="text-xs text-gray-500 dark:text-gray-400">
                            Current: <span className="font-mono text-gray-700 dark:text-gray-300">{formatCurrency(lastPx)}</span>
                          </div>
                          {/* Stop row */}
                          {stopPx !== null ? (
                            <div className="flex items-center gap-2 text-xs text-red-600 dark:text-red-400 font-mono">
                              <span>↓ Stop</span>
                              <span className="font-semibold">{formatCurrency(stopPx)}</span>
                              {stopDistPct !== null && (
                                <span className="text-gray-500">({stopDistPct.toFixed(1)}% away)</span>
                              )}
                              {riskAbs !== null && (
                                <span className="ml-auto text-gray-600 dark:text-gray-400">
                                  At risk: {formatCurrency(riskAbs)}
                                  {riskPct !== null && <span className="ml-1">({riskPct.toFixed(2)}% equity)</span>}
                                </span>
                              )}
                            </div>
                          ) : (
                            <div className="text-xs text-amber-600 dark:text-amber-400 font-medium">
                              ── No stop set ──
                            </div>
                          )}
                          {/* Target row */}
                          {targetPx !== null && (
                            <div className="flex items-center gap-2 text-xs text-green-600 dark:text-green-400 font-mono">
                              <span>↑ Target</span>
                              <span className="font-semibold">{formatCurrency(targetPx)}</span>
                              {tgtDistPct !== null && (
                                <span className="text-gray-500">({tgtDistPct.toFixed(1)}% away)</span>
                              )}
                              {rrRatio !== null && (
                                <span className="ml-1 text-gray-600 dark:text-gray-400">R:R 1:{rrRatio.toFixed(1)}</span>
                              )}
                              {currentR !== null && (
                                <span className={cn(
                                  'ml-auto font-semibold',
                                  currentR >= 0 ? 'text-green-600' : 'text-red-500'
                                )}>
                                  {currentR >= 0 ? '+' : ''}{currentR.toFixed(2)}r
                                </span>
                              )}
                            </div>
                          )}
                          {/* Current R when target not set but stop is */}
                          {targetPx === null && currentR !== null && (
                            <div className="text-xs text-gray-500 dark:text-gray-400">
                              Current R: <span className={cn('font-semibold', currentR >= 0 ? 'text-green-600' : 'text-red-500')}>
                                {currentR >= 0 ? '+' : ''}{currentR.toFixed(2)}r
                              </span>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                    {/* Risk Assessment summary panel */}
                    <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                      <p className="text-xs font-semibold text-gray-600 dark:text-gray-400 uppercase tracking-wide mb-2">Risk Assessment</p>
                      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                        <div className="flex justify-between">
                          <span className="text-gray-500">Positions</span>
                          <span className="font-medium">{posEntries.length}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-500">Exposure</span>
                          <span className="font-mono font-medium">
                            {formatCurrency(totalExposure)} <span className="text-gray-400">({exposurePct.toFixed(1)}%)</span>
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-500">At risk</span>
                          <span className="font-mono font-medium text-red-600 dark:text-red-400">
                            {formatCurrency(totalAtRisk)}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-500">Risk/equity</span>
                          <span className="font-mono font-medium">{totalRiskPct.toFixed(2)}%</span>
                        </div>
                      </div>
                      {positionsWithoutStop > 0 && (
                        <div className="mt-2 text-xs text-amber-600 dark:text-amber-400 font-medium">
                          ⚠ {positionsWithoutStop} position{positionsWithoutStop > 1 ? 's' : ''} {positionsWithoutStop > 1 ? 'have' : 'has'} no stop set
                        </div>
                      )}
                    </div>
                  </div>
                );
              })()}
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
                    <div className="mt-3 space-y-2 max-h-96 overflow-y-auto">
                      {plan.triggers.map((t) => (
                        <div key={t.id} className="p-2 bg-gray-50 dark:bg-gray-700 rounded text-xs border-l-2 border-l-blue-400">
                          <button
                            onClick={() => setExpandedTriggers((prev) => ({ ...prev, [t.id]: !prev[t.id] }))}
                            className="w-full flex items-center gap-2 text-left"
                          >
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
                            {expandedTriggers[t.id] ? (
                              <ChevronUp className="w-4 h-4 text-gray-500" />
                            ) : (
                              <ChevronDown className="w-4 h-4 text-gray-500" />
                            )}
                          </button>

                          {expandedTriggers[t.id] && (
                            <div className="mt-3 pl-5 space-y-2">
                              {(t.entry_rule && !isLiteralFalseRule(t.entry_rule)) && (
                                <div>
                                  <p className="text-[11px] font-semibold text-gray-500">ENTRY</p>
                                  <pre className="font-mono whitespace-pre-wrap break-words text-gray-700 dark:text-gray-200">{t.entry_rule}</pre>
                                </div>
                              )}
                              {(t.exit_rule && !isLiteralFalseRule(t.exit_rule)) && (
                                <div>
                                  <p className="text-[11px] font-semibold text-gray-500">EXIT</p>
                                  <pre className="font-mono whitespace-pre-wrap break-words text-gray-700 dark:text-gray-200">{t.exit_rule}</pre>
                                </div>
                              )}
                              {(t.hold_rule && !isLiteralFalseRule(t.hold_rule)) && (
                                <div>
                                  <p className="text-[11px] font-semibold text-gray-500">HOLD</p>
                                  <pre className="font-mono whitespace-pre-wrap break-words text-gray-700 dark:text-gray-200">{t.hold_rule}</pre>
                                </div>
                              )}
                              {(!t.entry_rule || isLiteralFalseRule(t.entry_rule)) &&
                                (!t.exit_rule || isLiteralFalseRule(t.exit_rule)) &&
                                (!t.hold_rule || isLiteralFalseRule(t.hold_rule)) && (
                                  <p className="text-xs text-gray-500 italic">
                                    No active conditions (rules are empty/disabled).
                                  </p>
                                )}
                            </div>
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

      {/* Live Candlestick Chart */}
      {selectedSessionId && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-blue-500" />
              Live Chart
            </h2>
            <div className="flex gap-2">
              {/* Instrument toggle */}
              {(() => {
                const symbols = session?.symbols ?? ['BTC-USD', 'ETH-USD'];
                return symbols.map((s: string) => (
                  <button
                    key={s}
                    onClick={() => setChartSymbol(s)}
                    className={cn(
                      'px-3 py-1 text-xs font-semibold rounded-full transition-colors',
                      chartSymbol === s
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300',
                    )}
                  >
                    {s.replace('-USD', '')}
                  </button>
                ));
              })()}
              {/* Timeframe selector */}
              <div className="flex gap-1 ml-2">
                {['1m', '5m', '15m', '1h', '4h', '1d', '1w', '1M'].map((tf) => (
                  <button
                    key={tf}
                    onClick={() => setChartTimeframe(tf)}
                    className={cn(
                      'px-2 py-1 text-xs font-mono rounded transition-colors',
                      chartTimeframe === tf
                        ? 'bg-gray-800 text-white dark:bg-gray-200 dark:text-gray-900'
                        : 'bg-gray-100 text-gray-500 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-400',
                    )}
                  >
                    {tf}
                  </button>
                ))}
              </div>
            </div>
          </div>
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
            <div className="xl:col-span-2">
              <CandlestickChart
                candles={candlesData?.candles ?? []}
                trades={trades.filter((t) => t.symbol === chartSymbol)}
                executions={chartExecutions}
                timeframe={chartTimeframe}
                stopPrice={selectedPositionMeta?.stop_price_abs ?? null}
                targetPrice={selectedPositionMeta?.target_price_abs ?? null}
                openPosition={selectedPositionQty !== 0 ? {
                  openedAt: selectedPositionMeta?.opened_at ?? null,
                  entrySide: selectedPositionMeta?.entry_side ?? null,
                  entryPrice: selectedEntryPrice,
                } : null}
              />
            </div>
            <div className="space-y-3">
              <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-3">
                <p className="text-xs uppercase tracking-wide font-semibold text-gray-500 mb-2">
                  Open Position Context
                </p>
                {selectedPositionQty === 0 ? (
                  <p className="text-xs text-gray-500">No open position on {chartSymbol.replace('-USD', '')}.</p>
                ) : (
                  <div className="space-y-1.5 text-xs">
                    <div className="flex justify-between gap-3">
                      <span className="text-gray-500">Direction</span>
                      <span className={cn(
                        'font-semibold',
                        (selectedPositionMeta?.entry_side ?? 'long') === 'short' ? 'text-red-600' : 'text-green-600'
                      )}>
                        {(selectedPositionMeta?.entry_side ?? 'long').toUpperCase()}
                      </span>
                    </div>
                    <div className="flex justify-between gap-3">
                      <span className="text-gray-500">Entry Time</span>
                      <span className="font-mono text-right">
                        {selectedPositionMeta?.opened_at ? formatDateTime(selectedPositionMeta.opened_at) : 'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between gap-3">
                      <span className="text-gray-500">Entry Price</span>
                      <span className="font-mono">{selectedEntryPrice != null ? formatCurrency(selectedEntryPrice) : 'N/A'}</span>
                    </div>
                    <div className="flex justify-between gap-3">
                      <span className="text-gray-500">Stop</span>
                      <span className={cn(
                        'font-mono',
                        selectedPositionMeta?.stop_price_abs == null && 'text-amber-600 dark:text-amber-400'
                      )}>
                        {selectedPositionMeta?.stop_price_abs != null
                          ? formatCurrency(selectedPositionMeta.stop_price_abs)
                          : 'Not set'}
                      </span>
                    </div>
                    <div className="flex justify-between gap-3">
                      <span className="text-gray-500">Target</span>
                      <span className={cn(
                        'font-mono',
                        selectedPositionMeta?.target_price_abs == null && 'text-gray-400'
                      )}>
                        {selectedPositionMeta?.target_price_abs != null
                          ? formatCurrency(selectedPositionMeta.target_price_abs)
                          : 'Not set'}
                      </span>
                    </div>
                  </div>
                )}
              </div>

              <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-3">
                <p className="text-xs uppercase tracking-wide font-semibold text-gray-500 mb-2">
                  Structure Snapshot (HTF)
                </p>
                {selectedStructureRows.length === 0 ? (
                  <p className="text-xs text-gray-500">
                    No structure snapshot yet. It appears after the first plan cycle.
                  </p>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <tbody>
                        {selectedStructureRows.map((row) => (
                          <tr key={row.key} className="border-b border-gray-100 dark:border-gray-700 last:border-0">
                            <td className="py-1.5 text-gray-500 pr-2">{row.label}</td>
                            <td className={cn('py-1.5 text-right font-mono', row.valueClassName)}>{row.value}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

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
              collapseActivityEvents(activityEvents.filter((ev) => ev.type !== 'tick'))
                .map((ev) => <ActivityRow key={ev.event_id} ev={ev} />)
            )}
          </div>
        </div>
      )}

      {/* Trade Sets — paired round-trip entry/exit view */}
      {tradeSets.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold">Trade Sets</h2>
            {tradeSetsData && (
              <div className="flex gap-4 text-sm">
                <span className="text-gray-500">
                  <span className="font-semibold text-gray-800 dark:text-gray-200">{tradeSetsData.total_completed_trades}</span> trades
                </span>
                <span className="text-gray-500">
                  Win rate: <span className={cn('font-semibold', tradeSetsData.win_rate_pct >= 50 ? 'text-green-600' : 'text-red-600')}>
                    {tradeSetsData.win_rate_pct.toFixed(1)}%
                  </span>
                </span>
                <span className="text-gray-500">
                  Net P&L: <span className={cn('font-semibold', tradeSetsData.total_net_pnl >= 0 ? 'text-green-600' : 'text-red-600')}>
                    {formatCurrency(tradeSetsData.total_net_pnl)}
                  </span>
                </span>
              </div>
            )}
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700 text-xs text-gray-500">
                  <th className="text-left py-2 px-2">Symbol</th>
                  <th className="text-left py-2 px-2">Entry</th>
                  <th className="text-left py-2 px-2">Exit</th>
                  <th className="text-right py-2 px-2">Hold</th>
                  <th className="text-right py-2 px-2">Entry Px</th>
                  <th className="text-right py-2 px-2">Exit Px</th>
                  <th className="text-right py-2 px-2">Qty</th>
                  <th className="text-right py-2 px-2">Fee</th>
                  <th className="text-right py-2 px-2">Net P&L</th>
                  <th className="text-right py-2 px-2">%</th>
                  <th className="text-left py-2 px-2">Setup</th>
                </tr>
              </thead>
              <tbody>
                {tradeSets.map((ts: PaperTradingTradeSet, idx: number) => {
                  const holdStr = ts.hold_minutes >= 60
                    ? `${(ts.hold_minutes / 60).toFixed(1)}h`
                    : `${ts.hold_minutes.toFixed(0)}m`;
                  return (
                    <tr key={idx} className={cn(
                      'border-b border-gray-100 dark:border-gray-700/50 text-xs',
                      ts.winner ? 'bg-green-50/30 dark:bg-green-900/10' : 'bg-red-50/30 dark:bg-red-900/10'
                    )}>
                      <td className="py-2 px-2 font-mono font-medium">{ts.symbol.replace('-USD', '')}</td>
                      <td className="py-2 px-2 text-gray-500">{formatDateTime(ts.entry_time)}</td>
                      <td className="py-2 px-2 text-gray-500">{formatDateTime(ts.exit_time)}</td>
                      <td className="py-2 px-2 text-right font-mono">{holdStr}</td>
                      <td className="py-2 px-2 text-right font-mono">{formatCurrency(ts.entry_price)}</td>
                      <td className="py-2 px-2 text-right font-mono">{formatCurrency(ts.exit_price)}</td>
                      <td className="py-2 px-2 text-right font-mono">{ts.qty.toFixed(4)}</td>
                      <td className="py-2 px-2 text-right text-gray-500">{formatCurrency(ts.fee)}</td>
                      <td className={cn('py-2 px-2 text-right font-semibold', ts.winner ? 'text-green-600' : 'text-red-600')}>
                        {formatCurrency(ts.net_pnl)}
                      </td>
                      <td className={cn('py-2 px-2 text-right font-semibold', ts.winner ? 'text-green-600' : 'text-red-600')}>
                        {ts.pnl_pct >= 0 ? '+' : ''}{ts.pnl_pct.toFixed(2)}%
                      </td>
                      <td className="py-2 px-2 text-gray-500 font-mono truncate max-w-32" title={ts.entry_trigger ?? ''}>
                        {ts.category ?? ts.entry_trigger ?? '-'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
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
                  <th className="text-right py-2 px-3">Fee</th>
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
                    <td className="py-2 px-3 text-right text-gray-500">
                      {trade.fee != null ? formatCurrency(trade.fee) : '-'}
                    </td>
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
// Activity helpers
// ─────────────────────────────────────────────────────────────────────────────

type ActivityEvent = { event_id: string; type: string; ts: string; payload: Record<string, any> };

/** Collapse consecutive trade_blocked events with same trigger_id+reason into one row with a count.
 *  Also dedupe consecutive eval_summary events per symbol into the latest one. */
function collapseActivityEvents(events: ActivityEvent[]): (ActivityEvent & { count?: number })[] {
  const out: (ActivityEvent & { count?: number })[] = [];
  for (const ev of events) {
    const last = out[out.length - 1];
    if (
      ev.type === 'trade_blocked' &&
      last?.type === 'trade_blocked' &&
      last.payload.trigger_id === ev.payload.trigger_id &&
      last.payload.reason === ev.payload.reason
    ) {
      last.count = (last.count ?? 1) + 1;
    } else if (
      ev.type === 'eval_summary' &&
      last?.type === 'eval_summary' &&
      last.payload.symbol === ev.payload.symbol
    ) {
      // Keep only the latest eval_summary per symbol — replace in place
      out[out.length - 1] = { ...ev, count: undefined };
    } else {
      out.push({ ...ev, count: undefined });
    }
  }
  return out;
}

type StructureRow = {
  key: string;
  label: string;
  value: string;
  valueClassName?: string;
};

function buildStructureRows(snapshot: Record<string, any> | null): StructureRow[] {
  if (!snapshot) return [];

  const timeframe = typeof snapshot.timeframe === 'string' ? snapshot.timeframe : null;
  const asOf = typeof snapshot.as_of === 'string' ? snapshot.as_of : null;
  const close = toNumber(snapshot.close);
  const htfDailyLow = toNumber(snapshot.htf_daily_low);
  const htfDailyHigh = toNumber(snapshot.htf_daily_high);
  const htf5dLow = toNumber(snapshot.htf_5d_low);
  const htf5dHigh = toNumber(snapshot.htf_5d_high);
  const dailyAtr = toNumber(snapshot.htf_daily_atr);
  const priceVsMid = toNumber(snapshot.htf_price_vs_daily_mid);
  const rsi = toNumber(snapshot.rsi_14);
  const trend = deriveTrendLabel(snapshot);
  const daysRangePct = toNumber(snapshot.htf_daily_range_pct);

  const rows: StructureRow[] = [
    { key: 'tf', label: 'Snapshot TF', value: timeframe ?? 'N/A' },
    { key: 'as_of', label: 'As Of', value: asOf ? formatDateTime(asOf) : 'N/A' },
    { key: 'trend', label: 'Trend State', value: trend.label, valueClassName: trend.className },
    { key: 'close', label: 'Close', value: close != null ? formatCurrency(close) : 'N/A' },
    { key: 'daily_low', label: 'Daily Low', value: htfDailyLow != null ? formatCurrency(htfDailyLow) : 'N/A' },
    { key: 'daily_high', label: 'Daily High', value: htfDailyHigh != null ? formatCurrency(htfDailyHigh) : 'N/A' },
    { key: 'week_low', label: '5D Low', value: htf5dLow != null ? formatCurrency(htf5dLow) : 'N/A' },
    { key: 'week_high', label: '5D High', value: htf5dHigh != null ? formatCurrency(htf5dHigh) : 'N/A' },
    { key: 'daily_atr', label: 'Daily ATR', value: dailyAtr != null ? formatNumber(dailyAtr, 2) : 'N/A' },
    { key: 'daily_range', label: 'Daily Range %', value: daysRangePct != null ? `${daysRangePct.toFixed(2)}%` : 'N/A' },
    {
      key: 'vs_mid',
      label: 'Price vs Daily Mid (ATR)',
      value: priceVsMid != null ? formatSigned(priceVsMid, 2) : 'N/A',
      valueClassName: priceVsMid == null ? '' : (priceVsMid >= 0 ? 'text-green-600' : 'text-red-600'),
    },
    {
      key: 'rsi',
      label: 'RSI 14',
      value: rsi != null ? rsi.toFixed(1) : 'N/A',
      valueClassName: rsi == null ? '' : (rsi >= 70 ? 'text-red-600' : (rsi <= 30 ? 'text-green-600' : '')),
    },
  ];

  return rows;
}

function toNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

function formatNumber(value: number, digits = 2): string {
  return value.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function formatSigned(value: number, digits = 2): string {
  return `${value >= 0 ? '+' : ''}${formatNumber(value, digits)}`;
}

function deriveTrendLabel(snapshot: Record<string, any>): { label: string; className: string } {
  const trendState = typeof snapshot.trend_state === 'string' ? snapshot.trend_state.toLowerCase() : '';
  if (trendState === 'uptrend') return { label: 'UPTREND', className: 'text-green-600' };
  if (trendState === 'downtrend') return { label: 'DOWNTREND', className: 'text-red-600' };
  if (trendState === 'sideways') return { label: 'SIDEWAYS', className: 'text-gray-500' };

  const smaShort = toNumber(snapshot.sma_short);
  const smaMedium = toNumber(snapshot.sma_medium);
  const smaLong = toNumber(snapshot.sma_long);
  if (smaShort != null && smaMedium != null && smaLong != null) {
    if (smaShort > smaMedium && smaMedium > smaLong) return { label: 'UPTREND*', className: 'text-green-600' };
    if (smaShort < smaMedium && smaMedium < smaLong) return { label: 'DOWNTREND*', className: 'text-red-600' };
  }
  return { label: 'SIDEWAYS', className: 'text-gray-500' };
}

// ─────────────────────────────────────────────────────────────────────────────
// Candlestick Chart
// ─────────────────────────────────────────────────────────────────────────────

interface CandlestickChartProps {
  candles: CandleBar[];
  trades: { timestamp: string; side: string; price: number }[];
  executions?: { ts: string; side: string; intent?: string; price: number; triggerId?: string }[];
  timeframe: string;
  stopPrice: number | null;
  targetPrice: number | null;
  openPosition?: {
    openedAt: string | null;
    entrySide: string | null;
    entryPrice: number | null;
  } | null;
}

function inferPriceDecimals(min: number, max: number): number {
  const span = Math.abs(max - min);
  if (!Number.isFinite(span) || span <= 0) return 2;
  const rawStep = span / 6;
  const magnitude = Math.floor(Math.log10(Math.max(rawStep, 1e-12)));
  let decimals = Math.max(0, Math.min(8, -magnitude + 1));
  const maxAbs = Math.max(Math.abs(min), Math.abs(max));
  if (maxAbs < 1) decimals = Math.max(decimals, 3);
  if (maxAbs < 0.1) decimals = Math.max(decimals, 4);
  return decimals;
}

function formatPriceWithDecimals(value: number, decimals: number): string {
  if (!Number.isFinite(value)) return 'N/A';
  return value.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function formatTimeAxis(value: number, timeframe: string): string {
  const dt = new Date(value);
  if (timeframe === '1d' || timeframe === '1w' || timeframe === '1M') {
    return dt.toLocaleDateString([], { month: 'short', day: 'numeric' });
  }
  if (timeframe === '1h' || timeframe === '4h') {
    return dt.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit' });
  }
  return dt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function mergePrimarySymbol(current: string, symbol: string): string {
  const incoming = symbol.trim().toUpperCase();
  const existing = current
    .split(',')
    .map((s) => s.trim().toUpperCase())
    .filter(Boolean)
    .filter((s) => s !== incoming);
  return [incoming, ...existing].join(', ');
}

function mapScreenerHypothesisToStrategyId(
  hypothesisOrTemplate: string | null | undefined,
  strategies: Array<{ id: string }>
): string | null {
  const id = (hypothesisOrTemplate || '').trim();
  if (!id) return null;
  const available = new Set(strategies.map((s) => s.id));
  if (available.has(id)) return id;

  const aliases: Record<string, string[]> = {
    compression_breakout: ['compression_breakout', 'volatility_breakout'],
    volatile_breakout: ['volatility_breakout', 'momentum_trend_following'],
    bull_trending: ['momentum_trend_following', 'aggressive_active', 'balanced_hybrid'],
    bear_defensive: ['conservative_defensive', 'balanced_hybrid'],
    range_mean_revert: ['mean_reversion', 'balanced_hybrid'],
    uncertain_wait: ['conservative_defensive', 'balanced_hybrid'],
  };
  for (const candidate of aliases[id] || []) {
    if (available.has(candidate)) return candidate;
  }
  return null;
}

function recommendedPlanIntervalHours(timeframe: string): number | null {
  const tf = timeframe.trim().toLowerCase();
  if (tf === '15m') return 1;
  if (tf === '1h') return 4;
  if (tf === '4h') return 8;
  return null;
}

function CandlestickChart({ candles, trades, executions = [], timeframe, stopPrice, targetPrice, openPosition }: CandlestickChartProps) {
  if (candles.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-400 text-sm">
        Loading candles…
      </div>
    );
  }

  // Build chart data with OHLCV + computed bar range for Recharts Bar
  const data = candles.map((c) => {
    const isUp = c.close >= c.open;
    const bodyLow = Math.min(c.open, c.close);
    const bodyHigh = Math.max(c.open, c.close);
    return {
      ...c,
      isUp,
      bodyRange: [bodyLow, bodyHigh] as [number, number],
      wickRange: [c.low, c.high] as [number, number],
    };
  });

  const firstCandleMs = data[0]?.time ?? 0;
  const lastCandleMs = data[data.length - 1]?.time ?? 0;

  // Build execution markers for trades in visible range.
  const markerInputs = executions.length > 0
    ? executions.map((e) => ({ ts: e.ts, side: e.side, intent: (e.intent || '').toLowerCase() }))
    : trades.map((t) => ({ ts: t.timestamp, side: (t.side || '').toLowerCase(), intent: '' }));
  const executionMarkers: { time: number; side: string; intent: string }[] = [];
  for (const m of markerInputs) {
    const markMs = new Date(m.ts).getTime();
    if (!Number.isFinite(markMs) || markMs < firstCandleMs || markMs > lastCandleMs) {
      continue;
    }
    let closestIdx = 0;
    let minDiff = Infinity;
    data.forEach((d, i) => {
      const diff = Math.abs(d.time - markMs);
      if (diff < minDiff) { minDiff = diff; closestIdx = i; }
    });
    executionMarkers.push({
      time: data[closestIdx]?.time,
      side: m.side,
      intent: m.intent,
    });
  }

  let openEntryMarkerTime: number | null = null;
  if (openPosition?.openedAt) {
    const openedMs = new Date(openPosition.openedAt).getTime();
    if (Number.isFinite(openedMs) && openedMs >= firstCandleMs && openedMs <= lastCandleMs) {
      let closestIdx = 0;
      let minDiff = Infinity;
      data.forEach((d, i) => {
        const diff = Math.abs(d.time - openedMs);
        if (diff < minDiff) { minDiff = diff; closestIdx = i; }
      });
      openEntryMarkerTime = data[closestIdx]?.time ?? null;
    }
  }

  const levelValues = [
    openPosition?.entryPrice,
    stopPrice,
    targetPrice,
  ].filter((v): v is number => v != null && Number.isFinite(v));
  const rawMin = Math.min(...candles.map((c) => c.low), ...(levelValues.length ? levelValues : [Infinity]));
  const rawMax = Math.max(...candles.map((c) => c.high), ...(levelValues.length ? levelValues : [-Infinity]));
  const baseSpan = Math.max(rawMax - rawMin, Math.abs(rawMax) * 0.002, 1e-9);
  const yPad = baseSpan * 0.03;
  const yMin = rawMin >= 0 ? Math.max(0, rawMin - yPad) : rawMin - yPad;
  const yMax = rawMax + yPad;
  const axisDecimals = inferPriceDecimals(yMin, yMax);

  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload?.length) return null;
    const d = payload[0]?.payload;
    if (!d) return null;
    return (
      <div className="bg-gray-900 text-white text-xs rounded px-3 py-2 shadow-lg">
        <p className="font-semibold mb-1">
          {new Date(d.time).toLocaleString([], {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
          })}
        </p>
        <p>O: {formatPriceWithDecimals(Number(d.open), axisDecimals)}  H: {formatPriceWithDecimals(Number(d.high), axisDecimals)}</p>
        <p>L: {formatPriceWithDecimals(Number(d.low), axisDecimals)}  C: {formatPriceWithDecimals(Number(d.close), axisDecimals)}</p>
        <p className="text-gray-400">Vol: {d.volume?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</p>
      </div>
    );
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-4 text-[11px] text-gray-500">
        <span className="inline-flex items-center gap-1">
          <span className="w-3 h-0.5 bg-cyan-500 inline-block" />
          Entry marker
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="w-3 h-0.5 bg-amber-500 inline-block" />
          Exit marker
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="w-3 h-0.5 bg-blue-500 inline-block" />
          Entry price
        </span>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" strokeOpacity={0.4} />
        <XAxis
          dataKey="time"
          type="number"
          domain={['dataMin', 'dataMax']}
          tick={{ fontSize: 10, fill: '#9ca3af' }}
          interval={Math.max(0, Math.floor(data.length / 8))}
          tickFormatter={(value) => formatTimeAxis(value, timeframe)}
          tickLine={false}
        />
        <YAxis
          domain={[yMin, yMax]}
          tick={{ fontSize: 10, fill: '#9ca3af' }}
          tickFormatter={(v) => formatPriceWithDecimals(v, axisDecimals)}
          width={80}
          tickLine={false}
        />
        <Tooltip content={<CustomTooltip />} />

        {/* Wick line (high-low) */}
        <Bar dataKey="wickRange" fill="transparent" stroke="transparent" barSize={1}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.isUp ? '#16a34a' : '#dc2626'} />
          ))}
        </Bar>

        {/* Candle body (open-close) */}
        <Bar dataKey="bodyRange" barSize={6}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.isUp ? '#16a34a' : '#dc2626'} fillOpacity={0.85} />
          ))}
        </Bar>

        {/* Close price line overlay */}
        <Line
          type="monotone"
          dataKey="close"
          stroke="#3b82f6"
          strokeWidth={1}
          dot={false}
          strokeOpacity={0.4}
        />

          {/* Entry/exit execution markers */}
          {executionMarkers.map((m, i) => {
            const isEntry = m.intent === 'entry';
            const color = isEntry ? '#06b6d4' : (m.intent === 'exit' ? '#f59e0b' : (m.side === 'buy' ? '#16a34a' : '#dc2626'));
            const labelText = isEntry
              ? (m.side === 'buy' ? 'ENTRY ▲' : 'ENTRY ▼')
              : (m.intent === 'exit'
                ? (m.side === 'sell' ? 'EXIT ▼' : 'EXIT ▲')
                : (m.side === 'buy' ? '▲' : '▼'));
            return (
              <ReferenceLine
                key={`exec-${i}`}
                x={m.time}
                stroke={color}
                strokeWidth={1.5}
                strokeDasharray="4 2"
                label={{
                  value: labelText,
                  fill: color,
                  fontSize: 10,
                }}
              />
            );
          })}

        {/* Open position entry marker (if visible in current candle window) */}
        {openEntryMarkerTime != null && (
          <ReferenceLine
            x={openEntryMarkerTime}
            stroke={openPosition?.entrySide === 'short' ? '#dc2626' : '#16a34a'}
            strokeWidth={1.5}
            strokeDasharray="2 2"
            label={{
              value: openPosition?.entrySide === 'short' ? 'entry ▼' : 'entry ▲',
              fill: openPosition?.entrySide === 'short' ? '#dc2626' : '#16a34a',
              fontSize: 10,
            }}
          />
        )}

        {/* Entry price reference line for active position */}
        {openPosition?.entryPrice != null && (
          <ReferenceLine
            y={openPosition.entryPrice}
            stroke="#3b82f6"
            strokeWidth={1}
            strokeDasharray="4 3"
            label={{ value: `entry ${formatPriceWithDecimals(openPosition.entryPrice, axisDecimals)}`, fill: '#3b82f6', fontSize: 10, position: 'right' }}
          />
        )}

        {/* Stop loss line */}
        {stopPrice != null && (
          <ReferenceLine
            y={stopPrice}
            stroke="#ef4444"
            strokeWidth={1.5}
            strokeDasharray="6 3"
            label={{ value: `stop ${formatPriceWithDecimals(stopPrice, axisDecimals)}`, fill: '#ef4444', fontSize: 10, position: 'right' }}
          />
        )}

        {/* Take profit target line */}
        {targetPrice != null && (
          <ReferenceLine
            y={targetPrice}
            stroke="#22c55e"
            strokeWidth={1.5}
            strokeDasharray="6 3"
            label={{ value: `target ${formatPriceWithDecimals(targetPrice, axisDecimals)}`, fill: '#22c55e', fontSize: 10, position: 'right' }}
          />
        )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Activity Feed Row
// ─────────────────────────────────────────────────────────────────────────────

function ActivityRow({ ev }: { ev: ActivityEvent & { count?: number } }) {
  const [expanded, setExpanded] = useState(false);

  const config = {
    tick: { icon: BarChart3, color: 'text-gray-400', bg: 'bg-gray-50 dark:bg-gray-700', label: 'Prices' },
    trigger_fired: { icon: ArrowUpRight, color: 'text-blue-500', bg: 'bg-blue-50 dark:bg-blue-900/20', label: 'Trigger Fired' },
    trade_blocked: { icon: Ban, color: 'text-orange-500', bg: 'bg-orange-50 dark:bg-orange-900/20', label: 'Blocked' },
    order_executed: { icon: CheckCircle2, color: 'text-green-500', bg: 'bg-green-50 dark:bg-green-900/20', label: 'Executed' },
    plan_generated: { icon: Zap, color: 'text-yellow-500', bg: 'bg-yellow-50 dark:bg-yellow-900/20', label: 'New Plan' },
    session_started: { icon: PlayCircle, color: 'text-green-500', bg: 'bg-green-50 dark:bg-green-900/20', label: 'Started' },
    session_stopped: { icon: StopCircle, color: 'text-red-500', bg: 'bg-red-50 dark:bg-red-900/20', label: 'Stopped' },
    eval_summary: { icon: Activity, color: 'text-gray-400', bg: 'bg-gray-50 dark:bg-gray-800', label: 'Watching' },
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
      case 'order_executed': {
        let s = `${p.side?.toUpperCase()} ${Number(p.quantity ?? 0).toFixed(6)} ${p.symbol} @ ${formatCurrency(p.price ?? 0)}`;
        if (p.stop_price != null) s += ` · stop ${formatCurrency(p.stop_price)}`;
        if (p.target_price != null) s += ` · target ${formatCurrency(p.target_price)}`;
        return s;
      }
      case 'plan_generated':
        return `${p.trigger_count ?? '?'} triggers · plan #${p.plan_index ?? '?'}`;
      case 'eval_summary':
        return `${p.symbol?.replace('-USD', '')} — ${p.triggers_evaluated ?? 0} triggers evaluated @ ${formatCurrency(p.price ?? 0)}${p.fired ? ` · ${p.fired} fired` : ' · none fired'}`;
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
        {ev.count != null && ev.count > 1 && (
          <span className="text-xs bg-orange-100 text-orange-600 dark:bg-orange-900/30 dark:text-orange-400 px-1.5 py-0.5 rounded-full font-mono flex-shrink-0">
            ×{ev.count}
          </span>
        )}
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
