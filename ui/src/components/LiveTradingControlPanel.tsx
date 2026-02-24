import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import {
  AlertTriangle,
  CheckCircle2,
  Loader2,
  RefreshCw,
  ShieldAlert,
  Sparkles,
} from 'lucide-react';

import { screenerAPI, type ScreenerRecommendationItem } from '../lib/api';
import { cn, formatDateTime } from '../lib/utils';

type OpsStatus = {
  mode?: string;
  live_trading_ack?: boolean;
  ts?: string;
};

async function getOpsStatus(): Promise<OpsStatus> {
  const response = await fetch('/status');
  if (!response.ok) {
    throw new Error(`Status request failed (${response.status})`);
  }
  return response.json();
}

type StagedSelection = {
  symbol: string;
  hypothesis: string;
  templateId?: string | null;
  timeframe: string;
  confidence: string;
  score: number;
  thesis: string;
};

export function LiveTradingControlPanel() {
  const [annotate, setAnnotate] = useState(false);
  const [confirmHighRisk, setConfirmHighRisk] = useState(false);
  const [staged, setStaged] = useState<StagedSelection | null>(null);

  const statusQuery = useQuery({
    queryKey: ['ops-status'],
    queryFn: getOpsStatus,
    refetchInterval: 15000,
    retry: false,
  });

  const preflightQuery = useQuery({
    queryKey: ['screener-session-preflight', 'live', annotate],
    queryFn: () => screenerAPI.getSessionPreflight('live', { annotate }),
    refetchInterval: 60000,
    retry: false,
  });

  const screenerErrorStatus = (preflightQuery.error as any)?.response?.status as number | undefined;
  const runScreenerNow = useMutation({
    mutationFn: () => screenerAPI.runOnce({ timeframe: '1h', lookback_bars: 50 }),
    onSuccess: async () => {
      await preflightQuery.refetch();
    },
  });
  const opsMode = statusQuery.data?.mode ?? 'unknown';
  const liveAck = Boolean(statusQuery.data?.live_trading_ack);
  const annotationApplied = Boolean(preflightQuery.data?.shortlist.annotation_meta?.applied);

  const stageCandidate = (item: ScreenerRecommendationItem) => {
    setStaged({
      symbol: item.symbol,
      hypothesis: item.hypothesis,
      templateId: item.template_id,
      timeframe: item.expected_hold_timeframe,
      confidence: item.confidence,
      score: item.composite_score,
      thesis: item.thesis,
    });
  };

  const canArm = !!staged && confirmHighRisk && liveAck;

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-red-100 dark:border-red-900/30 p-6">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <ShieldAlert className="w-5 h-5 text-red-500" />
            Live Trading Control Panel
          </h2>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Stage a screener-backed live candidate with explicit confirmation before any future live-start wiring.
          </p>
        </div>
        <div className="flex flex-wrap gap-2 text-xs">
          <span className={cn(
            'px-2 py-1 rounded-full border',
            opsMode === 'live'
              ? 'bg-red-50 text-red-700 border-red-200 dark:bg-red-900/20 dark:text-red-300 dark:border-red-800'
              : 'bg-gray-50 text-gray-700 border-gray-200 dark:bg-gray-700 dark:text-gray-200 dark:border-gray-600'
          )}>
            Mode: {opsMode}
          </span>
          <span className={cn(
            'px-2 py-1 rounded-full border',
            liveAck
              ? 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-900/20 dark:text-emerald-300 dark:border-emerald-800'
              : 'bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-900/20 dark:text-amber-300 dark:border-amber-800'
          )}>
            LIVE_TRADING_ACK: {liveAck ? 'enabled' : 'missing'}
          </span>
          <button
            type="button"
            onClick={() => {
              statusQuery.refetch();
              preflightQuery.refetch();
            }}
            disabled={statusQuery.isFetching || preflightQuery.isFetching}
            className="inline-flex items-center gap-1 px-2 py-1 rounded-full border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-60"
          >
            <RefreshCw className={cn('w-3.5 h-3.5', (statusQuery.isFetching || preflightQuery.isFetching) && 'animate-spin')} />
            Refresh
          </button>
          <button
            type="button"
            onClick={() => runScreenerNow.mutate()}
            disabled={runScreenerNow.isPending}
            className="inline-flex items-center gap-1 px-2 py-1 rounded-full border border-red-300 dark:border-red-700 hover:bg-red-50 dark:hover:bg-red-900/20 disabled:opacity-60"
            title="Run one screener pass now and refresh live preflight"
          >
            {runScreenerNow.isPending ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Sparkles className="w-3.5 h-3.5" />
            )}
            Run Screener Now
          </button>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-1 xl:grid-cols-5 gap-4">
        <div className="xl:col-span-3 rounded-lg border border-gray-200 dark:border-gray-700 p-4 space-y-3">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-sm font-medium">Screener Preflight (Live)</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Grouped shortlist by supported strategy hypothesis and hold timeframe.
              </p>
            </div>
            <label className="inline-flex items-center gap-2 text-xs">
              <input
                type="checkbox"
                className="w-4 h-4 text-violet-600 rounded"
                checked={annotate}
                onChange={(e) => setAnnotate(e.target.checked)}
              />
              LLM annotate / rerank
            </label>
          </div>

          {preflightQuery.isLoading && (
            <div className="text-sm text-gray-500 flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin" />
              Loading live preflight…
            </div>
          )}

          {!preflightQuery.isLoading && screenerErrorStatus === 404 && (
            <div className="rounded-md border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-900/10 p-3 text-sm text-amber-800 dark:text-amber-300">
              No screener preflight available yet. Run the universe screener workflow first.
            </div>
          )}

          {!preflightQuery.isLoading && preflightQuery.error && screenerErrorStatus !== 404 && (
            <div className="rounded-md border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/10 p-3 text-sm text-red-700 dark:text-red-300">
              Failed to load live preflight: {(preflightQuery.error as Error).message}
            </div>
          )}
          {runScreenerNow.error && (
            <div className="rounded-md border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/10 p-3 text-sm text-red-700 dark:text-red-300">
              Failed to run screener: {(runScreenerNow.error as Error).message}
            </div>
          )}
          {runScreenerNow.data && (
            <div className="rounded-md border border-emerald-200 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-900/10 p-3 text-sm text-emerald-700 dark:text-emerald-300">
              Screener refreshed: {runScreenerNow.data.top_candidates} candidates
              {runScreenerNow.data.selected_symbol ? `, selected ${runScreenerNow.data.selected_symbol}` : ''}.
            </div>
          )}

          {preflightQuery.data && (
            <>
              <div className="flex flex-wrap gap-2 text-xs">
                <span className="px-2 py-1 rounded-full border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/30">
                  {formatDateTime(preflightQuery.data.as_of)}
                </span>
                <span className="px-2 py-1 rounded-full border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/30">
                  {preflightQuery.data.shortlist.groups.length} groups
                </span>
                <span className={cn(
                  'px-2 py-1 rounded-full border',
                  annotationApplied
                    ? 'bg-violet-50 text-violet-700 border-violet-200 dark:bg-violet-900/20 dark:text-violet-300 dark:border-violet-800'
                    : 'bg-gray-50 text-gray-700 border-gray-200 dark:bg-gray-900/30 dark:text-gray-200 dark:border-gray-700'
                )}>
                  {annotationApplied ? 'LLM annotated' : 'Deterministic'}
                </span>
              </div>

              <div className="max-h-[420px] overflow-y-auto pr-1 space-y-3">
                {preflightQuery.data.shortlist.groups.map((group) => (
                  <div
                    key={`${group.hypothesis}:${group.timeframe}`}
                    className="rounded-lg border border-gray-200 dark:border-gray-700 p-3"
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-sm font-semibold">{group.label}</span>
                      <span className="text-[11px] font-mono px-1.5 py-0.5 rounded bg-red-50 text-red-700 dark:bg-red-900/20 dark:text-red-300">
                        {group.hypothesis}
                      </span>
                      <span className="text-[11px] font-mono px-1.5 py-0.5 rounded bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-200">
                        {group.timeframe}
                      </span>
                    </div>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{group.rationale}</p>
                    <div className="mt-2 space-y-2">
                      {group.recommendations.map((item) => (
                        <button
                          key={`${group.hypothesis}:${group.timeframe}:${item.symbol}:${item.rank_global}`}
                          type="button"
                          onClick={() => stageCandidate(item)}
                          className={cn(
                            'w-full text-left rounded-md border p-2.5 transition-colors',
                            staged?.symbol === item.symbol && staged?.hypothesis === item.hypothesis
                              ? 'border-red-300 bg-red-50 dark:border-red-700 dark:bg-red-900/10'
                              : 'border-gray-200 hover:bg-gray-50 dark:border-gray-700 dark:hover:bg-gray-700/30'
                          )}
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div className="min-w-0">
                              <div className="flex flex-wrap items-center gap-2">
                                <span className="font-mono font-semibold">{item.symbol}</span>
                                <span className="text-[11px] text-gray-500 font-mono">
                                  score {item.composite_score.toFixed(2)} · #{item.rank_global}
                                </span>
                                <span className={cn(
                                  'text-[11px] px-1.5 py-0.5 rounded',
                                  item.confidence === 'high'
                                    ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/20 dark:text-emerald-300'
                                    : item.confidence === 'medium'
                                      ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/20 dark:text-amber-300'
                                      : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-200'
                                )}>
                                  {item.confidence}
                                </span>
                              </div>
                              <p className="text-xs text-gray-600 dark:text-gray-400 mt-1 line-clamp-2">{item.thesis}</p>
                            </div>
                            <div className="shrink-0 text-xs text-gray-500 dark:text-gray-400 flex items-center gap-1">
                              <CheckCircle2 className={cn('w-3.5 h-3.5', staged?.symbol === item.symbol ? 'text-red-500' : 'text-gray-400')} />
                              Stage
                            </div>
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>

        <div className="xl:col-span-2 rounded-lg border border-red-200 dark:border-red-900/40 bg-red-50/40 dark:bg-red-900/10 p-4 space-y-4">
          <div>
            <p className="text-sm font-semibold flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-red-500" />
              Live Execution Staging
            </p>
            <p className="text-xs text-red-700/90 dark:text-red-300/90 mt-1">
              This panel stages a candidate and confirmation only. It does not place trades or start live execution yet.
            </p>
          </div>

          {staged ? (
            <div className="rounded-lg border border-red-200 dark:border-red-800 p-3 bg-white/80 dark:bg-gray-900/30 space-y-2">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-mono font-semibold">{staged.symbol}</span>
                <span className="text-[11px] px-1.5 py-0.5 rounded bg-red-100 text-red-700 dark:bg-red-900/20 dark:text-red-300">
                  {staged.hypothesis}
                </span>
                <span className="text-[11px] px-1.5 py-0.5 rounded bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-200 font-mono">
                  {staged.timeframe}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <p className="text-gray-500 dark:text-gray-400">Template</p>
                  <p className="font-mono">{staged.templateId || 'None'}</p>
                </div>
                <div>
                  <p className="text-gray-500 dark:text-gray-400">Confidence / Score</p>
                  <p className="font-mono">{staged.confidence} / {staged.score.toFixed(2)}</p>
                </div>
              </div>
              <p className="text-xs text-gray-600 dark:text-gray-300">{staged.thesis}</p>
            </div>
          ) : (
            <div className="rounded-lg border border-dashed border-red-200 dark:border-red-800 p-3 text-sm text-gray-500 dark:text-gray-400">
              Select a screener candidate to stage a live setup.
            </div>
          )}

          <label className="flex items-start gap-2 text-xs">
            <input
              type="checkbox"
              className="w-4 h-4 mt-0.5 text-red-600 rounded"
              checked={confirmHighRisk}
              onChange={(e) => setConfirmHighRisk(e.target.checked)}
            />
            <span>
              I understand this is a live trading context and staged selections require explicit operator review before any execution.
            </span>
          </label>

          <div className="rounded-md border border-gray-200 dark:border-gray-700 bg-white/70 dark:bg-gray-900/30 p-3">
            <p className="text-xs font-semibold flex items-center gap-1">
              <Sparkles className="w-3.5 h-3.5" />
              Ready State
            </p>
            <div className="mt-2 space-y-1 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-500">Candidate staged</span>
                <span className={staged ? 'text-emerald-600 dark:text-emerald-400' : 'text-gray-400'}>
                  {staged ? 'yes' : 'no'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Risk confirmation</span>
                <span className={confirmHighRisk ? 'text-emerald-600 dark:text-emerald-400' : 'text-gray-400'}>
                  {confirmHighRisk ? 'yes' : 'no'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">LIVE_TRADING_ACK</span>
                <span className={liveAck ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400'}>
                  {liveAck ? 'enabled' : 'missing'}
                </span>
              </div>
            </div>
            <button
              type="button"
              disabled={!canArm}
              className={cn(
                'mt-3 w-full inline-flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium border transition-colors',
                canArm
                  ? 'border-red-300 bg-red-600 text-white hover:bg-red-700'
                  : 'border-gray-300 text-gray-400 bg-gray-100 dark:bg-gray-800 dark:border-gray-700 cursor-not-allowed'
              )}
              title={canArm ? 'Future hook: send staged live configuration' : 'Stage a candidate, confirm risk, and enable LIVE_TRADING_ACK'}
            >
              <ShieldAlert className="w-4 h-4" />
              Arm Live Selection (UI-only)
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
