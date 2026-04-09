/**
 * ScannerPanel — live opportunity ranking from the R74 scanner.
 *
 * Shows the top-N symbols by opportunity_score_norm with per-component
 * score bars (vol_edge, structure_edge, trend_edge, liquidity_score).
 * Clicking a row expands the component_explanation breakdown.
 *
 * Refreshes every 60s (scanner cadence is 5-15 min; polling is cheap).
 */

import { useState, memo, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  ChevronDown,
  ChevronUp,
  RefreshCw,
  TrendingUp,
  Activity,
  Zap,
  Info,
} from 'lucide-react';
import { scannerAPI, type OpportunityCard } from '../lib/api';
import { cn } from '../lib/utils';

// ── Score colour thresholds ──────────────────────────────────────────────────

function scoreColor(norm: number): string {
  if (norm >= 0.6) return 'text-green-400';
  if (norm >= 0.4) return 'text-amber-400';
  return 'text-gray-400';
}

function scoreBgColor(norm: number): string {
  if (norm >= 0.6) return 'bg-green-500';
  if (norm >= 0.4) return 'bg-amber-500';
  return 'bg-gray-500';
}

function horizonBadge(horizon: string): string {
  switch (horizon) {
    case 'scalp':   return 'bg-blue-900 text-blue-300';
    case 'swing':   return 'bg-purple-900 text-purple-300';
    default:        return 'bg-slate-700 text-slate-300';
  }
}

// ── Component score mini-bar ──────────────────────────────────────────────────

function MiniBar({ value, label, positive = true }: { value: number; label: string; positive?: boolean }) {
  const pct = Math.round(value * 100);
  const color = positive
    ? value >= 0.6 ? 'bg-green-500' : value >= 0.3 ? 'bg-amber-500' : 'bg-gray-500'
    : value >= 0.5 ? 'bg-red-500' : value >= 0.2 ? 'bg-amber-500' : 'bg-green-600';

  return (
    <div className="flex items-center gap-1.5 text-xs">
      <span className="text-gray-400 w-20 shrink-0">{label}</span>
      <div className="flex-1 bg-gray-800 rounded-full h-1.5 min-w-[60px]">
        <div
          className={cn('h-1.5 rounded-full transition-all', color)}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-gray-300 w-8 text-right">{pct}%</span>
    </div>
  );
}

// ── Single card row ───────────────────────────────────────────────────────────

function CardRow({ card, rank }: { card: OpportunityCard; rank: number }) {
  const [expanded, setExpanded] = useState(false);
  const toggle = useCallback(() => setExpanded(e => !e), []);

  return (
    <div className="border border-gray-700 rounded-lg overflow-hidden">
      {/* Main row */}
      <button
        onClick={toggle}
        className="w-full flex items-center gap-3 p-3 hover:bg-gray-800/60 transition-colors text-left"
      >
        {/* Rank */}
        <span className="text-gray-500 text-xs w-4 shrink-0">{rank}</span>

        {/* Symbol */}
        <span className="font-mono text-white text-sm font-medium w-28 shrink-0">
          {card.symbol}
        </span>

        {/* Score bar */}
        <div className="flex-1 flex items-center gap-2">
          <div className="flex-1 bg-gray-800 rounded-full h-2 max-w-[120px]">
            <div
              className={cn('h-2 rounded-full transition-all', scoreBgColor(card.opportunity_score_norm))}
              style={{ width: `${Math.round(card.opportunity_score_norm * 100)}%` }}
            />
          </div>
          <span className={cn('text-sm font-semibold w-10 shrink-0', scoreColor(card.opportunity_score_norm))}>
            {Math.round(card.opportunity_score_norm * 100)}
          </span>
        </div>

        {/* Horizon badge */}
        <span className={cn('text-xs px-1.5 py-0.5 rounded font-medium w-16 text-center shrink-0', horizonBadge(card.expected_hold_horizon))}>
          {card.expected_hold_horizon}
        </span>

        {/* Component scores (compact) */}
        <div className="hidden md:flex items-center gap-2 text-xs">
          <span className="text-gray-500" title="Vol edge">
            <Zap className="w-3 h-3 inline mr-0.5" />
            {Math.round(card.vol_edge * 100)}
          </span>
          <span className="text-gray-500" title="Structure edge">
            <Activity className="w-3 h-3 inline mr-0.5" />
            {Math.round(card.structure_edge * 100)}
          </span>
          <span className="text-gray-500" title="Trend edge">
            <TrendingUp className="w-3 h-3 inline mr-0.5" />
            {Math.round(card.trend_edge * 100)}
          </span>
        </div>

        {/* Expand chevron */}
        <span className="text-gray-500 ml-auto shrink-0">
          {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </span>
      </button>

      {/* Expanded detail */}
      {expanded && (
        <div className="px-4 pb-4 bg-gray-900/40 border-t border-gray-700/50 space-y-3">
          {/* Component score bars */}
          <div className="pt-3 space-y-1.5">
            <MiniBar value={card.vol_edge}         label="Vol edge"    positive />
            <MiniBar value={card.structure_edge}   label="Structure"   positive />
            <MiniBar value={card.trend_edge}       label="Trend"       positive />
            <MiniBar value={card.liquidity_score}  label="Liquidity"   positive />
            <MiniBar value={card.spread_penalty}   label="Spread pen." positive={false} />
            <MiniBar value={card.instability_penalty} label="Instability" positive={false} />
          </div>

          {/* Structure levels */}
          {(card.nearest_support != null || card.nearest_resistance != null) && (
            <div className="text-xs text-gray-400 space-y-0.5">
              {card.nearest_support != null && (
                <div>Support: <span className="text-green-400">{card.nearest_support.toFixed(6)}</span></div>
              )}
              {card.nearest_resistance != null && (
                <div>Resistance: <span className="text-red-400">{card.nearest_resistance.toFixed(6)}</span></div>
              )}
              <div>Structure levels: {card.structure_levels_count}</div>
            </div>
          )}

          {/* Component explanations */}
          <div className="space-y-1">
            {Object.entries(card.component_explanation ?? {}).map(([key, desc]) => (
              <div key={key} className="flex gap-2 text-xs">
                <span className="text-gray-500 w-24 shrink-0">{key}:</span>
                <span className="text-gray-300">{desc}</span>
              </div>
            ))}
          </div>

          {/* Indicator freshness */}
          <div className="text-xs text-gray-600">
            Scored {new Date(card.scored_at).toLocaleTimeString()} ·
            Indicator as-of {new Date(card.indicator_as_of).toLocaleTimeString()}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main panel ────────────────────────────────────────────────────────────────

interface ScannerPanelProps {
  className?: string;
}

function ScannerPanelInner({ className }: ScannerPanelProps) {
  const queryClient = useQueryClient();

  const { data: ranking, isLoading, error, dataUpdatedAt } = useQuery({
    queryKey: ['scanner-opportunities'],
    queryFn: () => scannerAPI.getOpportunities(),
    refetchInterval: 60_000,   // refresh every 60s
    staleTime: 55_000,
  });

  const runOnceMutation = useMutation({
    mutationFn: () => scannerAPI.runOnce(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scanner-opportunities'] });
    },
  });
  const cards = Array.isArray(ranking?.cards) ? ranking.cards : [];
  const topN = Number(ranking?.top_n ?? 0);
  const universeSize = Number(ranking?.universe_size ?? 0);
  const scanDurationMs = Number(ranking?.scan_duration_ms ?? 0);

  return (
    <div className={cn('space-y-3', className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-blue-400" />
          <h3 className="text-sm font-semibold text-gray-200">Opportunity Scanner</h3>
          {ranking && (
            <span className="text-xs text-gray-500">
              ({topN}/{universeSize} symbols · {scanDurationMs}ms)
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {dataUpdatedAt > 0 && (
            <span className="text-xs text-gray-600">
              {new Date(dataUpdatedAt).toLocaleTimeString()}
            </span>
          )}
          <button
            onClick={() => runOnceMutation.mutate()}
            disabled={runOnceMutation.isPending}
            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors disabled:opacity-50"
            title="Trigger immediate scan"
          >
            <RefreshCw className={cn('w-4 h-4', runOnceMutation.isPending && 'animate-spin')} />
          </button>
        </div>
      </div>

      {/* Score legend */}
      <div className="flex items-center gap-4 text-xs text-gray-500">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-green-500 inline-block" /> High ≥60
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-amber-500 inline-block" /> Mid 40-60
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-gray-500 inline-block" /> Low &lt;40
        </span>
      </div>

      {/* Loading */}
      {isLoading && (
        <div className="text-center text-gray-500 py-6 text-sm">Loading scanner data…</div>
      )}

      {/* Error */}
      {error && !isLoading && (
        <div className="text-center text-red-400 py-4 text-sm">
          Failed to load scanner data. Is the ops API running?
        </div>
      )}

      {/* No data */}
      {!isLoading && !error && !ranking && (
        <div className="text-center text-gray-500 py-6 text-sm space-y-2">
          <Info className="w-5 h-5 mx-auto text-gray-600" />
          <p>No scan results yet.</p>
          <p className="text-xs">
            Run a paper trading session or click{' '}
            <button
              onClick={() => runOnceMutation.mutate()}
              className="text-blue-400 underline"
            >
              Scan now
            </button>
            .
          </p>
        </div>
      )}

      {/* Ranking cards */}
      {ranking && cards.length > 0 && (
        <div className="space-y-1.5">
          {cards.map((card, i) => (
            <CardRow key={card.symbol} card={card} rank={i + 1} />
          ))}
        </div>
      )}

      {ranking && cards.length === 0 && (
        <div className="text-center text-gray-500 py-4 text-sm">
          Scan completed — no symbols scored.
        </div>
      )}
    </div>
  );
}

export const ScannerPanel = memo(ScannerPanelInner);
export default ScannerPanel;
