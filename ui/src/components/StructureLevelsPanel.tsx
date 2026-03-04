/**
 * StructureLevelsPanel — compact S1 HTF anchors + S2 swing levels display.
 *
 * Replaces raw structure JSON in PaperTradingControl. Renders:
 *  - S1 Anchors: daily/5D/20D high/low reference levels
 *  - S2 Swings: swing highs/lows with ATR-distance bucket badge
 *  - Recent Events: latest 5 structure events (breaks, reclaims, range-outs)
 */

import React, { useMemo } from 'react';
import { cn, formatAssetPrice } from '../lib/utils';

// ─── Types (matches schemas/structure_engine.py) ─────────────────────────────

interface StructureLevel {
  price: number;
  kind: 'high' | 'low';
  role: 'anchor' | 'swing' | 'range_bound';
  timeframe?: string;
  label?: string;
  atr_distance?: number | null;
  atr_distance_pct?: number | null;
}

interface StructureEvent {
  event_type: 'break' | 'reclaim' | 'range_breakout' | 'structure_shift';
  level_price: number;
  direction: 'up' | 'down';
  bar_ts: string;
  policy_event_priority?: 'high' | 'medium' | 'low' | null;
}

interface StructureSnapshot {
  symbol: string;
  snapshot_id?: string;
  snapshot_hash?: string;
  levels?: StructureLevel[];
  events?: StructureEvent[];
  current_price?: number | null;
}

interface Props {
  snapshot: StructureSnapshot | null | undefined;
  currentPrice?: number | null;
  className?: string;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function formatPrice(p: number): string {
  return formatAssetPrice(p);
}

function atrBucketBadge(atrDist: number | null | undefined): { label: string; className: string } {
  if (atrDist == null) return { label: '?', className: 'bg-gray-100 text-gray-500 dark:bg-gray-700 dark:text-gray-400' };
  if (atrDist < 1.0) return { label: '🟢 near', className: 'bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-400' };
  if (atrDist < 2.5) return { label: '🟡 mid', className: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/40 dark:text-yellow-400' };
  return { label: '🔴 far', className: 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-400' };
}

function relativeTime(isoStr: string): string {
  try {
    const diff = (Date.now() - new Date(isoStr).getTime()) / 1000;
    if (diff < 60) return `${Math.round(diff)}s ago`;
    if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
    if (diff < 86400) return `${(diff / 3600).toFixed(1)}h ago`;
    return `${Math.round(diff / 86400)}d ago`;
  } catch {
    return isoStr;
  }
}

const EVENT_TYPE_LABEL: Record<string, string> = {
  break: '↓ break',
  reclaim: '↑ reclaim',
  range_breakout: '⬥ range_out',
  structure_shift: '⚡ shift',
};

// ─── Component ───────────────────────────────────────────────────────────────

export const StructureLevelsPanel: React.FC<Props> = ({ snapshot, currentPrice, className }) => {
  const price = currentPrice ?? snapshot?.current_price ?? null;
  const levels = snapshot?.levels ?? [];
  const events = snapshot?.events ?? [];

  const anchors = useMemo(
    () => levels.filter((l) => l.role === 'anchor'),
    [levels]
  );

  // Group anchor highs/lows by timeframe label
  const anchorMap = useMemo(() => {
    const map: Record<string, { high?: number; low?: number }> = {};
    for (const a of anchors) {
      const tf = a.timeframe ?? a.label ?? 'D-?';
      if (!map[tf]) map[tf] = {};
      if (a.kind === 'high') map[tf].high = a.price;
      if (a.kind === 'low') map[tf].low = a.price;
    }
    return map;
  }, [anchors]);

  const swings = useMemo(
    () => levels.filter((l) => l.role === 'swing').slice(0, 8),
    [levels]
  );

  const recentEvents = useMemo(() => events.slice(0, 5), [events]);

  if (!snapshot) {
    return (
      <div className={cn('text-xs text-gray-400 italic', className)}>
        No structure data available.
      </div>
    );
  }

  return (
    <div className={cn('space-y-3 text-xs', className)}>
      {/* S1 Anchors */}
      {Object.keys(anchorMap).length > 0 && (
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-1">
            S1 HTF Anchors
          </p>
          <table className="w-full">
            <tbody>
              {Object.entries(anchorMap).map(([tf, { high, low }]) => (
                <tr key={tf} className="border-b border-gray-100 dark:border-gray-700/50">
                  <td className="py-0.5 pr-2 font-mono font-semibold text-gray-500 dark:text-gray-400 w-12">
                    {tf}
                  </td>
                  <td className="py-0.5 pr-4 text-green-600 dark:text-green-400 font-mono">
                    {high != null ? `H ${formatPrice(high)}` : '—'}
                  </td>
                  <td className="py-0.5 text-red-600 dark:text-red-400 font-mono">
                    {low != null ? `L ${formatPrice(low)}` : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* S2 Swings */}
      {swings.length > 0 && (
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-1">
            S2 Swing Levels
          </p>
          <table className="w-full">
            <thead>
              <tr className="text-[10px] text-gray-400">
                <th className="text-left py-0.5 pr-2 font-normal">Level</th>
                <th className="text-left py-0.5 pr-2 font-normal">Role</th>
                <th className="text-left py-0.5 font-normal">ATR dist</th>
              </tr>
            </thead>
            <tbody>
              {swings.map((s, i) => {
                const badge = atrBucketBadge(s.atr_distance);
                const isAbove = price != null && s.price > price;
                return (
                  <tr key={i} className="border-b border-gray-100 dark:border-gray-700/50">
                    <td className={cn('py-0.5 pr-2 font-mono', isAbove ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400')}>
                      {formatPrice(s.price)}
                    </td>
                    <td className="py-0.5 pr-2 text-gray-500 capitalize">{s.kind}</td>
                    <td className="py-0.5">
                      <span className={cn('px-1 py-0.5 rounded text-[10px] font-medium', badge.className)}>
                        {badge.label}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Recent Events */}
      {recentEvents.length > 0 && (
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-1">
            Recent Events
          </p>
          <div className="space-y-0.5">
            {recentEvents.map((ev, i) => (
              <div key={i} className="flex items-center gap-2 font-mono">
                <span className="text-gray-500 dark:text-gray-400">
                  {EVENT_TYPE_LABEL[ev.event_type] ?? ev.event_type}
                </span>
                <span className={ev.direction === 'up' ? 'text-green-600' : 'text-red-600'}>
                  {formatPrice(ev.level_price)}
                </span>
                <span className="text-gray-400">{relativeTime(ev.bar_ts)}</span>
                {ev.policy_event_priority === 'high' && (
                  <span className="px-1 py-0.5 rounded bg-orange-100 text-orange-700 dark:bg-orange-900/40 dark:text-orange-400 text-[10px] font-medium">
                    high priority
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {anchors.length === 0 && swings.length === 0 && recentEvents.length === 0 && (
        <p className="text-gray-400 italic">Structure data present but no levels resolved yet.</p>
      )}
    </div>
  );
};

export default StructureLevelsPanel;
