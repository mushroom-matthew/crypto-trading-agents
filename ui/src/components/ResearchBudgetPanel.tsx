/**
 * ResearchBudgetPanel — configuration + live status for the R48 research budget.
 *
 * In the left-column config form:
 *   - Enable/disable toggle
 *   - Budget fraction input (% of initial cash)
 *   - Max loss input (% of research capital before auto-pause)
 *
 * When a session is running, shows live status below the (disabled) config fields:
 *   - Initial / Cash / P&L metrics
 *   - Utilization bar
 *   - Active experiment + playbook IDs
 *   - Paused banner with reason
 *   - Pending judge-suggested playbook edits with one-click Approve
 */

import { useState, memo, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  ChevronDown, ChevronUp, FlaskConical,
  AlertTriangle, CheckCircle, Loader2,
} from 'lucide-react';
import { researchAPI } from '../lib/api';
import { cn, formatCurrency } from '../lib/utils';
import { useDebouncedCallback } from '../hooks/useDebounce';

// ── Config shape (mirrors PaperTradingSessionConfig fields) ──────────────────

export interface ResearchBudgetSettings {
  research_budget_enabled?: boolean;
  research_budget_fraction?: number;   // 0.01 – 0.50
  research_max_loss_pct?: number;      // 5 – 100
}

// ── Props ────────────────────────────────────────────────────────────────────

interface ResearchBudgetPanelProps<T extends ResearchBudgetSettings> {
  config: T;
  onChange: (config: T) => void;
  disabled?: boolean;
  /** Pass the running session ID to show live status below the config knobs. */
  sessionId?: string;
}

// ── Panel ────────────────────────────────────────────────────────────────────

function ResearchBudgetPanelInner<T extends ResearchBudgetSettings>({
  config,
  onChange,
  disabled,
  sessionId,
}: ResearchBudgetPanelProps<T>) {
  const [isExpanded, setIsExpanded] = useState(false);
  const queryClient = useQueryClient();
  const debouncedOnChange = useDebouncedCallback(onChange, 150);

  const isRunning = !!sessionId && disabled;
  const hasSelectedSession = !!sessionId;

  // ── Live status queries (only when running + expanded) ──────────────────

  const budgetQuery = useQuery({
    queryKey: ['research-budget', sessionId],
    queryFn: () => researchAPI.getBudget(sessionId!),
    refetchInterval: isRunning ? 10000 : false,
    enabled: hasSelectedSession && isExpanded,
    retry: false,
  });

  const suggestionsQuery = useQuery({
    queryKey: ['research-edit-suggestions'],
    queryFn: () => researchAPI.getEditSuggestions(),
    refetchInterval: isRunning ? 30000 : false,
    enabled: isExpanded,
    retry: false,
  });

  const applyMutation = useMutation({
    mutationFn: ({ playbookId, suggestionId }: { playbookId: string; suggestionId: string }) =>
      researchAPI.applySuggestion(playbookId, suggestionId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['research-edit-suggestions'] }),
  });

  // ── Derived values ───────────────────────────────────────────────────────

  const enabled = config.research_budget_enabled ?? true;
  const fraction = config.research_budget_fraction ?? 0.10;
  const maxLoss = config.research_max_loss_pct ?? 50;
  const budget = budgetQuery.data;
  const suggestions = suggestionsQuery.data ?? [];
  const pnlPositive = (budget?.total_pnl ?? 0) >= 0;
  const utilization = budget?.initial_capital
    ? Math.max(0, 100 - ((budget.cash ?? 0) / budget.initial_capital) * 100)
    : 0;

  // ── Handlers ─────────────────────────────────────────────────────────────

  const handleEnabled = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    onChange({ ...config, research_budget_enabled: e.target.checked });
  }, [config, onChange]);

  const handleFraction = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const v = parseFloat(e.target.value);
    if (!isNaN(v)) debouncedOnChange({ ...config, research_budget_fraction: v / 100 });
  }, [config, debouncedOnChange]);

  const handleMaxLoss = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const v = parseFloat(e.target.value);
    if (!isNaN(v)) debouncedOnChange({ ...config, research_max_loss_pct: v });
  }, [config, debouncedOnChange]);

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <div className="border border-violet-200 dark:border-violet-800 rounded-lg overflow-hidden">
      {/* Header */}
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 flex items-center justify-between bg-violet-50 dark:bg-violet-900/20 hover:bg-violet-100 dark:hover:bg-violet-900/30 transition-colors"
      >
        <div className="flex items-center gap-2">
          <FlaskConical className="w-4 h-4 text-violet-600 dark:text-violet-400" />
          <span className="text-sm font-semibold text-violet-800 dark:text-violet-200">
            Research Budget
          </span>
          {!enabled && (
            <span className="px-1.5 py-0.5 text-xs rounded bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400">
              OFF
            </span>
          )}
          {enabled && (
            <span className="text-xs text-violet-600 dark:text-violet-400 font-mono">
              {(fraction * 100).toFixed(0)}%
            </span>
          )}
          {budget?.paused && (
            <span className="px-1.5 py-0.5 text-xs rounded bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300 font-medium">
              PAUSED
            </span>
          )}
          {suggestions.length > 0 && (
            <span className="px-1.5 py-0.5 text-xs rounded-full bg-amber-500 text-white font-bold">
              {suggestions.length}
            </span>
          )}
        </div>
        {isExpanded
          ? <ChevronUp className="w-4 h-4 text-violet-500" />
          : <ChevronDown className="w-4 h-4 text-violet-500" />}
      </button>

      {isExpanded && (
        <div className="p-4 space-y-4 bg-white dark:bg-gray-900">

          {/* ── Config knobs ── */}
          <div className="space-y-3">
            {/* Enable toggle */}
            <div className="flex items-center gap-3">
              <input
                type="checkbox"
                id="research_enabled"
                checked={enabled}
                onChange={handleEnabled}
                disabled={disabled}
                className="w-4 h-4 text-violet-600 rounded"
              />
              <label htmlFor="research_enabled" className="text-sm">
                Allocate a separate research capital pool
              </label>
            </div>

            {enabled && (
              <div className="grid grid-cols-2 gap-3 pl-7">
                {/* Budget fraction */}
                <div>
                  <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
                    Budget fraction (% of cash)
                  </label>
                  <div className="relative">
                    <input
                      type="number"
                      min={1} max={50} step={1}
                      defaultValue={(fraction * 100).toFixed(0)}
                      onChange={handleFraction}
                      disabled={disabled}
                      className="w-full px-3 py-1.5 pr-8 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-sm focus:ring-2 focus:ring-violet-500 focus:border-transparent disabled:opacity-60"
                    />
                    <span className="absolute right-2 top-1/2 -translate-y-1/2 text-xs text-gray-400">%</span>
                  </div>
                </div>

                {/* Max loss */}
                <div>
                  <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
                    Max loss before pause
                  </label>
                  <div className="relative">
                    <input
                      type="number"
                      min={5} max={100} step={5}
                      defaultValue={maxLoss.toFixed(0)}
                      onChange={handleMaxLoss}
                      disabled={disabled}
                      className="w-full px-3 py-1.5 pr-8 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-sm focus:ring-2 focus:ring-violet-500 focus:border-transparent disabled:opacity-60"
                    />
                    <span className="absolute right-2 top-1/2 -translate-y-1/2 text-xs text-gray-400">%</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* ── Session status (selected session, running or stopped) ── */}
          {hasSelectedSession && enabled && (
            <div className="border-t border-violet-100 dark:border-violet-800 pt-4 space-y-3">
              <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                {isRunning ? 'Live Status' : 'Selected Session Status'}
              </p>

              {budgetQuery.isLoading && (
                <div className="flex items-center gap-2 text-sm text-gray-400">
                  <Loader2 className="w-4 h-4 animate-spin" /> Loading…
                </div>
              )}

              {!budgetQuery.isLoading && budgetQuery.error && (
                <p className="text-xs text-red-600 dark:text-red-400">
                  Failed to load research budget: {(budgetQuery.error as Error).message}
                </p>
              )}

              {budget && !budget.research_enabled && (
                <p className="text-xs text-gray-400">{budget.message}</p>
              )}

              {budget?.research_enabled && (
                <>
                  {/* Paused banner */}
                  {budget.paused && (
                    <div className="flex items-start gap-2 rounded-md bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700 px-3 py-2">
                      <AlertTriangle className="w-4 h-4 text-amber-500 mt-0.5 shrink-0" />
                      <p className="text-xs text-amber-800 dark:text-amber-200">
                        <span className="font-semibold">Paused:</span>{' '}
                        {budget.pause_reason ?? 'max loss reached'}
                      </p>
                    </div>
                  )}

                  {/* Capital metrics */}
                  <div className="grid grid-cols-3 gap-2">
                    {[
                      { label: 'Initial', value: formatCurrency(budget.initial_capital ?? 0), cls: '' },
                      { label: 'Cash', value: formatCurrency(budget.cash ?? 0), cls: '' },
                      {
                        label: 'P&L',
                        value: `${pnlPositive ? '+' : ''}${formatCurrency(budget.total_pnl ?? 0)}`,
                        cls: pnlPositive ? 'text-green-600 dark:text-green-400' : 'text-red-500 dark:text-red-400',
                      },
                    ].map(({ label, value, cls }) => (
                      <div key={label} className="rounded-lg bg-gray-50 dark:bg-gray-800 p-2 text-center">
                        <p className="text-xs text-gray-400 mb-0.5">{label}</p>
                        <p className={cn('text-xs font-bold text-gray-800 dark:text-gray-100', cls)}>{value}</p>
                      </div>
                    ))}
                  </div>

                  {/* Utilization bar */}
                  <div>
                    <div className="flex justify-between text-xs text-gray-400 mb-1">
                      <span>Utilization</span>
                      <span>{utilization.toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 rounded-full bg-gray-200 dark:bg-gray-700">
                      <div
                        className="h-1.5 rounded-full bg-violet-500 transition-all"
                        style={{ width: `${Math.min(utilization, 100)}%` }}
                      />
                    </div>
                  </div>

                  {/* Active experiment / playbook */}
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <p className="text-gray-400 mb-0.5">Experiment</p>
                      <p className="font-mono truncate text-gray-700 dark:text-gray-300">
                        {budget.active_experiment_id ?? '—'}
                      </p>
                    </div>
                    <div>
                      <p className="text-gray-400 mb-0.5">Playbook</p>
                      <p className="font-mono truncate text-gray-700 dark:text-gray-300">
                        {budget.active_playbook_id ?? '—'}
                      </p>
                    </div>
                  </div>

                  <p className="text-xs text-gray-400">
                    Research trades:{' '}
                    <span className="font-semibold text-gray-600 dark:text-gray-300">
                      {budget.n_trades ?? 0}
                    </span>
                  </p>
                </>
              )}

              {/* Pending playbook edit suggestions */}
              {suggestions.length > 0 && (
                <div className="space-y-2 pt-1">
                  <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                    Judge-suggested playbook edits
                  </p>
                  {suggestions.map((s) => (
                    <div
                      key={s.suggestion_id}
                      className="rounded-lg border border-amber-200 dark:border-amber-700 bg-amber-50 dark:bg-amber-900/10 p-3 space-y-1.5"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-1 text-xs">
                          <span className="font-mono text-violet-700 dark:text-violet-300">{s.playbook_id}</span>
                          <span className="text-gray-400">§</span>
                          <span className="text-gray-600 dark:text-gray-300">{s.section}</span>
                        </div>
                        <button
                          type="button"
                          disabled={applyMutation.isPending}
                          onClick={() =>
                            applyMutation.mutate({ playbookId: s.playbook_id, suggestionId: s.suggestion_id })
                          }
                          className="flex items-center gap-1 px-2 py-1 text-xs rounded bg-green-600 hover:bg-green-700 text-white disabled:opacity-50 transition-colors"
                        >
                          {applyMutation.isPending
                            ? <Loader2 className="w-3 h-3 animate-spin" />
                            : <CheckCircle className="w-3 h-3" />}
                          Approve
                        </button>
                      </div>
                      <p className="text-xs text-gray-500 italic">{s.evidence_summary}</p>
                      <pre className="text-xs bg-white dark:bg-gray-800 rounded p-2 overflow-x-auto whitespace-pre-wrap text-gray-700 dark:text-gray-300 max-h-20 overflow-y-auto">
                        {s.suggested_text}
                      </pre>
                    </div>
                  ))}
                </div>
              )}

              {!suggestionsQuery.isLoading && suggestions.length === 0 && (
                <p className="text-xs text-gray-400">
                  No pending playbook edits from the judge yet.
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export const ResearchBudgetPanel = memo(ResearchBudgetPanelInner) as typeof ResearchBudgetPanelInner;
