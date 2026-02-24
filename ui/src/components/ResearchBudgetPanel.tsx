/**
 * ResearchBudgetPanel — live view of the R48 research budget state for a
 * running paper trading session.
 *
 * Shows:
 *  - Research capital: initial / remaining cash / cumulative P&L
 *  - Paused indicator + reason
 *  - Active experiment / playbook IDs
 *  - Trade count
 *  - Pending judge-suggested playbook edits (with one-click approve)
 */

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { FlaskConical, ChevronDown, ChevronUp, AlertTriangle, CheckCircle, Loader2 } from 'lucide-react';
import { researchAPI } from '../lib/api';
import { cn, formatCurrency } from '../lib/utils';

interface ResearchBudgetPanelProps {
  sessionId: string;
  isRunning: boolean;
}

export function ResearchBudgetPanel({ sessionId, isRunning }: ResearchBudgetPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const queryClient = useQueryClient();

  const budgetQuery = useQuery({
    queryKey: ['research-budget', sessionId],
    queryFn: () => researchAPI.getBudget(sessionId),
    refetchInterval: isRunning ? 10000 : false,
    enabled: !!sessionId && isExpanded,
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
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['research-edit-suggestions'] });
    },
  });

  const budget = budgetQuery.data;
  const suggestions = suggestionsQuery.data ?? [];

  // Derive display values
  const pnlPositive = (budget?.total_pnl ?? 0) >= 0;
  const utilization = budget?.initial_capital
    ? Math.max(0, 100 - ((budget.cash ?? 0) / budget.initial_capital) * 100)
    : 0;

  return (
    <div className="border border-violet-200 dark:border-violet-800 rounded-lg overflow-hidden">
      {/* Header / toggle */}
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
          {budget?.research_enabled && budget.paused && (
            <span className="px-1.5 py-0.5 text-xs rounded bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300 font-medium">
              PAUSED
            </span>
          )}
          {budget?.research_enabled && !budget.paused && isRunning && (
            <span className="w-2 h-2 rounded-full bg-violet-400 animate-pulse" />
          )}
          {suggestions.length > 0 && (
            <span className="px-1.5 py-0.5 text-xs rounded-full bg-amber-500 text-white font-bold">
              {suggestions.length} pending
            </span>
          )}
        </div>
        {isExpanded ? (
          <ChevronUp className="w-4 h-4 text-violet-500" />
        ) : (
          <ChevronDown className="w-4 h-4 text-violet-500" />
        )}
      </button>

      {isExpanded && (
        <div className="p-4 space-y-4 bg-white dark:bg-gray-900">

          {/* Loading state */}
          {budgetQuery.isLoading && (
            <div className="flex items-center gap-2 text-sm text-gray-500">
              <Loader2 className="w-4 h-4 animate-spin" />
              Loading research budget…
            </div>
          )}

          {/* Not enabled / no session */}
          {budget && !budget.research_enabled && (
            <p className="text-sm text-gray-500 dark:text-gray-400">
              {budget.message ?? 'Research budget not initialized for this session.'}
            </p>
          )}

          {/* Budget state */}
          {budget?.research_enabled && (
            <>
              {/* Paused banner */}
              {budget.paused && (
                <div className="flex items-start gap-2 rounded-md bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700 px-3 py-2">
                  <AlertTriangle className="w-4 h-4 text-amber-500 mt-0.5 shrink-0" />
                  <div className="text-sm text-amber-800 dark:text-amber-200">
                    <span className="font-semibold">Research paused:</span>{' '}
                    {budget.pause_reason ?? 'max loss reached'}
                  </div>
                </div>
              )}

              {/* Capital metrics */}
              <div className="grid grid-cols-3 gap-3">
                <div className="rounded-lg bg-gray-50 dark:bg-gray-800 p-3 text-center">
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Initial</p>
                  <p className="text-sm font-bold text-gray-800 dark:text-gray-100">
                    {formatCurrency(budget.initial_capital ?? 0)}
                  </p>
                </div>
                <div className="rounded-lg bg-gray-50 dark:bg-gray-800 p-3 text-center">
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Cash</p>
                  <p className="text-sm font-bold text-gray-800 dark:text-gray-100">
                    {formatCurrency(budget.cash ?? 0)}
                  </p>
                </div>
                <div className="rounded-lg bg-gray-50 dark:bg-gray-800 p-3 text-center">
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Total P&amp;L</p>
                  <p className={cn('text-sm font-bold', pnlPositive ? 'text-green-600 dark:text-green-400' : 'text-red-500 dark:text-red-400')}>
                    {pnlPositive ? '+' : ''}{formatCurrency(budget.total_pnl ?? 0)}
                  </p>
                </div>
              </div>

              {/* Utilization bar */}
              <div>
                <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mb-1">
                  <span>Capital utilization</span>
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
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>
                  <span className="text-gray-500 dark:text-gray-400 text-xs">Experiment</span>
                  <p className="font-mono text-xs truncate text-gray-800 dark:text-gray-100">
                    {budget.active_experiment_id ?? <span className="text-gray-400 italic">none</span>}
                  </p>
                </div>
                <div>
                  <span className="text-gray-500 dark:text-gray-400 text-xs">Playbook</span>
                  <p className="font-mono text-xs truncate text-gray-800 dark:text-gray-100">
                    {budget.active_playbook_id ?? <span className="text-gray-400 italic">none</span>}
                  </p>
                </div>
              </div>

              {/* Trade count */}
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Research trades: <span className="font-semibold text-gray-700 dark:text-gray-300">{budget.n_trades ?? 0}</span>
              </p>
            </>
          )}

          {/* Pending playbook edit suggestions */}
          {suggestions.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs font-semibold text-gray-600 dark:text-gray-300 uppercase tracking-wide">
                Judge-suggested playbook edits
              </p>
              {suggestions.map((s) => (
                <div
                  key={s.suggestion_id}
                  className="rounded-lg border border-amber-200 dark:border-amber-700 bg-amber-50 dark:bg-amber-900/10 p-3 space-y-1"
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono text-xs text-violet-700 dark:text-violet-300">
                        {s.playbook_id}
                      </span>
                      <span className="text-gray-400">§</span>
                      <span className="text-xs text-gray-600 dark:text-gray-300">{s.section}</span>
                    </div>
                    <button
                      type="button"
                      disabled={applyMutation.isPending}
                      onClick={() =>
                        applyMutation.mutate({ playbookId: s.playbook_id, suggestionId: s.suggestion_id })
                      }
                      className="flex items-center gap-1 px-2 py-1 text-xs rounded bg-green-600 hover:bg-green-700 text-white disabled:opacity-50 transition-colors"
                    >
                      {applyMutation.isPending ? (
                        <Loader2 className="w-3 h-3 animate-spin" />
                      ) : (
                        <CheckCircle className="w-3 h-3" />
                      )}
                      Approve
                    </button>
                  </div>
                  <p className="text-xs text-gray-600 dark:text-gray-400 italic">
                    {s.evidence_summary}
                  </p>
                  <pre className="text-xs bg-white dark:bg-gray-800 rounded p-2 overflow-x-auto whitespace-pre-wrap text-gray-700 dark:text-gray-300 max-h-24 overflow-y-auto">
                    {s.suggested_text}
                  </pre>
                </div>
              ))}
            </div>
          )}

          {/* No suggestions placeholder */}
          {!suggestionsQuery.isLoading && suggestions.length === 0 && (
            <p className="text-xs text-gray-400 dark:text-gray-500">
              No pending playbook edits. The judge will suggest changes here after validating experiments.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
