import { useQuery } from '@tanstack/react-query';
import { Brain, DollarSign, Calendar, TrendingUp } from 'lucide-react';
import { backtestAPI } from '../lib/api';
import { formatCurrency } from '../lib/utils';

export interface LLMInsightsProps {
  runId: string;
}

export function LLMInsights({ runId }: LLMInsightsProps) {
  const { data: insights, isLoading, error } = useQuery({
    queryKey: ['llm-insights', runId],
    queryFn: async () => {
      const response = await backtestAPI.getLLMInsights(runId);
      return response.data;
    },
    enabled: !!runId,
    retry: false, // Don't retry if backtest didn't use LLM strategist
  });

  if (isLoading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        <div className="flex items-center gap-3 mb-4">
          <Brain className="w-6 h-6 text-purple-500" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
            AI Strategy Insights
          </h2>
        </div>
        <div className="text-center py-8 text-gray-500">
          <div className="animate-pulse">Loading AI insights...</div>
        </div>
      </div>
    );
  }

  if (error || !insights) {
    // Backtest didn't use LLM strategist - return null to hide component
    return null;
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <Brain className="w-6 h-6 text-purple-500" />
        <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
          AI Strategy Insights
        </h2>
        <span className="ml-auto text-xs px-2 py-1 bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400 rounded">
          LLM Strategist
        </span>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {/* Total Plans Generated */}
        <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-lg">
          <div className="flex items-center gap-2 mb-1">
            <Calendar className="w-4 h-4 text-purple-500" />
            <span className="text-sm text-gray-600 dark:text-gray-400">Plans Generated</span>
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {insights.total_plans_generated}
          </div>
        </div>

        {/* Total LLM Cost */}
        <div className="p-4 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg">
          <div className="flex items-center gap-2 mb-1">
            <DollarSign className="w-4 h-4 text-green-500" />
            <span className="text-sm text-gray-600 dark:text-gray-400">Total AI Cost</span>
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            ${insights.total_cost_usd?.toFixed(4) || '0.0000'}
          </div>
        </div>

        {/* Final Portfolio */}
        <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg">
          <div className="flex items-center gap-2 mb-1">
            <TrendingUp className="w-4 h-4 text-blue-500" />
            <span className="text-sm text-gray-600 dark:text-gray-400">Final Cash</span>
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {formatCurrency(insights.final_cash || 0)}
          </div>
        </div>
      </div>

      {/* Final Positions */}
      {insights.final_positions && Object.keys(insights.final_positions).length > 0 && (
        <div className="mb-6">
          <h3 className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">
            Final Positions ({Object.keys(insights.final_positions).length})
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {Object.entries(insights.final_positions).map(([symbol, qty]) => (
              <div
                key={symbol}
                className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg flex items-center justify-between"
              >
                <span className="font-mono font-semibold text-sm text-gray-900 dark:text-gray-100">
                  {symbol}
                </span>
                <span className="text-sm text-gray-600 dark:text-gray-400 font-mono">
                  {Math.abs(qty as number).toFixed(6)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Strategy Plans Log */}
      {insights.plan_log && insights.plan_log.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">
            AI-Generated Strategy Plans ({insights.plan_log.length})
          </h3>
          <div className="max-h-96 overflow-y-auto space-y-3">
            {insights.plan_log.slice(0, 10).map((plan: any, idx: number) => (
              <div
                key={idx}
                className="p-4 bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 rounded-lg border border-purple-200 dark:border-purple-800"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-mono text-purple-600 dark:text-purple-400">
                    Plan #{idx + 1}
                  </span>
                  {plan.timestamp && (
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      {new Date(plan.timestamp).toLocaleString()}
                    </span>
                  )}
                </div>
                {plan.triggers && plan.triggers.length > 0 && (
                  <div className="text-sm text-gray-700 dark:text-gray-300">
                    <span className="font-semibold">{plan.triggers.length}</span> triggers defined
                  </div>
                )}
                {plan.market_regime && (
                  <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                    Market Regime: {plan.market_regime}
                  </div>
                )}
              </div>
            ))}
          </div>
          {insights.plan_log.length > 10 && (
            <div className="mt-3 text-center text-sm text-gray-500 dark:text-gray-400">
              Showing 10 of {insights.plan_log.length} plans
            </div>
          )}
        </div>
      )}

      {/* Cost Breakdown */}
      {insights.llm_costs && Object.keys(insights.llm_costs).length > 0 && (
        <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
          <h3 className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">
            AI Cost Breakdown
          </h3>
          <div className="space-y-2">
            {Object.entries(insights.llm_costs).map(([model, cost]) => (
              <div
                key={model}
                className="flex items-center justify-between text-sm p-2 bg-gray-50 dark:bg-gray-700/50 rounded"
              >
                <span className="text-gray-600 dark:text-gray-400 font-mono">{model}</span>
                <span className="font-semibold text-gray-900 dark:text-gray-100">
                  ${(cost as number).toFixed(6)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
