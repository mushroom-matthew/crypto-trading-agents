import { useQuery } from '@tanstack/react-query';
import { DollarSign, TrendingUp, PieChart, Wallet } from 'lucide-react';
import { backtestAPI } from '../lib/api';
import { formatCurrency, formatPercent } from '../lib/utils';
import { cn } from '../lib/utils';

export interface StateInspectorProps {
  runId: string;
  currentTimestamp: string;
  enabled?: boolean;
}

export function StateInspector({ runId, currentTimestamp, enabled = true }: StateInspectorProps) {
  const { data: state, isLoading } = useQuery({
    queryKey: ['portfolio-state', runId, currentTimestamp],
    queryFn: () => backtestAPI.getStateSnapshot(runId, currentTimestamp),
    enabled: enabled && !!currentTimestamp,
    staleTime: 60000, // Cache for 1 minute
  });

  if (isLoading || !state) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Portfolio State</h3>
        <div className="text-center py-8 text-gray-500">
          <div className="animate-pulse">Loading state...</div>
        </div>
      </div>
    );
  }

  const hasPositions = Object.keys(state.positions).length > 0;
  const isProfit = state.return_pct >= 0;

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 sticky top-4">
      <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">
        Portfolio State
      </h3>

      {/* Summary Cards */}
      <div className="space-y-4 mb-6">
        {/* Equity */}
        <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg">
          <div className="flex items-center gap-2 mb-1">
            <PieChart className="w-4 h-4 text-blue-500" />
            <span className="text-sm text-gray-600 dark:text-gray-400">Total Equity</span>
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {formatCurrency(state.equity)}
          </div>
        </div>

        {/* Return % */}
        <div className={cn(
          'p-4 rounded-lg',
          isProfit
            ? 'bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20'
            : 'bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20'
        )}>
          <div className="flex items-center gap-2 mb-1">
            <TrendingUp className={cn('w-4 h-4', isProfit ? 'text-green-500' : 'text-red-500')} />
            <span className="text-sm text-gray-600 dark:text-gray-400">Return</span>
          </div>
          <div className={cn(
            'text-2xl font-bold',
            isProfit ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
          )}>
            {formatPercent(state.return_pct)}
          </div>
        </div>

        {/* Cash */}
        <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-lg">
          <div className="flex items-center gap-2 mb-1">
            <DollarSign className="w-4 h-4 text-purple-500" />
            <span className="text-sm text-gray-600 dark:text-gray-400">Cash</span>
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {formatCurrency(state.cash)}
          </div>
        </div>

        {/* Realized P&L */}
        <div className={cn(
          'p-4 rounded-lg',
          state.pnl >= 0
            ? 'bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20'
            : 'bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20'
        )}>
          <div className="flex items-center gap-2 mb-1">
            <Wallet className={cn('w-4 h-4', state.pnl >= 0 ? 'text-green-500' : 'text-red-500')} />
            <span className="text-sm text-gray-600 dark:text-gray-400">Realized P&L</span>
          </div>
          <div className={cn(
            'text-xl font-bold',
            state.pnl >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
          )}>
            {formatCurrency(state.pnl)}
          </div>
        </div>
      </div>

      {/* Positions */}
      <div>
        <h4 className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">
          Active Positions ({Object.keys(state.positions).length})
        </h4>

        {!hasPositions ? (
          <div className="text-center py-6 text-gray-500 text-sm">
            No open positions
          </div>
        ) : (
          <div className="space-y-2">
            {Object.entries(state.positions).map(([symbol, qty]) => (
              <div
                key={symbol}
                className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-mono font-semibold text-sm text-gray-900 dark:text-gray-100">
                    {symbol}
                  </span>
                  <span className={cn(
                    'text-xs px-2 py-0.5 rounded',
                    qty > 0
                      ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400'
                      : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400'
                  )}>
                    {qty > 0 ? 'LONG' : 'SHORT'}
                  </span>
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  Qty: <span className="font-mono">{Math.abs(qty).toFixed(6)}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Timestamp */}
      <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
        <p className="text-xs text-gray-500 dark:text-gray-400">
          As of: {new Date(currentTimestamp).toLocaleString()}
        </p>
      </div>
    </div>
  );
}
