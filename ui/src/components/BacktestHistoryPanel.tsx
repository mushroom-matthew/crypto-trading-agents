import { useQuery } from '@tanstack/react-query';
import {
  Clock,
  CheckCircle,
  XCircle,
  Loader2,
  TrendingUp,
  TrendingDown,
  BarChart3,
  Calendar,
  Coins,
  RefreshCw
} from 'lucide-react';
import { backtestAPI, type BacktestListItem } from '../lib/api';
import { cn, formatCurrency, formatPercent, formatDateTime } from '../lib/utils';

export interface BacktestHistoryPanelProps {
  selectedRunId: string | null;
  onSelect: (runId: string) => void;
  maxItems?: number;
}

const StatusBadge = ({ status }: { status: string }) => {
  const config = {
    completed: { icon: CheckCircle, className: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' },
    running: { icon: Loader2, className: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400' },
    queued: { icon: Clock, className: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400' },
    failed: { icon: XCircle, className: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400' },
  }[status] || { icon: Clock, className: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300' };

  const Icon = config.icon;

  return (
    <span className={cn('inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium', config.className)}>
      <Icon className={cn('w-3 h-3', status === 'running' && 'animate-spin')} />
      {status}
    </span>
  );
};

const StrategyBadge = ({ strategy, strategyId }: { strategy?: string; strategyId?: string }) => {
  if (!strategy) return null;

  const isLLM = strategy === 'llm_strategist';
  const label = isLLM ? (strategyId || 'LLM') : 'Baseline';

  return (
    <span className={cn(
      'inline-flex items-center px-2 py-0.5 rounded text-xs font-medium',
      isLLM
        ? 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400'
        : 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
    )}>
      {label}
    </span>
  );
};

const MetricBadge = ({
  value,
  label,
  isPositive,
  icon: Icon
}: {
  value: string;
  label: string;
  isPositive?: boolean;
  icon?: React.ElementType;
}) => (
  <div className="flex items-center gap-1 text-xs">
    {Icon && <Icon className={cn('w-3 h-3', isPositive ? 'text-green-500' : 'text-red-500')} />}
    <span className={cn(
      'font-medium',
      isPositive !== undefined && (isPositive ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400')
    )}>
      {value}
    </span>
    <span className="text-gray-500 dark:text-gray-400">{label}</span>
  </div>
);

export function BacktestHistoryPanel({ selectedRunId, onSelect, maxItems = 10 }: BacktestHistoryPanelProps) {
  const { data: backtests = [], isLoading, isRefetching, refetch } = useQuery({
    queryKey: ['backtests-list'],
    queryFn: async () => {
      const response = await backtestAPI.listBacktests(undefined, maxItems);
      return response;
    },
    refetchInterval: 5000, // Auto-refresh every 5 seconds
  });

  if (isLoading) {
    return (
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">Backtest History</h3>
        </div>
        <div className="space-y-2">
          {[1, 2, 3].map((i) => (
            <div key={i} className="animate-pulse bg-gray-100 dark:bg-gray-700 h-20 rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  if (backtests.length === 0) {
    return (
      <div className="space-y-2">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">Backtest History</h3>
        <div className="text-center py-8 text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-dashed border-gray-300 dark:border-gray-600">
          <BarChart3 className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">No backtests yet</p>
          <p className="text-xs mt-1">Run your first backtest to see history</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Backtest History
          <span className="ml-2 text-xs text-gray-500 dark:text-gray-400">
            ({backtests.length} runs)
          </span>
        </h3>
        <button
          onClick={() => refetch()}
          className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
          title="Refresh list"
        >
          <RefreshCw className={cn('w-4 h-4', isRefetching && 'animate-spin')} />
        </button>
      </div>

      <div className="space-y-2 max-h-[400px] overflow-y-auto pr-1">
        {backtests.map((backtest: BacktestListItem) => {
          const isSelected = backtest.run_id === selectedRunId;
          const hasReturn = backtest.return_pct !== undefined && backtest.return_pct !== null;
          const isPositiveReturn = hasReturn && backtest.return_pct! > 0;

          return (
            <div
              key={backtest.run_id}
              onClick={() => {
                onSelect(backtest.run_id);
                localStorage.setItem('selectedBacktestRunId', backtest.run_id);
              }}
              className={cn(
                'p-3 rounded-lg border cursor-pointer transition-all duration-150',
                isSelected
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 ring-1 ring-blue-500'
                  : 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 hover:border-gray-300 dark:hover:border-gray-600 hover:shadow-sm'
              )}
            >
              {/* Header Row */}
              <div className="flex items-start justify-between gap-2 mb-2">
                <div className="flex items-center gap-2 flex-wrap">
                  <StatusBadge status={backtest.status} />
                  <StrategyBadge strategy={backtest.strategy} strategyId={backtest.strategy_id} />
                </div>
                {backtest.status === 'running' && backtest.progress > 0 && (
                  <span className="text-xs text-blue-600 dark:text-blue-400 font-medium">
                    {backtest.progress.toFixed(0)}%
                  </span>
                )}
              </div>

              {/* Symbols and Timeframe */}
              <div className="flex items-center gap-3 text-xs mb-2">
                {backtest.symbols && backtest.symbols.length > 0 && (
                  <div className="flex items-center gap-1">
                    <Coins className="w-3 h-3 text-gray-400" />
                    <span className="font-mono text-gray-700 dark:text-gray-300">
                      {backtest.symbols.join(', ')}
                    </span>
                  </div>
                )}
                {backtest.timeframe && (
                  <span className="text-gray-500 dark:text-gray-400">
                    {backtest.timeframe}
                  </span>
                )}
              </div>

              {/* Date Range */}
              {(backtest.start_date || backtest.end_date) && (
                <div className="flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400 mb-2">
                  <Calendar className="w-3 h-3" />
                  <span>
                    {backtest.start_date} → {backtest.end_date}
                  </span>
                </div>
              )}

              {/* Performance Metrics (for completed backtests) */}
              {backtest.status === 'completed' && (
                <div className="flex items-center gap-4 flex-wrap pt-2 border-t border-gray-100 dark:border-gray-700">
                  {hasReturn && (
                    <MetricBadge
                      value={formatPercent(backtest.return_pct!)}
                      label="return"
                      isPositive={isPositiveReturn}
                      icon={isPositiveReturn ? TrendingUp : TrendingDown}
                    />
                  )}
                  {backtest.total_trades !== undefined && backtest.total_trades !== null && (
                    <MetricBadge
                      value={backtest.total_trades.toString()}
                      label="trades"
                    />
                  )}
                  {backtest.sharpe_ratio !== undefined && backtest.sharpe_ratio !== null && (
                    <MetricBadge
                      value={backtest.sharpe_ratio.toFixed(2)}
                      label="sharpe"
                      isPositive={backtest.sharpe_ratio > 1}
                    />
                  )}
                  {backtest.max_drawdown_pct !== undefined && backtest.max_drawdown_pct !== null && (
                    <MetricBadge
                      value={formatPercent(backtest.max_drawdown_pct)}
                      label="dd"
                    />
                  )}
                </div>
              )}

              {/* Error Display */}
              {backtest.status === 'failed' && backtest.error && (
                <div className="mt-2 p-2 bg-red-50 dark:bg-red-900/20 rounded text-xs text-red-600 dark:text-red-400 truncate">
                  {backtest.error}
                </div>
              )}

              {/* Timestamp */}
              <div className="mt-2 text-xs text-gray-400 dark:text-gray-500">
                {backtest.completed_at
                  ? formatDateTime(backtest.completed_at)
                  : backtest.started_at
                    ? `Started ${formatDateTime(backtest.started_at)}`
                    : 'Pending...'}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
