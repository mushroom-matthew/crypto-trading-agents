import { useQuery } from '@tanstack/react-query';
import { ChevronDown, Clock, CheckCircle, XCircle } from 'lucide-react';
import { backtestAPI } from '../lib/api';
import { formatDateTime } from '../lib/utils';

export interface BacktestSelectorProps {
  selectedRunId: string | null;
  onSelect: (runId: string) => void;
}

export function BacktestSelector({ selectedRunId, onSelect }: BacktestSelectorProps) {
  const { data: backtests = [], isLoading } = useQuery({
    queryKey: ['backtests-list'],
    queryFn: async () => {
      const response = await backtestAPI.listBacktests();
      return response;
    },
    refetchInterval: 5000, // Refresh list every 5 seconds
  });

  if (isLoading) {
    return (
      <div className="animate-pulse bg-gray-100 dark:bg-gray-700 h-10 rounded-lg"></div>
    );
  }

  if (backtests.length === 0) {
    return (
      <div className="text-sm text-gray-500 dark:text-gray-400 italic">
        No backtests yet. Run your first backtest to see it here.
      </div>
    );
  }

  const selectedBacktest = backtests.find((b: any) => b.run_id === selectedRunId);

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
        Previous Backtests
      </label>
      <div className="relative">
        <select
          value={selectedRunId || ''}
          onChange={(e) => {
            if (e.target.value) {
              onSelect(e.target.value);
              // Persist to localStorage
              localStorage.setItem('selectedBacktestRunId', e.target.value);
            }
          }}
          className="w-full px-4 py-2 pr-10 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent appearance-none"
        >
          <option value="">Select a backtest...</option>
          {backtests.map((backtest: any) => (
            <option key={backtest.run_id} value={backtest.run_id}>
              {backtest.run_id.slice(0, 24)} - {backtest.status} - {backtest.completed_at ? formatDateTime(backtest.completed_at) : 'Running...'}
            </option>
          ))}
        </select>
        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
      </div>

      {/* Selected Backtest Info */}
      {selectedBacktest && (
        <div className="flex items-center gap-2 p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg text-sm">
          {selectedBacktest.status === 'completed' && (
            <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0" />
          )}
          {selectedBacktest.status === 'running' && (
            <Clock className="w-4 h-4 text-blue-500 animate-spin flex-shrink-0" />
          )}
          {selectedBacktest.status === 'failed' && (
            <XCircle className="w-4 h-4 text-red-500 flex-shrink-0" />
          )}
          <div className="flex-1 min-w-0">
            <div className="font-mono text-xs text-gray-600 dark:text-gray-400 truncate">
              {selectedBacktest.run_id}
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-500">
              Progress: {selectedBacktest.progress?.toFixed(1) || 0}%
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
