import { useQuery } from '@tanstack/react-query';
import { Activity, Clock, Brain, CheckCircle } from 'lucide-react';
import { backtestAPI } from '../lib/api';

export interface LiveProgressMonitorProps {
  runId: string;
  status: string;
}

export function LiveProgressMonitor({ runId, status }: LiveProgressMonitorProps) {
  const { data: progress } = useQuery({
    queryKey: ['backtest-progress', runId],
    queryFn: async () => {
      const response = await backtestAPI.getProgress(runId);
      return response.data;
    },
    enabled: status === 'running',
    refetchInterval: status === 'running' ? 1000 : false, // Poll every second when running
  });

  if (status !== 'running' || !progress) {
    return null;
  }

  return (
    <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-6 border border-blue-200 dark:border-blue-800">
      {/* Header */}
      <div className="flex items-center gap-3 mb-4">
        <Activity className="w-6 h-6 text-blue-500 animate-pulse" />
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
          Backtest Running - {progress.current_phase}
        </h3>
        {progress.strategy === 'llm_strategist' && (
          <span className="ml-auto text-xs px-2 py-1 bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400 rounded">
            AI Strategy
          </span>
        )}
      </div>

      {/* Progress Bar */}
      <div className="mb-4">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm text-gray-600 dark:text-gray-400">Progress</span>
          <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            {progress.progress_pct.toFixed(1)}%
          </span>
        </div>
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 overflow-hidden">
          <div
            className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-500 ease-out"
            style={{ width: `${progress.progress_pct}%` }}
          />
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
        {progress.total_candles && (
          <div className="p-3 bg-white dark:bg-gray-800 rounded-lg">
            <div className="flex items-center gap-2 mb-1">
              <Clock className="w-3 h-3 text-gray-500" />
              <span className="text-xs text-gray-500 dark:text-gray-400">Candles</span>
            </div>
            <div className="text-sm font-semibold text-gray-900 dark:text-gray-100">
              {progress.processed_candles || 0} / {progress.total_candles}
            </div>
          </div>
        )}

        {progress.strategy === 'llm_strategist' && (
          <>
            <div className="p-3 bg-white dark:bg-gray-800 rounded-lg">
              <div className="flex items-center gap-2 mb-1">
                <Brain className="w-3 h-3 text-purple-500" />
                <span className="text-xs text-gray-500 dark:text-gray-400">AI Plans</span>
              </div>
              <div className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                {progress.plans_generated}
              </div>
            </div>

            <div className="p-3 bg-white dark:bg-gray-800 rounded-lg">
              <div className="flex items-center gap-2 mb-1">
                <CheckCircle className="w-3 h-3 text-green-500" />
                <span className="text-xs text-gray-500 dark:text-gray-400">LLM Calls</span>
              </div>
              <div className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                {progress.llm_calls_made}
              </div>
            </div>
          </>
        )}

        {progress.current_timestamp && (
          <div className="p-3 bg-white dark:bg-gray-800 rounded-lg">
            <div className="flex items-center gap-2 mb-1">
              <Clock className="w-3 h-3 text-blue-500" />
              <span className="text-xs text-gray-500 dark:text-gray-400">Current Time</span>
            </div>
            <div className="text-xs font-mono text-gray-900 dark:text-gray-100">
              {new Date(progress.current_timestamp).toLocaleDateString()}
            </div>
          </div>
        )}
      </div>

      {/* Recent Logs */}
      {progress.latest_logs && progress.latest_logs.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold mb-2 text-gray-700 dark:text-gray-300">
            Recent Activity
          </h4>
          <div className="space-y-1 max-h-32 overflow-y-auto">
            {progress.latest_logs.slice().reverse().map((log: any, idx: number) => (
              <div
                key={idx}
                className="text-xs font-mono p-2 bg-white dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-700"
              >
                <span className="text-gray-500 dark:text-gray-400">
                  {new Date(log.timestamp).toLocaleTimeString()}
                </span>
                {' - '}
                <span className="text-gray-700 dark:text-gray-300">{log.message}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
