import { useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BacktestControl } from './components/BacktestControl';
import { LiveTradingMonitor } from './components/LiveTradingMonitor';
import { BarChart3, Activity } from 'lucide-react';
import { cn } from './lib/utils';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

type Tab = 'backtest' | 'live';

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('backtest');

  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        {/* Tab Navigation */}
        <div className="border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
          <div className="max-w-7xl mx-auto px-6">
            <nav className="flex gap-8">
              <button
                onClick={() => setActiveTab('backtest')}
                className={cn(
                  'flex items-center gap-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors',
                  activeTab === 'backtest'
                    ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:hover:text-gray-300'
                )}
              >
                <BarChart3 className="w-5 h-5" />
                Backtest Control
              </button>
              <button
                onClick={() => setActiveTab('live')}
                className={cn(
                  'flex items-center gap-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors',
                  activeTab === 'live'
                    ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:hover:text-gray-300'
                )}
              >
                <Activity className="w-5 h-5" />
                Live Trading Monitor
              </button>
            </nav>
          </div>
        </div>

        {/* Tab Content */}
        <div>
          {activeTab === 'backtest' && <BacktestControl />}
          {activeTab === 'live' && <LiveTradingMonitor />}
        </div>
      </div>
    </QueryClientProvider>
  );
}

export default App;
