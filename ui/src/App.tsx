import { lazy, Suspense, useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BarChart3, Activity, Wallet, Brain, PlayCircle } from 'lucide-react';
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

const BacktestControl = lazy(() =>
  import('./components/BacktestControl').then((module) => ({
    default: module.BacktestControl,
  }))
);
const PaperTradingControl = lazy(() =>
  import('./components/PaperTradingControl').then((module) => ({
    default: module.PaperTradingControl,
  }))
);
const LiveTradingMonitor = lazy(() =>
  import('./components/LiveTradingMonitor').then((module) => ({
    default: module.LiveTradingMonitor,
  }))
);
const AgentInspector = lazy(() =>
  import('./components/AgentInspector').then((module) => ({
    default: module.AgentInspector,
  }))
);
const WalletReconciliation = lazy(() => import('./components/WalletReconciliation'));

type Tab = 'backtest' | 'paper' | 'live' | 'wallets' | 'agents';

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
                onClick={() => setActiveTab('paper')}
                className={cn(
                  'flex items-center gap-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors',
                  activeTab === 'paper'
                    ? 'border-green-500 text-green-600 dark:text-green-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:hover:text-gray-300'
                )}
              >
                <PlayCircle className="w-5 h-5" />
                Paper Trading
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
              <button
                onClick={() => setActiveTab('agents')}
                className={cn(
                  'flex items-center gap-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors',
                  activeTab === 'agents'
                    ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:hover:text-gray-300'
                )}
              >
                <Brain className="w-5 h-5" />
                Agent Inspector
              </button>
              <button
                onClick={() => setActiveTab('wallets')}
                className={cn(
                  'flex items-center gap-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors',
                  activeTab === 'wallets'
                    ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:hover:text-gray-300'
                )}
              >
                <Wallet className="w-5 h-5" />
                Wallet Reconciliation
              </button>
            </nav>
          </div>
        </div>

        {/* Tab Content */}
        <div className="max-w-7xl mx-auto px-6 py-6">
          <Suspense fallback={<div className="text-sm text-gray-500">Loading...</div>}>
            {activeTab === 'backtest' && <BacktestControl />}
            {activeTab === 'paper' && <PaperTradingControl />}
            {activeTab === 'live' && <LiveTradingMonitor />}
            {activeTab === 'agents' && <AgentInspector />}
            {activeTab === 'wallets' && <WalletReconciliation />}
          </Suspense>
        </div>
      </div>
    </QueryClientProvider>
  );
}

export default App;
