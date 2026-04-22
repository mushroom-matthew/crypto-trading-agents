import { Component, type ErrorInfo, type ReactNode, lazy, Suspense, useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BarChart3, Activity, Wallet, Brain, PlayCircle } from 'lucide-react';
import { cn } from './lib/utils';
import { isEnabled, type FeatureFlag } from './lib/features';

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

type TabConfig = {
  id: Tab;
  label: string;
  icon: typeof BarChart3;
  activeClassName: string;
  feature: FeatureFlag;
};

class TabErrorBoundary extends Component<
  { children: ReactNode; tabLabel: string },
  { hasError: boolean; message: string | null }
> {
  constructor(props: { children: ReactNode; tabLabel: string }) {
    super(props);
    this.state = { hasError: false, message: null };
  }

  static getDerivedStateFromError(error: unknown) {
    return {
      hasError: true,
      message: error instanceof Error ? error.message : 'Unknown render error',
    };
  }

  componentDidCatch(error: unknown, info: ErrorInfo) {
    // Keep details in console for debugging while still rendering a fallback.
    console.error('Tab render error', { error, info, tab: this.props.tabLabel });
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700 dark:border-red-800 dark:bg-red-900/20 dark:text-red-300">
          <p className="font-semibold">{this.props.tabLabel} failed to render.</p>
          <p className="mt-1">
            {this.state.message ?? 'Unexpected UI error. Check browser console for details.'}
          </p>
        </div>
      );
    }
    return this.props.children;
  }
}

const TAB_LABELS: Record<Tab, string> = {
  backtest: 'Backtest Control',
  paper: 'Paper Trading',
  live: 'Live Trading Monitor',
  wallets: 'Wallet Reconciliation',
  agents: 'Agent Inspector',
};

const TAB_CONFIG: TabConfig[] = [
  {
    id: 'backtest',
    label: 'Backtest Control',
    icon: BarChart3,
    activeClassName: 'border-blue-500 text-blue-600 dark:text-blue-400',
    feature: 'backtesting',
  },
  {
    id: 'paper',
    label: 'Paper Trading',
    icon: PlayCircle,
    activeClassName: 'border-green-500 text-green-600 dark:text-green-400',
    feature: 'paper_trading',
  },
  {
    id: 'live',
    label: 'Live Trading Monitor',
    icon: Activity,
    activeClassName: 'border-blue-500 text-blue-600 dark:text-blue-400',
    feature: 'live_trading',
  },
  {
    id: 'agents',
    label: 'Agent Inspector',
    icon: Brain,
    activeClassName: 'border-blue-500 text-blue-600 dark:text-blue-400',
    feature: 'agents',
  },
  {
    id: 'wallets',
    label: 'Wallet Reconciliation',
    icon: Wallet,
    activeClassName: 'border-blue-500 text-blue-600 dark:text-blue-400',
    feature: 'wallets',
  },
];

function getDefaultTab(enabledTabs: TabConfig[]): Tab {
  if (enabledTabs.some((tab) => tab.id === 'paper')) {
    return 'paper';
  }
  return enabledTabs[0]?.id ?? 'paper';
}

function App() {
  const enabledTabs = TAB_CONFIG.filter((tab) => isEnabled(tab.feature));
  const [requestedTab, setRequestedTab] = useState<Tab>(() => getDefaultTab(enabledTabs));
  const activeTab = enabledTabs.some((tab) => tab.id === requestedTab)
    ? requestedTab
    : getDefaultTab(enabledTabs);

  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        {/* Tab Navigation */}
        <div className="border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
          <div className="max-w-7xl mx-auto px-6">
            <nav className="flex gap-8">
              {enabledTabs.map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setRequestedTab(tab.id)}
                    className={cn(
                      'flex items-center gap-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors',
                      activeTab === tab.id
                        ? tab.activeClassName
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:hover:text-gray-300'
                    )}
                  >
                    <Icon className="w-5 h-5" />
                    {tab.label}
                  </button>
                );
              })}
            </nav>
          </div>
        </div>

        {/* Tab Content */}
        <div className="max-w-7xl mx-auto px-6 py-6">
          {enabledTabs.length === 0 ? (
            <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-200">
              No UI features are enabled. Set `VITE_FEATURES=paper` or `VITE_FEATURES=all`.
            </div>
          ) : (
            <Suspense fallback={<div className="text-sm text-gray-500">Loading...</div>}>
              <TabErrorBoundary key={activeTab} tabLabel={TAB_LABELS[activeTab]}>
                {activeTab === 'backtest' && <BacktestControl />}
                {activeTab === 'paper' && <PaperTradingControl />}
                {activeTab === 'live' && <LiveTradingMonitor />}
                {activeTab === 'agents' && <AgentInspector />}
                {activeTab === 'wallets' && <WalletReconciliation />}
              </TabErrorBoundary>
            </Suspense>
          )}
        </div>
      </div>
    </QueryClientProvider>
  );
}

export default App;
