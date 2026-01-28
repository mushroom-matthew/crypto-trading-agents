import { useState, useCallback, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { TrendingUp, TrendingDown, AlertCircle, DollarSign, PieChart, Activity, Radio } from 'lucide-react';
import { cn, formatCurrency, formatDateTime } from '../lib/utils';
import { MarketTicker } from './MarketTicker';
import { EventTimeline } from './EventTimeline';
import { useWebSocket, type WebSocketMessage } from '../hooks/useWebSocket';
import { buildWebSocketUrl } from '../lib/websocket';

// API types (matching backend schemas)
interface Position {
  symbol: string;
  qty: number;
  avg_entry_price: number;
  mark_price?: number;
  unrealized_pnl?: number;
  timestamp: string;
}

interface PortfolioSummary {
  cash: number;
  equity: number;
  day_pnl?: number;
  total_pnl?: number;
  positions_count: number;
  updated_at: string;
}

interface Fill {
  order_id: string;
  symbol: string;
  side: string;
  qty: number;
  price: number;
  timestamp: string;
  run_id?: string;
  correlation_id?: string;
  // Risk stats (Phase 6 trade-level visibility)
  fee?: number;
  pnl?: number;
  trigger_id?: string;
  risk_used_abs?: number;
  actual_risk_at_stop?: number;
  stop_distance?: number;
  r_multiple?: number;
}

interface BlockEvent {
  timestamp: string;
  symbol: string;
  side: string;
  qty: number;
  reason: string;
  detail?: string;
  trigger_id: string;
  correlation_id?: string;
}

interface RiskBudget {
  date: string;
  budget_total: number;
  budget_used: number;
  budget_available: number;
  utilization_pct: number;
  allocations: any[];
}

interface BlockReason {
  reason: string;
  count: number;
}

// API client
const liveAPI = {
  getPositions: async (): Promise<Position[]> => {
    const response = await fetch('/live/positions');
    return response.json();
  },
  getPortfolio: async (): Promise<PortfolioSummary> => {
    const response = await fetch('/live/portfolio');
    return response.json();
  },
  getFills: async (limit = 50): Promise<Fill[]> => {
    const response = await fetch(`/live/fills?limit=${limit}`);
    return response.json();
  },
  getBlocks: async (limit = 100): Promise<BlockEvent[]> => {
    const response = await fetch(`/live/blocks?limit=${limit}`);
    return response.json();
  },
  getRiskBudget: async (): Promise<RiskBudget> => {
    const response = await fetch('/live/risk-budget');
    return response.json();
  },
  getBlockReasons: async (): Promise<{ run_id: string; reasons: BlockReason[] }> => {
    const response = await fetch('/live/block-reasons');
    return response.json();
  },
};

export function LiveTradingMonitor() {
  // WebSocket state for real-time updates
  const [liveFills, setLiveFills] = useState<Fill[]>([]);
  const [livePositions, setLivePositions] = useState<Position[]>([]);
  const [liveBlocks, setLiveBlocks] = useState<BlockEvent[]>([]);
  const [livePortfolio, setLivePortfolio] = useState<PortfolioSummary | null>(null);

  // WebSocket connection
  const wsUrl = buildWebSocketUrl('/ws/live');

  // Handle WebSocket messages
  const handleWebSocketMessage = useCallback((message: WebSocketMessage) => {
    switch (message.type) {
      case 'fill':
        const fill: Fill = {
          order_id: message.payload.order_id,
          symbol: message.payload.symbol,
          side: message.payload.side,
          qty: message.payload.qty,
          price: message.payload.price,
          timestamp: message.payload.timestamp || new Date().toISOString(),
          run_id: message.payload.run_id,
          correlation_id: message.payload.correlation_id,
        };
        setLiveFills((prev) => [fill, ...prev].slice(0, 50));
        break;

      case 'position_update':
        const position: Position = {
          symbol: message.payload.symbol,
          qty: message.payload.qty,
          avg_entry_price: message.payload.avg_entry_price,
          mark_price: message.payload.mark_price,
          unrealized_pnl: message.payload.unrealized_pnl,
          timestamp: message.payload.timestamp || new Date().toISOString(),
        };
        setLivePositions((prev) => {
          const filtered = prev.filter(p => p.symbol !== position.symbol);
          return position.qty === 0 ? filtered : [position, ...filtered];
        });
        break;

      case 'trade_blocked':
        const block: BlockEvent = {
          timestamp: message.payload.timestamp || new Date().toISOString(),
          symbol: message.payload.symbol,
          side: message.payload.side,
          qty: message.payload.qty,
          reason: message.payload.reason,
          detail: message.payload.detail,
          trigger_id: message.payload.trigger_id,
          correlation_id: message.payload.correlation_id,
        };
        setLiveBlocks((prev) => [block, ...prev].slice(0, 100));
        break;

      case 'portfolio_update':
        setLivePortfolio({
          cash: message.payload.cash,
          equity: message.payload.equity,
          day_pnl: message.payload.day_pnl,
          total_pnl: message.payload.total_pnl,
          positions_count: message.payload.positions_count,
          updated_at: message.payload.updated_at || new Date().toISOString(),
        });
        break;
    }
  }, []);

  const { isConnected } = useWebSocket(wsUrl, {}, handleWebSocketMessage);

  // Query all live data with auto-refresh (reduced intervals since WebSocket provides real-time)
  const { data: portfolio } = useQuery({
    queryKey: ['portfolio'],
    queryFn: liveAPI.getPortfolio,
    refetchInterval: isConnected ? 30000 : 5000, // 30s with WS, 5s without
  });

  const { data: positions = [] } = useQuery({
    queryKey: ['positions'],
    queryFn: liveAPI.getPositions,
    refetchInterval: isConnected ? 30000 : 3000, // 30s with WS, 3s without
  });

  const { data: fills = [] } = useQuery({
    queryKey: ['fills'],
    queryFn: () => liveAPI.getFills(50),
    refetchInterval: isConnected ? 30000 : 2000, // 30s with WS, 2s without
  });

  const { data: blocks = [] } = useQuery({
    queryKey: ['blocks'],
    queryFn: () => liveAPI.getBlocks(100),
    refetchInterval: isConnected ? 30000 : 5000, // 30s with WS, 5s without
  });

  const { data: riskBudget } = useQuery({
    queryKey: ['risk-budget'],
    queryFn: liveAPI.getRiskBudget,
    refetchInterval: 10000,
  });

  const { data: blockReasons } = useQuery({
    queryKey: ['block-reasons'],
    queryFn: liveAPI.getBlockReasons,
    refetchInterval: 10000,
  });
  const liveRunId = blockReasons?.run_id;

  // Merge WebSocket data with polling data
  const displayPortfolio = livePortfolio || portfolio;
  const displayPositions = useMemo(() => {
    if (livePositions.length > 0) {
      const merged = [...livePositions];
      positions.forEach(pos => {
        if (!merged.some(p => p.symbol === pos.symbol)) {
          merged.push(pos);
        }
      });
      return merged;
    }
    return positions;
  }, [livePositions, positions]);

  const displayFills = useMemo(() => {
    if (liveFills.length > 0) {
      const merged = [...liveFills];
      fills.forEach(fill => {
        if (!merged.some(f => f.order_id === fill.order_id)) {
          merged.push(fill);
        }
      });
      return merged.slice(0, 50);
    }
    return fills;
  }, [liveFills, fills]);

  const displayBlocks = useMemo(() => {
    if (liveBlocks.length > 0) {
      const merged = [...liveBlocks];
      blocks.forEach(block => {
        if (!merged.some(b => b.trigger_id === block.trigger_id)) {
          merged.push(block);
        }
      });
      return merged.slice(0, 100);
    }
    return blocks;
  }, [liveBlocks, blocks]);

  return (
    <div className="space-y-6">
      {/* Market Ticker */}
      <MarketTicker />

      <div className="p-6 space-y-6 max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Live Trading Monitor</h1>
            <p className="text-gray-500 dark:text-gray-400 mt-1">
              Real-time positions, fills, and risk monitoring
            </p>
        </div>
        <div className="flex items-center gap-2 text-sm">
          {isConnected ? (
            <>
              <Radio className="w-4 h-4 animate-pulse text-green-500" />
              <span className="text-green-600 dark:text-green-400 font-medium">Live (WebSocket)</span>
            </>
          ) : (
            <>
              <Activity className="w-4 h-4 text-yellow-500" />
              <span className="text-yellow-600 dark:text-yellow-400">Live (Polling)</span>
            </>
          )}
        </div>
      </div>

      {/* Portfolio Summary Cards */}
      {displayPortfolio && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <SummaryCard
            icon={<DollarSign className="w-5 h-5" />}
            label="Cash"
            value={formatCurrency(displayPortfolio.cash)}
            iconColor="text-blue-500"
          />
          <SummaryCard
            icon={<PieChart className="w-5 h-5" />}
            label="Equity"
            value={formatCurrency(displayPortfolio.equity)}
            iconColor="text-purple-500"
          />
          <SummaryCard
            icon={displayPortfolio.day_pnl && displayPortfolio.day_pnl >= 0 ? <TrendingUp className="w-5 h-5" /> : <TrendingDown className="w-5 h-5" />}
            label="Day P&L"
            value={displayPortfolio.day_pnl !== undefined ? formatCurrency(displayPortfolio.day_pnl) : 'N/A'}
            iconColor={displayPortfolio.day_pnl && displayPortfolio.day_pnl >= 0 ? 'text-green-500' : 'text-red-500'}
          />
          <SummaryCard
            icon={<Activity className="w-5 h-5" />}
            label="Positions"
            value={displayPortfolio.positions_count.toString()}
            iconColor="text-orange-500"
          />
        </div>
      )}

      {/* Risk Budget Gauge */}
      {riskBudget && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Daily Risk Budget</h2>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-500">Utilization</span>
                <span className="font-semibold">{riskBudget.utilization_pct.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4">
                <div
                  className={cn(
                    'h-4 rounded-full transition-all duration-300',
                    riskBudget.utilization_pct < 50 && 'bg-green-500',
                    riskBudget.utilization_pct >= 50 && riskBudget.utilization_pct < 80 && 'bg-yellow-500',
                    riskBudget.utilization_pct >= 80 && 'bg-red-500'
                  )}
                  style={{ width: `${Math.min(riskBudget.utilization_pct, 100)}%` }}
                />
              </div>
            </div>
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div>
                <p className="text-gray-500">Total Budget</p>
                <p className="font-semibold">{formatCurrency(riskBudget.budget_total)}</p>
              </div>
              <div>
                <p className="text-gray-500">Used</p>
                <p className="font-semibold">{formatCurrency(riskBudget.budget_used)}</p>
              </div>
              <div>
                <p className="text-gray-500">Available</p>
                <p className="font-semibold">{formatCurrency(riskBudget.budget_available)}</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Positions Table */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Current Positions</h2>
        {displayPositions.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <Activity className="w-12 h-12 mx-auto mb-2 opacity-50" />
            <p>No open positions</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-sm text-gray-500 border-b border-gray-200 dark:border-gray-700">
                  <th className="pb-3">Symbol</th>
                  <th className="pb-3 text-right">Quantity</th>
                  <th className="pb-3 text-right">Avg Entry</th>
                  <th className="pb-3 text-right">Mark Price</th>
                  <th className="pb-3 text-right">Unrealized P&L</th>
                  <th className="pb-3 text-right">Updated</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {displayPositions.map((pos, idx) => (
                  <tr key={idx} className="text-sm">
                    <td className="py-3 font-mono font-semibold">{pos.symbol}</td>
                    <td className="py-3 text-right">{pos.qty.toFixed(8)}</td>
                    <td className="py-3 text-right">{formatCurrency(pos.avg_entry_price)}</td>
                    <td className="py-3 text-right">{pos.mark_price ? formatCurrency(pos.mark_price) : 'N/A'}</td>
                    <td className={cn('py-3 text-right font-semibold', pos.unrealized_pnl && pos.unrealized_pnl >= 0 ? 'text-green-500' : 'text-red-500')}>
                      {pos.unrealized_pnl ? formatCurrency(pos.unrealized_pnl) : 'N/A'}
                    </td>
                    <td className="py-3 text-right text-gray-500">{formatDateTime(pos.timestamp)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Fills */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Recent Fills</h2>
          {displayFills.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <p>No recent fills</p>
            </div>
          ) : (
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {displayFills.slice(0, 10).map((fill, idx) => (
                <div key={idx} className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg text-sm">
                  <div className="flex justify-between items-center">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-semibold">{fill.symbol}</span>
                        <span className={cn('px-2 py-0.5 rounded text-xs font-semibold', fill.side === 'BUY' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400')}>
                          {fill.side}
                        </span>
                        {fill.trigger_id && (
                          <span className="text-xs text-gray-500 truncate max-w-24" title={fill.trigger_id}>
                            {fill.trigger_id}
                          </span>
                        )}
                      </div>
                      <div className="text-gray-500 text-xs mt-1">{formatDateTime(fill.timestamp)}</div>
                    </div>
                    <div className="text-right">
                      <div className="font-semibold">{fill.qty.toFixed(8)}</div>
                      <div className="text-gray-500 text-xs">@ {formatCurrency(fill.price)}</div>
                    </div>
                  </div>
                  {/* Risk stats row */}
                  {(fill.pnl != null || fill.risk_used_abs != null || fill.r_multiple != null) && (
                    <div className="flex gap-4 mt-2 pt-2 border-t border-gray-200 dark:border-gray-600 text-xs">
                      {fill.pnl != null && (
                        <div>
                          <span className="text-gray-500">P&L: </span>
                          <span className={cn('font-semibold', fill.pnl >= 0 ? 'text-green-500' : 'text-red-500')}>
                            {formatCurrency(fill.pnl)}
                          </span>
                        </div>
                      )}
                      {fill.risk_used_abs != null && (
                        <div>
                          <span className="text-gray-500">Risk: </span>
                          <span className="font-semibold">{formatCurrency(fill.risk_used_abs)}</span>
                        </div>
                      )}
                      {fill.r_multiple != null && (
                        <div>
                          <span className="text-gray-500">R: </span>
                          <span className={cn('font-semibold', fill.r_multiple >= 0 ? 'text-green-500' : 'text-red-500')}>
                            {fill.r_multiple.toFixed(2)}
                          </span>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Block Events */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Block Events</h2>
          {displayBlocks.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <p>No blocked trades</p>
            </div>
          ) : (
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {displayBlocks.slice(0, 10).map((block, idx) => (
                <div key={idx} className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-sm">
                  <div className="flex items-start gap-2">
                    <AlertCircle className="w-4 h-4 text-red-500 mt-0.5 flex-shrink-0" />
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-semibold">{block.symbol}</span>
                        <span className="text-gray-500">{block.side}</span>
                        <span className="text-gray-500">{block.qty.toFixed(8)}</span>
                      </div>
                      <div className="text-red-600 dark:text-red-400 font-semibold mt-1">{block.reason}</div>
                      {block.detail && <div className="text-gray-500 text-xs mt-1">{block.detail}</div>}
                      <div className="text-gray-500 text-xs mt-1">{formatDateTime(block.timestamp)}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Block Reasons Summary */}
      {blockReasons && blockReasons.reasons.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Block Reasons Summary</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {blockReasons.reasons.map((reason, idx) => (
              <div key={idx} className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                <p className="text-sm text-gray-500 mb-1">{reason.reason}</p>
                <p className="text-2xl font-bold">{reason.count}</p>
              </div>
            ))}
          </div>
        </div>
      )}

        {/* Event Timeline */}
        {liveRunId ? (
          <EventTimeline limit={30} runId={liveRunId} />
        ) : (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 text-sm text-gray-500 dark:text-gray-400">
            Waiting for an active live run to show the event timeline.
          </div>
        )}
      </div>
    </div>
  );
}

// Helper component for summary cards
interface SummaryCardProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  iconColor?: string;
}

function SummaryCard({ icon, label, value, iconColor = 'text-gray-500' }: SummaryCardProps) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
      <div className="flex items-center gap-3">
        <div className={cn('p-3 rounded-lg bg-gray-100 dark:bg-gray-700', iconColor)}>{icon}</div>
        <div>
          <p className="text-sm text-gray-500 dark:text-gray-400">{label}</p>
          <p className="text-2xl font-bold">{value}</p>
        </div>
      </div>
    </div>
  );
}
