import { useQuery } from '@tanstack/react-query';
import { TrendingUp, TrendingDown, Activity } from 'lucide-react';
import { marketAPI, type MarketTick } from '../lib/api';
import { cn, formatCurrency, formatDateTime } from '../lib/utils';
import { useState, useEffect, useCallback } from 'react';
import { useWebSocket, type WebSocketMessage } from '../hooks/useWebSocket';

export function MarketTicker() {
  const [priceChanges, setPriceChanges] = useState<Record<string, number>>({});
  const [wsTicks, setWsTicks] = useState<MarketTick[]>([]);

  // Initial data load with query (fallback)
  const { data: initialTicks = [], isLoading } = useQuery({
    queryKey: ['market-ticks'],
    queryFn: () => marketAPI.getTicks(undefined, 20),
    refetchInterval: 5000, // Reduced frequency since WebSocket provides real-time updates
  });

  // WebSocket connection for real-time ticks
  const wsUrl = `ws://${window.location.hostname}:8081/ws/market`;

  const handleWebSocketMessage = useCallback((message: WebSocketMessage) => {
    // Only process tick events
    if (message.type === 'tick' || message.type === 'market_tick') {
      const tick: MarketTick = {
        symbol: message.payload.symbol,
        price: message.payload.price,
        volume: message.payload.volume,
        timestamp: message.timestamp,
        source: message.source,
      };

      setWsTicks((prev) => {
        // Add new tick and keep last 20 ticks
        const updated = [tick, ...prev].slice(0, 20);
        return updated;
      });
    }
  }, []);

  const { isConnected } = useWebSocket(wsUrl, {}, handleWebSocketMessage);

  // Merge WebSocket ticks with initial query data
  const ticks = wsTicks.length > 0 ? wsTicks : initialTicks;

  // Track price changes for color coding
  useEffect(() => {
    if (ticks.length > 0) {
      const newChanges: Record<string, number> = {};

      ticks.forEach((tick) => {
        const prevPrice = priceChanges[tick.symbol];
        if (prevPrice !== undefined) {
          newChanges[tick.symbol] = tick.price - prevPrice;
        } else {
          newChanges[tick.symbol] = 0;
        }
      });

      setPriceChanges((prev) => ({
        ...prev,
        ...Object.fromEntries(ticks.map((t) => [t.symbol, t.price])),
      }));
    }
  }, [ticks]);

  // Group ticks by symbol (show most recent for each)
  const latestTicks = ticks.reduce((acc, tick) => {
    if (!acc[tick.symbol] || new Date(tick.timestamp) > new Date(acc[tick.symbol].timestamp)) {
      acc[tick.symbol] = tick;
    }
    return acc;
  }, {} as Record<string, MarketTick>);

  const ticksArray = Object.values(latestTicks);

  if (isLoading && ticksArray.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 py-2 px-6">
        <div className="flex items-center gap-2 text-sm text-gray-500">
          <Activity className="w-4 h-4 animate-pulse" />
          <span>Loading market data...</span>
        </div>
      </div>
    );
  }

  if (ticksArray.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 py-2 px-6">
        <div className="flex items-center gap-2 text-sm text-gray-500">
          <Activity className="w-4 h-4" />
          <span>No market data available</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-gray-800 dark:to-gray-900 border-b border-gray-200 dark:border-gray-700">
      <div className="max-w-7xl mx-auto px-6 py-3">
        <div className="flex items-center gap-6 overflow-x-auto">
          {/* Live indicator */}
          <div className="flex items-center gap-2 flex-shrink-0">
            <Activity
              className={cn(
                'w-4 h-4',
                isConnected ? 'animate-pulse text-green-500' : 'text-yellow-500'
              )}
            />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              {isConnected ? 'Live Market (WebSocket)' : 'Market Data (Polling)'}
            </span>
          </div>

          {/* Price tickers */}
          <div className="flex gap-6 overflow-x-auto scrollbar-hide">
            {ticksArray.map((tick) => {
              const change = priceChanges[tick.symbol] || 0;
              const isPositive = change > 0;
              const isNegative = change < 0;

              return (
                <div
                  key={tick.symbol}
                  className="flex items-center gap-3 flex-shrink-0 px-3 py-1 rounded-lg bg-white dark:bg-gray-800/50 shadow-sm"
                >
                  {/* Symbol */}
                  <span className="font-mono font-bold text-sm text-gray-900 dark:text-gray-100">
                    {tick.symbol}
                  </span>

                  {/* Price */}
                  <div className="flex items-center gap-1">
                    <span
                      className={cn(
                        'font-mono font-semibold text-lg',
                        isPositive && 'text-green-600 dark:text-green-400',
                        isNegative && 'text-red-600 dark:text-red-400',
                        !isPositive && !isNegative && 'text-gray-900 dark:text-gray-100'
                      )}
                    >
                      {formatCurrency(tick.price)}
                    </span>
                    {isPositive && <TrendingUp className="w-4 h-4 text-green-500" />}
                    {isNegative && <TrendingDown className="w-4 h-4 text-red-500" />}
                  </div>

                  {/* Volume (if available) */}
                  {tick.volume !== undefined && tick.volume !== null && (
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      Vol: {tick.volume.toFixed(2)}
                    </span>
                  )}

                  {/* Timestamp */}
                  <span className="text-xs text-gray-400 dark:text-gray-500">
                    {new Date(tick.timestamp).toLocaleTimeString()}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
