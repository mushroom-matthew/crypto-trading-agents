import { useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { backtestAPI } from '../lib/api';
import { TimelinePlayer } from './TimelinePlayer';
import { CandlestickChart } from './CandlestickChart';
import { IndicatorPanels } from './IndicatorPanels';
import { StateInspector } from './StateInspector';

export interface BacktestPlaybackViewerProps {
  runId: string;
  symbol: string;
}

export function BacktestPlaybackViewer({
  runId,
  symbol,
}: BacktestPlaybackViewerProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [mode, setMode] = useState<'candle' | 'event'>('candle');

  // Load candles with indicators (progressive loading)
  const { data: candles = [], isLoading: loadingCandles } = useQuery({
    queryKey: ['playback-candles', runId, symbol],
    queryFn: () => backtestAPI.getPlaybackCandles(runId, symbol, 0, 2000),
    staleTime: Infinity, // Cache indefinitely for playback
  });

  // Load trade events
  const { data: events = [], isLoading: loadingEvents } = useQuery({
    queryKey: ['playback-events', runId, symbol],
    queryFn: () => backtestAPI.getPlaybackEvents(runId, undefined, symbol),
    staleTime: Infinity,
  });

  // Load trades for chart annotations
  const { data: trades = [] } = useQuery({
    queryKey: ['playback-trades', runId],
    queryFn: () => backtestAPI.getTrades(runId, 1000),
    staleTime: Infinity,
  });

  useEffect(() => {
    setCurrentIndex(0);
  }, [symbol]);

  const filteredTrades = useMemo(() => {
    if (!symbol) {
      return trades;
    }
    return trades.filter((trade) => trade.symbol === symbol);
  }, [symbol, trades]);

  // Get current timestamp based on mode and index
  const currentTimestamp = useMemo(() => {
    if (mode === 'candle') {
      return candles[currentIndex]?.timestamp || '';
    } else {
      return events[currentIndex]?.timestamp || '';
    }
  }, [mode, currentIndex, candles, events]);

  const handleTimestampChange = (newIndex: number, _timestamp?: string) => {
    setCurrentIndex(newIndex);
  };

  const isLoading = loadingCandles || loadingEvents;

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-500">Loading backtest data...</p>
        </div>
      </div>
    );
  }

  if (candles.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        <p>No candle data available for this backtest</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Timeline Player Controls */}
      <TimelinePlayer
        totalCandles={candles.length}
        events={events}
        onTimestampChange={handleTimestampChange}
        currentIndex={currentIndex}
        mode={mode}
        onModeChange={setMode}
      />

      {/* Main Content: Charts + State Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Charts Section (2/3 width on large screens) */}
        <div className="lg:col-span-2 space-y-6">
          {/* Candlestick Chart */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">
              Price Chart
            </h3>
            <CandlestickChart
              candles={candles}
              trades={filteredTrades}
              currentIndex={currentIndex}
            />
          </div>

          {/* Indicator Panels */}
          <IndicatorPanels candles={candles} currentIndex={currentIndex} />
        </div>

        {/* State Inspector (1/3 width, sticky on large screens) */}
        <div className="lg:col-span-1">
          <StateInspector
            runId={runId}
            currentTimestamp={currentTimestamp}
            enabled={!!currentTimestamp}
          />
        </div>
      </div>

      {/* Info Panel */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <div className="flex-shrink-0 mt-0.5">
            <svg className="w-5 h-5 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div className="flex-1 text-sm text-blue-800 dark:text-blue-200">
            <p className="font-semibold mb-1">Interactive Playback Mode</p>
            <p>Use the timeline controls to step through the backtest. The charts show data up to the current position, and the state inspector displays portfolio metrics at each point in time.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
