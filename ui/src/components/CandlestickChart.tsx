import { useState, useMemo, useCallback } from 'react';
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Brush,
} from 'recharts';
import { ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';
import { formatCurrency, formatDateTime } from '../lib/utils';
import type { CandleWithIndicators, BacktestTrade } from '../lib/api';

export interface CandlestickChartProps {
  candles: CandleWithIndicators[];
  trades?: BacktestTrade[];
  currentIndex: number;
}

type OverlayType = 'sma_20' | 'sma_50' | 'ema_20' | 'bb';

export function CandlestickChart({
  candles,
  trades = [],
  currentIndex,
}: CandlestickChartProps) {
  const [selectedOverlays, setSelectedOverlays] = useState<OverlayType[]>(['sma_20', 'sma_50']);

  // Zoom state - indices into chartData
  const [zoomRange, setZoomRange] = useState<{ startIndex: number; endIndex: number } | null>(null);

  // Transform candle data for Recharts (workaround for no native candlestick)
  const chartData = useMemo(() => {
    const visibleCandles = candles.slice(0, currentIndex + 1);

    return visibleCandles.map((candle, idx) => {
      const isGreen = candle.close >= candle.open;

      return {
        index: idx,
        timestamp: candle.timestamp,
        // Candlestick body (area between open and close)
        bodyLow: Math.min(candle.open, candle.close),
        bodyHigh: Math.max(candle.open, candle.close),
        // Candlestick wicks
        wickLow: candle.low,
        wickHigh: candle.high,
        isGreen,
        // Original OHLC for tooltip
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
        volume: candle.volume,
        // Indicators
        sma_20: candle.sma_20,
        sma_50: candle.sma_50,
        ema_20: candle.ema_20,
        bb_upper: candle.bb_upper,
        bb_middle: candle.bb_middle,
        bb_lower: candle.bb_lower,
      };
    });
  }, [candles, currentIndex]);

  // Find trades that occurred in visible candles
  const visibleTrades = useMemo(() => {
    if (!trades.length || !chartData.length) return [];

    const firstTimestamp = new Date(chartData[0].timestamp).getTime();
    const lastTimestamp = new Date(chartData[chartData.length - 1].timestamp).getTime();

    return trades.filter((trade) => {
      const tradeTime = new Date(trade.timestamp).getTime();
      return tradeTime >= firstTimestamp && tradeTime <= lastTimestamp;
    });
  }, [trades, chartData]);

  const toggleOverlay = (overlay: OverlayType) => {
    setSelectedOverlays((prev) =>
      prev.includes(overlay)
        ? prev.filter((o) => o !== overlay)
        : [...prev, overlay]
    );
  };

  // Zoom control handlers
  const handleBrushChange = useCallback((range: { startIndex?: number; endIndex?: number }) => {
    if (range.startIndex !== undefined && range.endIndex !== undefined) {
      setZoomRange({ startIndex: range.startIndex, endIndex: range.endIndex });
    }
  }, []);

  const handleZoomIn = useCallback(() => {
    const total = chartData.length;
    if (total === 0) return;

    const currentStart = zoomRange?.startIndex ?? 0;
    const currentEnd = zoomRange?.endIndex ?? total - 1;
    const visibleCount = currentEnd - currentStart + 1;

    // Zoom in by 25% on each side (minimum 20 candles visible)
    const zoomAmount = Math.max(1, Math.floor(visibleCount * 0.125));
    const newStart = Math.min(currentStart + zoomAmount, currentEnd - 19);
    const newEnd = Math.max(currentEnd - zoomAmount, newStart + 19);

    setZoomRange({ startIndex: newStart, endIndex: newEnd });
  }, [chartData.length, zoomRange]);

  const handleZoomOut = useCallback(() => {
    const total = chartData.length;
    if (total === 0) return;

    const currentStart = zoomRange?.startIndex ?? 0;
    const currentEnd = zoomRange?.endIndex ?? total - 1;
    const visibleCount = currentEnd - currentStart + 1;

    // Zoom out by 25% on each side
    const zoomAmount = Math.max(1, Math.floor(visibleCount * 0.25));
    const newStart = Math.max(0, currentStart - zoomAmount);
    const newEnd = Math.min(total - 1, currentEnd + zoomAmount);

    // If we're at full range, clear zoom
    if (newStart === 0 && newEnd === total - 1) {
      setZoomRange(null);
    } else {
      setZoomRange({ startIndex: newStart, endIndex: newEnd });
    }
  }, [chartData.length, zoomRange]);

  const handleResetZoom = useCallback(() => {
    setZoomRange(null);
  }, []);

  // Compute visible range info for display
  const zoomInfo = useMemo(() => {
    const total = chartData.length;
    const visibleStart = zoomRange?.startIndex ?? 0;
    const visibleEnd = zoomRange?.endIndex ?? total - 1;
    const visibleCount = visibleEnd - visibleStart + 1;
    const zoomPercent = total > 0 ? Math.round((visibleCount / total) * 100) : 100;
    return { total, visibleCount, zoomPercent };
  }, [chartData.length, zoomRange]);

  const overlayConfig = {
    sma_20: { color: '#3b82f6', name: 'SMA 20' },
    sma_50: { color: '#8b5cf6', name: 'SMA 50' },
    ema_20: { color: '#10b981', name: 'EMA 20' },
    bb: { color: '#6b7280', name: 'Bollinger Bands' },
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload || !payload.length) return null;

    const data = payload[0].payload;

    return (
      <div className="bg-white dark:bg-gray-800 p-3 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
        <p className="text-xs text-gray-500 mb-2">{formatDateTime(data.timestamp)}</p>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
          <span className="text-gray-500">Open:</span>
          <span className="font-mono">{formatCurrency(data.open)}</span>
          <span className="text-gray-500">High:</span>
          <span className="font-mono text-green-500">{formatCurrency(data.high)}</span>
          <span className="text-gray-500">Low:</span>
          <span className="font-mono text-red-500">{formatCurrency(data.low)}</span>
          <span className="text-gray-500">Close:</span>
          <span className={`font-mono ${data.isGreen ? 'text-green-500' : 'text-red-500'}`}>
            {formatCurrency(data.close)}
          </span>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-4">
      {/* Control Bar */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        {/* Overlay Selector */}
        <div className="flex items-center gap-4 flex-wrap">
          <span className="text-sm text-gray-500">Overlays:</span>
          {Object.entries(overlayConfig).map(([key, config]) => (
            <label key={key} className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={selectedOverlays.includes(key as OverlayType)}
                onChange={() => toggleOverlay(key as OverlayType)}
                className="rounded border-gray-300 text-blue-500 focus:ring-blue-500"
              />
              <span className="text-sm flex items-center gap-1">
                <span
                  className="w-3 h-3 rounded"
                  style={{ backgroundColor: config.color }}
                />
                {config.name}
              </span>
            </label>
          ))}
        </div>

        {/* Zoom Controls */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">
            {zoomInfo.visibleCount} / {zoomInfo.total} candles ({zoomInfo.zoomPercent}%)
          </span>
          <div className="flex items-center gap-1 border border-gray-300 dark:border-gray-600 rounded-lg p-1">
            <button
              onClick={handleZoomIn}
              className="p-1.5 rounded hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Zoom In (show fewer candles)"
              disabled={zoomInfo.visibleCount <= 20}
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            <button
              onClick={handleZoomOut}
              className="p-1.5 rounded hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Zoom Out (show more candles)"
              disabled={zoomInfo.zoomPercent >= 100}
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <button
              onClick={handleResetZoom}
              className="p-1.5 rounded hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Reset Zoom (show all candles)"
              disabled={!zoomRange}
            >
              <Maximize2 className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={450}>
        <ComposedChart
          data={chartData}
          margin={{ top: 10, right: 30, left: 0, bottom: 40 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="timestamp"
            tickFormatter={(value) => {
              if (!value) return '';
              const date = new Date(value);
              return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
            }}
            stroke="#6b7280"
            style={{ fontSize: '12px' }}
          />
          <YAxis
            domain={['auto', 'auto']}
            tickFormatter={(value) => `$${value.toLocaleString()}`}
            stroke="#6b7280"
            style={{ fontSize: '12px' }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />

          {/* Candlestick wicks (high-low lines) */}
          <Line
            type="monotone"
            dataKey="wickHigh"
            stroke="#9ca3af"
            strokeWidth={1}
            dot={false}
            isAnimationActive={false}
            connectNulls
            name="Wicks"
            legendType="none"
          />
          <Line
            type="monotone"
            dataKey="wickLow"
            stroke="#9ca3af"
            strokeWidth={1}
            dot={false}
            isAnimationActive={false}
            connectNulls
            legendType="none"
          />

          {/* Candlestick bodies (open-close areas) - Green candles */}
          {chartData.some((d) => d.isGreen) && (
            <Area
              type="monotone"
              dataKey={(d: any) => (d.isGreen ? [d.bodyLow, d.bodyHigh] : null)}
              stroke="#10b981"
              fill="#10b981"
              fillOpacity={0.6}
              isAnimationActive={false}
              name="Green Candles"
              legendType="none"
            />
          )}

          {/* Candlestick bodies - Red candles */}
          {chartData.some((d) => !d.isGreen) && (
            <Area
              type="monotone"
              dataKey={(d: any) => (!d.isGreen ? [d.bodyLow, d.bodyHigh] : null)}
              stroke="#ef4444"
              fill="#ef4444"
              fillOpacity={0.6}
              isAnimationActive={false}
              name="Red Candles"
              legendType="none"
            />
          )}

          {/* Moving Average Overlays */}
          {selectedOverlays.includes('sma_20') && (
            <Line
              type="monotone"
              dataKey="sma_20"
              stroke={overlayConfig.sma_20.color}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
              connectNulls
              name={overlayConfig.sma_20.name}
            />
          )}

          {selectedOverlays.includes('sma_50') && (
            <Line
              type="monotone"
              dataKey="sma_50"
              stroke={overlayConfig.sma_50.color}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
              connectNulls
              name={overlayConfig.sma_50.name}
            />
          )}

          {selectedOverlays.includes('ema_20') && (
            <Line
              type="monotone"
              dataKey="ema_20"
              stroke={overlayConfig.ema_20.color}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
              connectNulls
              name={overlayConfig.ema_20.name}
            />
          )}

          {/* Bollinger Bands */}
          {selectedOverlays.includes('bb') && (
            <>
              <Line
                type="monotone"
                dataKey="bb_upper"
                stroke={overlayConfig.bb.color}
                strokeWidth={1}
                strokeDasharray="5 5"
                dot={false}
                isAnimationActive={false}
                connectNulls
                name="BB Upper"
              />
              <Line
                type="monotone"
                dataKey="bb_middle"
                stroke={overlayConfig.bb.color}
                strokeWidth={1}
                dot={false}
                isAnimationActive={false}
                connectNulls
                name="BB Middle"
              />
              <Line
                type="monotone"
                dataKey="bb_lower"
                stroke={overlayConfig.bb.color}
                strokeWidth={1}
                strokeDasharray="5 5"
                dot={false}
                isAnimationActive={false}
                connectNulls
                name="BB Lower"
              />
            </>
          )}

          {/* Trade Markers */}
          {visibleTrades.map((trade, idx) => {
            // Find the candle in chartData that matches the trade time
            const matchingCandle = chartData.find(
              (c) => new Date(c.timestamp).getTime() >= new Date(trade.timestamp).getTime()
            );

            if (!matchingCandle) return null;

            return (
              <ReferenceLine
                key={`trade-${idx}`}
                x={matchingCandle.timestamp}
                stroke={trade.side === 'BUY' ? '#10b981' : '#ef4444'}
                strokeWidth={2}
                strokeDasharray="3 3"
                label={{
                  value: trade.side,
                  position: 'top',
                  fill: trade.side === 'BUY' ? '#10b981' : '#ef4444',
                  fontSize: 12,
                }}
              />
            );
          })}

          {/* Brush for zooming (drag to select range) */}
          <Brush
            dataKey="timestamp"
            height={30}
            stroke="#8884d8"
            startIndex={zoomRange?.startIndex ?? 0}
            endIndex={zoomRange?.endIndex ?? chartData.length - 1}
            onChange={handleBrushChange}
            tickFormatter={(value) => {
              if (!value) return '';
              const date = new Date(value);
              return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
            }}
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Zoom Instructions */}
      <p className="text-xs text-gray-400 text-center">
        Drag the slider below the chart to zoom into a specific time range, or use the zoom buttons above.
      </p>
    </div>
  );
}
