import { useMemo } from 'react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { formatDateTime } from '../lib/utils';
import type { CandleWithIndicators } from '../lib/api';

export interface IndicatorPanelsProps {
  candles: CandleWithIndicators[];
  currentIndex: number;
}

export function IndicatorPanels({ candles, currentIndex }: IndicatorPanelsProps) {
  const visibleCandles = useMemo(() => {
    return candles.slice(0, currentIndex + 1);
  }, [candles, currentIndex]);

  const chartData = useMemo(() => {
    return visibleCandles.map((candle, idx) => ({
      index: idx,
      timestamp: candle.timestamp,
      rsi: candle.rsi_14,
      macd: candle.macd,
      macd_signal: candle.macd_signal,
      macd_hist: candle.macd_hist,
      atr: candle.atr_14,
    }));
  }, [visibleCandles]);

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || !payload.length) return null;

    const data = chartData[label];
    if (!data) return null;

    return (
      <div className="bg-white dark:bg-gray-800 p-2 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
        <p className="text-xs text-gray-500 mb-1">{formatDateTime(data.timestamp)}</p>
        <div className="space-y-1">
          {payload.map((entry: any) => (
            <div key={entry.name} className="flex items-center gap-2 text-xs">
              <span
                className="w-3 h-3 rounded"
                style={{ backgroundColor: entry.color }}
              />
              <span className="text-gray-700 dark:text-gray-300">
                {entry.name}: {entry.value !== null ? entry.value.toFixed(2) : 'N/A'}
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* RSI Panel */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
        <h3 className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">
          RSI (14) - Relative Strength Index
        </h3>
        <ResponsiveContainer width="100%" height={150}>
          <LineChart
            data={chartData}
            margin={{ top: 5, right: 30, left: 0, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="index"
              tickFormatter={(value) => {
                const candle = chartData[value];
                if (!candle) return '';
                const date = new Date(candle.timestamp);
                return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
              }}
              stroke="#6b7280"
              style={{ fontSize: '11px' }}
            />
            <YAxis
              domain={[0, 100]}
              ticks={[0, 30, 50, 70, 100]}
              stroke="#6b7280"
              style={{ fontSize: '11px' }}
            />
            <Tooltip content={<CustomTooltip />} />

            {/* Overbought/Oversold lines */}
            <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="3 3" label="Overbought" />
            <ReferenceLine y={50} stroke="#6b7280" strokeDasharray="1 1" />
            <ReferenceLine y={30} stroke="#10b981" strokeDasharray="3 3" label="Oversold" />

            <Line
              type="monotone"
              dataKey="rsi"
              stroke="#8b5cf6"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
              connectNulls
              name="RSI"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* MACD Panel */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
        <h3 className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">
          MACD (12, 26, 9) - Moving Average Convergence Divergence
        </h3>
        <ResponsiveContainer width="100%" height={150}>
          <BarChart
            data={chartData}
            margin={{ top: 5, right: 30, left: 0, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="index"
              tickFormatter={(value) => {
                const candle = chartData[value];
                if (!candle) return '';
                const date = new Date(candle.timestamp);
                return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
              }}
              stroke="#6b7280"
              style={{ fontSize: '11px' }}
            />
            <YAxis stroke="#6b7280" style={{ fontSize: '11px' }} />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={0} stroke="#6b7280" />

            {/* MACD Histogram */}
            <Bar
              dataKey="macd_hist"
              fill="#94a3b8"
              opacity={0.6}
              isAnimationActive={false}
              name="MACD Histogram"
            />

            {/* MACD Line */}
            <Line
              type="monotone"
              dataKey="macd"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
              connectNulls
              name="MACD"
            />

            {/* Signal Line */}
            <Line
              type="monotone"
              dataKey="macd_signal"
              stroke="#ef4444"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
              connectNulls
              name="Signal"
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* ATR Panel */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
        <h3 className="text-sm font-semibold mb-3 text-gray-700 dark:text-gray-300">
          ATR (14) - Average True Range
        </h3>
        <ResponsiveContainer width="100%" height={120}>
          <LineChart
            data={chartData}
            margin={{ top: 5, right: 30, left: 0, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="index"
              tickFormatter={(value) => {
                const candle = chartData[value];
                if (!candle) return '';
                const date = new Date(candle.timestamp);
                return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
              }}
              stroke="#6b7280"
              style={{ fontSize: '11px' }}
            />
            <YAxis stroke="#6b7280" style={{ fontSize: '11px' }} />
            <Tooltip content={<CustomTooltip />} />

            <Line
              type="monotone"
              dataKey="atr"
              stroke="#f59e0b"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
              connectNulls
              name="ATR"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
