import { useState, useEffect, useRef } from 'react';
import { Play, Pause, ChevronLeft, ChevronRight } from 'lucide-react';
import { cn } from '../lib/utils';

export interface TimelinePlayerProps {
  totalCandles: number;
  events: Array<{ timestamp: string }>;
  onTimestampChange: (index: number, timestamp: string) => void;
  currentIndex: number;
  mode: 'candle' | 'event';
  onModeChange: (mode: 'candle' | 'event') => void;
  disabled?: boolean;
}

type PlaybackSpeed = 0.5 | 1 | 2 | 5 | 10;

export function TimelinePlayer({
  totalCandles,
  events,
  onTimestampChange,
  currentIndex,
  mode,
  onModeChange,
  disabled = false,
}: TimelinePlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState<PlaybackSpeed>(1);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Auto-advance when playing
  useEffect(() => {
    if (isPlaying && !disabled) {
      const interval = 1000 / speed; // milliseconds between steps
      intervalRef.current = setInterval(() => {
        handleStepForward();
      }, interval);

      return () => {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
      };
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }
  }, [isPlaying, speed, currentIndex, mode, disabled]);

  const handleStepForward = () => {
    if (mode === 'candle') {
      if (currentIndex < totalCandles - 1) {
        onTimestampChange(currentIndex + 1, '');
      } else {
        setIsPlaying(false);
      }
    } else {
      // Event mode: jump to next event
      const nextEventIndex = events.findIndex(
        (_, idx) => idx > currentIndex && events[idx]
      );
      if (nextEventIndex !== -1) {
        onTimestampChange(nextEventIndex, events[nextEventIndex].timestamp);
      } else {
        setIsPlaying(false);
      }
    }
  };

  const handleStepBackward = () => {
    if (mode === 'candle') {
      if (currentIndex > 0) {
        onTimestampChange(currentIndex - 1, '');
      }
    } else {
      // Event mode: jump to previous event
      const prevEventIndex = [...events]
        .reverse()
        .findIndex((_, idx) => events.length - 1 - idx < currentIndex);

      if (prevEventIndex !== -1) {
        const actualIndex = events.length - 1 - prevEventIndex;
        onTimestampChange(actualIndex, events[actualIndex].timestamp);
      }
    }
  };

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handleScrub = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newIndex = parseInt(e.target.value, 10);
    onTimestampChange(newIndex, '');
    setIsPlaying(false);
  };

  const speeds: PlaybackSpeed[] = [0.5, 1, 2, 5, 10];

  const maxIndex = mode === 'candle' ? totalCandles - 1 : events.length - 1;
  const progress = maxIndex > 0 ? (currentIndex / maxIndex) * 100 : 0;

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
      <div className="flex items-center gap-4">
        {/* Mode Toggle */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => onModeChange('candle')}
            disabled={disabled}
            className={cn(
              'px-3 py-1.5 rounded text-sm font-medium transition-colors',
              mode === 'candle'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600',
              disabled && 'opacity-50 cursor-not-allowed'
            )}
          >
            Candle-by-Candle
          </button>
          <button
            onClick={() => onModeChange('event')}
            disabled={disabled || events.length === 0}
            className={cn(
              'px-3 py-1.5 rounded text-sm font-medium transition-colors',
              mode === 'event'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600',
              (disabled || events.length === 0) && 'opacity-50 cursor-not-allowed'
            )}
          >
            Events Only ({events.length})
          </button>
        </div>

        {/* Playback Controls */}
        <div className="flex items-center gap-2 flex-1">
          {/* Step Backward */}
          <button
            onClick={handleStepBackward}
            disabled={disabled || currentIndex === 0}
            className={cn(
              'p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors',
              (disabled || currentIndex === 0) && 'opacity-50 cursor-not-allowed'
            )}
            title="Step Backward"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>

          {/* Play/Pause */}
          <button
            onClick={handlePlayPause}
            disabled={disabled || currentIndex >= maxIndex}
            className={cn(
              'p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors',
              (disabled || currentIndex >= maxIndex) && 'opacity-50 cursor-not-allowed'
            )}
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? (
              <Pause className="w-5 h-5" />
            ) : (
              <Play className="w-5 h-5" />
            )}
          </button>

          {/* Step Forward */}
          <button
            onClick={handleStepForward}
            disabled={disabled || currentIndex >= maxIndex}
            className={cn(
              'p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors',
              (disabled || currentIndex >= maxIndex) && 'opacity-50 cursor-not-allowed'
            )}
            title="Step Forward"
          >
            <ChevronRight className="w-5 h-5" />
          </button>

          {/* Timeline Scrubber */}
          <div className="flex-1 mx-4">
            <input
              type="range"
              min="0"
              max={maxIndex}
              value={currentIndex}
              onChange={handleScrub}
              disabled={disabled}
              className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
              style={{
                background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${progress}%, #e5e7eb ${progress}%, #e5e7eb 100%)`,
              }}
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>{currentIndex + 1}</span>
              <span>{maxIndex + 1}</span>
            </div>
          </div>
        </div>

        {/* Speed Control */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-500">Speed:</span>
          <select
            value={speed}
            onChange={(e) => setSpeed(parseFloat(e.target.value) as PlaybackSpeed)}
            disabled={disabled}
            className={cn(
              'px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700',
              disabled && 'opacity-50 cursor-not-allowed'
            )}
          >
            {speeds.map((s) => (
              <option key={s} value={s}>
                {s}x
              </option>
            ))}
          </select>
        </div>

        {/* Progress Display */}
        <div className="text-sm text-gray-500">
          {progress.toFixed(1)}%
        </div>
      </div>
    </div>
  );
}
