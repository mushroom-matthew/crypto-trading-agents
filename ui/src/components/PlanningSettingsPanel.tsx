import { useState, memo, useCallback } from 'react';
import { ChevronDown, ChevronUp, Calendar } from 'lucide-react';
import { numberOrFallback, parseOptionalInteger, parseOptionalNumber } from '../lib/utils';
import { useDebouncedCallback } from '../hooks/useDebounce';

export interface PlanningSettings {
  max_trades_per_day?: number;
  max_triggers_per_symbol_per_day?: number;
  judge_cadence_hours?: number;
  judge_check_after_trades?: number;
  replan_on_day_boundary?: boolean;
  debug_trigger_sample_rate?: number;
  debug_trigger_max_samples?: number;
  indicator_debug_mode?: string;
  indicator_debug_keys?: string[];
}

interface PlanningSettingsPanelProps<T extends PlanningSettings> {
  config: T;
  onChange: (config: T) => void;
  disabled?: boolean;
  showJudgeCadence?: boolean;
  showDayBoundaryReplan?: boolean;
}

function PlanningSettingsPanelInner<T extends PlanningSettings>({
  config,
  onChange,
  disabled,
  showJudgeCadence = true,
  showDayBoundaryReplan = false,
}: PlanningSettingsPanelProps<T>) {
  const [isExpanded, setIsExpanded] = useState(false);

  // Debounce config updates to prevent lag on rapid input changes
  const debouncedOnChange = useDebouncedCallback(onChange, 150);

  // Memoized handlers for inputs
  const handleMaxTradesChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const next = parseOptionalInteger(e.target.value);
    if (next === undefined) return;
    debouncedOnChange({ ...config, max_trades_per_day: next });
  }, [config, debouncedOnChange]);

  const handleMaxTriggersChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const next = parseOptionalInteger(e.target.value);
    if (next === undefined) return;
    debouncedOnChange({ ...config, max_triggers_per_symbol_per_day: next });
  }, [config, debouncedOnChange]);

  const handleJudgeCadenceChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const next = parseOptionalNumber(e.target.value);
    if (next === undefined) return;
    debouncedOnChange({ ...config, judge_cadence_hours: next });
  }, [config, debouncedOnChange]);

  const handleJudgeCheckAfterChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const next = parseOptionalInteger(e.target.value);
    if (next === undefined) return;
    debouncedOnChange({ ...config, judge_check_after_trades: next });
  }, [config, debouncedOnChange]);

  const handleReplanChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    onChange({ ...config, replan_on_day_boundary: e.target.checked });
  }, [config, onChange]);

  const handleDebugSampleRateChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const next = parseOptionalNumber(e.target.value);
    if (next === undefined) return;
    debouncedOnChange({ ...config, debug_trigger_sample_rate: next });
  }, [config, debouncedOnChange]);

  const handleDebugMaxSamplesChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const next = parseOptionalInteger(e.target.value);
    if (next === undefined) return;
    debouncedOnChange({ ...config, debug_trigger_max_samples: next });
  }, [config, debouncedOnChange]);

  const handleDebugModeChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    onChange({ ...config, indicator_debug_mode: e.target.value });
  }, [config, onChange]);

  const handleDebugKeysChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const keys = e.target.value
      .split(',')
      .map((entry) => entry.trim())
      .filter(Boolean);
    debouncedOnChange({ ...config, indicator_debug_keys: keys });
  }, [config, debouncedOnChange]);

  return (
    <div className="border border-slate-200 dark:border-slate-700 rounded-lg overflow-hidden">
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 flex items-center justify-between bg-slate-50 dark:bg-slate-900/20 hover:bg-slate-100 dark:hover:bg-slate-900/30 transition-colors"
        disabled={disabled}
      >
        <div className="flex items-center gap-2">
          <Calendar className="w-5 h-5 text-slate-600 dark:text-slate-300" />
          <span className="font-medium text-slate-800 dark:text-slate-100">
            Advanced Planning
          </span>
        </div>
        {isExpanded ? (
          <ChevronUp className="w-5 h-5 text-slate-600 dark:text-slate-300" />
        ) : (
          <ChevronDown className="w-5 h-5 text-slate-600 dark:text-slate-300" />
        )}
      </button>

      {isExpanded && (
        <div className="p-4 space-y-4 bg-white dark:bg-gray-800">
          <p className="text-xs text-slate-500">
            Plan caps apply to the LLM strategist plan (trade count + trigger budgets),
            not the risk engine. Use for aggressive backtests without touching risk limits.
          </p>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">
                Max Trades/Day
              </label>
              <input
                type="number"
                value={numberOrFallback(config.max_trades_per_day, 10)}
                onChange={handleMaxTradesChange}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                min={1}
                max={200}
                step={1}
                disabled={disabled}
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">
                Max Triggers/Symbol/Day
              </label>
              <input
                type="number"
                value={numberOrFallback(config.max_triggers_per_symbol_per_day, 5)}
                onChange={handleMaxTriggersChange}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                min={1}
                max={50}
                step={1}
                disabled={disabled}
              />
            </div>

            {showJudgeCadence && (
              <div>
                <label className="block text-sm font-medium mb-1">
                  Judge Cadence (hours)
                </label>
                <input
                  type="number"
                  value={numberOrFallback(config.judge_cadence_hours, 4)}
                  onChange={handleJudgeCadenceChange}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                  min={1}
                  max={24}
                  step={1}
                  disabled={disabled}
                />
                <p className="text-xs text-slate-400 mt-1">
                  How often judge evaluates strategy quality
                </p>
              </div>
            )}

            {showJudgeCadence && (
              <div>
                <label className="block text-sm font-medium mb-1">
                  Judge Check After Trades
                </label>
                <input
                  type="number"
                  value={numberOrFallback(config.judge_check_after_trades, 3)}
                  onChange={handleJudgeCheckAfterChange}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                  min={1}
                  max={100}
                  step={1}
                  disabled={disabled}
                />
                <p className="text-xs text-slate-400 mt-1">
                  Triggers a judge run after N trades, even if cadence has not elapsed
                </p>
              </div>
            )}

            {showDayBoundaryReplan && (
              <div className="col-span-2">
                <label className="block text-sm font-medium mb-1">
                  Start-of-Day Replan
                </label>
                <div className="flex items-center gap-3">
                  <input
                    type="checkbox"
                    checked={config.replan_on_day_boundary ?? true}
                    onChange={handleReplanChange}
                    className="w-4 h-4 text-blue-600 rounded"
                    disabled={disabled}
                  />
                  <span className="text-sm text-slate-700 dark:text-slate-300">
                    Replan at day boundary (start of day)
                  </span>
                </div>
                <p className="text-xs text-slate-400 mt-1">
                  When off, replans happen only on judge triggers.
                </p>
              </div>
            )}

            <div>
              <label className="block text-sm font-medium mb-1">
                Trigger Debug Sample Rate
              </label>
              <input
                type="number"
                value={numberOrFallback(config.debug_trigger_sample_rate, 0)}
                onChange={handleDebugSampleRateChange}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                min={0}
                max={1}
                step={0.05}
                disabled={disabled}
              />
              <p className="text-xs text-slate-400 mt-1">
                Fraction of trigger evaluations to trace (0-1)
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">
                Trigger Debug Sample Max
              </label>
              <input
                type="number"
                value={numberOrFallback(config.debug_trigger_max_samples, 100)}
                onChange={handleDebugMaxSamplesChange}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                min={1}
                max={1000}
                step={1}
                disabled={disabled}
              />
              <p className="text-xs text-slate-400 mt-1">
                Max trigger traces to store per run
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">
                Indicator Debug Mode
              </label>
              <select
                value={config.indicator_debug_mode ?? 'off'}
                onChange={handleDebugModeChange}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                disabled={disabled}
              >
                <option value="off">Off</option>
                <option value="full">Full Snapshot</option>
                <option value="keys">Key List</option>
              </select>
              <p className="text-xs text-slate-400 mt-1">
                Capture indicators per bar for debugging
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">
                Indicator Debug Keys
              </label>
              <input
                type="text"
                value={(config.indicator_debug_keys ?? []).join(', ')}
                onChange={handleDebugKeysChange}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                placeholder="rsi_14, ema_short, ema_long"
                disabled={disabled || (config.indicator_debug_mode ?? 'off') !== 'keys'}
              />
              <p className="text-xs text-slate-400 mt-1">
                Comma-separated keys (used when mode = keys)
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Wrap with React.memo to prevent re-renders when parent state changes but props are the same
export const PlanningSettingsPanel = memo(PlanningSettingsPanelInner) as typeof PlanningSettingsPanelInner;
