import { useState } from 'react';
import { ChevronDown, ChevronUp, Calendar } from 'lucide-react';
import { numberOrFallback, parseOptionalInteger, parseOptionalNumber } from '../lib/utils';

export interface PlanningSettings {
  max_trades_per_day?: number;
  max_triggers_per_symbol_per_day?: number;
  judge_cadence_hours?: number;
  judge_check_after_trades?: number;
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
}

export function PlanningSettingsPanel<T extends PlanningSettings>({
  config,
  onChange,
  disabled,
  showJudgeCadence = true,
}: PlanningSettingsPanelProps<T>) {
  const [isExpanded, setIsExpanded] = useState(false);

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
                onChange={(e) => {
                  const next = parseOptionalInteger(e.target.value);
                  if (next === undefined) {
                    return;
                  }
                  onChange({ ...config, max_trades_per_day: next });
                }}
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
                onChange={(e) => {
                  const next = parseOptionalInteger(e.target.value);
                  if (next === undefined) {
                    return;
                  }
                  onChange({ ...config, max_triggers_per_symbol_per_day: next });
                }}
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
                  onChange={(e) => {
                    const next = parseOptionalNumber(e.target.value);
                    if (next === undefined) {
                      return;
                    }
                    onChange({ ...config, judge_cadence_hours: next });
                  }}
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
                  onChange={(e) => {
                    const next = parseOptionalInteger(e.target.value);
                    if (next === undefined) {
                      return;
                    }
                    onChange({ ...config, judge_check_after_trades: next });
                  }}
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

            <div>
              <label className="block text-sm font-medium mb-1">
                Trigger Debug Sample Rate
              </label>
              <input
                type="number"
                value={numberOrFallback(config.debug_trigger_sample_rate, 0)}
                onChange={(e) => {
                  const next = parseOptionalNumber(e.target.value);
                  if (next === undefined) {
                    return;
                  }
                  onChange({ ...config, debug_trigger_sample_rate: next });
                }}
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
                onChange={(e) => {
                  const next = parseOptionalInteger(e.target.value);
                  if (next === undefined) {
                    return;
                  }
                  onChange({ ...config, debug_trigger_max_samples: next });
                }}
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
                onChange={(e) =>
                  onChange({ ...config, indicator_debug_mode: e.target.value })
                }
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
                onChange={(e) => {
                  const keys = e.target.value
                    .split(',')
                    .map((entry) => entry.trim())
                    .filter(Boolean);
                  onChange({ ...config, indicator_debug_keys: keys });
                }}
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
