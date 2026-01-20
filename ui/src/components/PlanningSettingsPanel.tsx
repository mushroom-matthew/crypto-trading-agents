import { useState } from 'react';
import { ChevronDown, ChevronUp, Calendar } from 'lucide-react';

export interface PlanningSettings {
  max_trades_per_day?: number;
  max_triggers_per_symbol_per_day?: number;
  judge_cadence_hours?: number;
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
                value={config.max_trades_per_day ?? 10}
                onChange={(e) =>
                  onChange({ ...config, max_trades_per_day: parseInt(e.target.value) })
                }
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
                value={config.max_triggers_per_symbol_per_day ?? 5}
                onChange={(e) =>
                  onChange({ ...config, max_triggers_per_symbol_per_day: parseInt(e.target.value) })
                }
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
                  value={config.judge_cadence_hours ?? 4}
                  onChange={(e) =>
                    onChange({ ...config, judge_cadence_hours: parseFloat(e.target.value) })
                  }
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
          </div>
        </div>
      )}
    </div>
  );
}
