import { useState } from 'react';
import { ChevronDown, ChevronUp, FlaskConical } from 'lucide-react';
import { numberOrFallback, parseOptionalInteger, parseOptionalNumber } from '../lib/utils';

export interface LearningBookSettings {
  learning_book_enabled?: boolean;
  learning_daily_risk_budget_pct?: number;
  learning_max_trades_per_day?: number;
}

interface LearningBookPanelProps<T extends LearningBookSettings> {
  config: T;
  onChange: (config: T) => void;
  disabled?: boolean;
}

export function LearningBookPanel<T extends LearningBookSettings>({
  config,
  onChange,
  disabled,
}: LearningBookPanelProps<T>) {
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
          <FlaskConical className="w-5 h-5 text-slate-600 dark:text-slate-300" />
          <span className="font-medium text-slate-800 dark:text-slate-100">
            Learning Book
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
            Enable a small learning book that runs experimental trades with a
            capped risk budget, separate from the main strategy.
          </p>

          <div className="flex items-center gap-3">
            <input
              type="checkbox"
              id="learningBookEnabled"
              checked={config.learning_book_enabled ?? false}
              onChange={(e) =>
                onChange({ ...config, learning_book_enabled: e.target.checked })
              }
              className="w-4 h-4 text-blue-600 rounded"
              disabled={disabled}
            />
            <label htmlFor="learningBookEnabled" className="text-sm">
              Enable Learning Book
            </label>
          </div>

          {config.learning_book_enabled && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">
                  Daily Risk Budget %
                </label>
                <input
                  type="number"
                  value={numberOrFallback(config.learning_daily_risk_budget_pct, 1.0)}
                  onChange={(e) => {
                    const next = parseOptionalNumber(e.target.value);
                    if (next === undefined) {
                      return;
                    }
                    onChange({ ...config, learning_daily_risk_budget_pct: next });
                  }}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                  min={0.1}
                  max={10}
                  step={0.1}
                  disabled={disabled}
                />
                <p className="text-xs text-slate-400 mt-1">
                  Max % of equity risked per day on learning trades
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">
                  Max Trades / Day
                </label>
                <input
                  type="number"
                  value={numberOrFallback(config.learning_max_trades_per_day, 3)}
                  onChange={(e) => {
                    const next = parseOptionalInteger(e.target.value);
                    if (next === undefined) {
                      return;
                    }
                    onChange({ ...config, learning_max_trades_per_day: next });
                  }}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                  min={1}
                  max={20}
                  step={1}
                  disabled={disabled}
                />
                <p className="text-xs text-slate-400 mt-1">
                  Maximum learning trades per day
                </p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
