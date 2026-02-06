import { useState, memo, useCallback } from 'react';
import { ChevronDown, ChevronUp, FlaskConical } from 'lucide-react';
import { numberOrFallback, parseOptionalInteger, parseOptionalNumber } from '../lib/utils';
import { useDebouncedCallback } from '../hooks/useDebounce';

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

function LearningBookPanelInner<T extends LearningBookSettings>({
  config,
  onChange,
  disabled,
}: LearningBookPanelProps<T>) {
  const [isExpanded, setIsExpanded] = useState(false);

  // Debounce config updates to prevent lag on rapid input changes
  const debouncedOnChange = useDebouncedCallback(onChange, 150);

  // Memoized handlers
  const handleEnabledChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    onChange({ ...config, learning_book_enabled: e.target.checked });
  }, [config, onChange]);

  const handleRiskBudgetChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const next = parseOptionalNumber(e.target.value);
    if (next === undefined) return;
    debouncedOnChange({ ...config, learning_daily_risk_budget_pct: next });
  }, [config, debouncedOnChange]);

  const handleMaxTradesChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const next = parseOptionalInteger(e.target.value);
    if (next === undefined) return;
    debouncedOnChange({ ...config, learning_max_trades_per_day: next });
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
              onChange={handleEnabledChange}
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
                  onChange={handleRiskBudgetChange}
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
                  onChange={handleMaxTradesChange}
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

// Wrap with React.memo to prevent re-renders when parent state changes but props are the same
export const LearningBookPanel = memo(LearningBookPanelInner) as typeof LearningBookPanelInner;
