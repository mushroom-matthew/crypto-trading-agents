import { useState, memo, useCallback } from 'react';
import { ChevronDown, ChevronUp, AlertTriangle, Info, Zap, Shield, Target } from 'lucide-react';
import { numberOrFallback, parseOptionalNumber } from '../lib/utils';
import { useDebouncedCallback } from '../hooks/useDebounce';

// Generic type for aggressive trading settings shared by BacktestConfig and PaperTradingSessionConfig
export interface AggressiveSettings {
  // Risk Engine Parameters
  max_position_risk_pct?: number;
  max_symbol_exposure_pct?: number;
  max_portfolio_exposure_pct?: number;
  max_daily_loss_pct?: number;
  max_daily_risk_budget_pct?: number;

  // Whipsaw / Anti-Flip-Flop Controls
  min_hold_hours?: number;
  min_flat_hours?: number;
  confidence_override_threshold?: string | null;
  conflicting_signal_policy?: 'ignore' | 'exit' | 'reverse' | 'defer';

  // Execution Gating
  min_price_move_pct?: number;

  // Walk-Away Threshold
  walk_away_enabled?: boolean;
  walk_away_profit_target_pct?: number;

  // Flattening Options
  flatten_positions_daily?: boolean;
}

interface AggressiveSettingsPanelProps<T extends AggressiveSettings> {
  config: T;
  onChange: (config: T) => void;
  disabled?: boolean;
}

// Whipsaw preset configurations
const WHIPSAW_PRESETS = {
  conservative: {
    min_hold_hours: 2.0,
    min_flat_hours: 2.0,
    confidence_override_threshold: 'A',
    label: 'Conservative (default)',
    description: 'Full protection against rapid position changes',
  },
  moderate: {
    min_hold_hours: 1.0,
    min_flat_hours: 1.0,
    confidence_override_threshold: 'B',
    label: 'Moderate',
    description: 'Balanced protection with faster trading allowed',
  },
  aggressive: {
    min_hold_hours: 0.25,
    min_flat_hours: 0.25,
    confidence_override_threshold: 'C',
    label: 'Aggressive',
    description: 'Minimal protection for active trading',
  },
  disabled: {
    min_hold_hours: 0,
    min_flat_hours: 0,
    confidence_override_threshold: null,
    label: 'Disabled (Scalper Mode)',
    description: 'No protection - maximum trade frequency',
  },
};

function AggressiveSettingsPanelInner<T extends AggressiveSettings>({ config, onChange, disabled }: AggressiveSettingsPanelProps<T>) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [whipsawPreset, setWhipsawPreset] = useState<string>('conservative');

  // Debounce config updates to prevent lag on rapid input changes
  const debouncedOnChange = useDebouncedCallback(onChange, 150);

  const applyWhipsawPreset = useCallback((presetKey: string) => {
    setWhipsawPreset(presetKey);
    const preset = WHIPSAW_PRESETS[presetKey as keyof typeof WHIPSAW_PRESETS];
    if (preset) {
      onChange({
        ...config,
        min_hold_hours: preset.min_hold_hours,
        min_flat_hours: preset.min_flat_hours,
        confidence_override_threshold: preset.confidence_override_threshold,
      });
    }
  }, [config, onChange]);

  const portfolioExposure = numberOrFallback(config.max_portfolio_exposure_pct, 80);
  const isLeveraged = portfolioExposure > 100;
  const isScalperMode = config.min_hold_hours === 0 && config.min_flat_hours === 0;
  const conflictPolicy = config.conflicting_signal_policy ?? 'reverse';

  // Memoized handlers for numeric inputs
  const handlePositionRiskChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const next = parseOptionalNumber(e.target.value);
    if (next === undefined) return;
    debouncedOnChange({ ...config, max_position_risk_pct: next });
  }, [config, debouncedOnChange]);

  const handlePortfolioExposureChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const next = parseOptionalNumber(e.target.value);
    if (next === undefined) return;
    debouncedOnChange({ ...config, max_portfolio_exposure_pct: next });
  }, [config, debouncedOnChange]);

  const handleSymbolExposureChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const next = parseOptionalNumber(e.target.value);
    if (next === undefined) return;
    debouncedOnChange({ ...config, max_symbol_exposure_pct: next });
  }, [config, debouncedOnChange]);

  const handleDailyLossChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const next = parseOptionalNumber(e.target.value);
    if (next === undefined) return;
    debouncedOnChange({ ...config, max_daily_loss_pct: next });
  }, [config, debouncedOnChange]);

  const handleDailyRiskBudgetChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const next = parseOptionalNumber(e.target.value);
    if (next === undefined) return;
    debouncedOnChange({ ...config, max_daily_risk_budget_pct: next });
  }, [config, debouncedOnChange]);

  const handleMinPriceMoveChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const next = parseOptionalNumber(e.target.value);
    if (next === undefined) return;
    debouncedOnChange({ ...config, min_price_move_pct: next });
  }, [config, debouncedOnChange]);

  const handleMinHoldHoursChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const next = parseOptionalNumber(e.target.value);
    if (next === undefined) return;
    debouncedOnChange({ ...config, min_hold_hours: next });
    setWhipsawPreset('custom');
  }, [config, debouncedOnChange]);

  const handleMinFlatHoursChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const next = parseOptionalNumber(e.target.value);
    if (next === undefined) return;
    debouncedOnChange({ ...config, min_flat_hours: next });
    setWhipsawPreset('custom');
  }, [config, debouncedOnChange]);

  const handleConflictPolicyChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    onChange({ ...config, conflicting_signal_policy: e.target.value as AggressiveSettings['conflicting_signal_policy'] });
  }, [config, onChange]);

  const handleWalkAwayEnabledChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    onChange({ ...config, walk_away_enabled: e.target.checked });
  }, [config, onChange]);

  const handleWalkAwayTargetChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const next = parseOptionalNumber(e.target.value);
    if (next === undefined) return;
    debouncedOnChange({ ...config, walk_away_profit_target_pct: next });
  }, [config, debouncedOnChange]);

  const handleFlattenDailyChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    onChange({ ...config, flatten_positions_daily: e.target.checked });
  }, [config, onChange]);

  return (
    <div className="border border-orange-200 dark:border-orange-800 rounded-lg overflow-hidden">
      {/* Header */}
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 flex items-center justify-between bg-orange-50 dark:bg-orange-900/20 hover:bg-orange-100 dark:hover:bg-orange-900/30 transition-colors"
        disabled={disabled}
      >
        <div className="flex items-center gap-2">
          <Zap className="w-5 h-5 text-orange-500" />
          <span className="font-medium text-orange-700 dark:text-orange-300">
            Advanced Trading Settings
          </span>
          {(isLeveraged || isScalperMode) && (
            <span className="px-2 py-0.5 text-xs font-medium bg-orange-200 dark:bg-orange-700 text-orange-800 dark:text-orange-100 rounded">
              {isLeveraged ? `${((config.max_portfolio_exposure_pct ?? 100) / 100).toFixed(1)}x Leverage` : 'Scalper Mode'}
            </span>
          )}
        </div>
        {isExpanded ? (
          <ChevronUp className="w-5 h-5 text-orange-500" />
        ) : (
          <ChevronDown className="w-5 h-5 text-orange-500" />
        )}
      </button>

      {/* Content */}
      {isExpanded && (
        <div className="p-4 space-y-6 bg-white dark:bg-gray-800">
          {/* Risk Warning */}
          {(isLeveraged || isScalperMode || config.walk_away_enabled) && (
            <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg flex items-start gap-2">
              <AlertTriangle className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
              <div className="text-sm text-yellow-700 dark:text-yellow-300">
                <strong>Risk Notice:</strong> You have enabled aggressive trading settings.
                {isLeveraged && ' Leverage amplifies both gains AND losses.'}
                {isScalperMode && ' Disabled whipsaw protection may lead to rapid position changes.'}
              </div>
            </div>
          )}

          {/* Section: Risk Parameters */}
          <div className="space-y-4">
            <h3 className="flex items-center gap-2 font-medium text-gray-900 dark:text-gray-100">
              <Shield className="w-4 h-4" />
              Risk Parameters
            </h3>

            <div className="grid grid-cols-2 gap-4">
              {/* Position Risk % */}
              <div>
                <label className="block text-sm font-medium mb-1">
                  Position Risk %
                </label>
                <input
                  type="number"
                  value={numberOrFallback(config.max_position_risk_pct, 2)}
                  onChange={handlePositionRiskChange}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                  min={0.1}
                  max={20}
                  step={0.1}
                  disabled={disabled}
                />
                <p className="text-xs text-gray-500 mt-1">Risk per trade (default: 2%)</p>
              </div>

              {/* Portfolio Exposure % (Leverage) */}
              <div>
                <label className="block text-sm font-medium mb-1">
                  Portfolio Exposure %
                  {portfolioExposure > 100 && (
                    <span className="ml-2 text-orange-500 font-bold">
                      ({(portfolioExposure / 100).toFixed(1)}x)
                    </span>
                  )}
                </label>
                <input
                  type="number"
                  value={portfolioExposure}
                  onChange={handlePortfolioExposureChange}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                  min={10}
                  max={500}
                  step={1}
                  disabled={disabled}
                />
                <p className="text-xs text-gray-500 mt-1">&gt;100% = leverage (default: 80%)</p>
              </div>

              {/* Symbol Exposure % */}
              <div>
                <label className="block text-sm font-medium mb-1">
                  Symbol Exposure %
                </label>
                <input
                  type="number"
                  value={numberOrFallback(config.max_symbol_exposure_pct, 25)}
                  onChange={handleSymbolExposureChange}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                  min={5}
                  max={100}
                  step={1}
                  disabled={disabled}
                />
                <p className="text-xs text-gray-500 mt-1">Max per symbol (default: 25%)</p>
              </div>

              {/* Daily Loss Limit */}
              <div>
                <label className="block text-sm font-medium mb-1">
                  Daily Loss Limit %
                </label>
                <input
                  type="number"
                  value={numberOrFallback(config.max_daily_loss_pct, 3)}
                  onChange={handleDailyLossChange}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                  min={1}
                  max={50}
                  step={0.1}
                  disabled={disabled}
                />
                <p className="text-xs text-gray-500 mt-1">Stop trading limit (default: 3%)</p>
              </div>

              {/* Daily Risk Budget */}
              <div>
                <label className="block text-sm font-medium mb-1">
                  Daily Risk Budget %
                </label>
                <input
                  type="number"
                  value={numberOrFallback(config.max_daily_risk_budget_pct, 10)}
                  onChange={handleDailyRiskBudgetChange}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                  min={1}
                  max={50}
                  step={0.5}
                  disabled={disabled}
                />
                <p className="text-xs text-gray-500 mt-1">Cumulative risk cap per day (default: 10%)</p>
              </div>
            </div>
          </div>

          {/* Section: Execution Gating */}
          <div className="space-y-4 pt-4 border-t border-gray-200 dark:border-gray-700">
            <h3 className="flex items-center gap-2 font-medium text-gray-900 dark:text-gray-100">
              <Zap className="w-4 h-4" />
              Execution Gating
            </h3>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">
                  Min Price Move %
                </label>
                <input
                  type="number"
                  value={numberOrFallback(config.min_price_move_pct, 0.5)}
                  onChange={handleMinPriceMoveChange}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                  min={0}
                  max={10}
                  step={0.1}
                  disabled={disabled}
                />
                <p className="text-xs text-gray-500 mt-1">Lower = more signals (default: 0.5%)</p>
              </div>
            </div>
          </div>

          {/* Section: Whipsaw Protection */}
          <div className="space-y-4 pt-4 border-t border-gray-200 dark:border-gray-700">
            <h3 className="flex items-center gap-2 font-medium text-gray-900 dark:text-gray-100">
              <Shield className="w-4 h-4" />
              Whipsaw Protection
              <div className="group relative">
                <Info className="w-4 h-4 text-gray-400 cursor-help" />
                <div className="invisible group-hover:visible absolute left-0 top-6 z-10 w-64 p-2 bg-gray-800 text-white text-xs rounded shadow-lg">
                  Prevents rapid position changes (flip-flopping). Disable for high-frequency scalping.
                </div>
              </div>
            </h3>

            {/* Preset Selector */}
            <div>
              <label className="block text-sm font-medium mb-1">
                Protection Level
              </label>
              <select
                value={whipsawPreset}
                onChange={(e) => applyWhipsawPreset(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                disabled={disabled}
              >
                {Object.entries(WHIPSAW_PRESETS).map(([key, preset]) => (
                  <option key={key} value={key}>
                    {preset.label}
                  </option>
                ))}
              </select>
              <p className="text-xs text-gray-500 mt-1">
                {WHIPSAW_PRESETS[whipsawPreset as keyof typeof WHIPSAW_PRESETS]?.description}
              </p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              {/* Min Hold Hours */}
              <div>
                <label className="block text-sm font-medium mb-1">
                  Min Hold Time (hours)
                </label>
                <input
                  type="number"
                  value={numberOrFallback(config.min_hold_hours, 2)}
                  onChange={handleMinHoldHoursChange}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                  min={0}
                  max={24}
                  step={0.25}
                  disabled={disabled}
                />
                <p className="text-xs text-gray-500 mt-1">0 = exit anytime</p>
              </div>

              {/* Min Flat Hours (Cooldown) */}
              <div>
                <label className="block text-sm font-medium mb-1">
                  Trade Cooldown (hours)
                </label>
                <input
                  type="number"
                  value={numberOrFallback(config.min_flat_hours, 2)}
                  onChange={handleMinFlatHoursChange}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                  min={0}
                  max={24}
                  step={0.25}
                  disabled={disabled}
                />
                <p className="text-xs text-gray-500 mt-1">0 = immediate re-entry</p>
              </div>
            </div>
          </div>

          {/* Section: Conflicting Signal Policy */}
          <div className="space-y-4 pt-4 border-t border-gray-200 dark:border-gray-700">
            <h3 className="flex items-center gap-2 font-medium text-gray-900 dark:text-gray-100">
              <AlertTriangle className="w-4 h-4" />
              Conflicting Signal Policy
              <div className="group relative">
                <Info className="w-4 h-4 text-gray-400 cursor-help" />
                <div className="invisible group-hover:visible absolute left-0 top-6 z-10 w-64 p-2 bg-gray-800 text-white text-xs rounded shadow-lg">
                  Decide what happens when an opposing entry signal fires while already in a position.
                </div>
              </div>
            </h3>

            <div>
              <label className="block text-sm font-medium mb-1">
                Resolver Policy
              </label>
              <select
                value={conflictPolicy}
                onChange={handleConflictPolicyChange}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                disabled={disabled}
              >
                <option value="ignore">Ignore (hold)</option>
                <option value="exit">Exit (flatten)</option>
                <option value="reverse">Reverse (flip)</option>
                <option value="defer">Defer (wait)</option>
              </select>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs text-gray-500">
              <div>
                <span className="font-medium text-gray-700 dark:text-gray-300">Ignore</span>: drawdown during reversals
              </div>
              <div>
                <span className="font-medium text-gray-700 dark:text-gray-300">Exit</span>: missed continuation
              </div>
              <div>
                <span className="font-medium text-gray-700 dark:text-gray-300">Reverse</span>: whipsaw + fees
              </div>
              <div>
                <span className="font-medium text-gray-700 dark:text-gray-300">Defer</span>: opportunity cost
              </div>
            </div>
          </div>

          {/* Section: Walk-Away Threshold */}
          <div className="space-y-4 pt-4 border-t border-gray-200 dark:border-gray-700">
            <h3 className="flex items-center gap-2 font-medium text-gray-900 dark:text-gray-100">
              <Target className="w-4 h-4" />
              Walk-Away Threshold
              <div className="group relative">
                <Info className="w-4 h-4 text-gray-400 cursor-help" />
                <div className="invisible group-hover:visible absolute left-0 top-6 z-10 w-64 p-2 bg-gray-800 text-white text-xs rounded shadow-lg">
                  Stop trading for the day after reaching your profit target. Locks in gains and prevents giving back profits.
                </div>
              </div>
            </h3>

            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={config.walk_away_enabled ?? false}
                  onChange={handleWalkAwayEnabledChange}
                  className="w-4 h-4 rounded border-gray-300 text-orange-500 focus:ring-orange-500"
                  disabled={disabled}
                />
                <span className="text-sm">Enable Walk-Away Mode</span>
              </label>
            </div>

            {config.walk_away_enabled && (
              <div>
                <label className="block text-sm font-medium mb-1">
                  Profit Target %
                </label>
                <input
                  type="number"
                  value={numberOrFallback(config.walk_away_profit_target_pct, 25)}
                  onChange={handleWalkAwayTargetChange}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                  min={1}
                  max={100}
                  step={1}
                  disabled={disabled}
                />
                <p className="text-xs text-gray-500 mt-1">
                  Stop trading after reaching {config.walk_away_profit_target_pct ?? 25}% daily return
                </p>
              </div>
            )}
          </div>

          {/* Section: Position Flattening */}
          <div className="space-y-4 pt-4 border-t border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={config.flatten_positions_daily ?? false}
                  onChange={handleFlattenDailyChange}
                  className="w-4 h-4 rounded border-gray-300 text-orange-500 focus:ring-orange-500"
                  disabled={disabled}
                />
                <span className="text-sm">Flatten Positions Daily</span>
              </label>
              <div className="group relative">
                <Info className="w-4 h-4 text-gray-400 cursor-help" />
                <div className="invisible group-hover:visible absolute left-0 top-6 z-10 w-64 p-2 bg-gray-800 text-white text-xs rounded shadow-lg">
                  Close all positions at end of each trading day. Reduces overnight risk.
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Wrap with React.memo to prevent re-renders when parent state changes but props are the same
export const AggressiveSettingsPanel = memo(AggressiveSettingsPanelInner) as typeof AggressiveSettingsPanelInner;
