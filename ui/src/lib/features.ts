export type FeatureFlag =
  | 'paper_trading'
  | 'backtesting'
  | 'live_trading'
  | 'wallets'
  | 'agents';

const DEFAULT_FEATURES = new Set<FeatureFlag>(['paper_trading']);

const FEATURE_ALIASES: Record<string, FeatureFlag> = {
  paper: 'paper_trading',
  papertrading: 'paper_trading',
  paper_trading: 'paper_trading',
  backtest: 'backtesting',
  backtesting: 'backtesting',
  live: 'live_trading',
  live_trading: 'live_trading',
  livefills: 'live_trading',
  live_fills_monitor: 'live_trading',
  wallets: 'wallets',
  wallet: 'wallets',
  multi_wallet: 'wallets',
  wallet_reconciliation: 'wallets',
  agents: 'agents',
  agent_inspector: 'agents',
};

function normalizeFlag(flag: string): string {
  return flag.trim().toLowerCase().replace(/[\s-]+/g, '_');
}

function toFeatureFlag(flag: string): FeatureFlag | null {
  const normalized = normalizeFlag(flag);
  return FEATURE_ALIASES[normalized] ?? null;
}

function parseEnabledFlags(rawValue: string | undefined): Set<FeatureFlag> {
  if (!rawValue?.trim()) {
    return new Set(DEFAULT_FEATURES);
  }

  const requested = rawValue
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);

  if (requested.some((item) => normalizeFlag(item) === 'all')) {
    return new Set<FeatureFlag>(['paper_trading', 'backtesting', 'live_trading', 'wallets', 'agents']);
  }

  const enabled = new Set<FeatureFlag>();
  for (const item of requested) {
    const flag = toFeatureFlag(item);
    if (flag) {
      enabled.add(flag);
    }
  }

  return enabled.size > 0 ? enabled : new Set(DEFAULT_FEATURES);
}

const enabledFeatures = parseEnabledFlags(import.meta.env.VITE_FEATURES);

export function isEnabled(flag: string): boolean {
  const normalized = toFeatureFlag(flag);
  return normalized ? enabledFeatures.has(normalized) : false;
}

export function getEnabledFeatures(): FeatureFlag[] {
  return Array.from(enabledFeatures);
}
