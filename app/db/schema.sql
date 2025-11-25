-- Raw schema for initial database bootstrap.

CREATE TYPE wallet_type AS ENUM ('COINBASE_SPOT', 'COINBASE_TRADING', 'EXTERNAL');
CREATE TYPE ledger_side AS ENUM ('debit', 'credit');
CREATE TYPE order_side AS ENUM ('buy', 'sell');
CREATE TYPE order_type AS ENUM ('limit', 'market');
CREATE TYPE reservation_state AS ENUM ('active', 'consumed', 'canceled');

CREATE TABLE wallets (
    wallet_id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    coinbase_account_id TEXT NULL,
    portfolio_id TEXT NULL,
    type wallet_type NOT NULL,
    tradeable_fraction NUMERIC(4, 3) DEFAULT 0.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE balances (
    id BIGSERIAL PRIMARY KEY,
    wallet_id BIGINT NOT NULL REFERENCES wallets(wallet_id) ON DELETE CASCADE,
    currency TEXT NOT NULL,
    available NUMERIC(24, 12) NOT NULL,
    hold NUMERIC(24, 12) NOT NULL DEFAULT 0,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX ix_balances_wallet_currency ON balances (wallet_id, currency);

CREATE TABLE ledger_entries (
    entry_id BIGSERIAL PRIMARY KEY,
    wallet_id BIGINT NOT NULL REFERENCES wallets(wallet_id) ON DELETE CASCADE,
    currency TEXT NOT NULL,
    amount NUMERIC(24, 12) NOT NULL,
    side ledger_side NOT NULL,
    balance_after NUMERIC(24, 12) NOT NULL,
    source TEXT NOT NULL,
    external_tx_id TEXT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX ix_ledger_entries_wallet_created_at ON ledger_entries (wallet_id, created_at);

CREATE TABLE orders (
    order_id BIGSERIAL PRIMARY KEY,
    wallet_id BIGINT NOT NULL REFERENCES wallets(wallet_id) ON DELETE CASCADE,
    coinbase_order_id TEXT NOT NULL UNIQUE,
    product_id TEXT NOT NULL,
    side order_side NOT NULL,
    order_type order_type NOT NULL,
    price NUMERIC(24, 12),
    qty NUMERIC(24, 12) NOT NULL,
    status TEXT NOT NULL,
    filled_qty NUMERIC(24, 12) NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX ix_orders_wallet_created_at ON orders (wallet_id, created_at);

CREATE TABLE transfers (
    transfer_id BIGSERIAL PRIMARY KEY,
    wallet_id_from BIGINT NOT NULL REFERENCES wallets(wallet_id) ON DELETE CASCADE,
    wallet_id_to BIGINT NOT NULL REFERENCES wallets(wallet_id) ON DELETE CASCADE,
    currency TEXT NOT NULL,
    amount NUMERIC(24, 12) NOT NULL,
    coinbase_tx_id TEXT NULL,
    status TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX ix_transfers_wallet_created_at ON transfers (wallet_id_from, created_at);

CREATE TABLE reservations (
    res_id BIGSERIAL PRIMARY KEY,
    wallet_id BIGINT NOT NULL REFERENCES wallets(wallet_id) ON DELETE CASCADE,
    currency TEXT NOT NULL,
    amount NUMERIC(24, 12) NOT NULL,
    state reservation_state NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    idempotency_key TEXT NOT NULL UNIQUE
);
CREATE INDEX ix_reservations_wallet_state ON reservations (wallet_id, state);

CREATE TABLE fees_snapshots (
    id BIGSERIAL PRIMARY KEY,
    portfolio_id TEXT NOT NULL,
    maker_rate NUMERIC(12, 8) NOT NULL,
    taker_rate NUMERIC(12, 8) NOT NULL,
    tier_name TEXT NULL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX ix_fees_snapshots_portfolio ON fees_snapshots (portfolio_id);

CREATE TABLE cost_estimates (
    id BIGSERIAL PRIMARY KEY,
    order_id BIGINT REFERENCES orders(order_id) ON DELETE SET NULL,
    ex_fee NUMERIC(24, 12) NOT NULL,
    spread NUMERIC(24, 12) NOT NULL,
    slippage NUMERIC(24, 12) NOT NULL,
    transfer_fee NUMERIC(24, 12) NOT NULL,
    total_cost NUMERIC(24, 12) NOT NULL,
    decision BOOLEAN NOT NULL,
    override_flag BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX ix_cost_estimates_created_at ON cost_estimates (created_at);
