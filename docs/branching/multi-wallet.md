# Branch: multi-wallet

## Purpose
Add multi-wallet architecture with Phantom/Solana and Ethereum read-only support, plus reconciliation and UI updates.

## Source Plans
- docs/MULTI_WALLET_INTEGRATION_PLAN.md (Phases 1-7)

## Scope
- Wallet schema updates and migration (new wallet types, blockchain/public_address fields).
- Wallet provider abstraction and Solana/Ethereum provider stubs.
- Reconciliation updated to use provider factory across wallets.
- CLI/API endpoints for adding wallets and fetching transactions.
- UI updates for multi-wallet display and add-wallet flows.
- Settings/env updates and dependency additions for Solana SDK.
- Tests and documentation updates.

## Out of Scope / Deferred
- Phase 2 trading integration with browser signing.
- DEX integrations or real on-chain trade execution.

## Key Files
- app/db/models.py
- app/db/migrations/versions/0003_multi_wallet_support.py
- app/wallets/**
- app/ledger/reconciliation.py
- app/cli/main.py
- ops_api/routers/wallets.py
- app/core/config.py
- pyproject.toml
- ui/src/components/WalletReconciliation.tsx
- ui/src/components/AddWalletDialog.tsx
- ui/src/components/WalletCard.tsx
- ui/src/lib/api.ts
- docs/MULTI_WALLET_GUIDE.md
- docs/WALLET_PROVIDER_ARCHITECTURE.md

## Dependencies / Coordination
- Coordinate with aws-deploy if Secrets Manager or env vars overlap.
- Ensure migrations are isolated and backward compatible with Coinbase wallets.

## Acceptance Criteria
- Phantom/Solana wallets can be added via CLI/API using public address.
- Reconciliation supports Coinbase + Phantom wallets in one pass.
- UI displays wallet type badges, addresses, and balances for all wallet types.
- Tests cover provider factory, Phantom balances, and multi-wallet reconciliation.
- Documentation added for operators and developers.

## Test Plan (required before commit)
- uv run pytest tests/wallets/test_phantom_provider.py -vv
- uv run pytest tests/wallets/test_factory.py -vv
- uv run pytest tests/ledger/test_multi_wallet_reconciliation.py -vv
- uv run python -c "from app.wallets import factory"

If tests cannot be run, obtain user-run output and paste it below before committing.

## Human Verification (required)
- Use the CLI to add a Phantom/Solana wallet (test address) and run reconciliation; confirm balances show.
- Open the UI and verify the wallet card displays blockchain type, truncated address, and balances.
- Paste CLI output and UI observations in the Human Verification Evidence section.

## Git Workflow (explicit)
```bash
# Start from updated main
git checkout main
git pull

# Create branch
git checkout -b multi-wallet

# Work, then review changes
git status
git diff

# Stage changes (adjust list as needed based on actual edits)
git add app/db/models.py \
  app/db/migrations/versions/0003_multi_wallet_support.py \
  app/wallets \
  app/ledger/reconciliation.py \
  app/cli/main.py \
  ops_api/routers/wallets.py \
  app/core/config.py \
  pyproject.toml \
  ui/src/components/WalletReconciliation.tsx \
  ui/src/components/AddWalletDialog.tsx \
  ui/src/components/WalletCard.tsx \
  ui/src/lib/api.ts \
  docs/MULTI_WALLET_GUIDE.md \
  docs/WALLET_PROVIDER_ARCHITECTURE.md

# Run tests (must succeed or be explicitly approved by user with pasted output)
uv run pytest tests/wallets/test_phantom_provider.py -vv
uv run pytest tests/wallets/test_factory.py -vv
uv run pytest tests/ledger/test_multi_wallet_reconciliation.py -vv
uv run python -c "from app.wallets import factory"

# Commit ONLY after test evidence is captured below
git commit -m "Wallets: multi-provider support and Phantom reconciliation"
```

## Change Log (update during implementation)
- YYYY-MM-DD: Summary of changes, files touched, and decisions.

## Test Evidence (append results before commit)

## Human Verification Evidence (append results before commit when required)

