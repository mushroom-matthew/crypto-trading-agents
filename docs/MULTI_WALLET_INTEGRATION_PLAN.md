# Multi-Wallet Integration Plan: Phantom + Multi-Chain Support

## Executive Summary

Add support for Phantom wallet and multi-chain wallet architecture to the crypto trading system. This plan implements a **hybrid approach**: read-only balance monitoring now, with architecture designed for future trading integration. Supports multiple Solana wallets (Phantom, Solflare) and Ethereum wallets, with manual signing through browser extensions for future trading.

**Timeline**: 1-2 months
**Scope**: Phase 1 (Read-only) + Architecture for Phase 2 (Trading)

---

## Current State Analysis

### Existing Wallet Architecture
- **Database**: `wallets` table with enum `wallet_type` (COINBASE_SPOT, COINBASE_TRADING, EXTERNAL)
- **Models**: `app/db/models.py` - `Wallet` class with Coinbase-specific fields (`coinbase_account_id`, `portfolio_id`)
- **Integration**: `app/coinbase/client.py` - CoinbaseClient handles authentication and API calls
- **Reconciliation**: `app/ledger/reconciliation.py` - Reconciler queries Coinbase for balances
- **CLI**: `app/cli/main.py` - Commands for seeding wallets from Coinbase
- **UI**: `ui/src/components/WalletReconciliation.tsx` - Wallet reconciliation dashboard

### Limitations
1. Wallet model is tightly coupled to Coinbase
2. WalletType enum is hardcoded in database schema
3. No abstraction for different blockchain providers
4. Reconciliation logic assumes Coinbase API

---

## Implementation Plan

### Phase 1: Architecture Foundation (Week 1-2)

#### 1.1 Database Schema Evolution

**Objective**: Make wallet model blockchain-agnostic

**Changes to `app/db/models.py`:**
```python
class WalletType(enum.Enum):
    # Existing
    COINBASE_SPOT = "COINBASE_SPOT"
    COINBASE_TRADING = "COINBASE_TRADING"
    EXTERNAL = "EXTERNAL"

    # New - Solana wallets
    PHANTOM = "PHANTOM"
    SOLFLARE = "SOLFLARE"
    SOLANA_GENERIC = "SOLANA_GENERIC"

    # New - Ethereum wallets
    METAMASK = "METAMASK"
    ETHEREUM_GENERIC = "ETHEREUM_GENERIC"
```

**New fields for `Wallet` model:**
```python
class Wallet(Base):
    # Existing fields...
    coinbase_account_id: Mapped[Optional[str]]  # Keep for backward compat
    portfolio_id: Mapped[Optional[str]]  # Keep for backward compat

    # New generic fields
    blockchain: Mapped[Optional[str]]  # 'solana', 'ethereum', 'coinbase'
    public_address: Mapped[Optional[str]]  # Blockchain public key/address
    provider_metadata: Mapped[Optional[dict]]  # JSON field for provider-specific data
```

**Migration**: `app/db/migrations/versions/0003_multi_wallet_support.py`
- Add new enum values to `wallet_type`
- Add `blockchain`, `public_address`, `provider_metadata` columns
- Backfill existing wallets: set `blockchain='coinbase'`

#### 1.2 Wallet Provider Abstraction

**Create**: `app/wallets/` package

**Structure**:
```
app/wallets/
├── __init__.py
├── base.py              # Abstract base class
├── coinbase.py          # Existing Coinbase logic (refactored)
├── solana/
│   ├── __init__.py
│   ├── base.py          # Solana wallet interface
│   ├── phantom.py       # Phantom-specific implementation
│   ├── solflare.py      # Solflare-specific implementation
│   └── rpc_client.py    # Solana RPC client
└── ethereum/
    ├── __init__.py
    ├── base.py          # Ethereum wallet interface
    ├── metamask.py      # MetaMask integration
    └── web3_client.py   # Web3 client wrapper
```

**Base Interface** (`app/wallets/base.py`):
```python
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Dict, List

class WalletProvider(ABC):
    """Abstract base class for all wallet providers."""

    @abstractmethod
    async def get_balances(self, public_address: str) -> Dict[str, Decimal]:
        """Query wallet balances by public address."""
        pass

    @abstractmethod
    async def get_transactions(
        self,
        public_address: str,
        limit: int = 100
    ) -> List[Transaction]:
        """Get transaction history."""
        pass

    @abstractmethod
    async def validate_address(self, address: str) -> bool:
        """Validate if address is correctly formatted."""
        pass

    # Future: Trading support
    # @abstractmethod
    # async def sign_transaction(self, tx: Transaction) -> SignedTransaction:
    #     pass
```

#### 1.3 Solana/Phantom Integration

**Dependencies** (add to `pyproject.toml`):
```toml
solana = "^0.36.0"  # Solana Python SDK
solders = "^0.27.0"  # Solana SDK (already exists!)
anchorpy = "^0.20.0"  # Optional: For Solana program interactions
```

**Implementation** (`app/wallets/solana/phantom.py`):
```python
from app.wallets.solana.base import SolanaWalletProvider
from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey

class PhantomWalletProvider(SolanaWalletProvider):
    """Phantom wallet integration via Solana RPC."""

    def __init__(self, rpc_url: str = "https://api.mainnet-beta.solana.com"):
        self.client = AsyncClient(rpc_url)

    async def get_balances(self, public_address: str) -> Dict[str, Decimal]:
        """Query SOL and SPL token balances."""
        pubkey = Pubkey.from_string(public_address)

        # Get SOL balance
        sol_balance = await self.client.get_balance(pubkey)

        # Get SPL token accounts
        token_accounts = await self.client.get_token_accounts_by_owner(
            pubkey,
            {"programId": TOKEN_PROGRAM_ID}
        )

        # Parse and return balances
        balances = {"SOL": Decimal(sol_balance.value) / 10**9}
        # ... parse token accounts ...
        return balances
```

### Phase 2: Reconciliation Update (Week 2-3)

#### 2.1 Wallet Provider Factory

**Create**: `app/wallets/factory.py`
```python
from app.wallets.base import WalletProvider
from app.wallets.coinbase import CoinbaseWalletProvider
from app.wallets.solana.phantom import PhantomWalletProvider
from app.wallets.ethereum.metamask import MetaMaskProvider
from app.db.models import WalletType

class WalletProviderFactory:
    """Factory to create appropriate wallet provider."""

    @staticmethod
    def create(wallet_type: WalletType, **kwargs) -> WalletProvider:
        providers = {
            WalletType.COINBASE_SPOT: CoinbaseWalletProvider,
            WalletType.COINBASE_TRADING: CoinbaseWalletProvider,
            WalletType.PHANTOM: PhantomWalletProvider,
            WalletType.SOLFLARE: SolflareWalletProvider,
            WalletType.METAMASK: MetaMaskProvider,
        }

        provider_class = providers.get(wallet_type)
        if not provider_class:
            raise ValueError(f"Unsupported wallet type: {wallet_type}")

        return provider_class(**kwargs)
```

#### 2.2 Multi-Provider Reconciliation

**Update**: `app/ledger/reconciliation.py`

**Changes**:
- Replace direct Coinbase API calls with `WalletProviderFactory`
- Loop through all wallets and use appropriate provider
- Support different blockchain explorers for drift detection

```python
class Reconciler:
    async def reconcile(
        self,
        threshold: Decimal = Decimal("0.0001")
    ) -> ReconciliationReport:
        """Reconcile all wallets across all providers."""

        async with self._db.session() as session:
            wallets = await session.execute(select(Wallet))
            wallets = wallets.scalars().all()

        drift_records = []
        for wallet in wallets:
            provider = WalletProviderFactory.create(
                wallet.type,
                settings=self._settings
            )

            # Get blockchain balance
            if wallet.public_address:
                balances = await provider.get_balances(wallet.public_address)
            else:
                # Legacy Coinbase flow
                balances = await provider.get_balances_by_account_id(
                    wallet.coinbase_account_id
                )

            # Compare with ledger
            for currency, blockchain_balance in balances.items():
                ledger_balance = await self._get_ledger_balance(
                    wallet.wallet_id,
                    currency
                )
                drift = blockchain_balance - ledger_balance

                drift_records.append(DriftRecord(
                    wallet_id=wallet.wallet_id,
                    wallet_name=wallet.name,
                    currency=currency,
                    ledger_balance=ledger_balance,
                    blockchain_balance=blockchain_balance,
                    drift=drift,
                    within_threshold=abs(drift) <= threshold
                ))

        return ReconciliationReport(entries=drift_records)
```

### Phase 3: CLI & API Updates (Week 3-4)

#### 3.1 CLI Commands

**Add to `app/cli/main.py`**:
```python
@wallet_group.command()
def add_phantom(
    public_address: str,
    name: Optional[str] = None,
    tradeable_fraction: Decimal = Decimal("0")
):
    """Add a Phantom wallet by public address."""
    # Validate Solana address
    # Create wallet entry
    # Query initial balances
    pass

@wallet_group.command()
def add_solana(
    public_address: str,
    provider: str = "phantom",  # phantom, solflare, generic
    name: Optional[str] = None
):
    """Add a Solana wallet (generic or specific provider)."""
    pass

@wallet_group.command()
def add_ethereum(
    public_address: str,
    provider: str = "metamask",
    name: Optional[str] = None
):
    """Add an Ethereum wallet."""
    pass
```

#### 3.2 API Endpoints

**Update**: `ops_api/routers/wallets.py`

**New endpoints**:
```python
@router.post("/wallets/add-phantom")
async def add_phantom_wallet(
    public_address: str,
    name: Optional[str] = None
):
    """Add Phantom wallet via API."""
    # Validate address
    # Create wallet
    # Query balances
    # Return wallet info
    pass

@router.get("/wallets/{wallet_id}/transactions")
async def get_wallet_transactions(
    wallet_id: int,
    limit: int = 100
):
    """Get transaction history for any wallet type."""
    # Get wallet
    # Create provider
    # Query transactions
    pass
```

### Phase 4: UI Updates (Week 4-5)

#### 4.1 Multi-Wallet UI

**Update**: `ui/src/components/WalletReconciliation.tsx`

**Changes**:
- Display blockchain type badge (Coinbase/Solana/Ethereum)
- Show public address for blockchain wallets
- Add "Add Wallet" button with provider selector
- Transaction history viewer

**New Component**: `ui/src/components/AddWalletDialog.tsx`
```tsx
export function AddWalletDialog() {
  const [provider, setProvider] = useState<'phantom' | 'solflare' | 'metamask'>();
  const [publicAddress, setPublicAddress] = useState('');

  // Form to add new wallet
  // Validates address format
  // Calls POST /wallets/add-{provider}
}
```

#### 4.2 Wallet Cards

**New**: `ui/src/components/WalletCard.tsx`
- Display wallet icon (Phantom logo, Coinbase logo, etc.)
- Show blockchain type
- Display public address (truncated) with copy button
- Balance list with currency icons
- Transaction history link

### Phase 5: Configuration & Environment (Week 5-6)

#### 5.1 Environment Variables

**Add to `.env.example`:**
```bash
# Solana Configuration
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
SOLANA_RPC_TIMEOUT=30

# Ethereum Configuration
ETHEREUM_RPC_URL=https://mainnet.infura.io/v3/YOUR_PROJECT_ID
ETHEREUM_CHAIN_ID=1

# Wallet Provider Settings
ENABLE_PHANTOM=true
ENABLE_SOLFLARE=true
ENABLE_METAMASK=false  # Future
```

#### 5.2 Settings Update

**Update**: `app/core/config.py`
```python
class Settings(BaseSettings):
    # Existing Coinbase settings...

    # Solana settings
    solana_rpc_url: str = "https://api.mainnet-beta.solana.com"
    solana_rpc_timeout: int = 30

    # Ethereum settings (future)
    ethereum_rpc_url: Optional[str] = None
    ethereum_chain_id: int = 1

    # Feature flags
    enable_phantom: bool = True
    enable_solflare: bool = True
    enable_metamask: bool = False
```

### Phase 6: Testing (Week 6-7)

#### 6.1 Unit Tests

**Create**:
- `tests/wallets/test_phantom_provider.py` - Mock Solana RPC responses
- `tests/wallets/test_factory.py` - Provider factory logic
- `tests/ledger/test_multi_wallet_reconciliation.py` - Multi-provider reconciliation

#### 6.2 Integration Tests

**Create**:
- `tests/integration/test_phantom_balance_query.py` - Live Solana testnet queries
- `tests/integration/test_wallet_seeding.py` - CLI wallet addition flow

### Phase 7: Documentation (Week 7-8)

#### 7.1 User Documentation

**Update**: `README.md`
- Multi-wallet support section
- How to add Phantom/Solana wallets
- Reconciliation with multiple providers

**Create**: `docs/MULTI_WALLET_GUIDE.md`
- Detailed guide for each wallet type
- Provider comparison table
- Address validation requirements
- Troubleshooting

#### 7.2 Developer Documentation

**Create**: `docs/WALLET_PROVIDER_ARCHITECTURE.md`
- How wallet providers work
- Adding new blockchain support
- Provider interface documentation

---

## Critical Files to Modify

### Backend
1. `app/db/models.py` - Add wallet fields and enum values
2. `app/db/migrations/versions/0003_multi_wallet_support.py` - NEW migration
3. `app/wallets/` - NEW package (entire directory structure)
4. `app/ledger/reconciliation.py` - Multi-provider reconciliation
5. `app/cli/main.py` - Add wallet commands
6. `ops_api/routers/wallets.py` - New endpoints
7. `app/core/config.py` - Solana/Ethereum settings
8. `pyproject.toml` - Add Solana dependencies

### Frontend
9. `ui/src/components/WalletReconciliation.tsx` - Multi-wallet display
10. `ui/src/components/AddWalletDialog.tsx` - NEW component
11. `ui/src/components/WalletCard.tsx` - NEW component
12. `ui/src/lib/api.ts` - Add wallet endpoints

### Documentation
13. `README.md` - Update with multi-wallet info
14. `docs/MULTI_WALLET_GUIDE.md` - NEW
15. `docs/WALLET_PROVIDER_ARCHITECTURE.md` - NEW

---

## Dependencies to Add

```toml
[tool.poetry.dependencies]
# Solana SDK (already exists: solders = "^0.27.1")
solana = "^0.36.0"  # Full Solana SDK
# anchorpy = "^0.20.0"  # Optional: Solana program interactions

# Ethereum (already exists: web3 = "^6.0.0")
# eth-account = "^0.13.0"  # Already exists

# Utilities
base58 = "^2.1.0"  # Already exists - for address encoding
```

---

## Risk Assessment

### Low Risk
- ✅ Read-only balance queries (no trading risk)
- ✅ Backward compatible database changes
- ✅ Existing Coinbase functionality unchanged

### Medium Risk
- ⚠️ RPC endpoint reliability (public Solana RPCs can be slow/unreliable)
- ⚠️ Address validation (incorrect addresses could cause errors)
- ⚠️ Rate limiting on public RPC endpoints

### Mitigation Strategies
1. **RPC Reliability**: Support multiple RPC endpoints with fallback
2. **Address Validation**: Strict validation before saving to database
3. **Rate Limiting**: Implement request caching and rate limiting
4. **Error Handling**: Graceful degradation if blockchain RPC is down

---

## Future Enhancements (Phase 2: Trading)

### Trading Integration Architecture (Not in initial scope)

**Browser Extension Integration**:
1. User connects Phantom extension to UI
2. UI requests transaction signature
3. Extension prompts user to approve
4. Signed transaction sent to backend
5. Backend submits to blockchain

**Required Changes**:
- `ui/src/hooks/usePhantomWallet.ts` - Browser extension detection
- WebSocket for transaction signing flow
- Transaction builder in backend
- Solana DEX integration (Jupiter, Raydium, etc.)

---

## Success Criteria

### Phase 1 Complete When:
- [ ] Can add Phantom wallet by public address via CLI
- [ ] Balance queries work for Phantom wallets
- [ ] Reconciliation supports Phantom + Coinbase simultaneously
- [ ] UI displays all wallet types with correct icons/labels
- [ ] Transaction history visible for Phantom wallets
- [ ] Tests cover new wallet providers
- [ ] Documentation complete

### Future Phase 2 Goals:
- [ ] Browser extension integration
- [ ] Transaction signing workflow
- [ ] Solana DEX trading support
- [ ] Multi-wallet portfolio view

---

## Estimated Effort

**Total**: 6-8 weeks (1-2 months)

**Breakdown**:
- Week 1-2: Database schema + Provider abstraction (16-24 hours)
- Week 2-3: Solana/Phantom integration + Reconciliation (16-24 hours)
- Week 3-4: CLI + API endpoints (12-16 hours)
- Week 4-5: UI components (16-20 hours)
- Week 5-6: Configuration + Integration (8-12 hours)
- Week 6-7: Testing (12-16 hours)
- Week 7-8: Documentation + Polish (8-12 hours)

**Total Effort**: 88-124 hours (~3 months at part-time pace)

---

## Notes

- Solana SDK (`solders`) already exists in dependencies!
- Web3 library already exists for Ethereum support
- Architecture supports adding more chains (Cardano, Polkadot, etc.) easily
- Manual signing design prevents private key storage risks
- Read-only approach is safe for initial deployment
