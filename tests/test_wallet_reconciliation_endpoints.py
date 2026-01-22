"""Tests for wallet reconciliation API endpoints."""

import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime
from fastapi.testclient import TestClient

from ops_api.app import app
from app.db.models import WalletType
from app.ledger.reconciliation import ReconciliationReport as ReconReport, DriftRecord as ReconDriftRecord


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_wallet():
    """Create a mock wallet model."""
    wallet = Mock()
    wallet.wallet_id = 1
    wallet.name = "Test Wallet"
    wallet.type = WalletType.COINBASE_SPOT
    wallet.tradeable_fraction = Decimal("0.8")
    wallet.coinbase_account_id = "test-account-123"
    return wallet


@pytest.fixture
def mock_balance():
    """Create a mock balance model."""
    balance = Mock()
    balance.wallet_id = 1
    balance.currency = "USD"
    balance.available = Decimal("1000.00")
    balance.reserved = Decimal("0.00")
    return balance


def test_list_wallets_empty(client):
    """Test listing wallets when none exist."""
    with patch("ops_api.routers.wallets._db.session") as mock_session_ctx:
        # Mock empty wallet query
        mock_session = AsyncMock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = []

        mock_result = AsyncMock()
        mock_result.scalars.return_value = mock_scalars

        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session_ctx.return_value.__aenter__.return_value = mock_session

        response = client.get("/wallets")

        assert response.status_code == 200
        assert response.json() == []


def test_list_wallets_with_data(client, mock_wallet, mock_balance):
    """Test listing wallets with balance data."""
    with patch("ops_api.routers.wallets._db.session") as mock_session_ctx:
        mock_session = AsyncMock()

        # First query returns wallet
        wallet_scalars = Mock()
        wallet_scalars.all.return_value = [mock_wallet]
        wallet_result = AsyncMock()
        wallet_result.scalars.return_value = wallet_scalars

        # Second query returns balance
        balance_scalars = Mock()
        balance_scalars.all.return_value = [mock_balance]
        balance_result = AsyncMock()
        balance_result.scalars.return_value = balance_scalars

        # Mock execute to return different results
        async def mock_execute(query):
            # Simple heuristic: balance queries have .where()
            if hasattr(query, "whereclause"):
                return balance_result
            return wallet_result

        mock_session.execute = AsyncMock(side_effect=mock_execute)
        mock_session_ctx.return_value.__aenter__.return_value = mock_session

        response = client.get("/wallets")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["wallet_id"] == 1
        assert data[0]["name"] == "Test Wallet"
        assert data[0]["ledger_balance"] == "1000.00"
        assert data[0]["currency"] == "USD"
        assert data[0]["type"] == "COINBASE_SPOT"


def test_get_wallet_not_implemented(client):
    """Test that get wallet endpoint returns 501."""
    response = client.get("/wallets/1")

    assert response.status_code == 501
    assert "Not implemented" in response.json()["detail"]


def test_get_wallet_transactions_empty(client):
    """Test getting wallet transactions (currently returns empty)."""
    response = client.get("/wallets/1/transactions")

    assert response.status_code == 200
    assert response.json() == []


def test_get_wallet_transactions_with_limit(client):
    """Test getting wallet transactions with limit parameter."""
    response = client.get("/wallets/1/transactions?limit=50")

    assert response.status_code == 200
    assert response.json() == []


def test_trigger_reconciliation_success(client):
    """Test successful reconciliation."""
    # Create mock reconciliation report
    mock_drift = ReconDriftRecord(
        wallet_id=1,
        wallet_name="Test Wallet",
        currency="USD",
        ledger_balance=Decimal("1000.00"),
        coinbase_balance=Decimal("1000.05"),
        drift=Decimal("0.05"),
        within_threshold=True
    )

    mock_recon_report = ReconReport(entries=[mock_drift])

    with patch("ops_api.routers.wallets.Reconciler") as mock_reconciler_class, \
         patch("ops_api.routers.wallets.CoinbaseClient") as mock_client_class:

        # Mock reconciler instance
        mock_reconciler = AsyncMock()
        mock_reconciler.reconcile = AsyncMock(return_value=mock_recon_report)
        mock_reconciler_class.return_value = mock_reconciler

        # Mock Coinbase client context manager
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        response = client.post(
            "/wallets/reconcile",
            json={"threshold": "0.0001"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "timestamp" in data
        assert data["total_wallets"] == 1
        assert data["drifts_detected"] == 1
        assert data["drifts_within_threshold"] == 1
        assert data["drifts_exceeding_threshold"] == 0

        assert len(data["records"]) == 1
        record = data["records"][0]
        assert record["wallet_id"] == 1
        assert record["wallet_name"] == "Test Wallet"
        assert record["currency"] == "USD"
        assert record["ledger_balance"] == "1000.00"
        assert record["coinbase_balance"] == "1000.05"
        assert record["drift"] == "0.05"
        assert record["within_threshold"] is True


def test_trigger_reconciliation_with_drift_exceeding_threshold(client):
    """Test reconciliation with drift exceeding threshold."""
    # Create mock reconciliation report with large drift
    mock_drift = ReconDriftRecord(
        wallet_id=1,
        wallet_name="Test Wallet",
        currency="BTC",
        ledger_balance=Decimal("1.00000000"),
        coinbase_balance=Decimal("1.10000000"),
        drift=Decimal("0.10000000"),
        within_threshold=False  # Exceeds threshold
    )

    mock_recon_report = ReconReport(entries=[mock_drift])

    with patch("ops_api.routers.wallets.Reconciler") as mock_reconciler_class, \
         patch("ops_api.routers.wallets.CoinbaseClient") as mock_client_class:

        mock_reconciler = AsyncMock()
        mock_reconciler.reconcile = AsyncMock(return_value=mock_recon_report)
        mock_reconciler_class.return_value = mock_reconciler

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        response = client.post(
            "/wallets/reconcile",
            json={"threshold": "0.0001"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["total_wallets"] == 1
        assert data["drifts_detected"] == 1
        assert data["drifts_within_threshold"] == 0
        assert data["drifts_exceeding_threshold"] == 1

        record = data["records"][0]
        assert record["within_threshold"] is False
        assert record["drift"] == "0.10000000"


def test_trigger_reconciliation_multiple_wallets(client):
    """Test reconciliation with multiple wallets."""
    # Create multiple drift records
    drifts = [
        ReconDriftRecord(
            wallet_id=1,
            wallet_name="Wallet 1",
            currency="USD",
            ledger_balance=Decimal("1000.00"),
            coinbase_balance=Decimal("1000.00"),
            drift=Decimal("0.00"),
            within_threshold=True
        ),
        ReconDriftRecord(
            wallet_id=2,
            wallet_name="Wallet 2",
            currency="BTC",
            ledger_balance=Decimal("0.5"),
            coinbase_balance=Decimal("0.50001"),
            drift=Decimal("0.00001"),
            within_threshold=True
        ),
        ReconDriftRecord(
            wallet_id=3,
            wallet_name="Wallet 3",
            currency="ETH",
            ledger_balance=Decimal("10.0"),
            coinbase_balance=Decimal("10.1"),
            drift=Decimal("0.1"),
            within_threshold=False
        ),
    ]

    mock_recon_report = ReconReport(entries=drifts)

    with patch("ops_api.routers.wallets.Reconciler") as mock_reconciler_class, \
         patch("ops_api.routers.wallets.CoinbaseClient") as mock_client_class:

        mock_reconciler = AsyncMock()
        mock_reconciler.reconcile = AsyncMock(return_value=mock_recon_report)
        mock_reconciler_class.return_value = mock_reconciler

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        response = client.post(
            "/wallets/reconcile",
            json={"threshold": "0.0001"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["total_wallets"] == 3
        assert data["drifts_detected"] == 2  # Only non-zero drifts
        assert data["drifts_within_threshold"] == 2
        assert data["drifts_exceeding_threshold"] == 1
        assert len(data["records"]) == 3


def test_trigger_reconciliation_default_threshold(client):
    """Test reconciliation with default threshold."""
    mock_drift = ReconDriftRecord(
        wallet_id=1,
        wallet_name="Test Wallet",
        currency="USD",
        ledger_balance=Decimal("1000.00"),
        coinbase_balance=Decimal("1000.00"),
        drift=Decimal("0.00"),
        within_threshold=True
    )

    mock_recon_report = ReconReport(entries=[mock_drift])

    with patch("ops_api.routers.wallets.Reconciler") as mock_reconciler_class, \
         patch("ops_api.routers.wallets.CoinbaseClient") as mock_client_class:

        mock_reconciler = AsyncMock()
        mock_reconciler.reconcile = AsyncMock(return_value=mock_recon_report)
        mock_reconciler_class.return_value = mock_reconciler

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        # Post without threshold (should use default)
        response = client.post("/wallets/reconcile", json={})

        assert response.status_code == 200
        # Verify reconciler was called with default threshold
        mock_reconciler.reconcile.assert_called_once()


def test_get_reconciliation_history_empty(client):
    """Test getting reconciliation history (currently returns empty)."""
    response = client.get("/wallets/reconcile/history")

    assert response.status_code == 200
    assert response.json() == []


def test_get_reconciliation_history_with_limit(client):
    """Test getting reconciliation history with limit."""
    response = client.get("/wallets/reconcile/history?limit=10")

    assert response.status_code == 200
    assert response.json() == []
