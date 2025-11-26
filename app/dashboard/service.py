"""Process supervision and ledger summary services for the dashboard."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Optional

from sqlalchemy import select, func

from app.core.logging import get_logger
from app.db.models import Balance, Wallet
from app.db.repo import Database


LOG = get_logger(__name__)
LOG_DIR_NAME = "dashboard"


@dataclass
class ManagedProcess:
    name: str
    command: list[str]
    env: dict[str, str]
    cwd: Path
    process: Optional[asyncio.subprocess.Process] = None
    log_path: Optional[Path] = None
    log_handle: Optional[IO[bytes]] = None
    last_start_error: Optional[str] = None


class ProcessSupervisor:
    """Launch and monitor long-running stack processes."""

    def __init__(self, base_env: dict[str, str], workspace: Path) -> None:
        self._base_env = base_env
        self._workspace = workspace
        self._processes: dict[str, ManagedProcess] = {}
        self._log_dir = self._workspace / "logs" / LOG_DIR_NAME
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def register(self, proc: ManagedProcess) -> None:
        self._processes[proc.name] = proc

    async def start(self, name: str) -> None:
        proc = self._processes[name]
        if proc.process and proc.process.returncode is None:
            return
        if proc.log_path is None:
            proc.log_path = self._log_dir / f"{proc.name}.log"
        env = {**self._base_env, **proc.env}
        LOG.info("Starting process", name=name, cmd=proc.command, log=str(proc.log_path))
        try:
            proc.log_handle = open(proc.log_path, "ab", buffering=0)
            proc.process = await asyncio.create_subprocess_exec(
                *proc.command,
                env=env,
                cwd=str(proc.cwd),
                stdout=proc.log_handle,
                stderr=asyncio.subprocess.STDOUT,
            )
            proc.last_start_error = None
        except Exception as exc:  # pragma: no cover - surfaced in UI
            LOG.error("Failed to start process", name=name, error=str(exc))
            proc.last_start_error = str(exc)
            if proc.log_handle:
                proc.log_handle.close()
                proc.log_handle = None
            raise

    async def stop(self, name: str) -> None:
        proc = self._processes[name]
        if proc.process and proc.process.returncode is None:
            LOG.info("Stopping process", name=name)
            proc.process.terminate()
            try:
                await asyncio.wait_for(proc.process.wait(), timeout=10)
            except asyncio.TimeoutError:
                proc.process.kill()
        if proc.log_handle:
            proc.log_handle.close()
            proc.log_handle = None
        proc.process = None

    def status(self, name: str) -> dict[str, Optional[str]]:
        proc = self._processes[name]
        running = proc.process and proc.process.returncode is None
        return {
            "name": proc.name,
            "running": bool(running),
            "pid": proc.process.pid if proc.process else None,
            "returncode": proc.process.returncode if proc.process else None,
            "log_path": str(proc.log_path) if proc.log_path else None,
            "last_start_error": proc.last_start_error,
        }

    def all_status(self) -> list[dict[str, Optional[str]]]:
        return [self.status(name) for name in self._processes]


class LedgerSummaryService:
    """Read-only ledger summaries for the dashboard."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def wallet_overview(self) -> list[dict[str, str]]:
        async with self._db.session() as session:
            stmt = (
                select(
                    Wallet.wallet_id,
                    Wallet.name,
                    Wallet.coinbase_account_id,
                    Wallet.portfolio_id,
                    Wallet.tradeable_fraction,
                    Wallet.type,
                    func.coalesce(Balance.currency, "").label("currency"),
                    func.coalesce(Balance.available, 0).label("available"),
                    func.coalesce(Balance.hold, 0).label("hold"),
                )
                .outerjoin(Balance, Balance.wallet_id == Wallet.wallet_id)
                .order_by(Wallet.wallet_id, Balance.currency)
            )
            result = await session.execute(stmt)
            rows = result.fetchall()

        summaries: dict[int, dict[str, object]] = {}
        for row in rows:
            entry = summaries.setdefault(
                row.wallet_id,
                {
                    "wallet_id": row.wallet_id,
                    "name": row.name,
                    "coinbase_account_id": row.coinbase_account_id,
                    "portfolio_id": row.portfolio_id,
                    "tradeable_fraction": str(row.tradeable_fraction),
                    "type": row.type.value if row.type else None,
                    "balances": [],
                },
            )
            if row.currency:
                entry["balances"].append(
                    {
                        "currency": row.currency,
                        "available": str(row.available),
                        "hold": str(row.hold),
                    }
                )
        return list(summaries.values())
