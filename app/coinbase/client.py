"""Shared Coinbase API client utilities."""

from __future__ import annotations

import base64
import hmac
import json
import time
from typing import Any, Mapping, MutableMapping, Optional, Literal
from urllib.parse import urlencode, urlparse
import hashlib

import httpx
from cdp.auth.utils.http import GetAuthHeadersOptions, get_auth_headers
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from tenacity import AsyncRetrying, RetryError, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import Settings, get_settings
from app.core.errors import CoinbaseAPIError
from app.core.idempotency import IdempotencyKey
from app.core.logging import get_logger


LOG = get_logger(__name__)


class CoinbaseClient:
    """Async Coinbase API client with request signing and retries."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._client: Optional[httpx.AsyncClient] = None
        self._key_preview = self._preview_secret(self._settings.coinbase_api_key.get_secret_value())
        self._secret_preview = self._preview_secret(self._settings.coinbase_api_secret.get_secret_value())
        self._passphrase_preview = (
            self._preview_secret(self._settings.coinbase_passphrase.get_secret_value())
            if self._settings.coinbase_passphrase
            else None
        )
        self._sign_mode: Literal["hmac", "ecdsa", "cdp"] = "hmac"
        self._hmac_secret: Optional[bytes] = None
        self._ecdsa_key: Optional[ec.EllipticCurvePrivateKey] = None
        self._base_host = urlparse(str(self._settings.coinbase_base_url)).hostname or ""
        self._use_cdp_auth = self._should_use_cdp_auth()
        if self._use_cdp_auth:
            self._sign_mode = "cdp"
        else:
            self._initialise_signing_material()

    async def __aenter__(self) -> "CoinbaseClient":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        await self.close()

    @staticmethod
    def _preview_secret(value: str, visible: int = 4) -> str:
        if not value:
            return ""
        if len(value) <= visible * 2:
            return value
        return f"{value[:visible]}...{value[-visible:]}"

    def _should_use_cdp_auth(self) -> bool:
        """Detect whether the configured credentials require CDP JWT auth."""

        api_key_value = self._settings.coinbase_api_key.get_secret_value()
        secret_value = self._settings.coinbase_api_secret.get_secret_value()
        return api_key_value.startswith("organizations/") or "BEGIN" in secret_value

    def _initialise_signing_material(self) -> None:
        secret_value = self._settings.coinbase_api_secret.get_secret_value()
        try:
            self._ecdsa_key = serialization.load_pem_private_key(
                secret_value.encode("utf-8"),
                password=None,
            )
            self._sign_mode = "ecdsa"
        except ValueError:
            # Fallback to HMAC (legacy key format)
            try:
                self._hmac_secret = base64.b64decode(secret_value)
            except Exception:
                self._hmac_secret = secret_value.encode("utf-8")
            self._sign_mode = "hmac"

    async def start(self) -> None:
        """Initialise the underlying httpx client."""

        if self._client is None:
            timeout = httpx.Timeout(self._settings.http_timeout_seconds)
            self._client = httpx.AsyncClient(base_url=str(self._settings.coinbase_base_url), timeout=timeout)

    async def close(self) -> None:
        """Dispose of the HTTP client."""

        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get(
        self,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Issue an authenticated GET request."""

        return await self._request("GET", path, params=params)

    async def post(
        self,
        path: str,
        *,
        json_payload: Optional[Mapping[str, Any]] = None,
        idempotency_key: Optional[IdempotencyKey | str] = None,
    ) -> dict[str, Any]:
        """Issue an authenticated POST request."""

        return await self._request("POST", path, json_payload=json_payload, idempotency_key=idempotency_key)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json_payload: Optional[Mapping[str, Any]] = None,
        idempotency_key: Optional[IdempotencyKey | str] = None,
    ) -> dict[str, Any]:
        """Issue a signed request with retry/backoff."""

        if self._client is None:
            raise RuntimeError("Client not started")

        body_dict = json_payload or None
        body = json.dumps(body_dict, separators=(",", ":"), sort_keys=True) if body_dict else ""
        query = f"?{urlencode(sorted(params.items()))}" if params else ""
        signed_path = f"{path}{query}"
        headers = self._build_headers(method, signed_path, body, body_dict)

        if idempotency_key:
            key_value = idempotency_key.value if isinstance(idempotency_key, IdempotencyKey) else idempotency_key
            headers["Idempotency-Key"] = key_value

        async def do_request() -> dict[str, Any]:
            response = await self._client.request(
                method,
                path,
                params=params,
                content=body if body else None,
                headers=headers,
            )
            if response.status_code >= 400:
                payload = self._safe_json(response)
                log_context = {
                    "method": method,
                    "base_url": str(self._client.base_url),
                    "path": path,
                    "signed_path": signed_path,
                    "params": params,
                    "body": body or None,
                    "status": response.status_code,
                    "auth_mode": self._sign_mode,
                    "key_preview": self._key_preview,
                    "secret_preview": self._secret_preview,
                    "signature_preview": self._preview_secret(headers.get("CB-ACCESS-SIGN", ""), visible=6),
                    "timestamp": headers.get("CB-ACCESS-TIMESTAMP"),
                    "passphrase_present": bool(self._settings.coinbase_passphrase),
                    "portfolio_header": headers.get("CB-ACCESS-PORTFOLIO"),
                }
                LOG.warning(
                    "Coinbase request failed",
                    payload=payload,
                    **log_context,
                )
                if response.status_code in (429, 500, 502, 503, 504):
                    raise TemporaryCoinbaseError(response.status_code, payload)
                detail = payload.get("message") if isinstance(payload, dict) else None
                message = f"Coinbase API returned error ({response.status_code})"
                if detail:
                    message = f"{message}: {detail}"
                raise CoinbaseAPIError(message, response.status_code, payload=payload)
            return self._safe_json(response)

        try:
            async for attempt in AsyncRetrying(
                reraise=True,
                stop=stop_after_attempt(self._settings.http_retry_attempts),
                wait=wait_exponential(multiplier=self._settings.http_retry_backoff_seconds, max=10),
                retry=retry_if_exception_type(TemporaryCoinbaseError),
            ):
                with attempt:
                    return await do_request()
        except RetryError as exc:
            err = exc.last_attempt.exception()
            if isinstance(err, TemporaryCoinbaseError):
                raise CoinbaseAPIError(
                    f"Coinbase request failed after retries ({err.status_code})",
                    err.status_code,
                    payload=err.payload,
                ) from err
            raise

    def _build_headers(
        self,
        method: str,
        path: str,
        body: str,
        body_payload: Optional[Mapping[str, Any]],
    ) -> MutableMapping[str, str]:
        if self._use_cdp_auth:
            return self._build_cdp_headers(method, path, body_payload)
        return self._build_legacy_headers(method, path, body)

    def _build_cdp_headers(
        self,
        method: str,
        path: str,
        body_payload: Optional[Mapping[str, Any]],
    ) -> MutableMapping[str, str]:
        wallet_secret = (
            self._settings.coinbase_wallet_secret.get_secret_value()
            if self._settings.coinbase_wallet_secret
            else None
        )
        options = GetAuthHeadersOptions(
            api_key_id=self._settings.coinbase_api_key.get_secret_value(),
            api_key_secret=self._settings.coinbase_api_secret.get_secret_value(),
            request_method=method.upper(),
            request_host=self._base_host,
            request_path=path,
            request_body=dict(body_payload) if body_payload is not None else None,
            wallet_secret=wallet_secret,
            source="cta",
        )
        headers: MutableMapping[str, str] = get_auth_headers(options)
        if self._settings.coinbase_portfolio_id:
            headers["CB-ACCESS-PORTFOLIO"] = self._settings.coinbase_portfolio_id
        return headers

    def _build_legacy_headers(self, method: str, path: str, body: str) -> MutableMapping[str, str]:
        timestamp = str(int(time.time()))
        message = f"{timestamp}{method.upper()}{path}{body}"
        if self._sign_mode == "ecdsa" and self._ecdsa_key:
            signature = self._ecdsa_key.sign(
                message.encode("utf-8"),
                ec.ECDSA(hashes.SHA256()),
            )
            signature_b64 = base64.b64encode(signature).decode("utf-8")
        else:
            if not self._hmac_secret:
                secret = self._settings.coinbase_api_secret.get_secret_value().encode("utf-8")
            else:
                secret = self._hmac_secret
            signature = hmac.new(secret, message.encode("utf-8"), hashlib.sha256).digest()
            signature_b64 = base64.b64encode(signature).decode("utf-8")

        headers: MutableMapping[str, str] = {
            "CB-ACCESS-KEY": self._settings.coinbase_api_key.get_secret_value(),
            "CB-ACCESS-SIGN": signature_b64,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }
        if self._settings.coinbase_passphrase:
            headers["CB-ACCESS-PASSPHRASE"] = self._settings.coinbase_passphrase.get_secret_value()
        if self._settings.coinbase_portfolio_id:
            headers["CB-ACCESS-PORTFOLIO"] = self._settings.coinbase_portfolio_id
        return headers

    @staticmethod
    def _safe_json(response: httpx.Response) -> dict[str, Any]:
        try:
            return response.json()
        except json.JSONDecodeError:
            LOG.warning("Failed to decode JSON response", status_code=response.status_code, text=response.text[:256])
            return {}


class TemporaryCoinbaseError(RuntimeError):
    """Internal helper used for retry logic."""

    def __init__(self, status_code: int, payload: Optional[dict[str, Any]] = None) -> None:
        super().__init__(f"Temporary Coinbase error {status_code}")
        self.status_code = status_code
        self.payload = payload or {}


__all__ = ["CoinbaseClient"]
