"""Shared utilities for Temporal client connection and management."""

import os
import asyncio
from typing import Optional
from temporalio.client import Client, TLSConfig
from agents.constants import DEFAULT_TEMPORAL_ADDRESS, DEFAULT_TEMPORAL_NAMESPACE
from agents.logging_utils import get_logger

logger = get_logger(__name__)

# Shared Temporal client instance
_temporal_client: Optional[Client] = None
_client_lock = asyncio.Lock()


def _build_tls_config() -> Optional[TLSConfig]:
    """Build TLSConfig from env vars when Temporal Cloud certs are present.

    Supports two modes:
    - Inline content (Fly.io / ECS): set TEMPORAL_TLS_CERT_CONTENT and
      TEMPORAL_TLS_KEY_CONTENT to the PEM text directly (no filesystem needed).
    - File paths (local dev): set TEMPORAL_TLS_CERT and TEMPORAL_TLS_KEY to
      paths of the client cert and private key.
    When none are set, returns None and the client connects without TLS.
    """
    cert_content = os.environ.get("TEMPORAL_TLS_CERT_CONTENT")
    key_content = os.environ.get("TEMPORAL_TLS_KEY_CONTENT")
    if cert_content and key_content:
        return TLSConfig(
            client_cert=cert_content.encode(),
            client_private_key=key_content.encode(),
        )

    cert_path = os.environ.get("TEMPORAL_TLS_CERT")
    key_path = os.environ.get("TEMPORAL_TLS_KEY")
    if not (cert_path and key_path):
        return None
    with open(cert_path, "rb") as fh:
        client_cert = fh.read()
    with open(key_path, "rb") as fh:
        client_private_key = fh.read()
    return TLSConfig(client_cert=client_cert, client_private_key=client_private_key)


async def get_temporal_client(
    address: Optional[str] = None,
    namespace: Optional[str] = None
) -> Client:
    """Get or create a singleton Temporal client connection.

    Automatically enables mTLS when TEMPORAL_TLS_CERT and TEMPORAL_TLS_KEY
    env vars are set (Temporal Cloud).  Falls back to plain gRPC for local dev.

    Parameters
    ----------
    address:
        Temporal server address, defaults to TEMPORAL_ADDRESS env var
    namespace:
        Temporal namespace, defaults to TEMPORAL_NAMESPACE env var

    Returns
    -------
    Client
        Connected Temporal client instance
    """
    global _temporal_client

    if _temporal_client is None:
        async with _client_lock:
            # Double-check inside lock
            if _temporal_client is None:
                temporal_address = address or os.environ.get("TEMPORAL_ADDRESS", DEFAULT_TEMPORAL_ADDRESS)
                temporal_namespace = namespace or os.environ.get("TEMPORAL_NAMESPACE", DEFAULT_TEMPORAL_NAMESPACE)
                tls = _build_tls_config()

                if tls:
                    logger.info(
                        "Connecting to Temporal Cloud at %s (ns=%s) with mTLS",
                        temporal_address, temporal_namespace,
                    )
                else:
                    logger.info("Connecting to Temporal at %s (ns=%s)", temporal_address, temporal_namespace)

                _temporal_client = await Client.connect(
                    temporal_address, namespace=temporal_namespace, tls=tls
                )
                logger.info("Temporal client ready")

    return _temporal_client


async def connect_temporal(
    address: Optional[str] = None,
    namespace: Optional[str] = None
) -> Client:
    """Create a new Temporal client connection (non-singleton).

    Automatically enables mTLS when TEMPORAL_TLS_CERT and TEMPORAL_TLS_KEY
    env vars are set (Temporal Cloud).

    Parameters
    ----------
    address:
        Temporal server address, defaults to TEMPORAL_ADDRESS env var
    namespace:
        Temporal namespace, defaults to TEMPORAL_NAMESPACE env var

    Returns
    -------
    Client
        New Temporal client instance
    """
    temporal_address = address or os.environ.get("TEMPORAL_ADDRESS", DEFAULT_TEMPORAL_ADDRESS)
    temporal_namespace = namespace or os.environ.get("TEMPORAL_NAMESPACE", DEFAULT_TEMPORAL_NAMESPACE)
    tls = _build_tls_config()

    if tls:
        logger.info(
            "Creating new Temporal Cloud connection to %s (ns=%s) with mTLS",
            temporal_address, temporal_namespace,
        )
    else:
        logger.info("Creating new Temporal connection to %s (ns=%s)", temporal_address, temporal_namespace)

    return await Client.connect(temporal_address, namespace=temporal_namespace, tls=tls)