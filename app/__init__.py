"""Core application package for Coinbase-enabled trading stack."""

from importlib import metadata


def get_version() -> str:
    """Return the installed package version."""
    try:
        return metadata.version("crypto-trading-agents")
    except metadata.PackageNotFoundError:  # pragma: no cover - fallback for editable installs
        return "0.0.0"


__all__ = ["get_version"]
