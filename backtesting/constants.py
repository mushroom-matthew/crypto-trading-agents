"""Backtesting constants and versioning.

Bump ENGINE_SEMVER on each merged runbook that changes strategy behavior.
Training data MUST be stratified by ENGINE_SEMVER.
"""

from __future__ import annotations

import os

# Semver of the strategy engine.
# Bump when strategy-affecting runbooks merge (entry logic, risk rules, etc.)
ENGINE_SEMVER: str = os.environ.get("ENGINE_SEMVER", "0.5.0")
