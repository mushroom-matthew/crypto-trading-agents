"""Feature schema version constants.

Bump FEATURE_SCHEMA_VERSION whenever IndicatorSnapshot fields change:
  - Fields added: increment minor (1.2.0 → 1.3.0)
  - Fields removed or renamed: increment major (1.3.0 → 2.0.0)

This constant is stored in every SetupEvent and SignalEvent row.
Training data MUST be stratified by feature_schema_version before fitting.
Never pool events across major versions.
"""

FEATURE_SCHEMA_VERSION = "1.2.0"
# History:
#   1.0.0 — initial IndicatorSnapshot (base indicators through scalper fields)
#   1.1.0 — added Fibonacci + expansion/contraction ratios
#   1.2.0 — added 15 candlestick fields (R38) + 13 htf_* fields (R41)
