from ops_api.routers.paper_trading import (
    _index_trade_set_triggers,
    _resolve_trade_set_rule_fields,
)


def test_index_trade_set_triggers_keeps_latest_trigger_definition():
    catalog = _index_trade_set_triggers([
        {
            "triggers": [
                {
                    "id": "BTC_MR_LONG_PROBE",
                    "entry_rule": "old_entry",
                    "exit_rule": "old_exit",
                }
            ]
        },
        {
            "triggers": [
                {
                    "id": "BTC_MR_LONG_PROBE",
                    "entry_rule": "new_entry",
                    "exit_rule": "new_exit",
                    "hold_rule": "new_hold",
                }
            ]
        },
    ])

    assert catalog["BTC_MR_LONG_PROBE"]["entry_rule"] == "new_entry"
    assert catalog["BTC_MR_LONG_PROBE"]["exit_rule"] == "new_exit"
    assert catalog["BTC_MR_LONG_PROBE"]["hold_rule"] == "new_hold"


def test_resolve_trade_set_rule_fields_prefers_overrides_then_catalog():
    fields = _resolve_trade_set_rule_fields(
        entry_tx={"trigger_id": "BTC_MR_LONG_PROBE"},
        exit_tx={"trigger_id": "BTC_MR_LONG_PROBE_exit"},
        entry_override={
            "entry_rule": "close < vwap",
            "planned_exit_rule": "target_hit",
            "hold_rule": "not stop_hit",
            "timeframe": "5m",
            "trigger_category": "mean_reversion",
        },
        exit_override={"exit_rule": "stop_hit or target_hit", "timeframe": "5m"},
        trigger_catalog={
            "BTC_MR_LONG_PROBE": {
                "id": "BTC_MR_LONG_PROBE",
                "entry_rule": "catalog_entry",
                "exit_rule": "catalog_exit",
                "category": "mean_reversion",
                "timeframe": "15m",
            },
            "BTC_MR_LONG_PROBE_exit": {
                "id": "BTC_MR_LONG_PROBE_exit",
                "exit_rule": "ema_fast < ema_very_fast",
                "timeframe": "1m",
            },
        },
    )

    assert fields["entry_trigger"] == "BTC_MR_LONG_PROBE"
    assert fields["exit_trigger"] == "BTC_MR_LONG_PROBE_exit"
    assert fields["entry_rule"] == "close < vwap"
    assert fields["planned_exit_rule"] == "target_hit"
    assert fields["executed_exit_rule"] == "stop_hit or target_hit"
    assert fields["hold_rule"] == "not stop_hit"
    assert fields["entry_timeframe"] == "5m"
    assert fields["exit_timeframe"] == "5m"
    assert fields["category"] == "mean_reversion"
