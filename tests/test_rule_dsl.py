from agents.strategies.rule_dsl import RuleEvaluator


def test_rule_dsl_supports_arithmetic_binops():
    evaluator = RuleEvaluator()
    context = {
        "timeframe": "1h",
        "atr_14": 3.0,
        "tf_4h_atr_14": 10.0,
    }
    assert evaluator.evaluate("timeframe=='1h' and atr_14 < 0.5 * tf_4h_atr_14", context)
