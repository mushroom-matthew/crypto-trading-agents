from agents.strategies.rule_dsl import RuleEvaluator


def test_rule_dsl_supports_arithmetic_binops():
    evaluator = RuleEvaluator()
    context = {
        "timeframe": "1h",
        "atr_14": 3.0,
        "tf_4h_atr_14": 10.0,
    }
    assert evaluator.evaluate("timeframe=='1h' and atr_14 < 0.5 * tf_4h_atr_14", context)


def test_rule_dsl_supports_in_operator():
    evaluator = RuleEvaluator()
    context = {"vol_state": "high", "position": "flat"}
    assert evaluator.evaluate("vol_state in ['high', 'extreme']", context)
    assert not evaluator.evaluate("vol_state in ['low', 'medium']", context)


def test_rule_dsl_supports_not_in_operator():
    evaluator = RuleEvaluator()
    context = {"position": "long"}
    assert evaluator.evaluate("position not in ['flat']", context)
    assert not evaluator.evaluate("position not in ['long', 'short']", context)


def test_rule_dsl_in_with_compound_expression():
    evaluator = RuleEvaluator()
    context = {"rsi_14": 55.0, "vol_state": "extreme"}
    assert evaluator.evaluate("rsi_14 > 50 and vol_state in ['high', 'extreme']", context)
    context2 = {"rsi_14": 45.0, "vol_state": "extreme"}
    assert not evaluator.evaluate("rsi_14 > 50 and vol_state in ['high', 'extreme']", context2)
