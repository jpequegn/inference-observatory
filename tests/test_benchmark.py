from unittest.mock import patch, MagicMock

from observatory.benchmark import (
    DEFAULT_MODELS,
    estimate_cost,
    format_summary_notification,
    run_benchmark,
)


def test_default_models_all_providers():
    assert "ollama" in DEFAULT_MODELS
    assert "mlx" in DEFAULT_MODELS
    assert "openai" in DEFAULT_MODELS
    assert "anthropic" in DEFAULT_MODELS


def test_estimate_cost_local_free():
    costs = estimate_cost(["ollama", "mlx"])
    assert costs["ollama"] == 0.0
    assert costs["mlx"] == 0.0


def test_estimate_cost_api_positive():
    costs = estimate_cost(["openai"])
    assert costs["openai"] > 0


def test_estimate_cost_anthropic_positive():
    costs = estimate_cost(["anthropic"])
    assert costs["anthropic"] > 0


def test_estimate_cost_scales_with_repeats():
    costs_1 = estimate_cost(["openai"], repeats=1)
    costs_5 = estimate_cost(["openai"], repeats=5)
    assert costs_5["openai"] == costs_1["openai"] * 5


def test_estimate_cost_scales_with_tasks():
    costs_10 = estimate_cost(["openai"], num_tasks=10)
    costs_20 = estimate_cost(["openai"], num_tasks=20)
    assert abs(costs_20["openai"] - costs_10["openai"] * 2) < 0.001


def test_format_summary_notification():
    summary = {
        "total_runs": 100,
        "total_errors": 2,
        "total_cost": 0.1234,
        "providers": {"a/m1": {}, "b/m2": {}},
    }
    msg = format_summary_notification(summary)
    assert "100" in msg
    assert "2" in msg
    assert "$0.1234" in msg
    assert "2 models" in msg


def test_format_summary_notification_zero():
    summary = {
        "total_runs": 0,
        "total_errors": 0,
        "total_cost": 0.0,
        "providers": {},
    }
    msg = format_summary_notification(summary)
    assert "0" in msg
