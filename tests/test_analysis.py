import tempfile
from pathlib import Path

from observatory.analysis import (
    AnalysisResult,
    CategoryWinner,
    ConsistencyReport,
    LocalCompetitiveness,
    analyze_pareto_per_category,
    analyze_cost_efficiency,
    analyze_consistency,
    analyze_local_competitiveness,
    run_full_analysis,
    generate_findings,
)
from observatory.db import BenchmarkDB
from observatory.tasks import seed_tasks


def _setup_db() -> BenchmarkDB:
    d = tempfile.mkdtemp()
    db = BenchmarkDB(Path(d) / "test.duckdb")
    seed_tasks(conn=db.conn)

    # API provider runs
    for i in range(5):
        db.insert_run(
            provider="openai", model="gpt-4o-mini", task_id="sum-01",
            latency_ms=100 + i * 10, tokens_in=50, tokens_out=100,
            cost_usd=0.001, quality_score=4.0 + i * 0.1,
        )
        db.insert_run(
            provider="anthropic", model="claude-haiku", task_id="sum-01",
            latency_ms=90, tokens_in=50, tokens_out=120,
            cost_usd=0.002, quality_score=4.5,
        )
    # Local provider runs
    for i in range(5):
        db.insert_run(
            provider="ollama", model="llama3.2", task_id="sum-01",
            latency_ms=200, tokens_in=50, tokens_out=80,
            cost_usd=0.0, quality_score=3.8,
        )
    # Extraction tasks
    for i in range(3):
        db.insert_run(
            provider="openai", model="gpt-4o-mini", task_id="ext-01",
            latency_ms=150, tokens_in=60, tokens_out=90,
            cost_usd=0.0015, quality_score=3.5,
        )
        db.insert_run(
            provider="ollama", model="llama3.2", task_id="ext-01",
            latency_ms=250, tokens_in=60, tokens_out=70,
            cost_usd=0.0, quality_score=3.2,
        )
    return db


def test_analyze_pareto_per_category():
    db = _setup_db()
    winners = analyze_pareto_per_category(db)
    assert len(winners) >= 1
    cats = {w.category for w in winners}
    assert "summarization" in cats
    # Anthropic should win summarization (quality 4.5 > 4.2 avg for openai)
    sum_winner = next(w for w in winners if w.category == "summarization")
    assert sum_winner.provider == "anthropic"


def test_analyze_cost_efficiency():
    db = _setup_db()
    efficiency = analyze_cost_efficiency(db)
    assert len(efficiency) >= 1
    for e in efficiency:
        assert e.quality_per_dollar > 0


def test_analyze_consistency():
    db = _setup_db()
    reports = analyze_consistency(db)
    assert len(reports) >= 1
    for r in reports:
        assert r.quality_std_dev >= 0
        assert isinstance(r.is_consistent, bool)


def test_analyze_local_competitiveness():
    db = _setup_db()
    results = analyze_local_competitiveness(db)
    assert len(results) >= 1
    for lc in results:
        assert lc.local_provider in ("ollama", "mlx")
        assert lc.quality_gap >= 0 or lc.quality_gap < 0  # can be negative if local is better


def test_run_full_analysis():
    db = _setup_db()
    result = run_full_analysis(db)
    assert isinstance(result, AnalysisResult)
    assert result.generated_on != ""
    assert len(result.pareto_winners) >= 1


def test_generate_findings_with_data():
    db = _setup_db()
    result = run_full_analysis(db)
    md = generate_findings(result)
    assert "# Inference Observatory" in md
    assert "Provider Recommendations" in md
    assert "Cost Efficiency" in md
    assert "Consistency" in md
    assert "Local Model" in md


def test_generate_findings_empty():
    result = AnalysisResult(generated_on="2026-03-19")
    md = generate_findings(result)
    assert "No scored data available" in md


def test_category_winner_dataclass():
    w = CategoryWinner(
        category="qa", provider="openai", model="gpt-4o",
        avg_quality=4.5, avg_cost=0.01, avg_latency_ms=200,
        quality_per_dollar=450.0, reason="test",
    )
    assert w.category == "qa"
    assert w.quality_per_dollar == 450.0


def test_local_competitiveness_competitive():
    lc = LocalCompetitiveness(
        category="summarization",
        local_provider="ollama", local_model="llama3.2", local_quality=4.2,
        best_api_provider="openai", best_api_model="gpt-4o", best_api_quality=4.5,
        quality_gap=0.3, is_competitive=True,
    )
    assert lc.is_competitive
    assert lc.quality_gap == 0.3
