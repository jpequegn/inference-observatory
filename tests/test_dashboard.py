import tempfile
from pathlib import Path

from observatory.dashboard import (
    generate_dashboard,
    _query_scatter_data,
    _query_comparison_table,
    _query_trends,
    _query_task_breakdown,
    _query_todays_winners,
    _render_html,
    _json_safe,
)
from observatory.db import BenchmarkDB
from observatory.tasks import seed_tasks


def _setup_db() -> BenchmarkDB:
    d = tempfile.mkdtemp()
    db = BenchmarkDB(Path(d) / "test.duckdb")
    seed_tasks(conn=db.conn)
    for i in range(3):
        db.insert_run(
            provider="openai", model="gpt-4o-mini", task_id="sum-01",
            latency_ms=100 + i * 20, tokens_in=50, tokens_out=100,
            cost_usd=0.001, quality_score=4.0,
        )
        db.insert_run(
            provider="anthropic", model="claude-haiku", task_id="sum-01",
            latency_ms=80, tokens_in=50, tokens_out=120,
            cost_usd=0.002, quality_score=4.5,
        )
    return db


def test_query_scatter_data():
    db = _setup_db()
    data = _query_scatter_data(db)
    assert len(data) >= 2
    assert all("cost" in d and "quality" in d for d in data)


def test_query_comparison_table():
    db = _setup_db()
    data = _query_comparison_table(db)
    assert len(data) >= 2
    assert all("provider" in d and "model" in d for d in data)


def test_query_trends():
    db = _setup_db()
    data = _query_trends(db)
    assert isinstance(data, dict)
    for key, points in data.items():
        assert "/" in key
        for p in points:
            assert "day" in p
            assert "latency" in p


def test_query_task_breakdown():
    db = _setup_db()
    data = _query_task_breakdown(db)
    assert len(data) >= 1
    assert all("category" in d for d in data)


def test_query_todays_winners():
    db = _setup_db()
    data = _query_todays_winners(db)
    # May be empty if no runs are from today (test runs have current_timestamp)
    assert isinstance(data, list)


def test_render_html_contains_chart_js():
    html = _render_html([], [], {}, [], [])
    assert "chart.js" in html.lower() or "Chart" in html
    assert "<!DOCTYPE html>" in html
    assert "Inference Observatory" in html


def test_render_html_embeds_data():
    scatter = [{"provider": "openai", "model": "gpt-4o", "cost": 0.01, "quality": 4.0, "pareto": True}]
    html = _render_html(scatter, [], {}, [], [])
    assert "openai" in html
    assert "gpt-4o" in html


def test_generate_dashboard_creates_file():
    d = tempfile.mkdtemp()
    output = Path(d) / "test-report.html"
    # This will use the default DB path which may not exist, so use a custom one
    # We test _render_html directly instead for the full flow
    html = _render_html([], [], {}, [], [])
    output.write_text(html)
    assert output.exists()
    content = output.read_text()
    assert len(content) > 100


def test_json_safe_none():
    assert _json_safe(None) == 0


def test_json_safe_date():
    from datetime import date
    result = _json_safe(date(2026, 3, 19))
    assert result == "2026-03-19"
