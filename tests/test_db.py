import tempfile
from pathlib import Path

from observatory.db import BenchmarkDB, get_connection
from observatory.tasks import seed_tasks


def _setup_db() -> BenchmarkDB:
    """Create a temp DB with tasks seeded and some sample runs."""
    d = tempfile.mkdtemp()
    db_path = Path(d) / "test.duckdb"
    db = BenchmarkDB(db_path)
    seed_tasks(conn=db.conn)

    # Insert sample runs for two providers
    for i in range(5):
        db.insert_run(
            provider="openai", model="gpt-4o-mini", task_id="sum-01",
            latency_ms=100 + i * 20, tokens_in=50, tokens_out=100,
            cost_usd=0.001, quality_score=4.0 + i * 0.1,
        )
        db.insert_run(
            provider="anthropic", model="claude-haiku-4-5-20251001", task_id="sum-01",
            latency_ms=80 + i * 10, tokens_in=50, tokens_out=120,
            cost_usd=0.002, quality_score=4.5,
        )
    # A second task for openai
    for i in range(3):
        db.insert_run(
            provider="openai", model="gpt-4o-mini", task_id="ext-01",
            latency_ms=200 + i * 30, tokens_in=60, tokens_out=80,
            cost_usd=0.0015, quality_score=3.5,
        )
    return db


def test_insert_run_returns_id():
    d = tempfile.mkdtemp()
    db = BenchmarkDB(Path(d) / "test.duckdb")
    seed_tasks(conn=db.conn)
    run_id = db.insert_run(
        provider="openai", model="gpt-4o-mini", task_id="sum-01",
        latency_ms=150, tokens_in=50, tokens_out=100, cost_usd=0.001,
    )
    assert isinstance(run_id, str)
    assert len(run_id) == 36  # UUID format


def test_get_provider_summary():
    db = _setup_db()
    rows = db.get_provider_summary("openai")
    assert len(rows) == 1  # one model
    row = rows[0]
    assert row["model"] == "gpt-4o-mini"
    assert row["total_runs"] == 8  # 5 sum-01 + 3 ext-01
    assert row["avg_latency_ms"] > 0
    assert row["total_cost"] > 0


def test_get_provider_summary_empty():
    db = _setup_db()
    rows = db.get_provider_summary("nonexistent")
    assert rows == []


def test_get_task_breakdown():
    db = _setup_db()
    rows = db.get_task_breakdown("sum-01")
    assert len(rows) == 2  # openai + anthropic
    providers = {r["provider"] for r in rows}
    assert providers == {"openai", "anthropic"}


def test_get_task_breakdown_empty():
    db = _setup_db()
    rows = db.get_task_breakdown("nonexistent-task")
    assert rows == []


def test_get_trend():
    db = _setup_db()
    rows = db.get_trend("openai", "gpt-4o-mini", "latency_ms", days=30)
    assert len(rows) >= 1
    row = rows[0]
    assert "avg_value" in row
    assert "min_value" in row
    assert "max_value" in row
    assert row["runs"] > 0


def test_get_trend_invalid_metric():
    db = _setup_db()
    try:
        db.get_trend("openai", "gpt-4o-mini", "invalid_metric")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid metric" in str(e)


def test_get_pareto_front():
    db = _setup_db()
    rows = db.get_pareto_front()
    assert len(rows) >= 1
    # All Pareto-optimal models should have quality scores
    for r in rows:
        assert r["avg_quality"] is not None
        assert r["avg_cost"] is not None


def test_indexes_created():
    d = tempfile.mkdtemp()
    db = BenchmarkDB(Path(d) / "test.duckdb")
    indexes = db.conn.execute(
        "SELECT index_name FROM duckdb_indexes()"
    ).fetchall()
    index_names = {i[0] for i in indexes}
    assert "idx_runs_timestamp" in index_names
    assert "idx_runs_provider" in index_names
    assert "idx_runs_task_id" in index_names
