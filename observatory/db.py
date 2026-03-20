import uuid
from dataclasses import asdict
from pathlib import Path

import duckdb

DB_PATH = Path("observatory.duckdb")

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    category VARCHAR NOT NULL,
    prompt TEXT NOT NULL,
    expected_output TEXT,
    difficulty VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id VARCHAR PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT current_timestamp,
    provider VARCHAR NOT NULL,
    model VARCHAR NOT NULL,
    task_id VARCHAR NOT NULL REFERENCES tasks(id),
    latency_ms DOUBLE,
    tokens_in INTEGER,
    tokens_out INTEGER,
    cost_usd DOUBLE,
    quality_score DOUBLE,
    output_text TEXT,
    avg_cpu_percent DOUBLE,
    avg_memory_percent DOUBLE,
    run_index INTEGER DEFAULT 0,
    batch_id VARCHAR,
    grounding_score DOUBLE
);
"""

INDEXES = """
CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(timestamp);
CREATE INDEX IF NOT EXISTS idx_runs_provider ON runs(provider);
CREATE INDEX IF NOT EXISTS idx_runs_task_id ON runs(task_id);
"""


def get_connection(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(db_path))
    conn.execute(SCHEMA)
    conn.execute(INDEXES)
    return conn


class BenchmarkDB:
    """High-level query interface for benchmark results."""

    def __init__(self, db_path: Path = DB_PATH):
        self.conn = get_connection(db_path)

    def insert_run(
        self,
        provider: str,
        model: str,
        task_id: str,
        latency_ms: float,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        output_text: str = "",
        quality_score: float | None = None,
        avg_cpu_percent: float = 0.0,
        avg_memory_percent: float = 0.0,
        run_index: int = 0,
        batch_id: str | None = None,
    ) -> str:
        """Insert a single run result. Returns the generated run ID."""
        run_id = str(uuid.uuid4())
        self.conn.execute(
            """
            INSERT INTO runs (id, provider, model, task_id, latency_ms, tokens_in, tokens_out,
                              cost_usd, output_text, quality_score, avg_cpu_percent,
                              avg_memory_percent, run_index, batch_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [run_id, provider, model, task_id, latency_ms, tokens_in, tokens_out,
             cost_usd, output_text, quality_score, avg_cpu_percent, avg_memory_percent,
             run_index, batch_id],
        )
        return run_id

    def get_provider_summary(self, provider: str, days: int = 30) -> list[dict]:
        """Average latency, cost, and quality per model for a provider over N days."""
        rows = self.conn.execute(
            f"""
            SELECT
                model,
                COUNT(*) as total_runs,
                AVG(latency_ms) as avg_latency_ms,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms) as p50_latency_ms,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_latency_ms,
                SUM(cost_usd) as total_cost,
                AVG(cost_usd) as avg_cost,
                AVG(quality_score) as avg_quality,
                AVG(tokens_out) * 1000.0 / NULLIF(AVG(latency_ms), 0) as avg_tok_per_sec
            FROM runs
            WHERE provider = ?
              AND timestamp >= current_timestamp - INTERVAL '{days}' DAY
            GROUP BY model
            ORDER BY model
            """,
            [provider],
        ).fetchall()
        columns = ["model", "total_runs", "avg_latency_ms", "p50_latency_ms",
                    "p95_latency_ms", "total_cost", "avg_cost", "avg_quality", "avg_tok_per_sec"]
        return [dict(zip(columns, row)) for row in rows]

    def get_task_breakdown(self, task_id: str) -> list[dict]:
        """All providers/models side by side for a single task."""
        rows = self.conn.execute(
            """
            SELECT
                provider,
                model,
                COUNT(*) as runs,
                AVG(latency_ms) as avg_latency_ms,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms) as p50_latency_ms,
                AVG(cost_usd) as avg_cost,
                AVG(quality_score) as avg_quality,
                AVG(tokens_out) * 1000.0 / NULLIF(AVG(latency_ms), 0) as avg_tok_per_sec
            FROM runs
            WHERE task_id = ?
            GROUP BY provider, model
            ORDER BY avg_quality DESC NULLS LAST, avg_latency_ms ASC
            """,
            [task_id],
        ).fetchall()
        columns = ["provider", "model", "runs", "avg_latency_ms", "p50_latency_ms",
                    "avg_cost", "avg_quality", "avg_tok_per_sec"]
        return [dict(zip(columns, row)) for row in rows]

    def get_trend(self, provider: str, model: str, metric: str, days: int = 30) -> list[dict]:
        """Daily trend of a metric over time.

        Valid metrics: latency_ms, cost_usd, quality_score, tokens_out.
        """
        valid_metrics = {"latency_ms", "cost_usd", "quality_score", "tokens_out"}
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric: {metric}. Choose from: {valid_metrics}")

        rows = self.conn.execute(
            f"""
            SELECT
                DATE_TRUNC('day', timestamp) as day,
                AVG({metric}) as avg_value,
                MIN({metric}) as min_value,
                MAX({metric}) as max_value,
                COUNT(*) as runs
            FROM runs
            WHERE provider = ?
              AND model = ?
              AND timestamp >= current_timestamp - INTERVAL '{days}' DAY
            GROUP BY day
            ORDER BY day
            """,
            [provider, model],
        ).fetchall()
        columns = ["day", "avg_value", "min_value", "max_value", "runs"]
        return [dict(zip(columns, row)) for row in rows]

    def get_pareto_front(self) -> list[dict]:
        """Cost vs. quality Pareto frontier — models not dominated on both axes."""
        rows = self.conn.execute(
            """
            WITH model_stats AS (
                SELECT
                    provider,
                    model,
                    AVG(cost_usd) as avg_cost,
                    AVG(quality_score) as avg_quality
                FROM runs
                WHERE quality_score IS NOT NULL
                GROUP BY provider, model
            )
            SELECT
                a.provider,
                a.model,
                a.avg_cost,
                a.avg_quality
            FROM model_stats a
            WHERE NOT EXISTS (
                SELECT 1 FROM model_stats b
                WHERE b.avg_cost <= a.avg_cost
                  AND b.avg_quality >= a.avg_quality
                  AND (b.avg_cost < a.avg_cost OR b.avg_quality > a.avg_quality)
            )
            ORDER BY a.avg_cost ASC
            """,
        ).fetchall()
        columns = ["provider", "model", "avg_cost", "avg_quality"]
        return [dict(zip(columns, row)) for row in rows]
