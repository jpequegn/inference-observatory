import duckdb
from pathlib import Path

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
    output_text TEXT
);
"""


def get_connection(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(db_path))
    conn.execute(SCHEMA)
    return conn
