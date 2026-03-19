import json
import tempfile
from pathlib import Path

from observatory.db import get_connection
from observatory.tasks import load_tasks, seed_tasks


def test_load_tasks():
    tasks = load_tasks()
    assert len(tasks) == 20
    categories = {t["category"] for t in tasks}
    assert categories == {"summarization", "extraction", "trend_analysis", "qa"}
    for cat in categories:
        assert sum(1 for t in tasks if t["category"] == cat) == 5


def test_db_schema():
    with tempfile.TemporaryDirectory() as d:
        conn = get_connection(Path(d) / "test.duckdb")
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert "tasks" in table_names
        assert "runs" in table_names


def test_seed_tasks():
    with tempfile.TemporaryDirectory() as d:
        conn = get_connection(Path(d) / "test.duckdb")
        count = seed_tasks(conn=conn)
        assert count == 20
        rows = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()
        assert rows[0] == 20


def test_task_fields():
    tasks = load_tasks()
    required_fields = {"id", "name", "category", "prompt", "difficulty"}
    for task in tasks:
        assert required_fields.issubset(task.keys()), f"Task {task.get('id')} missing fields"


def test_task_difficulties():
    tasks = load_tasks()
    valid = {"easy", "medium", "hard"}
    for task in tasks:
        assert task["difficulty"] in valid, f"Task {task['id']} has invalid difficulty"
