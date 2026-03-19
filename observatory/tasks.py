import json
from pathlib import Path

from observatory.db import get_connection

TASKS_FILE = Path("benchmarks/tasks.jsonl")


def load_tasks(path: Path = TASKS_FILE) -> list[dict]:
    tasks = []
    with open(path) as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def seed_tasks(path: Path = TASKS_FILE, conn=None) -> int:
    tasks = load_tasks(path)
    if conn is None:
        conn = get_connection()
    count = 0
    for task in tasks:
        conn.execute(
            """
            INSERT OR REPLACE INTO tasks (id, name, category, prompt, expected_output, difficulty)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                task["id"],
                task["name"],
                task["category"],
                task["prompt"],
                task.get("expected_output", ""),
                task["difficulty"],
            ],
        )
        count += 1
    return count
