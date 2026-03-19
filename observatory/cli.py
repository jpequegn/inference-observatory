import typer
from rich.console import Console
from rich.table import Table

from observatory.db import get_connection
from observatory.tasks import seed_tasks

app = typer.Typer(help="Inference Observatory - LLM benchmark suite")
console = Console()


@app.command()
def init():
    """Initialize the database and load benchmark tasks."""
    conn = get_connection()
    count = seed_tasks()
    console.print(f"[green]Database initialized. Loaded {count} benchmark tasks.[/green]")


@app.command()
def tasks():
    """List all benchmark tasks."""
    conn = get_connection()
    rows = conn.execute("SELECT id, name, category, difficulty FROM tasks ORDER BY category, id").fetchall()
    table = Table(title="Benchmark Tasks")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Category", style="magenta")
    table.add_column("Difficulty", style="yellow")
    for row in rows:
        table.add_row(*row)
    console.print(table)


if __name__ == "__main__":
    app()
