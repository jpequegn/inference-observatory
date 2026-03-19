import uuid

import typer
from rich.console import Console
from rich.table import Table

from observatory.db import get_connection
from observatory.runners import RUNNERS
from observatory.tasks import load_tasks, seed_tasks

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


@app.command()
def run(
    provider: str = typer.Argument(help="Provider name: ollama, mlx, openai, anthropic"),
    model: str = typer.Argument(help="Model name to use"),
    task_id: str = typer.Option(None, "--task", "-t", help="Run a single task by ID"),
):
    """Run benchmark tasks against a provider."""
    if provider not in RUNNERS:
        console.print(f"[red]Unknown provider: {provider}. Choose from: {', '.join(RUNNERS)}[/red]")
        raise typer.Exit(1)

    runner_cls = RUNNERS[provider]
    runner = runner_cls(model=model)
    all_tasks = load_tasks()

    if task_id:
        all_tasks = [t for t in all_tasks if t["id"] == task_id]
        if not all_tasks:
            console.print(f"[red]Task not found: {task_id}[/red]")
            raise typer.Exit(1)

    conn = get_connection()
    seed_tasks(conn=conn)

    for task in all_tasks:
        console.print(f"  Running [cyan]{task['id']}[/cyan] on [magenta]{provider}/{model}[/magenta]...", end=" ")
        result = runner.run(task)
        if result.error:
            console.print(f"[red]ERROR: {result.error}[/red]")
        else:
            console.print(
                f"[green]OK[/green] {result.latency_ms:.0f}ms, "
                f"{result.tokens_in}+{result.tokens_out} tokens, "
                f"${result.cost_usd:.6f}"
            )
            run_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO runs (id, provider, model, task_id, latency_ms, tokens_in, tokens_out, cost_usd, output_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [run_id, provider, model, task["id"], result.latency_ms,
                 result.tokens_in, result.tokens_out, result.cost_usd, result.output],
            )


if __name__ == "__main__":
    app()
