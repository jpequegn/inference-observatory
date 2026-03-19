import uuid

import typer
from rich.console import Console
from rich.table import Table

from observatory.db import get_connection
from observatory.metrics import BenchmarkMetrics
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
    repeats: int = typer.Option(5, "--repeats", "-n", help="Number of times to run each task"),
):
    """Run benchmark tasks against a provider with repeated measurements."""
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
    batch_id = str(uuid.uuid4())
    is_local = runner.is_local

    console.print(f"Running {len(all_tasks)} tasks x {repeats} repeats on [magenta]{provider}/{model}[/magenta]")
    console.print(f"Batch: [dim]{batch_id}[/dim]\n")

    for task in all_tasks:
        bm = BenchmarkMetrics(task_id=task["id"], provider=provider, model=model)

        for i in range(repeats):
            console.print(
                f"  [{i+1}/{repeats}] [cyan]{task['id']}[/cyan]...",
                end=" ",
            )
            result = runner.run(task, collect_system_metrics=is_local)

            if result.error:
                bm.errors += 1
                console.print(f"[red]ERROR: {result.error}[/red]")
                continue

            bm.latencies_ms.append(result.latency_ms)
            bm.tokens_in.append(result.tokens_in)
            bm.tokens_out.append(result.tokens_out)
            bm.costs_usd.append(result.cost_usd)
            bm.system_snapshots.extend(result.system_snapshots)

            avg_cpu = 0.0
            avg_mem = 0.0
            if result.system_snapshots:
                avg_cpu = sum(s.cpu_percent for s in result.system_snapshots) / len(result.system_snapshots)
                avg_mem = sum(s.memory_percent for s in result.system_snapshots) / len(result.system_snapshots)

            console.print(
                f"[green]{result.latency_ms:.0f}ms[/green] "
                f"{result.tokens_out}tok "
                f"${result.cost_usd:.6f}"
                + (f" cpu:{avg_cpu:.0f}% mem:{avg_mem:.0f}%" if is_local else "")
            )

            run_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO runs (id, provider, model, task_id, latency_ms, tokens_in, tokens_out,
                                  cost_usd, output_text, avg_cpu_percent, avg_memory_percent,
                                  run_index, batch_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [run_id, provider, model, task["id"], result.latency_ms,
                 result.tokens_in, result.tokens_out, result.cost_usd, result.output,
                 avg_cpu, avg_mem, i, batch_id],
            )

        if bm.latencies_ms:
            console.print(
                f"  [bold]Summary:[/bold] p50={bm.p50_latency:.0f}ms p95={bm.p95_latency:.0f}ms "
                f"avg={bm.avg_latency:.0f}ms {bm.avg_tokens_per_sec:.1f}tok/s "
                f"cost=${bm.total_cost:.6f} errors={bm.errors}\n"
            )


@app.command()
def metrics(
    provider: str = typer.Option(None, "--provider", "-p", help="Filter by provider"),
    model: str = typer.Option(None, "--model", "-m", help="Filter by model"),
):
    """Show aggregated benchmark metrics (p50, p95 latency, throughput, cost)."""
    conn = get_connection()

    where_clauses = []
    params = []
    if provider:
        where_clauses.append("provider = ?")
        params.append(provider)
    if model:
        where_clauses.append("model = ?")
        params.append(model)

    where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    rows = conn.execute(
        f"""
        SELECT
            provider,
            model,
            task_id,
            COUNT(*) as runs,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms) as p50,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95,
            AVG(latency_ms) as avg_ms,
            SUM(tokens_out) * 1000.0 / NULLIF(SUM(latency_ms), 0) as tok_per_sec,
            SUM(cost_usd) as total_cost,
            AVG(avg_cpu_percent) as avg_cpu,
            AVG(avg_memory_percent) as avg_mem
        FROM runs
        {where}
        GROUP BY provider, model, task_id
        ORDER BY provider, model, task_id
        """,
        params,
    ).fetchall()

    if not rows:
        console.print("[yellow]No benchmark data found. Run some benchmarks first.[/yellow]")
        return

    table = Table(title="Benchmark Metrics")
    table.add_column("Provider", style="magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Task", style="white")
    table.add_column("Runs", justify="right")
    table.add_column("p50 (ms)", justify="right", style="green")
    table.add_column("p95 (ms)", justify="right", style="yellow")
    table.add_column("Avg (ms)", justify="right")
    table.add_column("tok/s", justify="right", style="blue")
    table.add_column("Cost ($)", justify="right")
    table.add_column("CPU%", justify="right", style="dim")
    table.add_column("Mem%", justify="right", style="dim")

    for row in rows:
        table.add_row(
            row[0], row[1], row[2], str(row[3]),
            f"{row[4]:.0f}", f"{row[5]:.0f}", f"{row[6]:.0f}",
            f"{row[7]:.1f}" if row[7] else "—",
            f"{row[8]:.6f}",
            f"{row[9]:.0f}" if row[9] else "—",
            f"{row[10]:.0f}" if row[10] else "—",
        )
    console.print(table)


if __name__ == "__main__":
    app()
