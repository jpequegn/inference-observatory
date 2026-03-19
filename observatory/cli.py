import uuid

import typer
from rich.console import Console
from rich.table import Table

from observatory.benchmark import (
    DEFAULT_MODELS,
    estimate_cost,
    format_summary_notification,
    run_benchmark,
    send_macos_notification,
)
from observatory.analysis import run_full_analysis, generate_findings
from observatory.dashboard import generate_dashboard
from observatory.db import BenchmarkDB, get_connection
from observatory.metrics import BenchmarkMetrics
from observatory.quality import QualityJudge
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


@app.command()
def score(
    provider: str = typer.Option(None, "--provider", "-p", help="Filter by provider"),
    model: str = typer.Option(None, "--model", "-m", help="Filter by model"),
    task_id: str = typer.Option(None, "--task", "-t", help="Score a single task"),
    check_consistency: bool = typer.Option(False, "--consistency", "-c", help="Run 3x and report variance"),
):
    """Score existing run outputs using LLM-as-judge (claude-sonnet-4-6)."""
    conn = get_connection()

    where_clauses = ["output_text IS NOT NULL", "output_text != ''"]
    params = []
    if provider:
        where_clauses.append("r.provider = ?")
        params.append(provider)
    if model:
        where_clauses.append("r.model = ?")
        params.append(model)
    if task_id:
        where_clauses.append("r.task_id = ?")
        params.append(task_id)

    where = f"WHERE {' AND '.join(where_clauses)}"

    rows = conn.execute(
        f"""
        SELECT r.id, r.provider, r.model, r.task_id, r.output_text, t.category, t.prompt
        FROM runs r JOIN tasks t ON r.task_id = t.id
        {where}
        ORDER BY r.provider, r.model, r.task_id
        """,
        params,
    ).fetchall()

    if not rows:
        console.print("[yellow]No runs to score. Run benchmarks first.[/yellow]")
        return

    judge = QualityJudge()
    console.print(f"Scoring {len(rows)} runs with [magenta]{judge.model}[/magenta]...\n")

    for run_id, prov, mod, tid, output, category, prompt in rows:
        console.print(f"  [cyan]{tid}[/cyan] ({prov}/{mod})...", end=" ")

        if check_consistency:
            result, std_dev = judge.score_with_consistency(category, prompt, output)
            if result.error:
                console.print(f"[red]ERROR: {result.error}[/red]")
                continue
            consistency = "[green]STABLE[/green]" if std_dev < 0.3 else "[yellow]VARIABLE[/yellow]"
            console.print(
                f"[green]{result.overall}/5[/green] std={std_dev} {consistency} "
                f"({', '.join(f'{k}:{v}' for k, v in result.criteria_scores.items())})"
            )
        else:
            result = judge.score(category, prompt, output)
            if result.error:
                console.print(f"[red]ERROR: {result.error}[/red]")
                continue
            console.print(
                f"[green]{result.overall}/5[/green] "
                f"({', '.join(f'{k}:{v}' for k, v in result.criteria_scores.items())})"
            )

        conn.execute(
            "UPDATE runs SET quality_score = ? WHERE id = ?",
            [result.overall, run_id],
        )

    console.print("\n[green]Quality scores saved to database.[/green]")


@app.command()
def summary(
    provider: str = typer.Argument(help="Provider to summarize"),
    days: int = typer.Option(30, "--days", "-d", help="Look back N days"),
):
    """Show provider summary: avg latency, cost, quality per model."""
    db = BenchmarkDB()
    rows = db.get_provider_summary(provider, days)
    if not rows:
        console.print(f"[yellow]No data for provider '{provider}' in the last {days} days.[/yellow]")
        return

    table = Table(title=f"Provider Summary: {provider} (last {days} days)")
    table.add_column("Model", style="cyan")
    table.add_column("Runs", justify="right")
    table.add_column("Avg (ms)", justify="right")
    table.add_column("p50 (ms)", justify="right", style="green")
    table.add_column("p95 (ms)", justify="right", style="yellow")
    table.add_column("Cost ($)", justify="right")
    table.add_column("Avg Cost", justify="right")
    table.add_column("Quality", justify="right", style="blue")
    table.add_column("tok/s", justify="right")

    for r in rows:
        table.add_row(
            r["model"], str(r["total_runs"]),
            f"{r['avg_latency_ms']:.0f}", f"{r['p50_latency_ms']:.0f}", f"{r['p95_latency_ms']:.0f}",
            f"{r['total_cost']:.6f}", f"{r['avg_cost']:.6f}",
            f"{r['avg_quality']:.1f}" if r["avg_quality"] else "—",
            f"{r['avg_tok_per_sec']:.1f}" if r["avg_tok_per_sec"] else "—",
        )
    console.print(table)


@app.command()
def breakdown(
    task_id: str = typer.Argument(help="Task ID to break down"),
):
    """Compare all providers/models on a single task."""
    db = BenchmarkDB()
    rows = db.get_task_breakdown(task_id)
    if not rows:
        console.print(f"[yellow]No data for task '{task_id}'.[/yellow]")
        return

    table = Table(title=f"Task Breakdown: {task_id}")
    table.add_column("Provider", style="magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Runs", justify="right")
    table.add_column("Avg (ms)", justify="right")
    table.add_column("p50 (ms)", justify="right", style="green")
    table.add_column("Avg Cost", justify="right")
    table.add_column("Quality", justify="right", style="blue")
    table.add_column("tok/s", justify="right")

    for r in rows:
        table.add_row(
            r["provider"], r["model"], str(r["runs"]),
            f"{r['avg_latency_ms']:.0f}", f"{r['p50_latency_ms']:.0f}",
            f"{r['avg_cost']:.6f}",
            f"{r['avg_quality']:.1f}" if r["avg_quality"] else "—",
            f"{r['avg_tok_per_sec']:.1f}" if r["avg_tok_per_sec"] else "—",
        )
    console.print(table)


@app.command()
def trend(
    provider: str = typer.Argument(help="Provider name"),
    model: str = typer.Argument(help="Model name"),
    metric: str = typer.Argument(help="Metric: latency_ms, cost_usd, quality_score, tokens_out"),
    days: int = typer.Option(30, "--days", "-d", help="Look back N days"),
):
    """Show daily trend of a metric over time."""
    db = BenchmarkDB()
    try:
        rows = db.get_trend(provider, model, metric, days)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    if not rows:
        console.print(f"[yellow]No trend data for {provider}/{model}.[/yellow]")
        return

    table = Table(title=f"Trend: {provider}/{model} — {metric} (last {days} days)")
    table.add_column("Day", style="cyan")
    table.add_column("Avg", justify="right", style="green")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Runs", justify="right")

    for r in rows:
        day_str = str(r["day"])[:10]
        table.add_row(
            day_str,
            f"{r['avg_value']:.2f}",
            f"{r['min_value']:.2f}",
            f"{r['max_value']:.2f}",
            str(r["runs"]),
        )
    console.print(table)


@app.command()
def pareto():
    """Show cost vs. quality Pareto frontier."""
    db = BenchmarkDB()
    rows = db.get_pareto_front()
    if not rows:
        console.print("[yellow]No scored runs found. Run benchmarks and score them first.[/yellow]")
        return

    table = Table(title="Pareto Frontier: Cost vs. Quality")
    table.add_column("Provider", style="magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Avg Cost ($)", justify="right")
    table.add_column("Avg Quality", justify="right", style="green")

    for r in rows:
        table.add_row(
            r["provider"], r["model"],
            f"{r['avg_cost']:.6f}",
            f"{r['avg_quality']:.1f}",
        )
    console.print(table)


@app.command()
def benchmark(
    providers: list[str] = typer.Option(None, "--provider", "-p", help="Providers to benchmark (repeatable)"),
    repeats: int = typer.Option(5, "--repeats", "-n", help="Runs per task per model"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip cost confirmation"),
    notify: bool = typer.Option(True, "--notify/--no-notify", help="Send macOS notification on completion"),
):
    """Run full benchmark across all providers and models."""
    if not providers:
        providers = list(DEFAULT_MODELS.keys())

    # Estimate cost and confirm
    costs = estimate_cost(providers, num_tasks=20, repeats=repeats)
    total_cost = sum(costs.values())

    console.print("[bold]Estimated cost:[/bold]")
    for p, c in costs.items():
        console.print(f"  {p}: ${c:.4f}")
    console.print(f"  [bold]Total: ${total_cost:.4f}[/bold]\n")

    if total_cost > 2.0 and not yes:
        confirm = typer.confirm(f"Estimated cost ${total_cost:.4f} exceeds $2. Continue?")
        if not confirm:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(0)

    console.print(f"Starting benchmark: {len(providers)} providers, {repeats} repeats per task...\n")
    result = run_benchmark(providers=providers, repeats=repeats)

    console.print(f"\n[green]Benchmark complete![/green]")
    console.print(f"  Batch: {result['batch_id']}")
    console.print(f"  Total runs: {result['total_runs']}")
    console.print(f"  Total errors: {result['total_errors']}")
    console.print(f"  Total cost: ${result['total_cost']:.4f}")

    # Auto-generate dashboard report
    report_path = generate_dashboard()
    console.print(f"  Report: [link=file://{report_path.resolve()}]{report_path}[/link]")

    if notify:
        msg = format_summary_notification(result)
        send_macos_notification("Inference Observatory", msg)


@app.command()
def dashboard(
    output: str = typer.Option(None, "--output", "-o", help="Output HTML file path"),
):
    """Generate static HTML dashboard from benchmark results."""
    output_path = Path(output) if output else None
    path = generate_dashboard(output_path)
    console.print(f"[green]Dashboard generated: {path}[/green]")


@app.command()
def install_schedule():
    """Install launchd plist for daily automated benchmarks at 6am."""
    import os
    plist_dir = Path.home() / "Library" / "LaunchAgents"
    plist_dir.mkdir(parents=True, exist_ok=True)
    plist_path = plist_dir / "com.inference-observatory.plist"

    log_dir = Path.home() / ".inference-observatory" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    working_dir = Path.cwd()
    python_path = subprocess.run(
        ["which", "python3"], capture_output=True, text=True
    ).stdout.strip()

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.inference-observatory</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>-m</string>
        <string>observatory.cli</string>
        <string>benchmark</string>
        <string>--yes</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{working_dir}</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>6</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>{log_dir}/benchmark.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/benchmark-error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin')}</string>
        <key>OPENAI_API_KEY</key>
        <string>{os.environ.get('OPENAI_API_KEY', '')}</string>
        <key>ANTHROPIC_API_KEY</key>
        <string>{os.environ.get('ANTHROPIC_API_KEY', '')}</string>
    </dict>
</dict>
</plist>"""

    plist_path.write_text(plist_content)
    console.print(f"[green]Plist written to {plist_path}[/green]")

    # Load the agent
    subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
    result = subprocess.run(["launchctl", "load", str(plist_path)], capture_output=True, text=True)
    if result.returncode == 0:
        console.print("[green]Schedule installed! Daily benchmark at 6:00 AM.[/green]")
    else:
        console.print(f"[yellow]launchctl load returned: {result.stderr}[/yellow]")
        console.print(f"You can manually load with: launchctl load {plist_path}")

    console.print(f"Logs: {log_dir}/")


@app.command()
def uninstall_schedule():
    """Uninstall the launchd daily benchmark schedule."""
    plist_path = Path.home() / "Library" / "LaunchAgents" / "com.inference-observatory.plist"
    if not plist_path.exists():
        console.print("[yellow]No schedule found.[/yellow]")
        return

    subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
    plist_path.unlink()
    console.print("[green]Schedule uninstalled.[/green]")


@app.command()
def analyze(
    days: int = typer.Option(30, "--days", "-d", help="Analysis window in days"),
    output: str = typer.Option("FINDINGS.md", "--output", "-o", help="Output file path"),
):
    """Analyze benchmark data and generate FINDINGS.md recommendations."""
    db = BenchmarkDB()
    result = run_full_analysis(db, days)

    console.print(f"[bold]Analysis ({days}-day window):[/bold]\n")

    if result.pareto_winners:
        console.print("[bold cyan]Category Winners:[/bold cyan]")
        for w in result.pareto_winners:
            console.print(f"  {w.category}: [green]{w.provider}/{w.model}[/green] Q={w.avg_quality}/5")
    else:
        console.print("[yellow]No scored data available yet.[/yellow]")

    if result.cost_efficiency:
        console.print("\n[bold cyan]Cost Efficiency:[/bold cyan]")
        for w in result.cost_efficiency:
            console.print(f"  {w.category}: [green]{w.provider}/{w.model}[/green] {w.quality_per_dollar:.0f} Q/$")

    if result.consistency:
        console.print("\n[bold cyan]Consistency:[/bold cyan]")
        for c in result.consistency:
            status = "[green]stable[/green]" if c.is_consistent else "[yellow]variable[/yellow]"
            console.print(f"  {c.provider}/{c.model}: std={c.quality_std_dev:.3f} {status}")

    if result.local_competitiveness:
        console.print("\n[bold cyan]Local vs API:[/bold cyan]")
        for lc in result.local_competitiveness:
            verdict = "[green]competitive[/green]" if lc.is_competitive else "[red]not competitive[/red]"
            console.print(f"  {lc.category}: gap={lc.quality_gap:.1f} {verdict}")

    findings = generate_findings(result)
    Path(output).write_text(findings)
    console.print(f"\n[green]Findings written to {output}[/green]")


if __name__ == "__main__":
    app()
