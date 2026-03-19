"""Automated benchmark runner across all configured providers."""

import subprocess
import uuid
from datetime import datetime
from pathlib import Path

from observatory.db import BenchmarkDB, get_connection
from observatory.runners import RUNNERS
from observatory.runners.openai_runner import PRICING as OPENAI_PRICING
from observatory.runners.anthropic_runner import PRICING as ANTHROPIC_PRICING, MODEL_ALIASES
from observatory.tasks import load_tasks, seed_tasks

# Default models per provider
DEFAULT_MODELS = {
    "ollama": ["llama3.2", "gemma3", "mistral"],
    "mlx": ["mlx-community/Llama-3.2-3B-Instruct-4bit"],
    "openai": ["gpt-4o-mini", "gpt-4o"],
    "anthropic": ["claude-haiku-4-5", "claude-sonnet-4-6"],
}

# Average tokens per task (estimated for cost projection)
AVG_TOKENS_PER_TASK = {"input": 500, "output": 300}


def estimate_cost(
    providers: list[str],
    num_tasks: int = 20,
    repeats: int = 5,
) -> dict[str, float]:
    """Estimate cost per provider for a full benchmark run.

    Returns dict of provider -> estimated cost in USD.
    """
    estimates = {}
    for provider in providers:
        if provider in ("ollama", "mlx"):
            estimates[provider] = 0.0
            continue

        models = DEFAULT_MODELS.get(provider, [])
        total = 0.0
        for model in models:
            if provider == "openai":
                prices = OPENAI_PRICING.get(model, {"input": 0, "output": 0})
            elif provider == "anthropic":
                resolved = MODEL_ALIASES.get(model, model)
                prices = ANTHROPIC_PRICING.get(resolved, {"input": 0, "output": 0})
            else:
                prices = {"input": 0, "output": 0}

            per_run = (
                AVG_TOKENS_PER_TASK["input"] * prices["input"]
                + AVG_TOKENS_PER_TASK["output"] * prices["output"]
            ) / 1_000_000
            total += per_run * num_tasks * repeats

        estimates[provider] = round(total, 4)

    return estimates


def run_benchmark(
    providers: list[str] | None = None,
    repeats: int = 5,
    log_path: Path | None = None,
) -> dict:
    """Run all benchmark tasks across specified providers.

    Returns a summary dict with results per provider/model.
    """
    if providers is None:
        providers = list(DEFAULT_MODELS.keys())

    tasks = load_tasks()
    db = BenchmarkDB()
    seed_tasks(conn=db.conn)
    batch_id = str(uuid.uuid4())

    results = {}
    total_runs = 0
    total_errors = 0

    for provider in providers:
        if provider not in RUNNERS:
            continue

        models = DEFAULT_MODELS.get(provider, [])
        runner_cls = RUNNERS[provider]

        for model in models:
            key = f"{provider}/{model}"
            runner = runner_cls(model=model)
            model_results = {"runs": 0, "errors": 0, "latencies": [], "costs": []}

            for task in tasks:
                for i in range(repeats):
                    result = runner.run(task, collect_system_metrics=runner.is_local)

                    if result.error:
                        model_results["errors"] += 1
                        total_errors += 1
                        continue

                    avg_cpu = 0.0
                    avg_mem = 0.0
                    if result.system_snapshots:
                        avg_cpu = sum(s.cpu_percent for s in result.system_snapshots) / len(result.system_snapshots)
                        avg_mem = sum(s.memory_percent for s in result.system_snapshots) / len(result.system_snapshots)

                    db.insert_run(
                        provider=provider,
                        model=model if provider != "anthropic" else MODEL_ALIASES.get(model, model),
                        task_id=task["id"],
                        latency_ms=result.latency_ms,
                        tokens_in=result.tokens_in,
                        tokens_out=result.tokens_out,
                        cost_usd=result.cost_usd,
                        output_text=result.output,
                        avg_cpu_percent=avg_cpu,
                        avg_memory_percent=avg_mem,
                        run_index=i,
                        batch_id=batch_id,
                    )

                    model_results["runs"] += 1
                    model_results["latencies"].append(result.latency_ms)
                    model_results["costs"].append(result.cost_usd)
                    total_runs += 1

            results[key] = model_results

    summary = {
        "batch_id": batch_id,
        "timestamp": datetime.now().isoformat(),
        "total_runs": total_runs,
        "total_errors": total_errors,
        "total_cost": sum(
            sum(r["costs"]) for r in results.values()
        ),
        "providers": results,
    }

    return summary


def send_macos_notification(title: str, message: str):
    """Send a macOS notification via osascript."""
    script = f'display notification "{message}" with title "{title}"'
    subprocess.run(["osascript", "-e", script], capture_output=True)


def format_summary_notification(summary: dict) -> str:
    """Format a 3-line summary for notification."""
    lines = [
        f"Runs: {summary['total_runs']} | Errors: {summary['total_errors']}",
        f"Cost: ${summary['total_cost']:.4f}",
        f"Providers: {len(summary['providers'])} models benchmarked",
    ]
    return "\n".join(lines)
