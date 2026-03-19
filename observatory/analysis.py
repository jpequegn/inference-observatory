"""Automated analysis engine for generating provider recommendations."""

import statistics
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from observatory.db import BenchmarkDB


@dataclass
class CategoryWinner:
    category: str
    provider: str
    model: str
    avg_quality: float
    avg_cost: float
    avg_latency_ms: float
    quality_per_dollar: float
    reason: str = ""


@dataclass
class ConsistencyReport:
    provider: str
    model: str
    quality_std_dev: float
    latency_std_dev: float
    quality_scores: list[float] = field(default_factory=list)
    is_consistent: bool = True  # std_dev < 0.5


@dataclass
class LocalCompetitiveness:
    category: str
    local_provider: str
    local_model: str
    local_quality: float
    best_api_provider: str
    best_api_model: str
    best_api_quality: float
    quality_gap: float
    is_competitive: bool  # gap < 0.5


@dataclass
class AnalysisResult:
    """Full analysis output."""
    pareto_winners: list[CategoryWinner] = field(default_factory=list)
    cost_efficiency: list[CategoryWinner] = field(default_factory=list)
    consistency: list[ConsistencyReport] = field(default_factory=list)
    local_competitiveness: list[LocalCompetitiveness] = field(default_factory=list)
    generated_on: str = ""


LOCAL_PROVIDERS = {"ollama", "mlx"}


def analyze_pareto_per_category(db: BenchmarkDB, days: int = 30) -> list[CategoryWinner]:
    """Find Pareto-optimal provider per task category."""
    rows = db.conn.execute(f"""
        SELECT t.category, r.provider, r.model,
            AVG(r.quality_score) as avg_quality,
            AVG(r.cost_usd) as avg_cost,
            AVG(r.latency_ms) as avg_latency
        FROM runs r JOIN tasks t ON r.task_id = t.id
        WHERE r.quality_score IS NOT NULL
          AND r.timestamp >= current_timestamp - INTERVAL '{days}' DAY
        GROUP BY t.category, r.provider, r.model
        ORDER BY t.category, avg_quality DESC
    """).fetchall()

    winners = {}
    for cat, prov, model, quality, cost, latency in rows:
        qpd = quality / cost if cost > 0 else float("inf")
        if cat not in winners or quality > winners[cat].avg_quality:
            winners[cat] = CategoryWinner(
                category=cat, provider=prov, model=model,
                avg_quality=round(quality, 2), avg_cost=round(cost, 6),
                avg_latency_ms=round(latency, 0), quality_per_dollar=round(qpd, 1),
                reason=f"Highest quality ({quality:.1f}/5) in {cat}",
            )
    return list(winners.values())


def analyze_cost_efficiency(db: BenchmarkDB, days: int = 30) -> list[CategoryWinner]:
    """Best quality-per-dollar across all providers per category."""
    rows = db.conn.execute(f"""
        SELECT t.category, r.provider, r.model,
            AVG(r.quality_score) as avg_quality,
            AVG(r.cost_usd) as avg_cost,
            AVG(r.latency_ms) as avg_latency
        FROM runs r JOIN tasks t ON r.task_id = t.id
        WHERE r.quality_score IS NOT NULL AND r.cost_usd > 0
          AND r.timestamp >= current_timestamp - INTERVAL '{days}' DAY
        GROUP BY t.category, r.provider, r.model
        ORDER BY t.category
    """).fetchall()

    best: dict[str, CategoryWinner] = {}
    for cat, prov, model, quality, cost, latency in rows:
        qpd = quality / cost if cost > 0 else 0
        if cat not in best or qpd > best[cat].quality_per_dollar:
            best[cat] = CategoryWinner(
                category=cat, provider=prov, model=model,
                avg_quality=round(quality, 2), avg_cost=round(cost, 6),
                avg_latency_ms=round(latency, 0), quality_per_dollar=round(qpd, 1),
                reason=f"Best quality/$ ratio ({qpd:.0f} Q/$) in {cat}",
            )
    return list(best.values())


def analyze_consistency(db: BenchmarkDB, days: int = 30) -> list[ConsistencyReport]:
    """Day-to-day quality consistency per provider/model."""
    rows = db.conn.execute(f"""
        SELECT provider, model, DATE_TRUNC('day', timestamp) as day,
            AVG(quality_score) as daily_quality, AVG(latency_ms) as daily_latency
        FROM runs
        WHERE quality_score IS NOT NULL
          AND timestamp >= current_timestamp - INTERVAL '{days}' DAY
        GROUP BY provider, model, day
        ORDER BY provider, model, day
    """).fetchall()

    grouped: dict[str, dict] = {}
    for prov, model, day, quality, latency in rows:
        key = f"{prov}/{model}"
        if key not in grouped:
            grouped[key] = {"provider": prov, "model": model, "qualities": [], "latencies": []}
        grouped[key]["qualities"].append(quality)
        grouped[key]["latencies"].append(latency)

    reports = []
    for key, data in grouped.items():
        q_std = statistics.stdev(data["qualities"]) if len(data["qualities"]) > 1 else 0.0
        l_std = statistics.stdev(data["latencies"]) if len(data["latencies"]) > 1 else 0.0
        reports.append(ConsistencyReport(
            provider=data["provider"], model=data["model"],
            quality_std_dev=round(q_std, 3), latency_std_dev=round(l_std, 1),
            quality_scores=data["qualities"],
            is_consistent=q_std < 0.5,
        ))
    return reports


def analyze_local_competitiveness(db: BenchmarkDB, days: int = 30) -> list[LocalCompetitiveness]:
    """Where are local models competitive with API providers?"""
    rows = db.conn.execute(f"""
        SELECT t.category, r.provider, r.model, AVG(r.quality_score) as avg_quality
        FROM runs r JOIN tasks t ON r.task_id = t.id
        WHERE r.quality_score IS NOT NULL
          AND r.timestamp >= current_timestamp - INTERVAL '{days}' DAY
        GROUP BY t.category, r.provider, r.model
    """).fetchall()

    by_cat: dict[str, dict] = {}
    for cat, prov, model, quality in rows:
        by_cat.setdefault(cat, {"local": [], "api": []})
        entry = {"provider": prov, "model": model, "quality": quality}
        if prov in LOCAL_PROVIDERS:
            by_cat[cat]["local"].append(entry)
        else:
            by_cat[cat]["api"].append(entry)

    results = []
    for cat, data in by_cat.items():
        if not data["local"] or not data["api"]:
            continue
        best_local = max(data["local"], key=lambda x: x["quality"])
        best_api = max(data["api"], key=lambda x: x["quality"])
        gap = best_api["quality"] - best_local["quality"]
        results.append(LocalCompetitiveness(
            category=cat,
            local_provider=best_local["provider"], local_model=best_local["model"],
            local_quality=round(best_local["quality"], 2),
            best_api_provider=best_api["provider"], best_api_model=best_api["model"],
            best_api_quality=round(best_api["quality"], 2),
            quality_gap=round(gap, 2),
            is_competitive=gap < 0.5,
        ))
    return results


def run_full_analysis(db: BenchmarkDB | None = None, days: int = 30) -> AnalysisResult:
    """Run all analyses and return structured results."""
    if db is None:
        db = BenchmarkDB()
    return AnalysisResult(
        pareto_winners=analyze_pareto_per_category(db, days),
        cost_efficiency=analyze_cost_efficiency(db, days),
        consistency=analyze_consistency(db, days),
        local_competitiveness=analyze_local_competitiveness(db, days),
        generated_on=date.today().isoformat(),
    )


def generate_findings(result: AnalysisResult) -> str:
    """Generate FINDINGS.md from analysis results."""
    lines = [
        "# Inference Observatory — Findings",
        f"\n_Generated: {result.generated_on}_\n",
        "## Provider Recommendations by Category\n",
    ]

    if result.pareto_winners:
        for w in result.pareto_winners:
            lines.append(
                f"**For {w.category}**, use **{w.provider}/{w.model}** — "
                f"{w.reason}. Avg latency: {w.avg_latency_ms:.0f}ms, "
                f"cost: ${w.avg_cost:.6f}/run.\n"
            )
    else:
        lines.append("_No scored data available yet. Run benchmarks and score outputs first._\n")

    lines.append("\n## Cost Efficiency (Quality per Dollar)\n")
    if result.cost_efficiency:
        for w in result.cost_efficiency:
            lines.append(
                f"- **{w.category}**: {w.provider}/{w.model} — "
                f"{w.quality_per_dollar:.0f} quality/$, "
                f"Q={w.avg_quality:.1f}/5\n"
            )
    else:
        lines.append("_No API cost data available yet._\n")

    lines.append("\n## Consistency (Day-to-Day Stability)\n")
    if result.consistency:
        lines.append("| Provider | Model | Quality Std Dev | Consistent? |")
        lines.append("|----------|-------|-----------------|-------------|")
        for c in sorted(result.consistency, key=lambda x: x.quality_std_dev):
            status = "Yes" if c.is_consistent else "No"
            lines.append(f"| {c.provider} | {c.model} | {c.quality_std_dev:.3f} | {status} |")
        lines.append("")
    else:
        lines.append("_Not enough multi-day data for consistency analysis._\n")

    lines.append("\n## Local Model Competitiveness\n")
    if result.local_competitiveness:
        for lc in result.local_competitiveness:
            verdict = "competitive" if lc.is_competitive else "not competitive"
            lines.append(
                f"- **{lc.category}**: Local ({lc.local_provider}/{lc.local_model}, "
                f"Q={lc.local_quality:.1f}) vs API ({lc.best_api_provider}/{lc.best_api_model}, "
                f"Q={lc.best_api_quality:.1f}) — gap={lc.quality_gap:.1f}, **{verdict}**\n"
            )
        lines.append("\n> **When to use local**: ")
        competitive_cats = [lc.category for lc in result.local_competitiveness if lc.is_competitive]
        if competitive_cats:
            lines.append(f"Local models are good enough for: {', '.join(competitive_cats)}.\n")
        else:
            lines.append("Local models are not yet competitive in any category.\n")
    else:
        lines.append("_No local vs API comparison data available._\n")

    lines.append("\n---\n")
    lines.append("_This analysis updates automatically. Re-run with `observatory analyze` to refresh._\n")

    return "\n".join(lines)
