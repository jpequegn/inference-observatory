"""Generate static HTML dashboard with embedded Chart.js visualizations."""

import json
from datetime import date, datetime
from pathlib import Path

from observatory.db import BenchmarkDB

REPORTS_DIR = Path("reports")


def _query_scatter_data(db: BenchmarkDB) -> list[dict]:
    """Cost vs Quality scatter data with Pareto frontier."""
    rows = db.conn.execute("""
        SELECT provider, model, AVG(cost_usd) as avg_cost, AVG(quality_score) as avg_quality
        FROM runs WHERE quality_score IS NOT NULL
        GROUP BY provider, model
    """).fetchall()
    pareto = {(r["provider"], r["model"]) for r in db.get_pareto_front()}
    return [
        {"provider": r[0], "model": r[1], "cost": r[2], "quality": r[3],
         "pareto": (r[0], r[1]) in pareto}
        for r in rows
    ]


def _query_comparison_table(db: BenchmarkDB) -> list[dict]:
    """Provider comparison: latency, cost, quality side by side."""
    rows = db.conn.execute("""
        SELECT provider, model, COUNT(*) as runs,
            AVG(latency_ms) as avg_latency, AVG(cost_usd) as avg_cost,
            AVG(quality_score) as avg_quality,
            AVG(tokens_out) * 1000.0 / NULLIF(AVG(latency_ms), 0) as tok_per_sec
        FROM runs GROUP BY provider, model ORDER BY provider, model
    """).fetchall()
    cols = ["provider", "model", "runs", "avg_latency", "avg_cost", "avg_quality", "tok_per_sec"]
    return [dict(zip(cols, r)) for r in rows]


def _query_trends(db: BenchmarkDB, days: int = 90) -> dict[str, list[dict]]:
    """Quality and latency trends per provider/model over time."""
    rows = db.conn.execute(f"""
        SELECT provider, model, DATE_TRUNC('day', timestamp) as day,
            AVG(latency_ms) as avg_latency, AVG(quality_score) as avg_quality
        FROM runs
        WHERE timestamp >= current_timestamp - INTERVAL '{days}' DAY
        GROUP BY provider, model, day ORDER BY provider, model, day
    """).fetchall()
    trends: dict[str, list[dict]] = {}
    for r in rows:
        key = f"{r[0]}/{r[1]}"
        trends.setdefault(key, []).append({
            "day": str(r[2])[:10], "latency": r[3], "quality": r[4],
        })
    return trends


def _query_task_breakdown(db: BenchmarkDB) -> list[dict]:
    """Best provider per task category."""
    rows = db.conn.execute("""
        SELECT t.category, r.provider, r.model, AVG(r.quality_score) as avg_quality,
            AVG(r.latency_ms) as avg_latency, AVG(r.cost_usd) as avg_cost
        FROM runs r JOIN tasks t ON r.task_id = t.id
        WHERE r.quality_score IS NOT NULL
        GROUP BY t.category, r.provider, r.model
        ORDER BY t.category, avg_quality DESC
    """).fetchall()
    cols = ["category", "provider", "model", "avg_quality", "avg_latency", "avg_cost"]
    return [dict(zip(cols, r)) for r in rows]


def _query_todays_winners(db: BenchmarkDB) -> list[dict]:
    """Best quality/cost ratio per category for today."""
    rows = db.conn.execute("""
        SELECT t.category, r.provider, r.model,
            AVG(r.quality_score) as avg_quality, AVG(r.cost_usd) as avg_cost,
            AVG(r.quality_score) / NULLIF(AVG(r.cost_usd), 0) as quality_per_dollar
        FROM runs r JOIN tasks t ON r.task_id = t.id
        WHERE r.quality_score IS NOT NULL
          AND DATE_TRUNC('day', r.timestamp) = current_date
        GROUP BY t.category, r.provider, r.model
        ORDER BY t.category, quality_per_dollar DESC NULLS LAST
    """).fetchall()
    cols = ["category", "provider", "model", "avg_quality", "avg_cost", "quality_per_dollar"]
    return [dict(zip(cols, r)) for r in rows]


def _json_safe(obj):
    """Make objects JSON serializable."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if obj is None:
        return 0
    return obj


def generate_dashboard(output_path: Path | None = None) -> Path:
    """Generate a static HTML dashboard and return the file path."""
    db = BenchmarkDB()

    scatter = _query_scatter_data(db)
    comparison = _query_comparison_table(db)
    trends = _query_trends(db)
    task_breakdown = _query_task_breakdown(db)
    winners = _query_todays_winners(db)

    if output_path is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = REPORTS_DIR / f"{date.today().isoformat()}.html"

    html = _render_html(scatter, comparison, trends, task_breakdown, winners)
    output_path.write_text(html)
    return output_path


def _render_html(
    scatter: list[dict],
    comparison: list[dict],
    trends: dict[str, list[dict]],
    task_breakdown: list[dict],
    winners: list[dict],
) -> str:
    scatter_json = json.dumps(scatter, default=_json_safe)
    comparison_json = json.dumps(comparison, default=_json_safe)
    trends_json = json.dumps(trends, default=_json_safe)
    task_breakdown_json = json.dumps(task_breakdown, default=_json_safe)
    winners_json = json.dumps(winners, default=_json_safe)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Inference Observatory — {date.today().isoformat()}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, system-ui, sans-serif; background: #0d1117; color: #e6edf3; padding: 2rem; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; }}
  h2 {{ font-size: 1.3rem; margin: 2rem 0 1rem; color: #58a6ff; }}
  .subtitle {{ color: #8b949e; margin-bottom: 2rem; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-bottom: 2rem; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1.5rem; }}
  .card.full {{ grid-column: 1 / -1; }}
  canvas {{ max-height: 400px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th, td {{ padding: 0.6rem 1rem; text-align: left; border-bottom: 1px solid #21262d; }}
  th {{ color: #8b949e; font-weight: 600; }}
  .winner {{ color: #3fb950; font-weight: 600; }}
  .pareto {{ color: #f0883e; }}
  @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>Inference Observatory</h1>
<p class="subtitle">Benchmark Report — {date.today().isoformat()}</p>

<div class="grid">
  <div class="card">
    <h2>Cost vs. Quality</h2>
    <canvas id="scatterChart"></canvas>
  </div>
  <div class="card">
    <h2>Today's Winners</h2>
    <table>
      <thead><tr><th>Category</th><th>Provider</th><th>Model</th><th>Quality</th><th>Cost</th></tr></thead>
      <tbody id="winnersBody"></tbody>
    </table>
  </div>
  <div class="card full">
    <h2>Provider Comparison</h2>
    <table>
      <thead><tr><th>Provider</th><th>Model</th><th>Runs</th><th>Avg Latency</th><th>Avg Cost</th><th>Quality</th><th>tok/s</th></tr></thead>
      <tbody id="comparisonBody"></tbody>
    </table>
  </div>
  <div class="card">
    <h2>Latency Trends (90d)</h2>
    <canvas id="latencyTrendChart"></canvas>
  </div>
  <div class="card">
    <h2>Quality Trends (90d)</h2>
    <canvas id="qualityTrendChart"></canvas>
  </div>
  <div class="card full">
    <h2>Best Provider per Category</h2>
    <table>
      <thead><tr><th>Category</th><th>Provider</th><th>Model</th><th>Quality</th><th>Latency</th><th>Cost</th></tr></thead>
      <tbody id="taskBreakdownBody"></tbody>
    </table>
  </div>
</div>

<script>
const scatter = {scatter_json};
const comparison = {comparison_json};
const trends = {trends_json};
const taskBreakdown = {task_breakdown_json};
const winners = {winners_json};

const colors = ['#58a6ff','#3fb950','#f0883e','#bc8cff','#f778ba','#79c0ff','#56d364','#e3b341'];

// Scatter chart
new Chart(document.getElementById('scatterChart'), {{
  type: 'scatter',
  data: {{
    datasets: [{{
      label: 'Models',
      data: scatter.filter(d => !d.pareto).map(d => ({{ x: d.cost, y: d.quality, label: d.provider+'/'+d.model }})),
      backgroundColor: '#58a6ff',
      pointRadius: 6,
    }}, {{
      label: 'Pareto Frontier',
      data: scatter.filter(d => d.pareto).map(d => ({{ x: d.cost, y: d.quality, label: d.provider+'/'+d.model }})),
      backgroundColor: '#f0883e',
      pointRadius: 8,
      pointStyle: 'star',
    }}]
  }},
  options: {{
    scales: {{
      x: {{ title: {{ display: true, text: 'Avg Cost ($)', color: '#8b949e' }}, grid: {{ color: '#21262d' }}, ticks: {{ color: '#8b949e' }} }},
      y: {{ title: {{ display: true, text: 'Avg Quality (1-5)', color: '#8b949e' }}, grid: {{ color: '#21262d' }}, ticks: {{ color: '#8b949e' }} }}
    }},
    plugins: {{
      tooltip: {{ callbacks: {{ label: ctx => ctx.raw.label + ' ($' + ctx.raw.x.toFixed(4) + ', Q' + ctx.raw.y.toFixed(1) + ')' }} }},
      legend: {{ labels: {{ color: '#e6edf3' }} }}
    }}
  }}
}});

// Comparison table
const compBody = document.getElementById('comparisonBody');
comparison.forEach(r => {{
  compBody.innerHTML += `<tr><td>${{r.provider}}</td><td>${{r.model}}</td><td>${{r.runs}}</td>` +
    `<td>${{r.avg_latency?.toFixed(0) || '—'}}ms</td><td>${{r.avg_cost?.toFixed(6) || '—'}}</td>` +
    `<td>${{r.avg_quality?.toFixed(1) || '—'}}</td><td>${{r.tok_per_sec?.toFixed(1) || '—'}}</td></tr>`;
}});

// Winners table
const winBody = document.getElementById('winnersBody');
const seen = new Set();
winners.forEach(r => {{
  if (!seen.has(r.category)) {{
    seen.add(r.category);
    winBody.innerHTML += `<tr><td>${{r.category}}</td><td class="winner">${{r.provider}}</td>` +
      `<td class="winner">${{r.model}}</td><td>${{r.avg_quality?.toFixed(1) || '—'}}</td>` +
      `<td>${{r.avg_cost?.toFixed(6) || '—'}}</td></tr>`;
  }}
}});

// Trend charts
const trendKeys = Object.keys(trends);
function makeTrendChart(canvasId, metric) {{
  const datasets = trendKeys.map((key, i) => ({{
    label: key,
    data: trends[key].map(d => ({{ x: d.day, y: d[metric] || 0 }})),
    borderColor: colors[i % colors.length],
    backgroundColor: 'transparent',
    tension: 0.3,
    pointRadius: 2,
  }}));
  new Chart(document.getElementById(canvasId), {{
    type: 'line',
    data: {{ datasets }},
    options: {{
      scales: {{
        x: {{ type: 'category', labels: [...new Set(trendKeys.flatMap(k => trends[k].map(d => d.day)))].sort(),
              grid: {{ color: '#21262d' }}, ticks: {{ color: '#8b949e', maxTicksLimit: 10 }} }},
        y: {{ grid: {{ color: '#21262d' }}, ticks: {{ color: '#8b949e' }} }}
      }},
      plugins: {{ legend: {{ labels: {{ color: '#e6edf3' }} }} }}
    }}
  }});
}}
makeTrendChart('latencyTrendChart', 'latency');
makeTrendChart('qualityTrendChart', 'quality');

// Task breakdown table
const tbBody = document.getElementById('taskBreakdownBody');
const seenCat = new Set();
taskBreakdown.forEach(r => {{
  const isFirst = !seenCat.has(r.category);
  if (isFirst) seenCat.add(r.category);
  tbBody.innerHTML += `<tr><td>${{r.category}}</td><td class="${{isFirst ? 'winner' : ''}}">${{r.provider}}</td>` +
    `<td class="${{isFirst ? 'winner' : ''}}">${{r.model}}</td><td>${{r.avg_quality?.toFixed(1) || '—'}}</td>` +
    `<td>${{r.avg_latency?.toFixed(0) || '—'}}ms</td><td>${{r.avg_cost?.toFixed(6) || '—'}}</td></tr>`;
}});
</script>
</body>
</html>"""
