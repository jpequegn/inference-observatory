import time
from observatory.metrics import BenchmarkMetrics, SystemMonitor, SystemSnapshot


def test_benchmark_metrics_p50_p95():
    bm = BenchmarkMetrics(task_id="t1", provider="test", model="m1")
    bm.latencies_ms = [100, 200, 300, 400, 500]
    assert bm.p50_latency == 300
    assert bm.p95_latency == 500


def test_benchmark_metrics_empty():
    bm = BenchmarkMetrics()
    assert bm.p50_latency == 0.0
    assert bm.p95_latency == 0.0
    assert bm.avg_latency == 0.0
    assert bm.total_cost == 0.0
    assert bm.avg_cpu == 0.0
    assert bm.avg_memory == 0.0
    assert bm.avg_tokens_per_sec == 0.0


def test_benchmark_metrics_avg_latency():
    bm = BenchmarkMetrics()
    bm.latencies_ms = [100, 200, 300]
    assert bm.avg_latency == 200.0


def test_benchmark_metrics_total_cost():
    bm = BenchmarkMetrics()
    bm.costs_usd = [0.001, 0.002, 0.003]
    assert abs(bm.total_cost - 0.006) < 1e-9


def test_benchmark_metrics_tokens_per_sec():
    bm = BenchmarkMetrics()
    bm.tokens_out = [100, 100]
    bm.latencies_ms = [1000, 1000]  # 2 seconds total
    assert bm.avg_tokens_per_sec == 100.0  # 200 tokens / 2 seconds


def test_benchmark_metrics_system_averages():
    bm = BenchmarkMetrics()
    bm.system_snapshots = [
        SystemSnapshot(cpu_percent=50.0, memory_percent=60.0),
        SystemSnapshot(cpu_percent=70.0, memory_percent=80.0),
    ]
    assert bm.avg_cpu == 60.0
    assert bm.avg_memory == 70.0


def test_system_monitor_collects_snapshots():
    monitor = SystemMonitor(interval=0.1)
    monitor.start()
    time.sleep(0.35)
    snapshots = monitor.stop()
    assert len(snapshots) >= 2
    for s in snapshots:
        assert 0 <= s.cpu_percent <= 100 or s.cpu_percent > 100  # multi-core can exceed 100
        assert 0 <= s.memory_percent <= 100


def test_system_monitor_stop_idempotent():
    monitor = SystemMonitor(interval=0.1)
    monitor.start()
    monitor.stop()
    result = monitor.stop()  # should not raise
    assert isinstance(result, list)
