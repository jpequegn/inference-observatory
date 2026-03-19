import statistics
import threading
import time
from dataclasses import dataclass, field

import psutil


@dataclass
class SystemSnapshot:
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    # MPS utilization not directly available via psutil;
    # we track whether Apple GPU is likely active via process checks
    mps_active: bool = False


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics from multiple runs of the same task."""
    task_id: str = ""
    provider: str = ""
    model: str = ""
    latencies_ms: list[float] = field(default_factory=list)
    tokens_in: list[int] = field(default_factory=list)
    tokens_out: list[int] = field(default_factory=list)
    costs_usd: list[float] = field(default_factory=list)
    system_snapshots: list[SystemSnapshot] = field(default_factory=list)
    errors: int = 0

    @property
    def p50_latency(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_l = sorted(self.latencies_ms)
        return sorted_l[len(sorted_l) // 2]

    @property
    def p95_latency(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_l = sorted(self.latencies_ms)
        idx = int(len(sorted_l) * 0.95)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    @property
    def avg_latency(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return statistics.mean(self.latencies_ms)

    @property
    def total_cost(self) -> float:
        return sum(self.costs_usd)

    @property
    def avg_cpu(self) -> float:
        if not self.system_snapshots:
            return 0.0
        return statistics.mean(s.cpu_percent for s in self.system_snapshots)

    @property
    def avg_memory(self) -> float:
        if not self.system_snapshots:
            return 0.0
        return statistics.mean(s.memory_percent for s in self.system_snapshots)

    @property
    def avg_tokens_per_sec(self) -> float:
        total_tokens = sum(self.tokens_out)
        total_secs = sum(self.latencies_ms) / 1000
        if total_secs == 0:
            return 0.0
        return total_tokens / total_secs


class SystemMonitor:
    """Collects system metrics in a background thread during a run."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.snapshots: list[SystemSnapshot] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._stop.clear()
        self.snapshots = []
        self._thread = threading.Thread(target=self._collect, daemon=True)
        self._thread.start()

    def stop(self) -> list[SystemSnapshot]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return self.snapshots

    def _collect(self):
        while not self._stop.is_set():
            snap = SystemSnapshot(
                cpu_percent=psutil.cpu_percent(interval=None),
                memory_percent=psutil.virtual_memory().percent,
            )
            self.snapshots.append(snap)
            self._stop.wait(self.interval)
