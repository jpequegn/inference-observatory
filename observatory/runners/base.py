from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time

from observatory.metrics import SystemMonitor, SystemSnapshot


@dataclass
class RunResult:
    output: str = ""
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    system_snapshots: list[SystemSnapshot] = field(default_factory=list)


class BaseRunner(ABC):
    provider: str
    model: str
    is_local: bool = False  # Override in local runners

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def _call(self, prompt: str) -> RunResult:
        """Execute the prompt against the provider and return a RunResult."""
        ...

    def run(self, task: dict, collect_system_metrics: bool = False) -> RunResult:
        """Run a benchmark task and measure latency."""
        prompt = task["prompt"]
        monitor = None
        if collect_system_metrics and self.is_local:
            monitor = SystemMonitor()
            monitor.start()

        start = time.perf_counter()
        try:
            result = self._call(prompt)
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            if monitor:
                monitor.stop()
            return RunResult(latency_ms=elapsed, error=str(e))

        elapsed = (time.perf_counter() - start) * 1000
        result.latency_ms = elapsed
        if monitor:
            result.system_snapshots = monitor.stop()
        return result
