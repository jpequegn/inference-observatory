from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time


@dataclass
class RunResult:
    output: str = ""
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    error: str | None = None


class BaseRunner(ABC):
    provider: str
    model: str

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def _call(self, prompt: str) -> RunResult:
        """Execute the prompt against the provider and return a RunResult."""
        ...

    def run(self, task: dict) -> RunResult:
        """Run a benchmark task and measure latency."""
        prompt = task["prompt"]
        start = time.perf_counter()
        try:
            result = self._call(prompt)
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return RunResult(latency_ms=elapsed, error=str(e))
        elapsed = (time.perf_counter() - start) * 1000
        result.latency_ms = elapsed
        return result
