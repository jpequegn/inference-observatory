import subprocess
import json

from observatory.runners.base import BaseRunner, RunResult


class MLXRunner(BaseRunner):
    provider = "mlx"
    is_local = True

    def __init__(self, model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"):
        super().__init__(model)

    def _call(self, prompt: str) -> RunResult:
        result = subprocess.run(
            [
                "python3", "-m", "mlx_lm.generate",
                "--model", self.model,
                "--prompt", prompt,
                "--max-tokens", "1024",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return RunResult(error=result.stderr.strip())

        output = result.stdout.strip()
        # mlx_lm.generate prints the generated text directly
        return RunResult(
            output=output,
            cost_usd=0.0,  # local inference, no cost
        )
