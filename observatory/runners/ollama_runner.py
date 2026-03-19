import ollama as ollama_client

from observatory.runners.base import BaseRunner, RunResult

DEFAULT_MODELS = ["llama3.2", "gemma3", "mistral"]


class OllamaRunner(BaseRunner):
    provider = "ollama"
    is_local = True

    def __init__(self, model: str = "llama3.2"):
        super().__init__(model)

    def _call(self, prompt: str) -> RunResult:
        response = ollama_client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        msg = response["message"]
        tokens_in = response.get("prompt_eval_count", 0)
        tokens_out = response.get("eval_count", 0)
        return RunResult(
            output=msg["content"],
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=0.0,  # local inference, no cost
        )
