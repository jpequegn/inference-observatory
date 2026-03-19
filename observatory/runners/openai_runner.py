from openai import OpenAI

from observatory.runners.base import BaseRunner, RunResult

# Pricing per 1M tokens (as of early 2026)
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


class OpenAIRunner(BaseRunner):
    provider = "openai"

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__(model)
        self.client = OpenAI()

    def _call(self, prompt: str) -> RunResult:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        choice = response.choices[0]
        usage = response.usage
        tokens_in = usage.prompt_tokens
        tokens_out = usage.completion_tokens

        prices = PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        cost = (tokens_in * prices["input"] + tokens_out * prices["output"]) / 1_000_000

        return RunResult(
            output=choice.message.content or "",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
        )
