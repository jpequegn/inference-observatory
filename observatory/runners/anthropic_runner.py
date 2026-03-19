from anthropic import Anthropic

from observatory.runners.base import BaseRunner, RunResult

# Pricing per 1M tokens (as of early 2026)
PRICING = {
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-6-20260320": {"input": 3.00, "output": 15.00},
}

# Convenient aliases
MODEL_ALIASES = {
    "claude-haiku-4-5": "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6": "claude-sonnet-4-6-20260320",
}


class AnthropicRunner(BaseRunner):
    provider = "anthropic"

    def __init__(self, model: str = "claude-haiku-4-5"):
        resolved = MODEL_ALIASES.get(model, model)
        super().__init__(resolved)
        self.client = Anthropic()

    def _call(self, prompt: str) -> RunResult:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        output = response.content[0].text
        tokens_in = response.usage.input_tokens
        tokens_out = response.usage.output_tokens

        prices = PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        cost = (tokens_in * prices["input"] + tokens_out * prices["output"]) / 1_000_000

        return RunResult(
            output=output,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
        )
