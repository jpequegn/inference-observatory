from observatory.runners.base import BaseRunner, RunResult
from observatory.runners.ollama_runner import OllamaRunner
from observatory.runners.mlx_runner import MLXRunner
from observatory.runners.openai_runner import OpenAIRunner
from observatory.runners.anthropic_runner import AnthropicRunner

RUNNERS = {
    "ollama": OllamaRunner,
    "mlx": MLXRunner,
    "openai": OpenAIRunner,
    "anthropic": AnthropicRunner,
}

__all__ = [
    "BaseRunner",
    "RunResult",
    "OllamaRunner",
    "MLXRunner",
    "OpenAIRunner",
    "AnthropicRunner",
    "RUNNERS",
]
