from unittest.mock import MagicMock, patch
from observatory.runners.base import BaseRunner, RunResult
from observatory.runners import RUNNERS


SAMPLE_TASK = {
    "id": "test-01",
    "name": "Test Task",
    "category": "qa",
    "prompt": "What is 2+2?",
    "expected_output": "4",
    "difficulty": "easy",
}


class ConcreteRunner(BaseRunner):
    provider = "test"

    def _call(self, prompt: str) -> RunResult:
        return RunResult(output="4", tokens_in=10, tokens_out=5)


class ErrorRunner(BaseRunner):
    provider = "test"

    def _call(self, prompt: str) -> RunResult:
        raise RuntimeError("connection failed")


def test_run_result_defaults():
    r = RunResult()
    assert r.output == ""
    assert r.latency_ms == 0.0
    assert r.tokens_in == 0
    assert r.tokens_out == 0
    assert r.cost_usd == 0.0
    assert r.error is None


def test_base_runner_measures_latency():
    runner = ConcreteRunner(model="test-model")
    result = runner.run(SAMPLE_TASK)
    assert result.output == "4"
    assert result.latency_ms > 0
    assert result.error is None


def test_base_runner_catches_errors():
    runner = ErrorRunner(model="test-model")
    result = runner.run(SAMPLE_TASK)
    assert result.error == "connection failed"
    assert result.latency_ms > 0


def test_all_runners_registered():
    assert "ollama" in RUNNERS
    assert "mlx" in RUNNERS
    assert "openai" in RUNNERS
    assert "anthropic" in RUNNERS


def test_ollama_runner_call():
    with patch("observatory.runners.ollama_runner.ollama_client") as mock:
        mock.chat.return_value = {
            "message": {"content": "Answer: 4"},
            "prompt_eval_count": 15,
            "eval_count": 8,
        }
        from observatory.runners.ollama_runner import OllamaRunner
        runner = OllamaRunner(model="llama3.2")
        result = runner.run(SAMPLE_TASK)
        assert result.output == "Answer: 4"
        assert result.tokens_in == 15
        assert result.tokens_out == 8
        assert result.cost_usd == 0.0
        assert result.error is None


def test_openai_runner_call():
    with patch("observatory.runners.openai_runner.OpenAI") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Answer: 4"))]
        mock_response.usage = MagicMock(prompt_tokens=20, completion_tokens=10)
        mock_client.chat.completions.create.return_value = mock_response

        from observatory.runners.openai_runner import OpenAIRunner
        runner = OpenAIRunner(model="gpt-4o-mini")
        result = runner.run(SAMPLE_TASK)
        assert result.output == "Answer: 4"
        assert result.tokens_in == 20
        assert result.tokens_out == 10
        assert result.cost_usd > 0
        assert result.error is None


def test_anthropic_runner_call():
    with patch("observatory.runners.anthropic_runner.Anthropic") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Answer: 4")]
        mock_response.usage = MagicMock(input_tokens=20, output_tokens=10)
        mock_client.messages.create.return_value = mock_response

        from observatory.runners.anthropic_runner import AnthropicRunner
        runner = AnthropicRunner(model="claude-haiku-4-5")
        result = runner.run(SAMPLE_TASK)
        assert result.output == "Answer: 4"
        assert result.tokens_in == 20
        assert result.tokens_out == 10
        assert result.cost_usd > 0
        assert result.error is None
