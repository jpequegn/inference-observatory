import json
from unittest.mock import MagicMock, patch

from observatory.grounding import (
    GroundingJudge,
    GroundingResult,
    HallucinationReport,
    GROUNDING_CATEGORIES,
    _build_grounding_prompt,
)


def test_grounding_categories():
    assert "summarization" in GROUNDING_CATEGORIES
    assert "qa" in GROUNDING_CATEGORIES


def test_build_grounding_prompt():
    prompt = _build_grounding_prompt("Source text here", "Output text here")
    assert "Source text here" in prompt
    assert "Output text here" in prompt
    assert "total_claims" in prompt
    assert "grounded_claims" in prompt


def test_grounding_result_defaults():
    r = GroundingResult(grounding_score=0.8, total_claims=10, grounded_claims=8)
    assert r.grounding_score == 0.8
    assert r.ungrounded_claims == []
    assert r.citations_found == 0
    assert r.error is None


def test_grounding_judge_empty_input():
    judge = GroundingJudge.__new__(GroundingJudge)
    result = judge.score("", "some output")
    assert result.error is not None
    assert result.grounding_score == 0.0


def test_grounding_judge_score():
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps({
        "total_claims": 5,
        "grounded_claims": 4,
        "ungrounded_claims": ["CEO founded in 2020"],
        "citations_found": 2,
        "uncertainty_expressed": True,
        "reasoning": "Mostly grounded",
    }))]

    with patch("observatory.grounding.Anthropic") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.messages.create.return_value = mock_response

        judge = GroundingJudge()
        result = judge.score("Source about a company", "Output about the company")
        assert result.grounding_score == 0.8
        assert result.total_claims == 5
        assert result.grounded_claims == 4
        assert len(result.ungrounded_claims) == 1
        assert result.citations_found == 2
        assert result.uncertainty_expressed is True
        assert result.error is None


def test_grounding_judge_handles_code_blocks():
    json_str = json.dumps({
        "total_claims": 3, "grounded_claims": 3,
        "ungrounded_claims": [], "citations_found": 0,
        "uncertainty_expressed": False, "reasoning": "All grounded",
    })
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=f"```json\n{json_str}\n```")]

    with patch("observatory.grounding.Anthropic") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.messages.create.return_value = mock_response

        judge = GroundingJudge()
        result = judge.score("Source", "Output")
        assert result.grounding_score == 1.0
        assert result.total_claims == 3


def test_hallucination_report_empty():
    judge = GroundingJudge.__new__(GroundingJudge)
    report = judge.build_hallucination_report("openai", "gpt-4o", [])
    assert report.total_outputs == 0
    assert report.avg_grounding_score == 0.0


def test_hallucination_report_with_data():
    results = [
        {"total_claims": 5, "grounded_claims": 4, "ungrounded_claims": ["x"],
         "citations_found": 1, "uncertainty_expressed": True, "reasoning": "ok"},
        {"total_claims": 3, "grounded_claims": 3, "ungrounded_claims": [],
         "citations_found": 0, "uncertainty_expressed": False, "reasoning": "good"},
    ]
    call_count = {"n": 0}

    def mock_create(**kwargs):
        resp = MagicMock()
        resp.content = [MagicMock(text=json.dumps(results[call_count["n"]]))]
        call_count["n"] += 1
        return resp

    with patch("observatory.grounding.Anthropic") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.messages.create.side_effect = mock_create

        judge = GroundingJudge()
        outputs = [
            {"source_text": "src1", "output_text": "out1", "task_id": "sum-01"},
            {"source_text": "src2", "output_text": "out2", "task_id": "sum-02"},
        ]
        report = judge.build_hallucination_report("openai", "gpt-4o", outputs)
        assert report.total_outputs == 2
        assert report.avg_grounding_score > 0
        assert report.hallucination_rate == 0.5  # 1 of 2 had ungrounded claims
        assert report.expresses_uncertainty_rate == 0.5
        assert len(report.flagged_outputs) == 1


def test_hallucination_report_dataclass():
    report = HallucinationReport(
        provider="ollama", model="llama3.2", total_outputs=10,
        avg_grounding_score=0.7, hallucination_rate=0.3,
        expresses_uncertainty_rate=0.2,
    )
    assert report.provider == "ollama"
    assert report.hallucination_rate == 0.3
