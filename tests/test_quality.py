import json
from unittest.mock import MagicMock, patch

from observatory.quality import (
    QualityJudge,
    QualityScore,
    RUBRICS,
    _build_judge_prompt,
    compute_extraction_precision_recall,
)


def test_rubrics_cover_all_categories():
    assert set(RUBRICS.keys()) == {"summarization", "extraction", "trend_analysis", "qa"}
    for cat, rubric in RUBRICS.items():
        assert len(rubric["criteria"]) == 3
        assert rubric["max_score"] == 5


def test_build_judge_prompt_contains_criteria():
    prompt = _build_judge_prompt("summarization", "Summarize this", "Some output")
    assert "accuracy" in prompt
    assert "completeness" in prompt
    assert "conciseness" in prompt
    assert "Summarize this" in prompt
    assert "Some output" in prompt


def test_quality_score_dataclass():
    qs = QualityScore(
        overall=4.2,
        criteria_scores={"accuracy": 4.0, "completeness": 4.5},
        reasoning="Good output",
    )
    assert qs.overall == 4.2
    assert qs.error is None


def test_quality_judge_unknown_category():
    judge = QualityJudge.__new__(QualityJudge)  # skip __init__ to avoid API client
    result = judge.score("nonexistent", "prompt", "output")
    assert result.error is not None
    assert "Unknown category" in result.error


def test_quality_judge_score():
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps({
        "criteria_scores": {"accuracy": 4.0, "completeness": 5.0, "conciseness": 3.0},
        "overall": 4.0,
        "reasoning": "Good summary",
    }))]

    with patch("observatory.quality.Anthropic") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.messages.create.return_value = mock_response

        judge = QualityJudge()
        result = judge.score("summarization", "Summarize this", "Key points: ...")
        assert result.overall == 4.0
        assert result.criteria_scores["accuracy"] == 4.0
        assert result.error is None


def test_quality_judge_handles_markdown_code_blocks():
    json_str = json.dumps({
        "criteria_scores": {"correctness": 5.0, "grounding": 4.0, "completeness": 4.0},
        "overall": 4.3,
        "reasoning": "Accurate",
    })
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=f"```json\n{json_str}\n```")]

    with patch("observatory.quality.Anthropic") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.messages.create.return_value = mock_response

        judge = QualityJudge()
        result = judge.score("qa", "What is X?", "X is Y")
        assert result.overall == 4.3
        assert result.error is None


def test_quality_judge_score_with_consistency():
    scores = [
        {"criteria_scores": {"accuracy": 4.0}, "overall": 4.0, "reasoning": "r1"},
        {"criteria_scores": {"accuracy": 4.0}, "overall": 4.2, "reasoning": "r2"},
        {"criteria_scores": {"accuracy": 4.0}, "overall": 4.1, "reasoning": "r3"},
    ]
    call_count = {"n": 0}

    def mock_create(**kwargs):
        resp = MagicMock()
        resp.content = [MagicMock(text=json.dumps(scores[call_count["n"]]))]
        call_count["n"] += 1
        return resp

    with patch("observatory.quality.Anthropic") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.messages.create.side_effect = mock_create

        judge = QualityJudge()
        result, std_dev = judge.score_with_consistency("summarization", "Prompt", "Output")
        assert result.overall == 4.1  # mean of 4.0, 4.2, 4.1
        assert std_dev < 0.3  # should be stable


def test_extraction_precision_recall_perfect():
    p, r, f1 = compute_extraction_precision_recall(
        ["Apple", "Google"], ["Apple", "Google"]
    )
    assert p == 1.0
    assert r == 1.0
    assert f1 == 1.0


def test_extraction_precision_recall_partial():
    p, r, f1 = compute_extraction_precision_recall(
        ["Apple", "Microsoft"], ["Apple", "Google"]
    )
    assert p == 0.5  # 1/2 extracted are correct
    assert r == 0.5  # 1/2 ground truth found
    assert f1 == 0.5


def test_extraction_precision_recall_empty():
    p, r, f1 = compute_extraction_precision_recall([], ["Apple"])
    assert p == 0.0
    assert r == 0.0
    assert f1 == 0.0


def test_extraction_precision_recall_case_insensitive():
    p, r, f1 = compute_extraction_precision_recall(
        ["apple", "GOOGLE"], ["Apple", "Google"]
    )
    assert p == 1.0
    assert r == 1.0
