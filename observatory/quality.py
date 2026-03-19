import json
import statistics
from dataclasses import dataclass

from anthropic import Anthropic

JUDGE_MODEL = "claude-sonnet-4-6-20260320"

RUBRICS = {
    "summarization": {
        "criteria": [
            ("accuracy", "Does the summary accurately reflect the source content without fabrication?"),
            ("completeness", "Does the summary cover all key points from the source?"),
            ("conciseness", "Is the summary appropriately brief without unnecessary detail?"),
        ],
        "max_score": 5,
    },
    "extraction": {
        "criteria": [
            ("precision", "Are all extracted items actually present in the source?"),
            ("recall", "Are all relevant items from the source captured?"),
            ("format", "Is the output in the requested format with correct structure?"),
        ],
        "max_score": 5,
    },
    "trend_analysis": {
        "criteria": [
            ("insight_quality", "Are the identified trends meaningful and non-obvious?"),
            ("evidence_quality", "Are trends supported with specific references to the source material?"),
            ("coherence", "Is the analysis logically structured and internally consistent?"),
        ],
        "max_score": 5,
    },
    "qa": {
        "criteria": [
            ("correctness", "Is the answer factually correct based on the source?"),
            ("grounding", "Is the answer grounded in the source material rather than external knowledge?"),
            ("completeness", "Does the answer fully address the question asked?"),
        ],
        "max_score": 5,
    },
}


@dataclass
class QualityScore:
    overall: float  # 1-5
    criteria_scores: dict[str, float]  # criterion_name -> score
    reasoning: str
    error: str | None = None


def _build_judge_prompt(category: str, task_prompt: str, output: str) -> str:
    rubric = RUBRICS[category]
    criteria_text = "\n".join(
        f"- **{name}** (1-5): {desc}" for name, desc in rubric["criteria"]
    )
    return f"""You are an expert quality judge for LLM outputs. Score the following output on a 1-5 scale for each criterion.

## Task Category: {category}

## Evaluation Criteria
{criteria_text}

## Original Task Prompt
{task_prompt}

## Output to Evaluate
{output}

## Instructions
Rate each criterion from 1 (poor) to 5 (excellent). Then provide an overall score (1-5) as the average rounded to one decimal.

Respond with ONLY valid JSON in this exact format:
{{
  "criteria_scores": {{"criterion_name": score, ...}},
  "overall": score,
  "reasoning": "Brief explanation of scores"
}}"""


class QualityJudge:
    """Uses a fixed LLM (claude-sonnet-4-6) to score output quality."""

    def __init__(self):
        self.client = Anthropic()
        self.model = JUDGE_MODEL

    def score(self, category: str, task_prompt: str, output: str) -> QualityScore:
        if category not in RUBRICS:
            return QualityScore(
                overall=0.0,
                criteria_scores={},
                reasoning="",
                error=f"Unknown category: {category}",
            )

        prompt = _build_judge_prompt(category, task_prompt, output)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            # Handle potential markdown code blocks
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(text)
            return QualityScore(
                overall=float(data["overall"]),
                criteria_scores={k: float(v) for k, v in data["criteria_scores"].items()},
                reasoning=data.get("reasoning", ""),
            )
        except Exception as e:
            return QualityScore(
                overall=0.0, criteria_scores={}, reasoning="", error=str(e)
            )

    def score_with_consistency(
        self, category: str, task_prompt: str, output: str, runs: int = 3
    ) -> tuple[QualityScore, float]:
        """Score multiple times and return the average score + standard deviation."""
        scores = []
        all_criteria: dict[str, list[float]] = {}
        reasonings = []

        for _ in range(runs):
            result = self.score(category, task_prompt, output)
            if result.error:
                continue
            scores.append(result.overall)
            reasonings.append(result.reasoning)
            for k, v in result.criteria_scores.items():
                all_criteria.setdefault(k, []).append(v)

        if not scores:
            return QualityScore(
                overall=0.0, criteria_scores={}, reasoning="", error="All scoring attempts failed"
            ), 0.0

        avg_overall = statistics.mean(scores)
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0
        avg_criteria = {k: statistics.mean(v) for k, v in all_criteria.items()}

        return QualityScore(
            overall=round(avg_overall, 1),
            criteria_scores={k: round(v, 1) for k, v in avg_criteria.items()},
            reasoning=reasonings[0],  # use first reasoning as representative
        ), round(std_dev, 2)


def compute_extraction_precision_recall(
    extracted: list[str], ground_truth: list[str]
) -> tuple[float, float, float]:
    """Compute exact-match precision, recall, and F1 for extraction tasks."""
    extracted_set = {item.lower().strip() for item in extracted}
    truth_set = {item.lower().strip() for item in ground_truth}

    if not extracted_set:
        return 0.0, 0.0, 0.0

    true_positives = len(extracted_set & truth_set)
    precision = true_positives / len(extracted_set) if extracted_set else 0.0
    recall = true_positives / len(truth_set) if truth_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return round(precision, 3), round(recall, 3), round(f1, 3)
