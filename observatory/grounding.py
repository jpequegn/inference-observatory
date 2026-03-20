"""Grounding and attribution scoring — detect hallucinations and measure trustworthiness."""

import json
from dataclasses import dataclass, field

from anthropic import Anthropic

JUDGE_MODEL = "claude-sonnet-4-6-20260320"

GROUNDING_CATEGORIES = {"summarization", "qa"}


@dataclass
class GroundingResult:
    grounding_score: float  # 0.0-1.0, % of claims traceable to source
    total_claims: int
    grounded_claims: int
    ungrounded_claims: list[str] = field(default_factory=list)
    citations_found: int = 0
    uncertainty_expressed: bool = False
    reasoning: str = ""
    error: str | None = None


@dataclass
class HallucinationReport:
    provider: str
    model: str
    total_outputs: int
    avg_grounding_score: float
    hallucination_rate: float  # % of outputs with ungrounded claims
    expresses_uncertainty_rate: float  # % of outputs that hedge appropriately
    flagged_outputs: list[dict] = field(default_factory=list)


def _build_grounding_prompt(source_text: str, output: str) -> str:
    return f"""You are an expert fact-checker. Analyze the following LLM output and determine which claims are grounded in the source material.

## Source Material
{source_text}

## LLM Output to Evaluate
{output}

## Instructions
1. Identify every factual claim in the output
2. For each claim, determine if it is directly traceable to the source material
3. Count any citations or source references in the output
4. Note if the output expresses uncertainty or hedging (e.g., "according to the transcript", "it appears that")

Respond with ONLY valid JSON:
{{
  "total_claims": <int>,
  "grounded_claims": <int>,
  "ungrounded_claims": ["claim 1 not in source", "claim 2 not in source"],
  "citations_found": <int - number of explicit source references>,
  "uncertainty_expressed": <bool - does the output hedge or express uncertainty>,
  "reasoning": "Brief explanation"
}}"""


class GroundingJudge:
    """Scores how well LLM outputs are grounded in source material."""

    def __init__(self):
        self.client = Anthropic()
        self.model = JUDGE_MODEL

    def score(self, source_text: str, output: str) -> GroundingResult:
        """Score grounding of a single output against source material."""
        if not source_text or not output:
            return GroundingResult(
                grounding_score=0.0, total_claims=0, grounded_claims=0,
                error="Empty source or output",
            )

        prompt = _build_grounding_prompt(source_text, output)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(text)

            total = int(data["total_claims"])
            grounded = int(data["grounded_claims"])
            score = grounded / total if total > 0 else 0.0

            return GroundingResult(
                grounding_score=round(score, 3),
                total_claims=total,
                grounded_claims=grounded,
                ungrounded_claims=data.get("ungrounded_claims", []),
                citations_found=int(data.get("citations_found", 0)),
                uncertainty_expressed=bool(data.get("uncertainty_expressed", False)),
                reasoning=data.get("reasoning", ""),
            )
        except Exception as e:
            return GroundingResult(
                grounding_score=0.0, total_claims=0, grounded_claims=0,
                error=str(e),
            )

    def build_hallucination_report(
        self, provider: str, model: str, outputs: list[dict],
    ) -> HallucinationReport:
        """Score multiple outputs and build a provider-level hallucination report.

        Each output dict should have: source_text, output_text, task_id
        """
        results = []
        flagged = []

        for entry in outputs:
            result = self.score(entry["source_text"], entry["output_text"])
            if result.error:
                continue
            results.append(result)
            if result.ungrounded_claims:
                flagged.append({
                    "task_id": entry.get("task_id", ""),
                    "grounding_score": result.grounding_score,
                    "ungrounded_claims": result.ungrounded_claims,
                })

        if not results:
            return HallucinationReport(
                provider=provider, model=model, total_outputs=0,
                avg_grounding_score=0.0, hallucination_rate=0.0,
                expresses_uncertainty_rate=0.0,
            )

        avg_score = sum(r.grounding_score for r in results) / len(results)
        halluc_rate = sum(1 for r in results if r.ungrounded_claims) / len(results)
        uncert_rate = sum(1 for r in results if r.uncertainty_expressed) / len(results)

        return HallucinationReport(
            provider=provider, model=model, total_outputs=len(results),
            avg_grounding_score=round(avg_score, 3),
            hallucination_rate=round(halluc_rate, 3),
            expresses_uncertainty_rate=round(uncert_rate, 3),
            flagged_outputs=flagged,
        )
