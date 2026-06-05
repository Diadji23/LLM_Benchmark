import logging
import re

import httpx
from rouge_score import rouge_scorer

from .metrics import BenchmarkResult

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"

_JUDGE_PROMPT = """\
Rate the following answer from 1 to 10 based on accuracy, clarity, and completeness.
Reply with a single integer only.

Question: {prompt}
Answer: {response}

Score:"""


def compute_rouge_l(response: str, reference: str) -> float:
    """Return ROUGE-L F1 score between response and reference."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, response)["rougeL"].fmeasure


async def llm_judge(
    prompt: str,
    response: str,
    judge_model: str = "llama3.2:1b",
) -> float:
    """Ask a local model to score a response from 1 to 10."""
    judge_input = _JUDGE_PROMPT.format(prompt=prompt, response=response)

    # Non-streaming: we only need the final answer, no TTFT to measure here.
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": judge_model, "prompt": judge_input, "stream": False},
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()

    match = re.search(r"\d+", text)
    if not match:
        logger.warning("Judge returned non-numeric response: %r", text)
        return 0.0

    return min(float(match.group()), 10.0)


async def evaluate(
    result: BenchmarkResult,
    reference: str | None = None,
    judge_model: str = "llama3.2:1b",
) -> dict[str, float]:
    """Compute quality scores for a BenchmarkResult.

    Args:
        result: The benchmark result to evaluate.
        reference: Optional ground-truth answer for ROUGE scoring.
        judge_model: Ollama model used as judge.

    Returns:
        Dict with "rouge_l" (if reference given) and "llm_judge" keys.
    """
    scores: dict[str, float] = {}

    if reference:
        scores["rouge_l"] = compute_rouge_l(result.response, reference)
        logger.info("model=%s rouge_l=%.3f", result.model, scores["rouge_l"])

    scores["llm_judge"] = await llm_judge(result.prompt, result.response, judge_model)
    logger.info("model=%s llm_judge=%.1f", result.model, scores["llm_judge"])

    return scores
