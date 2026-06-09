import logging
from dataclasses import dataclass

from .metrics import BenchmarkResult

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    model: str
    avg_latency: float
    avg_throughput: float
    avg_judge_score: float | None
    reason: str


def _aggregate(
    results: list[BenchmarkResult],
    scores: dict[int, dict[str, float]] | None,
) -> dict[str, dict]:
    """Compute per-model averages from raw results and quality scores."""
    scores = scores or {}
    buckets: dict[str, dict] = {}

    for i, r in enumerate(results):
        if r.model not in buckets:
            buckets[r.model] = {"latency": [], "throughput": [], "judge": []}
        buckets[r.model]["latency"].append(r.total_latency)
        buckets[r.model]["throughput"].append(r.throughput)
        if "llm_judge" in scores.get(i, {}):
            buckets[r.model]["judge"].append(scores[i]["llm_judge"])

    aggregated = {}
    for model, data in buckets.items():
        aggregated[model] = {
            "avg_latency": sum(data["latency"]) / len(data["latency"]),
            "avg_throughput": sum(data["throughput"]) / len(data["throughput"]),
            "avg_judge": sum(data["judge"]) / len(data["judge"]) if data["judge"] else None,
        }

    return aggregated


def recommend(
    results: list[BenchmarkResult],
    scores: dict[int, dict[str, float]] | None = None,
    max_latency: float | None = None,
    min_quality: float | None = None,
    optimize_for: str = "latency",
) -> Recommendation | None:
    """Recommend the best model given constraints.

    Args:
        results: All benchmark results.
        scores: Optional quality scores keyed by result index.
        max_latency: Reject models with avg latency above this value (seconds).
        min_quality: Reject models with avg LLM judge score below this value (1-10).
        optimize_for: Ranking criterion — "latency", "throughput", or "quality".

    Returns:
        Recommendation for the best model, or None if no model passes the constraints.
    """
    if optimize_for not in ("latency", "throughput", "quality"):
        raise ValueError(f"optimize_for must be 'latency', 'throughput', or 'quality'")

    aggregated = _aggregate(results, scores)

    # --- Filter ---
    candidates = {}
    for model, data in aggregated.items():
        if max_latency is not None and data["avg_latency"] > max_latency:
            logger.info("Rejected %s: avg_latency=%.3fs > max=%.3fs", model, data["avg_latency"], max_latency)
            continue
        if min_quality is not None:
            if data["avg_judge"] is None:
                logger.info("Rejected %s: no judge scores available", model)
                continue
            if data["avg_judge"] < min_quality:
                logger.info("Rejected %s: avg_judge=%.1f < min=%.1f", model, data["avg_judge"], min_quality)
                continue
        candidates[model] = data

    if not candidates:
        logger.warning("No model passed the constraints (max_latency=%s, min_quality=%s)", max_latency, min_quality)
        return None

    # --- Rank ---
    if optimize_for == "latency":
        best = min(candidates, key=lambda m: candidates[m]["avg_latency"])
        reason = f"lowest avg latency ({candidates[best]['avg_latency']:.3f}s)"
    elif optimize_for == "throughput":
        best = max(candidates, key=lambda m: candidates[m]["avg_throughput"])
        reason = f"highest avg throughput ({candidates[best]['avg_throughput']:.1f} tok/s)"
    else:  # quality
        best = max(candidates, key=lambda m: candidates[m]["avg_judge"] or 0)
        reason = f"highest avg judge score ({candidates[best]['avg_judge']:.1f}/10)"

    data = candidates[best]
    logger.info("Recommended model: %s — %s", best, reason)

    return Recommendation(
        model=best,
        avg_latency=round(data["avg_latency"], 3),
        avg_throughput=round(data["avg_throughput"], 1),
        avg_judge_score=round(data["avg_judge"], 1) if data["avg_judge"] is not None else None,
        reason=reason,
    )
