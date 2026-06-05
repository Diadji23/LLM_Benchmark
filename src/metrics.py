from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    model: str
    prompt: str
    response: str
    ttft: float
    total_latency: float
    throughput: float
    num_tokens: int


def compute_metrics(
    start_time: float,
    first_token_time: float,
    end_time: float,
    num_tokens: int,
) -> dict[str, float]:
    ttft = first_token_time - start_time
    total_latency = end_time - start_time
    throughput = num_tokens / total_latency if total_latency > 0 else 0.0
    return {"TTFT": ttft, "total latency": total_latency, "throughput": throughput}
