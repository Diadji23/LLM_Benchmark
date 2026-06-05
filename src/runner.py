import json
import logging
import time

import httpx

from .metrics import BenchmarkResult, compute_metrics

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
DEFAULT_TIMEOUT = 120.0


async def stream_ollama(
    model: str,
    prompt: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> BenchmarkResult:
    """Stream a prompt to Ollama and return measured performance metrics."""
    payload = {"model": model, "prompt": prompt, "stream": True}

    full_response_parts: list[str] = []
    first_token_time: float | None = None
    num_tokens: int = 0

    start_time = time.perf_counter()

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", _GENERATE_URL, json=payload) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line:
                    continue

                chunk: dict = json.loads(line)
                token_text = chunk.get("response", "")

                # Record timestamp on the very first non-empty token
                if first_token_time is None and token_text:
                    first_token_time = time.perf_counter()

                full_response_parts.append(token_text)

                if chunk.get("done"):
                    # eval_count is Ollama's own token count — more accurate than counting chunks
                    num_tokens = chunk.get("eval_count", 0)
                    break

    end_time = time.perf_counter()

    if first_token_time is None:
        first_token_time = end_time
        logger.warning("Empty response from model=%s prompt=%.60r", model, prompt)

    metrics = compute_metrics(start_time, first_token_time, end_time, num_tokens)

    return BenchmarkResult(
        model=model,
        prompt=prompt,
        response="".join(full_response_parts),
        ttft=metrics["TTFT"],
        total_latency=metrics["total latency"],
        throughput=metrics["throughput"],
        num_tokens=num_tokens,
    )


async def run_benchmark(
    models: list[str],
    prompts: list[str],
) -> list[BenchmarkResult]:
    """Run benchmark sequentially across all (model, prompt) pairs."""
    results: list[BenchmarkResult] = []

    for model in models:
        for prompt in prompts:
            logger.info("Running model=%s prompt=%.60r", model, prompt)
            try:
                result = await stream_ollama(model, prompt)
                results.append(result)
                logger.info(
                    "model=%s ttft=%.3fs total=%.3fs throughput=%.1f tok/s",
                    model, result.ttft, result.total_latency, result.throughput,
                )
            except httpx.ConnectError:
                logger.error("Cannot reach Ollama at %s — run `ollama serve`.", OLLAMA_BASE_URL)
                raise
            except httpx.HTTPStatusError as exc:
                logger.error("Model %r not available (HTTP %d).", model, exc.response.status_code)
                raise

    return results
