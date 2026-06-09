import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from src.agent import recommend
from src.evaluator import evaluate
from src.reporter import generate_html, generate_json
from src.runner import run_benchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark local Ollama models.")
    parser.add_argument(
        "--models", nargs="+", default=["llama3.2:1b"],
        help="Ollama models to benchmark (default: llama3.2:1b)",
    )
    parser.add_argument(
        "--questions", type=Path, default=Path("benchmarks/questions.json"),
        help="Path to questions JSON file",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="Directory for report output (default: results/)",
    )
    parser.add_argument(
        "--max-latency", type=float, default=None,
        help="Max acceptable avg latency in seconds for recommendation",
    )
    parser.add_argument(
        "--min-quality", type=float, default=None,
        help="Min acceptable avg judge score (1-10) for recommendation",
    )
    parser.add_argument(
        "--optimize-for", choices=["latency", "throughput", "quality"], default="latency",
        help="Criterion to optimize when recommending (default: latency)",
    )
    parser.add_argument(
        "--judge-model", default="llama3.2:1b",
        help="Ollama model used as LLM judge (default: llama3.2:1b)",
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Skip LLM-as-judge evaluation (faster)",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    # Load questions
    if not args.questions.exists():
        logger.error("Questions file not found: %s", args.questions)
        sys.exit(1)

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    prompts = [q["prompt"] for q in questions]
    logger.info("Loaded %d questions from %s", len(questions), args.questions)

    # Run benchmark
    logger.info("Starting benchmark on models: %s", args.models)
    results = await run_benchmark(args.models, prompts)

    # Evaluate quality
    scores: dict[int, dict[str, float]] = {}
    for i, result in enumerate(results):
        prompt_idx = i % len(prompts)
        reference = questions[prompt_idx].get("reference")
        scores[i] = await evaluate(
            result,
            reference=reference,
            judge_model=args.judge_model if not args.no_judge else None,
        )

    # Save reports
    args.output_dir.mkdir(exist_ok=True)
    json_path = args.output_dir / "report.json"
    html_path = args.output_dir / "report.html"

    generate_json(results, scores, output_path=json_path)
    generate_html(results, scores, output_path=html_path)
    logger.info("Reports saved → %s | %s", json_path, html_path)

    # Recommend
    rec = recommend(
        results,
        scores,
        max_latency=args.max_latency,
        min_quality=args.min_quality,
        optimize_for=args.optimize_for,
    )

    print("\n" + "=" * 50)
    if rec:
        print(f"  Recommended model : {rec.model}")
        print(f"  Avg latency       : {rec.avg_latency}s")
        print(f"  Avg throughput    : {rec.avg_throughput} tok/s")
        if rec.avg_judge_score is not None:
            print(f"  Avg judge score   : {rec.avg_judge_score}/10")
        print(f"  Reason            : {rec.reason}")
    else:
        print("  No model passed the given constraints.")
    print("=" * 50 + "\n")


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
