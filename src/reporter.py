import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .metrics import BenchmarkResult

logger = logging.getLogger(__name__)


def _build_report(
    results: list[BenchmarkResult],
    scores: dict[int, dict[str, float]] | None = None,
) -> dict:
    """Assemble the full report dict from results and optional quality scores.

    Args:
        results: Benchmark results in order.
        scores: Optional mapping of result index → {"rouge_l": ..., "llm_judge": ...}.
    """
    scores = scores or {}

    rows = []
    for i, r in enumerate(results):
        row = asdict(r)
        row.update(scores.get(i, {}))
        rows.append(row)

    # Aggregate per model
    summary: dict[str, dict] = {}
    for row in rows:
        model = row["model"]
        if model not in summary:
            summary[model] = {"ttft": [], "throughput": [], "llm_judge": []}
        summary[model]["ttft"].append(row["total_latency"])
        summary[model]["throughput"].append(row["throughput"])
        if "llm_judge" in row:
            summary[model]["llm_judge"].append(row["llm_judge"])

    aggregated = {}
    for model, data in summary.items():
        aggregated[model] = {
            "avg_latency": round(sum(data["ttft"]) / len(data["ttft"]), 3),
            "avg_throughput": round(sum(data["throughput"]) / len(data["throughput"]), 1),
        }
        if data["llm_judge"]:
            aggregated[model]["avg_llm_judge"] = round(
                sum(data["llm_judge"]) / len(data["llm_judge"]), 1
            )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_runs": len(results),
        "summary": aggregated,
        "results": rows,
    }


def generate_json(
    results: list[BenchmarkResult],
    scores: dict[int, dict[str, float]] | None = None,
    output_path: Path | None = None,
) -> str:
    """Return the benchmark report as a JSON string, and optionally save it."""
    report = _build_report(results, scores)
    content = json.dumps(report, indent=2, ensure_ascii=False)

    if output_path:
        output_path.write_text(content, encoding="utf-8")
        logger.info("JSON report saved to %s", output_path)

    return content


def generate_html(
    results: list[BenchmarkResult],
    scores: dict[int, dict[str, float]] | None = None,
    output_path: Path | None = None,
) -> str:
    """Return the benchmark report as an HTML string, and optionally save it."""
    report = _build_report(results, scores)

    summary_rows = ""
    for model, data in report["summary"].items():
        judge_cell = f"{data.get('avg_llm_judge', '—')}/10"
        summary_rows += (
            f"<tr><td>{model}</td>"
            f"<td>{data['avg_latency']}s</td>"
            f"<td>{data['avg_throughput']} tok/s</td>"
            f"<td>{judge_cell}</td></tr>\n"
        )

    detail_rows = ""
    for row in report["results"]:
        rouge = f"{row['rouge_l']:.3f}" if "rouge_l" in row else "—"
        judge = f"{row['llm_judge']}/10" if "llm_judge" in row else "—"
        detail_rows += (
            f"<tr>"
            f"<td>{row['model']}</td>"
            f"<td class='prompt'>{row['prompt'][:80]}</td>"
            f"<td>{row['ttft']:.3f}s</td>"
            f"<td>{row['total_latency']:.3f}s</td>"
            f"<td>{row['throughput']:.1f}</td>"
            f"<td>{row['num_tokens']}</td>"
            f"<td>{rouge}</td>"
            f"<td>{judge}</td>"
            f"</tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>LLM Benchmark Report</title>
  <link rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
  <style>
    body {{ padding: 2rem; }}
    .prompt {{ max-width: 300px; font-size: 0.85rem; color: #555; }}
  </style>
</head>
<body>
  <h1 class="mb-1">LLM Benchmark Report</h1>
  <p class="text-muted">Generated: {report['generated_at']} — {report['total_runs']} runs</p>

  <h2 class="mt-4">Summary per model</h2>
  <table class="table table-bordered table-sm w-auto">
    <thead class="table-dark">
      <tr><th>Model</th><th>Avg latency</th><th>Avg throughput</th><th>Avg judge</th></tr>
    </thead>
    <tbody>{summary_rows}</tbody>
  </table>

  <h2 class="mt-4">All runs</h2>
  <table class="table table-striped table-sm">
    <thead class="table-dark">
      <tr>
        <th>Model</th><th>Prompt</th><th>TTFT</th><th>Latency</th>
        <th>Tok/s</th><th>Tokens</th><th>ROUGE-L</th><th>Judge</th>
      </tr>
    </thead>
    <tbody>{detail_rows}</tbody>
  </table>
</body>
</html>"""

    if output_path:
        output_path.write_text(html, encoding="utf-8")
        logger.info("HTML report saved to %s", output_path)

    return html
