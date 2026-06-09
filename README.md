# LLM Benchmark

Evaluate and compare local LLM models running via [Ollama](https://ollama.com).

## What it measures

| Metric | Description |
|--------|-------------|
| TTFT | Time To First Token |
| Latency | Total response time |
| Throughput | Tokens per second |
| ROUGE-L | Lexical similarity vs a reference answer |
| LLM Judge | Quality score 1–10 from a local judge model |

## Stack

- Python 3.11+
- [Ollama](https://ollama.com) running locally
- `httpx` — async streaming calls
- `rouge-score` — ROUGE-L evaluation
- `FastAPI` + `uvicorn` — REST API
- `pytest` — test suite

## Setup

```bash
git clone https://github.com/Diadji23/LLM_Benchmark.git
cd LLM_Benchmark

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements-dev.txt
```

Make sure Ollama is running and the models are available:

```bash
ollama serve
ollama pull llama3.2:1b
ollama pull mistral
ollama pull phi3
```

## Usage

```bash
python main.py
```

## Project structure

```
src/
  metrics.py     # BenchmarkResult dataclass + compute_metrics()
  runner.py      # Async Ollama streaming + timing measurement
  evaluator.py   # ROUGE-L + LLM-as-judge scoring
  reporter.py    # JSON and HTML report generation
  agent.py       # Automatic model recommendation
api/
  main.py        # FastAPI endpoints
benchmarks/
  questions.json # Test prompt dataset
tests/
main.py          # CLI entrypoint
```

## Supported models

Any model available in your local Ollama instance. Default: `llama3.2:1b`, `mistral`, `phi3`.
