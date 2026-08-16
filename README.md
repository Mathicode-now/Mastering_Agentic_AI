# Model Eval Framework

Local LLM evaluation framework for comparing small models (7B–9B) across 10 task types using Ollama.

## Setup

```bash
# 1. Install with all extras
pip3 install -e ".[dev,dashboard,scoring]"

# 2. Install Ollama and pull models (takes ~18 GB)
bash scripts/setup_ollama.sh

# 3. Verify everything is ready
python3 scripts/preflight.py
```

Preflight should show all 4 models with green ✓:
- `qwen2.5-coder:7b` — Code-specialized baseline
- `mistral:7b` — General-purpose balanced baseline
- `llama3:8b` — General-purpose widely-adopted reference
- `gemma2:9b` — Larger model RAM/latency tradeoff

## Running Evaluations

```bash
# Run all 10 fields against all 4 models
python3 scripts/run_eval.py

# Run a single field
python3 scripts/run_eval.py --field classification

# Run specific models only
python3 scripts/run_eval.py --field code_generation --models qwen2.5-coder:7b mistral:7b

# Custom run name (defaults to eval_YYYYMMDD_HHMMSS)
python3 scripts/run_eval.py --run-name "baseline-v1"

# See available fields and models
python3 scripts/run_eval.py --list-fields
python3 scripts/run_eval.py --list-models
```

### Evaluation Fields

| Field | Tasks | Scoring | What it tests |
|-------|-------|---------|---------------|
| `Code_generation` | 5 | Code execution | Runs generated code, compares output |
| `Classification` | 5 | Exact match | Sentiment analysis accuracy |
| `Reasoning` | 5 | Contains | Logic, math, deduction |
| `Summarization` | 5 | ROUGE-L | Summary quality vs reference |
| `Rag_qa` | 5 | RAG composite | Faithfulness + relevance + correctness |
| `Creative_writing` | 5 | BERT-Score | Semantic quality vs reference |
| `Adversarial` | 5 | Contains | Safety — must refuse harmful requests |
| `Pii` | 5 | Not-contains | Must NOT leak given PII |
| `Bias` | 5 | Contains | Balanced, non-stereotyping responses |
| `Regression` | 5 | Exact match | Canary tests for consistency |

## Viewing Results

### Terminal Summary

After each run, a summary table prints automatically:

```
═══════════════════════════════════════════════════════════
Run: eval_20260815_210000 | Field: classification
═══════════════════════════════════════════════════════════
Model               Avg Score   Avg Latency   Tasks
───────────────────────────────────────────────────────────
Qwen2.5-Coder       0.80        1234ms        5
Mistral             0.60        987ms         5
═══════════════════════════════════════════════════════════
```

### Dashboard (Streamlit)

```bash
streamlit run src/dashboard/app.py
```

Opens at `http://localhost:8501` with 5 pages:

- **Overview** — Select a run, see heatmap + model comparison bars
- **Browse by Field** — Pick a field from dropdown, see all results across runs (no run selection needed)
- **Field Detail** — Drill into a specific run + field, per-task breakdown
- **Model Comparison** — Radar charts, token efficiency, side-by-side metrics
- **Gap Analysis** — Weighted rankings, improvement recommendations

### Direct DB Access

Results are stored in `results.db` (SQLite). Query directly:

```bash
sqlite3 results.db "SELECT model_id, task_field, AVG(score) FROM results GROUP BY model_id, task_field"
```

## Project Structure

```
config/
  models.yaml           — Model definitions (ID, RAM, role)
  timeouts.yaml         — Per-field generation timeouts
  scoring_weights.yaml  — Weights for gap analysis
tasks/
  <field>/prompts.yaml  — Evaluation prompts per field
src/
  model_provider.py     — Ollama HTTP client
  results_store.py      — SQLite persistence
  runner.py             — Evaluation orchestrator
  scoring/              — Scoring modules (exact, ROUGE, BERT, RAG)
  dashboard/            — Streamlit app + charts
scripts/
  setup_ollama.sh       — One-command Ollama + model setup
  preflight.py          — Connectivity checker
  run_eval.py           — CLI evaluation runner
tests/
  unit/                 — 103 unit tests
```

## Development

```bash
# Run tests
pytest

# Lint
ruff check src/ tests/ scripts/

# Type check
mypy src/
```
