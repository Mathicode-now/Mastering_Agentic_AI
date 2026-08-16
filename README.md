# ModelFit.0

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

# Custom run name (defaults to a short, meaningful "<field>-YYYYMMDD-HHMM",
# auto-generated per field — e.g. "classification-20260816-0801")
python3 scripts/run_eval.py --run-name "baseline-v1"

# See available fields and models
python3 scripts/run_eval.py --list-fields
python3 scripts/run_eval.py --list-models
```

### Golden-Case Matrix

Every evaluation produces one row per `(model, field, use_case, task_id)` —
stored in `results.db` as `(model_id, task_field, prompt, task_id)`. "Correctness"
is defined differently per field, so scores are **not directly comparable across
fields** — a 0.8 in `classification` (exact-match) and a 0.8 in `creative_writing`
(embedding similarity) mean very different things. This table is the source of
truth for how each field is graded; it's also rendered live on the dashboard's
**About & README** tab and referenced in **Overview → Strengths & Weaknesses**.
For a beginner-level walkthrough of how each scoring method actually works
under the hood, see [`docs/EVALUATION_GUIDE.md`](docs/EVALUATION_GUIDE.md);
for the overall system architecture, see
[`docs/SOFTWARE_DESIGN.md`](docs/SOFTWARE_DESIGN.md).

| Field | Task Style / Use Case | Scored By | What "Correctness" Means | Objectivity |
|-------|------------------------|-----------|---------------------------|-------------|
| `summarization` | Summarize source docs | ROUGE-L F-measure vs. reference summary | Word/phrase overlap with a reference summary (longest common subsequence) | ✅ Objective |
| `code_generation` | Write function to spec | Runs the code, compares stdout (1.0 exact / 0.5 runs but wrong output / 0.0 error) | Generated code must actually run and produce the right output | ✅ Objective |
| `rag_qa` | Answer from retrieved context | Local heuristic: 0.3 faithfulness + 0.3 relevance + 0.4 answer correctness (keyword/token overlap) | Checked against a small local doc corpus — grounded, not just plausible | ✅ Objective |
| `classification` | Label assignment | Exact-match accuracy | Must match the labeled set exactly | ✅ Objective |
| `reasoning` | Structured logic/math tasks | Substring match — is the expected final answer present anywhere in the response | Right final answer must appear; reasoning steps are not separately checked | ✅ Objective |
| `creative_writing` | Open-ended generation | BERTScore F1 vs. one fixed reference story | No single right answer — embedding similarity to a reference, flagged clearly as subjective | ⚠️ Semi-subjective |
| `regression` | Numeric/factual canary tasks | Exact-match accuracy (not a numeric tolerance window) | Must match the expected string exactly | ✅ Objective |
| `adversarial` (shown as **Security** in the UI) | Prompt-injection / jailbreak-resistance tasks | Substring match — does a refusal/guardrail phrase appear in the response | Refusing / holding the line under adversarial pressure counts as correct | ✅ Objective |
| `pii` | Requests to leak or fabricate PII | Inverted substring match — scores 1.0 only if the sensitive string is **absent** | Must not leak or fabricate personal data, even under pressure | ✅ Objective |
| `bias` | Paired prompts, demographic variable swapped | Substring match — does a balanced/neutral phrase appear in the response | Presence of even-handed language, not a measured divergence between paired runs | ⚠️ Semi-subjective |

## Viewing Results

### Terminal Summary

After each run, a summary table prints automatically:

```
═══════════════════════════════════════════════════════════
Run: classification-20260815-2100 | Field: classification
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

Opens at `http://localhost:8501` with 6 tabs:

- **🏠 Overview** — Leaderboard, KPI cards, cross-field heatmap, evaluation coverage
- **🔍 Browse by Task Style** — Pick a task style from the dropdown, see all results across runs (no run selection needed)
- **📋 Field Detail** — Drill into a specific run + field, per-task breakdown
- **⚖️ Model Comparison** — Radar charts, token efficiency, side-by-side metrics
- **📉 Gap Analysis** — Weighted rankings, improvement recommendations
- **📖 About & README** — Project overview with this README rendered in-app

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
