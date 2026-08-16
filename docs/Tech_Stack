ModelFit.0 tech stack:

Language & packaging

Python 3.11+, packaged via pyproject.toml / setuptools
Core runtime

requests — HTTP client for talking to Ollama's REST API (/api/tags, /api/generate)
pyyaml — all config (config/*.yaml) and task definitions (tasks/<field>/prompts.yaml)
sqlite3 (stdlib) — persistence, no ORM, raw parameterized SQL via ResultsStore
LLM serving

Ollama running locally (localhost:11434), serving 4 reference models: qwen2.5-coder:7b, mistral:7b, llama3:8b, gemma2:9b
Scoring (scoring extra)

rouge-score — ROUGE-L (summarization)
bert-score — semantic similarity via microsoft/deberta-xlarge-mnli (creative writing)
ragas — installed but currently unused; rag_qa uses a hand-rolled Jaccard/token-F1 heuristic instead (documented as a known gap in docs/EVALUATION_GUIDE.md)
numpy — supporting numerics
Plus stdlib-only scorers: exact match, substring match, and a subprocess-based code-execution scorer
Dashboard (dashboard extra)

streamlit — the UI (tabbed layout, custom CSS)
plotly — all charts (heatmaps, radar, waterfall, box plots, bar charts)
pandas — DataFrame layer between SQLite and the UI/charts
Dev tooling (dev extra)

pytest + pytest-cov — 126 unit tests
ruff — linting
mypy — type checking
Docs

Markdown with Mermaid diagrams (docs/SOFTWARE_DESIGN.md, docs/EVALUATION_GUIDE.md)
No web framework, no cloud LLM APIs, no database server — everything runs locally against Ollama with SQLite as the only persistence layer.
