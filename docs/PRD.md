# ModelFit.0 — Product Requirements Document

| | |
|---|---|
| **Status** | Living document — reflects the shipped v0.1.0 implementation |
| **Owner** | Mathicode-now |
| **Related docs** | [`SOFTWARE_DESIGN.md`](./SOFTWARE_DESIGN.md) · [`EVALUATION_GUIDE.md`](./EVALUATION_GUIDE.md) · [`Tech_Stack`](./Tech_Stack) |

## 1. Problem Statement

Teams choosing between small, locally-hostable open-weight LLMs (7B–9B
parameters) lack a lightweight, reproducible way to compare them across
the range of tasks a real product actually needs — code generation,
summarization, RAG question-answering, classification, reasoning,
creative writing, and safety-critical behaviors like adversarial
resistance, PII handling, and bias. Cloud eval platforms assume API
access to hosted models and per-token billing; general-purpose
benchmarks (MMLU, HellaSwag, etc.) don't map to task-specific product
requirements or run against a local Ollama deployment. There is no
existing, easy way to answer: *"of the 4 small models I can actually
run on my hardware, which one should I use for this feature — and
what am I trading off to get it?"*

## 2. Goals

1. **Run the same, fixed set of tasks against multiple local models**
   and produce comparable, storable results — no manual copy-pasting
   of prompts into different chat UIs.
2. **Score each response with a method appropriate to what
   "correctness" means for that task type**, rather than forcing one
   metric (e.g. exact match) onto tasks where it doesn't apply (e.g.
   creative writing).
3. **Make results explorable without a database client** — a
   dashboard a non-SQL user can open and immediately understand which
   model is strongest where, and why.
4. **Make trade-offs (accuracy vs. latency vs. RAM) explicit and
   configurable**, not just a raw accuracy leaderboard, since the
   target hardware is local/resource-constrained.
5. **Be honest about what's implemented vs. aspirational** — every
   scoring method documented should describe what the code actually
   does, not what a "gold standard" eval framework would ideally do
   (see [`EVALUATION_GUIDE.md`](./EVALUATION_GUIDE.md) for corrections
   already made against this principle).

### Non-goals

- **Not a general-purpose LLM benchmark suite.** It doesn't aim to
  reproduce MMLU/HELM/etc. or support arbitrary third-party task sets
  out of the box — its 10 fields and 5 tasks each are hand-authored
  and specific to this project's needs.
- **Not a hosted/multi-tenant service.** No auth, no cloud model
  providers, no concurrent-user support — it's a local CLI + local
  Streamlit dashboard against a local Ollama instance and a local
  SQLite file.
- **Not a training or fine-tuning tool.** It only evaluates models
  that already exist and are pulled into Ollama.
- **Not a real-time/production monitoring tool.** Runs are triggered
  manually or via the CLI; there's no scheduling, alerting, or
  regression-detection-in-CI built in today.

## 3. Target Users

- **Primary: a developer or technical evaluator** deciding which
  locally-hostable small model to standardize on for a given feature
  or product, who wants numbers instead of vibes, but doesn't have
  budget/infrastructure for a large-scale eval platform.
- **Secondary: the same person, later, as a "beginner" reader** of
  their own results — someone who wants to double-check *what a score
  actually measures* before trusting it (this is why
  `EVALUATION_GUIDE.md` exists as a first-class deliverable, not just
  code comments).

## 4. Functional Requirements

### 4.1 Task Fields & Prompts

- FR-1: The system supports 10 task fields — `code_generation`,
  `classification`, `reasoning`, `summarization`, `rag_qa`,
  `creative_writing`, `adversarial` (Security), `pii`, `bias`,
  `regression` — each with 5 hand-authored tasks in
  `tasks/<field>/prompts.yaml`.
- FR-2: Each field declares its own `scoring:` method (and, where
  needed, a `scoring_mode:` modifier) so field authors control how
  their tasks are graded without touching runner code.
- FR-3: New fields can be added by dropping a new
  `tasks/<field>/prompts.yaml` directory — no code change required to
  run it (see Extension Points in `SOFTWARE_DESIGN.md`).

### 4.2 Model Execution

- FR-4: The system evaluates any model available in a local Ollama
  instance, driven by `config/models.yaml`; the current reference set
  is `qwen2.5-coder:7b`, `mistral:7b`, `llama3:8b`, `gemma2:9b`.
- FR-5: A connectivity/model-availability check (`scripts/preflight.py`)
  must be runnable before an evaluation, so failures are caught before
  burning time on a multi-field run.
- FR-6: Generation requests must have configurable, per-field timeouts
  (`config/timeouts.yaml`) and retry with exponential backoff on
  connection errors (not on timeouts).
- FR-7: A timeout, generation error, or scoring error for one
  model/task must not abort the run — it is recorded as a `0.0`-score
  result with a diagnostic response string, and the run continues.

### 4.3 Scoring

- FR-8: Each response is scored 0.0–1.0 by the method declared for its
  field: exact match, substring (`contains`/`not_contains`), code
  execution (3-valued), ROUGE-L, BERTScore, or the local RAG composite
  heuristic (see `EVALUATION_GUIDE.md` for the full mechanics of each).
- FR-9: Scoring must degrade gracefully — a scoring exception is
  caught and recorded as `0.0`, not allowed to crash the run.

### 4.4 Persistence

- FR-10: Every result (`model_id`, `task_field`, `task_id`, `prompt`,
  `response`, `latency_ms`, `token_count`, `score`) is persisted to a
  local SQLite database (`results.db`), grouped under a named `run`.
- FR-11: Run names must be short and meaningful by default
  (`<field>-YYYYMMDD-HHMM`, auto-generated per field), with an
  optional user-supplied override.
- FR-12: The database must support querying results by a single run,
  by field across all runs, or by the latest run per field (for a
  cross-field leaderboard) — without requiring the caller to write raw
  SQL (`DashboardDataLoader`).

### 4.5 CLI

- FR-13: `scripts/run_eval.py` supports running all fields, a single
  `--field`, a subset of `--models`, a custom `--db` path, and
  `--list-fields`/`--list-models` introspection.
- FR-14: A human-readable summary table prints after each field
  completes (and on interruption/error, for whatever partial results
  exist), without requiring the dashboard to be open.

### 4.6 Dashboard

- FR-15: A Streamlit dashboard reads the same `results.db` and
  presents, across tabs: an **Overview** (leaderboard, strengths/
  weaknesses, cross-field heatmap, coverage), **Browse by Task Style**
  (field-first browsing with no run selection required), **Field
  Detail** (per-run, per-field drill-down), **Model Comparison**
  (radar charts, token efficiency), **Gap Analysis** (weighted
  ranking + recommendations), and **About & README** (renders the
  project's own docs in-app).
- FR-16: The dashboard must clearly communicate that scores are not
  comparable across fields — via a documented golden-case matrix
  surfaced in both the README and the UI.
- FR-17: A configurable weighting scheme
  (`config/scoring_weights.yaml`, applied by `GapAnalyzer`) combines
  per-field accuracy, latency, and RAM usage into a single ranked
  score per model, so "best overall" reflects the project's stated
  priorities rather than raw accuracy alone.

## 5. Non-Functional Requirements

- **NFR-1 (Local-only / privacy):** No prompts, responses, or scores
  leave the local machine — no cloud model calls, no telemetry. This
  matters specifically for the `pii` field, whose entire premise is
  testing whether sensitive data leaks; the harness itself must not
  introduce a new leakage path.
- **NFR-2 (Reproducibility):** Given the same models and task files, a
  re-run should be comparable to a prior run — run metadata records
  the field, models, and scoring type used, so results can be traced
  back to the configuration that produced them.
- **NFR-3 (Resilience):** A single flaky model response, timeout, or
  Ollama hiccup should not lose an entire evaluation run's results.
- **NFR-4 (Low setup cost):** A new contributor should be able to go
  from clone to first result with `pip install`, `setup_ollama.sh`,
  `preflight.py`, and one `run_eval.py` invocation — no cloud
  credentials, no external services beyond Ollama.
- **NFR-5 (Extensibility):** Adding a field, a model, or a scoring
  method should be a config/file addition plus a small, localized code
  change — not a rearchitecture (see `SOFTWARE_DESIGN.md` §9).
- **NFR-6 (Documentation accuracy):** Docs describing scoring behavior
  must match the implementation. This is treated as a requirement, not
  a nice-to-have, after the golden-case matrix was found to describe
  aspirational scoring (e.g. "Ragas faithfulness," "judge-model
  rubric") that didn't match the actual heuristics in `src/scoring/`.

## 6. Success Metrics

Since this is an internal evaluation tool rather than a customer-facing
product, success is measured operationally:

| Metric | Target |
|---|---|
| Task field coverage | 10/10 fields runnable end-to-end against all 4 reference models |
| Unit test coverage | All core modules (`runner`, `results_store`, `scoring/*`, `dashboard/*`) covered; suite passes on every change (126 tests as of this writing) |
| Time-to-first-result | Preflight → first field result in a single sitting, no cloud setup |
| Run failure containment | A single model/task failure never zeroes out an entire run's results |
| Docs/code parity | Every field's documented scoring method matches its `scoring:` value and the function `get_scorer()` actually dispatches to |

## 7. Known Limitations (tracked, not blocking)

- `rag_qa` scoring is a local Jaccard/token-F1 heuristic, not the
  `ragas` library (installed as a dependency but unused) or an
  LLM-as-judge — faithfulness checks can be fooled by paraphrased
  hallucinations that reuse context vocabulary.
- `creative_writing` scores similarity to **one fixed reference
  story** via BERTScore, not a rubric judged by a separate model — a
  genuinely good but stylistically different response can score lower
  than a mediocre but similar one.
- `reasoning` scoring checks only whether the final answer substring
  appears in the response; it does not validate the intermediate
  reasoning steps despite the field's name.
- `code_generation` correctness is a single stdout comparison, not a
  hidden multi-case unit test suite — a function correct only on the
  example shown isn't distinguished from one that's fully correct.
- 5 tasks per field is enough to spot large gaps between models but
  too small a sample to treat any single field's score as
  statistically robust.

## 8. Open Questions / Future Work

- Should `rag_qa` and `creative_writing` be upgraded to use an actual
  LLM-as-judge (local, via Ollama) now that the harness already has a
  model-calling path available?
- Should `code_generation` move to a hidden multi-case test harness
  (e.g. `pytest`-based) instead of single stdout comparison?
- Is there a need for CI-based regression detection (re-run
  `regression`'s canary tasks on every code change and fail if scores
  drop), or does this stay a manually-triggered tool?
- Should task counts per field grow beyond 5 for statistical
  confidence, and if so, does that change the per-field timeout
  budget in `config/timeouts.yaml`?
