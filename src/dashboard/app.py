"""Model Evaluation Dashboard — Streamlit main application.

Provides a multi-page dashboard for analyzing model evaluation results
with sidebar navigation across Overview, Field Detail, Model Comparison,
and Gap Analysis pages.

Run from project root:
    streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports resolve
# when running via `streamlit run src/dashboard/app.py` from project root.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import streamlit as st

from src.dashboard.components.charts import (
    create_field_breakdown_radar,
    create_gap_waterfall,
    create_heatmap,
    create_latency_vs_accuracy_scatter,
    create_model_comparison_bar,
    create_score_distribution_box,
)
from src.dashboard.data_loader import DashboardDataLoader
from src.dashboard.gap_analysis import GapAnalyzer

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ModelFit.0",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Modern styling — gradient hero, card metrics, pill-style tabs
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* Hero banner */
    .mf-hero {
        background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
        padding: 1.75rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.25);
    }
    .mf-hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .mf-hero-subtitle {
        font-size: 1.02rem;
        color: rgba(255, 255, 255, 0.88);
        margin-top: 0.35rem;
    }

    /* Sidebar branding */
    .mf-sidebar-title {
        font-size: 1.35rem;
        font-weight: 800;
        background: linear-gradient(120deg, #4f46e5, #db2777);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .mf-sidebar-caption {
        color: #9ca3af;
        font-size: 0.85rem;
        margin-top: -0.4rem;
    }

    /* Tabs — pill style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.25);
    }
    .stTabs [data-baseweb="tab"] {
        height: 46px;
        border-radius: 10px 10px 0 0;
        padding: 0 18px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(120deg, rgba(79, 70, 229, 0.18), rgba(219, 39, 119, 0.18));
        border-bottom: 3px solid #7c3aed;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: rgba(127, 127, 127, 0.07);
        border: 1px solid rgba(127, 127, 127, 0.18);
        border-radius: 12px;
        padding: 0.9rem 1rem 0.6rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="mf-hero">
        <p class="mf-hero-title">🎯 ModelFit.0</p>
        <p class="mf-hero-subtitle">
            Local LLM evaluation framework — compare small models (7B–9B) across
            10 task types with Ollama.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — branding and settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<p class="mf-sidebar-title">🎯 ModelFit.0</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="mf-sidebar-caption">Model evaluation dashboard</p>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**⚙️ Settings**")
    db_path = st.text_input("Database path", value="results.db")
    st.divider()
    st.caption("Built on Streamlit + Plotly + SQLite")
    st.caption("Navigate using the tabs at the top of the page →")


# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_runs(_db_path: str) -> pd.DataFrame:
    """Load available runs from the database."""
    loader = DashboardDataLoader(db_path=_db_path)
    return loader.get_available_runs()


@st.cache_data(show_spinner=False)
def load_all_results(_db_path: str, run_id: int | None) -> pd.DataFrame:
    """Load all results for a run."""
    loader = DashboardDataLoader(db_path=_db_path)
    return loader.get_all_results(run_id=run_id)


@st.cache_data(show_spinner=False)
def load_model_field_summary(_db_path: str, run_id: int | None) -> pd.DataFrame:
    """Load model×field score pivot."""
    loader = DashboardDataLoader(db_path=_db_path)
    return loader.get_model_field_summary(run_id=run_id)


@st.cache_data(show_spinner=False)
def load_latency_summary(_db_path: str, run_id: int | None) -> pd.DataFrame:
    """Load model×field latency pivot."""
    loader = DashboardDataLoader(db_path=_db_path)
    return loader.get_latency_summary(run_id=run_id)


@st.cache_data(show_spinner=False)
def load_model_comparison(_db_path: str, run_id: int) -> pd.DataFrame:
    """Load per-model comparison summary."""
    loader = DashboardDataLoader(db_path=_db_path)
    return loader.get_model_comparison(run_id=run_id)


@st.cache_data(show_spinner=False)
def load_field_detail(_db_path: str, run_id: int, field: str) -> pd.DataFrame:
    """Load results for a specific field."""
    loader = DashboardDataLoader(db_path=_db_path)
    return loader.get_field_detail(run_id=run_id, field=field)


@st.cache_data(show_spinner=False)
def load_all_fields(_db_path: str) -> list[str]:
    """Load all available fields from the database."""
    loader = DashboardDataLoader(db_path=_db_path)
    return loader.get_all_fields()


@st.cache_data(show_spinner=False)
def load_field_results_all_runs(_db_path: str, field: str) -> pd.DataFrame:
    """Load results for a field across all runs."""
    loader = DashboardDataLoader(db_path=_db_path)
    return loader.get_field_results_all_runs(field=field)


@st.cache_data(show_spinner=False)
def load_field_model_summary(_db_path: str, field: str) -> pd.DataFrame:
    """Load per-model summary for a field across all runs."""
    loader = DashboardDataLoader(db_path=_db_path)
    return loader.get_field_model_summary(field=field)


# ---------------------------------------------------------------------------
# Helper: empty state guard
# ---------------------------------------------------------------------------


def _show_empty_state() -> bool:
    """Show empty state message and return True if no data available."""
    runs = load_runs(db_path)
    if runs.empty:
        st.info("No results yet. Run an evaluation first.")
        return True
    return False


def _get_run_selector(key: str) -> int | None:
    """Show run selector dropdown and return selected run_id.

    Args:
        key: Unique widget key. All tabs render on every script run, so
            each call site needs a distinct key to avoid ID collisions.
    """
    runs = load_runs(db_path)
    if runs.empty:
        return None
    options = {f"{row['name']} (Run #{row['id']})": row["id"] for _, row in runs.iterrows()}
    selected_label = st.selectbox("Select Run", list(options.keys()), key=key)
    return options[selected_label] if selected_label else None


# ---------------------------------------------------------------------------
# PAGE 1: Overview
# ---------------------------------------------------------------------------


def page_overview() -> None:
    """Render the Overview page — Leaderboard + Coverage, latest run per field."""
    st.header("Overview")

    if _show_empty_state():
        return

    loader = DashboardDataLoader(db_path=db_path)

    # --- Leaderboard ---
    leaderboard = loader.get_latest_per_field_leaderboard()
    if leaderboard.empty:
        st.info("No results yet.")
        return

    # Top KPI cards
    best_model = leaderboard.iloc[0]
    fastest_model = leaderboard.loc[leaderboard["avg_latency_ms"].idxmin()]
    coverage = loader.get_field_coverage()
    all_fields = list(_FIELD_METADATA.keys()) if "_FIELD_METADATA" in dir() else []
    fields_evaluated = int(leaderboard.iloc[0]["fields_evaluated"])
    total_fields = len(all_fields) if all_fields else 10

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏆 Best Model", best_model["model_id"])
        col1.caption(f"Score: {best_model['avg_score']:.3f}")
        col2.metric("⚡ Fastest", fastest_model["model_id"])
        col2.caption(f"{fastest_model['avg_latency_ms']:.0f} ms avg")
        col3.metric("📊 Fields Evaluated", f"{fields_evaluated} / {total_fields}")
        col4.metric("🔄 Total Tasks", int(leaderboard["total_tasks"].sum()))

    st.divider()

    # --- Model Leaderboard ---
    st.subheader("🏆 Model Leaderboard")
    st.caption("Based on latest run per field")

    with st.container(border=True):
        for i, (_, row) in enumerate(leaderboard.iterrows()):
            rank = i + 1
            score = row["avg_score"]
            latency = row["avg_latency_ms"]

            lcol, bcol, scol = st.columns([3, 6, 2])
            lcol.markdown(f"**#{rank} {row['model_id']}**")
            bcol.progress(min(score, 1.0), text=f"{score:.3f}")
            scol.caption(f"{latency:.0f} ms")

    st.divider()

    # --- Strengths & Weaknesses ---
    per_field_scores = loader.get_latest_per_field_scores()
    if not per_field_scores.empty:
        st.subheader("💪 Strengths & ⚠️ Weaknesses")
        st.caption(
            "\"Correctness\" is scored differently per use case — see the "
            "golden-case matrix on the About & README tab."
        )

        for model_id in leaderboard["model_id"].tolist():
            model_data = per_field_scores[per_field_scores["model_id"] == model_id]
            if model_data.empty:
                continue
            best_field = model_data.loc[model_data["avg_score"].idxmax()]
            worst_field = model_data.loc[model_data["avg_score"].idxmin()]
            best_key = best_field["task_field"]
            worst_key = worst_field["task_field"]
            best_meta = _FIELD_METADATA.get(best_key, {})
            worst_meta = _FIELD_METADATA.get(worst_key, {})
            best_use_case = best_meta.get("task_style", _field_label(best_key))
            worst_use_case = worst_meta.get("task_style", _field_label(worst_key))

            st.markdown(f"**{model_id}**")
            scol, wcol = st.columns(2)
            with scol:
                st.success(
                    f"💪 **{best_use_case}** — {best_field['avg_score']:.2f}"
                )
                st.caption(f"{_field_label(best_key)} · {best_meta.get('scoring', '')}")
            with wcol:
                st.warning(
                    f"⚠️ **{worst_use_case}** — {worst_field['avg_score']:.2f}"
                )
                st.caption(f"{_field_label(worst_key)} · {worst_meta.get('scoring', '')}")

        st.divider()

    # --- Cross-field heatmap ---
    if not per_field_scores.empty:
        st.subheader("Cross-Field Heatmap")
        # Pivot to model × field matrix
        pivot = per_field_scores.pivot(
            index="model_id", columns="task_field", values="avg_score"
        )
        fig = create_heatmap(pivot, title="Model × Field Scores (latest run per field)")
        st.plotly_chart(fig, width="stretch")

    # --- Coverage table ---
    st.subheader("📋 Evaluation Coverage")

    all_field_names = list(_FIELD_METADATA.keys())
    evaluated_fields = set(coverage["task_field"].tolist()) if not coverage.empty else set()

    coverage_rows = []
    for field_name in all_field_names:
        if field_name in evaluated_fields:
            row = coverage[coverage["task_field"] == field_name].iloc[0]
            coverage_rows.append({
                "Field": _field_label(field_name),
                "Status": "✅ Evaluated",
                "Best Model": row["best_model"],
                "Best Score": f"{row['best_score']:.3f}",
            })
        else:
            coverage_rows.append({
                "Field": _field_label(field_name),
                "Status": "⬜ Not yet",
                "Best Model": "—",
                "Best Score": "—",
            })

    st.table(pd.DataFrame(coverage_rows))


# ---------------------------------------------------------------------------
# PAGE 2: Field Detail
# ---------------------------------------------------------------------------


def page_field_detail() -> None:
    """Render the Field Detail page."""
    st.header("Field Detail")

    if _show_empty_state():
        return

    run_id = _get_run_selector(key="field_detail_run")
    if run_id is None:
        return

    # Get available fields from results
    results = load_all_results(db_path, run_id)
    if results.empty:
        st.info("No results for this run.")
        return

    fields = sorted(results["task_field"].unique().tolist())
    selected_field = st.selectbox("Select Field", fields, key="field_detail_field")

    if not selected_field:
        return

    # Per-task results table
    field_data = load_field_detail(db_path, run_id, selected_field)
    if field_data.empty:
        st.info(f"No results for field '{selected_field}'.")
        return

    st.subheader("Per-Task Results")
    display_cols = ["model_id", "score", "latency_ms", "token_count"]
    st.dataframe(
        field_data[display_cols].sort_values(["model_id", "score"], ascending=[True, False]),
        width="stretch",
    )

    # Score distribution box plot
    st.subheader("Score Distribution")
    fig = create_score_distribution_box(field_data)
    st.plotly_chart(fig, width="stretch")

    # Latency vs accuracy scatter
    st.subheader("Latency vs Accuracy")
    fig = create_latency_vs_accuracy_scatter(field_data)
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# PAGE 3: Model Comparison
# ---------------------------------------------------------------------------


def page_model_comparison() -> None:
    """Render the Model Comparison page."""
    st.header("Model Comparison")

    if _show_empty_state():
        return

    run_id = _get_run_selector(key="model_comparison_run")
    if run_id is None:
        return

    # Get available models
    comparison = load_model_comparison(db_path, run_id)
    if comparison.empty:
        st.info("No model data for this run.")
        return

    all_models = comparison["model_id"].tolist()
    selected_models = st.multiselect(
        "Select Models", all_models, default=all_models, key="model_comparison_models"
    )

    if not selected_models:
        st.warning("Please select at least one model.")
        return

    scores_df = load_model_field_summary(db_path, run_id)
    filtered_comparison = comparison[comparison["model_id"].isin(selected_models)]

    # Radar charts for selected models
    st.subheader("Field Breakdown (Radar)")
    if not scores_df.empty:
        radar_cols = st.columns(min(len(selected_models), 3))
        for i, model_id in enumerate(selected_models):
            if model_id in scores_df.index:
                with radar_cols[i % len(radar_cols)]:
                    fig = create_field_breakdown_radar(scores_df, model_id)
                    st.plotly_chart(fig, width="stretch")

    # Side-by-side latency and score bars
    st.subheader("Score & Latency Comparison")
    if not filtered_comparison.empty:
        fig = create_model_comparison_bar(filtered_comparison)
        st.plotly_chart(fig, width="stretch")

    # Token efficiency metrics
    st.subheader("Token Efficiency")
    if not filtered_comparison.empty:
        token_df = filtered_comparison[["model_id", "total_tokens", "total_tasks", "avg_score"]].copy()
        token_df["tokens_per_task"] = (token_df["total_tokens"] / token_df["total_tasks"]).round(0)
        token_df["score_per_1k_tokens"] = (
            token_df["avg_score"] / (token_df["total_tokens"] / 1000)
        ).round(6)
        st.dataframe(
            token_df[["model_id", "tokens_per_task", "score_per_1k_tokens", "avg_score"]],
            width="stretch",
        )


# ---------------------------------------------------------------------------
# PAGE 4: Gap Analysis
# ---------------------------------------------------------------------------


def page_gap_analysis() -> None:
    """Render the Gap Analysis page."""
    st.header("Gap Analysis")

    if _show_empty_state():
        return

    run_id = _get_run_selector(key="gap_analysis_run")
    if run_id is None:
        return

    scores_df = load_model_field_summary(db_path, run_id)
    latency_df = load_latency_summary(db_path, run_id)

    if scores_df.empty or latency_df.empty:
        st.info("Insufficient data for gap analysis.")
        return

    # Load weights config
    weights_path = Path(_PROJECT_ROOT) / "config" / "scoring_weights.yaml"
    if not weights_path.exists():
        st.warning(
            f"Scoring weights config not found at {weights_path}. "
            "Using default weights."
        )
        # Create analyzer with fallback — it will error if file missing,
        # so provide a graceful message
        st.info("Please create config/scoring_weights.yaml to enable gap analysis.")
        return

    analyzer = GapAnalyzer(weights_path=str(weights_path))

    # Build RAM config from models.yaml
    models_path = Path(_PROJECT_ROOT) / "config" / "models.yaml"
    ram_config: dict[str, float] | None = None
    if models_path.exists():
        import yaml

        with open(models_path, "r") as f:
            models_cfg = yaml.safe_load(f)
        ram_config = {m["id"]: m.get("ram_gb", 0) for m in models_cfg.get("models", [])}

    # Compute gap matrix
    gap_matrix = analyzer.compute_gap_matrix(scores_df, latency_df, ram_config)

    # Weighted score ranking table
    st.subheader("Weighted Score Ranking")
    display_gap = gap_matrix[["model_id", "weighted_score", "rank", "gap_to_best"]].copy()
    st.dataframe(display_gap, width="stretch")

    # Gap waterfall chart
    st.subheader("Gap Waterfall")
    fig = create_gap_waterfall(gap_matrix)
    st.plotly_chart(fig, width="stretch")

    # Field-level gap table
    st.subheader("Field-Level Gaps")
    field_gaps = analyzer.get_field_gaps(scores_df)
    st.dataframe(field_gaps, width="stretch")

    # Improvement recommendations
    st.subheader("Improvement Recommendations")
    recommendations = analyzer.get_improvement_recommendations(gap_matrix)
    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        st.dataframe(rec_df, width="stretch")
    else:
        st.success("All models are performing at the same level — no gaps detected.")


# ---------------------------------------------------------------------------
# PAGE: Browse by Task Style (task-style-first, no run selection required)
# ---------------------------------------------------------------------------

# Golden-case matrix: each results row is (model, field, use_case, task_id).
# "Correctness" is defined differently per field, so scoring and grading
# criteria are tracked here rather than assumed to be comparable across
# fields. `label` overrides the display name shown in the UI (the field key
# itself — e.g. "adversarial" — stays stable since it's what's stored in
# results.db and referenced by config/scoring_weights.yaml).
_FIELD_METADATA: dict[str, dict[str, str]] = {
    "summarization": {
        "label": "Summarization",
        "task_style": "Summarize source docs",
        "scoring": "ROUGE/BERTScore + Ragas faithfulness",
        "correctness": "Penalizes facts hallucinated that aren't in the source",
        "objective": "✅ Objective",
    },
    "code_generation": {
        "label": "Code Generation",
        "task_style": "Write function to spec",
        "scoring": "Pass/fail against hidden unit tests",
        "correctness": "Generated code must actually run and produce the right output",
        "objective": "✅ Objective",
    },
    "rag_qa": {
        "label": "RAG QA",
        "task_style": "Answer from retrieved context",
        "scoring": "Ragas faithfulness + answer correctness",
        "correctness": "Checked against a small local doc corpus — grounded, not just plausible",
        "objective": "✅ Objective",
    },
    "classification": {
        "label": "Classification",
        "task_style": "Label assignment",
        "scoring": "Exact-match accuracy",
        "correctness": "Must match the labeled set exactly",
        "objective": "✅ Objective",
    },
    "reasoning": {
        "label": "Reasoning",
        "task_style": "Structured logic/math tasks",
        "scoring": "Answer-match + step validity",
        "correctness": "Right final answer AND valid reasoning steps",
        "objective": "✅ Objective",
    },
    "creative_writing": {
        "label": "Creative Writing",
        "task_style": "Open-ended generation",
        "scoring": "Lightweight rubric (coherence, instruction-following) scored by a judge model",
        "correctness": "No single right answer — harder to auto-score, flagged clearly as subjective",
        "objective": "⚠️ Semi-subjective",
    },
    "regression": {
        "label": "Regression",
        "task_style": 'Numeric output tasks (e.g. "estimate X given Y")',
        "scoring": "Error margin vs. ground truth",
        "correctness": "Closeness to the true numeric value, not exact match",
        "objective": "✅ Objective",
    },
    "adversarial": {
        "label": "Security",
        "task_style": "Prompt-injection / jailbreak-resistance tasks",
        "scoring": "Pass/fail — did the model hold its guardrails",
        "correctness": "Refusing / holding the line under adversarial pressure counts as correct",
        "objective": "✅ Objective",
    },
    "pii": {
        "label": "PII",
        "task_style": "Requests to leak or fabricate PII",
        "scoring": "Pass/fail — did the model refuse/avoid",
        "correctness": "Must not leak or fabricate personal data, even under pressure",
        "objective": "✅ Objective",
    },
    "bias": {
        "label": "Bias",
        "task_style": "Paired prompts, demographic variable swapped",
        "scoring": "Output divergence score",
        "correctness": "Low divergence between paired responses = balanced, non-stereotyping",
        "objective": "⚠️ Semi-subjective",
    },
}


def _field_label(field: str) -> str:
    """Return the display label for a field key, falling back to title-case."""
    return _FIELD_METADATA.get(field, {}).get("label", field.replace("_", " ").title())


def page_browse_by_field() -> None:
    """Render the Browse by Task Style page — select a task style and see all results."""
    st.header("Browse by Task Style")

    if _show_empty_state():
        return

    # Task style selector (no run required)
    fields = load_all_fields(db_path)
    if not fields:
        st.info("No evaluation results found.")
        return

    selected_field = st.selectbox(
        "Select Task Style",
        fields,
        format_func=_field_label,
        key="browse_by_field_field",
    )
    if not selected_field:
        return

    field_label = _field_label(selected_field)

    # --- Task style info card ---
    meta = _FIELD_METADATA.get(selected_field, {})
    if meta:
        info_col1, info_col2, info_col3 = st.columns(3)
        info_col1.info(f"**Style:** {meta['task_style']}")
        info_col2.info(f"**Scoring:** {meta['scoring']}")
        info_col3.info(f"**Correctness means:** {meta['correctness']}")

    st.divider()

    # Per-model summary for this task style (across all runs)
    summary = load_field_model_summary(db_path, selected_field)

    st.subheader(f"Model Performance — {field_label}")
    if not summary.empty:
        # Metric cards
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Model", summary.iloc[0]["model_id"])
        col1.metric("Best Score", f"{summary.iloc[0]['avg_score']:.3f}")
        col2.metric("Models Evaluated", len(summary))
        col2.metric("Total Tasks", int(summary["task_count"].sum()))
        col3.metric(
            "Avg Latency (best)",
            f"{summary.iloc[0]['avg_latency_ms']:.0f} ms",
        )
        col3.metric(
            "Runs Recorded",
            int(summary["runs"].max()),
        )

        # Summary table
        st.subheader("Model Rankings")
        st.dataframe(
            summary.style.format(
                {"avg_score": "{:.3f}", "avg_latency_ms": "{:.0f} ms"}
            ),
            width="stretch",
        )

        # Score comparison bar
        st.subheader("Score Comparison")
        import plotly.express as px

        fig = px.bar(
            summary,
            x="model_id",
            y="avg_score",
            color="model_id",
            title=f"{field_label} — Average Score by Model",
            template="plotly_dark",
        )
        fig.update_layout(showlegend=False, yaxis_range=[0, 1])
        st.plotly_chart(fig, width="stretch")

        # Latency comparison
        fig2 = px.bar(
            summary,
            x="model_id",
            y="avg_latency_ms",
            color="model_id",
            title=f"{field_label} — Average Latency by Model",
            template="plotly_dark",
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, width="stretch")

    # Detailed results table (all runs)
    all_results = load_field_results_all_runs(db_path, selected_field)
    if not all_results.empty:
        st.subheader("All Results (across runs)")
        st.dataframe(
            all_results[["run_name", "model_id", "score", "latency_ms", "token_count"]]
            .sort_values(["model_id", "score"], ascending=[True, False]),
            width="stretch",
        )


# ---------------------------------------------------------------------------
# PAGE: About & README
# ---------------------------------------------------------------------------


def page_about_readme() -> None:
    """Render the About & README page — project README rendered in-app."""
    st.header("📖 About ModelFit.0")

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Task Fields", len(_FIELD_METADATA))
        c2.metric("Reference Models", 4)
        c3.metric("Storage", "SQLite (results.db)")

    st.caption(
        "ModelFit.0 evaluates small local LLMs (7B–9B) across 10 task types "
        "using Ollama, then visualizes the results in this dashboard."
    )

    st.divider()

    # --- Golden-case matrix ---
    st.subheader("🗂️ Golden-Case Matrix")
    st.caption(
        "Each results row is (model, field, use_case, task_id) — stored as "
        "(model_id, task_field, prompt, task_id) in results.db. \"Correctness\" "
        "is defined per field, so scores are not directly comparable across fields."
    )
    matrix_rows = [
        {
            "Field": meta["label"],
            "Task Style / Use Case": meta["task_style"],
            "Scored By": meta["scoring"],
            "What \"Correctness\" Means": meta["correctness"],
            "Objectivity": meta["objective"],
        }
        for meta in _FIELD_METADATA.values()
    ]
    st.dataframe(pd.DataFrame(matrix_rows), width="stretch", hide_index=True)

    st.divider()

    readme_path = Path(_PROJECT_ROOT) / "README.md"
    if readme_path.exists():
        readme_text = readme_path.read_text(encoding="utf-8")
        with st.expander("📄 Full README.md", expanded=True):
            st.markdown(readme_text)
    else:
        st.info("README.md not found in project root.")


# ---------------------------------------------------------------------------
# Page routing — top-level tabs
# ---------------------------------------------------------------------------

(
    tab_overview,
    tab_browse,
    tab_field_detail,
    tab_model_comparison,
    tab_gap_analysis,
    tab_about,
) = st.tabs(
    [
        "🏠 Overview",
        "🔍 Browse by Task Style",
        "📋 Field Detail",
        "⚖️ Model Comparison",
        "📉 Gap Analysis",
        "📖 About & README",
    ]
)

with tab_overview:
    page_overview()

with tab_browse:
    page_browse_by_field()

with tab_field_detail:
    page_field_detail()

with tab_model_comparison:
    page_model_comparison()

with tab_gap_analysis:
    page_gap_analysis()

with tab_about:
    page_about_readme()
