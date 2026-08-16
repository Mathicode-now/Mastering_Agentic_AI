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

st.set_page_config(page_title="Model Eval Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar — navigation and settings
# ---------------------------------------------------------------------------

st.sidebar.title("Model Eval Dashboard")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Browse by Field", "Field Detail", "Model Comparison", "Gap Analysis"],
)

db_path = st.sidebar.text_input("Database path", value="results.db")


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


def _get_run_selector() -> int | None:
    """Show run selector dropdown and return selected run_id."""
    runs = load_runs(db_path)
    if runs.empty:
        return None
    options = {f"{row['name']} (Run #{row['id']})": row["id"] for _, row in runs.iterrows()}
    selected_label = st.selectbox("Select Run", list(options.keys()))
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

        for model_id in leaderboard["model_id"].tolist():
            model_data = per_field_scores[per_field_scores["model_id"] == model_id]
            if model_data.empty:
                continue
            best_field = model_data.loc[model_data["avg_score"].idxmax()]
            worst_field = model_data.loc[model_data["avg_score"].idxmin()]
            st.markdown(
                f"**{model_id}:** "
                f"💪 {best_field['task_field']} ({best_field['avg_score']:.2f}) · "
                f"⚠️ {worst_field['task_field']} ({worst_field['avg_score']:.2f})"
            )

        st.divider()

    # --- Cross-field heatmap ---
    if not per_field_scores.empty:
        st.subheader("Cross-Field Heatmap")
        # Pivot to model × field matrix
        pivot = per_field_scores.pivot(
            index="model_id", columns="task_field", values="avg_score"
        )
        fig = create_heatmap(pivot, title="Model × Field Scores (latest run per field)")
        st.plotly_chart(fig, use_container_width=True)

    # --- Coverage table ---
    st.subheader("📋 Evaluation Coverage")

    all_field_names = list(_FIELD_METADATA.keys())
    evaluated_fields = set(coverage["task_field"].tolist()) if not coverage.empty else set()

    coverage_rows = []
    for field_name in all_field_names:
        if field_name in evaluated_fields:
            row = coverage[coverage["task_field"] == field_name].iloc[0]
            coverage_rows.append({
                "Field": field_name,
                "Status": "✅ Evaluated",
                "Best Model": row["best_model"],
                "Best Score": f"{row['best_score']:.3f}",
            })
        else:
            coverage_rows.append({
                "Field": field_name,
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

    run_id = _get_run_selector()
    if run_id is None:
        return

    # Get available fields from results
    results = load_all_results(db_path, run_id)
    if results.empty:
        st.info("No results for this run.")
        return

    fields = sorted(results["task_field"].unique().tolist())
    selected_field = st.selectbox("Select Field", fields)

    if not selected_field:
        return

    # Per-task results table
    field_data = load_field_detail(db_path, run_id, selected_field)
    if field_data.empty:
        st.info(f"No results for field '{selected_field}'.")
        return

    st.subheader("Per-Task Results")
    display_cols = ["model_id", "task_id", "score", "latency_ms", "token_count"]
    st.dataframe(
        field_data[display_cols].sort_values(["model_id", "score"], ascending=[True, False]),
        use_container_width=True,
    )

    # Score distribution box plot
    st.subheader("Score Distribution")
    fig = create_score_distribution_box(field_data)
    st.plotly_chart(fig, use_container_width=True)

    # Latency vs accuracy scatter
    st.subheader("Latency vs Accuracy")
    fig = create_latency_vs_accuracy_scatter(field_data)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 3: Model Comparison
# ---------------------------------------------------------------------------


def page_model_comparison() -> None:
    """Render the Model Comparison page."""
    st.header("Model Comparison")

    if _show_empty_state():
        return

    run_id = _get_run_selector()
    if run_id is None:
        return

    # Get available models
    comparison = load_model_comparison(db_path, run_id)
    if comparison.empty:
        st.info("No model data for this run.")
        return

    all_models = comparison["model_id"].tolist()
    selected_models = st.multiselect("Select Models", all_models, default=all_models)

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
                    st.plotly_chart(fig, use_container_width=True)

    # Side-by-side latency and score bars
    st.subheader("Score & Latency Comparison")
    if not filtered_comparison.empty:
        fig = create_model_comparison_bar(filtered_comparison)
        st.plotly_chart(fig, use_container_width=True)

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
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# PAGE 4: Gap Analysis
# ---------------------------------------------------------------------------


def page_gap_analysis() -> None:
    """Render the Gap Analysis page."""
    st.header("Gap Analysis")

    if _show_empty_state():
        return

    run_id = _get_run_selector()
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
    st.dataframe(display_gap, use_container_width=True)

    # Gap waterfall chart
    st.subheader("Gap Waterfall")
    fig = create_gap_waterfall(gap_matrix)
    st.plotly_chart(fig, use_container_width=True)

    # Field-level gap table
    st.subheader("Field-Level Gaps")
    field_gaps = analyzer.get_field_gaps(scores_df)
    st.dataframe(field_gaps, use_container_width=True)

    # Improvement recommendations
    st.subheader("Improvement Recommendations")
    recommendations = analyzer.get_improvement_recommendations(gap_matrix)
    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        st.dataframe(rec_df, use_container_width=True)
    else:
        st.success("All models are performing at the same level — no gaps detected.")


# ---------------------------------------------------------------------------
# PAGE: Browse by Field (field-first, no run selection required)
# ---------------------------------------------------------------------------

# Field metadata: task style, scoring method, objectivity
_FIELD_METADATA: dict[str, dict[str, str]] = {
    "summarization": {
        "task_style": "Summarize source docs",
        "scoring": "ROUGE/BERTScore + faithfulness",
        "objective": "✅ Objective",
    },
    "code_generation": {
        "task_style": "Write function to spec",
        "scoring": "Hidden unit tests, pass/fail",
        "objective": "✅ Objective",
    },
    "rag_qa": {
        "task_style": "Answer from retrieved context",
        "scoring": "Faithfulness + answer correctness",
        "objective": "✅ Objective",
    },
    "classification": {
        "task_style": "Label assignment",
        "scoring": "Exact-match accuracy",
        "objective": "✅ Objective",
    },
    "reasoning": {
        "task_style": "Logic/math problems",
        "scoring": "Answer-match + step validity",
        "objective": "✅ Objective",
    },
    "creative_writing": {
        "task_style": "Open-ended generation",
        "scoring": "Rubric scored (coherence, instruction-following)",
        "objective": "⚠️ Semi-subjective",
    },
    "regression": {
        "task_style": "Numeric estimation",
        "scoring": "Error margin vs. ground truth",
        "objective": "✅ Objective",
    },
    "adversarial": {
        "task_style": "Prompt-injection / jailbreak attempts",
        "scoring": "Pass/fail — did guardrails hold",
        "objective": "✅ Objective",
    },
    "pii": {
        "task_style": "Requests to leak/fabricate PII",
        "scoring": "Pass/fail — did model refuse/avoid",
        "objective": "✅ Objective",
    },
    "bias": {
        "task_style": "Paired prompts, demographic variable swapped",
        "scoring": "Output divergence score",
        "objective": "⚠️ Semi-subjective",
    },
}


def page_browse_by_field() -> None:
    """Render the Browse by Field page — select a field and see all results."""
    st.header("Browse by Field")

    if _show_empty_state():
        return

    # Field selector (no run required)
    fields = load_all_fields(db_path)
    if not fields:
        st.info("No evaluation results found.")
        return

    selected_field = st.selectbox("Select Field", fields)
    if not selected_field:
        return

    # --- Field info card ---
    meta = _FIELD_METADATA.get(selected_field, {})
    if meta:
        info_col1, info_col2 = st.columns(2)
        info_col1.info(f"**Task Style:** {meta['task_style']}")
        info_col2.info(f"**Scoring:** {meta['scoring']}")

    st.divider()

    # Per-model summary for this field (across all runs)
    summary = load_field_model_summary(db_path, selected_field)

    st.subheader(f"Model Performance — {selected_field}")
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
            use_container_width=True,
        )

        # Score comparison bar
        st.subheader("Score Comparison")
        import plotly.express as px

        fig = px.bar(
            summary,
            x="model_id",
            y="avg_score",
            color="model_id",
            title=f"{selected_field} — Average Score by Model",
            template="plotly_dark",
        )
        fig.update_layout(showlegend=False, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        # Latency comparison
        fig2 = px.bar(
            summary,
            x="model_id",
            y="avg_latency_ms",
            color="model_id",
            title=f"{selected_field} — Average Latency by Model",
            template="plotly_dark",
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Detailed results table (all runs)
    all_results = load_field_results_all_runs(db_path, selected_field)
    if not all_results.empty:
        st.subheader("All Results (across runs)")
        st.dataframe(
            all_results[["run_name", "model_id", "task_id", "score", "latency_ms", "token_count"]]
            .sort_values(["model_id", "score"], ascending=[True, False]),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------

if page == "Overview":
    page_overview()
elif page == "Browse by Field":
    page_browse_by_field()
elif page == "Field Detail":
    page_field_detail()
elif page == "Model Comparison":
    page_model_comparison()
elif page == "Gap Analysis":
    page_gap_analysis()
