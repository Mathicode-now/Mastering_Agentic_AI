"""Reusable Plotly chart factory functions for the evaluation dashboard.

All functions accept pandas DataFrames and return plotly.graph_objects.Figure
instances. No side effects — charts are not displayed, only constructed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Consistent dark template for all charts
_TEMPLATE = "plotly_dark"


def create_heatmap(
    scores_df: pd.DataFrame,
    title: str = "Model × Field Scores",
) -> go.Figure:
    """Create a heatmap of model scores across task fields.

    Args:
        scores_df: Pivot table with model_id as index and task_field as
            columns. Values are average scores (0-1).
        title: Chart title.

    Returns:
        Plotly Figure with green (high) to red (low) color scale.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=scores_df.values,
            x=scores_df.columns.tolist(),
            y=scores_df.index.tolist(),
            colorscale=[[0, "red"], [0.5, "yellow"], [1, "green"]],
            zmin=0,
            zmax=1,
            text=np.round(scores_df.values, 3),
            texttemplate="%{text}",
            hovertemplate="Model: %{y}<br>Field: %{x}<br>Score: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Task Field",
        yaxis_title="Model",
        template=_TEMPLATE,
    )
    return fig


def create_model_comparison_bar(comparison_df: pd.DataFrame) -> go.Figure:
    """Create a grouped bar chart comparing models on score and latency.

    Uses dual y-axes: left for avg_score, right for avg_latency_ms.

    Args:
        comparison_df: DataFrame with columns: model_id, avg_score,
            avg_latency_ms, total_tasks, total_tokens.

    Returns:
        Plotly Figure with grouped bars and dual y-axis.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=comparison_df["model_id"],
            y=comparison_df["avg_score"],
            name="Avg Score",
            marker_color="#2ecc71",
            offsetgroup=0,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(
            x=comparison_df["model_id"],
            y=comparison_df["avg_latency_ms"],
            name="Avg Latency (ms)",
            marker_color="#e74c3c",
            offsetgroup=1,
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Model Comparison: Score vs Latency",
        barmode="group",
        template=_TEMPLATE,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_yaxes(title_text="Avg Score", secondary_y=False)
    fig.update_yaxes(title_text="Avg Latency (ms)", secondary_y=True)

    return fig


def create_latency_vs_accuracy_scatter(results_df: pd.DataFrame) -> go.Figure:
    """Create a scatter plot of latency vs accuracy colored by model.

    Args:
        results_df: DataFrame with columns: model_id, latency_ms, score,
            token_count.

    Returns:
        Plotly Figure scatter with x=latency_ms, y=score, color=model_id,
        size=token_count.
    """
    fig = go.Figure()

    models = results_df["model_id"].unique()
    colors = _generate_color_palette(len(models))

    for model, color in zip(models, colors):
        subset = results_df[results_df["model_id"] == model]
        fig.add_trace(
            go.Scatter(
                x=subset["latency_ms"],
                y=subset["score"],
                mode="markers",
                name=model,
                marker={
                    "size": _normalize_sizes(subset["token_count"]),
                    "color": color,
                    "opacity": 0.7,
                    "line": {"width": 1, "color": "white"},
                },
                hovertemplate=(
                    f"<b>{model}</b><br>"
                    "Latency: %{x:.0f}ms<br>"
                    "Score: %{y:.3f}<br>"
                    "Tokens: %{customdata[0]}<extra></extra>"
                ),
                customdata=subset[["token_count"]].values,
            )
        )

    fig.update_layout(
        title="Latency vs Accuracy",
        xaxis_title="Latency (ms)",
        yaxis_title="Score",
        template=_TEMPLATE,
    )
    return fig


def create_field_breakdown_radar(
    scores_df: pd.DataFrame, model_id: str
) -> go.Figure:
    """Create a radar chart showing one model's scores across all fields.

    Args:
        scores_df: Pivot table with model_id as index and task_field as
            columns. Values are average scores (0-1).
        model_id: The model to display.

    Returns:
        Plotly Figure radar chart for the specified model.
    """
    fields = scores_df.columns.tolist()
    values = scores_df.loc[model_id].values.tolist()

    # Close the radar polygon
    values_closed = values + [values[0]]
    fields_closed = fields + [fields[0]]

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values_closed,
            theta=fields_closed,
            fill="toself",
            name=model_id,
            line={"color": "#3498db"},
            fillcolor="rgba(52, 152, 219, 0.3)",
        )
    )

    fig.update_layout(
        title=f"Field Breakdown: {model_id}",
        polar={
            "radialaxis": {"visible": True, "range": [0, 1]},
        },
        template=_TEMPLATE,
    )
    return fig


def create_gap_waterfall(gap_df: pd.DataFrame) -> go.Figure:
    """Create a waterfall chart showing gap_to_best per model.

    Args:
        gap_df: DataFrame from GapAnalyzer.compute_gap_matrix() with
            columns: model_id, weighted_score, rank, gap_to_best.

    Returns:
        Plotly Figure waterfall showing each model's gap to the best.
    """
    # Sort by rank so best model is first
    sorted_df = gap_df.sort_values("rank").reset_index(drop=True)

    measures = []
    values = []
    for i, row in sorted_df.iterrows():
        if i == 0:
            measures.append("absolute")
            values.append(row["weighted_score"])
        else:
            measures.append("relative")
            values.append(-row["gap_to_best"] if i == 1 else -(row["gap_to_best"] - sorted_df.iloc[i - 1]["gap_to_best"]))

    fig = go.Figure(
        data=go.Waterfall(
            x=sorted_df["model_id"].tolist(),
            y=values,
            measure=measures,
            text=[f"{v:.4f}" for v in sorted_df["weighted_score"]],
            textposition="outside",
            connector={"line": {"color": "rgba(255,255,255,0.3)"}},
            increasing={"marker": {"color": "#2ecc71"}},
            decreasing={"marker": {"color": "#e74c3c"}},
            totals={"marker": {"color": "#3498db"}},
        )
    )

    fig.update_layout(
        title="Gap to Best Model (Waterfall)",
        yaxis_title="Weighted Score",
        template=_TEMPLATE,
        showlegend=False,
    )
    return fig


def create_score_distribution_box(results_df: pd.DataFrame) -> go.Figure:
    """Create a box plot of score distributions per model, colored by field.

    Args:
        results_df: DataFrame with columns: model_id, score, task_field.

    Returns:
        Plotly Figure box plot with x=model_id, y=score, color=task_field.
    """
    fig = go.Figure()

    fields = results_df["task_field"].unique()
    colors = _generate_color_palette(len(fields))

    for field, color in zip(fields, colors):
        subset = results_df[results_df["task_field"] == field]
        fig.add_trace(
            go.Box(
                x=subset["model_id"],
                y=subset["score"],
                name=field,
                marker_color=color,
                boxmean=True,
            )
        )

    fig.update_layout(
        title="Score Distribution by Model and Field",
        xaxis_title="Model",
        yaxis_title="Score",
        boxmode="group",
        template=_TEMPLATE,
        legend_title_text="Task Field",
    )
    return fig


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _normalize_sizes(
    series: pd.Series, min_size: int = 6, max_size: int = 30
) -> list[float]:
    """Normalize a numeric series to marker sizes for scatter plots.

    Args:
        series: Numeric series to normalize.
        min_size: Minimum marker size in pixels.
        max_size: Maximum marker size in pixels.

    Returns:
        List of marker sizes.
    """
    if series.max() == series.min():
        return [int((min_size + max_size) / 2)] * len(series)
    normalized = (series - series.min()) / (series.max() - series.min())
    return (normalized * (max_size - min_size) + min_size).tolist()


def _generate_color_palette(n: int) -> list[str]:
    """Generate n distinct colors from Plotly's qualitative palette.

    Args:
        n: Number of colors needed.

    Returns:
        List of hex color strings.
    """
    palette = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    ]
    # Repeat if more colors needed than palette provides
    return [palette[i % len(palette)] for i in range(n)]
