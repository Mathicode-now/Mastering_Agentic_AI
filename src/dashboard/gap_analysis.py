"""Weighted scoring module for computing gap analysis across models.

Computes composite scores incorporating accuracy, latency, and RAM efficiency
with configurable field-level and metric-level weights.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


class GapAnalyzer:
    """Analyzes performance gaps between models using weighted scoring.

    Loads field weights and metric weights from a YAML config file and
    provides methods to compute weighted scores, gap matrices, field-level
    gaps, and improvement recommendations.
    """

    def __init__(self, weights_path: str = "config/scoring_weights.yaml") -> None:
        """Initialize GapAnalyzer with weights from YAML config.

        Args:
            weights_path: Path to the scoring weights YAML file.
        """
        path = Path(weights_path)
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        self.field_weights: dict[str, float] = config["field_weights"]
        self.metric_weights: dict[str, float] = config["metric_weights"]

    def compute_weighted_score(
        self,
        model_scores: dict[str, float],
        model_latency: dict[str, float],
        model_ram: dict[str, float] | None = None,
    ) -> float:
        """Compute a single weighted score for a model.

        Formula per field:
            field_weight * (accuracy_weight * score
                          + latency_weight * latency_norm
                          + ram_weight * ram_norm)

        Latency normalized: 1 - (latency / max_latency) so lower is better.
        RAM normalized: 1 - (ram / max_ram) so lower is better.

        Args:
            model_scores: Mapping of field name to accuracy score (0-1).
            model_latency: Mapping of field name to latency value (seconds).
            model_ram: Optional mapping of field name to RAM usage (GB).

        Returns:
            Composite weighted score (higher is better).
        """
        accuracy_weight = self.metric_weights.get("accuracy", 0.6)
        latency_weight = self.metric_weights.get("latency", 0.2)
        ram_weight = self.metric_weights.get("ram_efficiency", 0.2)

        # Normalize latency: 1 - (val / max) so lower latency scores higher
        max_latency = max(model_latency.values()) if model_latency else 1.0
        max_latency = max_latency if max_latency > 0 else 1.0

        # Normalize RAM if provided
        max_ram = 1.0
        if model_ram:
            max_ram = max(model_ram.values()) if model_ram else 1.0
            max_ram = max_ram if max_ram > 0 else 1.0

        total_score = 0.0
        for field, field_weight in self.field_weights.items():
            score = model_scores.get(field, 0.0)
            latency = model_latency.get(field, 0.0)
            latency_norm = 1.0 - (latency / max_latency)

            ram_norm = 0.0
            if model_ram:
                ram = model_ram.get(field, 0.0)
                ram_norm = 1.0 - (ram / max_ram)

            field_score = (
                accuracy_weight * score
                + latency_weight * latency_norm
                + ram_weight * ram_norm
            )
            total_score += field_weight * field_score

        return total_score

    def compute_gap_matrix(
        self,
        scores_df: pd.DataFrame,
        latency_df: pd.DataFrame,
        ram_config: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Compute gap analysis matrix across all models.

        Args:
            scores_df: Model×field pivot table of accuracy scores.
                       Index = model_id, columns = field names.
            latency_df: Model×field pivot table of latency values.
                        Index = model_id, columns = field names.
            ram_config: Optional mapping of model_id to RAM usage (GB).

        Returns:
            DataFrame with columns: model_id, weighted_score, rank,
            gap_to_best, strengths, weaknesses.
        """
        results: list[dict] = []

        for model_id in scores_df.index:
            model_scores = scores_df.loc[model_id].to_dict()
            model_latency = latency_df.loc[model_id].to_dict()

            # For RAM, use the same value across all fields for a given model
            model_ram: dict[str, float] | None = None
            if ram_config and model_id in ram_config:
                model_ram = {
                    field: ram_config[model_id]
                    for field in self.field_weights
                }

            weighted_score = self.compute_weighted_score(
                model_scores, model_latency, model_ram
            )

            # Identify strengths and weaknesses by per-field accuracy
            field_scores = {
                f: model_scores.get(f, 0.0) for f in self.field_weights
            }
            sorted_fields = sorted(
                field_scores.items(), key=lambda x: x[1], reverse=True
            )
            strengths = [f for f, _ in sorted_fields[:3]]
            weaknesses = [f for f, _ in sorted_fields[-3:]]

            results.append(
                {
                    "model_id": model_id,
                    "weighted_score": round(weighted_score, 4),
                    "strengths": strengths,
                    "weaknesses": weaknesses,
                }
            )

        df = pd.DataFrame(results)
        df = df.sort_values("weighted_score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        best_score = df["weighted_score"].max()
        df["gap_to_best"] = round(best_score - df["weighted_score"], 4)

        return df[
            ["model_id", "weighted_score", "rank", "gap_to_best", "strengths", "weaknesses"]
        ]

    def get_field_gaps(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-field gap analysis showing best and worst models.

        Args:
            scores_df: Model×field pivot table of accuracy scores.
                       Index = model_id, columns = field names.

        Returns:
            DataFrame with columns: field, best_model, best_score,
            worst_model, worst_score, gap.
        """
        rows: list[dict] = []

        for field in scores_df.columns:
            col = scores_df[field]
            best_model = col.idxmax()
            best_score = col.max()
            worst_model = col.idxmin()
            worst_score = col.min()
            gap = best_score - worst_score

            rows.append(
                {
                    "field": field,
                    "best_model": best_model,
                    "best_score": round(best_score, 4),
                    "worst_model": worst_model,
                    "worst_score": round(worst_score, 4),
                    "gap": round(gap, 4),
                }
            )

        return pd.DataFrame(rows).sort_values("gap", ascending=False).reset_index(
            drop=True
        )

    def get_improvement_recommendations(
        self, gap_matrix: pd.DataFrame
    ) -> list[dict]:
        """Generate improvement recommendations based on gap analysis.

        Prioritizes models with the largest gap_to_best and focuses
        recommendations on their weakest fields.

        Args:
            gap_matrix: Output from compute_gap_matrix().

        Returns:
            List of dicts with keys: model, recommendation, priority,
            expected_impact.
        """
        recommendations: list[dict] = []

        for _, row in gap_matrix.iterrows():
            gap = row["gap_to_best"]
            if gap == 0:
                continue  # Skip the best model

            # Determine priority from gap magnitude
            if gap > 0.15:
                priority = "high"
            elif gap > 0.05:
                priority = "medium"
            else:
                priority = "low"

            weaknesses = row["weaknesses"]
            weak_fields_str = ", ".join(weaknesses[:2])

            recommendation = (
                f"Improve performance on {weak_fields_str} to close "
                f"the {gap:.2%} gap to the leading model."
            )

            # Expected impact: addressing top weakness with its field weight
            top_weakness = weaknesses[0] if weaknesses else None
            expected_impact = (
                self.field_weights.get(top_weakness, 0.0) * gap
                if top_weakness
                else 0.0
            )

            recommendations.append(
                {
                    "model": row["model_id"],
                    "recommendation": recommendation,
                    "priority": priority,
                    "expected_impact": round(expected_impact, 4),
                }
            )

        # Sort by priority (high first) then expected impact
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(
            key=lambda r: (priority_order.get(r["priority"], 3), -r["expected_impact"])
        )

        return recommendations
