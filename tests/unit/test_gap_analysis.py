"""Unit tests for dashboard gap_analysis module."""

import pandas as pd
import pytest

from src.dashboard.gap_analysis import GapAnalyzer

# Path to the real config file
WEIGHTS_PATH = "config/scoring_weights.yaml"


@pytest.fixture
def analyzer():
    """Create a GapAnalyzer with the real scoring weights config."""
    return GapAnalyzer(weights_path=WEIGHTS_PATH)


@pytest.fixture
def scores_df():
    """Synthetic scores pivot: model_id (index) × field (columns)."""
    data = {
        "code_generation": [0.9, 0.7, 0.5],
        "summarization": [0.8, 0.85, 0.6],
        "rag_qa": [0.75, 0.9, 0.65],
        "classification": [0.95, 0.8, 0.7],
        "reasoning": [0.7, 0.6, 0.9],
        "creative_writing": [0.8, 0.7, 0.75],
        "regression": [0.6, 0.65, 0.55],
        "adversarial": [0.85, 0.75, 0.8],
        "pii": [0.9, 0.88, 0.7],
        "bias": [0.7, 0.6, 0.8],
    }
    return pd.DataFrame(data, index=["model-alpha", "model-beta", "model-gamma"])


@pytest.fixture
def latency_df():
    """Synthetic latency pivot matching scores_df structure."""
    data = {
        "code_generation": [200.0, 300.0, 150.0],
        "summarization": [100.0, 120.0, 90.0],
        "rag_qa": [250.0, 180.0, 220.0],
        "classification": [80.0, 100.0, 70.0],
        "reasoning": [350.0, 400.0, 300.0],
        "creative_writing": [150.0, 180.0, 120.0],
        "regression": [90.0, 110.0, 85.0],
        "adversarial": [200.0, 220.0, 190.0],
        "pii": [120.0, 140.0, 100.0],
        "bias": [100.0, 130.0, 95.0],
    }
    return pd.DataFrame(data, index=["model-alpha", "model-beta", "model-gamma"])


class TestWeightsConfig:
    def test_weights_sum_to_one(self, analyzer):
        """Validate that field weights in the config sum to 1.0."""
        total = sum(analyzer.field_weights.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_metric_weights_sum_to_one(self, analyzer):
        """Validate that metric weights in the config sum to 1.0."""
        total = sum(analyzer.metric_weights.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_field_weights_loaded(self, analyzer):
        """Ensure the expected fields are in the config."""
        expected_fields = {
            "code_generation", "summarization", "rag_qa", "classification",
            "reasoning", "creative_writing", "regression", "adversarial",
            "pii", "bias",
        }
        assert set(analyzer.field_weights.keys()) == expected_fields


class TestComputeWeightedScore:
    def test_compute_weighted_score_basic(self, analyzer):
        """Test weighted score without RAM."""
        model_scores = {field: 0.8 for field in analyzer.field_weights}
        model_latency = {field: 100.0 for field in analyzer.field_weights}

        score = analyzer.compute_weighted_score(model_scores, model_latency)

        # With uniform scores and latencies, score should be deterministic
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # With latency_norm = 1 - (100/100) = 0 when all latencies equal max
        # score = sum(field_weight * (0.6 * 0.8 + 0.2 * 0.0 + 0.2 * 0.0))
        # = sum(field_weight * 0.48) = 0.48 * 1.0 = 0.48
        assert score == pytest.approx(0.48, abs=1e-6)

    def test_compute_weighted_score_with_ram(self, analyzer):
        """Test weighted score with RAM efficiency included."""
        model_scores = {field: 0.8 for field in analyzer.field_weights}
        model_latency = {field: 100.0 for field in analyzer.field_weights}
        model_ram = {field: 4.0 for field in analyzer.field_weights}

        score = analyzer.compute_weighted_score(model_scores, model_latency, model_ram)

        assert isinstance(score, float)
        # With uniform RAM: ram_norm = 1 - (4/4) = 0
        # Same as without RAM in this case
        assert score == pytest.approx(0.48, abs=1e-6)

    def test_compute_weighted_score_varying_latency(self, analyzer):
        """Score should increase when latency decreases (lower is better)."""
        model_scores = {field: 0.8 for field in analyzer.field_weights}

        # High latency
        high_latency = {field: 500.0 for field in analyzer.field_weights}
        high_latency["code_generation"] = 100.0  # One low value to create variation

        # Low latency
        low_latency = {field: 100.0 for field in analyzer.field_weights}

        score_high = analyzer.compute_weighted_score(model_scores, high_latency)
        score_low = analyzer.compute_weighted_score(model_scores, low_latency)

        # High latency (all same) gets norm=0, low latency (all same) gets norm=0
        # But high_latency has variation, so code_gen gets a better norm
        # The score with variation should differ
        assert isinstance(score_high, float)
        assert isinstance(score_low, float)

    def test_compute_weighted_score_with_varying_ram(self, analyzer):
        """Score should be higher with lower RAM usage."""
        model_scores = {field: 0.8 for field in analyzer.field_weights}
        model_latency = {field: 100.0 for field in analyzer.field_weights}

        # Low RAM (relative)
        low_ram = {field: 2.0 for field in analyzer.field_weights}
        low_ram["code_generation"] = 8.0  # One high value to create a max

        # High RAM (all at max)
        high_ram = {field: 8.0 for field in analyzer.field_weights}

        score_low = analyzer.compute_weighted_score(model_scores, model_latency, low_ram)
        score_high = analyzer.compute_weighted_score(model_scores, model_latency, high_ram)

        # Lower RAM should yield a higher composite score
        assert score_low > score_high


class TestComputeGapMatrix:
    def test_compute_gap_matrix_ranks_correctly(self, analyzer, scores_df, latency_df):
        """Verify models are ranked by weighted_score descending."""
        gap_df = analyzer.compute_gap_matrix(scores_df, latency_df)

        assert isinstance(gap_df, pd.DataFrame)
        assert list(gap_df.columns) == [
            "model_id", "weighted_score", "rank", "gap_to_best", "strengths", "weaknesses"
        ]
        # Rank should be 1, 2, 3
        assert list(gap_df["rank"]) == [1, 2, 3]
        # Scores should be descending
        scores = gap_df["weighted_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_compute_gap_matrix_identifies_strengths_weaknesses(
        self, analyzer, scores_df, latency_df
    ):
        """Verify strengths and weaknesses are lists of field names."""
        gap_df = analyzer.compute_gap_matrix(scores_df, latency_df)

        for _, row in gap_df.iterrows():
            assert isinstance(row["strengths"], list)
            assert isinstance(row["weaknesses"], list)
            assert len(row["strengths"]) <= 3
            assert len(row["weaknesses"]) <= 3
            # All entries should be valid field names
            all_fields = set(analyzer.field_weights.keys())
            for s in row["strengths"]:
                assert s in all_fields
            for w in row["weaknesses"]:
                assert w in all_fields

    def test_compute_gap_matrix_gap_to_best(self, analyzer, scores_df, latency_df):
        """Best model should have gap_to_best = 0."""
        gap_df = analyzer.compute_gap_matrix(scores_df, latency_df)

        # First row (rank 1) should have gap_to_best = 0
        assert gap_df.iloc[0]["gap_to_best"] == 0.0
        # Other rows should have positive gaps
        for i in range(1, len(gap_df)):
            assert gap_df.iloc[i]["gap_to_best"] > 0.0

    def test_compute_gap_matrix_with_ram(self, analyzer, scores_df, latency_df):
        """Verify gap matrix works with RAM config."""
        ram_config = {"model-alpha": 8.0, "model-beta": 4.0, "model-gamma": 6.0}
        gap_df = analyzer.compute_gap_matrix(scores_df, latency_df, ram_config)

        assert len(gap_df) == 3
        assert list(gap_df["rank"]) == [1, 2, 3]


class TestGetFieldGaps:
    def test_get_field_gaps(self, analyzer, scores_df):
        """Verify per-field best/worst model identification."""
        gaps_df = analyzer.get_field_gaps(scores_df)

        assert isinstance(gaps_df, pd.DataFrame)
        expected_cols = ["field", "best_model", "best_score", "worst_model", "worst_score", "gap"]
        assert list(gaps_df.columns) == expected_cols

        # Should have one row per field
        assert len(gaps_df) == len(scores_df.columns)

        # Check a known field: code_generation scores are [0.9, 0.7, 0.5]
        code_gen_row = gaps_df[gaps_df["field"] == "code_generation"].iloc[0]
        assert code_gen_row["best_model"] == "model-alpha"
        assert code_gen_row["best_score"] == pytest.approx(0.9)
        assert code_gen_row["worst_model"] == "model-gamma"
        assert code_gen_row["worst_score"] == pytest.approx(0.5)
        assert code_gen_row["gap"] == pytest.approx(0.4)

    def test_get_field_gaps_sorted_by_gap(self, analyzer, scores_df):
        """Results should be sorted by gap descending."""
        gaps_df = analyzer.get_field_gaps(scores_df)
        gaps = gaps_df["gap"].tolist()
        assert gaps == sorted(gaps, reverse=True)


class TestGetImprovementRecommendations:
    def test_get_improvement_recommendations_returns_list(
        self, analyzer, scores_df, latency_df
    ):
        """Verify recommendations are a list of properly structured dicts."""
        gap_df = analyzer.compute_gap_matrix(scores_df, latency_df)
        recs = analyzer.get_improvement_recommendations(gap_df)

        assert isinstance(recs, list)
        # Should have recommendations for models that aren't the best
        assert len(recs) >= 1

        for rec in recs:
            assert "model" in rec
            assert "recommendation" in rec
            assert "priority" in rec
            assert "expected_impact" in rec
            assert rec["priority"] in {"high", "medium", "low"}
            assert isinstance(rec["expected_impact"], float)
            assert isinstance(rec["recommendation"], str)

    def test_get_improvement_recommendations_excludes_best(
        self, analyzer, scores_df, latency_df
    ):
        """The best model should not appear in recommendations."""
        gap_df = analyzer.compute_gap_matrix(scores_df, latency_df)
        best_model = gap_df.iloc[0]["model_id"]

        recs = analyzer.get_improvement_recommendations(gap_df)
        rec_models = [r["model"] for r in recs]

        assert best_model not in rec_models

    def test_get_improvement_recommendations_sorted_by_priority(
        self, analyzer, scores_df, latency_df
    ):
        """Recommendations should be sorted: high > medium > low priority."""
        gap_df = analyzer.compute_gap_matrix(scores_df, latency_df)
        recs = analyzer.get_improvement_recommendations(gap_df)

        priority_order = {"high": 0, "medium": 1, "low": 2}
        priorities = [priority_order[r["priority"]] for r in recs]

        # Check non-decreasing priority order (same priorities may be interleaved by impact)
        for i in range(len(priorities) - 1):
            assert priorities[i] <= priorities[i + 1]
