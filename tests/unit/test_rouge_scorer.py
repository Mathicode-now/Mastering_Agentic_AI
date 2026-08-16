"""Unit tests for src/scoring/rouge.py.

Uses pytest.importorskip to skip all tests if rouge_score is not installed,
ensuring tests work without the scoring extras.
"""

import pytest

rouge_score = pytest.importorskip("rouge_score", reason="rouge_score not installed")

from src.scoring.rouge import (
    rouge_1_score,
    rouge_2_score,
    rouge_aggregate,
    rouge_l_score,
)

# --- rouge_l_score ---


class TestRougeLScore:
    """Tests for ROUGE-L F-measure scoring."""

    def test_rouge_l_identical_strings(self):
        """Identical strings should score 1.0."""
        text = "The quick brown fox jumps over the lazy dog"
        assert rouge_l_score(text, text) == 1.0

    def test_rouge_l_completely_different(self):
        """Completely different strings should score below 0.3."""
        response = "alpha beta gamma delta epsilon"
        reference = "one two three four five six seven"
        score = rouge_l_score(response, reference)
        assert score < 0.3

    def test_rouge_l_partial_overlap(self):
        """Partially overlapping strings should score between 0.3 and 1.0."""
        response = "The quick brown fox jumps over the lazy dog"
        reference = "The quick brown cat sits on the lazy mat"
        score = rouge_l_score(response, reference)
        assert 0.3 < score < 1.0

    def test_rouge_l_empty_string_response(self):
        """Empty response returns 0.0."""
        assert rouge_l_score("", "some reference text") == 0.0

    def test_rouge_l_empty_string_reference(self):
        """Empty reference returns 0.0."""
        assert rouge_l_score("some response text", "") == 0.0

    def test_rouge_l_whitespace_only(self):
        """Whitespace-only input returns 0.0."""
        assert rouge_l_score("   ", "some text") == 0.0


# --- rouge_1_score ---


class TestRouge1Score:
    """Tests for ROUGE-1 F-measure scoring."""

    def test_rouge_1_identical(self):
        """Identical strings should score 1.0."""
        text = "The model generates accurate responses"
        assert rouge_1_score(text, text) == 1.0

    def test_rouge_1_partial_overlap(self):
        """Partial overlap should produce a mid-range score."""
        response = "The model generates accurate responses quickly"
        reference = "The model produces correct answers efficiently"
        score = rouge_1_score(response, reference)
        assert 0.0 < score < 1.0

    def test_rouge_1_empty_string(self):
        """Empty input returns 0.0."""
        assert rouge_1_score("", "reference") == 0.0


# --- rouge_2_score ---


class TestRouge2Score:
    """Tests for ROUGE-2 F-measure scoring."""

    def test_rouge_2_identical(self):
        """Identical strings should score 1.0."""
        text = "natural language processing is important"
        assert rouge_2_score(text, text) == 1.0

    def test_rouge_2_no_bigram_overlap(self):
        """No bigram overlap should score 0.0."""
        response = "alpha beta gamma"
        reference = "one two three"
        assert rouge_2_score(response, reference) == 0.0

    def test_rouge_2_empty_string(self):
        """Empty input returns 0.0."""
        assert rouge_2_score("", "reference") == 0.0


# --- rouge_aggregate ---


class TestRougeAggregate:
    """Tests for aggregate ROUGE scoring."""

    def test_rouge_aggregate_returns_all_keys(self):
        """Aggregate result contains rouge1, rouge2, and rougeL keys."""
        result = rouge_aggregate("hello world foo", "hello world bar")
        assert "rouge1" in result
        assert "rouge2" in result
        assert "rougeL" in result
        # All values should be floats between 0 and 1
        for value in result.values():
            assert isinstance(value, float)
            assert 0.0 <= value <= 1.0

    def test_rouge_aggregate_identical(self):
        """Identical strings produce all 1.0 scores."""
        text = "The quick brown fox jumps over the lazy dog"
        result = rouge_aggregate(text, text)
        assert result["rouge1"] == 1.0
        assert result["rouge2"] == 1.0
        assert result["rougeL"] == 1.0

    def test_rouge_aggregate_empty_returns_zeros(self):
        """Empty input returns all zeros."""
        result = rouge_aggregate("", "reference text")
        assert result == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
