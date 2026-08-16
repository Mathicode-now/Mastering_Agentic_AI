"""Unit tests for src/scoring/semantic.py.

All tests mock the bert_score library so they run without downloading
ML models. Tests are skipped entirely if bert_score is not installed
(since the module raises RuntimeError on import without it).
"""

import importlib
from unittest.mock import MagicMock, patch

import pytest

# Check if bert_score is available for the import to succeed
_bert_score_available = importlib.util.find_spec("bert_score") is not None

pytestmark = pytest.mark.skipif(
    not _bert_score_available,
    reason="bert_score not installed — skipping semantic scorer tests",
)


@pytest.fixture(autouse=True)
def _mock_bert_scorer():
    """Mock BERTScorer to avoid model downloads in every test.

    Yields a mock that returns synthetic tensor-like values from score().
    """
    mock_tensor = MagicMock()
    mock_tensor.__getitem__ = lambda self, idx: MagicMock(item=lambda: 0.95)

    mock_scorer_instance = MagicMock()
    mock_scorer_instance.score.return_value = (
        mock_tensor,  # precision
        mock_tensor,  # recall
        mock_tensor,  # f1
    )

    with patch("src.scoring.semantic.BERTScorer", return_value=mock_scorer_instance):
        # Reset the cached singleton so the mock takes effect
        import src.scoring.semantic as sem_module

        sem_module._scorer_instance = None
        sem_module._scorer_model_type = ""
        yield mock_scorer_instance


class TestBertScoreF1:
    """Tests for bert_score_f1 scoring function."""

    def test_bert_score_identical(self):
        """Identical strings should produce a high score (mocked to 0.95)."""
        from src.scoring.semantic import bert_score_f1

        score = bert_score_f1("The cat sat on the mat", "The cat sat on the mat")
        assert score > 0.8

    def test_bert_score_different(self):
        """Different strings still go through the mock — verifies the call path."""
        from src.scoring.semantic import bert_score_f1

        score = bert_score_f1(
            "Machine learning is fascinating",
            "The weather is nice today",
        )
        # Mock returns 0.95 regardless, but we verify the function works
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_bert_score_empty_string_response(self):
        """Empty response returns 0.0 without calling the model."""
        from src.scoring.semantic import bert_score_f1

        score = bert_score_f1("", "some reference")
        assert score == 0.0

    def test_bert_score_empty_string_reference(self):
        """Empty reference returns 0.0 without calling the model."""
        from src.scoring.semantic import bert_score_f1

        score = bert_score_f1("some response", "")
        assert score == 0.0

    def test_bert_score_whitespace_only(self):
        """Whitespace-only input returns 0.0."""
        from src.scoring.semantic import bert_score_f1

        assert bert_score_f1("   ", "reference") == 0.0
        assert bert_score_f1("response", "   \n\t  ") == 0.0

    def test_bert_score_calls_scorer_with_correct_args(self, _mock_bert_scorer):
        """Verify BERTScorer.score is called with lists of stripped strings."""
        from src.scoring.semantic import bert_score_f1

        bert_score_f1("  hello world  ", "  reference text  ")

        _mock_bert_scorer.score.assert_called_once_with(
            cands=["hello world"],
            refs=["reference text"],
        )
