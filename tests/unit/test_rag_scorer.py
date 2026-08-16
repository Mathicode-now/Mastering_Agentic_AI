"""Unit tests for src/scoring/rag.py.

All tests use the local heuristic implementations directly — no mocking
or external dependencies needed.
"""


from src.scoring.rag import (
    answer_correctness,
    faithfulness_score,
    rag_composite_score,
    relevance_score,
)

# --- faithfulness_score ---


class TestFaithfulnessScore:
    """Tests for faithfulness (response supported by context)."""

    def test_faithfulness_high_overlap(self):
        """Response sentences derived from context should score high."""
        context = (
            "Python is a high-level programming language. "
            "It was created by Guido van Rossum. "
            "Python emphasizes code readability and simplicity."
        )
        response = (
            "Python is a high-level programming language. "
            "It was created by Guido van Rossum. "
            "Python emphasizes code readability."
        )
        score = faithfulness_score(response, context)
        assert score >= 0.5

    def test_faithfulness_no_overlap(self):
        """Response unrelated to context should score low."""
        context = "The solar system has eight planets orbiting the sun."
        response = "Machine learning models require extensive training data."
        score = faithfulness_score(response, context)
        assert score < 0.3

    def test_faithfulness_empty_response(self):
        """Empty response returns 0.0."""
        assert faithfulness_score("", "some context") == 0.0

    def test_faithfulness_empty_context(self):
        """Empty context returns 0.0."""
        assert faithfulness_score("some response", "") == 0.0


# --- relevance_score ---


class TestRelevanceScore:
    """Tests for relevance (response addresses the question)."""

    def test_relevance_addresses_question(self):
        """Response containing question keywords should score high."""
        question = "What is the capital of France?"
        response = "The capital of France is Paris, a major European city."
        score = relevance_score(response, question)
        assert score > 0.5

    def test_relevance_irrelevant(self):
        """Response not addressing the question should score low."""
        question = "What is the capital of France?"
        response = "Machine learning uses neural networks for pattern recognition."
        score = relevance_score(response, question)
        assert score < 0.3

    def test_relevance_empty_response(self):
        """Empty response returns 0.0."""
        assert relevance_score("", "What is Python?") == 0.0

    def test_relevance_empty_question(self):
        """Empty question returns 0.0."""
        assert relevance_score("Python is a language", "") == 0.0


# --- answer_correctness ---


class TestAnswerCorrectness:
    """Tests for answer correctness (token F1 vs ground truth)."""

    def test_answer_correctness_exact(self):
        """Identical answer should score 1.0."""
        text = "Paris is the capital of France"
        score = answer_correctness(text, text)
        assert score == 1.0

    def test_answer_correctness_partial(self):
        """Partial overlap should score between 0.3 and 0.9."""
        response = "Paris is the capital of France and a beautiful city"
        ground_truth = "The capital of France is Paris"
        score = answer_correctness(response, ground_truth)
        assert 0.3 < score < 0.9

    def test_answer_correctness_wrong(self):
        """Completely wrong answer should score low."""
        response = "Tokyo is located in Japan"
        ground_truth = "Paris is the capital of France"
        score = answer_correctness(response, ground_truth)
        assert score < 0.3

    def test_answer_correctness_empty_response(self):
        """Empty response returns 0.0."""
        assert answer_correctness("", "expected answer") == 0.0

    def test_answer_correctness_empty_ground_truth(self):
        """Empty ground truth returns 0.0."""
        assert answer_correctness("some answer", "") == 0.0


# --- rag_composite_score ---


class TestRagCompositeScore:
    """Tests for the weighted composite RAG score."""

    def test_rag_composite_combines_scores(self):
        """Composite score should be a weighted combination of sub-scores."""
        context = "Python is a programming language created by Guido van Rossum."
        question = "What is Python?"
        response = "Python is a programming language created by Guido van Rossum."
        ground_truth = "Python is a programming language."

        composite = rag_composite_score(response, context, question, ground_truth)

        # Verify it's a reasonable combination (all sub-scores should be high here)
        assert 0.5 < composite <= 1.0

        # Verify weights: faith=0.3, relevance=0.3, correctness=0.4
        faith = faithfulness_score(response, context)
        relevance = relevance_score(response, question)
        correctness = answer_correctness(response, ground_truth)
        expected = 0.3 * faith + 0.3 * relevance + 0.4 * correctness
        assert abs(composite - expected) < 1e-9

    def test_rag_composite_empty_response(self):
        """Empty response returns 0.0."""
        score = rag_composite_score("", "context", "question", "truth")
        assert score == 0.0

    def test_rag_empty_inputs(self):
        """Empty/whitespace inputs yield 0.0."""
        assert rag_composite_score("", "ctx", "q", "gt") == 0.0
        assert rag_composite_score("   ", "ctx", "q", "gt") == 0.0

    def test_rag_composite_low_when_unrelated(self):
        """Unrelated response should produce a low composite score."""
        context = "The Earth orbits the Sun in approximately 365 days."
        question = "How long does Earth take to orbit the Sun?"
        response = "Quantum computing uses qubits for parallel processing."
        ground_truth = "Earth takes about 365 days to orbit the Sun."

        score = rag_composite_score(response, context, question, ground_truth)
        assert score < 0.4
