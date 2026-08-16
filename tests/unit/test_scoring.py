"""Unit tests for src/scoring/exact_match.py.

All tests run without network access or Ollama.
"""

import pytest

from src.scoring.exact_match import (
    code_execution_score,
    contains_match,
    exact_match,
    get_scorer,
    normalize_response,
)

# --- exact_match ---


def test_exact_match_identical():
    """Identical strings score 1.0."""
    assert exact_match("hello world", "hello world") == 1.0


def test_exact_match_different():
    """Different strings score 0.0."""
    assert exact_match("hello world", "goodbye world") == 0.0


def test_exact_match_whitespace_stripped():
    """Leading/trailing whitespace is stripped before comparison."""
    assert exact_match("  hello world  ", "\nhello world\n") == 1.0


# --- contains_match ---


def test_contains_match_found():
    """Substring present scores 1.0."""
    assert contains_match("The answer is 42.", "42") == 1.0


def test_contains_match_not_found():
    """Substring absent scores 0.0."""
    assert contains_match("The answer is 42.", "99") == 0.0


def test_contains_match_case_insensitive():
    """Matching is case-insensitive."""
    assert contains_match("Hello World", "HELLO WORLD") == 1.0


# --- normalize_response ---


def test_normalize_response_strips_code_fences():
    """Code fences are removed, inner content preserved."""
    response = "```python\nprint('hello')\n```"
    result = normalize_response(response)
    assert result == "print('hello')"
    assert "```" not in result


def test_normalize_response_plain_text():
    """Plain text is returned stripped but unchanged."""
    response = "  just some text  "
    assert normalize_response(response) == "just some text"


# --- code_execution_score ---


def test_code_execution_score_correct():
    """Code that produces correct output scores 1.0."""
    response = "```python\nprint('hello')\n```"
    assert code_execution_score(response, "hello", language="python") == 1.0


def test_code_execution_score_wrong_output():
    """Code that runs but produces wrong output scores 0.5."""
    response = "```python\nprint('goodbye')\n```"
    assert code_execution_score(response, "hello", language="python") == 0.5


def test_code_execution_score_error():
    """Code that raises an error scores 0.0."""
    response = "```python\nraise ValueError('oops')\n```"
    assert code_execution_score(response, "anything", language="python") == 0.0


def test_code_execution_score_timeout():
    """Code that exceeds timeout scores 0.0."""
    response = "```python\nimport time\ntime.sleep(10)\nprint('done')\n```"
    assert code_execution_score(response, "done", language="python", timeout=0.5) == 0.0


# --- get_scorer ---


def test_get_scorer_returns_correct_function():
    """Factory returns the expected scorer for each known type."""
    assert get_scorer("exact_match") is exact_match
    assert get_scorer("contains") is contains_match
    assert get_scorer("code_execution") is code_execution_score


def test_get_scorer_unknown_raises():
    """Unknown scoring type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown scoring type"):
        get_scorer("nonexistent_scorer")


# --- get_scorer: new scoring types ---


def test_get_scorer_rouge_type():
    """get_scorer('rouge') returns rouge_l_score function."""
    pytest.importorskip("rouge_score", reason="rouge_score not installed")
    scorer = get_scorer("rouge")
    from src.scoring.rouge import rouge_l_score

    assert scorer is rouge_l_score


def test_get_scorer_bert_score_type():
    """get_scorer('bert_score') returns bert_score_f1 function."""
    pytest.importorskip("bert_score", reason="bert_score not installed")
    scorer = get_scorer("bert_score")
    from src.scoring.semantic import bert_score_f1

    assert scorer is bert_score_f1


def test_get_scorer_rag_type():
    """get_scorer('rag') returns rag_composite_score function."""
    scorer = get_scorer("rag")
    from src.scoring.rag import rag_composite_score

    assert scorer is rag_composite_score
