"""ROUGE scoring module for model evaluation.

Provides ROUGE-1, ROUGE-2, and ROUGE-L F-measure scores for comparing
model responses against reference texts.
"""

try:
    from rouge_score import rouge_scorer
except ImportError:
    raise RuntimeError(
        'Install scoring extras: pip install -e ".[scoring]"'
    )


def _get_scorer() -> rouge_scorer.RougeScorer:
    """Return a shared RougeScorer instance.

    Returns:
        A RougeScorer configured for rouge1, rouge2, and rougeL.
    """
    return rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


def rouge_l_score(response: str, reference: str) -> float:
    """Compute ROUGE-L F-measure between response and reference.

    Args:
        response: The model's response string.
        reference: The reference/expected string.

    Returns:
        ROUGE-L F-measure score between 0.0 and 1.0.
        Returns 0.0 if either input is empty.
    """
    if not response or not response.strip() or not reference or not reference.strip():
        return 0.0

    scorer = _get_scorer()
    scores = scorer.score(reference.strip(), response.strip())
    return float(scores['rougeL'].fmeasure)


def rouge_1_score(response: str, reference: str) -> float:
    """Compute ROUGE-1 F-measure between response and reference.

    Args:
        response: The model's response string.
        reference: The reference/expected string.

    Returns:
        ROUGE-1 F-measure score between 0.0 and 1.0.
        Returns 0.0 if either input is empty.
    """
    if not response or not response.strip() or not reference or not reference.strip():
        return 0.0

    scorer = _get_scorer()
    scores = scorer.score(reference.strip(), response.strip())
    return float(scores['rouge1'].fmeasure)


def rouge_2_score(response: str, reference: str) -> float:
    """Compute ROUGE-2 F-measure between response and reference.

    Args:
        response: The model's response string.
        reference: The reference/expected string.

    Returns:
        ROUGE-2 F-measure score between 0.0 and 1.0.
        Returns 0.0 if either input is empty.
    """
    if not response or not response.strip() or not reference or not reference.strip():
        return 0.0

    scorer = _get_scorer()
    scores = scorer.score(reference.strip(), response.strip())
    return float(scores['rouge2'].fmeasure)


def rouge_aggregate(response: str, reference: str) -> dict[str, float]:
    """Compute all ROUGE scores (1, 2, L) between response and reference.

    Args:
        response: The model's response string.
        reference: The reference/expected string.

    Returns:
        Dictionary with keys 'rouge1', 'rouge2', 'rougeL' mapping to
        their respective F-measure scores (0.0-1.0).
        Returns all zeros if either input is empty.
    """
    if not response or not response.strip() or not reference or not reference.strip():
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    scorer = _get_scorer()
    scores = scorer.score(reference.strip(), response.strip())
    return {
        'rouge1': float(scores['rouge1'].fmeasure),
        'rouge2': float(scores['rouge2'].fmeasure),
        'rougeL': float(scores['rougeL'].fmeasure),
    }
