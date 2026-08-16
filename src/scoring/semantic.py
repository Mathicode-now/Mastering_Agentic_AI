"""Semantic similarity scoring module using BERTScore.

Provides BERTScore F1 measurement for evaluating semantic similarity
between model responses and reference texts.
"""

try:
    from bert_score import BERTScorer
except ImportError:
    raise RuntimeError(
        'Install scoring extras: pip install -e ".[scoring]"'
    )

_scorer_instance: "BERTScorer | None" = None
_scorer_model_type: str = ""


def _get_scorer(model_type: str = 'microsoft/deberta-xlarge-mnli') -> "BERTScorer":
    """Return a cached BERTScorer singleton, initializing on first call.

    Uses module-level caching to avoid reloading the model on every
    scoring call. A new scorer is created if the model_type changes.

    Args:
        model_type: HuggingFace model identifier for BERTScore.

    Returns:
        A BERTScorer instance configured with the specified model.
    """
    global _scorer_instance, _scorer_model_type

    if _scorer_instance is None or _scorer_model_type != model_type:
        # Use fast tokenizer disabled to avoid integer overflow issues
        _scorer_instance = BERTScorer(
            model_type=model_type,
            lang='en',
            use_fast_tokenizer=False,
        )
        _scorer_model_type = model_type

    return _scorer_instance


def bert_score_f1(
    response: str,
    reference: str,
    model_type: str = 'microsoft/deberta-xlarge-mnli',
) -> float:
    """Compute BERTScore F1 between response and reference.

    Uses a pre-trained language model to measure semantic similarity
    between the candidate response and the reference text.

    Args:
        response: The model's response string.
        reference: The reference/expected string.
        model_type: HuggingFace model identifier for BERTScore computation.

    Returns:
        BERTScore F1 value between 0.0 and 1.0.
        Returns 0.0 if either input is empty.
    """
    if not response or not response.strip() or not reference or not reference.strip():
        return 0.0

    scorer = _get_scorer(model_type)
    # BERTScorer expects lists of strings
    _precision, _recall, f1 = scorer.score(
        cands=[response.strip()],
        refs=[reference.strip()],
    )
    return float(f1[0].item())
