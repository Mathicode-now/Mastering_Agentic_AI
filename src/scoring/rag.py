"""Simplified RAG evaluation scoring module.

Provides lightweight, local implementations of common RAG evaluation metrics
(faithfulness, relevance, answer correctness) that do NOT require an LLM judge
or async infrastructure.

These are heuristic approximations based on token/keyword overlap. For
production-grade RAG evaluation with LLM-as-judge, consider the full ragas
library with appropriate LLM configuration.
"""

import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens, removing punctuation.

    Args:
        text: Input text string.

    Returns:
        List of lowercase word tokens.
    """
    return re.findall(r'\b\w+\b', text.lower())


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using basic punctuation rules.

    Args:
        text: Input text string.

    Returns:
        List of non-empty sentence strings.
    """
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


def _jaccard_similarity(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Compute Jaccard similarity between two token lists.

    Args:
        tokens_a: First list of tokens.
        tokens_b: Second list of tokens.

    Returns:
        Jaccard similarity coefficient (0.0-1.0).
    """
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def _token_f1(candidate_tokens: list[str], reference_tokens: list[str]) -> float:
    """Compute token-level F1 score between candidate and reference.

    Args:
        candidate_tokens: Tokens from the candidate text.
        reference_tokens: Tokens from the reference text.

    Returns:
        F1 score (0.0-1.0) based on token overlap.
    """
    if not candidate_tokens or not reference_tokens:
        return 0.0

    candidate_counts = Counter(candidate_tokens)
    reference_counts = Counter(reference_tokens)

    # Count matching tokens (min of counts for each token)
    common = candidate_counts & reference_counts
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / sum(candidate_counts.values())
    recall = num_common / sum(reference_counts.values())

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def faithfulness_score(response: str, context: str) -> float:
    """Measure how much of the response is supported by the context.

    Splits the response into sentences and checks what fraction have
    sufficient keyword overlap (Jaccard similarity > 0.3) with the context.

    Args:
        response: The model's response string.
        context: The retrieved context/passage string.

    Returns:
        Fraction of response sentences supported by context (0.0-1.0).
        Returns 0.0 if either input is empty.
    """
    if not response or not response.strip() or not context or not context.strip():
        return 0.0

    sentences = _split_sentences(response)
    if not sentences:
        return 0.0

    context_tokens = _tokenize(context)
    if not context_tokens:
        return 0.0

    supported = 0
    for sentence in sentences:
        sentence_tokens = _tokenize(sentence)
        if not sentence_tokens:
            continue
        similarity = _jaccard_similarity(sentence_tokens, context_tokens)
        if similarity > 0.3:
            supported += 1

    return supported / len(sentences)


def relevance_score(response: str, question: str) -> float:
    """Measure if the response addresses the question.

    Computes normalized keyword overlap between the question and response.

    Args:
        response: The model's response string.
        question: The original question/query string.

    Returns:
        Normalized overlap score (0.0-1.0).
        Returns 0.0 if either input is empty.
    """
    if not response or not response.strip() or not question or not question.strip():
        return 0.0

    response_tokens = _tokenize(response)
    question_tokens = _tokenize(question)

    if not response_tokens or not question_tokens:
        return 0.0

    # What fraction of question tokens appear in the response
    question_set = set(question_tokens)
    response_set = set(response_tokens)

    if not question_set:
        return 0.0

    overlap = question_set & response_set
    return len(overlap) / len(question_set)


def answer_correctness(response: str, ground_truth: str) -> float:
    """Compute token-overlap F1 between response and ground truth.

    Uses token-level precision and recall to produce an F1 score measuring
    how well the response matches the expected ground truth.

    Args:
        response: The model's response string.
        ground_truth: The expected/correct answer string.

    Returns:
        Token F1 score (0.0-1.0).
        Returns 0.0 if either input is empty.
    """
    if not response or not response.strip() or not ground_truth or not ground_truth.strip():
        return 0.0

    response_tokens = _tokenize(response)
    truth_tokens = _tokenize(ground_truth)

    return _token_f1(response_tokens, truth_tokens)


def rag_composite_score(
    response: str,
    context: str,
    question: str,
    ground_truth: str,
) -> float:
    """Compute weighted composite RAG evaluation score.

    Combines faithfulness (0.3), relevance (0.3), and answer correctness (0.4)
    into a single score.

    Args:
        response: The model's response string.
        context: The retrieved context/passage string.
        question: The original question/query string.
        ground_truth: The expected/correct answer string.

    Returns:
        Weighted composite score (0.0-1.0).
        Returns 0.0 if response is empty.
    """
    if not response or not response.strip():
        return 0.0

    faith = faithfulness_score(response, context)
    relevance = relevance_score(response, question)
    correctness = answer_correctness(response, ground_truth)

    return 0.3 * faith + 0.3 * relevance + 0.4 * correctness
