"""Scoring module with multiple matching strategies for model evaluation."""

import os
import re
import subprocess
import tempfile
from collections.abc import Callable


def normalize_response(response: str) -> str:
    """Strip whitespace and remove markdown code fences, extracting content.

    Args:
        response: Raw response string potentially containing markdown formatting.

    Returns:
        Cleaned string with code fences removed and whitespace stripped.
    """
    text = response.strip()
    # Remove markdown code fences and extract inner content
    pattern = r"```(?:\w+)?\s*\n?(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    return text


def exact_match(response: str, expected: str) -> float:
    """Score 1.0 if stripped response exactly equals stripped expected, else 0.0.

    Args:
        response: The model's response string.
        expected: The expected/reference string.

    Returns:
        1.0 for exact match, 0.0 otherwise.
    """
    return 1.0 if response.strip() == expected.strip() else 0.0


def contains_match(response: str, expected: str) -> float:
    """Score 1.0 if expected substring is found in response (case-insensitive).

    Args:
        response: The model's response string.
        expected: The expected substring to search for.

    Returns:
        1.0 if expected is found within response, 0.0 otherwise.
    """
    return 1.0 if expected.strip().lower() in response.strip().lower() else 0.0


def _extract_code(response: str) -> str:
    """Extract code from markdown fenced code blocks.

    Looks for ```python ... ``` or ``` ... ``` patterns.
    Falls back to the raw response if no fences are found.

    Args:
        response: Response string potentially containing fenced code blocks.

    Returns:
        Extracted code string.
    """
    # Try ```python ... ``` first
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic ``` ... ```
    pattern = r"```\s*\n?(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # No fences found — use the raw response
    return response.strip()


def code_execution_score(
    response: str,
    expected: str,
    language: str = "python",
    timeout: float = 10.0,
) -> float:
    """Execute code extracted from response and compare output to expected.

    Extracts code from markdown code blocks in the response, writes it to a
    temp file, and runs it as a subprocess. Scores based on output comparison.

    Args:
        response: The model's response containing code (possibly in markdown fences).
        expected: The expected stdout output.
        language: Programming language (only 'python' supported currently).
        timeout: Maximum execution time in seconds.

    Returns:
        1.0 if stdout matches expected, 0.5 if code runs without error but
        output differs, 0.0 if code errors or times out.
    """
    if language != "python":
        return 0.0

    code = _extract_code(response)
    if not code:
        return 0.0

    tmp_dir = tempfile.mkdtemp()
    script_path = os.path.join(tmp_dir, "script.py")

    try:
        with open(script_path, "w") as f:
            f.write(code)

        result = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tmp_dir,
            check=False,
        )

        if result.returncode != 0:
            return 0.0

        actual_output = result.stdout.strip()
        expected_output = expected.strip()

        if actual_output == expected_output:
            return 1.0
        else:
            return 0.5

    except subprocess.TimeoutExpired:
        return 0.0
    except (OSError, subprocess.SubprocessError):
        return 0.0
    finally:
        # Clean up temp file
        try:
            os.remove(script_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass


def get_scorer(scoring_type: str) -> Callable:
    """Factory that returns the appropriate scoring function based on type string.

    Args:
        scoring_type: One of 'exact_match', 'contains', 'code_execution',
            'rouge', 'rouge_1', 'rouge_2', 'bert_score', or 'rag'.

    Returns:
        The corresponding scoring function.

    Note:
        Most scorers have signature (response, expected) -> float.
        The 'rag' scorer has signature (response, context, question, ground_truth) -> float.
        The 'code_execution' scorer has signature (response, expected, language=, timeout=).
        Callers (e.g. runner.py) must handle the different signatures.

    Raises:
        ValueError: If scoring_type is not recognized.
    """
    # Base scorers that are always available
    scorers: dict[str, Callable] = {
        "exact_match": exact_match,
        "contains": contains_match,
        "code_execution": code_execution_score,
    }

    if scoring_type in scorers:
        return scorers[scoring_type]

    # Lazy-imported scorers to avoid requiring heavy deps for basic usage
    if scoring_type in ("rouge", "rouge_1", "rouge_2"):
        from src.scoring.rouge import rouge_1_score, rouge_2_score, rouge_l_score

        rouge_scorers = {
            "rouge": rouge_l_score,
            "rouge_1": rouge_1_score,
            "rouge_2": rouge_2_score,
        }
        return rouge_scorers[scoring_type]

    if scoring_type == "bert_score":
        from src.scoring.semantic import bert_score_f1

        return bert_score_f1

    if scoring_type == "rag":
        from src.scoring.rag import rag_composite_score

        return rag_composite_score

    available = [
        "exact_match", "contains", "code_execution",
        "rouge", "rouge_1", "rouge_2", "bert_score", "rag",
    ]
    raise ValueError(
        f"Unknown scoring type: {scoring_type!r}. "
        f"Available types: {available}"
    )
