"""Unit tests for src/model_provider.py.

All HTTP calls are mocked — no running Ollama instance required.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.model_provider import OllamaClient, OllamaConnectionError

# --- Initialization ---


def test_init_defaults():
    """OllamaClient uses correct default values."""
    client = OllamaClient()
    assert client.base_url == "http://localhost:11434"
    assert client.timeout == 120
    assert client.retries == 3
    assert client.backoff_base == 2


def test_init_custom_params():
    """OllamaClient accepts and stores custom parameters."""
    client = OllamaClient(
        base_url="http://myhost:9999/",
        timeout=60,
        retries=5,
        backoff_base=3,
    )
    assert client.base_url == "http://myhost:9999", "Trailing slash should be stripped"
    assert client.timeout == 60
    assert client.retries == 5
    assert client.backoff_base == 3


# --- health_check ---


@patch("src.model_provider.requests.get")
def test_health_check_success(mock_get):
    """health_check returns True when server responds 200."""
    mock_get.return_value = MagicMock(status_code=200)
    client = OllamaClient()
    assert client.health_check() is True
    mock_get.assert_called_once_with(
        "http://localhost:11434/api/tags", timeout=120
    )


@patch("src.model_provider.requests.get")
def test_health_check_non_200(mock_get):
    """health_check returns False on non-200 status."""
    mock_get.return_value = MagicMock(status_code=503)
    client = OllamaClient()
    assert client.health_check() is False


@patch("src.model_provider.requests.get")
def test_health_check_connection_error(mock_get):
    """health_check returns False when ConnectionError is raised."""
    mock_get.side_effect = requests.ConnectionError("refused")
    client = OllamaClient()
    assert client.health_check() is False


@patch("src.model_provider.requests.get")
def test_health_check_timeout(mock_get):
    """health_check returns False when request times out."""
    mock_get.side_effect = requests.Timeout("timed out")
    client = OllamaClient()
    assert client.health_check() is False


# --- list_models ---


@patch("src.model_provider.time.sleep", return_value=None)
@patch("src.model_provider.requests.get")
def test_list_models_success(mock_get, _mock_sleep):
    """list_models returns parsed model names from response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "models": [
            {"name": "llama3:8b", "size": 4000000000},
            {"name": "mistral:7b", "size": 3500000000},
        ]
    }
    mock_get.return_value = mock_response

    client = OllamaClient()
    models = client.list_models()

    assert models == ["llama3:8b", "mistral:7b"], "Should extract model names"
    mock_get.assert_called_once()


@patch("src.model_provider.time.sleep", return_value=None)
@patch("src.model_provider.requests.get")
def test_list_models_empty(mock_get, _mock_sleep):
    """list_models returns empty list when no models available."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"models": []}
    mock_get.return_value = mock_response

    client = OllamaClient()
    assert client.list_models() == []


@patch("src.model_provider.time.sleep", return_value=None)
@patch("src.model_provider.requests.get")
def test_list_models_connection_error_raises(mock_get, _mock_sleep):
    """list_models raises OllamaConnectionError after retries exhausted."""
    mock_get.side_effect = requests.ConnectionError("refused")
    client = OllamaClient(retries=2)

    with pytest.raises(OllamaConnectionError):
        client.list_models()

    # Initial attempt + 2 retries = 3 total calls
    assert mock_get.call_count == 3


# --- generate ---


@patch("src.model_provider.time.sleep", return_value=None)
@patch("src.model_provider.requests.post")
def test_generate_success(mock_post, _mock_sleep):
    """generate returns correct dict from valid response."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "response": "Hello, world!",
        "total_duration": 5000000000,
        "eval_count": 42,
        "extra_field": "ignored",
    }
    mock_post.return_value = mock_response

    client = OllamaClient()
    result = client.generate("llama3:8b", "Say hello")

    assert result == {
        "response": "Hello, world!",
        "total_duration": 5000000000,
        "eval_count": 42,
    }, "Should return only response, total_duration, eval_count"

    mock_post.assert_called_once_with(
        "http://localhost:11434/api/generate",
        json={"model": "llama3:8b", "prompt": "Say hello", "stream": False},
        timeout=120,
    )


@patch("src.model_provider.time.sleep", return_value=None)
@patch("src.model_provider.requests.post")
def test_generate_with_custom_timeout(mock_post, _mock_sleep):
    """generate uses per-call timeout override when provided."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "response": "OK",
        "total_duration": 1000,
        "eval_count": 1,
    }
    mock_post.return_value = mock_response

    client = OllamaClient(timeout=120)
    client.generate("model", "prompt", timeout=30)

    _, kwargs = mock_post.call_args
    assert kwargs["timeout"] == 30, "Should use per-call timeout, not instance default"


@patch("src.model_provider.time.sleep", return_value=None)
@patch("src.model_provider.requests.post")
def test_generate_timeout_raises(mock_post, _mock_sleep):
    """generate raises TimeoutError on request timeout (no retry)."""
    mock_post.side_effect = requests.Timeout("timed out")

    client = OllamaClient(retries=3)
    with pytest.raises(TimeoutError, match="timed out"):
        client.generate("llama3:8b", "Hello")

    # TimeoutError is NOT retried — should be called only once
    assert mock_post.call_count == 1, "TimeoutError must not be retried"


@patch("src.model_provider.time.sleep", return_value=None)
@patch("src.model_provider.requests.post")
def test_generate_connection_error_retries(mock_post, _mock_sleep):
    """generate retries on connection errors then raises."""
    mock_post.side_effect = requests.ConnectionError("refused")

    client = OllamaClient(retries=2)
    with pytest.raises(OllamaConnectionError):
        client.generate("model", "prompt")

    # 1 initial + 2 retries = 3
    assert mock_post.call_count == 3, "Should attempt initial + retries times"


# --- Retry logic ---


@patch("src.model_provider.time.sleep", return_value=None)
@patch("src.model_provider.requests.get")
def test_retry_succeeds_on_second_attempt(mock_get, _mock_sleep):
    """Retry logic recovers when second attempt succeeds."""
    fail_response = requests.ConnectionError("refused")
    success_response = MagicMock()
    success_response.raise_for_status = MagicMock()
    success_response.json.return_value = {"models": [{"name": "phi:latest"}]}

    mock_get.side_effect = [fail_response, success_response]

    client = OllamaClient(retries=3)
    models = client.list_models()

    assert models == ["phi:latest"]
    assert mock_get.call_count == 2, "Should succeed on second try"


@patch("src.model_provider.time.sleep", return_value=None)
@patch("src.model_provider.requests.get")
def test_retry_exponential_backoff(mock_get, mock_sleep):
    """Retry sleeps with exponential backoff (base^attempt)."""
    mock_get.side_effect = requests.ConnectionError("refused")

    client = OllamaClient(retries=3, backoff_base=2)
    with pytest.raises(OllamaConnectionError):
        client.list_models()

    # Backoff sleeps: 2^0=1, 2^1=2, 2^2=4 (for attempts 0, 1, 2 before final fail)
    sleep_calls = [call.args[0] for call in mock_sleep.call_args_list]
    assert sleep_calls == [1, 2, 4], (
        f"Expected exponential backoff [1, 2, 4], got {sleep_calls}"
    )


@patch("src.model_provider.time.sleep", return_value=None)
@patch("src.model_provider.requests.post")
def test_retry_http_error_not_retried(mock_post, _mock_sleep):
    """HTTPError from raise_for_status is not retried."""
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
    mock_post.return_value = mock_response

    client = OllamaClient(retries=3)
    with pytest.raises(requests.HTTPError):
        client.generate("model", "prompt")

    assert mock_post.call_count == 1, "HTTPError must not be retried"
