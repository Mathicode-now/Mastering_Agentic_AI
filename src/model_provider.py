"""Ollama HTTP client for model evaluation framework.

Provides OllamaClient for interacting with a local Ollama instance,
including health checks, model listing, and text generation with
retry logic and exponential backoff.
"""

import logging
import time
from collections.abc import Callable
from typing import Any

import requests

logger = logging.getLogger(__name__)


class OllamaConnectionError(Exception):
    """Raised when unable to connect to the Ollama server."""



class OllamaClient:
    """HTTP client for the Ollama API.

    Supports health checks, model listing, and text generation
    with configurable timeouts and retry behavior.

    Args:
        base_url: Base URL of the Ollama server.
        timeout: Default request timeout in seconds.
        retries: Number of retry attempts for failed requests.
        backoff_base: Base for exponential backoff calculation.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 120,
        retries: int = 3,
        backoff_base: float = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.backoff_base = backoff_base

    def health_check(self) -> bool:
        """Check if the Ollama server is reachable.

        Sends a GET request to /api/tags and returns True if the
        server responds with HTTP 200.

        Returns:
            True if server is healthy, False otherwise.
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags", timeout=self.timeout
            )
            return response.status_code == 200
        except requests.ConnectionError:
            return False
        except requests.Timeout:
            return False

    def list_models(self) -> list[str]:
        """List all available models on the Ollama server.

        Returns:
            List of model name strings (e.g., ['llama3:8b', 'mistral:7b']).

        Raises:
            OllamaConnectionError: If unable to connect to the server.
        """

        def _fetch() -> list[str]:
            try:
                response = requests.get(
                    f"{self.base_url}/api/tags", timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            except requests.ConnectionError as e:
                raise OllamaConnectionError(
                    f"Cannot connect to Ollama at {self.base_url}: {e}"
                ) from e
            except requests.Timeout as e:
                raise OllamaConnectionError(
                    f"Timeout connecting to Ollama at {self.base_url}: {e}"
                ) from e

        return self._retry(_fetch, self.retries, self.backoff_base)

    def generate(
        self, model: str, prompt: str, timeout: float | None = None
    ) -> dict[str, Any]:
        """Generate a completion from the specified model.

        Sends a synchronous (non-streaming) generation request to Ollama.

        Args:
            model: Model identifier (e.g., 'qwen2.5-coder:7b').
            prompt: The input prompt text.
            timeout: Override timeout in seconds. Uses instance default if None.

        Returns:
            Dict with keys:
                - response (str): Generated text.
                - total_duration (int): Total duration in nanoseconds.
                - eval_count (int): Number of tokens evaluated.

        Raises:
            TimeoutError: If the request exceeds the timeout.
            OllamaConnectionError: If unable to connect to the server.
        """
        request_timeout = timeout if timeout is not None else self.timeout

        def _call() -> dict[str, Any]:
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                    timeout=request_timeout,
                )
                response.raise_for_status()
                data = response.json()
                return {
                    "response": data.get("response", ""),
                    "total_duration": data.get("total_duration", 0),
                    "eval_count": data.get("eval_count", 0),
                }
            except requests.Timeout as e:
                raise TimeoutError(
                    f"Generation timed out after {request_timeout}s "
                    f"for model '{model}'"
                ) from e
            except requests.ConnectionError as e:
                raise OllamaConnectionError(
                    f"Cannot connect to Ollama at {self.base_url}: {e}"
                ) from e

        return self._retry(_call, self.retries, self.backoff_base)

    def _retry(
        self,
        fn: Callable[[], Any],
        retries: int,
        backoff_base: float,
    ) -> Any:
        """Execute a function with exponential backoff retry logic.

        Retries on OllamaConnectionError. Does not retry on TimeoutError
        or other exceptions.

        Args:
            fn: Callable to execute.
            retries: Maximum number of retry attempts.
            backoff_base: Base for exponential backoff (sleep = base^attempt).

        Returns:
            The return value of fn() on success.

        Raises:
            The last exception raised by fn() after all retries exhausted.
        """
        last_exception: Exception | None = None

        for attempt in range(retries + 1):
            try:
                return fn()
            except OllamaConnectionError as e:
                last_exception = e
                if attempt < retries:
                    sleep_time = backoff_base**attempt
                    logger.warning(
                        "Attempt %d/%d failed: %s. Retrying in %.1fs...",
                        attempt + 1,
                        retries + 1,
                        e,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(
                        "All %d attempts failed. Last error: %s",
                        retries + 1,
                        e,
                    )
            except (TimeoutError, requests.HTTPError):
                raise

        raise last_exception  # type: ignore[misc]
