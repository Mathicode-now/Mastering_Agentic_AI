"""Preflight check: verify Ollama connectivity and model availability.

Run from project root:
    python scripts/preflight.py

Exits 0 if all configured models are available, 1 otherwise.
"""

import sys
from pathlib import Path

# Allow running as `python scripts/preflight.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.model_provider import OllamaClient, OllamaConnectionError

# ANSI color codes
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    """Print a green checkmark message."""
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    """Print a red cross message."""
    print(f"  {RED}✗{RESET} {msg}")


def warn(msg: str) -> None:
    """Print a yellow warning message."""
    print(f"  {YELLOW}⚠{RESET} {msg}")


def main() -> None:
    """Run preflight checks for Ollama connectivity and model availability."""
    config_path = Path(__file__).resolve().parent.parent / "config" / "models.yaml"

    # Load configuration
    print(f"\n{BOLD}Preflight Check{RESET}")
    print("=" * 40)

    if not config_path.exists():
        fail(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    configured_models = [m["id"] for m in config.get("models", [])]
    if not configured_models:
        fail("No models configured in models.yaml")
        sys.exit(1)

    # Create client
    client = OllamaClient()

    # Health check
    print(f"\n{BOLD}Ollama Server{RESET}")
    healthy = client.health_check()
    if healthy:
        ok(f"Connected to {client.base_url}")
    else:
        fail(f"Cannot reach Ollama at {client.base_url}")
        print(f"\n  {YELLOW}Hint: Is Ollama running? Try: ollama serve{RESET}")
        sys.exit(1)

    # List available models
    print(f"\n{BOLD}Model Availability{RESET}")
    try:
        available_models = client.list_models()
    except OllamaConnectionError as e:
        fail(f"Failed to list models: {e}")
        sys.exit(1)

    if not available_models:
        warn("No models pulled in Ollama")

    # Check each configured model
    available_count = 0
    for model_id in configured_models:
        # Match by exact name or by prefix (Ollama may append :latest)
        found = any(
            m == model_id or m.startswith(model_id.split(":")[0] + ":")
            for m in available_models
        )
        if found:
            ok(model_id)
            available_count += 1
        else:
            fail(f"{model_id} — not found (pull with: ollama pull {model_id})")

    # Summary
    total = len(configured_models)
    print(f"\n{BOLD}Summary{RESET}: {available_count}/{total} models available")

    if available_count == total:
        print(f"{GREEN}All checks passed.{RESET}\n")
        sys.exit(0)
    else:
        missing = total - available_count
        print(f"{RED}{missing} model(s) missing.{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
