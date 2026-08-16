"""Task runner that orchestrates model × task evaluation.

Loads model configurations, iterates over task fields and prompts,
invokes the Ollama API, scores responses, and persists results.
"""

import logging
import os
import time
from datetime import UTC, datetime

import yaml

from src.model_provider import OllamaClient, OllamaConnectionError
from src.results_store import ResultsStore
from src.scoring.exact_match import get_scorer

logger = logging.getLogger(__name__)


class EvalRunner:
    """Orchestrates model evaluation across task fields.

    Loads configurations, drives generation requests, scores results,
    and stores everything in the results database.

    Args:
        models_config_path: Path to the models YAML config file.
        timeouts_config_path: Path to the timeouts YAML config file.
        db_path: Path to the SQLite results database.
        ollama_url: Base URL for the Ollama server.
    """

    def __init__(
        self,
        models_config_path: str = "config/models.yaml",
        timeouts_config_path: str = "config/timeouts.yaml",
        db_path: str = "results.db",
        ollama_url: str = "http://localhost:11434",
    ) -> None:
        with open(models_config_path) as f:
            self.models_config = yaml.safe_load(f)

        with open(timeouts_config_path) as f:
            self.timeouts_config = yaml.safe_load(f)

        self.client = OllamaClient(
            base_url=ollama_url,
            timeout=self.timeouts_config["defaults"]["generation_timeout_seconds"],
            retries=self.timeouts_config["defaults"].get("retry_count", 3),
            backoff_base=self.timeouts_config["defaults"].get("retry_backoff_base", 2),
        )
        self.store = ResultsStore(db_path=db_path)

    def load_tasks(self, field: str) -> list[dict]:
        """Load task prompts for a given evaluation field.

        Args:
            field: The task field name (e.g., 'code_generation').

        Returns:
            List of task dictionaries from the prompts YAML file.

        Raises:
            FileNotFoundError: If the prompts file does not exist.
        """
        prompts_path = os.path.join("tasks", field, "prompts.yaml")
        with open(prompts_path) as f:
            data = yaml.safe_load(f)
        return data.get("tasks", [])

    def get_timeout(self, field: str) -> float:
        """Get the generation timeout for a specific field.

        Falls back to the default timeout if the field has no override.

        Args:
            field: The task field name.

        Returns:
            Timeout in seconds.
        """
        per_field = self.timeouts_config.get("per_field", {})
        if field in per_field:
            return float(
                per_field[field].get(
                    "generation_timeout_seconds",
                    self.timeouts_config["defaults"]["generation_timeout_seconds"],
                )
            )
        return float(self.timeouts_config["defaults"]["generation_timeout_seconds"])

    def run_field(
        self,
        field: str,
        model_ids: list[str] | None = None,
        run_name: str | None = None,
    ) -> int:
        """Run evaluation for a single task field across specified models.

        Creates a run, iterates over all model × task combinations,
        generates responses, scores them, and stores results.

        Args:
            field: The task field to evaluate (e.g., 'code_generation').
            model_ids: List of model identifiers to evaluate. If None, uses
                all models from the config file.
            run_name: Human-readable run name. Auto-generated if None.

        Returns:
            The run_id for the created evaluation run.

        Raises:
            OllamaConnectionError: If the Ollama server is unreachable.
            FileNotFoundError: If the task prompts file is missing.
        """
        # Verify Ollama connectivity before starting
        if not self.client.health_check():
            raise OllamaConnectionError(
                f"Cannot reach Ollama server at {self.client.base_url}"
            )

        tasks = self.load_tasks(field)
        prompts_path = os.path.join("tasks", field, "prompts.yaml")
        with open(prompts_path) as f:
            field_config = yaml.safe_load(f)
        scoring_type = field_config.get("scoring", "exact_match")
        scorer = get_scorer(scoring_type)

        if model_ids is None:
            model_ids = [m["id"] for m in self.models_config["models"]]

        if run_name is None:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            run_name = f"{field}_{timestamp}"

        run_id = self.store.create_run(
            run_name=run_name,
            metadata={"field": field, "models": model_ids, "scoring": scoring_type},
        )

        timeout = self.get_timeout(field)

        for model_id in model_ids:
            for task in tasks:
                task_id = task["id"]
                prompt = task["prompt"]
                expected = task.get("expected_output", "")

                start = time.perf_counter()
                try:
                    result = self.client.generate(
                        model=model_id, prompt=prompt, timeout=timeout
                    )
                    response_text = result["response"]
                    token_count = result.get("eval_count", 0)
                except TimeoutError:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    logger.warning(
                        "Timeout for model=%s task=%s after %.0fms",
                        model_id,
                        task_id,
                        elapsed_ms,
                    )
                    response_text = f"[TIMEOUT after {timeout}s]"
                    token_count = 0
                    score = 0.0
                    self.store.save_result(
                        run_id=run_id,
                        model_id=model_id,
                        task_field=field,
                        task_id=task_id,
                        prompt=prompt,
                        response=response_text,
                        latency_ms=elapsed_ms,
                        token_count=token_count,
                        score=score,
                    )
                    print(
                        f"[{model_id}] {task_id} ... score={score:.1f} "
                        f"latency={elapsed_ms:.0f}ms"
                    )
                    continue
                except OllamaConnectionError:
                    # Re-raise connection errors — don't silently skip
                    raise
                except (OSError, RuntimeError, ValueError) as e:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    logger.warning(
                        "Error for model=%s task=%s: %s", model_id, task_id, e
                    )
                    response_text = f"[ERROR: {e}]"
                    token_count = 0
                    score = 0.0
                    self.store.save_result(
                        run_id=run_id,
                        model_id=model_id,
                        task_field=field,
                        task_id=task_id,
                        prompt=prompt,
                        response=response_text,
                        latency_ms=elapsed_ms,
                        token_count=token_count,
                        score=score,
                    )
                    print(
                        f"[{model_id}] {task_id} ... score={score:.1f} "
                        f"latency={elapsed_ms:.0f}ms"
                    )
                    continue

                elapsed_ms = (time.perf_counter() - start) * 1000

                # Score the response
                try:
                    scoring_mode = task.get("scoring_mode", "")
                    if scoring_type == "code_execution":
                        language = task.get("language", "python")
                        score = scorer(response_text, expected, language=language)
                    elif scoring_type == "rag":
                        context = task.get("context", "")
                        question = task.get("prompt", "")
                        score = scorer(response_text, context, question, expected)
                    else:
                        score = scorer(response_text, expected)

                    # Handle scoring_mode inversions
                    if scoring_mode == "not_contains":
                        from src.scoring.exact_match import contains_match

                        score = 1.0 - contains_match(response_text, expected)
                except (ValueError, TypeError, OSError, OverflowError, RuntimeError) as e:
                    logger.warning(
                        "Scoring error for model=%s task=%s: %s",
                        model_id,
                        task_id,
                        e,
                    )
                    score = 0.0

                self.store.save_result(
                    run_id=run_id,
                    model_id=model_id,
                    task_field=field,
                    task_id=task_id,
                    prompt=prompt,
                    response=response_text,
                    latency_ms=elapsed_ms,
                    token_count=token_count,
                    score=score,
                )

                print(
                    f"[{model_id}] {task_id} ... score={score:.1f} "
                    f"latency={elapsed_ms:.0f}ms"
                )

        return run_id

    def run_all_fields(self, model_ids: list[str] | None = None) -> int:
        """Run evaluation across all available task fields.

        Scans the tasks/ directory for subdirectories containing prompts.yaml,
        runs each field, and returns the run_id from the last field evaluated.

        Args:
            model_ids: List of model identifiers to evaluate. If None, uses
                all models from the config file.

        Returns:
            The run_id of the last evaluation run created.

        Raises:
            OllamaConnectionError: If the Ollama server is unreachable.
            FileNotFoundError: If no task fields are found.
        """
        fields = []
        tasks_dir = "tasks"
        for entry in sorted(os.listdir(tasks_dir)):
            prompts_path = os.path.join(tasks_dir, entry, "prompts.yaml")
            if os.path.isdir(os.path.join(tasks_dir, entry)) and os.path.isfile(
                prompts_path
            ):
                fields.append(entry)

        if not fields:
            raise FileNotFoundError("No task fields with prompts.yaml found in tasks/")

        logger.info("Running evaluation for fields: %s", fields)

        last_run_id = -1
        for field in fields:
            print(f"\n{'='*60}")
            print(f"  Field: {field}")
            print(f"{'='*60}\n")
            last_run_id = self.run_field(field=field, model_ids=model_ids)

        return last_run_id

    def get_results_table(self, run_id: int) -> list[dict]:
        """Get formatted results for display.

        Args:
            run_id: The run ID to retrieve results for.

        Returns:
            List of result dictionaries with keys: model_id, task_field,
            task_id, score, latency_ms, token_count.
        """
        raw_results = self.store.get_run_results(run_id)
        table = []
        for r in raw_results:
            table.append(
                {
                    "model_id": r["model_id"],
                    "task_field": r["task_field"],
                    "task_id": r["task_id"],
                    "score": r["score"],
                    "latency_ms": round(r["latency_ms"], 1),
                    "token_count": r["token_count"],
                }
            )
        return table
