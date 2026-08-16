"""Unit tests for src/runner.py.

All OllamaClient calls are mocked — no running Ollama instance required.
Uses tmp_path for database files and real tasks/ YAML for integration-style tests.
"""

import os
from unittest.mock import patch

import pytest
import yaml

from src.runner import EvalRunner

# --- Fixtures ---


@pytest.fixture
def models_config(tmp_path):
    """Create a minimal models config YAML file."""
    config = {
        "models": [
            {"id": "test-model:7b", "name": "Test Model", "params": "7B", "ram_gb": 4.0}
        ]
    }
    path = tmp_path / "models.yaml"
    path.write_text(yaml.dump(config))
    return str(path)


@pytest.fixture
def timeouts_config(tmp_path):
    """Create a minimal timeouts config YAML file."""
    config = {
        "defaults": {
            "generation_timeout_seconds": 60,
            "retry_count": 1,
            "retry_backoff_base": 2,
        },
        "per_field": {
            "code_generation": {"generation_timeout_seconds": 180},
        },
    }
    path = tmp_path / "timeouts.yaml"
    path.write_text(yaml.dump(config))
    return str(path)


@pytest.fixture
def runner(tmp_path, models_config, timeouts_config):
    """Create an EvalRunner instance with tmp_path database."""
    db_path = str(tmp_path / "test_results.db")
    return EvalRunner(
        models_config_path=models_config,
        timeouts_config_path=timeouts_config,
        db_path=db_path,
        ollama_url="http://localhost:11434",
    )


# --- test_eval_runner_init ---


def test_eval_runner_init(tmp_path, models_config, timeouts_config):
    """EvalRunner loads configs and creates client + store on init."""
    db_path = str(tmp_path / "init_test.db")
    r = EvalRunner(
        models_config_path=models_config,
        timeouts_config_path=timeouts_config,
        db_path=db_path,
        ollama_url="http://testhost:9999",
    )
    assert r.models_config["models"][0]["id"] == "test-model:7b"
    assert r.timeouts_config["defaults"]["generation_timeout_seconds"] == 60
    assert r.client.base_url == "http://testhost:9999"
    assert r.store.db_path == db_path


# --- test_load_tasks_returns_list ---


def test_load_tasks_returns_list(runner):
    """load_tasks returns a list of task dicts from real tasks/ YAML."""
    # Use the real code_generation prompts file
    original_cwd = os.getcwd()
    os.chdir(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )  # project root
    try:
        tasks = runner.load_tasks("code_generation")
        assert isinstance(tasks, list)
        assert len(tasks) > 0
        # Each task should have id, prompt, expected_output
        assert "id" in tasks[0]
        assert "prompt" in tasks[0]
        assert "expected_output" in tasks[0]
    finally:
        os.chdir(original_cwd)


# --- test_get_timeout ---


def test_get_timeout_per_field(runner):
    """get_timeout returns per-field override when configured."""
    assert runner.get_timeout("code_generation") == 180.0


def test_get_timeout_default_fallback(runner):
    """get_timeout returns default when field has no override."""
    assert runner.get_timeout("unknown_field") == 60.0


# --- test_run_field ---


@patch("src.runner.OllamaClient.health_check", return_value=True)
@patch("src.runner.OllamaClient.generate")
def test_run_field_calls_generate_for_each_model_task(
    mock_generate, mock_health, runner, tmp_path
):
    """run_field calls generate for each model × task combination."""
    # Create a minimal tasks file
    tasks_dir = tmp_path / "tasks" / "test_field"
    tasks_dir.mkdir(parents=True)
    prompts = {
        "field": "test_field",
        "scoring": "exact_match",
        "tasks": [
            {"id": "t1", "prompt": "say hello", "expected_output": "hello"},
            {"id": "t2", "prompt": "say bye", "expected_output": "bye"},
        ],
    }
    (tasks_dir / "prompts.yaml").write_text(yaml.dump(prompts))

    mock_generate.return_value = {"response": "hello", "eval_count": 10}

    original_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        run_id = runner.run_field(
            field="test_field", model_ids=["test-model:7b"], run_name="test_run"
        )
    finally:
        os.chdir(original_cwd)

    # 1 model × 2 tasks = 2 calls
    assert mock_generate.call_count == 2
    assert isinstance(run_id, int)


@patch("src.runner.OllamaClient.health_check", return_value=True)
@patch("src.runner.OllamaClient.generate")
def test_run_field_handles_timeout(mock_generate, mock_health, runner, tmp_path):
    """run_field handles TimeoutError gracefully with score 0.0."""
    tasks_dir = tmp_path / "tasks" / "test_field"
    tasks_dir.mkdir(parents=True)
    prompts = {
        "field": "test_field",
        "scoring": "exact_match",
        "tasks": [
            {"id": "t1", "prompt": "say hello", "expected_output": "hello"},
        ],
    }
    (tasks_dir / "prompts.yaml").write_text(yaml.dump(prompts))

    mock_generate.side_effect = TimeoutError("timed out")

    original_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        run_id = runner.run_field(
            field="test_field", model_ids=["test-model:7b"], run_name="timeout_run"
        )
    finally:
        os.chdir(original_cwd)

    # Run completes (doesn't crash) with run_id
    assert isinstance(run_id, int)
    # Verify result was stored with score 0.0
    results = runner.store.get_run_results(run_id)
    assert len(results) == 1
    assert results[0]["score"] == 0.0
    assert "TIMEOUT" in results[0]["response"]


@patch("src.runner.OllamaClient.health_check", return_value=True)
@patch("src.runner.OllamaClient.generate")
def test_run_field_scores_results(mock_generate, mock_health, runner, tmp_path):
    """run_field scores responses correctly using configured scorer."""
    tasks_dir = tmp_path / "tasks" / "test_field"
    tasks_dir.mkdir(parents=True)
    prompts = {
        "field": "test_field",
        "scoring": "exact_match",
        "tasks": [
            {"id": "t1", "prompt": "say hello", "expected_output": "hello"},
            {"id": "t2", "prompt": "say bye", "expected_output": "bye"},
        ],
    }
    (tasks_dir / "prompts.yaml").write_text(yaml.dump(prompts))

    # First task matches, second doesn't
    mock_generate.side_effect = [
        {"response": "hello", "eval_count": 5},
        {"response": "wrong answer", "eval_count": 8},
    ]

    original_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        run_id = runner.run_field(
            field="test_field", model_ids=["test-model:7b"], run_name="score_run"
        )
    finally:
        os.chdir(original_cwd)

    results = runner.store.get_run_results(run_id)
    scores = {r["task_id"]: r["score"] for r in results}
    assert scores["t1"] == 1.0
    assert scores["t2"] == 0.0


@patch("src.runner.OllamaClient.health_check", return_value=True)
@patch("src.runner.OllamaClient.generate")
@patch("src.runner.ResultsStore.save_result")
def test_run_field_saves_to_store(
    mock_save, mock_generate, mock_health, tmp_path
):
    """run_field calls ResultsStore.save_result for each model × task."""
    # Need a fresh runner that uses the mocked store
    models_config = {
        "models": [{"id": "m1", "name": "M1", "params": "7B", "ram_gb": 4.0}]
    }
    timeouts_config = {
        "defaults": {"generation_timeout_seconds": 60, "retry_count": 1, "retry_backoff_base": 2},
        "per_field": {},
    }
    models_path = tmp_path / "models.yaml"
    models_path.write_text(yaml.dump(models_config))
    timeouts_path = tmp_path / "timeouts.yaml"
    timeouts_path.write_text(yaml.dump(timeouts_config))

    db_path = str(tmp_path / "store_test.db")
    r = EvalRunner(
        models_config_path=str(models_path),
        timeouts_config_path=str(timeouts_path),
        db_path=db_path,
    )

    tasks_dir = tmp_path / "tasks" / "test_field"
    tasks_dir.mkdir(parents=True)
    prompts = {
        "field": "test_field",
        "scoring": "exact_match",
        "tasks": [
            {"id": "t1", "prompt": "p1", "expected_output": "e1"},
            {"id": "t2", "prompt": "p2", "expected_output": "e2"},
        ],
    }
    (tasks_dir / "prompts.yaml").write_text(yaml.dump(prompts))

    mock_generate.return_value = {"response": "e1", "eval_count": 5}

    original_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        r.run_field(field="test_field", model_ids=["m1"], run_name="save_run")
    finally:
        os.chdir(original_cwd)

    # 1 model × 2 tasks = 2 save_result calls
    assert mock_save.call_count == 2
    # Verify save_result was called with expected kwargs
    first_call_kwargs = mock_save.call_args_list[0].kwargs
    assert first_call_kwargs["model_id"] == "m1"
    assert first_call_kwargs["task_id"] == "t1"
    assert first_call_kwargs["task_field"] == "test_field"
