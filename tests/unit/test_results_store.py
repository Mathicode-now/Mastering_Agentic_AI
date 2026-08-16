"""Unit tests for src/results_store.py.

Uses pytest tmp_path fixture for isolated SQLite databases.
No network access required.
"""

import sqlite3

import pytest

from src.results_store import ResultsStore

# --- Database initialization ---


def test_creates_db_and_tables(tmp_path):
    """ResultsStore creates the database file with runs and results tables."""
    db_file = str(tmp_path / "test.db")
    ResultsStore(db_path=db_file)

    conn = sqlite3.connect(db_file)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()

    assert "runs" in tables, "Should create 'runs' table"
    assert "results" in tables, "Should create 'results' table"


def test_creates_index(tmp_path):
    """ResultsStore creates the idx_results_run_model index."""
    db_file = str(tmp_path / "test.db")
    ResultsStore(db_path=db_file)

    conn = sqlite3.connect(db_file)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    )
    indexes = [row[0] for row in cursor.fetchall()]
    conn.close()

    assert "idx_results_run_model" in indexes, "Should create composite index"


def test_idempotent_init(tmp_path):
    """Creating ResultsStore twice on same DB does not error."""
    db_file = str(tmp_path / "test.db")
    ResultsStore(db_path=db_file)
    # Should not raise
    store = ResultsStore(db_path=db_file)
    assert store.db_path == db_file


# --- create_run ---


def test_create_run_returns_id(tmp_path):
    """create_run returns an integer ID."""
    store = ResultsStore(db_path=str(tmp_path / "test.db"))
    run_id = store.create_run("test-run")
    assert isinstance(run_id, int)
    assert run_id >= 1


def test_create_run_incrementing_ids(tmp_path):
    """Successive create_run calls return incrementing IDs."""
    store = ResultsStore(db_path=str(tmp_path / "test.db"))
    id1 = store.create_run("run-1")
    id2 = store.create_run("run-2")
    id3 = store.create_run("run-3")

    assert id2 == id1 + 1, "IDs should increment sequentially"
    assert id3 == id2 + 1, "IDs should increment sequentially"


def test_create_run_with_metadata(tmp_path):
    """create_run stores metadata as JSON and list_runs deserializes it."""
    store = ResultsStore(db_path=str(tmp_path / "test.db"))
    meta = {"version": "1.0", "models": ["a", "b"]}
    run_id = store.create_run("meta-run", metadata=meta)

    runs = store.list_runs()
    run = next(r for r in runs if r["id"] == run_id)
    assert run["metadata"] == meta, "Metadata should round-trip through JSON"


def test_create_run_without_metadata(tmp_path):
    """create_run without metadata stores None."""
    store = ResultsStore(db_path=str(tmp_path / "test.db"))
    run_id = store.create_run("no-meta-run")

    runs = store.list_runs()
    run = next(r for r in runs if r["id"] == run_id)
    assert run["metadata"] is None


# --- save_result and get_run_results ---


def test_save_result_returns_id(tmp_path):
    """save_result returns an integer result ID."""
    store = ResultsStore(db_path=str(tmp_path / "test.db"))
    run_id = store.create_run("run")
    result_id = store.save_result(
        run_id=run_id,
        model_id="llama3:8b",
        task_field="coding",
        task_id="task-001",
        prompt="Hello",
        response="World",
        latency_ms=150.5,
        token_count=10,
        score=0.85,
    )
    assert isinstance(result_id, int)
    assert result_id >= 1


def test_save_and_get_results_roundtrip(tmp_path):
    """Saved results are retrievable via get_run_results with correct values."""
    store = ResultsStore(db_path=str(tmp_path / "test.db"))
    run_id = store.create_run("roundtrip-run")

    store.save_result(
        run_id=run_id,
        model_id="model-a",
        task_field="math",
        task_id="t1",
        prompt="2+2=?",
        response="4",
        latency_ms=100.0,
        token_count=5,
        score=1.0,
    )
    store.save_result(
        run_id=run_id,
        model_id="model-b",
        task_field="coding",
        task_id="t2",
        prompt="def hello",
        response="def hello(): pass",
        latency_ms=200.0,
        token_count=12,
        score=0.9,
    )

    results = store.get_run_results(run_id)
    assert len(results) == 2, "Should return all results for the run"

    r1 = results[0]
    assert r1["model_id"] == "model-a"
    assert r1["task_field"] == "math"
    assert r1["task_id"] == "t1"
    assert r1["prompt"] == "2+2=?"
    assert r1["response"] == "4"
    assert r1["latency_ms"] == 100.0
    assert r1["token_count"] == 5
    assert r1["score"] == 1.0


def test_get_run_results_isolates_runs(tmp_path):
    """get_run_results only returns results for the specified run."""
    store = ResultsStore(db_path=str(tmp_path / "test.db"))
    run1 = store.create_run("run-1")
    run2 = store.create_run("run-2")

    store.save_result(run1, "m1", "f", "t1", "p", "r", 100, 5)
    store.save_result(run2, "m2", "f", "t2", "p", "r", 200, 10)

    results1 = store.get_run_results(run1)
    results2 = store.get_run_results(run2)

    assert len(results1) == 1
    assert results1[0]["model_id"] == "m1"
    assert len(results2) == 1
    assert results2[0]["model_id"] == "m2"


def test_save_result_score_none(tmp_path):
    """save_result works with score=None (optional field)."""
    store = ResultsStore(db_path=str(tmp_path / "test.db"))
    run_id = store.create_run("run")
    store.save_result(run_id, "model", "field", "t1", "p", "r", 50, 3, score=None)

    results = store.get_run_results(run_id)
    assert results[0]["score"] is None


# --- get_model_summary ---


def test_get_model_summary_computes_averages(tmp_path):
    """get_model_summary returns correct avg_latency, avg_score, total_tokens."""
    store = ResultsStore(db_path=str(tmp_path / "test.db"))
    run_id = store.create_run("summary-run")

    store.save_result(run_id, "modelX", "f", "t1", "p", "r", 100.0, 10, score=0.8)
    store.save_result(run_id, "modelX", "f", "t2", "p", "r", 200.0, 20, score=1.0)
    store.save_result(run_id, "modelX", "f", "t3", "p", "r", 300.0, 30, score=0.6)

    summary = store.get_model_summary(run_id, "modelX")

    assert summary["avg_latency_ms"] == pytest.approx(200.0), (
        "Average of 100, 200, 300 should be 200"
    )
    assert summary["avg_score"] == pytest.approx(0.8), (
        "Average of 0.8, 1.0, 0.6 should be 0.8"
    )
    assert summary["total_tokens"] == 60, "Sum of 10, 20, 30 should be 60"


def test_get_model_summary_filters_by_model(tmp_path):
    """get_model_summary only aggregates results for the specified model."""
    store = ResultsStore(db_path=str(tmp_path / "test.db"))
    run_id = store.create_run("run")

    store.save_result(run_id, "modelA", "f", "t1", "p", "r", 100, 10, score=1.0)
    store.save_result(run_id, "modelB", "f", "t2", "p", "r", 900, 90, score=0.1)

    summary_a = store.get_model_summary(run_id, "modelA")
    assert summary_a["avg_latency_ms"] == pytest.approx(100.0)
    assert summary_a["total_tokens"] == 10


def test_get_model_summary_with_null_scores(tmp_path):
    """get_model_summary handles NULL scores (AVG ignores NULLs in SQL)."""
    store = ResultsStore(db_path=str(tmp_path / "test.db"))
    run_id = store.create_run("run")

    store.save_result(run_id, "m", "f", "t1", "p", "r", 100, 5, score=0.5)
    store.save_result(run_id, "m", "f", "t2", "p", "r", 200, 10, score=None)

    summary = store.get_model_summary(run_id, "m")
    # SQL AVG ignores NULL, so avg_score = 0.5 (only one non-null score)
    assert summary["avg_score"] == pytest.approx(0.5)
    assert summary["avg_latency_ms"] == pytest.approx(150.0)
    assert summary["total_tokens"] == 15


def test_get_model_summary_no_results(tmp_path):
    """get_model_summary returns None values when no matching results exist."""
    store = ResultsStore(db_path=str(tmp_path / "test.db"))
    run_id = store.create_run("run")

    summary = store.get_model_summary(run_id, "nonexistent")
    assert summary["avg_latency_ms"] is None
    assert summary["avg_score"] is None
    assert summary["total_tokens"] is None


# --- list_runs ---


def test_list_runs_empty(tmp_path):
    """list_runs returns empty list when no runs exist."""
    store = ResultsStore(db_path=str(tmp_path / "test.db"))
    assert store.list_runs() == []


def test_list_runs_returns_all(tmp_path):
    """list_runs returns all created runs in order."""
    store = ResultsStore(db_path=str(tmp_path / "test.db"))
    store.create_run("alpha")
    store.create_run("beta")
    store.create_run("gamma")

    runs = store.list_runs()
    assert len(runs) == 3
    names = [r["name"] for r in runs]
    assert names == ["alpha", "beta", "gamma"], "Runs should be ordered by ID"


def test_list_runs_includes_created_at(tmp_path):
    """list_runs includes a created_at timestamp."""
    store = ResultsStore(db_path=str(tmp_path / "test.db"))
    store.create_run("timestamped")

    runs = store.list_runs()
    assert runs[0]["created_at"] is not None, "created_at should have a value"
