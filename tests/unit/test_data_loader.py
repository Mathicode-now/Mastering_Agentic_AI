"""Unit tests for dashboard data_loader module."""

import pandas as pd
import pytest

from src.dashboard.data_loader import DashboardDataLoader
from src.results_store import ResultsStore


@pytest.fixture
def db_path(tmp_path):
    """Return a temporary database path."""
    return str(tmp_path / "test_results.db")


@pytest.fixture
def populated_db(db_path):
    """Create a DB with known test data and return (db_path, run_id)."""
    store = ResultsStore(db_path=db_path)
    run_id = store.create_run("test-run-1")

    # Model A: 2 fields, 2 tasks each
    store.save_result(
        run_id=run_id,
        model_id="model-a",
        task_field="summarization",
        task_id="task-1",
        prompt="Summarize this",
        response="Summary A1",
        latency_ms=100.0,
        token_count=50,
        score=0.9,
    )
    store.save_result(
        run_id=run_id,
        model_id="model-a",
        task_field="summarization",
        task_id="task-2",
        prompt="Summarize that",
        response="Summary A2",
        latency_ms=120.0,
        token_count=60,
        score=0.8,
    )
    store.save_result(
        run_id=run_id,
        model_id="model-a",
        task_field="code_generation",
        task_id="task-3",
        prompt="Write code",
        response="def hello(): pass",
        latency_ms=200.0,
        token_count=80,
        score=0.7,
    )

    # Model B: same fields
    store.save_result(
        run_id=run_id,
        model_id="model-b",
        task_field="summarization",
        task_id="task-1",
        prompt="Summarize this",
        response="Summary B1",
        latency_ms=150.0,
        token_count=45,
        score=0.85,
    )
    store.save_result(
        run_id=run_id,
        model_id="model-b",
        task_field="code_generation",
        task_id="task-3",
        prompt="Write code",
        response="def hello(): return",
        latency_ms=250.0,
        token_count=70,
        score=0.6,
    )

    return db_path, run_id


class TestGetAllResults:
    def test_get_all_results_returns_dataframe(self, populated_db):
        db_path, _run_id = populated_db
        loader = DashboardDataLoader(db_path=db_path)

        df = loader.get_all_results()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        expected_cols = [
            "model_id", "task_field", "task_id", "prompt",
            "response", "latency_ms", "token_count", "score", "created_at",
        ]
        assert list(df.columns) == expected_cols
        assert set(df["model_id"].unique()) == {"model-a", "model-b"}

    def test_get_all_results_filtered_by_run(self, db_path):
        store = ResultsStore(db_path=db_path)
        run1 = store.create_run("run-1")
        run2 = store.create_run("run-2")

        store.save_result(run1, "m1", "field1", "t1", "p", "r", 100.0, 10, 0.5)
        store.save_result(run2, "m2", "field2", "t2", "p", "r", 200.0, 20, 0.6)

        loader = DashboardDataLoader(db_path=db_path)
        df = loader.get_all_results(run_id=run1)

        assert len(df) == 1
        assert df.iloc[0]["model_id"] == "m1"

    def test_get_all_results_empty_db(self, db_path):
        # Initialize DB but add no data
        ResultsStore(db_path=db_path)
        loader = DashboardDataLoader(db_path=db_path)

        df = loader.get_all_results()

        assert isinstance(df, pd.DataFrame)
        assert df.empty
        expected_cols = [
            "model_id", "task_field", "task_id", "prompt",
            "response", "latency_ms", "token_count", "score", "created_at",
        ]
        assert list(df.columns) == expected_cols


class TestGetModelFieldSummary:
    def test_get_model_field_summary_pivot(self, populated_db):
        db_path, run_id = populated_db
        loader = DashboardDataLoader(db_path=db_path)

        df = loader.get_model_field_summary(run_id=run_id)

        # Should be a pivot: index=model_id, columns=task_field
        assert "model-a" in df.index
        assert "model-b" in df.index
        assert "summarization" in df.columns
        assert "code_generation" in df.columns

        # model-a summarization: avg of 0.9 and 0.8 = 0.85
        assert df.loc["model-a", "summarization"] == pytest.approx(0.85)
        # model-a code_generation: 0.7
        assert df.loc["model-a", "code_generation"] == pytest.approx(0.7)
        # model-b summarization: 0.85
        assert df.loc["model-b", "summarization"] == pytest.approx(0.85)

    def test_get_model_field_summary_empty(self, db_path):
        ResultsStore(db_path=db_path)
        loader = DashboardDataLoader(db_path=db_path)

        df = loader.get_model_field_summary()
        assert df.empty


class TestGetLatencySummary:
    def test_get_latency_summary_pivot(self, populated_db):
        db_path, run_id = populated_db
        loader = DashboardDataLoader(db_path=db_path)

        df = loader.get_latency_summary(run_id=run_id)

        assert "model-a" in df.index
        assert "model-b" in df.index
        assert "summarization" in df.columns
        assert "code_generation" in df.columns

        # model-a summarization: avg of 100.0 and 120.0 = 110.0
        assert df.loc["model-a", "summarization"] == pytest.approx(110.0)
        # model-b code_generation: 250.0
        assert df.loc["model-b", "code_generation"] == pytest.approx(250.0)

    def test_get_latency_summary_empty(self, db_path):
        ResultsStore(db_path=db_path)
        loader = DashboardDataLoader(db_path=db_path)

        df = loader.get_latency_summary()
        assert df.empty


class TestGetModelComparison:
    def test_get_model_comparison(self, populated_db):
        db_path, run_id = populated_db
        loader = DashboardDataLoader(db_path=db_path)

        df = loader.get_model_comparison(run_id=run_id)

        assert isinstance(df, pd.DataFrame)
        expected_cols = ["model_id", "avg_score", "avg_latency_ms", "total_tasks", "total_tokens"]
        assert list(df.columns) == expected_cols

        # model-a: scores [0.9, 0.8, 0.7], latencies [100, 120, 200], tokens [50, 60, 80]
        row_a = df[df["model_id"] == "model-a"].iloc[0]
        assert row_a["avg_score"] == pytest.approx(0.8, rel=1e-3)
        assert row_a["avg_latency_ms"] == pytest.approx(140.0, rel=1e-3)
        assert row_a["total_tasks"] == 3
        assert row_a["total_tokens"] == 190

        # model-b: scores [0.85, 0.6], latencies [150, 250], tokens [45, 70]
        row_b = df[df["model_id"] == "model-b"].iloc[0]
        assert row_b["avg_score"] == pytest.approx(0.725, rel=1e-3)
        assert row_b["avg_latency_ms"] == pytest.approx(200.0, rel=1e-3)
        assert row_b["total_tasks"] == 2
        assert row_b["total_tokens"] == 115

    def test_get_model_comparison_empty_run(self, db_path):
        store = ResultsStore(db_path=db_path)
        run_id = store.create_run("empty-run")
        loader = DashboardDataLoader(db_path=db_path)

        df = loader.get_model_comparison(run_id=run_id)
        assert df.empty
        assert list(df.columns) == [
            "model_id", "avg_score", "avg_latency_ms", "total_tasks", "total_tokens"
        ]


class TestGetAvailableRuns:
    def test_get_available_runs(self, db_path):
        store = ResultsStore(db_path=db_path)
        store.create_run("first-run")
        store.create_run("second-run")

        loader = DashboardDataLoader(db_path=db_path)
        df = loader.get_available_runs()

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["id", "name", "created_at"]
        assert len(df) == 2
        # Ordered by id DESC
        assert df.iloc[0]["name"] == "second-run"
        assert df.iloc[1]["name"] == "first-run"

    def test_get_available_runs_empty(self, db_path):
        ResultsStore(db_path=db_path)
        loader = DashboardDataLoader(db_path=db_path)

        df = loader.get_available_runs()
        assert df.empty
        assert list(df.columns) == ["id", "name", "created_at"]


class TestGetFieldDetail:
    def test_get_field_detail_filters_correctly(self, populated_db):
        db_path, run_id = populated_db
        loader = DashboardDataLoader(db_path=db_path)

        df = loader.get_field_detail(run_id=run_id, field="summarization")

        # model-a has 2 summarization results, model-b has 1
        assert len(df) == 3
        assert all(df["task_field"] == "summarization")
        assert set(df["model_id"].unique()) == {"model-a", "model-b"}

    def test_get_field_detail_no_match(self, populated_db):
        db_path, run_id = populated_db
        loader = DashboardDataLoader(db_path=db_path)

        df = loader.get_field_detail(run_id=run_id, field="nonexistent_field")

        assert df.empty
        expected_cols = [
            "model_id", "task_field", "task_id", "prompt",
            "response", "latency_ms", "token_count", "score", "created_at",
        ]
        assert list(df.columns) == expected_cols
