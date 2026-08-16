"""Data access layer for the evaluation dashboard.

Queries the ResultsStore SQLite database and returns pandas DataFrames
suitable for visualization and analysis.
"""

import sqlite3
from typing import ClassVar

import pandas as pd

from src.results_store import ResultsStore


class DashboardDataLoader:
    """Loads evaluation data from SQLite and returns pandas DataFrames."""

    # Column definitions for empty DataFrame fallbacks
    _RESULTS_COLUMNS: ClassVar[list[str]] = [
        "model_id",
        "task_field",
        "task_id",
        "prompt",
        "response",
        "latency_ms",
        "token_count",
        "score",
        "created_at",
    ]
    _RUNS_COLUMNS: ClassVar[list[str]] = ["id", "name", "created_at"]
    _COMPARISON_COLUMNS: ClassVar[list[str]] = [
        "model_id",
        "avg_score",
        "avg_latency_ms",
        "total_tasks",
        "total_tokens",
    ]

    def __init__(self, db_path: str = "results.db") -> None:
        """Initialize the data loader.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._store = ResultsStore(db_path=db_path)

    def _query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame.

        Args:
            sql: SQL query string.
            params: Query parameters tuple.

        Returns:
            DataFrame with query results.
        """
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def get_all_results(self, run_id: int | None = None) -> pd.DataFrame:
        """Return all evaluation results, optionally filtered by run.

        Args:
            run_id: If provided, filter results to this run only.

        Returns:
            DataFrame with columns: model_id, task_field, task_id, prompt,
            response, latency_ms, token_count, score, created_at.
        """
        base_sql = """
            SELECT model_id, task_field, task_id, prompt, response,
                   latency_ms, token_count, score, created_at
            FROM results
        """
        if run_id is not None:
            df = self._query(f"{base_sql} WHERE run_id = ?", (run_id,))
        else:
            df = self._query(base_sql)

        if df.empty:
            return pd.DataFrame(columns=self._RESULTS_COLUMNS)
        return df

    def get_model_field_summary(
        self, run_id: int | None = None
    ) -> pd.DataFrame:
        """Pivot table of average scores: rows=model_id, columns=task_field.

        Args:
            run_id: If provided, filter to this run only.

        Returns:
            DataFrame pivot with model_id as index and task_field as columns,
            values are average scores.
        """
        df = self.get_all_results(run_id=run_id)
        if df.empty:
            return pd.DataFrame()
        return df.pivot_table(
            index="model_id",
            columns="task_field",
            values="score",
            aggfunc="mean",
        )

    def get_latency_summary(self, run_id: int | None = None) -> pd.DataFrame:
        """Pivot table of average latency: rows=model_id, columns=task_field.

        Args:
            run_id: If provided, filter to this run only.

        Returns:
            DataFrame pivot with model_id as index and task_field as columns,
            values are average latency in milliseconds.
        """
        df = self.get_all_results(run_id=run_id)
        if df.empty:
            return pd.DataFrame()
        return df.pivot_table(
            index="model_id",
            columns="task_field",
            values="latency_ms",
            aggfunc="mean",
        )

    def get_model_comparison(self, run_id: int) -> pd.DataFrame:
        """Per-model aggregated summary for a given run.

        Args:
            run_id: The run to summarize.

        Returns:
            DataFrame with columns: model_id, avg_score, avg_latency_ms,
            total_tasks, total_tokens.
        """
        sql = """
            SELECT
                model_id,
                AVG(score) AS avg_score,
                AVG(latency_ms) AS avg_latency_ms,
                COUNT(*) AS total_tasks,
                SUM(token_count) AS total_tokens
            FROM results
            WHERE run_id = ?
            GROUP BY model_id
            ORDER BY avg_score DESC
        """
        df = self._query(sql, (run_id,))
        if df.empty:
            return pd.DataFrame(columns=self._COMPARISON_COLUMNS)
        return df

    def get_available_runs(self) -> pd.DataFrame:
        """List all evaluation runs.

        Returns:
            DataFrame with columns: id, name, created_at.
        """
        sql = "SELECT id, name, created_at FROM runs ORDER BY id DESC"
        df = self._query(sql)
        if df.empty:
            return pd.DataFrame(columns=self._RUNS_COLUMNS)
        return df

    def get_field_detail(self, run_id: int, field: str) -> pd.DataFrame:
        """Get all results for a specific task field within a run.

        Args:
            run_id: The run to query.
            field: The task_field value to filter on.

        Returns:
            DataFrame with columns: model_id, task_field, task_id, prompt,
            response, latency_ms, token_count, score, created_at.
        """
        sql = """
            SELECT model_id, task_field, task_id, prompt, response,
                   latency_ms, token_count, score, created_at
            FROM results
            WHERE run_id = ? AND task_field = ?
            ORDER BY model_id, task_id
        """
        df = self._query(sql, (run_id, field))
        if df.empty:
            return pd.DataFrame(columns=self._RESULTS_COLUMNS)
        return df

    def get_all_fields(self) -> list[str]:
        """Return all distinct task_field values in the database.

        Returns:
            Sorted list of field names.
        """
        sql = "SELECT DISTINCT task_field FROM results ORDER BY task_field"
        df = self._query(sql)
        if df.empty:
            return []
        return df["task_field"].tolist()

    def get_field_results_all_runs(self, field: str) -> pd.DataFrame:
        """Get all results for a specific field across ALL runs.

        Args:
            field: The task_field value to filter on.

        Returns:
            DataFrame with columns: run_id, run_name, model_id, task_id,
            score, latency_ms, token_count, created_at.
        """
        sql = """
            SELECT r.run_id, runs.name AS run_name, r.model_id, r.task_id,
                   r.score, r.latency_ms, r.token_count, r.created_at
            FROM results r
            JOIN runs ON r.run_id = runs.id
            WHERE r.task_field = ?
            ORDER BY r.created_at DESC, r.model_id, r.task_id
        """
        df = self._query(sql, (field,))
        if df.empty:
            return pd.DataFrame(
                columns=["run_id", "run_name", "model_id", "task_id",
                         "score", "latency_ms", "token_count", "created_at"]
            )
        return df

    def get_field_model_summary(self, field: str) -> pd.DataFrame:
        """Get per-model summary for a field across all runs (latest run per model).

        Args:
            field: The task_field to summarize.

        Returns:
            DataFrame with columns: model_id, avg_score, avg_latency_ms, task_count, runs.
        """
        sql = """
            SELECT
                model_id,
                AVG(score) AS avg_score,
                AVG(latency_ms) AS avg_latency_ms,
                COUNT(*) AS task_count,
                COUNT(DISTINCT run_id) AS runs
            FROM results
            WHERE task_field = ?
            GROUP BY model_id
            ORDER BY avg_score DESC
        """
        df = self._query(sql, (field,))
        if df.empty:
            return pd.DataFrame(
                columns=["model_id", "avg_score", "avg_latency_ms", "task_count", "runs"]
            )
        return df

    def get_latest_per_field_leaderboard(self) -> pd.DataFrame:
        """Get model leaderboard using the latest run for each field.

        For each field, finds the most recent run that evaluated it,
        then aggregates scores across all fields per model.

        Returns:
            DataFrame with columns: model_id, avg_score, avg_latency_ms,
            fields_evaluated, total_tasks.
        """
        sql = """
            WITH latest_runs AS (
                SELECT task_field, MAX(run_id) AS run_id
                FROM results
                GROUP BY task_field
            ),
            latest_results AS (
                SELECT r.*
                FROM results r
                INNER JOIN latest_runs lr
                    ON r.task_field = lr.task_field AND r.run_id = lr.run_id
            )
            SELECT
                model_id,
                AVG(score) AS avg_score,
                AVG(latency_ms) AS avg_latency_ms,
                COUNT(DISTINCT task_field) AS fields_evaluated,
                COUNT(*) AS total_tasks
            FROM latest_results
            GROUP BY model_id
            ORDER BY avg_score DESC
        """
        df = self._query(sql)
        if df.empty:
            return pd.DataFrame(
                columns=["model_id", "avg_score", "avg_latency_ms",
                         "fields_evaluated", "total_tasks"]
            )
        return df

    def get_latest_per_field_scores(self) -> pd.DataFrame:
        """Get per-model, per-field scores using latest run for each field.

        Returns:
            DataFrame with columns: model_id, task_field, avg_score, avg_latency_ms.
        """
        sql = """
            WITH latest_runs AS (
                SELECT task_field, MAX(run_id) AS run_id
                FROM results
                GROUP BY task_field
            ),
            latest_results AS (
                SELECT r.*
                FROM results r
                INNER JOIN latest_runs lr
                    ON r.task_field = lr.task_field AND r.run_id = lr.run_id
            )
            SELECT
                model_id,
                task_field,
                AVG(score) AS avg_score,
                AVG(latency_ms) AS avg_latency_ms
            FROM latest_results
            GROUP BY model_id, task_field
            ORDER BY model_id, task_field
        """
        df = self._query(sql)
        if df.empty:
            return pd.DataFrame(
                columns=["model_id", "task_field", "avg_score", "avg_latency_ms"]
            )
        return df

    def get_field_coverage(self) -> pd.DataFrame:
        """Get evaluation coverage: which fields have been run, best model per field.

        Returns:
            DataFrame with columns: task_field, best_model, best_score,
            avg_latency_ms, last_run_name, evaluated.
        """
        sql = """
            WITH latest_runs AS (
                SELECT task_field, MAX(run_id) AS run_id
                FROM results
                GROUP BY task_field
            ),
            field_summary AS (
                SELECT
                    r.task_field,
                    r.model_id,
                    AVG(r.score) AS avg_score,
                    AVG(r.latency_ms) AS avg_latency_ms,
                    runs.name AS run_name
                FROM results r
                INNER JOIN latest_runs lr
                    ON r.task_field = lr.task_field AND r.run_id = lr.run_id
                INNER JOIN runs ON r.run_id = runs.id
                GROUP BY r.task_field, r.model_id
            ),
            best_per_field AS (
                SELECT
                    task_field,
                    model_id AS best_model,
                    avg_score AS best_score,
                    avg_latency_ms,
                    run_name AS last_run_name
                FROM field_summary
                WHERE (task_field, avg_score) IN (
                    SELECT task_field, MAX(avg_score)
                    FROM field_summary
                    GROUP BY task_field
                )
            )
            SELECT * FROM best_per_field
            ORDER BY task_field
        """
        df = self._query(sql)
        if df.empty:
            return pd.DataFrame(
                columns=["task_field", "best_model", "best_score",
                         "avg_latency_ms", "last_run_name"]
            )
        return df
