"""SQLite-backed results persistence for model evaluation runs."""

import json
import sqlite3


class ResultsStore:
    """Stores and retrieves model evaluation results using SQLite."""

    def __init__(self, db_path: str = "results.db") -> None:
        """Initialize the results store.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create database tables and indexes if they do not exist."""
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now')),
                    metadata TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER REFERENCES runs(id),
                    model_id TEXT NOT NULL,
                    task_field TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    token_count INTEGER NOT NULL,
                    score REAL,
                    created_at TEXT DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_results_run_model
                ON results(run_id, model_id)
                """
            )
            conn.commit()

    def create_run(self, run_name: str, metadata: dict | None = None) -> int:
        """Create a new evaluation run.

        Args:
            run_name: Human-readable name for the run.
            metadata: Optional dictionary of run metadata (stored as JSON).

        Returns:
            The integer ID of the newly created run.
        """
        meta_json = json.dumps(metadata) if metadata is not None else None
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            cursor = conn.execute(
                "INSERT INTO runs (name, metadata) VALUES (?, ?)",
                (run_name, meta_json),
            )
            conn.commit()
            return cursor.lastrowid

    def save_result(
        self,
        run_id: int,
        model_id: str,
        task_field: str,
        task_id: str,
        prompt: str,
        response: str,
        latency_ms: float,
        token_count: int,
        score: float | None = None,
    ) -> int:
        """Save a single evaluation result.

        Args:
            run_id: ID of the parent run.
            model_id: Identifier of the model evaluated.
            task_field: Category or field of the task.
            task_id: Unique identifier for the task.
            prompt: The prompt sent to the model.
            response: The model's response text.
            latency_ms: Response latency in milliseconds.
            token_count: Number of tokens in the response.
            score: Optional evaluation score.

        Returns:
            The integer ID of the newly inserted result row.
        """
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            cursor = conn.execute(
                """
                INSERT INTO results
                    (run_id, model_id, task_field, task_id, prompt, response,
                     latency_ms, token_count, score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    model_id,
                    task_field,
                    task_id,
                    prompt,
                    response,
                    latency_ms,
                    token_count,
                    score,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_run_results(self, run_id: int) -> list[dict]:
        """Retrieve all results for a given run.

        Args:
            run_id: ID of the run to query.

        Returns:
            List of result dictionaries with all columns.
        """
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM results WHERE run_id = ?", (run_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_model_summary(self, run_id: int, model_id: str) -> dict:
        """Get aggregated metrics for a model within a run.

        Args:
            run_id: ID of the run.
            model_id: Identifier of the model.

        Returns:
            Dictionary with avg_latency_ms, avg_score, and total_tokens.
        """
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            cursor = conn.execute(
                """
                SELECT
                    AVG(latency_ms) as avg_latency_ms,
                    AVG(score) as avg_score,
                    SUM(token_count) as total_tokens
                FROM results
                WHERE run_id = ? AND model_id = ?
                """,
                (run_id, model_id),
            )
            row = cursor.fetchone()
            return {
                "avg_latency_ms": row[0],
                "avg_score": row[1],
                "total_tokens": row[2],
            }

    def list_runs(self) -> list[dict]:
        """List all evaluation runs.

        Returns:
            List of run dictionaries with id, name, created_at, and metadata
            (metadata is deserialized from JSON).
        """
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT id, name, created_at, metadata FROM runs ORDER BY id"
            )
            runs = []
            for row in cursor.fetchall():
                run = dict(row)
                if run["metadata"] is not None:
                    run["metadata"] = json.loads(run["metadata"])
                runs.append(run)
            return runs
