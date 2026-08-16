"""CLI script to run model evaluations from the command line.

Usage:
    python scripts/run_eval.py --field code_generation
    python scripts/run_eval.py --models qwen2.5-coder:7b mistral:7b
    python scripts/run_eval.py --list-fields
    python scripts/run_eval.py --list-models
"""

import argparse
import os
import sys
from datetime import UTC, datetime

# Allow running from project root: python scripts/run_eval.py
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import yaml

from src.runner import EvalRunner


def list_fields() -> list[str]:
    """Scan tasks/ for directories containing prompts.yaml."""
    tasks_dir = os.path.join(_PROJECT_ROOT, "tasks")
    fields = []
    if not os.path.isdir(tasks_dir):
        return fields
    for entry in sorted(os.listdir(tasks_dir)):
        prompts_path = os.path.join(tasks_dir, entry, "prompts.yaml")
        if os.path.isdir(os.path.join(tasks_dir, entry)) and os.path.isfile(
            prompts_path
        ):
            fields.append(entry)
    return fields


def list_models(config_path: str) -> list[dict]:
    """Load models from config YAML."""
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return data.get("models", [])


def print_summary_table(results: list[dict], run_name: str, field: str) -> None:
    """Print a formatted summary table of evaluation results."""
    if not results:
        print("No results to display.")
        return

    # Aggregate by model
    model_stats: dict[str, dict] = {}
    for r in results:
        mid = r["model_id"]
        if mid not in model_stats:
            model_stats[mid] = {"scores": [], "latencies": [], "count": 0}
        model_stats[mid]["scores"].append(r["score"])
        model_stats[mid]["latencies"].append(r["latency_ms"])
        model_stats[mid]["count"] += 1

    # Sort by average score descending
    sorted_models = sorted(
        model_stats.items(),
        key=lambda x: sum(x[1]["scores"]) / len(x[1]["scores"]),
        reverse=True,
    )

    header_line = f"Run: {run_name} | Field: {field}"
    width = max(55, len(header_line) + 4)
    border = "\u2550" * width
    thin_border = "\u2500" * width

    print(f"\n{border}")
    print(f"  {header_line}")
    print(border)
    print(f"  {'Model':<20}{'Avg Score':<12}{'Avg Latency':<14}{'Tasks'}")
    print(f"  {thin_border}")

    for model_id, stats in sorted_models:
        avg_score = sum(stats["scores"]) / len(stats["scores"])
        avg_latency = sum(stats["latencies"]) / len(stats["latencies"])
        count = stats["count"]
        print(
            f"  {model_id:<20}{avg_score:<12.2f}{avg_latency:<14.0f}ms{count}"
        )

    print(f"{border}\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run model evaluations against Ollama-served LLMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python scripts/run_eval.py --field code_generation
  python scripts/run_eval.py --models qwen2.5-coder:7b mistral:7b
  python scripts/run_eval.py --list-fields
  python scripts/run_eval.py --list-models
""",
    )

    parser.add_argument(
        "--field",
        type=str,
        default=None,
        help="Run a specific evaluation field (e.g., 'code_generation'). "
        "If omitted, runs all available fields.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help="Specific model IDs to test (e.g., 'qwen2.5-coder:7b'). "
        "If omitted, uses all configured models.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        metavar="NAME",
        help="Custom name for this evaluation run. "
        "Defaults to '<field>-YYYYMMDD-HHMM', auto-generated per field.",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="results.db",
        metavar="PATH",
        help="Path to the results database (default: results.db).",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        metavar="URL",
        help="Ollama server URL (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--list-fields",
        action="store_true",
        help="List available task fields and exit.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List configured models and exit.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the evaluation CLI."""
    args = parse_args(argv)

    # Change to project root so relative paths in runner resolve correctly
    os.chdir(_PROJECT_ROOT)

    models_config_path = os.path.join("config", "models.yaml")
    timeouts_config_path = os.path.join("config", "timeouts.yaml")

    # Handle --list-fields
    if args.list_fields:
        fields = list_fields()
        if not fields:
            print("No task fields found in tasks/")
            return 0
        print("Available evaluation fields:")
        for f in fields:
            print(f"  • {f}")
        return 0

    # Handle --list-models
    if args.list_models:
        if not os.path.isfile(models_config_path):
            print(f"Models config not found: {models_config_path}")
            return 1
        models = list_models(models_config_path)
        if not models:
            print("No models configured.")
            return 0
        print("Configured models:")
        print(f"  {'ID':<25}{'Name':<20}{'Params':<8}{'RAM':<8}{'Role'}")
        print(f"  {'─'*80}")
        for m in models:
            print(
                f"  {m['id']:<25}{m.get('name', ''):<20}"
                f"{m.get('params', ''):<8}{m.get('ram_gb', ''):<8}"
                f"{m.get('role', '')}"
            )
        return 0

    # Shared timestamp for this invocation; each field gets its own short,
    # meaningful default name (e.g. "classification-20260816-0801") unless
    # the caller supplied --run-name explicitly.
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")

    # Validate field if specified
    if args.field:
        available = list_fields()
        if args.field not in available:
            print(f"Error: field '{args.field}' not found.")
            print(f"Available fields: {', '.join(available)}")
            return 1

    # Create the runner
    try:
        runner = EvalRunner(
            models_config_path=models_config_path,
            timeouts_config_path=timeouts_config_path,
            db_path=args.db,
            ollama_url=args.ollama_url,
        )
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {e}")
        return 1
    except (ValueError, OSError, yaml.YAMLError) as e:
        print(f"Error initializing runner: {e}")
        return 1

    # Run evaluation
    run_ids: list[int] = []
    run_names: dict[int, str] = {}
    try:
        if args.field:
            run_name = args.run_name or f"{args.field}-{timestamp}"
            print(f"Starting evaluation: field={args.field}, run_name={run_name}")
            print(f"Models: {args.models or 'all configured'}\n")
            run_id = runner.run_field(
                field=args.field,
                model_ids=args.models,
                run_name=run_name,
            )
            run_ids.append(run_id)
            run_names[run_id] = run_name
        else:
            fields = list_fields()
            if not fields:
                print("Error: No task fields found in tasks/")
                return 1
            if args.run_name:
                print(f"Starting full evaluation: run_name={args.run_name}")
            else:
                print(
                    "Starting full evaluation across "
                    f"{len(fields)} fields (auto-named per field, e.g. "
                    f"'{fields[0]}-{timestamp}')"
                )
            print(f"Models: {args.models or 'all configured'}\n")
            # run_all_fields doesn't accept run_name, so run each field individually
            for field in fields:
                print(f"\n{'='*60}")
                print(f"  Field: {field}")
                print(f"{'='*60}\n")
                run_name = args.run_name or f"{field}-{timestamp}"
                run_id = runner.run_field(
                    field=field,
                    model_ids=args.models,
                    run_name=run_name,
                )
                run_ids.append(run_id)
                run_names[run_id] = run_name

    except KeyboardInterrupt:
        print("\n\nInterrupted! Printing partial results...\n")
        for rid in run_ids:
            results = runner.get_results_table(rid)
            if results:
                # Determine field from results
                field_name = results[0]["task_field"] if results else "unknown"
                print_summary_table(results, run_names[rid], field_name)
        return 1
    except (OSError, RuntimeError, ValueError) as e:
        print(f"\nError during evaluation: {e}")
        # Print any partial results collected so far
        for rid in run_ids:
            results = runner.get_results_table(rid)
            if results:
                field_name = results[0]["task_field"] if results else "unknown"
                print_summary_table(results, run_names[rid], field_name)
        return 1

    # Print summary tables for all completed runs
    for rid in run_ids:
        results = runner.get_results_table(rid)
        if results:
            field_name = results[0]["task_field"] if results else "unknown"
            print_summary_table(results, run_names[rid], field_name)

    print("Evaluation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
