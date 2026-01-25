#!/usr/bin/env python3
"""
Parallel metadata generator for run_non_unique datasets.

This script scans run_non_unique/gnn_datasets_{N}tasks directories and enriches
each dataset by writing system_state_captured_unique.json using the existing
knative baseline logic (no placement generation).
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json  # noqa: E402
from src.executeknativecosim import run_knative_baseline_for_dataset, setup_logging  # noqa: E402


def _parse_task_counts(value: str) -> List[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [int(p) for p in parts]


def _iter_dataset_dirs(base_dir: Path) -> Iterable[Path]:
    if not base_dir.exists():
        return []
    return sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("ds_")])


def _worker(args: Tuple[str, str, Optional[int]]) -> Tuple[str, bool, str, float, str, int]:
    dataset_dir_str, sim_input_path_str, num_tasks = args
    dataset_dir = Path(dataset_dir_str)
    sim_input_path = Path(sim_input_path_str)
    logger = setup_logging(Path("."))
    start_time = time.time()
    exit_code = 0
    rtt_value = "N/A"
    try:
        ok = run_knative_baseline_for_dataset(
            dataset_dir, sim_input_path, logger, num_tasks=num_tasks
        )
        if ok:
            captured_path = dataset_dir / "system_state_captured_unique.json"
            if captured_path.exists():
                try:
                    with open(captured_path, "r") as f:
                        captured = json.load(f)
                    rtt_value = captured.get("total_rtt", "N/A")
                    if isinstance(rtt_value, (int, float)):
                        rtt_value = f"{rtt_value:.3f}"
                except Exception:
                    rtt_value = "N/A"
        else:
            exit_code = 1
    except Exception:
        ok = False
        exit_code = 1
    duration = time.time() - start_time
    return dataset_dir.name, ok, dataset_dir_str, duration, str(rtt_value), exit_code


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel knative metadata generator for run_non_unique datasets"
    )
    parser.add_argument(
        "--datasets-root",
        default=str(PROJECT_ROOT / "simulation_data" / "artifacts" / "run_non_unique"),
        help="Root directory containing gnn_datasets_{N}tasks folders",
    )
    parser.add_argument(
        "--task-counts",
        default="2,3",
        help="Comma-separated task counts to process (e.g., 2,3)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Max parallel workers (default: CPU count - 1)",
    )
    parser.add_argument(
        "--sim-input-path",
        default=str(PROJECT_ROOT / "data" / "nofs-ids"),
        help="Simulation inputs path",
    )
    args = parser.parse_args()

    task_counts = _parse_task_counts(args.task_counts)
    cpu_count = os.cpu_count() or 1
    max_workers = args.workers or max(1, cpu_count - 1)

    print("=== Knative metadata generation (parallel) ===")
    print(f"Datasets root: {args.datasets_root}")
    print(f"Task counts: {task_counts}")
    print(f"Workers: {max_workers}")
    print("")

    total_submitted = 0
    for n_tasks in task_counts:
        datasets_base = Path(args.datasets_root) / f"gnn_datasets_{n_tasks}tasks"
        if not datasets_base.exists():
            print(f"[{n_tasks} tasks] Skipping missing directory: {datasets_base}")
            continue

        ds_dirs = _iter_dataset_dirs(datasets_base)
        if not ds_dirs:
            print(f"[{n_tasks} tasks] No ds_* directories found in {datasets_base}")
            continue

        # Filter datasets that already have captured state
        pending = [
            d for d in ds_dirs if not (d / "system_state_captured_unique.json").exists()
        ]

        print(f"[{n_tasks} tasks] Total: {len(ds_dirs)} | Pending: {len(pending)}")
        if not pending:
            continue

        # Create progress log for this task count
        progress_log = PROJECT_ROOT / "logs" / f"knative_baseline_run_non_unique_{n_tasks}tasks_progress.txt"

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _worker, (str(d), args.sim_input_path, n_tasks)
                )
                for d in pending
            ]
            total_submitted += len(futures)

            completed = 0
            succeeded = 0
            for future in as_completed(futures):
                ds_name, ok, ds_path, duration, rtt_value, exit_code = future.result()
                completed += 1
                if ok:
                    succeeded += 1
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if ok:
                    line = f"{ds_name} SUCCESS {timestamp} {duration:.1f}s RTT={rtt_value}"
                else:
                    line = f"{ds_name} FAILED {timestamp} {duration:.1f}s exit_code={exit_code}"
                with open(progress_log, "a") as log_f:
                    log_f.write(line + "\n")
                if completed % 10 == 0 or completed == len(futures):
                    print(
                        f"[{n_tasks} tasks] Progress: {completed}/{len(futures)} "
                        f"(success: {succeeded})"
                    )

    print("")
    print(f"Submitted datasets: {total_submitted}")
    print("Done.")


if __name__ == "__main__":
    main()
