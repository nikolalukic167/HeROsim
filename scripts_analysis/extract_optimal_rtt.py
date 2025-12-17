#!/usr/bin/env python3
"""
Inspect all optimal_result.json files and list warmup vs workload RTTs.

Usage:
    python scripts/extract_optimal_rtt.py \
        --root /root/projects/my-herosim/simulation_data/artifacts/run9_all/gnn_datasets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import matplotlib.pyplot as plt  # type: ignore[import]


def load_task_results(json_path: Path) -> List[Dict[str, Any]]:
    with json_path.open("r") as handle:
        data = json.load(handle)
    return data.get("stats", {}).get("taskResults", [])


def split_elapsed_times(task_results: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    warmups: List[float] = []
    workload: List[float] = []
    for entry in task_results:
        task_id = entry.get("taskId")
        elapsed = float(entry.get("elapsedTime", 0.0))
        if task_id is None:
            continue
        if task_id < 0:
            warmups.append(elapsed)
        else:
            workload.append(elapsed)
    return {"warmups": warmups, "workload": workload}


def plot_distributions(
    base_dir: Path,
    warmup_all: List[float],
    workload_all: List[float],
    per_dataset: List[Tuple[str, float, float]],
) -> None:
    output_dir = base_dir / "rtt_analysis"
    output_dir.mkdir(exist_ok=True)

    if warmup_all:
        plt.figure(figsize=(8, 4))
        plt.hist(warmup_all, bins=min(40, max(10, len(warmup_all) // 5)), color="#E45756", edgecolor="white")
        plt.title("Warmup Task elapsedTime Distribution")
        plt.xlabel("elapsedTime (s)")
        plt.ylabel("Count")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        warmup_hist = output_dir / "warmup_hist.png"
        plt.savefig(warmup_hist, dpi=150)
        plt.close()
        print(f"Saved warmup histogram to {warmup_hist}")

    if workload_all:
        plt.figure(figsize=(8, 4))
        plt.hist(workload_all, bins=min(40, max(10, len(workload_all) // 5)), color="#4C78A8", edgecolor="white")
        plt.title("Workload Task elapsedTime Distribution")
        plt.xlabel("elapsedTime (s)")
        plt.ylabel("Count")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        workload_hist = output_dir / "workload_hist.png"
        plt.savefig(workload_hist, dpi=150)
        plt.close()
        print(f"Saved workload histogram to {workload_hist}")

    if per_dataset:
        labels = [name for name, _, _ in per_dataset]
        warm_totals = [total for _, total, _ in per_dataset]
        work_totals = [total for _, _, total in per_dataset]
        x = range(len(labels))
        plt.figure(figsize=(max(10, len(labels) * 0.6), 5))
        plt.bar(x, warm_totals, label="Warmup total", color="#E45756")
        plt.bar(x, work_totals, bottom=warm_totals, label="Workload total", color="#4C78A8")
        plt.xticks(x, labels, rotation=90)
        plt.ylabel("Sum of elapsedTime (s)")
        plt.title("Per-dataset elapsedTime totals")
        plt.legend()
        plt.tight_layout()
        stacked_bar = output_dir / "per_dataset_totals.png"
        plt.savefig(stacked_bar, dpi=150)
        plt.close()
        print(f"Saved per-dataset totals chart to {stacked_bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract warmup vs workload RTT components from optimal_result.json files.")
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Base directory containing ds_xxxxx subdirectories.",
    )
    args = parser.parse_args()

    base_dir = args.root.resolve()
    if not base_dir.exists():
        raise SystemExit(f"Base directory does not exist: {base_dir}")

    warmup_all: List[float] = []
    workload_all: List[float] = []
    per_dataset: List[Tuple[str, float, float]] = []

    for ds_dir in sorted(base_dir.glob("ds_*")):
        json_path = ds_dir / "optimal_result.json"
        if not json_path.exists():
            continue
        task_results = load_task_results(json_path)
        splits = split_elapsed_times(task_results)
        warmups = splits["warmups"]
        workload = splits["workload"]
        warm_total = sum(warmups)
        work_total = sum(workload)
        warmup_all.extend(warmups)
        workload_all.extend(workload)
        per_dataset.append((ds_dir.name, warm_total, work_total))
        print(f"{ds_dir.name}:")
        print(f"  warmup elapsedTimes ({len(warmups)}): {warmups}")
        print(f"  warmup total: {warm_total:.6f}s")
        print(f"  workload elapsedTimes ({len(workload)}): {workload}")
        print(f"  workload total: {work_total:.6f}s")

    if warmup_all or workload_all:
        print("\n=== Aggregate Totals ===")
        if warmup_all:
            warm_sum = sum(warmup_all)
            print(f"Warmup tasks → count={len(warmup_all)}, sum={warm_sum:.6f}s, mean={warm_sum / len(warmup_all):.6f}s")
        else:
            print("Warmup tasks → none")
        if workload_all:
            work_sum = sum(workload_all)
            print(f"Workload tasks → count={len(workload_all)}, sum={work_sum:.6f}s, mean={work_sum / len(workload_all):.6f}s")
        else:
            print("Workload tasks → none")

    if warmup_all or workload_all:
        plot_distributions(base_dir, warmup_all, workload_all, per_dataset)


if __name__ == "__main__":
    main()

