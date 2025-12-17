#!/usr/bin/env python3
"""
Minimal helper to inspect RTT distributions across placement summaries.

Usage:
    python scripts/analyze_rtt_csvs.py \
        --root /root/projects/my-herosim/simulation_data/artifacts/run9_all/gnn_datasets/ds_00000/placements_csv

This script aggregates RTT values from every placement_summary_*.csv in the
provided directory, prints descriptive statistics, and saves quick-look
visualizations (histogram + box plot) next to the data.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt  # type: ignore[import]


def read_rtts(csv_dir: Path) -> List[float]:
    rtts: List[float] = []
    for csv_path in sorted(csv_dir.glob("placement_summary_*.csv")):
        with csv_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    rtt = float(row["rtt"])
                except (KeyError, ValueError):
                    continue
                rtts.append(rtt)
    return rtts


def plot_distributions(rtts: List[float], output_dir: Path) -> None:
    if not rtts:
        print("No RTT values found; skipping plots.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rtts, bins=min(40, max(10, len(rtts) // 10)), color="#4C78A8", edgecolor="white")
    ax.set_title("RTT Histogram")
    ax.set_xlabel("Total RTT (s)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    hist_path = output_dir / "rtt_histogram.png"
    fig.tight_layout()
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"Saved histogram to {hist_path}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(rtts, vert=True, showfliers=False)
    ax.set_title("RTT Box Plot")
    ax.set_ylabel("Total RTT (s)")
    ax.grid(alpha=0.3)
    box_path = output_dir / "rtt_boxplot.png"
    fig.tight_layout()
    fig.savefig(box_path, dpi=150)
    plt.close(fig)
    print(f"Saved box plot to {box_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect RTT distributions in placement CSVs.")
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing placement_summary_*.csv files.",
    )
    args = parser.parse_args()

    csv_dir = args.root.resolve()
    if not csv_dir.exists():
        raise SystemExit(f"Directory does not exist: {csv_dir}")

    print(f"Scanning {csv_dir} for placement_summary_*.csv files...")
    rtts = read_rtts(csv_dir)
    print(f"Loaded {len(rtts)} RTT samples from {csv_dir}")

    if not rtts:
        return

    mean_rtt = statistics.fmean(rtts)
    median_rtt = statistics.median(rtts)
    min_rtt = min(rtts)
    max_rtt = max(rtts)

    print("RTT summary statistics:")
    print(f"  Mean:   {mean_rtt:.3f} s")
    print(f"  Median: {median_rtt:.3f} s")
    print(f"  Min:    {min_rtt:.3f} s")
    print(f"  Max:    {max_rtt:.3f} s")

    plot_distributions(rtts, csv_dir)


if __name__ == "__main__":
    main()

