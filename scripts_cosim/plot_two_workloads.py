#!/usr/bin/env python3
"""
Plot two workloads (e.g. 100-100 and 100-50) side by side in a single PNG,
for the knative, random, gnn and offload policies.

It reuses the helpers from scripts_cosim.plot_policy_comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import helpers from sibling module via package root
from scripts_cosim.plot_policy_comparison import (  # type: ignore[import]
    collect_metrics_for_workload,
    POLICY_COLORS,
)


def _filter_policies(
    metrics: List[Tuple[str, float]],
) -> List[Tuple[str, float]]:
    """Keep only knative, random, gnn, offload_network in the desired order."""
    order = ["knative", "random", "gnn", "offload_network"]
    by_label = {label: value for label, value in metrics}
    return [(label, by_label[label]) for label in order if label in by_label]


def main() -> None:
    # Adjust these if you want different workloads
    workloads = ["100-100", "100-50"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, workload in zip(axes, workloads):
        metrics = collect_metrics_for_workload(workload)
        metrics = _filter_policies(metrics)

        if not metrics:
            ax.set_title(f"{workload} (no data)")
            ax.axis("off")
            continue

        labels = [label for label, _ in metrics]
        values = [value for _, value in metrics]
        colors = [POLICY_COLORS.get(label, "#95a5a6") for label in labels]
        x_positions = range(len(labels))

        ax.bar(
            x_positions,
            values,
            color=colors,
            edgecolor="white",
            linewidth=1.2,
        )
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(labels)
        ax.set_title(workload)
        ax.set_ylabel("Total RTT (s)")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    fig.suptitle("Total RTT by policy", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = Path("simulation_data/results/policy_comparison_100-100_100-50.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved combined plot -> {out_path}")


if __name__ == "__main__":
    main()

