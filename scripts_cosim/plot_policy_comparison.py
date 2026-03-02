#!/usr/bin/env python3
"""
Simple plotting script to compare policies across workloads.

For each workload (e.g. 100-100), this script:
- looks into `simulation_data/results/<workload>` for JSON result files
- extracts `total_rtt` for selected policies
- creates a bar chart: x = policies, y = total_rtt

Notes:
- If a workload directory does not exist, it is skipped.
- If a result file is missing or does not match the expected policy/workload, it is skipped.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

# Use a non-interactive backend so the script works in headless/CLI environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_ROOT = BASE_DIR / "simulation_data" / "results"

# Filename convention: simulation_result_<policy>.json
RESULT_FILE_PREFIX = "simulation_result_"
RESULT_FILE_SUFFIX = ".json"

# Map human-friendly policy labels to the `policy` value stored in result JSONs
# (and used in filenames: simulation_result_<value>.json)
# NOTE: we use the *network-aware* knative variant here.
POLICY_MAP: Dict[str, str] = {
    "knative": "knative_network",
    "random": "random_network",
    "hrc": "herocache_network",
    "gnn": "gnn",
    "offload_network": "offload_network",
}

# Order for consistent bar ordering in charts
POLICY_ORDER: List[str] = ["knative", "random", "hrc", "gnn", "offload_network"]

# Max bytes to read from each result JSON; top-level policy/workload_file/total_rtt
# appear in the first few KB. Full files can be 2GB+ so we never load them entirely.
_HEAD_READ_BYTES = 8192


def _read_result_head(path: Path) -> Optional[Dict[str, object]]:
    """
    Read the first _HEAD_READ_BYTES of a result JSON and extract policy,
    workload_file, and total_rtt via regex. Returns None on error or missing fields.
    Use this instead of full json.load() because result files can be multi-GB.
    """
    try:
        with path.open("r") as f:
            chunk = f.read(_HEAD_READ_BYTES)
    except OSError:
        return None

    policy_m = re.search(r'"policy"\s*:\s*"([^"]+)"', chunk)
    workload_m = re.search(r'"workload_file"\s*:\s*"([^"]+)"', chunk)
    total_rtt_m = re.search(r'"total_rtt"\s*:\s*([\d.eE+-]+)', chunk)

    if not (policy_m and workload_m and total_rtt_m):
        return None

    try:
        total_rtt = float(total_rtt_m.group(1))
    except ValueError:
        return None

    return {
        "policy": policy_m.group(1),
        "workload_file": workload_m.group(1),
        "total_rtt": total_rtt,
    }


def _workload_id_in_path(workload: str, workload_file: str) -> bool:
    """True if the workload id (e.g. 100-100) appears in the workload_file path."""
    return workload in workload_file


def discover_workloads() -> List[str]:
    """
    Return workload ids (e.g. ["100-100", "200-200"]) by listing subdirs of
    RESULTS_ROOT that contain at least one simulation_result_*.json.
    """
    if not RESULTS_ROOT.exists() or not RESULTS_ROOT.is_dir():
        return []

    workloads: List[str] = []
    for d in sorted(RESULTS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        if any(d.glob(f"{RESULT_FILE_PREFIX}*{RESULT_FILE_SUFFIX}")):
            workloads.append(d.name)
    return workloads


def collect_metrics_for_workload(workload: str) -> List[Tuple[str, float]]:
    """
    For a given workload (e.g. "100-100"), collect (policy_label, total_rtt)
    for all policies that have a matching result file and correct workload in JSON.
    Uses filename convention simulation_result_<policy>.json; reads only file head.
    """
    results: List[Tuple[str, float]] = []
    workload_dir = RESULTS_ROOT / workload

    if not workload_dir.exists() or not workload_dir.is_dir():
        return results

    for label in POLICY_ORDER:
        policy_value = POLICY_MAP.get(label)
        if not policy_value:
            continue

        path = workload_dir / f"{RESULT_FILE_PREFIX}{policy_value}{RESULT_FILE_SUFFIX}"
        if not path.exists():
            continue

        data = _read_result_head(path)
        if not data:
            continue

        if data.get("policy") != policy_value:
            continue
        if not _workload_id_in_path(workload, str(data.get("workload_file", ""))):
            continue

        total_rtt = data.get("total_rtt")
        if isinstance(total_rtt, (int, float)):
            results.append((label, float(total_rtt)))

    return results


# Distinct colors per policy (knative, random, hrc, gnn)
POLICY_COLORS: Dict[str, str] = {
    "knative": "#2ecc71",   # green
    "random": "#e74c3c",    # red
    "hrc": "#3498db",       # blue
    "gnn": "#9b59b6",       # purple
    "offload_network": "#f1c40f", # yellow
}


def plot_workload_bar_chart(workload: str, metrics: List[Tuple[str, float]]) -> None:
    """
    Create and save a bar chart for a single workload.

    Bars: one per policy (label), value = total_rtt.
    """
    if not metrics:
        print(f"[INFO] No metrics to plot for workload={workload}, skipping chart.")
        return

    labels = [label for label, _ in metrics]
    values = [value for _, value in metrics]
    colors = [POLICY_COLORS.get(label, "#95a5a6") for label in labels]

    x_positions = range(len(labels))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        x_positions,
        values,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
    )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Total RTT (s)", fontsize=11)
    ax.set_title(f"Total RTT by policy – workload {workload}", fontsize=12, fontweight="medium")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    plt.tight_layout()

    # Save chart in the workload directory
    workload_dir = RESULTS_ROOT / workload
    workload_dir.mkdir(parents=True, exist_ok=True)
    output_path = workload_dir / f"policy_comparison_{workload}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"[INFO] Saved chart for workload={workload} -> {output_path}")


def main() -> None:
    workloads = discover_workloads()
    if not workloads:
        print(f"[INFO] No workload directories found under {RESULTS_ROOT}, nothing to plot.")
        return

    for workload in workloads:
        metrics = collect_metrics_for_workload(workload)
        plot_workload_bar_chart(workload, metrics)


if __name__ == "__main__":
    main()

