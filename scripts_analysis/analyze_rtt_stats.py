#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_rtts(placements_path: Path):
    rtts = []
    with placements_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "rtt" in obj:
                rtts.append(float(obj["rtt"]))
    return np.array(rtts, dtype=float)


def main():
    base_dir = Path("/root/projects/my-herosim/simulation_data/artifacts/run1650/gnn_datasets")

    all_rtts = []
    per_dataset_counts = {}

    # Iterate over all ds_* folders
    for ds_dir in sorted(base_dir.glob("ds_*")):
        placements_path = ds_dir / "placements" / "placements.jsonl"
        if not placements_path.exists():
            continue

        rtts = load_rtts(placements_path)
        if rtts.size == 0:
            continue

        dataset_id = ds_dir.name
        per_dataset_counts[dataset_id] = len(rtts)
        all_rtts.append(rtts)

        # Per-dataset basic statistics
        print(f"Dataset: {dataset_id}")
        print(f"  Number of placements: {len(rtts)}")
        print(f"  Min RTT:   {rtts.min():.6f}")
        print(f"  Max RTT:   {rtts.max():.6f}")
        print(f"  Mean RTT:  {rtts.mean():.6f}")
        print(f"  Median RTT:{np.median(rtts):.6f}")
        print(f"  Std RTT:   {rtts.std(ddof=1):.6f}")
        print()

    if not all_rtts:
        print("No RTTs found in any dataset.")
        return

    # Concatenate all RTTs for global view
    all_rtts_arr = np.concatenate(all_rtts)
    print("=== Global statistics across all datasets ===")
    print(f"Total datasets with placements: {len(per_dataset_counts)}")
    print(f"Total placements: {len(all_rtts_arr)}")
    print(f"Min RTT:   {all_rtts_arr.min():.6f}")
    print(f"Max RTT:   {all_rtts_arr.max():.6f}")
    print(f"Mean RTT:  {all_rtts_arr.mean():.6f}")
    print(f"Median RTT:{np.median(all_rtts_arr):.6f}")
    print(f"Std RTT:   {all_rtts_arr.std(ddof=1):.6f}")

    # ==== Visualization 1: Global histogram of RTTs ====
    plt.figure(figsize=(8, 5))
    plt.hist(all_rtts_arr, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    plt.title("RTT Histogram - All Datasets")
    plt.xlabel("RTT (seconds)")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.3)

    # ==== Visualization 2: Global ECDF ====
    sorted_rtts = np.sort(all_rtts_arr)
    y = np.linspace(0, 1, len(sorted_rtts), endpoint=False)

    plt.figure(figsize=(8, 5))
    plt.step(sorted_rtts, y, where="post", color="darkorange")
    plt.title("RTT Empirical CDF - All Datasets")
    plt.xlabel("RTT (seconds)")
    plt.ylabel("Fraction of placements ≤ RTT")
    plt.grid(alpha=0.3)

    # ==== Visualization 3: Per-dataset mean RTTs (bar chart) ====
    dataset_ids = list(per_dataset_counts.keys())
    mean_rtts = []
    for ds_id in dataset_ids:
        placements_path = base_dir / ds_id / "placements" / "placements.jsonl"
        rtts = load_rtts(placements_path)
        if rtts.size > 0:
            mean_rtts.append(rtts.mean())
        else:
            mean_rtts.append(np.nan)

    plt.figure(figsize=(10, 5))
    x = np.arange(len(dataset_ids))
    plt.bar(x, mean_rtts, color="seagreen", alpha=0.7)
    plt.xticks(x, dataset_ids, rotation=90)
    plt.title("Mean RTT per Dataset")
    plt.xlabel("Dataset ID")
    plt.ylabel("Mean RTT (seconds)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    # ==== Save merged RTT sample from up to 10 random datasets ====
    # We don't want separate files per dataset; instead, we take up to 10
    # random datasets, merge their RTTs, and save that combined distribution.
    if len(dataset_ids) > 0:
        rng = np.random.default_rng(seed=42)
        k = min(10, len(dataset_ids))
        sampled_ids = list(rng.choice(dataset_ids, size=k, replace=False))

        merged_sample = []
        for ds_id in sampled_ids:
            placements_path = base_dir / ds_id / "placements" / "placements.jsonl"
            rtts_ds = load_rtts(placements_path)
            if rtts_ds.size > 0:
                merged_sample.append(rtts_ds)

        if merged_sample:
            merged_arr = np.concatenate(merged_sample)
            out_dir = base_dir.parent / "analysis_results"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "rtt_merged_sample_10_datasets.npy"
            np.save(out_path, merged_arr)
            print()
            print(f"Saved merged RTT sample from {len(sampled_ids)} dataset(s) to: {out_path}")
            print(f"Sampled datasets: {sampled_ids}")

    plt.show()


if __name__ == "__main__":
    main()