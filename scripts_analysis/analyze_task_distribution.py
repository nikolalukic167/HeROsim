#!/usr/bin/env python3
"""
Analyze task results distribution across all datasets.
Sums metrics for tasks with taskId 1-4 and visualizes distribution.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# Paths
BASE_DIR = Path("simulation_data/artifacts/run10_all/gnn_datasets")
TASK_IDS = [0, 1, 2, 3, 4]  # taskIds 0-4 (5 tasks total)

# Storage for results
dataset_results = []
skipped_no_file = 0
skipped_no_tasks = 0
errors = 0

print(f"Scanning datasets in {BASE_DIR}...")
dataset_dirs = sorted([d for d in BASE_DIR.iterdir() if d.is_dir() and d.name.startswith("ds_")])
total_datasets = len(dataset_dirs)
print(f"Found {total_datasets} datasets\n")

# Process each dataset
for i, ds_dir in enumerate(dataset_dirs):
    optimal_result_path = ds_dir / "optimal_result.json"
    
    if not optimal_result_path.exists():
        skipped_no_file += 1
        print(f"⚠️  Skipping {ds_dir.name}: optimal_result.json not found")
        continue
    
    try:
        with open(optimal_result_path, 'r') as f:
            data = json.load(f)
        
        stats = data.get("stats", {})
        task_results = stats.get("taskResults", [])
        
        # Filter for taskIds 0-4
        filtered_tasks = [t for t in task_results if t.get("taskId") in TASK_IDS]
        
        if len(filtered_tasks) == 0:
            skipped_no_tasks += 1
            print(f"⚠️  {ds_dir.name}: No tasks with taskId in {TASK_IDS}")
            continue
        
        # Sum metrics
        sums = {
            "elapsedTime": sum(t.get("elapsedTime", 0) for t in filtered_tasks),
            "executionTime": sum(t.get("executionTime", 0) for t in filtered_tasks),
            "energy": sum(t.get("energy", 0) for t in filtered_tasks),
            "computeTime": sum(t.get("computeTime", 0) for t in filtered_tasks),
            "coldStartTime": sum(t.get("coldStartTime", 0) for t in filtered_tasks),
            "queueTime": sum(t.get("queueTime", 0) for t in filtered_tasks),
        }
        
        # Count tasks
        sums["taskCount"] = len(filtered_tasks)
        
        # Store with dataset name
        result = {
            "dataset": ds_dir.name,
            **sums
        }
        dataset_results.append(result)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(dataset_dirs)} datasets...")
            
    except Exception as e:
        errors += 1
        print(f"❌ Error processing {ds_dir.name}: {e}")
        continue

successful_datasets = len(dataset_results)
print(f"\n✅ Successfully processed {successful_datasets} datasets")
print(f"   📁 Total datasets found: {total_datasets}")
print(f"   ✅ Successfully analyzed: {successful_datasets}")
print(f"   ⚠️  Skipped (no file): {skipped_no_file}")
print(f"   ⚠️  Skipped (no matching tasks): {skipped_no_tasks}")
print(f"   ❌ Errors: {errors}\n")

# Convert to numpy arrays for statistics
elapsed_times = np.array([r["elapsedTime"] for r in dataset_results])
execution_times = np.array([r["executionTime"] for r in dataset_results])
energies = np.array([r["energy"] for r in dataset_results])
compute_times = np.array([r["computeTime"] for r in dataset_results])
task_counts = np.array([r["taskCount"] for r in dataset_results])

# Print statistics
print("=" * 70)
print("STATISTICS")
print("=" * 70)
print(f"\n📊 DATASET ANALYSIS:")
print(f"   Total datasets found: {total_datasets}")
print(f"   Successfully analyzed: {successful_datasets} ({100*successful_datasets/total_datasets:.1f}%)")
print(f"   Skipped (no file): {skipped_no_file}")
print(f"   Skipped (no matching tasks): {skipped_no_tasks}")
print(f"   Errors: {errors}")
print(f"\n📊 Task Count per Dataset: {task_counts.mean():.2f} ± {task_counts.std():.2f} (range: {task_counts.min()}-{task_counts.max()})")

print(f"\n⏱️  ELAPSED TIME ANALYSIS (sum per dataset, taskId 0-4):")
print(f"   Count: {len(elapsed_times)} datasets")
print(f"   Mean: {elapsed_times.mean():.4f}s")
print(f"   Std:  {elapsed_times.std():.4f}s")
print(f"   Min:  {elapsed_times.min():.4f}s (dataset: {dataset_results[np.argmin(elapsed_times)]['dataset']})")
print(f"   Max:  {elapsed_times.max():.4f}s (dataset: {dataset_results[np.argmax(elapsed_times)]['dataset']})")
print(f"   Median (50th percentile): {np.median(elapsed_times):.4f}s")
print(f"   25th percentile (Q1): {np.percentile(elapsed_times, 25):.4f}s")
print(f"   75th percentile (Q3): {np.percentile(elapsed_times, 75):.4f}s")
print(f"   IQR (Q3-Q1): {np.percentile(elapsed_times, 75) - np.percentile(elapsed_times, 25):.4f}s")
print(f"   10th percentile: {np.percentile(elapsed_times, 10):.4f}s")
print(f"   90th percentile: {np.percentile(elapsed_times, 90):.4f}s")
print(f"   Variance: {elapsed_times.var():.6f}s²")
print(f"   Coefficient of Variation: {(elapsed_times.std()/elapsed_times.mean()*100):.2f}%")

print(f"\n⚡ Execution Time (sum per dataset):")
print(f"   Mean: {execution_times.mean():.4f}s")
print(f"   Std:  {execution_times.std():.4f}s")
print(f"   Min:  {execution_times.min():.4f}s")
print(f"   Max:  {execution_times.max():.4f}s")
print(f"   Median: {np.median(execution_times):.4f}s")

print(f"\n🔋 Energy (sum per dataset):")
print(f"   Mean: {energies.mean():.2e} kWh")
print(f"   Std:  {energies.std():.2e} kWh")
print(f"   Min:  {energies.min():.2e} kWh")
print(f"   Max:  {energies.max():.2e} kWh")
print(f"   Median: {np.median(energies):.2e} kWh")

print(f"\n💻 Compute Time (sum per dataset):")
print(f"   Mean: {compute_times.mean():.4f}s")
print(f"   Std:  {compute_times.std():.4f}s")
print(f"   Min:  {compute_times.min():.4f}s")
print(f"   Max:  {compute_times.max():.4f}s")
print(f"   Median: {np.median(compute_times):.4f}s")

# Create visualizations
print(f"\n📈 Generating visualizations...")

# Focused Elapsed Time visualization
fig_elapsed, axes_elapsed = plt.subplots(2, 2, figsize=(16, 12))
fig_elapsed.suptitle(f'Elapsed Time Distribution Analysis (taskId 0-4) - {successful_datasets} datasets', 
                     fontsize=16, fontweight='bold')

# Histogram: Elapsed Time (detailed)
axes_elapsed[0, 0].hist(elapsed_times, bins=80, edgecolor='black', alpha=0.7, color='steelblue', density=False)
axes_elapsed[0, 0].axvline(elapsed_times.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {elapsed_times.mean():.3f}s')
axes_elapsed[0, 0].axvline(np.median(elapsed_times), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(elapsed_times):.3f}s')
axes_elapsed[0, 0].axvline(np.percentile(elapsed_times, 25), color='green', linestyle=':', linewidth=1.5, label=f'Q1: {np.percentile(elapsed_times, 25):.3f}s')
axes_elapsed[0, 0].axvline(np.percentile(elapsed_times, 75), color='green', linestyle=':', linewidth=1.5, label=f'Q3: {np.percentile(elapsed_times, 75):.3f}s')
axes_elapsed[0, 0].set_xlabel('Sum of Elapsed Time (s)', fontsize=11)
axes_elapsed[0, 0].set_ylabel('Number of Datasets', fontsize=11)
axes_elapsed[0, 0].set_title(f'Distribution of Total Elapsed Time (n={successful_datasets})', fontsize=12, fontweight='bold')
axes_elapsed[0, 0].legend(fontsize=9)
axes_elapsed[0, 0].grid(True, alpha=0.3)

# Density plot: Elapsed Time
axes_elapsed[0, 1].hist(elapsed_times, bins=80, edgecolor='black', alpha=0.7, color='steelblue', density=True)
axes_elapsed[0, 1].axvline(elapsed_times.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {elapsed_times.mean():.3f}s')
axes_elapsed[0, 1].axvline(np.median(elapsed_times), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(elapsed_times):.3f}s')
axes_elapsed[0, 1].set_xlabel('Sum of Elapsed Time (s)', fontsize=11)
axes_elapsed[0, 1].set_ylabel('Density', fontsize=11)
axes_elapsed[0, 1].set_title('Probability Density of Elapsed Time', fontsize=12, fontweight='bold')
axes_elapsed[0, 1].legend(fontsize=9)
axes_elapsed[0, 1].grid(True, alpha=0.3)

# Box plot: Elapsed Time
bp = axes_elapsed[1, 0].boxplot([elapsed_times], labels=['Elapsed Time'], patch_artist=True, 
                                 showmeans=True, meanline=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][0].set_alpha(0.7)
axes_elapsed[1, 0].set_ylabel('Sum of Elapsed Time (s)', fontsize=11)
axes_elapsed[1, 0].set_title('Box Plot of Elapsed Time Distribution', fontsize=12, fontweight='bold')
axes_elapsed[1, 0].grid(True, alpha=0.3, axis='y')
# Add statistics text
stats_text = f'Mean: {elapsed_times.mean():.3f}s\nMedian: {np.median(elapsed_times):.3f}s\nQ1: {np.percentile(elapsed_times, 25):.3f}s\nQ3: {np.percentile(elapsed_times, 75):.3f}s\nIQR: {np.percentile(elapsed_times, 75) - np.percentile(elapsed_times, 25):.3f}s'
axes_elapsed[1, 0].text(1.3, np.median(elapsed_times), stats_text, fontsize=9, 
                        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Cumulative Distribution Function
sorted_times = np.sort(elapsed_times)
cumulative = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
axes_elapsed[1, 1].plot(sorted_times, cumulative, linewidth=2, color='darkblue', label='CDF')
axes_elapsed[1, 1].axhline(0.5, color='orange', linestyle='--', linewidth=1.5, label='50th percentile (Median)')
axes_elapsed[1, 1].axvline(np.median(elapsed_times), color='orange', linestyle='--', linewidth=1.5)
axes_elapsed[1, 1].set_xlabel('Sum of Elapsed Time (s)', fontsize=11)
axes_elapsed[1, 1].set_ylabel('Cumulative Probability', fontsize=11)
axes_elapsed[1, 1].set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
axes_elapsed[1, 1].legend(fontsize=9)
axes_elapsed[1, 1].grid(True, alpha=0.3)
axes_elapsed[1, 1].set_ylim([0, 1])

plt.tight_layout()
output_path_elapsed = "task_distribution_elapsed_time.png"
plt.savefig(output_path_elapsed, dpi=150, bbox_inches='tight')
plt.close(fig_elapsed)
print(f"✅ Saved elapsed time visualization to {output_path_elapsed}")

# General comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Task Results Distribution Across Datasets (taskId 0-4) - {successful_datasets} datasets', 
             fontsize=16, fontweight='bold')

# Histogram: Elapsed Time
axes[0, 0].hist(elapsed_times, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].axvline(elapsed_times.mean(), color='red', linestyle='--', label=f'Mean: {elapsed_times.mean():.3f}s')
axes[0, 0].axvline(np.median(elapsed_times), color='orange', linestyle='--', label=f'Median: {np.median(elapsed_times):.3f}s')
axes[0, 0].set_xlabel('Sum of Elapsed Time (s)')
axes[0, 0].set_ylabel('Number of Datasets')
axes[0, 0].set_title('Distribution of Total Elapsed Time')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Histogram: Execution Time
axes[0, 1].hist(execution_times, bins=50, edgecolor='black', alpha=0.7, color='forestgreen')
axes[0, 1].axvline(execution_times.mean(), color='red', linestyle='--', label=f'Mean: {execution_times.mean():.3f}s')
axes[0, 1].axvline(np.median(execution_times), color='orange', linestyle='--', label=f'Median: {np.median(execution_times):.3f}s')
axes[0, 1].set_xlabel('Sum of Execution Time (s)')
axes[0, 1].set_ylabel('Number of Datasets')
axes[0, 1].set_title('Distribution of Total Execution Time')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Histogram: Energy
axes[1, 0].hist(energies, bins=50, edgecolor='black', alpha=0.7, color='darkorange')
axes[1, 0].axvline(energies.mean(), color='red', linestyle='--', label=f'Mean: {energies.mean():.2e} kWh')
axes[1, 0].axvline(np.median(energies), color='orange', linestyle='--', label=f'Median: {np.median(energies):.2e} kWh')
axes[1, 0].set_xlabel('Sum of Energy (kWh)')
axes[1, 0].set_ylabel('Number of Datasets')
axes[1, 0].set_title('Distribution of Total Energy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Scatter: Elapsed Time vs Energy
axes[1, 1].scatter(elapsed_times, energies, alpha=0.5, s=20, color='purple')
axes[1, 1].set_xlabel('Sum of Elapsed Time (s)')
axes[1, 1].set_ylabel('Sum of Energy (kWh)')
axes[1, 1].set_title('Elapsed Time vs Energy')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
output_path = "task_distribution_analysis.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ Saved visualization to {output_path}")

# Additional detailed plots
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Additional Task Metrics Distribution', fontsize=16, fontweight='bold')

# Compute Time
axes2[0].hist(compute_times, bins=50, edgecolor='black', alpha=0.7, color='teal')
axes2[0].axvline(compute_times.mean(), color='red', linestyle='--', label=f'Mean: {compute_times.mean():.3f}s')
axes2[0].axvline(np.median(compute_times), color='orange', linestyle='--', label=f'Median: {np.median(compute_times):.3f}s')
axes2[0].set_xlabel('Sum of Compute Time (s)')
axes2[0].set_ylabel('Number of Datasets')
axes2[0].set_title('Distribution of Total Compute Time')
axes2[0].legend()
axes2[0].grid(True, alpha=0.3)

# Execution Time vs Compute Time
axes2[1].scatter(execution_times, compute_times, alpha=0.5, s=20, color='crimson')
axes2[1].set_xlabel('Sum of Execution Time (s)')
axes2[1].set_ylabel('Sum of Compute Time (s)')
axes2[1].set_title('Execution Time vs Compute Time')
axes2[1].grid(True, alpha=0.3)

plt.tight_layout()
output_path2 = "task_distribution_analysis_2.png"
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"✅ Saved additional visualization to {output_path2}")

print("\n" + "=" * 70)
print("✅ Analysis complete!")
print("=" * 70)

