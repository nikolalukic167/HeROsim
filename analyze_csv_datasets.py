#!/usr/bin/env python3
"""Minimal script to analyze CSV datasets in gnn_datasets directory."""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from collections import defaultdict

def count_csv_files(base_dir):
    """Count CSV files in each dataset directory."""
    base_path = Path(base_dir)
    dataset_counts = {}
    
    # Sort dataset directories by numeric order
    dataset_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('ds_')],
                         key=lambda x: int(x.name.split('_')[1]))
    
    for ds_dir in dataset_dirs:
        # Count all CSV files recursively in this dataset directory
        csv_count = len(list(ds_dir.rglob('*.csv')))
        dataset_num = int(ds_dir.name.split('_')[1])
        dataset_counts[dataset_num] = csv_count
    
    return dataset_counts

def analyze_growth(csv_counts, dataset_nums):
    """Perform deep statistical analysis to prove it's a growing function."""
    counts_array = np.array(csv_counts)
    indices_array = np.array(dataset_nums)
    
    # Linear regression to find trend
    slope, intercept, r_value, p_value, std_err = stats.linregress(indices_array, counts_array)
    
    # Pearson correlation coefficient
    correlation, corr_p_value = stats.pearsonr(indices_array, counts_array)
    
    # Test for monotonic growth (Kendall's tau)
    tau, tau_p_value = stats.kendalltau(indices_array, counts_array)
    
    # Count increases vs decreases
    diffs = np.diff(counts_array)
    num_increases = np.sum(diffs > 0)
    num_decreases = np.sum(diffs < 0)
    num_stable = np.sum(diffs == 0)
    
    # Percentage of increases
    pct_increases = (num_increases / len(diffs)) * 100
    
    # Compare first half vs second half
    mid = len(counts_array) // 2
    first_half_mean = np.mean(counts_array[:mid])
    second_half_mean = np.mean(counts_array[mid:])
    
    # Window-based analysis (moving average)
    window_size = min(50, len(counts_array) // 10)
    if window_size > 1:
        moving_avg = np.convolve(counts_array, np.ones(window_size)/window_size, mode='valid')
        moving_avg_indices = indices_array[window_size-1:]
        moving_slope, _, _, _, _ = stats.linregress(moving_avg_indices, moving_avg)
    else:
        moving_avg = None
        moving_slope = None
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'correlation': correlation,
        'corr_p_value': corr_p_value,
        'tau': tau,
        'tau_p_value': tau_p_value,
        'num_increases': num_increases,
        'num_decreases': num_decreases,
        'num_stable': num_stable,
        'pct_increases': pct_increases,
        'first_half_mean': first_half_mean,
        'second_half_mean': second_half_mean,
        'moving_avg': moving_avg,
        'moving_slope': moving_slope,
        'moving_avg_indices': moving_avg_indices if window_size > 1 else None
    }

def visualize(counts):
    """Visualize CSV counts per dataset with deep analysis."""
    sorted_items = sorted(counts.items())
    dataset_nums = [item[0] for item in sorted_items]
    csv_counts = [item[1] for item in sorted_items]
    
    # Filter out datasets with no CSV files
    original_count = len(dataset_nums)
    filtered_data = [(idx, count) for idx, count in zip(dataset_nums, csv_counts) if count > 0]
    dataset_nums = [item[0] for item in filtered_data]
    csv_counts = [item[1] for item in filtered_data]
    excluded_count = original_count - len(dataset_nums)
    
    if excluded_count > 0:
        print(f"\nFiltered out {excluded_count} dataset(s) with 0 CSV files")
        print(f"Analyzing {len(dataset_nums)} datasets with CSV files")
    
    # Deep statistical analysis
    analysis = analyze_growth(csv_counts, dataset_nums)
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top plot: Raw data with trend line
    ax1 = axes[0]
    ax1.plot(dataset_nums, csv_counts, 'b-', linewidth=1.5, alpha=0.7, label='CSV Count per Dataset')
    
    # Add linear trend line
    trend_line = analysis['slope'] * np.array(dataset_nums) + analysis['intercept']
    ax1.plot(dataset_nums, trend_line, 'r--', linewidth=2, label=f"Linear Trend (slope={analysis['slope']:.4f})")
    
    # Add moving average if available
    if analysis['moving_avg'] is not None:
        ax1.plot(analysis['moving_avg_indices'], analysis['moving_avg'], 
                'g-', linewidth=2, alpha=0.8, label=f"Moving Average (window={min(50, len(dataset_nums)//10)})")
    
    ax1.set_xlabel('Dataset Index')
    ax1.set_ylabel('CSV Count per Dataset')
    ax1.set_title('CSV Dataset Analysis - Count per Dataset with Growth Trend')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Bottom plot: First differences (change between consecutive datasets)
    ax2 = axes[1]
    diffs = np.diff(csv_counts)
    diff_indices = dataset_nums[1:]
    colors = ['green' if d > 0 else 'red' if d < 0 else 'gray' for d in diffs]
    ax2.bar(diff_indices, diffs, color=colors, alpha=0.6, width=1)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Dataset Index')
    ax2.set_ylabel('Change in CSV Count')
    ax2.set_title('First Differences (Positive = Growth, Negative = Decline)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('csv_dataset_analysis.png', dpi=150)
    print(f"Visualization saved to csv_dataset_analysis.png")
    
    # Print detailed statistics
    total_csv_files = sum(csv_counts)
    num_datasets = len(dataset_nums)
    avg_csv_per_dataset = total_csv_files / num_datasets if num_datasets > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"BASIC SUMMARY")
    print(f"{'='*60}")
    print(f"  Datasets analyzed: {num_datasets} (excluded {excluded_count} with 0 CSV files)")
    print(f"  Total CSV files: {total_csv_files:,}")
    print(f"  Average CSV files per dataset: {avg_csv_per_dataset:.2f}")
    print(f"  Min CSV count: {min(csv_counts)}")
    print(f"  Max CSV count: {max(csv_counts)}")
    print(f"  Median CSV count: {np.median(csv_counts):.2f}")
    print(f"  Std deviation: {np.std(csv_counts):.2f}")
    print(f"  Dataset index range: {min(dataset_nums)} to {max(dataset_nums)}")
    
    print(f"\n{'='*60}")
    print(f"GROWTH ANALYSIS - PROVING IT'S A GROWING FUNCTION")
    print(f"{'='*60}")
    print(f"\n1. LINEAR REGRESSION:")
    print(f"   Slope: {analysis['slope']:.6f} (positive = growing)")
    print(f"   R²: {analysis['r_squared']:.4f} ({analysis['r_squared']*100:.2f}% variance explained)")
    print(f"   P-value: {analysis['p_value']:.2e}")
    print(f"   Interpretation: {'STRONG POSITIVE GROWTH' if analysis['slope'] > 0 and analysis['p_value'] < 0.05 else 'No significant growth' if analysis['p_value'] >= 0.05 else 'Declining'}")
    
    print(f"\n2. CORRELATION ANALYSIS:")
    print(f"   Pearson correlation: {analysis['correlation']:.4f}")
    print(f"   P-value: {analysis['corr_p_value']:.2e}")
    print(f"   Interpretation: {'STRONG POSITIVE CORRELATION' if analysis['correlation'] > 0.7 and analysis['corr_p_value'] < 0.05 else 'Moderate correlation' if analysis['correlation'] > 0.3 else 'Weak/no correlation'}")
    
    print(f"\n3. MONOTONIC GROWTH TEST (Kendall's Tau):")
    print(f"   Tau: {analysis['tau']:.4f}")
    print(f"   P-value: {analysis['tau_p_value']:.2e}")
    print(f"   Interpretation: {'MONOTONICALLY INCREASING' if analysis['tau'] > 0.5 and analysis['tau_p_value'] < 0.05 else 'Not monotonically increasing'}")
    
    print(f"\n4. INCREASE/DECREASE ANALYSIS:")
    print(f"   Number of increases: {analysis['num_increases']}")
    print(f"   Number of decreases: {analysis['num_decreases']}")
    print(f"   Number of stable: {analysis['num_stable']}")
    print(f"   Percentage increases: {analysis['pct_increases']:.2f}%")
    print(f"   Interpretation: {'GROWING TREND' if analysis['pct_increases'] > 50 else 'Declining/Stable trend'}")
    
    print(f"\n5. HALF COMPARISON:")
    print(f"   First half mean: {analysis['first_half_mean']:.2f}")
    print(f"   Second half mean: {analysis['second_half_mean']:.2f}")
    print(f"   Difference: {analysis['second_half_mean'] - analysis['first_half_mean']:.2f}")
    print(f"   Interpretation: {'SECOND HALF HAS MORE CSVs' if analysis['second_half_mean'] > analysis['first_half_mean'] else 'First half has more CSVs'}")
    
    if analysis['moving_slope'] is not None:
        print(f"\n6. MOVING AVERAGE TREND:")
        print(f"   Moving average slope: {analysis['moving_slope']:.6f}")
        print(f"   Interpretation: {'SMOOTHED GROWTH CONFIRMED' if analysis['moving_slope'] > 0 else 'Smoothed decline'}")
    
    print(f"\n{'='*60}")
    print(f"CONCLUSION:")
    # Multiple lines of evidence for growth
    has_positive_slope = analysis['slope'] > 0 and analysis['p_value'] < 0.05
    has_half_growth = analysis['second_half_mean'] > analysis['first_half_mean']
    has_moving_growth = analysis['moving_slope'] is not None and analysis['moving_slope'] > 0
    significant_correlation = analysis['correlation'] > 0 and analysis['corr_p_value'] < 0.05
    
    evidence_count = sum([has_positive_slope, has_half_growth, has_moving_growth, significant_correlation])
    
    if has_positive_slope and has_half_growth and has_moving_growth:
        print(f"✓ PROVEN: This is a GROWING FUNCTION")
        print(f"  Evidence:")
        if has_positive_slope:
            print(f"    • Positive linear slope (p={analysis['p_value']:.2e})")
        if has_half_growth:
            print(f"    • Second half has {analysis['second_half_mean'] - analysis['first_half_mean']:.2f} more CSVs on average")
        if has_moving_growth:
            print(f"    • Moving average shows positive trend")
        if significant_correlation:
            print(f"    • Significant positive correlation (r={analysis['correlation']:.4f})")
        print(f"  Note: While not strictly monotonic step-by-step, the overall trend is clearly upward")
    else:
        print(f"⚠ PARTIAL EVIDENCE: Growth detected but not conclusive")
        print(f"  Evidence score: {evidence_count}/4 indicators positive")
        print(f"  Review statistics above for details")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    base_dir = "/root/projects/my-herosim/simulation_data/artifacts/run10_all/gnn_datasets"
    
    print("Analyzing CSV datasets...")
    counts = count_csv_files(base_dir)
    
    visualize(counts)

