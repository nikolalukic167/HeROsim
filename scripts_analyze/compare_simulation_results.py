#!/usr/bin/env python3
"""
Compare GNN, Knative, and RoundRobin simulation results with visualizations.

This script loads two or three simulation result JSON files and creates comprehensive
comparisons with visualizations of key metrics.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10


def load_json_file(filepath: Path) -> Dict[str, Any]:
    """Load a JSON file, handling large files efficiently."""
    print(f"Loading {filepath}...")
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"  ✓ Loaded successfully ({len(str(data)) / 1024 / 1024:.2f} MB)")
        return data
    except Exception as e:
        print(f"  ✗ Error loading {filepath}: {e}")
        sys.exit(1)


def extract_key_metrics(stats: Dict[str, Any]) -> Dict[str, float]:
    """Extract key metrics from stats dictionary."""
    metrics = {
        'averageElapsedTime': stats.get('averageElapsedTime', 0),
        'averageQueueTime': stats.get('averageQueueTime', 0),
        'averageWaitTime': stats.get('averageWaitTime', 0),
        'averageExecutionTime': stats.get('averageExecutionTime', 0),
        'averageColdStartTime': stats.get('averageColdStartTime', 0),
        'averagePullTime': stats.get('averagePullTime', 0),
        'averageComputeTime': stats.get('averageComputeTime', 0),
        'averageCommunicationsTime': stats.get('averageCommunicationsTime', 0),
        'penaltyProportion': stats.get('penaltyProportion', 0),
        'coldStartProportion': stats.get('coldStartProportion', 0),
        'taskCacheHitsProportion': stats.get('taskCacheHitsProportion', 0),
        'nodeCacheHitsProportion': stats.get('nodeCacheHitsProportion', 0),
        'averageOccupation': stats.get('averageOccupation', 0),
        'unusedPlatforms': stats.get('unusedPlatforms', 0),
        'unusedNodes': stats.get('unusedNodes', 0),
        'energy': stats.get('energy', 0),
        'reclaimableEnergy': stats.get('reclaimableEnergy', 0),
        'offloadingRate': stats.get('offloadingRate', 0),
        'endTime': stats.get('endTime', 0),
    }
    return metrics


def create_comparison_table(gnn_data: Dict, knative_data: Dict, output_dir: Path, roundrobin_data: Dict = None):
    """Create a comparison table of key metrics."""
    gnn_stats = gnn_data.get('stats', {})
    knative_stats = knative_data.get('stats', {})
    roundrobin_stats = roundrobin_data.get('stats', {}) if roundrobin_data else None
    
    gnn_metrics = extract_key_metrics(gnn_stats)
    knative_metrics = extract_key_metrics(knative_stats)
    roundrobin_metrics = extract_key_metrics(roundrobin_stats) if roundrobin_stats else None
    
    # Create comparison
    metrics_to_compare = [
        ('Average Elapsed Time (s)', 'averageElapsedTime', 'lower'),
        ('Average Queue Time (s)', 'averageQueueTime', 'lower'),
        ('Average Wait Time (s)', 'averageWaitTime', 'lower'),
        ('Average Execution Time (s)', 'averageExecutionTime', 'lower'),
        ('Average Cold Start Time (s)', 'averageColdStartTime', 'lower'),
        ('Average Pull Time (s)', 'averagePullTime', 'lower'),
        ('Average Compute Time (s)', 'averageComputeTime', 'lower'),
        ('Penalty Proportion (%)', 'penaltyProportion', 'lower'),
        ('Cold Start Proportion (%)', 'coldStartProportion', 'lower'),
        ('Task Cache Hits (%)', 'taskCacheHitsProportion', 'higher'),
        ('Node Cache Hits (%)', 'nodeCacheHitsProportion', 'higher'),
        ('Average Occupation (%)', 'averageOccupation', 'higher'),
        ('Unused Platforms (%)', 'unusedPlatforms', 'lower'),
        ('Unused Nodes (%)', 'unusedNodes', 'lower'),
        ('Energy (kWh)', 'energy', 'lower'),
        ('Reclaimable Energy (kWh)', 'reclaimableEnergy', 'lower'),
        ('Offloading Rate (%)', 'offloadingRate', 'neutral'),
        ('Simulation End Time (s)', 'endTime', 'neutral'),
    ]
    
    has_roundrobin = roundrobin_metrics is not None
    
    print("\n" + "="*80)
    print("KEY METRICS COMPARISON")
    print("="*80)
    if has_roundrobin:
        print(f"{'Metric':<35} {'GNN':<18} {'Knative':<18} {'RoundRobin':<18} {'Best':<10}")
        print("-"*100)
    else:
        print(f"{'Metric':<35} {'GNN':<20} {'Knative':<20} {'Difference':<15}")
        print("-"*80)
    
    results = []
    for metric_name, metric_key, better in metrics_to_compare:
        gnn_val = gnn_metrics.get(metric_key, 0)
        knative_val = knative_metrics.get(metric_key, 0)
        roundrobin_val = roundrobin_metrics.get(metric_key, 0) if roundrobin_metrics else None
        
        # Format values
        def format_val(val):
            if abs(val) < 0.01:
                return f"{val:.6f}"
            elif abs(val) < 1:
                return f"{val:.4f}"
            else:
                return f"{val:.2f}"
        
        gnn_str = format_val(gnn_val)
        knative_str = format_val(knative_val)
        
        if has_roundrobin:
            roundrobin_str = format_val(roundrobin_val)
            # Determine best
            if better == 'lower':
                best_val = min(gnn_val, knative_val, roundrobin_val)
                if best_val == gnn_val:
                    best = "GNN"
                elif best_val == knative_val:
                    best = "Knative"
                else:
                    best = "RoundRobin"
            elif better == 'higher':
                best_val = max(gnn_val, knative_val, roundrobin_val)
                if best_val == gnn_val:
                    best = "GNN"
                elif best_val == knative_val:
                    best = "Knative"
                else:
                    best = "RoundRobin"
            else:
                best = "N/A"
            
            print(f"{metric_name:<35} {gnn_str:<18} {knative_str:<18} {roundrobin_str:<18} {best:<10}")
            
            results.append({
                'metric': metric_name,
                'gnn': gnn_val,
                'knative': knative_val,
                'roundrobin': roundrobin_val,
                'better': better,
                'winner': best
            })
        else:
            if knative_val != 0:
                diff_pct = ((gnn_val - knative_val) / knative_val) * 100
            else:
                diff_pct = 0 if gnn_val == 0 else float('inf')
            
            diff_abs = gnn_val - knative_val
            
            if abs(diff_pct) < 0.01:
                diff_str = f"{diff_abs:+.4f} ({diff_pct:+.2f}%)"
            else:
                diff_str = f"{diff_abs:+.2f} ({diff_pct:+.1f}%)"
            
            # Determine winner
            if better == 'lower':
                winner = "GNN" if gnn_val < knative_val else "Knative"
            elif better == 'higher':
                winner = "GNN" if gnn_val > knative_val else "Knative"
            else:
                winner = "N/A"
            
            print(f"{metric_name:<35} {gnn_str:<20} {knative_str:<20} {diff_str:<15}")
            
            results.append({
                'metric': metric_name,
                'gnn': gnn_val,
                'knative': knative_val,
                'diff_pct': diff_pct,
                'better': better,
                'winner': winner
            })
    
    print("="*80)
    
    # Save to file
    output_file = output_dir / "comparison_table.txt"
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("KEY METRICS COMPARISON\n")
        f.write("="*80 + "\n")
        if has_roundrobin:
            f.write(f"{'Metric':<35} {'GNN':<18} {'Knative':<18} {'RoundRobin':<18} {'Best':<10}\n")
            f.write("-"*100 + "\n")
            for r in results:
                f.write(f"{r['metric']:<35} {r['gnn']:<18.4f} {r['knative']:<18.4f} "
                       f"{r['roundrobin']:<18.4f} {r['winner']:<10}\n")
        else:
            f.write(f"{'Metric':<35} {'GNN':<20} {'Knative':<20} {'Difference':<15}\n")
            f.write("-"*80 + "\n")
            for r in results:
                f.write(f"{r['metric']:<35} {r['gnn']:<20.4f} {r['knative']:<20.4f} "
                       f"{r['diff_pct']:+.2f}%\n")
        f.write("="*80 + "\n")
    
    print(f"\n✓ Comparison table saved to {output_file}")
    return results


def plot_response_time_distribution(gnn_data: Dict, knative_data: Dict, output_dir: Path, roundrobin_data: Dict = None):
    """Plot response time distribution comparison."""
    gnn_dist = gnn_data.get('stats', {}).get('taskResponseTimeDistribution', [])
    knative_dist = knative_data.get('stats', {}).get('taskResponseTimeDistribution', [])
    roundrobin_dist = roundrobin_data.get('stats', {}).get('taskResponseTimeDistribution', []) if roundrobin_data else None
    
    if not gnn_dist or not knative_dist:
        print("  ⚠ No response time distribution data available")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # CDF plot
    percentiles = np.linspace(0, 100, len(gnn_dist))
    ax1.plot(gnn_dist, percentiles, label='GNN', linewidth=2, color='#2ecc71')
    ax1.plot(knative_dist, percentiles, label='Knative', linewidth=2, color='#e74c3c')
    if roundrobin_dist:
        ax1.plot(roundrobin_dist, percentiles, label='RoundRobin', linewidth=2, color='#3498db')
    ax1.set_xlabel('Response Time (s)', fontsize=12)
    ax1.set_ylabel('Cumulative Probability (%)', fontsize=12)
    ax1.set_title('Response Time CDF Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Histogram comparison
    ax2.hist(gnn_dist, bins=50, alpha=0.6, label='GNN', color='#2ecc71', density=True)
    ax2.hist(knative_dist, bins=50, alpha=0.6, label='Knative', color='#e74c3c', density=True)
    if roundrobin_dist:
        ax2.hist(roundrobin_dist, bins=50, alpha=0.6, label='RoundRobin', color='#3498db', density=True)
    ax2.set_xlabel('Response Time (s)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Response Time Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "response_time_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Response time distribution saved to {output_file}")


def plot_metrics_comparison(gnn_data: Dict, knative_data: Dict, output_dir: Path, roundrobin_data: Dict = None):
    """Create bar chart comparison of key metrics."""
    gnn_stats = gnn_data.get('stats', {})
    knative_stats = knative_data.get('stats', {})
    roundrobin_stats = roundrobin_data.get('stats', {}) if roundrobin_data else None
    
    # Select metrics for visualization
    metrics = {
        'Average Elapsed Time (s)': ('averageElapsedTime', 1),
        'Average Queue Time (s)': ('averageQueueTime', 1),
        'Penalty Proportion (%)': ('penaltyProportion', 1),
        'Cold Start Proportion (%)': ('coldStartProportion', 1),
        'Task Cache Hits (%)': ('taskCacheHitsProportion', 1),
        'Average Occupation (%)': ('averageOccupation', 1),
        'Energy (kWh)': ('energy', 1e-6),  # Scale for readability
    }
    
    gnn_values = []
    knative_values = []
    roundrobin_values = []
    metric_names = []
    
    for name, (key, scale) in metrics.items():
        gnn_val = gnn_stats.get(key, 0) * scale
        knative_val = knative_stats.get(key, 0) * scale
        roundrobin_val = (roundrobin_stats.get(key, 0) * scale) if roundrobin_stats else None
        
        if gnn_val > 0 or knative_val > 0 or (roundrobin_val and roundrobin_val > 0):  # Only include if at least one is non-zero
            gnn_values.append(gnn_val)
            knative_values.append(knative_val)
            if roundrobin_val is not None:
                roundrobin_values.append(roundrobin_val)
            metric_names.append(name)
    
    x = np.arange(len(metric_names))
    has_roundrobin = roundrobin_data is not None and len(roundrobin_values) > 0
    
    if has_roundrobin:
        width = 0.25
        offset = width
    else:
        width = 0.35
        offset = width
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars1 = ax.bar(x - offset, gnn_values, width, label='GNN', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, knative_values, width, label='Knative', color='#e74c3c', alpha=0.8)
    if has_roundrobin:
        bars3 = ax.bar(x + offset, roundrobin_values, width, label='RoundRobin', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    title = 'Key Metrics Comparison: GNN vs Knative vs RoundRobin' if has_roundrobin else 'Key Metrics Comparison: GNN vs Knative'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2] + ([bars3] if has_roundrobin else []):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_file = output_dir / "metrics_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Metrics comparison saved to {output_file}")


def plot_time_breakdown(gnn_data: Dict, knative_data: Dict, output_dir: Path, roundrobin_data: Dict = None):
    """Plot time breakdown comparison."""
    gnn_stats = gnn_data.get('stats', {})
    knative_stats = knative_data.get('stats', {})
    roundrobin_stats = roundrobin_data.get('stats', {}) if roundrobin_data else None
    
    time_components = {
        'Queue Time': 'averageQueueTime',
        'Wait Time': 'averageWaitTime',
        'Pull Time': 'averagePullTime',
        'Cold Start Time': 'averageColdStartTime',
        'Execution Time': 'averageExecutionTime',
        'Compute Time': 'averageComputeTime',
        'Communications Time': 'averageCommunicationsTime',
    }
    
    gnn_times = [gnn_stats.get(key, 0) for key in time_components.values()]
    knative_times = [knative_stats.get(key, 0) for key in time_components.values()]
    roundrobin_times = [roundrobin_stats.get(key, 0) for key in time_components.values()] if roundrobin_stats else None
    labels = list(time_components.keys())
    
    has_roundrobin = roundrobin_times is not None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Stacked bar chart
    if has_roundrobin:
        x = np.arange(3)
        width = 0.6
        x_labels = ['GNN', 'Knative', 'RoundRobin']
    else:
        x = np.arange(2)
        width = 0.6
        x_labels = ['GNN', 'Knative']
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    for i, label in enumerate(labels):
        gnn_val = gnn_times[i]
        knative_val = knative_times[i]
        roundrobin_val = roundrobin_times[i] if has_roundrobin else None
        
        if i == 0:
            ax1.bar(x[0], gnn_val, width, label=label, color=colors[i], bottom=0)
            ax1.bar(x[1], knative_val, width, color=colors[i], bottom=0)
            gnn_bottom = gnn_val
            knative_bottom = knative_val
            if has_roundrobin:
                ax1.bar(x[2], roundrobin_val, width, color=colors[i], bottom=0)
                roundrobin_bottom = roundrobin_val
        else:
            ax1.bar(x[0], gnn_val, width, label=label, color=colors[i], bottom=gnn_bottom)
            ax1.bar(x[1], knative_val, width, color=colors[i], bottom=knative_bottom)
            gnn_bottom += gnn_val
            knative_bottom += knative_val
            if has_roundrobin:
                ax1.bar(x[2], roundrobin_val, width, color=colors[i], bottom=roundrobin_bottom)
                roundrobin_bottom += roundrobin_val
    
    ax1.set_ylabel('Time (s)', fontsize=12)
    ax1.set_title('Time Component Breakdown (Stacked)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Side-by-side comparison
    x = np.arange(len(labels))
    if has_roundrobin:
        width = 0.25
        offset = width
    else:
        width = 0.35
        offset = width
    
    ax2.bar(x - offset, gnn_times, width, label='GNN', color='#2ecc71', alpha=0.8)
    ax2.bar(x, knative_times, width, label='Knative', color='#e74c3c', alpha=0.8)
    if has_roundrobin:
        ax2.bar(x + offset, roundrobin_times, width, label='RoundRobin', color='#3498db', alpha=0.8)
    
    ax2.set_xlabel('Time Component', fontsize=12)
    ax2.set_ylabel('Time (s)', fontsize=12)
    ax2.set_title('Time Component Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_dir / "time_breakdown.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Time breakdown saved to {output_file}")


def plot_proportions_comparison(gnn_data: Dict, knative_data: Dict, output_dir: Path, roundrobin_data: Dict = None):
    """Plot proportion metrics comparison."""
    gnn_stats = gnn_data.get('stats', {})
    knative_stats = knative_data.get('stats', {})
    roundrobin_stats = roundrobin_data.get('stats', {}) if roundrobin_data else None
    
    proportions = {
        'Penalty Proportion': 'penaltyProportion',
        'Cold Start Proportion': 'coldStartProportion',
        'Task Cache Hits': 'taskCacheHitsProportion',
        'Node Cache Hits': 'nodeCacheHitsProportion',
        'Unused Platforms': 'unusedPlatforms',
        'Unused Nodes': 'unusedNodes',
        'Offloading Rate': 'offloadingRate',
    }
    
    gnn_values = [gnn_stats.get(key, 0) for key in proportions.values()]
    knative_values = [knative_stats.get(key, 0) for key in proportions.values()]
    roundrobin_values = [roundrobin_stats.get(key, 0) for key in proportions.values()] if roundrobin_stats else None
    labels = list(proportions.keys())
    
    x = np.arange(len(labels))
    has_roundrobin = roundrobin_values is not None
    
    if has_roundrobin:
        width = 0.25
        offset = width
    else:
        width = 0.35
        offset = width
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars1 = ax.bar(x - offset, gnn_values, width, label='GNN', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, knative_values, width, label='Knative', color='#e74c3c', alpha=0.8)
    if has_roundrobin:
        bars3 = ax.bar(x + offset, roundrobin_values, width, label='RoundRobin', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Proportion Metrics', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    title = 'Proportion Metrics Comparison: GNN vs Knative vs RoundRobin' if has_roundrobin else 'Proportion Metrics Comparison: GNN vs Knative'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2] + ([bars3] if has_roundrobin else []):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_file = output_dir / "proportions_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Proportions comparison saved to {output_file}")


def plot_energy_comparison(gnn_data: Dict, knative_data: Dict, output_dir: Path, roundrobin_data: Dict = None):
    """Plot energy consumption comparison."""
    gnn_stats = gnn_data.get('stats', {})
    knative_stats = knative_data.get('stats', {})
    roundrobin_stats = roundrobin_data.get('stats', {}) if roundrobin_data else None
    
    energy_metrics = {
        'Total Energy': 'energy',
        'Reclaimable Energy': 'reclaimableEnergy',
    }
    
    gnn_energy = [gnn_stats.get(key, 0) for key in energy_metrics.values()]
    knative_energy = [knative_stats.get(key, 0) for key in energy_metrics.values()]
    roundrobin_energy = [roundrobin_stats.get(key, 0) for key in energy_metrics.values()] if roundrobin_stats else None
    labels = list(energy_metrics.keys())
    
    has_roundrobin = roundrobin_energy is not None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart
    x = np.arange(len(labels))
    if has_roundrobin:
        width = 0.25
        offset = width
    else:
        width = 0.35
        offset = width
    
    bars1 = ax1.bar(x - offset, gnn_energy, width, label='GNN', color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x, knative_energy, width, label='Knative', color='#e74c3c', alpha=0.8)
    if has_roundrobin:
        bars3 = ax1.bar(x + offset, roundrobin_energy, width, label='RoundRobin', color='#3498db', alpha=0.8)
    
    ax1.set_xlabel('Energy Type', fontsize=12)
    ax1.set_ylabel('Energy (kWh)', fontsize=12)
    ax1.set_title('Energy Consumption Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2] + ([bars3] if has_roundrobin else []):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.6f}',
                        ha='center', va='bottom', fontsize=9)
    
    # Pie chart for energy efficiency
    if has_roundrobin:
        if sum(gnn_energy) > 0 or sum(knative_energy) > 0 or sum(roundrobin_energy) > 0:
            total_gnn = sum(gnn_energy)
            total_knative = sum(knative_energy)
            total_roundrobin = sum(roundrobin_energy)
            
            ax2.pie([total_gnn, total_knative, total_roundrobin], 
                   labels=['GNN', 'Knative', 'RoundRobin'],
                   autopct='%1.2f%%',
                   colors=['#2ecc71', '#e74c3c', '#3498db'],
                   startangle=90)
            ax2.set_title('Total Energy Distribution', fontsize=14, fontweight='bold')
    else:
        if sum(gnn_energy) > 0 and sum(knative_energy) > 0:
            total_gnn = sum(gnn_energy)
            total_knative = sum(knative_energy)
            
            ax2.pie([total_gnn, total_knative], 
                   labels=['GNN', 'Knative'],
                   autopct='%1.2f%%',
                   colors=['#2ecc71', '#e74c3c'],
                   startangle=90)
            ax2.set_title('Total Energy Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / "energy_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Energy comparison saved to {output_file}")


def create_summary_dashboard(gnn_data: Dict, knative_data: Dict, output_dir: Path, roundrobin_data: Dict = None):
    """Create a comprehensive summary dashboard."""
    gnn_stats = gnn_data.get('stats', {})
    knative_stats = knative_data.get('stats', {})
    roundrobin_stats = roundrobin_data.get('stats', {}) if roundrobin_data else None
    
    has_roundrobin = roundrobin_stats is not None
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Response Time CDF
    ax1 = fig.add_subplot(gs[0, 0])
    gnn_dist = gnn_stats.get('taskResponseTimeDistribution', [])
    knative_dist = knative_stats.get('taskResponseTimeDistribution', [])
    roundrobin_dist = roundrobin_stats.get('taskResponseTimeDistribution', []) if roundrobin_stats else None
    if gnn_dist and knative_dist:
        percentiles = np.linspace(0, 100, len(gnn_dist))
        ax1.plot(gnn_dist, percentiles, label='GNN', linewidth=2, color='#2ecc71')
        ax1.plot(knative_dist, percentiles, label='Knative', linewidth=2, color='#e74c3c')
        if roundrobin_dist:
            ax1.plot(roundrobin_dist, percentiles, label='RoundRobin', linewidth=2, color='#3498db')
        ax1.set_xlabel('Response Time (s)')
        ax1.set_ylabel('Cumulative %')
        ax1.set_title('Response Time CDF', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Key Metrics Bar Chart
    ax2 = fig.add_subplot(gs[0, 1])
    metrics = ['Elapsed Time', 'Queue Time', 'Penalty %', 'Cache Hits %']
    gnn_vals = [
        gnn_stats.get('averageElapsedTime', 0),
        gnn_stats.get('averageQueueTime', 0),
        gnn_stats.get('penaltyProportion', 0),
        gnn_stats.get('taskCacheHitsProportion', 0)
    ]
    knative_vals = [
        knative_stats.get('averageElapsedTime', 0),
        knative_stats.get('averageQueueTime', 0),
        knative_stats.get('penaltyProportion', 0),
        knative_stats.get('taskCacheHitsProportion', 0)
    ]
    roundrobin_vals = [
        roundrobin_stats.get('averageElapsedTime', 0),
        roundrobin_stats.get('averageQueueTime', 0),
        roundrobin_stats.get('penaltyProportion', 0),
        roundrobin_stats.get('taskCacheHitsProportion', 0)
    ] if roundrobin_stats else None
    x = np.arange(len(metrics))
    if has_roundrobin:
        width = 0.25
        offset = width
    else:
        width = 0.35
        offset = width
    ax2.bar(x - offset, gnn_vals, width, label='GNN', color='#2ecc71', alpha=0.8)
    ax2.bar(x, knative_vals, width, label='Knative', color='#e74c3c', alpha=0.8)
    if has_roundrobin:
        ax2.bar(x + offset, roundrobin_vals, width, label='RoundRobin', color='#3498db', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, rotation=45, ha='right')
    ax2.set_ylabel('Value')
    ax2.set_title('Key Metrics', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Proportions
    ax3 = fig.add_subplot(gs[0, 2])
    props = ['Penalty', 'Cold Start', 'Cache Hits', 'Unused Platforms']
    gnn_props = [
        gnn_stats.get('penaltyProportion', 0),
        gnn_stats.get('coldStartProportion', 0),
        gnn_stats.get('taskCacheHitsProportion', 0),
        gnn_stats.get('unusedPlatforms', 0)
    ]
    knative_props = [
        knative_stats.get('penaltyProportion', 0),
        knative_stats.get('coldStartProportion', 0),
        knative_stats.get('taskCacheHitsProportion', 0),
        knative_stats.get('unusedPlatforms', 0)
    ]
    roundrobin_props = [
        roundrobin_stats.get('penaltyProportion', 0),
        roundrobin_stats.get('coldStartProportion', 0),
        roundrobin_stats.get('taskCacheHitsProportion', 0),
        roundrobin_stats.get('unusedPlatforms', 0)
    ] if roundrobin_stats else None
    x = np.arange(len(props))
    ax3.bar(x - offset, gnn_props, width, label='GNN', color='#2ecc71', alpha=0.8)
    ax3.bar(x, knative_props, width, label='Knative', color='#e74c3c', alpha=0.8)
    if has_roundrobin:
        ax3.bar(x + offset, roundrobin_props, width, label='RoundRobin', color='#3498db', alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(props, rotation=45, ha='right')
    ax3.set_ylabel('Percentage (%)')
    ax3.set_title('Proportions', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Time Breakdown
    ax4 = fig.add_subplot(gs[1, :])
    time_labels = ['Queue', 'Wait', 'Pull', 'Cold Start', 'Execution', 'Compute', 'Comm']
    time_keys = ['averageQueueTime', 'averageWaitTime', 'averagePullTime', 
                 'averageColdStartTime', 'averageExecutionTime', 'averageComputeTime', 
                 'averageCommunicationsTime']
    gnn_times = [gnn_stats.get(k, 0) for k in time_keys]
    knative_times = [knative_stats.get(k, 0) for k in time_keys]
    roundrobin_times = [roundrobin_stats.get(k, 0) for k in time_keys] if roundrobin_stats else None
    x = np.arange(len(time_labels))
    ax4.bar(x - offset, gnn_times, width, label='GNN', color='#2ecc71', alpha=0.8)
    ax4.bar(x, knative_times, width, label='Knative', color='#e74c3c', alpha=0.8)
    if has_roundrobin:
        ax4.bar(x + offset, roundrobin_times, width, label='RoundRobin', color='#3498db', alpha=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(time_labels)
    ax4.set_ylabel('Time (s)')
    ax4.set_title('Time Component Breakdown', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Energy
    ax5 = fig.add_subplot(gs[2, 0])
    energy_labels = ['Total', 'Reclaimable']
    gnn_energy = [gnn_stats.get('energy', 0), gnn_stats.get('reclaimableEnergy', 0)]
    knative_energy = [knative_stats.get('energy', 0), knative_stats.get('reclaimableEnergy', 0)]
    roundrobin_energy = [roundrobin_stats.get('energy', 0), roundrobin_stats.get('reclaimableEnergy', 0)] if roundrobin_stats else None
    x = np.arange(len(energy_labels))
    ax5.bar(x - offset, gnn_energy, width, label='GNN', color='#2ecc71', alpha=0.8)
    ax5.bar(x, knative_energy, width, label='Knative', color='#e74c3c', alpha=0.8)
    if has_roundrobin:
        ax5.bar(x + offset, roundrobin_energy, width, label='RoundRobin', color='#3498db', alpha=0.8)
    ax5.set_xticks(x)
    ax5.set_xticklabels(energy_labels)
    ax5.set_ylabel('Energy (kWh)')
    ax5.set_title('Energy Consumption', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Resource Utilization
    ax6 = fig.add_subplot(gs[2, 1])
    util_labels = ['Avg Occupation', 'Unused Platforms', 'Unused Nodes']
    gnn_util = [
        gnn_stats.get('averageOccupation', 0) * 100,
        gnn_stats.get('unusedPlatforms', 0),
        gnn_stats.get('unusedNodes', 0)
    ]
    knative_util = [
        knative_stats.get('averageOccupation', 0) * 100,
        knative_stats.get('unusedPlatforms', 0),
        knative_stats.get('unusedNodes', 0)
    ]
    roundrobin_util = [
        roundrobin_stats.get('averageOccupation', 0) * 100,
        roundrobin_stats.get('unusedPlatforms', 0),
        roundrobin_stats.get('unusedNodes', 0)
    ] if roundrobin_stats else None
    x = np.arange(len(util_labels))
    ax6.bar(x - offset, gnn_util, width, label='GNN', color='#2ecc71', alpha=0.8)
    ax6.bar(x, knative_util, width, label='Knative', color='#e74c3c', alpha=0.8)
    if has_roundrobin:
        ax6.bar(x + offset, roundrobin_util, width, label='RoundRobin', color='#3498db', alpha=0.8)
    ax6.set_xticks(x)
    ax6.set_xticklabels(util_labels, rotation=45, ha='right')
    ax6.set_ylabel('Percentage (%)')
    ax6.set_title('Resource Utilization', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Summary Text
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    gnn_elapsed = gnn_stats.get('averageElapsedTime', 0)
    knative_elapsed = knative_stats.get('averageElapsedTime', 0)
    roundrobin_elapsed = roundrobin_stats.get('averageElapsedTime', 0) if roundrobin_stats else None
    gnn_penalty = gnn_stats.get('penaltyProportion', 0)
    knative_penalty = knative_stats.get('penaltyProportion', 0)
    roundrobin_penalty = roundrobin_stats.get('penaltyProportion', 0) if roundrobin_stats else None
    
    if has_roundrobin:
        summary_text = f"""
    SIMULATION SUMMARY
    
    GNN Results:
    • Avg Elapsed Time: {gnn_elapsed:.2f}s
    • Penalty Rate: {gnn_penalty:.1f}%
    • End Time: {gnn_stats.get('endTime', 0):.1f}s
    • Total Tasks: {gnn_data.get('num_tasks', 0)}
    
    Knative Results:
    • Avg Elapsed Time: {knative_elapsed:.2f}s
    • Penalty Rate: {knative_penalty:.1f}%
    • End Time: {knative_stats.get('endTime', 0):.1f}s
    • Total Tasks: {knative_data.get('num_tasks', 0)}
    
    RoundRobin Results:
    • Avg Elapsed Time: {roundrobin_elapsed:.2f}s
    • Penalty Rate: {roundrobin_penalty:.1f}%
    • End Time: {roundrobin_stats.get('endTime', 0):.1f}s
    • Total Tasks: {roundrobin_data.get('num_tasks', 0)}
    """
    else:
        summary_text = f"""
    SIMULATION SUMMARY
    
    GNN Results:
    • Avg Elapsed Time: {gnn_elapsed:.2f}s
    • Penalty Rate: {gnn_penalty:.1f}%
    • End Time: {gnn_stats.get('endTime', 0):.1f}s
    • Total Tasks: {gnn_data.get('num_tasks', 0)}
    
    Knative Results:
    • Avg Elapsed Time: {knative_elapsed:.2f}s
    • Penalty Rate: {knative_penalty:.1f}%
    • End Time: {knative_stats.get('endTime', 0):.1f}s
    • Total Tasks: {knative_data.get('num_tasks', 0)}
    
    Comparison:
    • Elapsed Time Diff: {gnn_elapsed - knative_elapsed:+.2f}s
    • Penalty Rate Diff: {gnn_penalty - knative_penalty:+.1f}%
    """
    ax7.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    title = 'Simulation Results Comparison: GNN vs Knative vs RoundRobin' if has_roundrobin else 'Simulation Results Comparison: GNN vs Knative'
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    output_file = output_dir / "summary_dashboard.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Summary dashboard saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare GNN, Knative, and RoundRobin simulation results with visualizations'
    )
    parser.add_argument(
        '--gnn-file',
        type=str,
        default='/root/projects/my-herosim/simulation_data/results/simulation_result_gnn.json',
        help='Path to GNN simulation result JSON file'
    )
    parser.add_argument(
        '--knative-file',
        type=str,
        default='/root/projects/my-herosim/simulation_data/results/simulation_result_knative.json',
        help='Path to Knative simulation result JSON file'
    )
    parser.add_argument(
        '--roundrobin-file',
        type=str,
        default=None,
        help='Path to RoundRobin simulation result JSON file (optional)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/root/projects/my-herosim/simulation_data/results/comparison',
        help='Output directory for comparison results'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("SIMULATION RESULTS COMPARISON")
    print("="*80)
    print(f"GNN File: {args.gnn_file}")
    print(f"Knative File: {args.knative_file}")
    if args.roundrobin_file:
        print(f"RoundRobin File: {args.roundrobin_file}")
    print(f"Output Directory: {output_dir}")
    print("="*80)
    
    # Load data
    gnn_data = load_json_file(Path(args.gnn_file))
    knative_data = load_json_file(Path(args.knative_file))
    roundrobin_data = None
    if args.roundrobin_file:
        roundrobin_data = load_json_file(Path(args.roundrobin_file))
    
    # Create comparisons
    print("\n" + "="*80)
    print("GENERATING COMPARISONS")
    print("="*80)
    
    # 1. Comparison table
    print("\n1. Creating comparison table...")
    create_comparison_table(gnn_data, knative_data, output_dir, roundrobin_data)
    
    # 2. Visualizations
    print("\n2. Creating visualizations...")
    plot_response_time_distribution(gnn_data, knative_data, output_dir, roundrobin_data)
    plot_metrics_comparison(gnn_data, knative_data, output_dir, roundrobin_data)
    plot_time_breakdown(gnn_data, knative_data, output_dir, roundrobin_data)
    plot_proportions_comparison(gnn_data, knative_data, output_dir, roundrobin_data)
    plot_energy_comparison(gnn_data, knative_data, output_dir, roundrobin_data)
    create_summary_dashboard(gnn_data, knative_data, output_dir, roundrobin_data)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"All results saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()

