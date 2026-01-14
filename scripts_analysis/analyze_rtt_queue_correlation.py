#!/usr/bin/env python3
"""
Analyze timing breakdown of simulation results.

This script loads a simulation result JSON file and analyzes timing metrics
including execution time, cold start duration, image pull time, queue time,
initialization time, compute time, communications time, etc.

Usage:
    python scripts_analysis/analyze_rtt_queue_correlation.py <simulation_result.json>
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt

def load_simulation_result(filepath: Path) -> Dict:
    """Load simulation result JSON file."""
    print(f"Loading {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"  ✓ Loaded successfully")
    return data

def extract_task_metrics(data: Dict) -> List[Dict]:
    """Extract task-level metrics from simulation results."""
    task_results = data.get('stats', {}).get('taskResults', [])
    
    tasks = []
    for task in task_results:
        # Only include real tasks (taskId >= 0)
        if task.get('taskId') is not None and task.get('taskId') >= 0:
            tasks.append({
                'task_id': task.get('taskId'),
                'elapsed_time': task.get('elapsedTime', 0),
                'queue_time': task.get('queueTime', 0),
                'wait_time': task.get('waitTime', 0),
                'execution_time': task.get('executionTime', 0),
                'pull_time': task.get('pullTime', 0),
                'cold_start_time': task.get('coldStartTime', 0),
                'initialization_time': task.get('initializationTime', 0),
                'compute_time': task.get('computeTime', 0),
                'communications_time': task.get('communicationsTime', 0),
                'network_latency': task.get('networkLatency', 0),
                'cold_started': task.get('coldStarted', False),
                'cache_hit': task.get('cacheHit', False),
                'task_type': task.get('taskType', {}).get('name', 'unknown'),
            })
    
    return tasks

def calculate_statistics(tasks: List[Dict]) -> Dict:
    """Calculate comprehensive timing statistics."""
    metrics = {
        'elapsed_time': np.array([t['elapsed_time'] for t in tasks]),
        'queue_time': np.array([t['queue_time'] for t in tasks]),
        'wait_time': np.array([t['wait_time'] for t in tasks]),
        'execution_time': np.array([t['execution_time'] for t in tasks]),
        'pull_time': np.array([t['pull_time'] for t in tasks]),
        'cold_start_time': np.array([t['cold_start_time'] for t in tasks]),
        'initialization_time': np.array([t['initialization_time'] for t in tasks]),
        'compute_time': np.array([t['compute_time'] for t in tasks]),
        'communications_time': np.array([t['communications_time'] for t in tasks]),
        'network_latency': np.array([t['network_latency'] for t in tasks]),
    }
    
    # Calculate proportions
    elapsed = metrics['elapsed_time']
    non_zero_elapsed = elapsed[elapsed > 0]
    
    proportions = {}
    for key in ['queue_time', 'wait_time', 'execution_time', 'pull_time', 
                'cold_start_time', 'initialization_time', 'compute_time', 
                'communications_time', 'network_latency']:
        metric = metrics[key]
        if len(non_zero_elapsed) > 0:
            ratio = metric / elapsed
            proportions[key] = np.mean(ratio[elapsed > 0]) * 100
        else:
            proportions[key] = 0.0
    
    # Count cold starts and cache hits
    cold_start_count = sum(1 for t in tasks if t['cold_started'])
    cache_hit_count = sum(1 for t in tasks if t['cache_hit'])
    
    return {
        'num_tasks': len(tasks),
        'cold_start_count': cold_start_count,
        'cache_hit_count': cache_hit_count,
        'metrics': metrics,
        'proportions': proportions,
    }

def print_statistics(stats: Dict):
    """Print comprehensive timing statistics."""
    print("\n" + "="*80)
    print("Task Timing Breakdown Analysis")
    print("="*80)
    print(f"\nTotal tasks analyzed: {stats['num_tasks']}")
    print(f"Tasks with cold start: {stats['cold_start_count']} ({stats['cold_start_count']/stats['num_tasks']*100:.2f}%)")
    print(f"Tasks with cache hit: {stats['cache_hit_count']} ({stats['cache_hit_count']/stats['num_tasks']*100:.2f}%)")
    
    metrics = stats['metrics']
    proportions = stats['proportions']
    
    print("\n--- Summary Statistics ---")
    
    metric_labels = {
        'elapsed_time': 'RTT (elapsedTime)',
        'queue_time': 'Queue Time',
        'wait_time': 'Wait Time',
        'execution_time': 'Execution Time',
        'pull_time': 'Image Pull Time',
        'cold_start_time': 'Cold Start Time',
        'initialization_time': 'Initialization Time',
        'compute_time': 'Compute Time',
        'communications_time': 'Communications Time',
        'network_latency': 'Network Latency',
    }
    
    for key, label in metric_labels.items():
        if key in metrics:
            arr = metrics[key]
            print(f"\n{label}:")
            print(f"  Mean: {np.mean(arr):.4f}s, Std: {np.std(arr):.4f}s")
            print(f"  Min: {np.min(arr):.4f}s, Max: {np.max(arr):.4f}s")
            if key in proportions:
                print(f"  As % of RTT: {proportions[key]:.2f}%")
    
    # Time breakdown
    print("\n--- Time Breakdown (as % of RTT) ---")
    elapsed = metrics['elapsed_time']
    non_zero_elapsed = elapsed[elapsed > 0]
    
    if len(non_zero_elapsed) > 0:
        for key in ['queue_time', 'wait_time', 'execution_time', 'pull_time',
                    'cold_start_time', 'initialization_time', 'compute_time',
                    'communications_time', 'network_latency']:
            if key in proportions:
                print(f"  {metric_labels[key]}: {proportions[key]:.2f}%")

def create_visualizations(stats: Dict, output_dir: Path):
    """Create visualizations for timing metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = stats['metrics']
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # Plot 1: Queue Time vs RTT (scatter)
    ax1 = axes[0, 0]
    elapsed = metrics['elapsed_time']
    queue = metrics['queue_time']
    ax1.scatter(queue, elapsed, alpha=0.5, s=20)
    ax1.set_xlabel('Queue Time (s)', fontsize=11)
    ax1.set_ylabel('RTT / Elapsed Time (s)', fontsize=11)
    ax1.set_title('RTT vs Queue Time', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Execution Time vs RTT (scatter)
    ax2 = axes[0, 1]
    execution = metrics['execution_time']
    ax2.scatter(execution, elapsed, alpha=0.5, s=20, color='green')
    ax2.set_xlabel('Execution Time (s)', fontsize=11)
    ax2.set_ylabel('RTT / Elapsed Time (s)', fontsize=11)
    ax2.set_title('RTT vs Execution Time', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cold Start Time distribution
    ax3 = axes[1, 0]
    cold_start = metrics['cold_start_time']
    non_zero_cold_start = cold_start[cold_start > 0]
    if len(non_zero_cold_start) > 0:
        ax3.hist(non_zero_cold_start, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax3.set_xlabel('Cold Start Time (s)', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title(f'Cold Start Time Distribution\n({len(non_zero_cold_start)} tasks with cold start)', fontsize=12)
    else:
        ax3.text(0.5, 0.5, 'No cold starts', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Cold Start Time Distribution', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Image Pull Time distribution
    ax4 = axes[1, 1]
    pull_time = metrics['pull_time']
    non_zero_pull = pull_time[pull_time > 0]
    if len(non_zero_pull) > 0:
        ax4.hist(non_zero_pull, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax4.set_xlabel('Image Pull Time (s)', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.set_title(f'Image Pull Time Distribution\n({len(non_zero_pull)} tasks with image pull)', fontsize=12)
    else:
        ax4.text(0.5, 0.5, 'No image pulls', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Image Pull Time Distribution', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Queue Time distribution
    ax5 = axes[2, 0]
    ax5.hist(queue, bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax5.set_xlabel('Queue Time (s)', fontsize=11)
    ax5.set_ylabel('Frequency', fontsize=11)
    ax5.set_title('Queue Time Distribution', fontsize=12)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: RTT distribution
    ax6 = axes[2, 1]
    ax6.hist(elapsed, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax6.set_xlabel('RTT / Elapsed Time (s)', fontsize=11)
    ax6.set_ylabel('Frequency', fontsize=11)
    ax6.set_title('RTT Distribution', fontsize=12)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_dir / "rtt_queue_correlation.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Visualization saved to {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts_analysis/analyze_rtt_queue_correlation.py <simulation_result.json>")
        sys.exit(1)
    
    result_file = Path(sys.argv[1])
    if not result_file.exists():
        print(f"Error: File not found: {result_file}")
        sys.exit(1)
    
    # Load data
    data = load_simulation_result(result_file)
    
    # Extract task metrics
    tasks = extract_task_metrics(data)
    if not tasks:
        print("Error: No task results found in the simulation result file")
        sys.exit(1)
    
    print(f"  Found {len(tasks)} tasks")
    
    # Calculate statistics
    stats = calculate_statistics(tasks)
    
    # Print statistics
    print_statistics(stats)
    
    # Create visualizations
    output_dir = result_file.parent / "analysis"
    create_visualizations(stats, output_dir)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()
