#!/usr/bin/env python3
"""
Analyze workload file to understand task arrival patterns and estimate queue lengths.

This script analyzes:
1. Arrival rate distribution (events per second over time)
2. Task type distribution (dnn1 vs dnn2 ratio)
3. Client node distribution
4. Inter-arrival time statistics
5. Estimated queue buildup (based on arrival rates and processing assumptions)
6. Comparison to GNN dataset workload characteristics
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
import statistics

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def load_workload(workload_path: Path) -> Dict[str, Any]:
    """Load workload JSON file."""
    with open(workload_path, 'r') as f:
        return json.load(f)


def analyze_arrival_rate(events: List[Dict], duration: float, window_size: float = 1.0) -> Dict[str, Any]:
    """
    Analyze arrival rate distribution over time.
    
    Returns:
        Dictionary with arrival rate statistics and time-series data
    """
    # Group events by time windows
    time_windows = defaultdict(int)
    for event in events:
        timestamp = event['timestamp']
        window = int(timestamp / window_size)
        time_windows[window] += 1
    
    # Convert to sorted list
    windows = sorted(time_windows.keys())
    rates = [time_windows[w] / window_size for w in windows]
    times = [w * window_size for w in windows]
    
    # Calculate statistics
    mean_rate = statistics.mean(rates) if rates else 0
    median_rate = statistics.median(rates) if rates else 0
    std_rate = statistics.stdev(rates) if len(rates) > 1 else 0
    min_rate = min(rates) if rates else 0
    max_rate = max(rates) if rates else 0
    
    return {
        'mean': mean_rate,
        'median': median_rate,
        'std': std_rate,
        'min': min_rate,
        'max': max_rate,
        'times': times,
        'rates': rates,
        'total_events': len(events),
        'duration': duration,
        'expected_rps': len(events) / duration if duration > 0 else 0
    }


def analyze_task_types(events: List[Dict]) -> Dict[str, Any]:
    """Analyze distribution of task types (dnn1 vs dnn2)."""
    task_type_counts = Counter()
    task_type_timestamps = defaultdict(list)
    
    for event in events:
        app_name = event['application']['name']
        # Extract task type from application name (e.g., "nofs-dnn1" -> "dnn1")
        if 'dnn1' in app_name:
            task_type = 'dnn1'
        elif 'dnn2' in app_name:
            task_type = 'dnn2'
        else:
            task_type = 'unknown'
        
        task_type_counts[task_type] += 1
        task_type_timestamps[task_type].append(event['timestamp'])
    
    total = sum(task_type_counts.values())
    percentages = {k: (v / total * 100) if total > 0 else 0 for k, v in task_type_counts.items()}
    
    return {
        'counts': dict(task_type_counts),
        'percentages': percentages,
        'total': total,
        'timestamps': {k: v for k, v in task_type_timestamps.items()}
    }


def analyze_client_distribution(events: List[Dict]) -> Dict[str, Any]:
    """Analyze distribution of tasks across client nodes."""
    client_counts = Counter()
    client_timestamps = defaultdict(list)
    
    for event in events:
        node_name = event['node_name']
        client_counts[node_name] += 1
        client_timestamps[node_name].append(event['timestamp'])
    
    total = sum(client_counts.values())
    percentages = {k: (v / total * 100) if total > 0 else 0 for k, v in client_counts.items()}
    
    return {
        'counts': dict(client_counts),
        'percentages': percentages,
        'total': total,
        'num_clients': len(client_counts),
        'timestamps': {k: v for k, v in client_timestamps.items()}
    }


def analyze_inter_arrival_times(events: List[Dict]) -> Dict[str, Any]:
    """Analyze inter-arrival times between consecutive events."""
    if len(events) < 2:
        return {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0, 'times': []}
    
    timestamps = sorted([e['timestamp'] for e in events])
    inter_arrivals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    
    mean_iat = statistics.mean(inter_arrivals) if inter_arrivals else 0
    median_iat = statistics.median(inter_arrivals) if inter_arrivals else 0
    std_iat = statistics.stdev(inter_arrivals) if len(inter_arrivals) > 1 else 0
    min_iat = min(inter_arrivals) if inter_arrivals else 0
    max_iat = max(inter_arrivals) if inter_arrivals else 0
    
    return {
        'mean': mean_iat,
        'median': median_iat,
        'std': std_iat,
        'min': min_iat,
        'max': max_iat,
        'times': inter_arrivals
    }


def estimate_queue_buildup(
    events: List[Dict],
    duration: float,
    processing_time_per_task: float = 0.001,  # Assume 1ms processing time per task
    num_replicas: int = 1,
    window_size: float = 1.0
) -> Dict[str, Any]:
    """
    Estimate queue length over time based on arrival rate and processing capacity.
    
    This is a simplified model that assumes:
    - Tasks arrive at the rate observed in the workload
    - Each replica can process tasks at a constant rate
    - Queue builds up when arrival rate > processing capacity
    """
    # Group events by time windows
    time_windows = defaultdict(int)
    for event in events:
        timestamp = event['timestamp']
        window = int(timestamp / window_size)
        time_windows[window] += 1
    
    windows = sorted(time_windows.keys())
    
    # Calculate processing capacity per window
    processing_capacity = (num_replicas / processing_time_per_task) * window_size
    
    # Simulate queue buildup
    queue_lengths = []
    current_queue = 0
    
    for w in windows:
        arrivals = time_windows[w]
        processed = min(current_queue + arrivals, processing_capacity)
        current_queue = max(0, current_queue + arrivals - processed)
        queue_lengths.append(current_queue)
    
    # Calculate statistics
    mean_queue = statistics.mean(queue_lengths) if queue_lengths else 0
    max_queue = max(queue_lengths) if queue_lengths else 0
    times = [w * window_size for w in windows]
    
    return {
        'mean': mean_queue,
        'max': max_queue,
        'times': times,
        'queue_lengths': queue_lengths,
        'processing_capacity': processing_capacity,
        'assumptions': {
            'processing_time_per_task': processing_time_per_task,
            'num_replicas': num_replicas,
            'window_size': window_size
        }
    }


def compare_to_gnn_datasets(workload_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare current workload to GNN dataset characteristics.
    
    GNN datasets typically have:
    - 5 tasks per workload
    - Various task type ratios (50-50, 60-40, etc.)
    - 10 client nodes
    """
    num_events = len(workload_data['events'])
    task_types = analyze_task_types(workload_data['events'])
    
    gnn_typical = {
        'num_tasks': 5,
        'task_type_ratios': ['50-50', '60-40', '40-60'],
        'num_clients': 10
    }
    
    current = {
        'num_tasks': num_events,
        'dnn1_percentage': task_types['percentages'].get('dnn1', 0),
        'dnn2_percentage': task_types['percentages'].get('dnn2', 0),
        'num_clients': len(analyze_client_distribution(workload_data['events'])['counts'])
    }
    
    return {
        'gnn_typical': gnn_typical,
        'current': current,
        'comparison': {
            'task_count_ratio': current['num_tasks'] / gnn_typical['num_tasks'] if gnn_typical['num_tasks'] > 0 else 0,
            'is_similar_task_ratio': abs(current['dnn1_percentage'] - 50) < 10,  # Within 10% of 50-50
            'same_num_clients': current['num_clients'] == gnn_typical['num_clients']
        }
    }


def print_summary(workload_path: Path, analysis: Dict[str, Any]):
    """Print a text summary of the analysis."""
    print("=" * 80)
    print(f"Workload Analysis: {workload_path.name}")
    print("=" * 80)
    
    print("\n1. WORKLOAD OVERVIEW")
    print("-" * 80)
    print(f"  Total events: {analysis['arrival_rate']['total_events']:,}")
    print(f"  Duration: {analysis['arrival_rate']['duration']:.2f} seconds")
    print(f"  Expected RPS: {analysis['arrival_rate']['expected_rps']:.2f}")
    
    print("\n2. ARRIVAL RATE STATISTICS")
    print("-" * 80)
    arr = analysis['arrival_rate']
    print(f"  Mean RPS: {arr['mean']:.2f}")
    print(f"  Median RPS: {arr['median']:.2f}")
    print(f"  Std Dev RPS: {arr['std']:.2f}")
    print(f"  Min RPS: {arr['min']:.2f}")
    print(f"  Max RPS: {arr['max']:.2f}")
    
    print("\n3. TASK TYPE DISTRIBUTION")
    print("-" * 80)
    tt = analysis['task_types']
    for task_type, count in tt['counts'].items():
        pct = tt['percentages'][task_type]
        print(f"  {task_type}: {count:,} ({pct:.1f}%)")
    
    print("\n4. CLIENT NODE DISTRIBUTION")
    print("-" * 80)
    cd = analysis['client_distribution']
    print(f"  Number of client nodes: {cd['num_clients']}")
    print(f"  Top 5 clients by task count:")
    sorted_clients = sorted(cd['counts'].items(), key=lambda x: x[1], reverse=True)
    for client, count in sorted_clients[:5]:
        pct = cd['percentages'][client]
        print(f"    {client}: {count:,} ({pct:.1f}%)")
    
    print("\n5. INTER-ARRIVAL TIME STATISTICS")
    print("-" * 80)
    iat = analysis['inter_arrival']
    print(f"  Mean IAT: {iat['mean']:.6f} seconds ({1/iat['mean']:.2f} RPS if constant)")
    print(f"  Median IAT: {iat['median']:.6f} seconds")
    print(f"  Std Dev IAT: {iat['std']:.6f} seconds")
    print(f"  Min IAT: {iat['min']:.6f} seconds")
    print(f"  Max IAT: {iat['max']:.6f} seconds")
    
    print("\n6. ESTIMATED QUEUE BUILDUP")
    print("-" * 80)
    eq = analysis['estimated_queue']
    print(f"  Assumptions:")
    print(f"    Processing time per task: {eq['assumptions']['processing_time_per_task']*1000:.2f} ms")
    print(f"    Number of replicas: {eq['assumptions']['num_replicas']}")
    print(f"    Processing capacity: {eq['processing_capacity']:.2f} tasks/second")
    print(f"  Mean queue length: {eq['mean']:.2f} tasks")
    print(f"  Max queue length: {eq['max']:.2f} tasks")
    
    print("\n7. COMPARISON TO GNN DATASETS")
    print("-" * 80)
    comp = analysis['comparison']
    print(f"  GNN typical: {comp['gnn_typical']['num_tasks']} tasks")
    print(f"  Current: {comp['current']['num_tasks']:,} tasks")
    print(f"  Ratio: {comp['comparison']['task_count_ratio']:.1f}x")
    print(f"  Task ratio similarity: {comp['comparison']['is_similar_task_ratio']}")
    print(f"  Same number of clients: {comp['comparison']['same_num_clients']}")
    
    print("\n8. POTENTIAL ISSUES")
    print("-" * 80)
    issues = []
    
    # Check if arrival rate is too high
    if arr['mean'] > 100:
        issues.append(f"High arrival rate ({arr['mean']:.2f} RPS) may overwhelm system")
    
    # Check if queue buildup is significant
    if eq['max'] > 100:
        issues.append(f"Significant queue buildup (max {eq['max']:.2f} tasks)")
    
    # Check if workload is much larger than GNN datasets
    if comp['comparison']['task_count_ratio'] > 100:
        issues.append(f"Workload is {comp['comparison']['task_count_ratio']:.1f}x larger than GNN training data")
    
    # Check arrival rate variance
    if arr['std'] > arr['mean'] * 0.5:
        issues.append(f"High arrival rate variance (std={arr['std']:.2f}, mean={arr['mean']:.2f})")
    
    if issues:
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("  ✓ No obvious issues detected")
    
    print("\n" + "=" * 80)


def plot_analysis(workload_path: Path, analysis: Dict[str, Any], output_path: Path):
    """Generate plots of the analysis."""
    with PdfPages(output_path) as pdf:
        # Plot 1: Arrival rate over time
        fig, ax = plt.subplots(figsize=(12, 6))
        arr = analysis['arrival_rate']
        ax.plot(arr['times'], arr['rates'], alpha=0.7, linewidth=0.5)
        ax.axhline(y=arr['mean'], color='r', linestyle='--', label=f"Mean: {arr['mean']:.2f} RPS")
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Arrival Rate (events/second)')
        ax.set_title('Arrival Rate Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Plot 2: Task type distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        tt = analysis['task_types']
        ax1.bar(tt['counts'].keys(), tt['counts'].values())
        ax1.set_xlabel('Task Type')
        ax1.set_ylabel('Count')
        ax1.set_title('Task Type Distribution (Count)')
        ax2.bar(tt['percentages'].keys(), tt['percentages'].values())
        ax2.set_xlabel('Task Type')
        ax2.set_ylabel('Percentage')
        ax2.set_title('Task Type Distribution (Percentage)')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Plot 3: Client node distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        cd = analysis['client_distribution']
        sorted_clients = sorted(cd['counts'].items(), key=lambda x: x[1], reverse=True)
        clients = [c[0] for c in sorted_clients]
        counts = [c[1] for c in sorted_clients]
        ax.bar(range(len(clients)), counts)
        ax.set_xticks(range(len(clients)))
        ax.set_xticklabels(clients, rotation=45, ha='right')
        ax.set_xlabel('Client Node')
        ax.set_ylabel('Task Count')
        ax.set_title('Client Node Distribution')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Plot 4: Inter-arrival time distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        iat = analysis['inter_arrival']
        if iat['times']:
            ax.hist(iat['times'], bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(x=iat['mean'], color='r', linestyle='--', label=f"Mean: {iat['mean']:.6f}s")
            ax.set_xlabel('Inter-Arrival Time (seconds)')
            ax.set_ylabel('Frequency')
            ax.set_title('Inter-Arrival Time Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Plot 5: Estimated queue buildup
        fig, ax = plt.subplots(figsize=(12, 6))
        eq = analysis['estimated_queue']
        ax.plot(eq['times'], eq['queue_lengths'], alpha=0.7, linewidth=0.5)
        ax.axhline(y=eq['mean'], color='r', linestyle='--', label=f"Mean: {eq['mean']:.2f} tasks")
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Estimated Queue Length (tasks)')
        ax.set_title(f"Estimated Queue Buildup (assuming {eq['assumptions']['num_replicas']} replica(s), "
                     f"{eq['assumptions']['processing_time_per_task']*1000:.2f}ms processing time)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_workload.py <workload.json> [output.pdf]")
        sys.exit(1)
    
    workload_path = Path(sys.argv[1])
    if not workload_path.exists():
        print(f"Error: Workload file not found: {workload_path}")
        sys.exit(1)
    
    # Load workload
    print(f"Loading workload from {workload_path}...")
    workload_data = load_workload(workload_path)
    
    # Perform analysis
    print("Analyzing workload...")
    events = workload_data['events']
    duration = workload_data.get('duration', 0)
    
    analysis = {
        'arrival_rate': analyze_arrival_rate(events, duration),
        'task_types': analyze_task_types(events),
        'client_distribution': analyze_client_distribution(events),
        'inter_arrival': analyze_inter_arrival_times(events),
        'estimated_queue': estimate_queue_buildup(events, duration),
        'comparison': compare_to_gnn_datasets(workload_data)
    }
    
    # Print summary
    print_summary(workload_path, analysis)
    
    # Generate plots if output path provided
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
        print(f"\nGenerating plots to {output_path}...")
        plot_analysis(workload_path, analysis, output_path)
        print(f"Plots saved to {output_path}")
    
    # Save JSON summary
    json_output = workload_path.parent / f"{workload_path.stem}_analysis.json"
    with open(json_output, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_analysis = convert_to_serializable(analysis)
        json.dump(serializable_analysis, f, indent=2)
    
    print(f"\nJSON summary saved to {json_output}")


if __name__ == "__main__":
    main()

