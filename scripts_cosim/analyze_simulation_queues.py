#!/usr/bin/env python3
"""
Analyze queue distributions from real simulation results.

This script helps understand the gap between training and simulation conditions
by extracting queue statistics from simulation results.

Usage:
  python scripts_cosim/analyze_simulation_queues.py simulation_data/results/simulation_result_gnn.json

The output can inform co-simulation parameter adjustments.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any
import statistics


def analyze_task_results(task_results: List[Dict]) -> Dict[str, Any]:
    """Analyze task-level statistics."""
    # Filter to workload tasks (taskId >= 0)
    workload_tasks = [tr for tr in task_results if tr.get('taskId', -1) >= 0]
    
    if not workload_tasks:
        return {"error": "No workload tasks found"}
    
    # RTT distribution
    rtts = [tr.get('elapsedTime', 0) for tr in workload_tasks]
    
    # Platform usage distribution
    platform_usage = Counter()
    for tr in workload_tasks:
        exec_node = tr.get('executionNode', '?')
        exec_plat = tr.get('executionPlatform', '?')
        platform_usage[f"{exec_node}:{exec_plat}"] += 1
    
    # Node distribution
    node_usage = Counter(tr.get('executionNode', '?') for tr in workload_tasks)
    
    return {
        "total_tasks": len(workload_tasks),
        "rtt": {
            "min": min(rtts),
            "max": max(rtts),
            "mean": statistics.mean(rtts),
            "median": statistics.median(rtts),
            "sum": sum(rtts),
            "p95": sorted(rtts)[int(len(rtts) * 0.95)] if len(rtts) > 20 else max(rtts),
            "p99": sorted(rtts)[int(len(rtts) * 0.99)] if len(rtts) > 100 else max(rtts),
        },
        "platform_usage": {
            "unique_platforms": len(platform_usage),
            "top_10": platform_usage.most_common(10),
            "max_tasks_per_platform": max(platform_usage.values()) if platform_usage else 0,
            "min_tasks_per_platform": min(platform_usage.values()) if platform_usage else 0,
            "mean_tasks_per_platform": statistics.mean(platform_usage.values()) if platform_usage else 0,
        },
        "node_distribution": dict(node_usage.most_common(10)),
    }


def analyze_node_results(node_results: List[Dict]) -> Dict[str, Any]:
    """Analyze node and platform-level queue statistics."""
    queue_stats = []
    total_tasks_processed = 0
    
    for node_result in node_results:
        node_name = node_result.get('nodeName', f"node_{node_result.get('nodeId', -1)}")
        
        for plat_result in node_result.get('platformResults', []):
            plat_id = plat_result.get('platformId', -1)
            tasks_processed = plat_result.get('totalTasksProcessed', 0)
            queue_data = plat_result.get('queueStatistics', {})
            
            if isinstance(queue_data, dict):
                max_queue = queue_data.get('maxQueueLength', 0)
                avg_queue = queue_data.get('avgQueueLength', 0)
            else:
                max_queue = 0
                avg_queue = 0
            
            queue_stats.append({
                "platform": f"{node_name}:{plat_id}",
                "max_queue": max_queue,
                "avg_queue": avg_queue,
                "tasks_processed": tasks_processed,
            })
            total_tasks_processed += tasks_processed
    
    # Filter to platforms with activity
    active_platforms = [q for q in queue_stats if q['tasks_processed'] > 0]
    
    if not active_platforms:
        return {"error": "No active platforms found in node results"}
    
    max_queues = [q['max_queue'] for q in active_platforms]
    avg_queues = [q['avg_queue'] for q in active_platforms]
    
    return {
        "total_platforms": len(queue_stats),
        "active_platforms": len(active_platforms),
        "total_tasks_processed": total_tasks_processed,
        "queue_lengths": {
            "max_observed": max(max_queues) if max_queues else 0,
            "mean_of_max": statistics.mean(max_queues) if max_queues else 0,
            "mean_of_avg": statistics.mean(avg_queues) if avg_queues else 0,
        },
        "top_10_by_max_queue": sorted(active_platforms, key=lambda x: x['max_queue'], reverse=True)[:10],
    }


def compare_with_training(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Compare simulation stats with typical training conditions."""
    
    # Training conditions (from your training data analysis)
    training_conditions = {
        "batch_size": 5,
        "queue_norm_factor": 10.0,
        "typical_max_queue": 31,
        "typical_mean_queue": 4.5,
        "poisson_lambda": 6,
    }
    
    # Extract simulation conditions
    sim_max_queue = analysis.get("node_analysis", {}).get("queue_lengths", {}).get("max_observed", 0)
    sim_mean_queue = analysis.get("node_analysis", {}).get("queue_lengths", {}).get("mean_of_avg", 0)
    sim_max_tasks_per_platform = analysis.get("task_analysis", {}).get("platform_usage", {}).get("max_tasks_per_platform", 0)
    
    return {
        "training_conditions": training_conditions,
        "simulation_conditions": {
            "max_queue_observed": sim_max_queue,
            "mean_queue": sim_mean_queue,
            "max_tasks_per_platform": sim_max_tasks_per_platform,
        },
        "recommendations": {
            "queue_norm_factor": max(50.0, sim_max_queue / 2),  # Adjust normalization
            "poisson_lambda": max(6, int(sim_mean_queue * 2)),  # Increase training queue load
            "max_queue_for_training": max(100, int(sim_max_queue * 1.2)),  # Train with higher caps
        }
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_simulation_queues.py <simulation_result.json>")
        print("\nExample files:")
        print("  simulation_data/results/simulation_result_gnn.json")
        print("  simulation_data/results/simulation_result_knative_network.json")
        sys.exit(1)
    
    result_file = Path(sys.argv[1])
    if not result_file.exists():
        print(f"Error: File not found: {result_file}")
        sys.exit(1)
    
    print(f"Analyzing: {result_file}")
    print("=" * 60)
    
    with open(result_file, 'r') as f:
        result = json.load(f)
    
    stats = result.get('stats', {})
    
    # Task analysis
    task_results = stats.get('taskResults', [])
    task_analysis = analyze_task_results(task_results)
    
    print("\n=== TASK ANALYSIS ===")
    print(f"Total workload tasks: {task_analysis.get('total_tasks', 0)}")
    
    rtt = task_analysis.get('rtt', {})
    print(f"\nRTT Distribution:")
    print(f"  Min:    {rtt.get('min', 0):.4f}s")
    print(f"  Max:    {rtt.get('max', 0):.4f}s")
    print(f"  Mean:   {rtt.get('mean', 0):.4f}s")
    print(f"  Median: {rtt.get('median', 0):.4f}s")
    print(f"  P95:    {rtt.get('p95', 0):.4f}s")
    print(f"  P99:    {rtt.get('p99', 0):.4f}s")
    print(f"  Total:  {rtt.get('sum', 0):.2f}s")
    
    plat_usage = task_analysis.get('platform_usage', {})
    print(f"\nPlatform Usage:")
    print(f"  Unique platforms used: {plat_usage.get('unique_platforms', 0)}")
    print(f"  Max tasks on single platform: {plat_usage.get('max_tasks_per_platform', 0)}")
    print(f"  Mean tasks per platform: {plat_usage.get('mean_tasks_per_platform', 0):.1f}")
    print(f"\n  Top 10 busiest platforms:")
    for plat, count in plat_usage.get('top_10', []):
        print(f"    {plat}: {count} tasks")
    
    # Node analysis  
    node_results = stats.get('nodeResults', [])
    node_analysis = analyze_node_results(node_results)
    
    print("\n=== NODE/QUEUE ANALYSIS ===")
    print(f"Total platforms: {node_analysis.get('total_platforms', 0)}")
    print(f"Active platforms: {node_analysis.get('active_platforms', 0)}")
    
    queue_lengths = node_analysis.get('queue_lengths', {})
    print(f"\nQueue Length Statistics:")
    print(f"  Max observed: {queue_lengths.get('max_observed', 0)}")
    print(f"  Mean of max queues: {queue_lengths.get('mean_of_max', 0):.2f}")
    print(f"  Mean of avg queues: {queue_lengths.get('mean_of_avg', 0):.2f}")
    
    # Comparison
    analysis = {
        "task_analysis": task_analysis,
        "node_analysis": node_analysis,
    }
    comparison = compare_with_training(analysis)
    
    print("\n=== TRAINING vs SIMULATION COMPARISON ===")
    training = comparison.get('training_conditions', {})
    simulation = comparison.get('simulation_conditions', {})
    recommendations = comparison.get('recommendations', {})
    
    print(f"\n{'Metric':<35} {'Training':<15} {'Simulation':<15}")
    print("-" * 65)
    print(f"{'Max queue length':<35} {training['typical_max_queue']:<15} {simulation['max_queue_observed']:<15}")
    print(f"{'Mean queue length':<35} {training['typical_mean_queue']:<15.2f} {simulation['mean_queue']:<15.2f}")
    print(f"{'Queue norm factor':<35} {training['queue_norm_factor']:<15.1f} {'N/A':<15}")
    
    print("\n=== RECOMMENDATIONS FOR TRAINING ===")
    print(f"1. Set queue_norm_factor = {recommendations['queue_norm_factor']:.1f} (currently {training['queue_norm_factor']})")
    print(f"2. Set Poisson lambda = {recommendations['poisson_lambda']} (currently {training['poisson_lambda']})")
    print(f"3. Set max_queue = {recommendations['max_queue_for_training']} (currently {training['typical_max_queue']})")
    
    print("\n=== SOFT BLENDING RECOMMENDATION ===")
    if simulation['max_queue_observed'] > 50:
        print(f"High queue load detected. Recommend soft blending with:")
        print(f"  gnn_weight = 0.4 (trust queues more)")
        print(f"  queue_norm = {max(100, simulation['max_queue_observed'])}")
    else:
        print(f"Moderate queue load. Default soft blending should work:")
        print(f"  gnn_weight = 0.6")
        print(f"  queue_norm = 50.0")


if __name__ == "__main__":
    main()
