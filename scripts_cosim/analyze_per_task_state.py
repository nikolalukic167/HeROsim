#!/usr/bin/env python3
"""
Analyze Per-Task System State Evolution

NEW: Analyzes the per-task system state captures from GNN scheduler.
Each task has systemStateResult, fullQueueSnapshot, and temporalStateAtScheduling
captured at scheduling time, enabling detailed temporal analysis.

This script provides:
1. Temporal evolution of replicas, queues, and temporal state
2. Per-task state comparison at different simulation stages
3. State transition analysis
4. Correlation between system state and task performance
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import numpy as np
import pandas as pd

BASE_DIR = Path("/root/projects/my-herosim")
SIM_RESULT_FILE = BASE_DIR / "simulation_data/results/simulation_result_gnn.json"


def extract_per_task_states(sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Extract per-task system state from simulation results.
    
    Returns list of task states with full system state information.
    """
    if not SIM_RESULT_FILE.exists():
        print(f"Error: Simulation result file not found: {SIM_RESULT_FILE}")
        return []
    
    print(f"Loading per-task states from {SIM_RESULT_FILE}...")
    with open(SIM_RESULT_FILE, 'r') as f:
        data = json.load(f)
    
    task_results = data.get('stats', {}).get('taskResults', [])
    
    if sample_size:
        # Sample evenly across the simulation
        sample_indices = np.linspace(0, len(task_results) - 1, 
                                   min(sample_size, len(task_results)), dtype=int)
        task_results = [task_results[i] for i in sample_indices]
    
    per_task_states = []
    
    for tr in task_results:
        system_state = tr.get('systemStateResult')
        if not system_state:
            continue  # Skip tasks without system state
        
        # Extract all state information
        task_state = {
            "task_id": tr.get('taskId'),
            "task_type": tr.get('taskType', {}).get('name'),
            "scheduled_time": tr.get('scheduledTime'),
            "elapsed_time": tr.get('elapsedTime'),
            "queue_time": tr.get('queueTime'),
            "execution_node": tr.get('executionNode'),
            "execution_platform": tr.get('executionPlatform'),
            
            # System state from systemStateResult
            "timestamp": system_state.get('timestamp', tr.get('scheduledTime')),
            "replicas": system_state.get('replicas', {}),
            "available_resources": system_state.get('available_resources', {}),
            "scheduler_state": system_state.get('scheduler_state', {}),
            "queue_occupancy": system_state.get('queue_occupancy', {}),
            
            # Queue snapshots
            "queue_snapshot": tr.get('queueSnapshotAtScheduling', {}),
            "full_queue_snapshot": tr.get('fullQueueSnapshot', {}),
            
            # Temporal state
            "temporal_state": tr.get('temporalStateAtScheduling', {})
        }
        
        # Calculate derived metrics
        task_state["total_replicas"] = sum(len(v) for v in task_state["replicas"].values())
        task_state["dnn1_replicas"] = len(task_state["replicas"].get("dnn1", []))
        task_state["dnn2_replicas"] = len(task_state["replicas"].get("dnn2", []))
        
        if task_state["full_queue_snapshot"]:
            queue_values = [int(v) for v in task_state["full_queue_snapshot"].values()]
            task_state["mean_queue"] = float(np.mean(queue_values))
            task_state["max_queue"] = int(np.max(queue_values))
            task_state["total_queued"] = sum(queue_values)
            task_state["non_zero_queues"] = sum(1 for v in queue_values if v > 0)
        else:
            task_state["mean_queue"] = 0.0
            task_state["max_queue"] = 0
            task_state["total_queued"] = 0
            task_state["non_zero_queues"] = 0
        
        if task_state["temporal_state"]:
            comm_remaining = [
                state_dict.get('comm_remaining', 0.0)
                for state_dict in task_state["temporal_state"].values()
                if isinstance(state_dict, dict)
            ]
            task_state["mean_comm_remaining"] = float(np.mean(comm_remaining)) if comm_remaining else 0.0
        else:
            task_state["mean_comm_remaining"] = 0.0
        
        per_task_states.append(task_state)
    
    print(f"Extracted {len(per_task_states)} per-task states")
    return per_task_states


def analyze_temporal_evolution(per_task_states: List[Dict]) -> Dict[str, Any]:
    """Analyze how system state evolves over time."""
    if not per_task_states:
        return {}
    
    # Sort by timestamp
    sorted_states = sorted(per_task_states, key=lambda x: x["timestamp"])
    
    analysis = {
        "time_range": {
            "start": sorted_states[0]["timestamp"],
            "end": sorted_states[-1]["timestamp"],
            "duration": sorted_states[-1]["timestamp"] - sorted_states[0]["timestamp"]
        },
        "replica_evolution": {
            "timestamps": [s["timestamp"] for s in sorted_states],
            "total_replicas": [s["total_replicas"] for s in sorted_states],
            "dnn1_replicas": [s["dnn1_replicas"] for s in sorted_states],
            "dnn2_replicas": [s["dnn2_replicas"] for s in sorted_states],
            "initial": sorted_states[0]["total_replicas"],
            "final": sorted_states[-1]["total_replicas"],
            "mean": float(np.mean([s["total_replicas"] for s in sorted_states])),
            "max": int(np.max([s["total_replicas"] for s in sorted_states])),
            "min": int(np.min([s["total_replicas"] for s in sorted_states]))
        },
        "queue_evolution": {
            "timestamps": [s["timestamp"] for s in sorted_states],
            "mean_queues": [s["mean_queue"] for s in sorted_states],
            "max_queues": [s["max_queue"] for s in sorted_states],
            "total_queued": [s["total_queued"] for s in sorted_states],
            "non_zero_counts": [s["non_zero_queues"] for s in sorted_states],
            "initial_mean": sorted_states[0]["mean_queue"],
            "final_mean": sorted_states[-1]["mean_queue"],
            "mean": float(np.mean([s["mean_queue"] for s in sorted_states])),
            "max": int(np.max([s["max_queue"] for s in sorted_states]))
        },
        "temporal_evolution": {
            "timestamps": [s["timestamp"] for s in sorted_states],
            "mean_comm_remaining": [s["mean_comm_remaining"] for s in sorted_states],
            "initial": sorted_states[0]["mean_comm_remaining"],
            "final": sorted_states[-1]["mean_comm_remaining"],
            "mean": float(np.mean([s["mean_comm_remaining"] for s in sorted_states]))
        }
    }
    
    return analysis


def analyze_by_time_ranges(per_task_states: List[Dict]) -> Dict[str, Any]:
    """Analyze system state in different time ranges."""
    if not per_task_states:
        return {}
    
    # Define time ranges
    early = [s for s in per_task_states if s["timestamp"] < 10.0]
    mid = [s for s in per_task_states if 10.0 <= s["timestamp"] < 100.0]
    late = [s for s in per_task_states if s["timestamp"] >= 100.0]
    
    def analyze_range(states, name):
        if not states:
            return {}
        
        return {
            "count": len(states),
            "time_range": (min(s["timestamp"] for s in states), max(s["timestamp"] for s in states)),
            "replicas": {
                "mean_total": float(np.mean([s["total_replicas"] for s in states])),
                "mean_dnn1": float(np.mean([s["dnn1_replicas"] for s in states])),
                "mean_dnn2": float(np.mean([s["dnn2_replicas"] for s in states])),
                "max_total": int(np.max([s["total_replicas"] for s in states]))
            },
            "queues": {
                "mean": float(np.mean([s["mean_queue"] for s in states])),
                "max": int(np.max([s["max_queue"] for s in states])),
                "mean_total_queued": float(np.mean([s["total_queued"] for s in states]))
            },
            "temporal": {
                "mean_comm_remaining": float(np.mean([s["mean_comm_remaining"] for s in states]))
            },
            "performance": {
                "mean_queue_time": float(np.mean([s["queue_time"] for s in states if s["queue_time"]])),
                "mean_elapsed_time": float(np.mean([s["elapsed_time"] for s in states if s["elapsed_time"]]))
            }
        }
    
    return {
        "early": analyze_range(early, "early"),
        "mid": analyze_range(mid, "mid"),
        "late": analyze_range(late, "late")
    }


def analyze_state_performance_correlation(per_task_states: List[Dict]) -> Dict[str, Any]:
    """Analyze correlation between system state and task performance."""
    if not per_task_states:
        return {}
    
    # Extract metrics
    queue_times = [s["queue_time"] for s in per_task_states if s.get("queue_time")]
    elapsed_times = [s["elapsed_time"] for s in per_task_states if s.get("elapsed_time")]
    mean_queues = [s["mean_queue"] for s in per_task_states]
    total_replicas = [s["total_replicas"] for s in per_task_states]
    
    correlation = {}
    
    if len(queue_times) > 1 and len(mean_queues) > 1:
        # Correlation between queue length and queue time
        queue_corr = np.corrcoef(mean_queues[:len(queue_times)], queue_times)[0, 1]
        correlation["queue_length_vs_queue_time"] = float(queue_corr) if not np.isnan(queue_corr) else 0.0
    
    if len(elapsed_times) > 1 and len(total_replicas) > 1:
        # Correlation between replica count and elapsed time
        replica_corr = np.corrcoef(total_replicas[:len(elapsed_times)], elapsed_times)[0, 1]
        correlation["replica_count_vs_elapsed_time"] = float(replica_corr) if not np.isnan(replica_corr) else 0.0
    
    return correlation


def generate_analysis_report(per_task_states: List[Dict]) -> str:
    """Generate comprehensive analysis report."""
    report = []
    report.append("=" * 80)
    report.append("PER-TASK SYSTEM STATE ANALYSIS")
    report.append("Using per-task systemStateResult, fullQueueSnapshot, temporalStateAtScheduling")
    report.append("=" * 80)
    report.append("")
    
    if not per_task_states:
        report.append("No per-task state data found.")
        return "\n".join(report)
    
    report.append(f"Total tasks analyzed: {len(per_task_states)}")
    report.append("")
    
    # Temporal Evolution
    evolution = analyze_temporal_evolution(per_task_states)
    if evolution:
        report.append("1. TEMPORAL EVOLUTION")
        report.append("-" * 80)
        report.append(f"Time range: {evolution['time_range']['start']:.2f}s - {evolution['time_range']['end']:.2f}s")
        report.append(f"Duration: {evolution['time_range']['duration']:.2f}s")
        report.append("")
        
        if evolution.get("replica_evolution"):
            rep = evolution["replica_evolution"]
            report.append("Replica Evolution:")
            report.append(f"  Initial: {rep['initial']} replicas")
            report.append(f"  Final: {rep['final']} replicas")
            report.append(f"  Growth: {rep['final'] - rep['initial']:+d} replicas")
            report.append(f"  Mean: {rep['mean']:.2f} replicas")
            report.append(f"  Range: [{rep['min']}, {rep['max']}] replicas")
            report.append("")
        
        if evolution.get("queue_evolution"):
            queue = evolution["queue_evolution"]
            report.append("Queue Evolution:")
            report.append(f"  Initial mean: {queue['initial_mean']:.2f}")
            report.append(f"  Final mean: {queue['final_mean']:.2f}")
            report.append(f"  Buildup: {queue['final_mean'] - queue['initial_mean']:+.2f}")
            report.append(f"  Overall mean: {queue['mean']:.2f}")
            report.append(f"  Peak queue: {queue['max']}")
            report.append("")
        
        if evolution.get("temporal_evolution"):
            temp = evolution["temporal_evolution"]
            report.append("Temporal State Evolution:")
            report.append(f"  Initial mean comm_remaining: {temp['initial']:.4f}s")
            report.append(f"  Final mean comm_remaining: {temp['final']:.4f}s")
            report.append(f"  Change: {temp['final'] - temp['initial']:+.4f}s")
            report.append(f"  Overall mean: {temp['mean']:.4f}s")
            report.append("")
    
    # Time Range Analysis
    time_ranges = analyze_by_time_ranges(per_task_states)
    if time_ranges:
        report.append("2. SYSTEM STATE BY TIME RANGE")
        report.append("-" * 80)
        
        for range_name, range_data in time_ranges.items():
            if range_data:
                report.append(f"{range_name.replace('_', ' ').title()} Simulation:")
                report.append(f"  Tasks: {range_data['count']}")
                report.append(f"  Time range: {range_data['time_range'][0]:.2f}s - {range_data['time_range'][1]:.2f}s")
                
                if range_data.get("replicas"):
                    rep = range_data["replicas"]
                    report.append(f"  Replicas - Mean: {rep['mean_total']:.2f}, Max: {rep['max_total']}")
                    report.append(f"    dnn1: {rep['mean_dnn1']:.2f}, dnn2: {rep['mean_dnn2']:.2f}")
                
                if range_data.get("queues"):
                    q = range_data["queues"]
                    report.append(f"  Queues - Mean: {q['mean']:.2f}, Max: {q['max']}, Total queued: {q['mean_total_queued']:.2f}")
                
                if range_data.get("performance"):
                    perf = range_data["performance"]
                    report.append(f"  Performance - Queue time: {perf['mean_queue_time']:.2f}s, Elapsed: {perf['mean_elapsed_time']:.2f}s")
                
                report.append("")
    
    # Performance Correlation
    correlation = analyze_state_performance_correlation(per_task_states)
    if correlation:
        report.append("3. STATE-PERFORMANCE CORRELATION")
        report.append("-" * 80)
        for metric, corr_value in correlation.items():
            report.append(f"  {metric}: {corr_value:+.3f}")
            if abs(corr_value) > 0.3:
                direction = "positive" if corr_value > 0 else "negative"
                report.append(f"    ⚠️  Significant {direction} correlation detected")
        report.append("")
    
    # Summary Statistics
    report.append("4. SUMMARY STATISTICS")
    report.append("-" * 80)
    
    total_replicas = [s["total_replicas"] for s in per_task_states]
    mean_queues = [s["mean_queue"] for s in per_task_states]
    queue_times = [s["queue_time"] for s in per_task_states if s.get("queue_time")]
    
    report.append("Replicas:")
    report.append(f"  Mean: {np.mean(total_replicas):.2f}")
    report.append(f"  Range: [{np.min(total_replicas)}, {np.max(total_replicas)}]")
    report.append("")
    
    report.append("Queues:")
    report.append(f"  Mean: {np.mean(mean_queues):.2f}")
    report.append(f"  Range: [{np.min(mean_queues):.2f}, {np.max(mean_queues):.2f}]")
    report.append("")
    
    if queue_times:
        report.append("Queue Times:")
        report.append(f"  Mean: {np.mean(queue_times):.2f}s")
        report.append(f"  Range: [{np.min(queue_times):.2f}s, {np.max(queue_times):.2f}s]")
        report.append("")
    
    return "\n".join(report)


def main():
    """Main entry point."""
    print("=" * 80)
    print("PER-TASK SYSTEM STATE ANALYSIS")
    print("=" * 80)
    print()
    
    print("Extracting per-task states...")
    per_task_states = extract_per_task_states(sample_size=5000)
    
    if not per_task_states:
        print("No per-task state data found. Make sure simulation has run with updated GNN scheduler.")
        return
    
    print(f"Analyzing {len(per_task_states)} tasks...")
    print()
    
    print("Generating analysis report...")
    report = generate_analysis_report(per_task_states)
    print(report)
    
    # Save to file
    output_file = BASE_DIR / "scripts_cosim" / "PER_TASK_STATE_ANALYSIS.txt"
    with open(output_file, 'w') as f:
        f.write(report)
    
    print()
    print(f"✓ Analysis saved to: {output_file}")


if __name__ == "__main__":
    main()
