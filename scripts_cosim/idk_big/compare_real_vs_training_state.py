#!/usr/bin/env python3
"""
Compare Real Simulation System State vs Training/Co-Simulation State

UPDATED: Now uses per-task system state captures from GNN scheduler.
Each task now has systemStateResult, fullQueueSnapshot, and temporalStateAtScheduling
captured at scheduling time, enabling detailed per-timestamp analysis.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

BASE_DIR = Path("/root/projects/my-herosim")
ARTIFACTS_DIR = BASE_DIR / "simulation_data/artifacts/run_non_unique"
SIM_RESULT_FILE = BASE_DIR / "simulation_data/results/simulation_result_gnn.json"


def extract_training_queue_statistics(num_samples: int = 100) -> Dict[str, Any]:
    """Extract queue statistics from training datasets."""
    stats = {
        "queue_lengths": [],
        "temporal_states": {
            "current_task_remaining": [],
            "cold_start_remaining": [],
            "comm_remaining": []
        },
        "replica_counts": defaultdict(int),
        "platform_usage": defaultdict(int),
        "samples_analyzed": 0,
        "per_task_data": []  # NEW: Store per-task data for detailed analysis
    }
    
    # Sample from both 2-task and 3-task datasets
    for dataset_type in ["gnn_datasets_2tasks", "gnn_datasets_3tasks"]:
        dataset_dir = ARTIFACTS_DIR / dataset_type
        if not dataset_dir.exists():
            continue
        
        dataset_dirs = sorted(dataset_dir.glob("ds_*"))[:num_samples]
        
        for ds_dir in dataset_dirs:
            ssc_path = ds_dir / "system_state_captured_unique.json"
            if not ssc_path.exists():
                continue
            
            try:
                with open(ssc_path, 'r') as f:
                    data = json.load(f)
                
                task_placements = data.get('task_placements', [])
                if not task_placements:
                    continue
                
                # Extract per-task data
                for tp in task_placements:
                    task_data = {
                        "task_id": tp.get('task_id'),
                        "task_type": tp.get('task_type'),
                        "timestamp": data.get('timestamp', 0),
                        "queue_snapshot": tp.get('full_queue_snapshot', {}),
                        "temporal_state": tp.get('temporal_state_at_scheduling', {}),
                        "replicas": data.get('replicas', {})
                    }
                    stats["per_task_data"].append(task_data)
                    
                    # Extract queue snapshot
                    full_queue = tp.get('full_queue_snapshot', {})
                    for platform_key, queue_len in full_queue.items():
                        stats["queue_lengths"].append(int(queue_len))
                        stats["platform_usage"][platform_key] += 1
                    
                    # Extract temporal state
                    temp_state = tp.get('temporal_state_at_scheduling', {})
                    if isinstance(temp_state, dict):
                        for platform_key, state_dict in temp_state.items():
                            if isinstance(state_dict, dict):
                                stats["temporal_states"]["current_task_remaining"].append(
                                    state_dict.get('current_task_remaining', 0.0)
                                )
                                stats["temporal_states"]["cold_start_remaining"].append(
                                    state_dict.get('cold_start_remaining', 0.0)
                                )
                                stats["temporal_states"]["comm_remaining"].append(
                                    state_dict.get('comm_remaining', 0.0)
                                )
                
                # Extract replicas
                replicas = data.get('replicas', {})
                for task_type, replica_list in replicas.items():
                    stats["replica_counts"][task_type] += len(replica_list)
                
                stats["samples_analyzed"] += 1
                
            except Exception as e:
                print(f"  Warning: Failed to process {ds_dir.name}: {e}")
                continue
    
    return stats


def extract_real_simulation_statistics(sample_size: int = 2000) -> Dict[str, Any]:
    """
    Extract queue and system state statistics from real simulation.
    
    UPDATED: Now uses per-task systemStateResult, fullQueueSnapshot, and 
    temporalStateAtScheduling captured at scheduling time.
    """
    stats = {
        "queue_lengths": [],
        "queue_snapshots": [],
        "temporal_states": {
            "current_task_remaining": [],
            "cold_start_remaining": [],
            "comm_remaining": []
        },
        "replica_counts": defaultdict(int),
        "platform_usage": defaultdict(int),
        "gnn_decision_times": [],
        "queue_times": [],
        "elapsed_times": [],
        "samples_analyzed": 0,
        "per_task_data": [],  # NEW: Store per-task data with full system state
        "system_state_evolution": []  # NEW: Track system state over time
    }
    
    if not SIM_RESULT_FILE.exists():
        return stats
    
    print(f"Loading real simulation results from {SIM_RESULT_FILE}...")
    with open(SIM_RESULT_FILE, 'r') as f:
        data = json.load(f)
    
    task_results = data.get('stats', {}).get('taskResults', [])
    
    # Sample task results
    if task_results:
        sample_indices = np.linspace(0, len(task_results) - 1, 
                                   min(sample_size, len(task_results)), dtype=int)
        
        for idx in sample_indices:
            tr = task_results[idx]
            
            # NEW: Extract per-task system state (now available at scheduling time)
            system_state = tr.get('systemStateResult')
            full_queue = tr.get('fullQueueSnapshot', {})
            queue_snapshot = tr.get('queueSnapshotAtScheduling', {})
            temporal_state = tr.get('temporalStateAtScheduling', {})
            
            # Store per-task data with full system state
            task_data = {
                "task_id": tr.get('taskId'),
                "task_type": tr.get('taskType', {}).get('name'),
                "scheduled_time": tr.get('scheduledTime'),
                "system_state": system_state,
                "full_queue_snapshot": full_queue,
                "queue_snapshot": queue_snapshot,
                "temporal_state": temporal_state,
                "queue_time": tr.get('queueTime'),
                "elapsed_time": tr.get('elapsedTime')
            }
            stats["per_task_data"].append(task_data)
            
            # Extract system state metrics
            if system_state:
                timestamp = system_state.get('timestamp', 0)
                replicas = system_state.get('replicas', {})
                queue_occupancy = system_state.get('queue_occupancy', {})
                
                # Track system state evolution
                state_snapshot = {
                    "timestamp": timestamp,
                    "task_id": tr.get('taskId'),
                    "replica_counts": {k: len(v) for k, v in replicas.items()},
                    "total_replicas": sum(len(v) for v in replicas.values()),
                    "queue_occupancy": queue_occupancy
                }
                stats["system_state_evolution"].append(state_snapshot)
                
                # Extract replica counts
                for task_type, replica_list in replicas.items():
                    stats["replica_counts"][task_type] += len(replica_list)
            
            # Extract queue snapshots
            if full_queue:
                stats["queue_snapshots"].append(full_queue)
                for platform_key, queue_len in full_queue.items():
                    stats["queue_lengths"].append(int(queue_len))
                    stats["platform_usage"][platform_key] += 1
            
            # Extract temporal state
            if isinstance(temporal_state, dict):
                for platform_key, state_dict in temporal_state.items():
                    if isinstance(state_dict, dict):
                        stats["temporal_states"]["current_task_remaining"].append(
                            state_dict.get('current_task_remaining', 0.0)
                        )
                        stats["temporal_states"]["cold_start_remaining"].append(
                            state_dict.get('cold_start_remaining', 0.0)
                        )
                        stats["temporal_states"]["comm_remaining"].append(
                            state_dict.get('comm_remaining', 0.0)
                        )
            
            # Extract metrics
            if tr.get('gnn_decision_time'):
                stats["gnn_decision_times"].append(tr['gnn_decision_time'])
            if tr.get('queueTime'):
                stats["queue_times"].append(tr['queueTime'])
            if tr.get('elapsedTime'):
                stats["elapsed_times"].append(tr['elapsedTime'])
            
            stats["samples_analyzed"] += 1
    
    # Sample system states (periodic captures)
    system_states = data.get('stats', {}).get('systemStateResults', [])
    if system_states:
        sample_indices = [0, len(system_states) // 4, len(system_states) // 2, 
                        3 * len(system_states) // 4, len(system_states) - 1]
        stats['system_state_samples'] = []
        for idx in sample_indices:
            if idx < len(system_states):
                state = system_states[idx]
                stats['system_state_samples'].append({
                    "timestamp": state.get('timestamp'),
                    "replicas": {
                        k: len(v) for k, v in state.get('replicas', {}).items()
                    },
                    "queue_occupancy_keys": list(state.get('queue_occupancy', {}).keys()),
                    "queue_occupancy_sample": {
                        k: len(v) if isinstance(v, dict) else v
                        for k, v in list(state.get('queue_occupancy', {}).items())[:3]
                    } if state.get('queue_occupancy') else None
                })
    
    return stats


def analyze_temporal_evolution(real_stats: Dict) -> Dict[str, Any]:
    """
    Analyze how system state evolves over time in real simulation.
    
    NEW: Uses per-task system state captures to track evolution.
    """
    evolution = {
        "replica_evolution": [],
        "queue_evolution": [],
        "temporal_evolution": [],
        "state_transitions": []
    }
    
    if not real_stats.get("system_state_evolution"):
        return evolution
    
    # Sort by timestamp
    state_history = sorted(real_stats["system_state_evolution"], 
                          key=lambda x: x["timestamp"])
    
    # Analyze replica evolution
    timestamps = [s["timestamp"] for s in state_history]
    total_replicas = [s["total_replicas"] for s in state_history]
    dnn1_replicas = [s["replica_counts"].get("dnn1", 0) for s in state_history]
    dnn2_replicas = [s["replica_counts"].get("dnn2", 0) for s in state_history]
    
    evolution["replica_evolution"] = {
        "timestamps": timestamps,
        "total_replicas": total_replicas,
        "dnn1_replicas": dnn1_replicas,
        "dnn2_replicas": dnn2_replicas,
        "mean_total": float(np.mean(total_replicas)) if total_replicas else 0.0,
        "max_total": int(np.max(total_replicas)) if total_replicas else 0,
        "min_total": int(np.min(total_replicas)) if total_replicas else 0
    }
    
    # Analyze queue evolution from per-task data
    queue_lengths_by_time = []
    for task_data in real_stats.get("per_task_data", []):
        timestamp = task_data.get("scheduled_time", 0)
        full_queue = task_data.get("full_queue_snapshot", {})
        if full_queue:
            queue_values = [int(v) for v in full_queue.values()]
            if queue_values:
                queue_lengths_by_time.append({
                    "timestamp": timestamp,
                    "mean_queue": float(np.mean(queue_values)),
                    "max_queue": int(np.max(queue_values)),
                    "total_queued": sum(queue_values)
                })
    
    if queue_lengths_by_time:
        queue_lengths_by_time.sort(key=lambda x: x["timestamp"])
        evolution["queue_evolution"] = {
            "timestamps": [q["timestamp"] for q in queue_lengths_by_time],
            "mean_queues": [q["mean_queue"] for q in queue_lengths_by_time],
            "max_queues": [q["max_queue"] for q in queue_lengths_by_time],
            "total_queued": [q["total_queued"] for q in queue_lengths_by_time]
        }
    
    # Analyze temporal state evolution
    temporal_by_time = []
    for task_data in real_stats.get("per_task_data", []):
        timestamp = task_data.get("scheduled_time", 0)
        temporal_state = task_data.get("temporal_state", {})
        if temporal_state:
            current_task_remaining = []
            comm_remaining = []
            for platform_key, state_dict in temporal_state.items():
                if isinstance(state_dict, dict):
                    current_task_remaining.append(
                        state_dict.get('current_task_remaining', 0.0)
                    )
                    comm_remaining.append(
                        state_dict.get('comm_remaining', 0.0)
                    )
            
            if current_task_remaining:
                temporal_by_time.append({
                    "timestamp": timestamp,
                    "mean_current_task_remaining": float(np.mean(current_task_remaining)),
                    "mean_comm_remaining": float(np.mean(comm_remaining)) if comm_remaining else 0.0
                })
    
    if temporal_by_time:
        temporal_by_time.sort(key=lambda x: x["timestamp"])
        evolution["temporal_evolution"] = {
            "timestamps": [t["timestamp"] for t in temporal_by_time],
            "mean_current_task_remaining": [t["mean_current_task_remaining"] for t in temporal_by_time],
            "mean_comm_remaining": [t["mean_comm_remaining"] for t in temporal_by_time]
        }
    
    return evolution


def compare_per_task_state(train_stats: Dict, real_stats: Dict) -> Dict[str, Any]:
    """
    Compare system state at similar timestamps between training and real simulation.
    
    NEW: Uses per-task system state captures for detailed comparison.
    """
    comparison = {
        "early_simulation": {},  # t < 10s
        "mid_simulation": {},    # 10s <= t < 100s
        "late_simulation": {},   # t >= 100s
        "overall": {}
    }
    
    # Group training data by timestamp ranges
    train_early = [td for td in train_stats.get("per_task_data", []) 
                   if td.get("timestamp", 0) < 10.0]
    train_mid = [td for td in train_stats.get("per_task_data", []) 
                 if 10.0 <= td.get("timestamp", 0) < 100.0]
    train_late = [td for td in train_stats.get("per_task_data", []) 
                  if td.get("timestamp", 0) >= 100.0]
    
    # Group real simulation data by timestamp ranges
    real_early = [td for td in real_stats.get("per_task_data", []) 
                  if td.get("scheduled_time", 0) < 10.0]
    real_mid = [td for td in real_stats.get("per_task_data", []) 
                if 10.0 <= td.get("scheduled_time", 0) < 100.0]
    real_late = [td for td in real_stats.get("per_task_data", []) 
                 if td.get("scheduled_time", 0) >= 100.0]
    
    def analyze_group(train_group, real_group, name):
        """Analyze a group of tasks in a time range."""
        result = {
            "training_count": len(train_group),
            "real_count": len(real_group),
            "replica_comparison": {},
            "queue_comparison": {},
            "temporal_comparison": {}
        }
        
        # Compare replicas
        if train_group and real_group:
            train_replicas = []
            real_replicas = []
            
            for td in train_group:
                replicas = td.get("replicas", {})
                train_replicas.append(sum(len(v) for v in replicas.values()))
            
            for td in real_group:
                system_state = td.get("system_state")
                if system_state:
                    replicas = system_state.get("replicas", {})
                    real_replicas.append(sum(len(v) for v in replicas.values()))
            
            if train_replicas and real_replicas:
                result["replica_comparison"] = {
                    "training_mean": float(np.mean(train_replicas)),
                    "real_mean": float(np.mean(real_replicas)),
                    "difference": float(np.mean(real_replicas) - np.mean(train_replicas))
                }
            
            # Compare queues
            train_queues = []
            real_queues = []
            
            for td in train_group:
                queue = td.get("queue_snapshot", {})
                train_queues.extend([int(v) for v in queue.values()])
            
            for td in real_group:
                queue = td.get("full_queue_snapshot", {})
                real_queues.extend([int(v) for v in queue.values()])
            
            if train_queues and real_queues:
                result["queue_comparison"] = {
                    "training_mean": float(np.mean(train_queues)),
                    "real_mean": float(np.mean(real_queues)),
                    "difference": float(np.mean(real_queues) - np.mean(train_queues)),
                    "training_max": int(np.max(train_queues)),
                    "real_max": int(np.max(real_queues))
                }
        
        return result
    
    comparison["early_simulation"] = analyze_group(train_early, real_early, "early")
    comparison["mid_simulation"] = analyze_group(train_mid, real_mid, "mid")
    comparison["late_simulation"] = analyze_group(train_late, real_late, "late")
    
    # Overall comparison
    comparison["overall"] = analyze_group(
        train_stats.get("per_task_data", []),
        real_stats.get("per_task_data", []),
        "overall"
    )
    
    return comparison


def compare_statistics(train_stats: Dict, real_stats: Dict) -> Dict[str, Any]:
    """Compare training and real simulation statistics."""
    comparison = {
        "queue_lengths": {},
        "temporal_states": {},
        "replica_counts": {},
        "platform_usage": {},
        "discrepancies": [],
        "temporal_evolution": {},
        "per_task_comparison": {}
    }
    
    # Compare queue lengths
    train_queues = np.array(train_stats["queue_lengths"]) if train_stats["queue_lengths"] else np.array([])
    real_queues = np.array(real_stats["queue_lengths"]) if real_stats["queue_lengths"] else np.array([])
    
    if len(train_queues) > 0 and len(real_queues) > 0:
        comparison["queue_lengths"] = {
            "training": {
                "mean": float(np.mean(train_queues)),
                "median": float(np.median(train_queues)),
                "std": float(np.std(train_queues)),
                "min": int(np.min(train_queues)),
                "max": int(np.max(train_queues)),
                "p95": float(np.percentile(train_queues, 95)),
                "p99": float(np.percentile(train_queues, 99)),
                "non_zero_pct": float(np.sum(train_queues > 0) / len(train_queues) * 100)
            },
            "real": {
                "mean": float(np.mean(real_queues)),
                "median": float(np.median(real_queues)),
                "std": float(np.std(real_queues)),
                "min": int(np.min(real_queues)),
                "max": int(np.max(real_queues)),
                "p95": float(np.percentile(real_queues, 95)),
                "p99": float(np.percentile(real_queues, 99)),
                "non_zero_pct": float(np.sum(real_queues > 0) / len(real_queues) * 100)
            },
            "difference": {
                "mean_diff": float(np.mean(real_queues) - np.mean(train_queues)),
                "median_diff": float(np.median(real_queues) - np.median(train_queues)),
                "std_diff": float(np.std(real_queues) - np.std(train_queues)),
                "max_diff": int(np.max(real_queues) - np.max(train_queues))
            }
        }
        
        # Identify discrepancies
        mean_diff = comparison["queue_lengths"]["difference"]["mean_diff"]
        if abs(mean_diff) > 1.0:
            comparison["discrepancies"].append({
                "type": "queue_length_mean",
                "severity": "high" if abs(mean_diff) > 5.0 else "medium",
                "description": f"Mean queue length differs by {mean_diff:.2f} (training: {comparison['queue_lengths']['training']['mean']:.2f}, real: {comparison['queue_lengths']['real']['mean']:.2f})",
                "recommendation": "Check if co-simulation captures queue state correctly at scheduling time"
            })
    
    # Compare temporal states
    for temp_key in ["current_task_remaining", "cold_start_remaining", "comm_remaining"]:
        train_temps = np.array(train_stats["temporal_states"][temp_key]) if train_stats["temporal_states"][temp_key] else np.array([])
        real_temps = np.array(real_stats["temporal_states"][temp_key]) if real_stats["temporal_states"][temp_key] else np.array([])
        
        if len(train_temps) > 0 and len(real_temps) > 0:
            comparison["temporal_states"][temp_key] = {
                "training": {
                    "mean": float(np.mean(train_temps)),
                    "median": float(np.median(train_temps)),
                    "std": float(np.std(train_temps)),
                    "max": float(np.max(train_temps))
                },
                "real": {
                    "mean": float(np.mean(real_temps)),
                    "median": float(np.median(real_temps)),
                    "std": float(np.std(real_temps)),
                    "max": float(np.max(real_temps))
                },
                "difference": {
                    "mean_diff": float(np.mean(real_temps) - np.mean(train_temps))
                }
            }
            
            mean_diff = comparison["temporal_states"][temp_key]["difference"]["mean_diff"]
            if abs(mean_diff) > 0.1:
                comparison["discrepancies"].append({
                    "type": f"temporal_{temp_key}",
                    "severity": "medium",
                    "description": f"{temp_key} differs by {mean_diff:.4f}s (training: {comparison['temporal_states'][temp_key]['training']['mean']:.4f}s, real: {comparison['temporal_states'][temp_key]['real']['mean']:.4f}s)",
                    "recommendation": "Verify temporal state capture timing in co-simulation matches real simulation"
                })
    
    # Compare replica counts
    all_task_types = set(train_stats["replica_counts"].keys()) | set(real_stats["replica_counts"].keys())
    for task_type in all_task_types:
        train_count = train_stats["replica_counts"].get(task_type, 0) / max(train_stats["samples_analyzed"], 1)
        real_count = real_stats["replica_counts"].get(task_type, 0) / max(real_stats["samples_analyzed"], 1)
        
        comparison["replica_counts"][task_type] = {
            "training_avg": float(train_count),
            "real_avg": float(real_count),
            "difference": float(real_count - train_count)
        }
        
        if abs(real_count - train_count) > 0.5:
            comparison["discrepancies"].append({
                "type": f"replica_count_{task_type}",
                "severity": "medium",
                "description": f"Average {task_type} replicas differ by {abs(real_count - train_count):.2f} (training: {train_count:.2f}, real: {real_count:.2f})",
                "recommendation": "Check replica scaling logic in co-simulation vs real simulation"
            })
    
    # NEW: Analyze temporal evolution
    comparison["temporal_evolution"] = analyze_temporal_evolution(real_stats)
    
    # NEW: Per-task state comparison
    comparison["per_task_comparison"] = compare_per_task_state(train_stats, real_stats)
    
    # Compare platform usage
    all_platforms = set(train_stats["platform_usage"].keys()) | set(real_stats["platform_usage"].keys())
    train_platform_counts = Counter(train_stats["platform_usage"])
    real_platform_counts = Counter(real_stats["platform_usage"])
    
    top_platforms = sorted(all_platforms, 
                          key=lambda p: train_platform_counts.get(p, 0) + real_platform_counts.get(p, 0),
                          reverse=True)[:20]
    
    comparison["platform_usage"] = {
        "top_platforms": []
    }
    
    for platform in top_platforms:
        train_count = train_platform_counts.get(platform, 0)
        real_count = real_platform_counts.get(platform, 0)
        train_pct = train_count / max(train_stats["samples_analyzed"], 1) * 100
        real_pct = real_count / max(real_stats["samples_analyzed"], 1) * 100
        
        comparison["platform_usage"]["top_platforms"].append({
            "platform": platform,
            "training_count": train_count,
            "real_count": real_count,
            "training_pct": float(train_pct),
            "real_pct": float(real_pct),
            "difference_pct": float(real_pct - train_pct)
        })
    
    return comparison


def generate_detailed_report(train_stats: Dict, real_stats: Dict, comparison: Dict) -> str:
    """Generate a detailed comparison report."""
    report = []
    report.append("=" * 80)
    report.append("REAL SIMULATION vs TRAINING/CO-SIMULATION STATE COMPARISON")
    report.append("UPDATED: Using per-task system state captures")
    report.append("=" * 80)
    report.append("")
    
    # Summary
    report.append("SUMMARY")
    report.append("-" * 80)
    report.append(f"Training samples analyzed: {train_stats['samples_analyzed']}")
    report.append(f"Real simulation samples analyzed: {real_stats['samples_analyzed']}")
    report.append(f"Training per-task data points: {len(train_stats.get('per_task_data', []))}")
    report.append(f"Real per-task data points: {len(real_stats.get('per_task_data', []))}")
    report.append(f"Total discrepancies found: {len(comparison['discrepancies'])}")
    report.append("")
    
    # NEW: Temporal Evolution Analysis
    if comparison.get("temporal_evolution"):
        evo = comparison["temporal_evolution"]
        report.append("TEMPORAL EVOLUTION ANALYSIS")
        report.append("-" * 80)
        
        if evo.get("replica_evolution"):
            rep_evo = evo["replica_evolution"]
            report.append("Replica Evolution Over Time:")
            report.append(f"  Mean total replicas: {rep_evo['mean_total']:.2f}")
            report.append(f"  Range: [{rep_evo['min_total']}, {rep_evo['max_total']}]")
            if rep_evo.get('timestamps'):
                report.append(f"  Time range: [{rep_evo['timestamps'][0]:.2f}s, {rep_evo['timestamps'][-1]:.2f}s]")
            report.append("")
        
        if evo.get("queue_evolution"):
            queue_evo = evo["queue_evolution"]
            if queue_evo.get("timestamps"):
                report.append("Queue Evolution Over Time:")
                report.append(f"  Mean queue length: {np.mean(queue_evo['mean_queues']):.2f}")
                report.append(f"  Max queue length: {max(queue_evo['max_queues'])}")
                report.append(f"  Time range: [{queue_evo['timestamps'][0]:.2f}s, {queue_evo['timestamps'][-1]:.2f}s]")
                report.append("")
        
        if evo.get("temporal_evolution"):
            temp_evo = evo["temporal_evolution"]
            if temp_evo.get("timestamps"):
                report.append("Temporal State Evolution:")
                report.append(f"  Mean current_task_remaining: {np.mean(temp_evo['mean_current_task_remaining']):.4f}s")
                report.append(f"  Mean comm_remaining: {np.mean(temp_evo['mean_comm_remaining']):.4f}s")
                report.append("")
    
    # NEW: Per-Task State Comparison by Time Range
    if comparison.get("per_task_comparison"):
        per_task = comparison["per_task_comparison"]
        report.append("PER-TASK STATE COMPARISON BY TIME RANGE")
        report.append("-" * 80)
        
        for time_range in ["early_simulation", "mid_simulation", "late_simulation"]:
            if time_range in per_task:
                group = per_task[time_range]
                report.append(f"{time_range.replace('_', ' ').title()}:")
                report.append(f"  Training tasks: {group['training_count']}")
                report.append(f"  Real tasks: {group['real_count']}")
                
                if group.get("replica_comparison"):
                    rc = group["replica_comparison"]
                    if rc.get("training_mean") is not None:
                        report.append(f"  Replicas - Training: {rc['training_mean']:.2f}, Real: {rc['real_mean']:.2f}, Diff: {rc['difference']:+.2f}")
                
                if group.get("queue_comparison"):
                    qc = group["queue_comparison"]
                    if qc.get("training_mean") is not None:
                        report.append(f"  Queues - Training: {qc['training_mean']:.2f}, Real: {qc['real_mean']:.2f}, Diff: {qc['difference']:+.2f}")
                        report.append(f"  Max queues - Training: {qc['training_max']}, Real: {qc['real_max']}")
                report.append("")
    
    # Queue Length Statistics
    report.append("1. QUEUE LENGTH STATISTICS")
    report.append("-" * 80)
    if comparison.get("queue_lengths"):
        ql = comparison["queue_lengths"]
        report.append("Training (Co-Simulation):")
        train = ql["training"]
        report.append(f"  Mean: {train['mean']:.2f}")
        report.append(f"  Median: {train['median']:.2f}")
        report.append(f"  Std Dev: {train['std']:.2f}")
        report.append(f"  Range: [{train['min']}, {train['max']}]")
        report.append(f"  95th percentile: {train['p95']:.2f}")
        report.append(f"  99th percentile: {train['p99']:.2f}")
        report.append(f"  Non-zero percentage: {train['non_zero_pct']:.2f}%")
        report.append("")
        report.append("Real Simulation:")
        real = ql["real"]
        report.append(f"  Mean: {real['mean']:.2f}")
        report.append(f"  Median: {real['median']:.2f}")
        report.append(f"  Std Dev: {real['std']:.2f}")
        report.append(f"  Range: [{real['min']}, {real['max']}]")
        report.append(f"  95th percentile: {real['p95']:.2f}")
        report.append(f"  99th percentile: {real['p99']:.2f}")
        report.append(f"  Non-zero percentage: {real['non_zero_pct']:.2f}%")
        report.append("")
        report.append("Differences:")
        diff = ql["difference"]
        report.append(f"  Mean difference: {diff['mean_diff']:+.2f} (real - training)")
        report.append(f"  Median difference: {diff['median_diff']:+.2f}")
        report.append(f"  Std Dev difference: {diff['std_diff']:+.2f}")
        report.append(f"  Max difference: {diff['max_diff']:+d}")
        report.append("")
        
        # Interpretation
        if abs(diff['mean_diff']) > 1.0:
            report.append("⚠️  SIGNIFICANT DIFFERENCE DETECTED")
            if diff['mean_diff'] > 0:
                report.append("   Real simulation has HIGHER queue lengths on average.")
                report.append("   This suggests co-simulation may underestimate queue congestion.")
            else:
                report.append("   Real simulation has LOWER queue lengths on average.")
                report.append("   This suggests co-simulation may overestimate queue congestion.")
        report.append("")
    
    # Temporal State Statistics
    report.append("2. TEMPORAL STATE STATISTICS")
    report.append("-" * 80)
    if comparison.get("temporal_states"):
        for temp_key, temp_data in comparison["temporal_states"].items():
            report.append(f"{temp_key.replace('_', ' ').title()}:")
            train = temp_data["training"]
            real = temp_data["real"]
            diff = temp_data["difference"]
            report.append(f"  Training mean: {train['mean']:.4f}s (std: {train['std']:.4f}s)")
            report.append(f"  Real mean: {real['mean']:.4f}s (std: {real['std']:.4f}s)")
            report.append(f"  Difference: {diff['mean_diff']:+.4f}s")
            if abs(diff['mean_diff']) > 0.1:
                report.append(f"  ⚠️  Significant difference detected")
            report.append("")
    
    # Replica Counts
    report.append("3. REPLICA COUNTS")
    report.append("-" * 80)
    if comparison.get("replica_counts"):
        for task_type, counts in comparison["replica_counts"].items():
            report.append(f"{task_type}:")
            report.append(f"  Training average: {counts['training_avg']:.2f}")
            report.append(f"  Real average: {counts['real_avg']:.2f}")
            report.append(f"  Difference: {counts['difference']:+.2f}")
            if abs(counts['difference']) > 0.5:
                report.append(f"  ⚠️  Significant difference detected")
            report.append("")
    
    # Platform Usage
    report.append("4. PLATFORM USAGE (Top 20)")
    report.append("-" * 80)
    if comparison.get("platform_usage") and comparison["platform_usage"].get("top_platforms"):
        report.append(f"{'Platform':<30} {'Training %':<15} {'Real %':<15} {'Difference':<15}")
        report.append("-" * 75)
        for plat_info in comparison["platform_usage"]["top_platforms"][:20]:
            plat = plat_info["platform"]
            train_pct = plat_info["training_pct"]
            real_pct = plat_info["real_pct"]
            diff_pct = plat_info["difference_pct"]
            marker = "⚠️" if abs(diff_pct) > 5.0 else "  "
            report.append(f"{marker} {plat:<28} {train_pct:>6.2f}%      {real_pct:>6.2f}%      {diff_pct:>+6.2f}%")
    report.append("")
    
    # Real Simulation Metrics
    report.append("5. REAL SIMULATION METRICS")
    report.append("-" * 80)
    if real_stats.get("gnn_decision_times"):
        gnn_times = np.array(real_stats["gnn_decision_times"])
        report.append(f"GNN Decision Times:")
        report.append(f"  Mean: {np.mean(gnn_times):.6f}s")
        report.append(f"  Median: {np.median(gnn_times):.6f}s")
        report.append(f"  Max: {np.max(gnn_times):.6f}s")
        report.append("")
    
    if real_stats.get("queue_times"):
        queue_times = np.array(real_stats["queue_times"])
        report.append(f"Queue Times:")
        report.append(f"  Mean: {np.mean(queue_times):.2f}s")
        report.append(f"  Median: {np.median(queue_times):.2f}s")
        report.append(f"  Max: {np.max(queue_times):.2f}s")
        report.append("")
    
    # Discrepancies and Recommendations
    report.append("6. DISCREPANCIES AND RECOMMENDATIONS")
    report.append("-" * 80)
    if comparison.get("discrepancies"):
        high_severity = [d for d in comparison["discrepancies"] if d["severity"] == "high"]
        medium_severity = [d for d in comparison["discrepancies"] if d["severity"] == "medium"]
        
        if high_severity:
            report.append("HIGH SEVERITY:")
            for disc in high_severity:
                report.append(f"  ⚠️  {disc['type']}: {disc['description']}")
                report.append(f"     Recommendation: {disc['recommendation']}")
                report.append("")
        
        if medium_severity:
            report.append("MEDIUM SEVERITY:")
            for disc in medium_severity:
                report.append(f"  • {disc['type']}: {disc['description']}")
                report.append(f"    Recommendation: {disc['recommendation']}")
                report.append("")
    else:
        report.append("No significant discrepancies detected.")
        report.append("")
    
    # Action Items
    report.append("7. ACTION ITEMS FOR IMPROVEMENT")
    report.append("-" * 80)
    action_items = []
    
    if comparison.get("queue_lengths"):
        ql_diff = comparison["queue_lengths"]["difference"]["mean_diff"]
        if abs(ql_diff) > 1.0:
            action_items.append({
                "priority": "HIGH" if abs(ql_diff) > 5.0 else "MEDIUM",
                "item": "Queue Length Capture",
                "description": f"Mean queue length differs by {ql_diff:.2f}. Review queue snapshot timing in co-simulation."
            })
    
    if comparison.get("temporal_states"):
        for temp_key, temp_data in comparison["temporal_states"].items():
            if abs(temp_data["difference"]["mean_diff"]) > 0.1:
                action_items.append({
                    "priority": "MEDIUM",
                    "item": f"Temporal State: {temp_key}",
                    "description": f"Temporal state differs by {temp_data['difference']['mean_diff']:.4f}s. Verify capture timing."
                })
    
    if action_items:
        for item in sorted(action_items, key=lambda x: x["priority"]):
            report.append(f"[{item['priority']}] {item['item']}")
            report.append(f"  {item['description']}")
            report.append("")
    else:
        report.append("No immediate action items identified.")
        report.append("")
    
    return "\n".join(report)


def main():
    """Main entry point."""
    print("=" * 80)
    print("REAL SIMULATION vs TRAINING/CO-SIMULATION STATE COMPARISON")
    print("UPDATED: Using per-task system state captures")
    print("=" * 80)
    print()
    
    print("Step 1: Extracting training/co-simulation statistics...")
    train_stats = extract_training_queue_statistics(num_samples=200)
    print(f"  Analyzed {train_stats['samples_analyzed']} training samples")
    print(f"  Found {len(train_stats['queue_lengths'])} queue length measurements")
    print(f"  Found {len(train_stats['per_task_data'])} per-task data points")
    print()
    
    print("Step 2: Extracting real simulation statistics (with per-task state)...")
    real_stats = extract_real_simulation_statistics(sample_size=2000)
    print(f"  Analyzed {real_stats['samples_analyzed']} real simulation samples")
    print(f"  Found {len(real_stats['queue_lengths'])} queue length measurements")
    print(f"  Found {len(real_stats['per_task_data'])} per-task data points")
    print(f"  Found {len(real_stats.get('system_state_evolution', []))} system state evolution points")
    print()
    
    print("Step 3: Comparing statistics...")
    comparison = compare_statistics(train_stats, real_stats)
    print(f"  Found {len(comparison['discrepancies'])} discrepancies")
    print()
    
    print("Step 4: Generating report...")
    report = generate_detailed_report(train_stats, real_stats, comparison)
    print(report)
    
    # Save to file
    output_file = BASE_DIR / "scripts_cosim" / "REAL_VS_TRAINING_COMPARISON.txt"
    with open(output_file, 'w') as f:
        f.write(report)
    
    print()
    print(f"✓ Detailed comparison saved to: {output_file}")


if __name__ == "__main__":
    main()
