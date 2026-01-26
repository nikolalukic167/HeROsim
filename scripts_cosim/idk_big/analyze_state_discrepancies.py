#!/usr/bin/env python3
"""
Deep Dive: Analyze Root Causes of State Discrepancies

UPDATED: Now uses per-task system state captures from GNN scheduler.
Each task has systemStateResult captured at scheduling time, enabling
detailed analysis of system state evolution and discrepancies.

This script performs detailed analysis of specific discrepancies:
1. Queue length differences (why real sim has much higher queues)
2. Replica count differences (why training has replicas but real doesn't)
3. Temporal state timing issues
4. System state capture timing differences
5. NEW: System state evolution over time
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import numpy as np

BASE_DIR = Path("/root/projects/my-herosim")
ARTIFACTS_DIR = BASE_DIR / "simulation_data/artifacts/run_non_unique"
SIM_RESULT_FILE = BASE_DIR / "simulation_data/results/simulation_result_gnn.json"


def analyze_replica_discrepancy() -> Dict[str, Any]:
    """
    Analyze why replica counts differ so dramatically.
    
    UPDATED: Uses per-task systemStateResult to track replica evolution.
    """
    analysis = {
        "training_replicas": {},
        "real_replicas": {},
        "replica_evolution": {},
        "root_cause": []
    }
    
    # Analyze training replicas
    print("Analyzing training replica state...")
    training_samples = []
    for dataset_type in ["gnn_datasets_2tasks", "gnn_datasets_3tasks"]:
        dataset_dir = ARTIFACTS_DIR / dataset_type
        if not dataset_dir.exists():
            continue
        
        for ds_dir in sorted(dataset_dir.glob("ds_*"))[:50]:
            ssc_path = ds_dir / "system_state_captured_unique.json"
            if not ssc_path.exists():
                continue
            
            try:
                with open(ssc_path, 'r') as f:
                    data = json.load(f)
                
                replicas = data.get('replicas', {})
                timestamp = data.get('timestamp', 0)
                num_tasks = data.get('num_tasks', 0)
                
                training_samples.append({
                    "dataset": ds_dir.name,
                    "timestamp": timestamp,
                    "num_tasks": num_tasks,
                    "replicas": {
                        k: len(v) for k, v in replicas.items()
                    },
                    "total_replicas": sum(len(v) for v in replicas.values())
                })
            except Exception as e:
                continue
    
    if training_samples:
        analysis["training_replicas"] = {
            "sample_count": len(training_samples),
            "avg_total_replicas": np.mean([s["total_replicas"] for s in training_samples]),
            "avg_dnn1_replicas": np.mean([s["replicas"].get("dnn1", 0) for s in training_samples]),
            "avg_dnn2_replicas": np.mean([s["replicas"].get("dnn2", 0) for s in training_samples]),
            "samples": training_samples[:10]  # First 10 for inspection
        }
    
    # NEW: Analyze real simulation replicas from per-task systemStateResult
    print("Analyzing real simulation replica state (from per-task captures)...")
    if SIM_RESULT_FILE.exists():
        with open(SIM_RESULT_FILE, 'r') as f:
            data = json.load(f)
        
        task_results = data.get('stats', {}).get('taskResults', [])
        real_samples = []
        replica_evolution = []
        
        # Extract replica state from each task's systemStateResult
        sample_indices = np.linspace(0, len(task_results) - 1, 
                                   min(1000, len(task_results)), dtype=int)
        
        for idx in sample_indices:
            tr = task_results[idx]
            system_state = tr.get('systemStateResult')
            
            if system_state:
                timestamp = system_state.get('timestamp', tr.get('scheduledTime', 0))
                replicas = system_state.get('replicas', {})
                
                replica_info = {
                    k: len(v) for k, v in replicas.items()
                }
                total_replicas = sum(replica_info.values())
                
                real_samples.append({
                    "task_id": tr.get('taskId'),
                    "timestamp": timestamp,
                    "replicas": replica_info,
                    "total_replicas": total_replicas
                })
                
                replica_evolution.append({
                    "timestamp": timestamp,
                    "task_id": tr.get('taskId'),
                    "total_replicas": total_replicas,
                    "dnn1_replicas": replica_info.get("dnn1", 0),
                    "dnn2_replicas": replica_info.get("dnn2", 0)
                })
        
        if real_samples:
            analysis["real_replicas"] = {
                "sample_count": len(real_samples),
                "avg_total_replicas": np.mean([s["total_replicas"] for s in real_samples]),
                "avg_dnn1_replicas": np.mean([s["replicas"].get("dnn1", 0) for s in real_samples]),
                "avg_dnn2_replicas": np.mean([s["replicas"].get("dnn2", 0) for s in real_samples]),
                "samples": real_samples[:10]
            }
            
            # Analyze replica evolution
            if replica_evolution:
                replica_evolution.sort(key=lambda x: x["timestamp"])
                analysis["replica_evolution"] = {
                    "timestamps": [r["timestamp"] for r in replica_evolution],
                    "total_replicas": [r["total_replicas"] for r in replica_evolution],
                    "dnn1_replicas": [r["dnn1_replicas"] for r in replica_evolution],
                    "dnn2_replicas": [r["dnn2_replicas"] for r in replica_evolution],
                    "mean_total": float(np.mean([r["total_replicas"] for r in replica_evolution])),
                    "max_total": int(np.max([r["total_replicas"] for r in replica_evolution])),
                    "min_total": int(np.min([r["total_replicas"] for r in replica_evolution]))
                }
    
    # Root cause analysis
    if analysis["training_replicas"] and analysis["real_replicas"]:
        train_avg = analysis["training_replicas"]["avg_total_replicas"]
        real_avg = analysis["real_replicas"]["avg_total_replicas"]
        
        if train_avg > 10 and real_avg < 1:
            analysis["root_cause"].append({
                "issue": "Replica State Mismatch",
                "description": f"Training shows {train_avg:.1f} replicas on average, but real simulation shows {real_avg:.1f}",
                "likely_cause": "Co-simulation captures state AFTER warmup tasks create replicas, but real simulation starts from zero (autoscaling from zero)",
                "impact": "HIGH - This affects all queue and temporal state measurements",
                "recommendation": "Ensure co-simulation captures state at the SAME point as real simulation (before first task batch, not after warmup)"
            })
        elif abs(train_avg - real_avg) < 5:
            analysis["root_cause"].append({
                "issue": "Replica State Alignment Improved",
                "description": f"Replica counts are now closer: training {train_avg:.1f}, real {real_avg:.1f}",
                "likely_cause": "Proactive replica creation in GNN scheduler is working",
                "impact": "LOW - State alignment is better",
                "recommendation": "Continue monitoring replica creation patterns"
            })
    
    return analysis


def analyze_queue_timing() -> Dict[str, Any]:
    """
    Analyze queue length differences and timing.
    
    UPDATED: Uses per-task fullQueueSnapshot to track queue evolution.
    """
    analysis = {
        "queue_distributions": {},
        "queue_evolution": {},
        "queue_timing": {},
        "root_cause": []
    }
    
    # Extract queue lengths from training at different stages
    print("Analyzing queue timing in training data...")
    training_queues_by_stage = defaultdict(list)
    
    for dataset_type in ["gnn_datasets_2tasks"]:
        dataset_dir = ARTIFACTS_DIR / dataset_type
        if not dataset_dir.exists():
            continue
        
        for ds_dir in sorted(dataset_dir.glob("ds_*"))[:100]:
            ssc_path = ds_dir / "system_state_captured_unique.json"
            if not ssc_path.exists():
                continue
            
            try:
                with open(ssc_path, 'r') as f:
                    data = json.load(f)
                
                task_placements = data.get('task_placements', [])
                if not task_placements:
                    continue
                
                # Get queue snapshot from first task (scheduling time)
                full_queue = task_placements[0].get('full_queue_snapshot', {})
                queue_values = [int(v) for v in full_queue.values()]
                
                if queue_values:
                    training_queues_by_stage["at_scheduling"].extend(queue_values)
                
            except Exception:
                continue
    
    # NEW: Extract queue lengths from real simulation per-task captures
    print("Analyzing queue timing in real simulation (from per-task captures)...")
    real_queues_by_stage = defaultdict(list)
    queue_evolution = []
    
    if SIM_RESULT_FILE.exists():
        with open(SIM_RESULT_FILE, 'r') as f:
            data = json.load(f)
        
        task_results = data.get('stats', {}).get('taskResults', [])
        
        # Sample task results
        sample_indices = np.linspace(0, len(task_results) - 1, 
                                   min(1000, len(task_results)), dtype=int)
        
        for idx in sample_indices:
            tr = task_results[idx]
            full_queue = tr.get('fullQueueSnapshot', {})
            scheduled_time = tr.get('scheduledTime', 0)
            queue_values = [int(v) for v in full_queue.values()]
            
            if queue_values:
                real_queues_by_stage["at_scheduling"].extend(queue_values)
                
                # Track queue evolution
                queue_evolution.append({
                    "timestamp": scheduled_time,
                    "task_id": tr.get('taskId'),
                    "mean_queue": float(np.mean(queue_values)),
                    "max_queue": int(np.max(queue_values)),
                    "total_queued": sum(queue_values),
                    "non_zero_count": sum(1 for v in queue_values if v > 0)
                })
    
    # Compare distributions
    if training_queues_by_stage["at_scheduling"] and real_queues_by_stage["at_scheduling"]:
        train_queues = np.array(training_queues_by_stage["at_scheduling"])
        real_queues = np.array(real_queues_by_stage["at_scheduling"])
        
        analysis["queue_distributions"] = {
            "training": {
                "mean": float(np.mean(train_queues)),
                "median": float(np.median(train_queues)),
                "p95": float(np.percentile(train_queues, 95)),
                "p99": float(np.percentile(train_queues, 99)),
                "max": int(np.max(train_queues)),
                "non_zero_pct": float(np.sum(train_queues > 0) / len(train_queues) * 100)
            },
            "real": {
                "mean": float(np.mean(real_queues)),
                "median": float(np.median(real_queues)),
                "p95": float(np.percentile(real_queues, 95)),
                "p99": float(np.percentile(real_queues, 99)),
                "max": int(np.max(real_queues)),
                "non_zero_pct": float(np.sum(real_queues > 0) / len(real_queues) * 100)
            }
        }
        
        # NEW: Analyze queue evolution
        if queue_evolution:
            queue_evolution.sort(key=lambda x: x["timestamp"])
            analysis["queue_evolution"] = {
                "timestamps": [q["timestamp"] for q in queue_evolution],
                "mean_queues": [q["mean_queue"] for q in queue_evolution],
                "max_queues": [q["max_queue"] for q in queue_evolution],
                "total_queued": [q["total_queued"] for q in queue_evolution],
                "non_zero_counts": [q["non_zero_count"] for q in queue_evolution],
                "early_mean": float(np.mean([q["mean_queue"] for q in queue_evolution[:100]])),
                "late_mean": float(np.mean([q["mean_queue"] for q in queue_evolution[-100:]]))
            }
        
        # Root cause analysis
        mean_diff = analysis["queue_distributions"]["real"]["mean"] - analysis["queue_distributions"]["training"]["mean"]
        if mean_diff > 10:
            analysis["root_cause"].append({
                "issue": "Queue Length Mismatch",
                "description": f"Real simulation has mean queue length {mean_diff:.1f} higher than training",
                "likely_cause": "Real simulation accumulates queues over time (long-running), while co-simulation captures state for isolated batches",
                "impact": "HIGH - GNN trained on low-queue data but deployed in high-queue environment",
                "recommendation": "Generate training data with longer-running simulations or capture queue state at different simulation stages"
            })
    
    return analysis


def analyze_system_state_capture_timing() -> Dict[str, Any]:
    """
    Analyze when system state is captured in training vs real simulation.
    
    UPDATED: Uses per-task systemStateResult timestamps for detailed analysis.
    """
    analysis = {
        "training_capture": {},
        "real_capture": {},
        "timing_differences": [],
        "capture_frequency": {}
    }
    
    # Analyze training capture timing
    print("Analyzing system state capture timing...")
    training_timestamps = []
    training_task_counts = []
    
    for dataset_type in ["gnn_datasets_2tasks"]:
        dataset_dir = ARTIFACTS_DIR / dataset_type
        if not dataset_dir.exists():
            continue
        
        for ds_dir in sorted(dataset_dir.glob("ds_*"))[:100]:
            ssc_path = ds_dir / "system_state_captured_unique.json"
            if not ssc_path.exists():
                continue
            
            try:
                with open(ssc_path, 'r') as f:
                    data = json.load(f)
                
                timestamp = data.get('timestamp', 0)
                num_tasks = data.get('num_tasks', 0)
                
                training_timestamps.append(timestamp)
                training_task_counts.append(num_tasks)
            except Exception:
                continue
    
    if training_timestamps:
        analysis["training_capture"] = {
            "avg_timestamp": float(np.mean(training_timestamps)),
            "min_timestamp": float(np.min(training_timestamps)),
            "max_timestamp": float(np.max(training_timestamps)),
            "avg_task_count": float(np.mean(training_task_counts)),
            "sample_count": len(training_timestamps)
        }
    
    # NEW: Analyze real simulation capture timing from per-task systemStateResult
    print("Analyzing real simulation capture timing (from per-task captures)...")
    if SIM_RESULT_FILE.exists():
        with open(SIM_RESULT_FILE, 'r') as f:
            data = json.load(f)
        
        task_results = data.get('stats', {}).get('taskResults', [])
        real_timestamps = []
        
        # Extract timestamps from per-task systemStateResult
        for tr in task_results[:1000]:  # Sample first 1000
            system_state = tr.get('systemStateResult')
            if system_state:
                timestamp = system_state.get('timestamp', tr.get('scheduledTime', 0))
                real_timestamps.append(timestamp)
        
        if real_timestamps:
            analysis["real_capture"] = {
                "avg_timestamp": float(np.mean(real_timestamps)),
                "min_timestamp": float(np.min(real_timestamps)),
                "max_timestamp": float(np.max(real_timestamps)),
                "total_captures": len(real_timestamps),
                "simulation_duration": float(np.max(real_timestamps)) - float(np.min(real_timestamps)) if len(real_timestamps) > 1 else 0.0
            }
            
            # Analyze capture frequency
            if len(real_timestamps) > 1:
                time_diffs = np.diff(sorted(real_timestamps))
                analysis["capture_frequency"] = {
                    "mean_interval": float(np.mean(time_diffs)),
                    "median_interval": float(np.median(time_diffs)),
                    "min_interval": float(np.min(time_diffs)),
                    "max_interval": float(np.max(time_diffs))
                }
    
    # Identify timing differences
    if analysis["training_capture"] and analysis["real_capture"]:
        train_avg_time = analysis["training_capture"]["avg_timestamp"]
        real_avg_time = analysis["real_capture"]["avg_timestamp"]
        
        if train_avg_time < 10 and real_avg_time > 100:
            analysis["timing_differences"].append({
                "issue": "Capture Timing Mismatch",
                "description": f"Training captures state at ~{train_avg_time:.1f}s, real simulation at ~{real_avg_time:.1f}s",
                "impact": "Training sees early system state, real simulation sees later (more congested) state",
                "recommendation": "Capture training state at multiple simulation timestamps to match real deployment scenarios"
            })
        elif abs(train_avg_time - real_avg_time) < 50:
            analysis["timing_differences"].append({
                "issue": "Capture Timing Alignment Improved",
                "description": f"Timing is now closer: training ~{train_avg_time:.1f}s, real ~{real_avg_time:.1f}s",
                "impact": "LOW - Better alignment between training and real simulation",
                "recommendation": "Continue monitoring timing alignment"
            })
    
    return analysis


def analyze_per_task_state_evolution() -> Dict[str, Any]:
    """
    NEW: Analyze how system state evolves per task in real simulation.
    
    Uses per-task systemStateResult to track state changes.
    """
    analysis = {
        "state_transitions": [],
        "replica_growth": {},
        "queue_buildup": {},
        "temporal_changes": {}
    }
    
    if not SIM_RESULT_FILE.exists():
        return analysis
    
    print("Analyzing per-task system state evolution...")
    with open(SIM_RESULT_FILE, 'r') as f:
        data = json.load(f)
    
    task_results = data.get('stats', {}).get('taskResults', [])
    
    # Extract state for each task
    state_history = []
    for tr in task_results[:2000]:  # Sample first 2000 tasks
        system_state = tr.get('systemStateResult')
        if not system_state:
            continue
        
        timestamp = system_state.get('timestamp', tr.get('scheduledTime', 0))
        replicas = system_state.get('replicas', {})
        queue_occupancy = system_state.get('queue_occupancy', {})
        full_queue = tr.get('fullQueueSnapshot', {})
        temporal_state = tr.get('temporalStateAtScheduling', {})
        
        state_history.append({
            "task_id": tr.get('taskId'),
            "timestamp": timestamp,
            "total_replicas": sum(len(v) for v in replicas.values()),
            "dnn1_replicas": len(replicas.get("dnn1", [])),
            "dnn2_replicas": len(replicas.get("dnn2", [])),
            "mean_queue": float(np.mean([int(v) for v in full_queue.values()])) if full_queue else 0.0,
            "max_queue": int(np.max([int(v) for v in full_queue.values()])) if full_queue else 0,
            "temporal_mean_comm": float(np.mean([
                state_dict.get('comm_remaining', 0.0)
                for state_dict in temporal_state.values()
                if isinstance(state_dict, dict)
            ])) if temporal_state else 0.0
        })
    
    if state_history:
        state_history.sort(key=lambda x: x["timestamp"])
        
        # Analyze replica growth
        total_replicas = [s["total_replicas"] for s in state_history]
        analysis["replica_growth"] = {
            "initial": total_replicas[0] if total_replicas else 0,
            "final": total_replicas[-1] if total_replicas else 0,
            "growth": total_replicas[-1] - total_replicas[0] if len(total_replicas) > 1 else 0,
            "mean": float(np.mean(total_replicas)),
            "max": int(np.max(total_replicas))
        }
        
        # Analyze queue buildup
        mean_queues = [s["mean_queue"] for s in state_history]
        max_queues = [s["max_queue"] for s in state_history]
        analysis["queue_buildup"] = {
            "initial_mean": mean_queues[0] if mean_queues else 0.0,
            "final_mean": mean_queues[-1] if mean_queues else 0.0,
            "buildup": mean_queues[-1] - mean_queues[0] if len(mean_queues) > 1 else 0.0,
            "max_queue": int(np.max(max_queues)) if max_queues else 0
        }
        
        # Analyze temporal changes
        comm_means = [s["temporal_mean_comm"] for s in state_history]
        analysis["temporal_changes"] = {
            "initial_mean_comm": comm_means[0] if comm_means else 0.0,
            "final_mean_comm": comm_means[-1] if comm_means else 0.0,
            "change": comm_means[-1] - comm_means[0] if len(comm_means) > 1 else 0.0
        }
    
    return analysis


def generate_root_cause_report(analyses: Dict[str, Dict]) -> str:
    """Generate a root cause analysis report."""
    report = []
    report.append("=" * 80)
    report.append("ROOT CAUSE ANALYSIS: STATE DISCREPANCIES")
    report.append("UPDATED: Using per-task system state captures")
    report.append("=" * 80)
    report.append("")
    
    # Replica Analysis
    if "replica" in analyses:
        rep_analysis = analyses["replica"]
        report.append("1. REPLICA STATE ANALYSIS")
        report.append("-" * 80)
        
        if rep_analysis.get("training_replicas"):
            train = rep_analysis["training_replicas"]
            report.append("Training (Co-Simulation):")
            report.append(f"  Average total replicas: {train['avg_total_replicas']:.2f}")
            report.append(f"  Average dnn1 replicas: {train['avg_dnn1_replicas']:.2f}")
            report.append(f"  Average dnn2 replicas: {train['avg_dnn2_replicas']:.2f}")
            report.append(f"  Sample datasets: {train['sample_count']}")
            report.append("")
            report.append("  Sample states:")
            for sample in train.get("samples", [])[:5]:
                report.append(f"    {sample['dataset']}: {sample['total_replicas']} replicas at t={sample['timestamp']:.2f}s ({sample['num_tasks']} tasks)")
            report.append("")
        
        if rep_analysis.get("real_replicas"):
            real = rep_analysis["real_replicas"]
            report.append("Real Simulation (from per-task captures):")
            report.append(f"  Average total replicas: {real['avg_total_replicas']:.2f}")
            report.append(f"  Average dnn1 replicas: {real['avg_dnn1_replicas']:.2f}")
            report.append(f"  Average dnn2 replicas: {real['avg_dnn2_replicas']:.2f}")
            report.append("")
            report.append("  Sample states:")
            for sample in real.get("samples", [])[:5]:
                report.append(f"    Task {sample['task_id']}: {sample['total_replicas']} replicas at t={sample['timestamp']:.2f}s")
            report.append("")
        
        # NEW: Replica evolution
        if rep_analysis.get("replica_evolution"):
            evo = rep_analysis["replica_evolution"]
            report.append("Replica Evolution Over Time:")
            report.append(f"  Mean: {evo['mean_total']:.2f}")
            report.append(f"  Range: [{evo['min_total']}, {evo['max_total']}]")
            report.append(f"  Time range: [{evo['timestamps'][0]:.2f}s, {evo['timestamps'][-1]:.2f}s]")
            report.append("")
        
        if rep_analysis.get("root_cause"):
            report.append("ROOT CAUSE:")
            for cause in rep_analysis["root_cause"]:
                report.append(f"  ⚠️  {cause['issue']}")
                report.append(f"     {cause['description']}")
                report.append(f"     Likely Cause: {cause['likely_cause']}")
                report.append(f"     Impact: {cause['impact']}")
                report.append(f"     Recommendation: {cause['recommendation']}")
                report.append("")
    
    # Queue Timing Analysis
    if "queue" in analyses:
        queue_analysis = analyses["queue"]
        report.append("2. QUEUE TIMING ANALYSIS")
        report.append("-" * 80)
        
        if queue_analysis.get("queue_distributions"):
            dist = queue_analysis["queue_distributions"]
            report.append("Queue Length Distributions:")
            report.append("")
            report.append("Training (Co-Simulation):")
            train = dist["training"]
            report.append(f"  Mean: {train['mean']:.2f}")
            report.append(f"  Median: {train['median']:.2f}")
            report.append(f"  95th percentile: {train['p95']:.2f}")
            report.append(f"  99th percentile: {train['p99']:.2f}")
            report.append(f"  Max: {train['max']}")
            report.append(f"  Non-zero: {train['non_zero_pct']:.2f}%")
            report.append("")
            report.append("Real Simulation:")
            real = dist["real"]
            report.append(f"  Mean: {real['mean']:.2f}")
            report.append(f"  Median: {real['median']:.2f}")
            report.append(f"  95th percentile: {real['p95']:.2f}")
            report.append(f"  99th percentile: {real['p99']:.2f}")
            report.append(f"  Max: {real['max']}")
            report.append(f"  Non-zero: {real['non_zero_pct']:.2f}%")
            report.append("")
        
        # NEW: Queue evolution
        if queue_analysis.get("queue_evolution"):
            evo = queue_analysis["queue_evolution"]
            report.append("Queue Evolution Over Time:")
            report.append(f"  Early mean (first 100 tasks): {evo['early_mean']:.2f}")
            report.append(f"  Late mean (last 100 tasks): {evo['late_mean']:.2f}")
            report.append(f"  Buildup: {evo['late_mean'] - evo['early_mean']:.2f}")
            report.append(f"  Max queue observed: {max(evo['max_queues'])}")
            report.append("")
        
        if queue_analysis.get("root_cause"):
            report.append("ROOT CAUSE:")
            for cause in queue_analysis["root_cause"]:
                report.append(f"  ⚠️  {cause['issue']}")
                report.append(f"     {cause['description']}")
                report.append(f"     Likely Cause: {cause['likely_cause']}")
                report.append(f"     Impact: {cause['impact']}")
                report.append(f"     Recommendation: {cause['recommendation']}")
                report.append("")
    
    # Timing Analysis
    if "timing" in analyses:
        timing_analysis = analyses["timing"]
        report.append("3. SYSTEM STATE CAPTURE TIMING")
        report.append("-" * 80)
        
        if timing_analysis.get("training_capture"):
            train = timing_analysis["training_capture"]
            report.append("Training Capture:")
            report.append(f"  Average timestamp: {train['avg_timestamp']:.2f}s")
            report.append(f"  Range: [{train['min_timestamp']:.2f}s, {train['max_timestamp']:.2f}s]")
            report.append(f"  Average task count: {train['avg_task_count']:.1f}")
            report.append("")
        
        if timing_analysis.get("real_capture"):
            real = timing_analysis["real_capture"]
            report.append("Real Simulation Capture (from per-task systemStateResult):")
            report.append(f"  Average timestamp: {real['avg_timestamp']:.2f}s")
            report.append(f"  Range: [{real['min_timestamp']:.2f}s, {real['max_timestamp']:.2f}s]")
            report.append(f"  Simulation duration: {real['simulation_duration']:.2f}s")
            report.append(f"  Total captures: {real['total_captures']}")
            report.append("")
        
        # NEW: Capture frequency
        if timing_analysis.get("capture_frequency"):
            freq = timing_analysis["capture_frequency"]
            report.append("Capture Frequency (Real Simulation):")
            report.append(f"  Mean interval: {freq['mean_interval']:.4f}s")
            report.append(f"  Median interval: {freq['median_interval']:.4f}s")
            report.append(f"  Range: [{freq['min_interval']:.4f}s, {freq['max_interval']:.4f}s]")
            report.append("")
        
        if timing_analysis.get("timing_differences"):
            report.append("TIMING DIFFERENCES:")
            for diff in timing_analysis["timing_differences"]:
                report.append(f"  ⚠️  {diff['issue']}")
                report.append(f"     {diff['description']}")
                report.append(f"     Impact: {diff['impact']}")
                report.append(f"     Recommendation: {diff['recommendation']}")
                report.append("")
    
    # NEW: Per-Task State Evolution
    if "evolution" in analyses:
        evo_analysis = analyses["evolution"]
        report.append("4. PER-TASK SYSTEM STATE EVOLUTION")
        report.append("-" * 80)
        
        if evo_analysis.get("replica_growth"):
            rg = evo_analysis["replica_growth"]
            report.append("Replica Growth:")
            report.append(f"  Initial: {rg['initial']} replicas")
            report.append(f"  Final: {rg['final']} replicas")
            report.append(f"  Growth: {rg['growth']:+d} replicas")
            report.append(f"  Mean: {rg['mean']:.2f} replicas")
            report.append(f"  Peak: {rg['max']} replicas")
            report.append("")
        
        if evo_analysis.get("queue_buildup"):
            qb = evo_analysis["queue_buildup"]
            report.append("Queue Buildup:")
            report.append(f"  Initial mean: {qb['initial_mean']:.2f}")
            report.append(f"  Final mean: {qb['final_mean']:.2f}")
            report.append(f"  Buildup: {qb['buildup']:+.2f}")
            report.append(f"  Peak queue: {qb['max_queue']}")
            report.append("")
        
        if evo_analysis.get("temporal_changes"):
            tc = evo_analysis["temporal_changes"]
            report.append("Temporal State Changes:")
            report.append(f"  Initial mean comm_remaining: {tc['initial_mean_comm']:.4f}s")
            report.append(f"  Final mean comm_remaining: {tc['final_mean_comm']:.4f}s")
            report.append(f"  Change: {tc['change']:+.4f}s")
            report.append("")
    
    # Summary and Recommendations
    report.append("5. SUMMARY AND RECOMMENDATIONS")
    report.append("-" * 80)
    report.append("")
    report.append("KEY FINDINGS:")
    report.append("")
    
    # Check if proactive replica creation is working
    if "replica" in analyses:
        rep = analyses["replica"]
        if rep.get("real_replicas") and rep.get("training_replicas"):
            real_avg = rep["real_replicas"]["avg_total_replicas"]
            train_avg = rep["training_replicas"]["avg_total_replicas"]
            if abs(real_avg - train_avg) < 10:
                report.append("✓ Replica counts are now better aligned (proactive creation working)")
            else:
                report.append("⚠️  Replica counts still differ significantly")
            report.append("")
    
    report.append("1. REPLICA STATE MISMATCH (CRITICAL)")
    report.append("   - Training data captures state AFTER warmup tasks create replicas")
    report.append("   - Real simulation starts from ZERO (autoscaling from zero)")
    report.append("   - Proactive replica creation in scheduler helps but may need more")
    report.append("")
    report.append("2. QUEUE LENGTH MISMATCH (HIGH)")
    report.append("   - Real simulation accumulates queues over long-running execution")
    report.append("   - Training captures isolated batch states (low queues)")
    report.append("   - Per-task captures show queue buildup over time")
    report.append("")
    report.append("3. TEMPORAL STATE TIMING (MEDIUM)")
    report.append("   - Communication remaining time differs significantly")
    report.append("   - Per-task captures enable detailed temporal analysis")
    report.append("")
    report.append("ACTIONABLE RECOMMENDATIONS:")
    report.append("")
    report.append("1. Fix Replica State Capture:")
    report.append("   - Modify co-simulation to capture state BEFORE first task batch")
    report.append("   - Match real simulation's autoscaling-from-zero behavior")
    report.append("   - Or: Generate training data with zero initial replicas")
    report.append("")
    report.append("2. Improve Queue State Representation:")
    report.append("   - Generate training data with longer-running simulations")
    report.append("   - Capture queue state at multiple simulation timestamps")
    report.append("   - Include high-queue scenarios in training data")
    report.append("")
    report.append("3. Align Temporal State Capture:")
    report.append("   - Ensure temporal state capture timing matches real simulation")
    report.append("   - Verify communication time calculations are consistent")
    report.append("")
    report.append("4. Data Generation Strategy:")
    report.append("   - Generate diverse training scenarios:")
    report.append("     * Early simulation (low queues, few replicas)")
    report.append("     * Mid simulation (moderate queues, some replicas)")
    report.append("     * Late simulation (high queues, many replicas)")
    report.append("   - This will make GNN robust to different system states")
    report.append("")
    
    return "\n".join(report)


def main():
    """Main entry point."""
    print("=" * 80)
    print("ROOT CAUSE ANALYSIS: STATE DISCREPANCIES")
    print("UPDATED: Using per-task system state captures")
    print("=" * 80)
    print()
    
    analyses = {}
    
    print("Analyzing replica state discrepancy...")
    analyses["replica"] = analyze_replica_discrepancy()
    print()
    
    print("Analyzing queue timing...")
    analyses["queue"] = analyze_queue_timing()
    print()
    
    print("Analyzing system state capture timing...")
    analyses["timing"] = analyze_system_state_capture_timing()
    print()
    
    print("Analyzing per-task state evolution...")
    analyses["evolution"] = analyze_per_task_state_evolution()
    print()
    
    print("Generating root cause report...")
    report = generate_root_cause_report(analyses)
    print(report)
    
    # Save to file
    output_file = BASE_DIR / "scripts_cosim" / "ROOT_CAUSE_ANALYSIS.txt"
    with open(output_file, 'w') as f:
        f.write(report)
    
    print()
    print(f"✓ Root cause analysis saved to: {output_file}")


if __name__ == "__main__":
    main()
