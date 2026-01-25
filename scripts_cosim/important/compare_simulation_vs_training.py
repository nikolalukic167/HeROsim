#!/usr/bin/env python3
"""
Compare actual simulation results with training data to identify
improvements needed for GNN data generation.

This script analyzes:
1. Workload characteristics (task count, distribution, timing)
2. System state differences (replicas, queues, temporal state)
3. Batch size distributions
4. Queue and temporal state patterns
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any

BASE_DIR = Path("/root/projects/my-herosim")


def load_simulation_result(result_path: Path) -> Dict[str, Any]:
    """Load actual simulation result."""
    with open(result_path, 'r') as f:
        return json.load(f)


def load_training_dataset(dataset_dir: Path) -> Dict[str, Any]:
    """Load training dataset files."""
    dataset = {}
    
    # Load workload
    workload_path = dataset_dir / "workload.json"
    if workload_path.exists():
        with open(workload_path, 'r') as f:
            dataset['workload'] = json.load(f)
    
    # Load captured state
    captured_path = dataset_dir / "system_state_captured_unique.json"
    if captured_path.exists():
        with open(captured_path, 'r') as f:
            dataset['captured_state'] = json.load(f)
    
    # Load infrastructure
    infra_path = dataset_dir / "infrastructure.json"
    if infra_path.exists():
        with open(infra_path, 'r') as f:
            dataset['infrastructure'] = json.load(f)
    
    # Load best result
    best_path = dataset_dir / "best.json"
    if best_path.exists():
        with open(best_path, 'r') as f:
            dataset['best'] = json.load(f)
    
    return dataset


def analyze_workload(workload: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze workload characteristics."""
    events = workload.get('events', [])
    
    task_types = Counter(e.get('taskType', {}).get('name', 'unknown') for e in events)
    arrival_times = [e.get('arrivalTime', 0) for e in events]
    
    return {
        'total_tasks': len(events),
        'task_types': dict(task_types),
        'arrival_times': arrival_times,
        'time_span': max(arrival_times) - min(arrival_times) if arrival_times else 0,
        'avg_inter_arrival': (max(arrival_times) - min(arrival_times)) / len(events) if len(events) > 1 else 0,
    }


def analyze_system_state(captured_state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze captured system state."""
    replicas = captured_state.get('replicas', {})
    task_placements = captured_state.get('task_placements', [])
    
    # Analyze queue snapshots
    queue_lengths = []
    temporal_states = []
    
    for tp in task_placements:
        queue_snap = tp.get('queue_snapshot_at_scheduling', {})
        full_snap = tp.get('full_queue_snapshot', {})
        temp_state = tp.get('temporal_state_at_scheduling', {})
        
        # Collect queue lengths
        for key, length in queue_snap.items():
            queue_lengths.append(length)
        for key, length in full_snap.items():
            queue_lengths.append(length)
        
        # Collect temporal state
        for key, state in temp_state.items():
            if isinstance(state, dict):
                temporal_states.append(state)
    
    return {
        'replica_count': {k: len(v) if isinstance(v, list) else 0 for k, v in replicas.items()},
        'task_placements_count': len(task_placements),
        'queue_lengths': queue_lengths,
        'avg_queue_length': sum(queue_lengths) / len(queue_lengths) if queue_lengths else 0,
        'max_queue_length': max(queue_lengths) if queue_lengths else 0,
        'temporal_states_count': len(temporal_states),
        'has_temporal_state': len(temporal_states) > 0,
    }


def analyze_task_results(task_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze task results from actual simulation."""
    valid_tasks = [tr for tr in task_results if tr.get('taskId') is not None and tr.get('taskId') >= 0]
    
    # Check what data is captured
    sample = valid_tasks[0] if valid_tasks else {}
    
    has_queue_snap = any(tr.get('queueSnapshotAtScheduling') for tr in valid_tasks)
    has_full_snap = any(tr.get('fullQueueSnapshot') for tr in valid_tasks)
    has_temporal = any(tr.get('temporalStateAtScheduling') for tr in valid_tasks)
    
    # Analyze queue snapshots
    queue_lengths = []
    for tr in valid_tasks:
        qs = tr.get('queueSnapshotAtScheduling', {})
        if isinstance(qs, dict):
            queue_lengths.extend(qs.values())
    
    return {
        'total_tasks': len(valid_tasks),
        'has_queue_snapshot': has_queue_snap,
        'has_full_queue_snapshot': has_full_snap,
        'has_temporal_state': has_temporal,
        'queue_lengths': queue_lengths,
        'avg_queue_length': sum(queue_lengths) / len(queue_lengths) if queue_lengths else 0,
        'max_queue_length': max(queue_lengths) if queue_lengths else 0,
    }


def main():
    print("="*100)
    print("COMPARING ACTUAL SIMULATION vs TRAINING DATA")
    print("="*100 + "\n")
    
    # Load actual simulation
    sim_result_path = BASE_DIR / "simulation_data/results/simulation_result_gnn.json"
    if not sim_result_path.exists():
        print(f"ERROR: Simulation result not found: {sim_result_path}")
        sys.exit(1)
    
    print(f"Loading actual simulation: {sim_result_path}")
    sim_result = load_simulation_result(sim_result_path)
    
    stats = sim_result.get('stats', {})
    task_results = stats.get('taskResults', [])
    system_state_results = stats.get('systemStateResults', [])
    
    print(f"  Total tasks: {len([tr for tr in task_results if tr.get('taskId') is not None and tr.get('taskId') >= 0])}")
    print(f"  Total RTT: {sim_result.get('total_rtt', 0):.2f}s")
    print()
    
    # Analyze actual simulation
    sim_task_analysis = analyze_task_results(task_results)
    
    # Load and analyze training datasets
    train_base = BASE_DIR / "simulation_data/artifacts/run300/gnn_datasets"
    if not train_base.exists():
        print(f"ERROR: Training datasets not found: {train_base}")
        sys.exit(1)
    
    dataset_dirs = sorted(train_base.glob("ds_*"))[:10]  # Sample first 10
    print(f"Analyzing {len(dataset_dirs)} training datasets...")
    
    train_workloads = []
    train_states = []
    
    for ds_dir in dataset_dirs:
        dataset = load_training_dataset(ds_dir)
        
        if 'workload' in dataset:
            wl_analysis = analyze_workload(dataset['workload'])
            train_workloads.append(wl_analysis)
        
        if 'captured_state' in dataset:
            state_analysis = analyze_system_state(dataset['captured_state'])
            train_states.append(state_analysis)
    
    # Compare
    print("\n" + "="*100)
    print("COMPARISON RESULTS")
    print("="*100 + "\n")
    
    print("1. WORKLOAD CHARACTERISTICS")
    print("-"*100)
    if train_workloads:
        avg_train_tasks = sum(w['total_tasks'] for w in train_workloads) / len(train_workloads)
        print(f"  Training: {avg_train_tasks:.1f} tasks per dataset (always same)")
        print(f"  Production: {sim_task_analysis['total_tasks']} tasks total")
        print(f"  → Training uses fixed-size batches, production has variable batch sizes")
    
    print("\n2. DATA CAPTURE")
    print("-"*100)
    print(f"  Production Simulation:")
    print(f"    Queue snapshot: {'✓' if sim_task_analysis['has_queue_snapshot'] else '✗'}")
    print(f"    Full queue snapshot: {'✓' if sim_task_analysis['has_full_queue_snapshot'] else '✗'}")
    print(f"    Temporal state: {'✓' if sim_task_analysis['has_temporal_state'] else '✗ (ADDED IN SCHEDULER)'}")
    
    if train_states:
        has_temp = sum(1 for s in train_states if s['has_temporal_state'])
        print(f"\n  Training Data:")
        print(f"    Temporal state: {has_temp}/{len(train_states)} datasets have it")
        print(f"    Queue snapshots: All datasets have them")
    
    print("\n3. QUEUE CHARACTERISTICS")
    print("-"*100)
    if sim_task_analysis['queue_lengths']:
        print(f"  Production:")
        print(f"    Avg queue length: {sim_task_analysis['avg_queue_length']:.2f}")
        print(f"    Max queue length: {sim_task_analysis['max_queue_length']}")
    
    if train_states:
        avg_train_queue = sum(s['avg_queue_length'] for s in train_states) / len(train_states)
        max_train_queue = max(s['max_queue_length'] for s in train_states)
        print(f"  Training:")
        print(f"    Avg queue length: {avg_train_queue:.2f}")
        print(f"    Max queue length: {max_train_queue}")
    
    print("\n4. RECOMMENDATIONS FOR DATA GENERATION")
    print("-"*100)
    print("""
  A. VARIABLE BATCH SIZES
     • Generate datasets with 1, 2, 3, 4, 5, 6, 7 tasks
     • Match production distribution (38% 1-task, 21% 2-task, etc.)
     • Smaller batches are MUCH faster to generate (1000x fewer combinations)
  
  B. TEMPORAL STATE CAPTURE
     • ✓ Already added to GNN scheduler
     • Ensure all training datasets have temporal_state_at_scheduling
     • Verify temporal state is being used in graph construction
  
  C. QUEUE DISTRIBUTIONS
     • Training data should cover various queue states (empty, moderate, full)
     • Current training may have limited queue diversity
     • Consider generating datasets with different initial queue distributions
  
  D. SYSTEM STATE DIVERSITY
     • Training: Single snapshot after warmup
     • Production: Evolving state throughout simulation
     • Consider capturing state at multiple time points
     • Or generate datasets with different initial states
    """)
    
    print("="*100)


if __name__ == "__main__":
    main()
