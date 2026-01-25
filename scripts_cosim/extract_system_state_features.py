#!/usr/bin/env python3
"""
Extract and Visualize System State Features

This script extracts system state features from:
1. Training datasets (artifacts)
2. Real simulation results

And shows how they're used in GNN training and inference.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

BASE_DIR = Path("/root/projects/my-herosim")
ARTIFACTS_DIR = BASE_DIR / "simulation_data/artifacts/run_non_unique"
SIM_RESULT_FILE = BASE_DIR / "simulation_data/results/simulation_result_gnn.json"


def extract_training_features(dataset_name: str = "ds_00000") -> Dict[str, Any]:
    """Extract features from a training dataset."""
    dataset_dir = ARTIFACTS_DIR / "gnn_datasets_2tasks" / dataset_name
    
    result = {
        "dataset": dataset_name,
        "queue_snapshot": {},
        "temporal_state": {},
        "replicas": {},
        "platform_features_sample": []
    }
    
    # Load system state
    ssc_path = dataset_dir / "system_state_captured_unique.json"
    if ssc_path.exists():
        with open(ssc_path, 'r') as f:
            data = json.load(f)
        
        task_placements = data.get('task_placements', [])
        if task_placements:
            # Get queue snapshot
            full_queue = task_placements[0].get('full_queue_snapshot', {})
            result['queue_snapshot'] = full_queue
            
            # Get temporal state (merge across tasks)
            merged_temporal = {}
            for tp in task_placements:
                temp_state = tp.get('temporal_state_at_scheduling', {})
                if isinstance(temp_state, dict):
                    for key, state_dict in temp_state.items():
                        if isinstance(state_dict, dict):
                            merged_temporal[key] = {
                                'current_task_remaining': state_dict.get('current_task_remaining', 0.0),
                                'cold_start_remaining': state_dict.get('cold_start_remaining', 0.0),
                                'comm_remaining': state_dict.get('comm_remaining', 0.0)
                            }
            result['temporal_state'] = merged_temporal
    
    # Load optimal result for replicas
    opt_path = dataset_dir / "optimal_result.json"
    if opt_path.exists():
        with open(opt_path, 'r') as f:
            opt_data = json.load(f)
        
        system_states = opt_data.get('stats', {}).get('systemStateResults', [])
        if system_states:
            final_state = system_states[-1]
            result['replicas'] = final_state.get('replicas', {})
    
    # Sample platform features
    if result['queue_snapshot']:
        sample_keys = list(result['queue_snapshot'].keys())[:5]
        for key in sample_keys:
            queue_len = result['queue_snapshot'].get(key, 0)
            temp = result['temporal_state'].get(key, {})
            result['platform_features_sample'].append({
                "platform": key,
                "queue_length": queue_len,
                "current_task_remaining": temp.get('current_task_remaining', 0.0),
                "cold_start_remaining": temp.get('cold_start_remaining', 0.0),
                "comm_remaining": temp.get('comm_remaining', 0.0)
            })
    
    return result


def extract_simulation_features(sample_size: int = 10) -> Dict[str, Any]:
    """Extract features from real simulation results."""
    result = {
        "file": str(SIM_RESULT_FILE),
        "exists": SIM_RESULT_FILE.exists(),
        "samples": []
    }
    
    if not SIM_RESULT_FILE.exists():
        return result
    
    with open(SIM_RESULT_FILE, 'r') as f:
        data = json.load(f)
    
    task_results = data.get('stats', {}).get('taskResults', [])
    system_states = data.get('stats', {}).get('systemStateResults', [])
    
    # Sample task results
    sample_indices = np.linspace(0, len(task_results) - 1, min(sample_size, len(task_results)), dtype=int)
    
    for idx in sample_indices:
        tr = task_results[idx]
        sample = {
            "task_id": tr.get('taskId'),
            "task_type": tr.get('taskType', {}).get('name'),
            "source_node": tr.get('sourceNode'),
            "execution_node": tr.get('executionNode'),
            "execution_platform": tr.get('executionPlatform'),
            "elapsed_time": tr.get('elapsedTime'),
            "queue_time": tr.get('queueTime'),
            "queue_snapshot": tr.get('queueSnapshotAtScheduling', {}),
            "full_queue_snapshot": tr.get('fullQueueSnapshot', {}),
            "temporal_state": tr.get('temporalStateAtScheduling', {}),
            "gnn_decision_time": tr.get('gnn_decision_time', 0.0)
        }
        result['samples'].append(sample)
    
    # Sample system states
    if system_states:
        state_sample_indices = [0, len(system_states) // 2, len(system_states) - 1]
        result['system_state_samples'] = []
        for idx in state_sample_indices:
            if idx < len(system_states):
                state = system_states[idx]
                result['system_state_samples'].append({
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
    
    return result


def compare_features() -> str:
    """Compare features between training and simulation."""
    report = []
    report.append("=" * 80)
    report.append("SYSTEM STATE FEATURES COMPARISON")
    report.append("=" * 80)
    report.append("")
    
    # Training features
    report.append("TRAINING DATASET FEATURES (ds_00000)")
    report.append("-" * 80)
    train_features = extract_training_features()
    report.append(f"Dataset: {train_features['dataset']}")
    report.append(f"Queue snapshot entries: {len(train_features['queue_snapshot'])}")
    report.append(f"Temporal state entries: {len(train_features['temporal_state'])}")
    report.append(f"Replica types: {list(train_features['replicas'].keys())}")
    report.append("")
    report.append("Sample Platform Features:")
    for pf in train_features['platform_features_sample'][:3]:
        report.append(f"  Platform: {pf['platform']}")
        report.append(f"    Queue length: {pf['queue_length']}")
        report.append(f"    Current task remaining: {pf['current_task_remaining']:.4f}s")
        report.append(f"    Cold start remaining: {pf['cold_start_remaining']:.4f}s")
        report.append(f"    Comm remaining: {pf['comm_remaining']:.4f}s")
    report.append("")
    
    # Simulation features
    report.append("REAL SIMULATION FEATURES")
    report.append("-" * 80)
    sim_features = extract_simulation_features()
    if sim_features['exists']:
        report.append(f"File: {sim_features['file']}")
        report.append(f"Task samples: {len(sim_features['samples'])}")
        report.append("")
        report.append("Sample Task Results:")
        for i, sample in enumerate(sim_features['samples'][:3]):
            report.append(f"  Task {i+1} (ID: {sample['task_id']}):")
            report.append(f"    Type: {sample['task_type']}")
            report.append(f"    Source: {sample['source_node']} -> Exec: {sample['execution_node']}:{sample['execution_platform']}")
            report.append(f"    Elapsed: {sample['elapsed_time']:.4f}s (Queue: {sample['queue_time']:.4f}s)")
            report.append(f"    GNN decision time: {sample['gnn_decision_time']:.6f}s")
            report.append(f"    Queue snapshot size: {len(sample.get('queue_snapshot', {}))}")
            report.append(f"    Full queue snapshot size: {len(sample.get('full_queue_snapshot', {}))}")
            report.append(f"    Temporal state size: {len(sample.get('temporal_state', {}))}")
        report.append("")
        
        if sim_features.get('system_state_samples'):
            report.append("System State Samples:")
            for ss in sim_features['system_state_samples']:
                report.append(f"  Timestamp: {ss['timestamp']:.2f}s")
                report.append(f"    Replicas: {ss['replicas']}")
                if ss.get('queue_occupancy_sample'):
                    report.append(f"    Queue occupancy sample: {ss['queue_occupancy_sample']}")
    else:
        report.append(f"File not found: {sim_features['file']}")
    report.append("")
    
    # Feature mapping
    report.append("FEATURE MAPPING: TRAINING -> INFERENCE")
    report.append("-" * 80)
    report.append("")
    report.append("Training Data Source          ->  Inference Source")
    report.append("-" * 80)
    report.append("system_state_captured_unique.json")
    report.append("  .task_placements[0]")
    report.append("    .full_queue_snapshot      ->  queue_snapshot (from scheduler)")
    report.append("    .temporal_state_at_scheduling -> temporal_state (estimated if missing)")
    report.append("")
    report.append("optimal_result.json")
    report.append("  .stats.systemStateResults[-1]")
    report.append("    .replicas                 ->  SystemState.replicas (live)")
    report.append("")
    report.append("Platform Features (13 dims):")
    report.append("  1. Platform type one-hot (5) -> Same")
    report.append("  2. Has dnn1 replica (1)      -> Same")
    report.append("  3. Has dnn2 replica (1)       -> Same")
    report.append("  4. Queue length (1)           -> Adaptive normalization")
    report.append("  5. Current task remaining (1) -> Estimated from queue")
    report.append("  6. Cold start remaining (1)    -> Estimated")
    report.append("  7. Comm remaining (1)         -> Estimated")
    report.append("  8. Target concurrency (1)     -> Calculated")
    report.append("  9. Usage ratio (1)             -> Calculated")
    report.append("")
    
    return "\n".join(report)


def main():
    """Main entry point."""
    print("Extracting system state features...")
    print()
    
    report = compare_features()
    print(report)
    
    # Save to file
    output_file = BASE_DIR / "scripts_cosim" / "SYSTEM_STATE_FEATURES.txt"
    with open(output_file, 'w') as f:
        f.write(report)
    
    print()
    print(f"✓ Features analysis saved to: {output_file}")


if __name__ == "__main__":
    main()
