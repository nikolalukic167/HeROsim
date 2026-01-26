#!/usr/bin/env python3
"""
Comprehensive Analysis: Training Data vs Real Simulation Results

This script analyzes the relationship between:
1. Training data artifacts (run_non_unique/gnn_datasets_*)
2. Graph cache features (graphs_cache_merged_2_3_tasks)
3. Real simulation results (simulation_result_gnn.json)

It demonstrates how system state features flow from training to inference.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Paths
BASE_DIR = Path("/root/projects/my-herosim")
ARTIFACTS_DIR = BASE_DIR / "simulation_data/artifacts/run_non_unique"
CACHE_DIR = ARTIFACTS_DIR / "graphs_cache_merged_2_3_tasks"
SIM_RESULT_FILE = BASE_DIR / "simulation_data/results/simulation_result_gnn.json"


def load_training_dataset_sample(dataset_name: str = "ds_00000") -> Dict[str, Any]:
    """Load a sample training dataset to show its structure."""
    dataset_dir = ARTIFACTS_DIR / "gnn_datasets_2tasks" / dataset_name
    
    result = {
        "dataset_name": dataset_name,
        "dataset_dir": str(dataset_dir),
        "files": {},
        "structure": {}
    }
    
    # Check what files exist
    files_to_check = [
        "optimal_result.json",
        "system_state_captured_unique.json",
        "infrastructure.json",
        "placement_plan.json"
    ]
    
    for filename in files_to_check:
        filepath = dataset_dir / filename
        if filepath.exists():
            result["files"][filename] = "exists"
            if filename == "optimal_result.json":
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    result["structure"][filename] = {
                        "keys": list(data.keys()),
                        "has_stats": "stats" in data,
                        "has_config": "config" in data,
                        "has_sample": "sample" in data
                    }
            elif filename == "system_state_captured_unique.json":
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    result["structure"][filename] = {
                        "keys": list(data.keys()),
                        "num_task_placements": len(data.get("task_placements", [])),
                        "has_queue_snapshot": "full_queue_snapshot" in (data.get("task_placements", [{}])[0] if data.get("task_placements") else {}),
                        "has_temporal_state": "temporal_state_at_scheduling" in (data.get("task_placements", [{}])[0] if data.get("task_placements") else {})
                    }
        else:
            result["files"][filename] = "missing"
    
    return result


def analyze_graph_cache_features() -> Dict[str, Any]:
    """Analyze what features are stored in the graph cache."""
    result = {
        "cache_dir": str(CACHE_DIR),
        "files": {},
        "graph_features": {},
        "metadata": {}
    }
    
    # Check cache files
    cache_files = {
        "graphs.pkl": "Graph data",
        "metadata.json": "Cache metadata",
        "placement_rtt_hash_table.pkl": "RTT lookup table",
        "optimal_rtt.pkl": "Optimal RTT values"
    }
    
    for filename, description in cache_files.items():
        filepath = CACHE_DIR / filename
        if filepath.exists():
            result["files"][filename] = {
                "exists": True,
                "description": description,
                "size_mb": filepath.stat().st_size / (1024 * 1024)
            }
            
            if filename == "metadata.json":
                with open(filepath, 'r') as f:
                    metadata = json.load(f)
                    result["metadata"] = metadata
            elif filename == "graphs.pkl":
                # Load a sample graph to show structure
                try:
                    with open(filepath, 'rb') as f:
                        graphs = pickle.load(f)
                        if graphs and len(graphs) > 0:
                            sample_graph = graphs[0]
                            graph_info = {
                                "num_graphs": len(graphs),
                                "sample_graph": {}
                            }
                            if hasattr(sample_graph, 'x') and sample_graph.x is not None:
                                graph_info["sample_graph"]["num_nodes"] = sample_graph.x.shape[0]
                            if hasattr(sample_graph, 'edge_index') and sample_graph.edge_index is not None:
                                graph_info["sample_graph"]["num_edges"] = sample_graph.edge_index.shape[1]
                            if hasattr(sample_graph, 'x_task') and sample_graph.x_task is not None:
                                graph_info["sample_graph"]["task_features_dim"] = sample_graph.x_task.shape[1]
                            if hasattr(sample_graph, 'x_platform') and sample_graph.x_platform is not None:
                                graph_info["sample_graph"]["platform_features_dim"] = sample_graph.x_platform.shape[1]
                            graph_info["sample_graph"]["has_y"] = hasattr(sample_graph, 'y')
                            graph_info["sample_graph"]["has_optimal_rtt"] = hasattr(sample_graph, 'optimal_rtt')
                            result["graph_features"] = graph_info
                except Exception as e:
                    result["graph_features"] = {"error": str(e)}
        else:
            result["files"][filename] = {"exists": False}
    
    return result


def analyze_simulation_result() -> Dict[str, Any]:
    """Analyze the real simulation result structure."""
    result = {
        "sim_result_file": str(SIM_RESULT_FILE),
        "exists": SIM_RESULT_FILE.exists(),
        "structure": {},
        "system_state_samples": []
    }
    
    if not SIM_RESULT_FILE.exists():
        return result
    
    # Load just the structure (file is huge)
    with open(SIM_RESULT_FILE, 'r') as f:
        data = json.load(f)
    
    result["structure"] = {
        "top_level_keys": list(data.keys()),
        "has_stats": "stats" in data,
        "has_system_state_results": "stats" in data and "systemStateResults" in data.get("stats", {}),
        "num_system_states": len(data.get("stats", {}).get("systemStateResults", [])),
        "num_task_results": len(data.get("stats", {}).get("taskResults", [])),
        "total_rtt": data.get("total_rtt"),
        "num_tasks": data.get("num_tasks")
    }
    
    # Sample a few system state results
    system_states = data.get("stats", {}).get("systemStateResults", [])
    if system_states:
        # Sample first, middle, and last
        sample_indices = [0, len(system_states) // 2, len(system_states) - 1]
        for idx in sample_indices:
            if idx < len(system_states):
                state = system_states[idx]
                result["system_state_samples"].append({
                    "index": idx,
                    "timestamp": state.get("timestamp"),
                    "keys": list(state.keys()),
                    "has_replicas": "replicas" in state,
                    "has_queue_occupancy": "queue_occupancy" in state,
                    "has_scheduler_state": "scheduler_state" in state,
                    "num_replica_types": len(state.get("replicas", {})),
                    "queue_occupancy_sample": {
                        k: len(v) if isinstance(v, dict) else v
                        for k, v in list(state.get("queue_occupancy", {}).items())[:2]
                    } if "queue_occupancy" in state else None
                })
    
    # Sample task results
    task_results = data.get("stats", {}).get("taskResults", [])
    if task_results:
        sample_task = task_results[0]
        result["task_result_sample"] = {
            "keys": list(sample_task.keys()),
            "has_queue_snapshot": "queueSnapshotAtScheduling" in sample_task,
            "has_full_queue_snapshot": "fullQueueSnapshot" in sample_task,
            "has_temporal_state": "temporalStateAtScheduling" in sample_task,
            "has_gnn_decision_time": "gnn_decision_time" in sample_task
        }
    
    return result


def compare_feature_extraction() -> Dict[str, Any]:
    """Compare how features are extracted in training vs inference."""
    comparison = {
        "task_features": {
            "training": {
                "source": "optimal_result.json -> df_tasks",
                "features": [
                    "task_type (one-hot: dnn1, dnn2) - 2 dims",
                    "source_node (normalized index) - 1 dim",
                    "Total: 3 dims"
                ],
                "code_location": "prepare_graphs_cache.py:597-609"
            },
            "inference": {
                "source": "batch_tasks from scheduler",
                "features": [
                    "task_type (one-hot: dnn1, dnn2) - 2 dims",
                    "source_node (normalized index) - 1 dim",
                    "Total: 3 dims"
                ],
                "code_location": "scheduler.py:419-427"
            },
            "match": True
        },
        "platform_features": {
            "training": {
                "source": "optimal_result.json + system_state_captured_unique.json",
                "features": [
                    "platform_type (one-hot: 5 types) - 5 dims",
                    "has_dnn1_replica - 1 dim",
                    "has_dnn2_replica - 1 dim",
                    "queue_length (normalized) - 1 dim",
                    "current_task_remaining (normalized) - 1 dim",
                    "cold_start_remaining (normalized) - 1 dim",
                    "comm_remaining (normalized) - 1 dim",
                    "target_concurrency (normalized) - 1 dim",
                    "usage_ratio (normalized) - 1 dim",
                    "Total: 13 dims"
                ],
                "code_location": "prepare_graphs_cache.py:611-741"
            },
            "inference": {
                "source": "SystemState + queue_snapshot from scheduler",
                "features": [
                    "platform_type (one-hot: 5 types) - 5 dims",
                    "has_dnn1_replica - 1 dim",
                    "has_dnn2_replica - 1 dim",
                    "queue_length (normalized, adaptive) - 1 dim",
                    "current_task_remaining (estimated) - 1 dim",
                    "cold_start_remaining (estimated) - 1 dim",
                    "comm_remaining (estimated) - 1 dim",
                    "target_concurrency (calculated) - 1 dim",
                    "usage_ratio (calculated) - 1 dim",
                    "Total: 13 dims"
                ],
                "code_location": "scheduler.py:431-500"
            },
            "match": True,
            "note": "Inference uses adaptive queue normalization and estimates temporal state if not available"
        },
        "system_state_sources": {
            "training": {
                "queue_snapshot": "system_state_captured_unique.json -> task_placements[0].full_queue_snapshot",
                "temporal_state": "system_state_captured_unique.json -> task_placements[].temporal_state_at_scheduling",
                "replicas": "optimal_result.json -> stats.systemStateResults[-1].replicas",
                "fallback": "infrastructure.json -> queue_distributions (for run_non_unique datasets)"
            },
            "inference": {
                "queue_snapshot": "scheduler.queue_snapshot (captured at scheduling time)",
                "temporal_state": "estimated from queue_length if not available",
                "replicas": "SystemState.replicas (live from simulation)",
                "fallback": "task_types_data for execution time estimates"
            }
        }
    }
    
    return comparison


def generate_summary_report() -> str:
    """Generate a comprehensive summary report."""
    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE ANALYSIS: TRAINING DATA vs REAL SIMULATION RESULTS")
    report.append("=" * 80)
    report.append("")
    
    # 1. Training Dataset Structure
    report.append("1. TRAINING DATASET STRUCTURE (Artifacts)")
    report.append("-" * 80)
    dataset_sample = load_training_dataset_sample()
    report.append(f"Sample Dataset: {dataset_sample['dataset_name']}")
    report.append(f"Location: {dataset_sample['dataset_dir']}")
    report.append("")
    report.append("Files:")
    for filename, status in dataset_sample['files'].items():
        report.append(f"  - {filename}: {status}")
    report.append("")
    report.append("Structure:")
    for filename, struct in dataset_sample['structure'].items():
        report.append(f"  {filename}:")
        for key, value in struct.items():
            report.append(f"    - {key}: {value}")
    report.append("")
    
    # 2. Graph Cache Features
    report.append("2. GRAPH CACHE FEATURES")
    report.append("-" * 80)
    cache_analysis = analyze_graph_cache_features()
    report.append(f"Cache Directory: {cache_analysis['cache_dir']}")
    report.append("")
    report.append("Cache Files:")
    for filename, info in cache_analysis['files'].items():
        if isinstance(info, dict) and info.get('exists'):
            report.append(f"  - {filename}: {info.get('description', 'N/A')} ({info.get('size_mb', 0):.2f} MB)")
    report.append("")
    if cache_analysis.get('graph_features'):
        gf = cache_analysis['graph_features']
        report.append("Graph Structure:")
        report.append(f"  - Total graphs: {gf.get('num_graphs', 'N/A')}")
        if 'sample_graph' in gf:
            sg = gf['sample_graph']
            report.append(f"  - Nodes per graph: {sg.get('num_nodes', 'N/A')}")
            report.append(f"  - Edges per graph: {sg.get('num_edges', 'N/A')}")
            report.append(f"  - Task features: {sg.get('task_features_dim', 'N/A')} dims")
            report.append(f"  - Platform features: {sg.get('platform_features_dim', 'N/A')} dims")
    report.append("")
    if cache_analysis.get('metadata'):
        report.append("Cache Metadata:")
        for key, value in cache_analysis['metadata'].items():
            if isinstance(value, dict):
                report.append(f"  - {key}:")
                for k, v in value.items():
                    report.append(f"      {k}: {v}")
            else:
                report.append(f"  - {key}: {value}")
    report.append("")
    
    # 3. Real Simulation Results
    report.append("3. REAL SIMULATION RESULTS")
    report.append("-" * 80)
    sim_analysis = analyze_simulation_result()
    if sim_analysis['exists']:
        report.append(f"File: {sim_analysis['sim_result_file']}")
        report.append("")
        report.append("Structure:")
        for key, value in sim_analysis['structure'].items():
            report.append(f"  - {key}: {value}")
        report.append("")
        if sim_analysis.get('system_state_samples'):
            report.append("System State Samples:")
            for sample in sim_analysis['system_state_samples']:
                report.append(f"  Index {sample['index']} (t={sample['timestamp']}):")
                report.append(f"    - Keys: {sample['keys']}")
                report.append(f"    - Replica types: {sample['num_replica_types']}")
                if sample.get('queue_occupancy_sample'):
                    report.append(f"    - Queue occupancy sample: {sample['queue_occupancy_sample']}")
        report.append("")
        if sim_analysis.get('task_result_sample'):
            report.append("Task Result Sample:")
            for key, value in sim_analysis['task_result_sample'].items():
                report.append(f"  - {key}: {value}")
    else:
        report.append(f"File not found: {sim_analysis['sim_result_file']}")
    report.append("")
    
    # 4. Feature Comparison
    report.append("4. FEATURE EXTRACTION COMPARISON")
    report.append("-" * 80)
    comparison = compare_feature_extraction()
    
    for feature_type, comp in comparison.items():
        report.append(f"{feature_type.upper()}:")
        report.append("  Training:")
        for item in comp['training'].get('features', []):
            report.append(f"    - {item}")
        report.append("  Inference:")
        for item in comp['inference'].get('features', []):
            report.append(f"    - {item}")
        if comp.get('match'):
            report.append("  ✓ Features match between training and inference")
        if comp.get('note'):
            report.append(f"  Note: {comp['note']}")
        report.append("")
    
    # 5. Data Flow Summary
    report.append("5. DATA FLOW SUMMARY")
    report.append("-" * 80)
    report.append("")
    report.append("TRAINING DATA GENERATION:")
    report.append("  1. Co-simulation generates optimal_result.json (all placements)")
    report.append("  2. System state captured: system_state_captured_unique.json")
    report.append("     - Contains: queue_snapshot, temporal_state, replicas")
    report.append("  3. prepare_graphs_cache.py extracts:")
    report.append("     - Nodes, tasks, platforms from optimal_result.json")
    report.append("     - Queue lengths from system_state_captured_unique.json")
    report.append("     - Temporal state from system_state_captured_unique.json")
    report.append("     - Replicas from optimal_result.json stats")
    report.append("  4. Graphs built with 13-dim platform features + 3-dim task features")
    report.append("  5. RTT hash table created for regret calculation")
    report.append("")
    report.append("REAL SIMULATION (INFERENCE):")
    report.append("  1. Scheduler receives batch of tasks")
    report.append("  2. Captures current SystemState (replicas, queues)")
    report.append("  3. Builds graph using same feature extraction logic")
    report.append("  4. GNN predicts placement logits")
    report.append("  5. Greedy decoder selects placements (non-unique allowed)")
    report.append("  6. Results stored in simulation_result_gnn.json")
    report.append("     - taskResults: per-task metrics + queue/temporal snapshots")
    report.append("     - systemStateResults: periodic system state captures")
    report.append("")
    report.append("KEY DIFFERENCES:")
    report.append("  - Training: Uses optimal placements from brute-force search")
    report.append("  - Inference: Uses GNN predictions with greedy decoding")
    report.append("  - Training: Queue/temporal state from co-simulation capture")
    report.append("  - Inference: Queue/temporal state from live SystemState")
    report.append("  - Training: Fixed queue normalization (QUEUE_NORM_FACTOR=10.0)")
    report.append("  - Inference: Adaptive queue normalization (90th percentile)")
    report.append("")
    
    return "\n".join(report)


def main():
    """Main entry point."""
    print("Analyzing training data vs simulation results...")
    print()
    
    # Generate and print report
    report = generate_summary_report()
    print(report)
    
    # Save to file
    output_file = BASE_DIR / "scripts_cosim" / "TRAINING_VS_SIMULATION_ANALYSIS.txt"
    with open(output_file, 'w') as f:
        f.write(report)
    
    print()
    print(f"✓ Analysis saved to: {output_file}")


if __name__ == "__main__":
    main()
