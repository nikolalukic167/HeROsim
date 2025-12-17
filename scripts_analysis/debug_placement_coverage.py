#!/usr/bin/env python3
"""
Debug script to analyze placement coverage discrepancies.

This script compares:
1. Number of feasible platform positions per task (from graph construction)
2. Total possible placement combinations (product of feasible platforms)
3. Number of actual placement CSV files (from co-simulation)
4. Shows where combinations are being limited
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pandas as pd
import numpy as np

# Configuration
BASE_DIR = Path("/root/projects/my-herosim/simulation_data/artifacts/run10_all/gnn_datasets")
CACHE_DIR = BASE_DIR.parent / "graphs_cache"

# Load task-platform compatibility (same as in prepare_graphs_cache.py)
TASK_PLATFORM_COMPATIBILITY = {
    'dnn1': ['rpiCpu', 'xavierGpu', 'xavierCpu'],
    'dnn2': ['rpiCpu', 'xavierGpu', 'xavierCpu']
}


def count_feasible_platforms_for_task(
    task_type: str,
    source_node_name: str,
    df_platforms: pd.DataFrame,
    df_nodes: pd.DataFrame,
    network_map_by_node: Dict[str, Dict],
    check_replicas: bool = False
) -> int:
    """
    Count feasible platform positions for a single task.
    
    Args:
        check_replicas: If True, only count platforms with replicas (like executecosimulation.py)
                       If False, count all compatible platforms (like graph construction)
    """
    # Get platforms compatible with this task type
    allowed_types = TASK_PLATFORM_COMPATIBILITY.get(task_type, [])
    compatible_platforms = df_platforms[df_platforms['platform_type'].isin(allowed_types)]
    
    if len(compatible_platforms) == 0:
        return 0
    
    # Filter by replicas if requested (like executecosimulation.py does)
    if check_replicas:
        if task_type == 'dnn1':
            compatible_platforms = compatible_platforms[compatible_platforms['has_dnn1_replica'] == True]
        elif task_type == 'dnn2':
            compatible_platforms = compatible_platforms[compatible_platforms['has_dnn2_replica'] == True]
        else:
            return 0
    
    # Get network-feasible nodes
    network_map = network_map_by_node.get(source_node_name, {})
    feasible_nodes = [source_node_name] + list(network_map.keys())
    
    # Count platforms on feasible nodes
    feasible_platforms = compatible_platforms[
        compatible_platforms['node_name'].isin(feasible_nodes)
    ]
    
    return len(feasible_platforms)


def analyze_dataset(dataset_id: str, dataset_dict: Dict) -> Dict:
    """Analyze a single dataset for placement coverage."""
    df_nodes = dataset_dict['nodes']
    df_tasks = dataset_dict['tasks']
    df_platforms = dataset_dict['platforms']
    
    # Build network map (same as graph construction)
    network_map_by_node = {
        row.node_name: row.network_map 
        for row in df_nodes.itertuples(index=False)
    }
    
    # Count feasible platforms per task (TWO ways: with and without replica filtering)
    task_feasible_counts_all = []  # All compatible platforms (like graph construction)
    task_feasible_counts_replicas = []  # Only platforms with replicas (like executecosimulation.py)
    
    for _, task_row in df_tasks.iterrows():
        task_type = task_row['task_type']
        source_node = task_row['source_node']
        
        # Count all compatible platforms (graph construction logic)
        feasible_count_all = count_feasible_platforms_for_task(
            task_type, source_node, df_platforms, df_nodes, network_map_by_node, check_replicas=False
        )
        task_feasible_counts_all.append(feasible_count_all)
        
        # Count only platforms with replicas (executecosimulation.py logic)
        feasible_count_replicas = count_feasible_platforms_for_task(
            task_type, source_node, df_platforms, df_nodes, network_map_by_node, check_replicas=True
        )
        task_feasible_counts_replicas.append(feasible_count_replicas)
    
    # Calculate total possible combinations (TWO ways)
    total_possible_all = 1  # All compatible platforms
    for count in task_feasible_counts_all:
        if count > 0:
            total_possible_all *= count
        else:
            total_possible_all = 0
            break
    
    total_possible_replicas = 1  # Only platforms with replicas
    for count in task_feasible_counts_replicas:
        if count > 0:
            total_possible_replicas *= count
        else:
            total_possible_replicas = 0
            break
    
    # Count actual CSV files
    dataset_dir = dataset_dict.get('dataset_dir', BASE_DIR / dataset_id)
    placements_csv_dir = dataset_dir / "placements_csv"
    num_csv_files = 0
    if placements_csv_dir.exists():
        num_csv_files = len(list(placements_csv_dir.glob("placement_summary_*.csv")))
    
    # Count JSON files (if they exist)
    placements_dir = dataset_dir / "placements"
    num_json_files = 0
    if placements_dir.exists():
        num_json_files = len(list(placements_dir.glob("placement_summary_*.json")))
    
    return {
        'dataset_id': dataset_id,
        'num_tasks': len(df_tasks),
        'task_feasible_counts_all': task_feasible_counts_all,  # All compatible platforms
        'task_feasible_counts_replicas': task_feasible_counts_replicas,  # Only with replicas
        'total_possible_all': total_possible_all,  # All compatible platforms
        'total_possible_replicas': total_possible_replicas,  # Only with replicas
        'num_csv_files': num_csv_files,
        'num_json_files': num_json_files,
        'coverage_percent_all': (num_csv_files / total_possible_all * 100) if total_possible_all > 0 else 0.0,
        'coverage_percent_replicas': (num_csv_files / total_possible_replicas * 100) if total_possible_replicas > 0 else 0.0,
        'missing_combinations_all': max(0, total_possible_all - num_csv_files),
        'missing_combinations_replicas': max(0, total_possible_replicas - num_csv_files),
        'replica_limitation': total_possible_all - total_possible_replicas  # How many combinations are lost due to replica filtering
    }


def load_all_datasets(base_dir: Path) -> Dict[str, Dict]:
    """Load all datasets (same as prepare_graphs_cache.py)."""
    from prepare_graphs_cache import extract_dataset_to_dataframes, load_all_datasets as _load_all
    
    # Import the function from prepare_graphs_cache
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        return _load_all(base_dir)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return {}

def check_early_stopping_config():
    """Check if early stopping is configured in executecosimulation.py."""
    executecosim_path = Path("/root/projects/my-herosim/src/executecosimulation.py")
    
    if not executecosim_path.exists():
        return None, None
    
    with open(executecosim_path, 'r') as f:
        content = f.read()
    
    # Find early stopping config
    import re
    patience_match = re.search(r'early_stop_patience\s*=\s*int\(bf_cfg\.get\([\'"]early_stop_patience[\'"],\s*(\d+)\)\)', content)
    patience = int(patience_match.group(1)) if patience_match else None
        
    return patience


def main():
    print("="*80)
    print("PLACEMENT COVERAGE DEBUGGER")
    print("="*80)
    print()
    
    # Check early stopping
    patience = check_early_stopping_config()
    if patience:
        print(f"⚠️  Early stopping configured: patience = {patience}")
    
    print()
    
    # Load all datasets
    print("Loading datasets...")
    all_datasets = load_all_datasets(BASE_DIR)
    
    if len(all_datasets) == 0:
        print("ERROR: No datasets loaded!")
        return
    
    print(f"Loaded {len(all_datasets)} datasets")
    print()
    
    # Analyze each dataset
    print("Analyzing datasets...")
    results = []
    
    for dataset_id, dataset_dict in list(all_datasets.items())[:20]:  # Analyze first 20 for speed
        try:
            result = analyze_dataset(dataset_id, dataset_dict)
            results.append(result)
        except Exception as e:
            print(f"  Error analyzing {dataset_id}: {e}")
            continue
    
    if not results:
        print("No results to analyze!")
        return
    
    # Print summary statistics
    print()
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    total_possible_all_sum = sum(r['total_possible_all'] for r in results)
    total_possible_replicas_sum = sum(r['total_possible_replicas'] for r in results)
    total_csv_all = sum(r['num_csv_files'] for r in results)
    total_json_all = sum(r['num_json_files'] for r in results)
    total_replica_limitation = sum(r['replica_limitation'] for r in results)
    
    print(f"Total datasets analyzed: {len(results)}")
    print()
    print("📊 ALL COMPATIBLE PLATFORMS (graph construction logic):")
    print(f"  Total possible combinations: {total_possible_all_sum:,}")
    print(f"  Total CSV files found: {total_csv_all:,}")
    print(f"  Overall coverage: {total_csv_all / total_possible_all_sum * 100:.2f}%" if total_possible_all_sum > 0 else "N/A")
    print(f"  Missing combinations: {total_possible_all_sum - total_csv_all:,}")
    print()
    print("📊 PLATFORMS WITH REPLICAS (executecosimulation.py logic):")
    print(f"  Total possible combinations: {total_possible_replicas_sum:,}")
    print(f"  Total CSV files found: {total_csv_all:,}")
    print(f"  Overall coverage: {total_csv_all / total_possible_replicas_sum * 100:.2f}%" if total_possible_replicas_sum > 0 else "N/A")
    print(f"  Missing combinations: {total_possible_replicas_sum - total_csv_all:,}")
    print()
    print(f"⚠️  REPLICA LIMITATION:")
    print(f"  Combinations lost due to replica filtering: {total_replica_limitation:,}")
    print(f"  This is the discrepancy between graph construction and simulation!")
    print()
    
    # Show per-dataset breakdown
    print("="*80)
    print("PER-DATASET BREAKDOWN (first 10)")
    print("="*80)
    
    for i, result in enumerate(results[:10]):
        print(f"\n[{i+1}] {result['dataset_id']}")
        print(f"    Tasks: {result['num_tasks']}")
        print(f"    Feasible platforms per task (ALL): {result['task_feasible_counts_all']}")
        print(f"    Feasible platforms per task (REPLICAS): {result['task_feasible_counts_replicas']}")
        print(f"    Total possible (ALL): {result['total_possible_all']:,}")
        print(f"    Total possible (REPLICAS): {result['total_possible_replicas']:,}")
        print(f"    CSV files found: {result['num_csv_files']}")
        print(f"    JSON files found: {result['num_json_files']}")
        print(f"    Coverage (ALL): {result['coverage_percent_all']:.2f}%")
        print(f"    Coverage (REPLICAS): {result['coverage_percent_replicas']:.2f}%")
        print(f"    Missing (ALL): {result['missing_combinations_all']:,}")
        print(f"    Missing (REPLICAS): {result['missing_combinations_replicas']:,}")
        print(f"    Lost to replica filtering: {result['replica_limitation']:,}")
        
        if result['replica_limitation'] > 0:
            print(f"    ⚠️  REPLICA DISCREPANCY: Graph considers {result['total_possible_all']:,} but simulation only {result['total_possible_replicas']:,}")
    
    # Find datasets with biggest discrepancies
    print()
    print("="*80)
    print("TOP 5 DATASETS WITH BIGGEST DISCREPANCIES")
    print("="*80)
    
    sorted_results = sorted(
        results, 
        key=lambda x: x['missing_combinations_replicas'], 
        reverse=True
    )[:5]
    
    for i, result in enumerate(sorted_results):
        print(f"\n[{i+1}] {result['dataset_id']}")
        print(f"    Possible (ALL): {result['total_possible_all']:,}")
        print(f"    Possible (REPLICAS): {result['total_possible_replicas']:,}")
        print(f"    CSV files: {result['num_csv_files']}")
        print(f"    Missing (ALL): {result['missing_combinations_all']:,} ({100 - result['coverage_percent_all']:.2f}%)")
        print(f"    Missing (REPLICAS): {result['missing_combinations_replicas']:,} ({100 - result['coverage_percent_replicas']:.2f}%)")
        print(f"    Lost to replicas: {result['replica_limitation']:,}")
    
    # Check for datasets with no CSV files
    no_files = sum(1 for r in results if r['num_csv_files'] == 0)
    if no_files > 0:
        print(f"\n⚠️  WARNING: {no_files} datasets have NO CSV files!")
        print(f"   Possible causes:")
        print(f"   - Timeout (3600s per dataset)")
        print(f"   - Simulation errors")
        print(f"   - JSON to CSV conversion not run")
    
    # Check replica limitation
    print()
    print("="*80)
    print("REPLICA LIMITATION ANALYSIS")
    print("="*80)
    
    datasets_with_replica_limitation = [r for r in results if r['replica_limitation'] > 0]
    if datasets_with_replica_limitation:
        total_lost = sum(r['replica_limitation'] for r in datasets_with_replica_limitation)
        print(f"⚠️  {len(datasets_with_replica_limitation)} datasets are limited by replica configuration!")
        print(f"   Total combinations lost: {total_lost:,}")
        print(f"   This means graph construction considers MORE platforms than simulation!")
        print(f"   Graph construction: All compatible platforms")
        print(f"   Simulation: Only platforms with replicas created")
        print(f"   This is why predicted placements don't match the hash table!")
    else:
        print("✅ No replica limitation found - graph and simulation use same platform filtering")
    
    print()
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

