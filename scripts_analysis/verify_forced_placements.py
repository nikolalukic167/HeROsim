#!/usr/bin/env python3
"""
Verification script to check if forced placements in placement_plan align with 
executionNode and executionPlatform in taskResults for tasks 0, 1, 2, 3, 4.

For each dataset in simulation_data/gnn_datasets:
1. Load optimal_result.json
2. Extract placement_plan from sample.placement_plan (lines 452-473)
3. Extract taskResults from stats.taskResults
4. Map node_id to node_name using infrastructure.nodes
5. Compare forced placements with actual executionNode and executionPlatform
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

BASE_DIR = Path("/root/projects/my-herosim/simulation_data/gnn_datasets")


def build_node_id_to_name_mapping(infrastructure: Dict) -> Dict[int, str]:
    """Build mapping from node_id (index) to node_name."""
    nodes = infrastructure.get("nodes", [])
    return {i: node.get("node_name", f"node_{i}") for i, node in enumerate(nodes)}


def verify_dataset(dataset_dir: Path) -> Tuple[bool, List[Dict]]:
    """
    Verify forced placements for a single dataset.
    
    Returns:
        (all_aligned, mismatches)
    """
    optimal_result_path = dataset_dir / "optimal_result.json"
    
    if not optimal_result_path.exists():
        return False, [{"error": f"optimal_result.json not found"}]
    
    try:
        with open(optimal_result_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return False, [{"error": f"Failed to load JSON: {e}"}]
    
    # Extract placement_plan
    sample = data.get("sample", {})
    placement_plan = sample.get("placement_plan", {})
    
    # Extract infrastructure for node mapping
    config = data.get("config", {})
    infrastructure = config.get("infrastructure", {})
    node_id_to_name = build_node_id_to_name_mapping(infrastructure)
    
    # Extract task results
    stats = data.get("stats", {})
    task_results = stats.get("taskResults", [])
    
    # Build task_id -> task_result mapping
    task_results_by_id = {tr.get("taskId"): tr for tr in task_results}
    
    mismatches = []
    all_aligned = True
    
    # Check tasks 0, 1, 2, 3, 4
    for task_id in [0, 1, 2, 3, 4]:
        task_id_str = str(task_id)
        
        # Get forced placement
        if task_id_str not in placement_plan:
            mismatches.append({
                "task_id": task_id,
                "error": f"No forced placement found in placement_plan"
            })
            all_aligned = False
            continue
        
        forced_placement = placement_plan[task_id_str]
        if not isinstance(forced_placement, list) or len(forced_placement) < 2:
            mismatches.append({
                "task_id": task_id,
                "error": f"Invalid forced placement format: {forced_placement}"
            })
            all_aligned = False
            continue
        
        forced_node_id, forced_platform_id = forced_placement[0], forced_placement[1]
        
        # Map node_id to node_name
        if forced_node_id not in node_id_to_name:
            mismatches.append({
                "task_id": task_id,
                "error": f"Node ID {forced_node_id} not found in infrastructure"
            })
            all_aligned = False
            continue
        
        expected_node_name = node_id_to_name[forced_node_id]
        expected_platform_id = str(forced_platform_id)  # executionPlatform is a string
        
        # Get actual task result
        if task_id not in task_results_by_id:
            mismatches.append({
                "task_id": task_id,
                "error": f"Task result not found for taskId {task_id}"
            })
            all_aligned = False
            continue
        
        task_result = task_results_by_id[task_id]
        actual_node_name = task_result.get("executionNode", "")
        actual_platform_id = task_result.get("executionPlatform", "")
        
        # Compare
        node_match = actual_node_name == expected_node_name
        platform_match = actual_platform_id == expected_platform_id
        
        if not node_match or not platform_match:
            mismatches.append({
                "task_id": task_id,
                "expected": {
                    "node": expected_node_name,
                    "platform": expected_platform_id
                },
                "actual": {
                    "node": actual_node_name,
                    "platform": actual_platform_id
                },
                "node_match": node_match,
                "platform_match": platform_match
            })
            all_aligned = False
    
    return all_aligned, mismatches


def main():
    """Main function to verify all datasets."""
    if not BASE_DIR.exists():
        print(f"Error: Base directory not found: {BASE_DIR}")
        return
    
    # Find all dataset directories
    dataset_dirs = sorted([d for d in BASE_DIR.iterdir() if d.is_dir() and d.name.startswith("ds_")])
    
    if not dataset_dirs:
        print(f"No dataset directories found in {BASE_DIR}")
        return
    
    print(f"Found {len(dataset_dirs)} datasets to verify\n")
    
    total_datasets = 0
    aligned_datasets = 0
    all_mismatches = []
    
    for dataset_dir in tqdm(dataset_dirs, desc="Verifying datasets"):
        total_datasets += 1
        all_aligned, mismatches = verify_dataset(dataset_dir)
        
        if all_aligned:
            aligned_datasets += 1
        else:
            all_mismatches.append({
                "dataset": dataset_dir.name,
                "mismatches": mismatches
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Verification Summary")
    print(f"{'='*60}")
    print(f"Total datasets: {total_datasets}")
    print(f"Aligned datasets: {aligned_datasets}")
    print(f"Datasets with mismatches: {total_datasets - aligned_datasets}")
    
    # Print detailed mismatches
    if all_mismatches:
        print(f"\n{'='*60}")
        print(f"Detailed Mismatches")
        print(f"{'='*60}")
        for dataset_info in all_mismatches:
            print(f"\nDataset: {dataset_info['dataset']}")
            for mismatch in dataset_info['mismatches']:
                if 'error' in mismatch:
                    print(f"  Task {mismatch['task_id']}: ERROR - {mismatch['error']}")
                else:
                    print(f"  Task {mismatch['task_id']}:")
                    print(f"    Expected: node={mismatch['expected']['node']}, platform={mismatch['expected']['platform']}")
                    print(f"    Actual:   node={mismatch['actual']['node']}, platform={mismatch['actual']['platform']}")
                    print(f"    Node match: {mismatch['node_match']}, Platform match: {mismatch['platform_match']}")
    else:
        print("\n✓ All datasets are aligned!")


if __name__ == "__main__":
    main()

