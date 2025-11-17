#!/usr/bin/env python3
"""
Verification script to check that optimal placements never use cold platforms.

For each dataset in run9_all/gnn_datasets:
1. Load optimal_result.json
2. Extract placement_plan (optimal placements)
3. Extract replicas from systemStateResults
4. Check if each optimal placement uses a warm platform (has replica)
5. Report any violations
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
from tqdm import tqdm

BASE_DIR = Path("/root/projects/my-herosim/simulation_data/artifacts/run10_all/gnn_datasets")


def load_replicas_from_optimal_result(optimal_result_path: Path) -> Dict[str, Set[Tuple[str, int]]]:
    """
    Load replicas from optimal_result.json.
    
    Returns:
        Dict mapping task_type -> set of (node_name, platform_id) tuples
    """
    try:
        with open(optimal_result_path, 'r') as f:
            data = json.load(f)
        
        stats = data.get('stats', {})
        system_state_results = stats.get('systemStateResults', [])
        
        if not system_state_results:
            return {}
        
        # Get the last system state (most recent)
        last_state = system_state_results[-1]
        
        # Parse replicas: task_type -> set of (node_name, platform_id)
        replicas_dict = {}
        replicas_raw = last_state.get('replicas', {})
        for task_type, replica_list in replicas_raw.items():
            if isinstance(replica_list, list):
                replicas_dict[task_type] = set()
                for replica in replica_list:
                    if isinstance(replica, list) and len(replica) >= 2:
                        node_name, platform_id = replica[0], replica[1]
                        replicas_dict[task_type].add((node_name, platform_id))
        
        return replicas_dict
    except Exception as e:
        print(f"  Error loading replicas from {optimal_result_path}: {e}")
        return {}


def verify_dataset(dataset_dir: Path) -> Tuple[bool, List[Dict]]:
    """
    Verify that optimal placements in a dataset only use warm platforms.
    
    Returns:
        (is_valid, list_of_violations)
    """
    optimal_result_path = dataset_dir / "optimal_result.json"
    if not optimal_result_path.exists():
        return True, []  # Skip if no optimal_result.json
    
    try:
        with open(optimal_result_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return True, [{"error": f"Failed to load JSON: {e}"}]
    
    # Extract infrastructure nodes for node_id -> node_name mapping
    infra_nodes = data.get("config", {}).get("infrastructure", {}).get("nodes", [])
    node_id_to_name = {i: node.get("node_name", f"node_{i}") for i, node in enumerate(infra_nodes)}
    
    # Extract placement plan
    placement_plan = data.get("sample", {}).get("placement_plan", {})
    if not placement_plan:
        return True, []  # No placements to check
    
    # Extract task results to get task types
    task_results = data.get("stats", {}).get("taskResults", [])
    task_id_to_type = {}
    for tr in task_results:
        task_id = tr.get("taskId")
        task_type = tr.get("taskType", {}).get("name", "unknown")
        if task_id is not None:
            task_id_to_type[task_id] = task_type
    
    # Load replicas
    replicas = load_replicas_from_optimal_result(optimal_result_path)
    
    # Check each placement
    violations = []
    for task_id_str, placement in placement_plan.items():
        try:
            task_id = int(task_id_str)
        except (ValueError, TypeError):
            continue
        
        if not isinstance(placement, list) or len(placement) < 2:
            continue
        
        node_id, platform_id = placement[0], placement[1]
        
        # Skip if node_id or platform_id is None
        if node_id is None or platform_id is None:
            continue
        
        # Get node name
        node_name = node_id_to_name.get(node_id)
        if node_name is None:
            continue
        
        # Get task type
        task_type = task_id_to_type.get(task_id, "unknown")
        
        # Check if this platform has a warm replica for this task type
        task_type_replicas = replicas.get(task_type, set())
        is_warm = (node_name, platform_id) in task_type_replicas
        
        if not is_warm:
            violations.append({
                "task_id": task_id,
                "task_type": task_type,
                "node_id": node_id,
                "node_name": node_name,
                "platform_id": platform_id,
                "reason": f"Platform {platform_id} on node {node_name} does not have warm replica for task type {task_type}"
            })
    
    return len(violations) == 0, violations


def main():
    print("="*80)
    print("VERIFYING OPTIMAL PLACEMENTS USE ONLY WARM PLATFORMS")
    print("="*80)
    print()
    
    dataset_dirs = sorted(BASE_DIR.glob("ds_*"))
    print(f"Found {len(dataset_dirs)} dataset directories")
    print()
    
    total_violations = 0
    datasets_with_violations = []
    datasets_checked = 0
    datasets_skipped = 0
    
    for dataset_dir in tqdm(dataset_dirs, desc="Checking datasets", unit="dataset"):
        is_valid, violations = verify_dataset(dataset_dir)
        datasets_checked += 1
        
        if violations:
            total_violations += len(violations)
            datasets_with_violations.append({
                "dataset_id": dataset_dir.name,
                "violations": violations
            })
    
    print()
    print("="*80)
    print("VERIFICATION RESULTS")
    print("="*80)
    print(f"Datasets checked: {datasets_checked}")
    print(f"Datasets with violations: {len(datasets_with_violations)}")
    print(f"Total violations: {total_violations}")
    print()
    
    if datasets_with_violations:
        print("⚠️  VIOLATIONS FOUND:")
        print()
        for ds_info in datasets_with_violations[:10]:  # Show first 10
            print(f"Dataset: {ds_info['dataset_id']}")
            for v in ds_info['violations']:
                print(f"  Task {v['task_id']} ({v['task_type']}): {v['reason']}")
            print()
        
        if len(datasets_with_violations) > 10:
            print(f"... and {len(datasets_with_violations) - 10} more datasets with violations")
    else:
        print("✅ All optimal placements use warm platforms!")
        print("   No violations found.")
    
    print()
    print("="*80)


if __name__ == "__main__":
    main()



