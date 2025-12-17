#!/usr/bin/env python3
"""
Analysis script to compare available_resources and replicas in systemStateResults.

For each dataset in simulation_data/gnn_datasets:
1. Load optimal_result.json
2. Extract the last systemStateResult
3. Analyze differences between available_resources and replicas:
   - Platforms in replicas but not removed from available_resources (overlap - should not exist)
   - Platforms in available_resources that are also in replicas (inconsistency)
   - Platforms that are neither (missing platforms)
   - Summary statistics
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from tqdm import tqdm

BASE_DIR = Path("/root/projects/my-herosim/simulation_data/gnn_datasets")


def extract_replicas_by_node(replicas: Dict) -> Dict[str, Set[int]]:
    """
    Extract replicas grouped by node_name.
    
    Returns:
        Dict[node_name, set of platform_ids]
    """
    node_replicas = defaultdict(set)
    
    for task_type, replica_list in replicas.items():
        for replica in replica_list:
            if isinstance(replica, list) and len(replica) >= 2:
                node_name, platform_id = replica[0], replica[1]
                node_replicas[node_name].add(platform_id)
    
    return dict(node_replicas)


def analyze_dataset(dataset_dir: Path) -> Dict:
    """
    Analyze available_resources vs replicas for a single dataset.
    
    Returns:
        Dict with analysis results
    """
    optimal_result_path = dataset_dir / "optimal_result.json"
    
    if not optimal_result_path.exists():
        return {"error": "optimal_result.json not found"}
    
    try:
        with open(optimal_result_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return {"error": f"Failed to load JSON: {e}"}
    
    # Extract the last systemStateResult
    stats = data.get("stats", {})
    system_state_results = stats.get("systemStateResults", [])
    
    if not system_state_results:
        return {"error": "No systemStateResults found"}
    
    last_state = system_state_results[-1]
    available_resources = last_state.get("available_resources", {})
    replicas = last_state.get("replicas", {})
    
    # Extract replicas by node
    replicas_by_node = extract_replicas_by_node(replicas)
    
    # Build infrastructure mapping to get all platforms per node
    config = data.get("config", {})
    infrastructure = config.get("infrastructure", {})
    nodes = infrastructure.get("nodes", [])
    
    # Build expected platforms per node (from infrastructure)
    # Platform IDs are assigned sequentially: 0, 1, 2, ... across all nodes
    expected_platforms_by_node = {}
    platform_id = 0
    for node in nodes:
        node_name = node.get("node_name", "")
        platforms = node.get("platforms", [])
        platform_ids = list(range(platform_id, platform_id + len(platforms)))
        expected_platforms_by_node[node_name] = set(platform_ids)
        platform_id += len(platforms)
    
    # Analyze each node
    analysis = {
        "dataset": dataset_dir.name,
        "nodes": {},
        "summary": {
            "total_nodes": len(available_resources),
            "nodes_with_overlaps": 0,
            "nodes_with_missing_platforms": 0,
            "total_overlaps": 0,
            "total_missing": 0,
            "total_available_only": 0,
            "total_replica_only": 0
        }
    }
    
    all_nodes = set(available_resources.keys()) | set(replicas_by_node.keys()) | set(expected_platforms_by_node.keys())
    
    for node_name in all_nodes:
        available = set(available_resources.get(node_name, []))
        replica_platforms = replicas_by_node.get(node_name, set())
        expected = expected_platforms_by_node.get(node_name, set())
        
        # Find overlaps (platforms in both available and replicas - should not exist)
        overlaps = available & replica_platforms
        
        # Find platforms only in available (unused)
        available_only = available - replica_platforms
        
        # Find platforms only in replicas (allocated, correctly removed from available)
        replica_only = replica_platforms - available
        
        # Find missing platforms (in expected but neither in available nor replicas)
        missing = expected - available - replica_platforms
        
        # Find unexpected platforms (in available or replicas but not in expected)
        unexpected_available = available - expected
        unexpected_replica = replica_platforms - expected
        
        node_analysis = {
            "available_count": len(available),
            "replica_count": len(replica_platforms),
            "expected_count": len(expected),
            "overlaps": sorted(list(overlaps)),
            "overlap_count": len(overlaps),
            "available_only": sorted(list(available_only)),
            "available_only_count": len(available_only),
            "replica_only": sorted(list(replica_only)),
            "replica_only_count": len(replica_only),
            "missing": sorted(list(missing)),
            "missing_count": len(missing),
            "unexpected_available": sorted(list(unexpected_available)),
            "unexpected_replica": sorted(list(unexpected_replica)),
        }
        
        analysis["nodes"][node_name] = node_analysis
        
        # Update summary
        if overlaps:
            analysis["summary"]["nodes_with_overlaps"] += 1
            analysis["summary"]["total_overlaps"] += len(overlaps)
        if missing:
            analysis["summary"]["nodes_with_missing_platforms"] += 1
            analysis["summary"]["total_missing"] += len(missing)
        
        analysis["summary"]["total_available_only"] += len(available_only)
        analysis["summary"]["total_replica_only"] += len(replica_only)
    
    return analysis


def main():
    """Main function to analyze all datasets."""
    if not BASE_DIR.exists():
        print(f"Error: Base directory not found: {BASE_DIR}")
        return
    
    # Find all dataset directories
    dataset_dirs = sorted([d for d in BASE_DIR.iterdir() if d.is_dir() and d.name.startswith("ds_")])
    
    if not dataset_dirs:
        print(f"No dataset directories found in {BASE_DIR}")
        return
    
    print(f"Found {len(dataset_dirs)} datasets to analyze\n")
    
    all_analyses = []
    
    for dataset_dir in tqdm(dataset_dirs, desc="Analyzing datasets"):
        analysis = analyze_dataset(dataset_dir)
        if "error" not in analysis:
            all_analyses.append(analysis)
    
    # Print summary across all datasets
    print(f"\n{'='*80}")
    print(f"Summary Across All Datasets")
    print(f"{'='*80}")
    
    total_overlaps = sum(a["summary"]["total_overlaps"] for a in all_analyses)
    total_missing = sum(a["summary"]["total_missing"] for a in all_analyses)
    total_available_only = sum(a["summary"]["total_available_only"] for a in all_analyses)
    total_replica_only = sum(a["summary"]["total_replica_only"] for a in all_analyses)
    nodes_with_overlaps = sum(a["summary"]["nodes_with_overlaps"] for a in all_analyses)
    nodes_with_missing = sum(a["summary"]["nodes_with_missing_platforms"] for a in all_analyses)
    
    print(f"Total datasets analyzed: {len(all_analyses)}")
    print(f"\nIssues Found:")
    print(f"  Nodes with overlaps (platform in both available and replicas): {nodes_with_overlaps}")
    print(f"  Total overlapping platforms: {total_overlaps}")
    print(f"  Nodes with missing platforms: {nodes_with_missing}")
    print(f"  Total missing platforms: {total_missing}")
    print(f"\nNormal State:")
    print(f"  Platforms only in available_resources (unused): {total_available_only}")
    print(f"  Platforms only in replicas (allocated): {total_replica_only}")
    if total_available_only + total_replica_only > 0:
        total_accounted = total_available_only + total_replica_only
        print(f"  Allocation rate: {total_replica_only / total_accounted * 100:.1f}% allocated, {total_available_only / total_accounted * 100:.1f}% unused")
    
    # Print detailed analysis for datasets with issues
    datasets_with_issues = [
        a for a in all_analyses 
        if a["summary"]["total_overlaps"] > 0 or a["summary"]["total_missing"] > 0
    ]
    
    if datasets_with_issues:
        print(f"\n{'='*80}")
        print(f"Detailed Analysis for Datasets with Issues ({len(datasets_with_issues)} datasets)")
        print(f"{'='*80}")
        
        for analysis in datasets_with_issues:
            print(f"\nDataset: {analysis['dataset']}")
            print(f"  Overlaps: {analysis['summary']['total_overlaps']}, Missing: {analysis['summary']['total_missing']}")
            
            for node_name, node_data in sorted(analysis["nodes"].items()):
                if node_data["overlap_count"] > 0 or node_data["missing_count"] > 0:
                    print(f"    Node: {node_name}")
                    if node_data["overlap_count"] > 0:
                        print(f"      ⚠️  OVERLAPS: {node_data['overlaps']} (in both available and replicas)")
                    if node_data["missing_count"] > 0:
                        print(f"      ⚠️  MISSING: {node_data['missing']} (expected but not found)")
                    if node_data["unexpected_available"]:
                        print(f"      ⚠️  Unexpected in available: {node_data['unexpected_available']}")
                    if node_data["unexpected_replica"]:
                        print(f"      ⚠️  Unexpected in replicas: {node_data['unexpected_replica']}")
    else:
        print(f"\n✓ No issues found! All datasets are consistent.")
    
    # Print statistics for a sample dataset
    if all_analyses:
        print(f"\n{'='*80}")
        print(f"Sample Analysis (first dataset)")
        print(f"{'='*80}")
        sample = all_analyses[0]
        print(f"Dataset: {sample['dataset']}")
        
        # Show replica breakdown by task type
        sample_result_path = BASE_DIR / sample['dataset'] / "optimal_result.json"
        try:
            with open(sample_result_path, 'r') as f:
                sample_data = json.load(f)
            stats = sample_data.get("stats", {})
            system_state_results = stats.get("systemStateResults", [])
            if system_state_results:
                last_state = system_state_results[-1]
                replicas = last_state.get("replicas", {})
                print(f"\nReplica Breakdown by Task Type:")
                for task_type, replica_list in replicas.items():
                    print(f"  {task_type}: {len(replica_list)} replicas")
        except:
            pass
        
        print(f"\nNode Breakdown:")
        for node_name in sorted(sample["nodes"].keys())[:10]:  # Show first 10 nodes
            node_data = sample["nodes"][node_name]
            print(f"  {node_name}:")
            print(f"    Available: {node_data['available_count']}, Replicas: {node_data['replica_count']}, Expected: {node_data['expected_count']}")
            if node_data["available_only"]:
                print(f"    Available only: {node_data['available_only'][:5]}{'...' if len(node_data['available_only']) > 5 else ''}")
            if node_data["replica_only"]:
                print(f"    Replica only: {node_data['replica_only'][:5]}{'...' if len(node_data['replica_only']) > 5 else ''}")


if __name__ == "__main__":
    main()

