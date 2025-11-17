#!/usr/bin/env python3
"""
Analyze simulation results for:
1. Cold start detection
2. Inference-time placement constraints (based on DeterminedScheduler logic)
3. Graph filtering recommendations

This script analyzes optimal_result.json files to understand:
- Which tasks had cold starts
- What placement constraints would apply at inference time
- How to filter graphs to match inference-time constraints
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict


def load_optimal_result(result_path: Path) -> Dict:
    """Load an optimal_result.json file."""
    with open(result_path, 'r') as f:
        return json.load(f)


def analyze_cold_starts(result: Dict) -> Dict:
    """
    Analyze cold starts in simulation results.
    
    Returns:
        Dict with cold start statistics
    """
    task_results = result.get('stats', {}).get('taskResults', [])
    
    # Filter out warmup tasks (taskId < 0)
    workload_tasks = [tr for tr in task_results if tr.get('taskId', 0) >= 0]
    
    if not workload_tasks:
        return {
            'has_cold_starts': False,
            'total_tasks': 0,
            'cold_start_count': 0,
            'cold_start_ratio': 0.0,
            'cold_start_time_sum': 0.0,
            'cold_start_time_avg': 0.0
        }
    
    cold_started = [tr for tr in workload_tasks if tr.get('coldStarted', False)]
    cold_start_times = [tr.get('coldStartTime', 0.0) for tr in workload_tasks]
    
    # Deep analysis: Check patterns in cold starts
    cold_start_patterns = analyze_cold_start_patterns(result, cold_started, workload_tasks)
    
    return {
        'has_cold_starts': len(cold_started) > 0,
        'total_tasks': len(workload_tasks),
        'cold_start_count': len(cold_started),
        'cold_start_ratio': len(cold_started) / len(workload_tasks) if workload_tasks else 0.0,
        'cold_start_time_sum': sum(cold_start_times),
        'cold_start_time_avg': sum(cold_start_times) / len(cold_start_times) if cold_start_times else 0.0,
        'cold_started_task_ids': [tr.get('taskId') for tr in cold_started],
        'cold_start_times': cold_start_times,
        'patterns': cold_start_patterns
    }


def analyze_cold_start_patterns(result: Dict, cold_started: List[Dict], all_tasks: List[Dict]) -> Dict:
    """
    Analyze patterns in cold starts to identify root causes.
    
    Checks:
    1. Are cold starts happening on platforms with replicas?
    2. Are cold starts correlated with task type?
    3. Are platforms being shared between task types?
    4. Are cold starts happening when previous task type differs?
    """
    if not cold_started:
        return {
            'total_cold_starts': 0,
            'pattern_analysis': 'No cold starts to analyze'
        }
    
    # Get infrastructure to check replicas
    config = result.get('config', {})
    infrastructure = config.get('infrastructure', {})
    nodes_config = infrastructure.get('nodes', [])
    
    # Get replica information from final system state
    system_state_results = result.get('stats', {}).get('systemStateResults', [])
    final_replicas = {}
    if system_state_results:
        final_state = system_state_results[-1]
        replicas_dict = final_state.get('replicas', {})
        for task_type, replica_list in replicas_dict.items():
            replica_set = set()
            for replica in replica_list:
                if isinstance(replica, list) and len(replica) >= 2:
                    node_name = str(replica[0])
                    platform_id = int(replica[1])
                    replica_set.add((node_name, platform_id))
            final_replicas[task_type] = replica_set
    
    # Analyze cold starts
    cold_by_task_type = defaultdict(int)
    cold_by_platform = defaultdict(int)
    cold_on_replica_platforms = 0
    cold_with_wrong_task_type = 0
    platform_task_history = {}  # Track what task types ran on each platform
    
    # Build platform->node mapping
    platform_to_node = {}
    node_id_to_name = {i: node['node_name'] for i, node in enumerate(nodes_config)}
    platform_id = 0
    for node in nodes_config:
        for _ in node.get('platforms', []):
            node_name = node['node_name']
            platform_to_node[platform_id] = node_name
            platform_id += 1
    
    # Process all tasks to build history
    for task in sorted(all_tasks, key=lambda t: t.get('startedTime', 0.0)):
        task_id = task.get('taskId')
        task_type = task.get('taskType', {}).get('name')
        platform_info = task.get('platform', {})
        platform_id_str = task.get('executionPlatform', '')
        node_name = task.get('executionNode', '')
        
        if not task_type or not platform_id_str or not node_name:
            continue
        
        # Try to get platform ID
        try:
            # Platform ID might be in executionPlatform field
            # Or we need to find it from platform info
            platform_id_val = None
            if platform_id_str.isdigit():
                platform_id_val = int(platform_id_str)
            elif node_name in platform_to_node.values():
                # Try to find platform ID from node
                pass
            
            key = (node_name, platform_id_val) if platform_id_val is not None else (node_name, None)
            
            if key not in platform_task_history:
                platform_task_history[key] = []
            platform_task_history[key].append({
                'task_id': task_id,
                'task_type': task_type,
                'time': task.get('startedTime', 0.0),
                'cold_started': task.get('coldStarted', False)
            })
        except Exception:
            pass
    
    # Analyze cold starts
    for task in cold_started:
        task_type = task.get('taskType', {}).get('name')
        node_name = task.get('executionNode', '')
        platform_id_str = task.get('executionPlatform', '')
        
        cold_by_task_type[task_type] += 1
        
        if node_name and platform_id_str:
            try:
                platform_id_val = int(platform_id_str) if platform_id_str.isdigit() else None
                key = (node_name, platform_id_val) if platform_id_val else (node_name, None)
                cold_by_platform[key] += 1
                
                # Check if this platform has a replica for this task type
                if task_type in final_replicas:
                    if (node_name, platform_id_val) in final_replicas[task_type]:
                        cold_on_replica_platforms += 1
                    else:
                        # Platform doesn't have replica for this task type
                        cold_with_wrong_task_type += 1
                
                # Check platform history for task type mismatch
                if key in platform_task_history:
                    history = platform_task_history[key]
                    # Find previous task on this platform
                    task_time = task.get('startedTime', 0.0)
                    previous_tasks = [h for h in history if h['time'] < task_time]
                    if previous_tasks:
                        last_task = max(previous_tasks, key=lambda h: h['time'])
                        if last_task['task_type'] != task_type:
                            cold_with_wrong_task_type += 1
            except Exception:
                pass
    
    return {
        'total_cold_starts': len(cold_started),
        'cold_by_task_type': dict(cold_by_task_type),
        'cold_by_platform': {f"{node}:{pid}": count for (node, pid), count in cold_by_platform.items()},
        'cold_on_replica_platforms': cold_on_replica_platforms,
        'cold_with_wrong_task_type': cold_with_wrong_task_type,
        'cold_on_replica_ratio': cold_on_replica_platforms / len(cold_started) if cold_started else 0.0,
        'cold_wrong_type_ratio': cold_with_wrong_task_type / len(cold_started) if cold_started else 0.0,
        'platform_sharing_issue': cold_with_wrong_task_type > 0
    }


def get_inference_time_replicas(result: Dict) -> Dict[str, Set[Tuple[str, int]]]:
    """
    Extract available replicas at inference time from systemStateResults.
    
    This represents what would be available at inference time (based on final system state).
    
    Returns:
        Dict mapping task_type -> Set of (node_name, platform_id) tuples
    """
    system_state_results = result.get('stats', {}).get('systemStateResults', [])
    
    if not system_state_results:
        return {}
    
    # Use the final system state (last snapshot)
    final_state = system_state_results[-1]
    replicas_dict = final_state.get('replicas', {})
    
    # Convert to our format: task_type -> Set[(node_name, platform_id)]
    inference_replicas = {}
    
    for task_type, replica_list in replicas_dict.items():
        replica_set = set()
        # Replicas are stored as [node_name, platform_id] lists
        for replica in replica_list:
            if isinstance(replica, list) and len(replica) >= 2:
                node_name = str(replica[0])
                platform_id = int(replica[1])
                replica_set.add((node_name, platform_id))
        inference_replicas[task_type] = replica_set
    
    return inference_replicas


def get_inference_time_valid_placements(
    result: Dict,
    task_type: str,
    source_node_name: str,
    inference_replicas: Dict[str, Set[Tuple[str, int]]],
    nodes_config: List[Dict]
) -> Set[Tuple[str, int]]:
    """
    Get valid placements at inference time based on DeterminedScheduler._get_valid_replicas logic.
    
    Rules:
    1. Include if it's the task's source node (local execution always allowed)
    2. Include if it's a server node (not starting with 'client_node') AND has network connectivity
    3. Exclude client nodes that aren't the source node
    
    Args:
        result: Full result dictionary
        task_type: Task type name (e.g., 'dnn1', 'dnn2')
        source_node_name: Source node name for the task
        inference_replicas: Replicas available at inference time
        nodes_config: List of node configurations from infrastructure
    
    Returns:
        Set of valid (node_name, platform_id) tuples
    """
    # Get network maps for each node
    node_network_maps = {
        node['node_name']: node.get('network_map', {})
        for node in nodes_config
    }
    
    # Get available replicas for this task type
    task_replicas = inference_replicas.get(task_type, set())
    
    valid_placements = set()
    
    for node_name, platform_id in task_replicas:
        # Rule 1: Local execution always allowed
        if node_name == source_node_name:
            valid_placements.add((node_name, platform_id))
        
        # Rule 2: Server nodes with network connectivity
        elif not node_name.startswith('client_node'):
            # Check if this server node has network connectivity to source
            network_map = node_network_maps.get(node_name, {})
            if source_node_name in network_map:
                valid_placements.add((node_name, platform_id))
        
        # Rule 3: Client nodes (that aren't the source) are excluded
        # (No action needed, they're not added)
    
    return valid_placements


def analyze_task_placements_vs_inference_constraints(result: Dict) -> Dict:
    """
    Analyze whether optimal placements in the result would be valid at inference time.
    
    Returns:
        Dict with analysis results
    """
    config = result.get('config', {})
    infrastructure = config.get('infrastructure', {})
    nodes_config = infrastructure.get('nodes', [])
    
    # Get optimal placements
    placement_plan = result.get('sample', {}).get('placement_plan', {})
    
    # Get inference-time replicas
    inference_replicas = get_inference_time_replicas(result)
    
    # Get task results
    task_results = result.get('stats', {}).get('taskResults', [])
    workload_tasks = [tr for tr in task_results if tr.get('taskId', 0) >= 0]
    
    if not workload_tasks:
        return {
            'total_tasks': 0,
            'valid_at_inference': 0,
            'invalid_at_inference': 0,
            'invalid_reasons': {}
        }
    
    valid_count = 0
    invalid_count = 0
    invalid_reasons = defaultdict(list)
    
    # Build node_id to node_name mapping
    node_id_to_name = {i: node['node_name'] for i, node in enumerate(nodes_config)}
    
    for task_result in workload_tasks:
        task_id = task_result.get('taskId')
        task_type = task_result.get('taskType', {}).get('name')
        source_node_name = task_result.get('sourceNode', '')
        execution_node = task_result.get('executionNode', '')
        execution_platform = task_result.get('executionPlatform', '')
        
        if not task_type or not source_node_name:
            continue
        
        # Get optimal placement for this task
        placement = placement_plan.get(str(task_id))
        if not placement or not isinstance(placement, list) or len(placement) < 2:
            invalid_count += 1
            invalid_reasons['no_placement'].append(task_id)
            continue
        
        opt_node_id, opt_platform_id = placement[0], placement[1]
        opt_node_name = node_id_to_name.get(opt_node_id)
        
        if not opt_node_name:
            invalid_count += 1
            invalid_reasons['invalid_node_id'].append(task_id)
            continue
        
        # Check if this placement would be valid at inference time
        valid_placements = get_inference_time_valid_placements(
            result, task_type, source_node_name, inference_replicas, nodes_config
        )
        
        placement_tuple = (opt_node_name, opt_platform_id)
        
        if placement_tuple in valid_placements:
            valid_count += 1
        else:
            invalid_count += 1
            # Determine reason
            if opt_node_name == source_node_name:
                reason = 'local_not_in_replicas'
            elif opt_node_name.startswith('client_node'):
                reason = 'client_node_not_source'
            else:
                # Check network connectivity
                node_network_map = next(
                    (n.get('network_map', {}) for n in nodes_config if n['node_name'] == opt_node_name),
                    {}
                )
                if source_node_name not in node_network_map:
                    reason = 'no_network_connectivity'
                else:
                    reason = 'not_in_replicas'
            
            invalid_reasons[reason].append(task_id)
    
    return {
        'total_tasks': len(workload_tasks),
        'valid_at_inference': valid_count,
        'invalid_at_inference': invalid_count,
        'invalid_ratio': invalid_count / len(workload_tasks) if workload_tasks else 0.0,
        'invalid_reasons': dict(invalid_reasons)
    }


def compare_graph_edges_vs_inference_constraints(
    result: Dict,
    graph_edge_info: Optional[Dict] = None
) -> Dict:
    """
    Compare what edges the graph has vs what would be valid at inference time.
    
    This helps understand if graph construction should be filtered further.
    
    Args:
        result: Full result dictionary
        graph_edge_info: Optional dict with graph edge information (if available)
    
    Returns:
        Dict with comparison results
    """
    config = result.get('config', {})
    infrastructure = config.get('infrastructure', {})
    nodes_config = infrastructure.get('nodes', [])
    
    # Get inference-time replicas
    inference_replicas = get_inference_time_replicas(result)
    
    # Get task results to understand task requirements
    task_results = result.get('stats', {}).get('taskResults', [])
    workload_tasks = [tr for tr in task_results if tr.get('taskId', 0) >= 0]
    
    if not workload_tasks:
        return {'analysis': 'no_workload_tasks'}
    
    # Analyze per task
    task_analysis = []
    
    for task_result in workload_tasks:
        task_id = task_result.get('taskId')
        task_type = task_result.get('taskType', {}).get('name')
        source_node_name = task_result.get('sourceNode', '')
        
        if not task_type or not source_node_name:
            continue
        
        # Get valid placements at inference time
        valid_placements = get_inference_time_valid_placements(
            result, task_type, source_node_name, inference_replicas, nodes_config
        )
        
        task_analysis.append({
            'task_id': task_id,
            'task_type': task_type,
            'source_node': source_node_name,
            'valid_placement_count': len(valid_placements),
            'valid_placements': list(valid_placements)
        })
    
    # Summary statistics
    valid_counts = [ta['valid_placement_count'] for ta in task_analysis]
    
    return {
        'tasks_analyzed': len(task_analysis),
        'avg_valid_placements_per_task': sum(valid_counts) / len(valid_counts) if valid_counts else 0.0,
        'min_valid_placements': min(valid_counts) if valid_counts else 0,
        'max_valid_placements': max(valid_counts) if valid_counts else 0,
        'tasks_with_no_valid_placements': sum(1 for c in valid_counts if c == 0),
        'task_details': task_analysis
    }


def analyze_dataset(dataset_dir: Path) -> Dict:
    """Analyze a single dataset directory."""
    optimal_result_path = dataset_dir / "optimal_result.json"
    
    if not optimal_result_path.exists():
        return {'error': 'optimal_result.json not found'}
    
    result = load_optimal_result(optimal_result_path)
    
    # Analyze cold starts
    cold_start_analysis = analyze_cold_starts(result)
    
    # Analyze placement constraints
    placement_analysis = analyze_task_placements_vs_inference_constraints(result)
    
    # Compare with inference constraints
    edge_comparison = compare_graph_edges_vs_inference_constraints(result)
    
    # Get inference-time replicas
    inference_replicas = get_inference_time_replicas(result)
    
    return {
        'dataset_id': dataset_dir.name,
        'cold_starts': cold_start_analysis,
        'placement_constraints': placement_analysis,
        'edge_comparison': edge_comparison,
        'inference_replicas': {
            task_type: len(replicas)
            for task_type, replicas in inference_replicas.items()
        }
    }


def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze simulation results for cold starts and inference-time constraints'
    )
    parser.add_argument(
        'base_dir',
        type=Path,
        help='Base directory containing gnn_datasets (e.g., simulation_data/artifacts/run9_all/gnn_datasets)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output JSON file for analysis results'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print summary statistics'
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    if not base_dir.exists():
        print(f"Error: Base directory {base_dir} does not exist")
        sys.exit(1)
    
    # Find all dataset directories
    dataset_dirs = sorted(base_dir.glob("ds_*"))
    
    if not dataset_dirs:
        print(f"Error: No dataset directories found in {base_dir}")
        sys.exit(1)
    
    print(f"Analyzing {len(dataset_dirs)} datasets...")
    
    results = []
    
    for dataset_dir in dataset_dirs:
        print(f"  Analyzing {dataset_dir.name}...", end=' ')
        try:
            analysis = analyze_dataset(dataset_dir)
            results.append(analysis)
            
            if 'error' in analysis:
                print(f"ERROR: {analysis['error']}")
            else:
                has_cold = analysis['cold_starts']['has_cold_starts']
                invalid_ratio = analysis['placement_constraints'].get('invalid_ratio', 0.0)
                print(f"Cold starts: {has_cold}, Invalid placements: {invalid_ratio*100:.1f}%")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'dataset_id': dataset_dir.name,
                'error': str(e)
            })
    
    # Summary statistics
    if args.summary:
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        valid_results = [r for r in results if 'error' not in r]
        
        if valid_results:
            # Cold start statistics
            datasets_with_cold_starts = sum(
                1 for r in valid_results
                if r.get('cold_starts', {}).get('has_cold_starts', False)
            )
            
            avg_cold_start_ratio = sum(
                r.get('cold_starts', {}).get('cold_start_ratio', 0.0)
                for r in valid_results
            ) / len(valid_results)
            
            # Placement constraint statistics
            avg_invalid_ratio = sum(
                r.get('placement_constraints', {}).get('invalid_ratio', 0.0)
                for r in valid_results
            ) / len(valid_results)
            
            datasets_with_invalid_placements = sum(
                1 for r in valid_results
                if r.get('placement_constraints', {}).get('invalid_at_inference', 0) > 0
            )
            
            print(f"\nCold Starts:")
            print(f"  Datasets with cold starts: {datasets_with_cold_starts} / {len(valid_results)}")
            print(f"  Average cold start ratio: {avg_cold_start_ratio*100:.2f}%")
            
            print(f"\nInference-Time Placement Constraints:")
            print(f"  Datasets with invalid placements: {datasets_with_invalid_placements} / {len(valid_results)}")
            print(f"  Average invalid placement ratio: {avg_invalid_ratio*100:.2f}%")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()


"""
   python src/notebooks/new/analyze_inference_placement_constraints.py \
       simulation_data/artifacts/run9_all/gnn_datasets \
       --summary \
       --output analysis_results.json
"""