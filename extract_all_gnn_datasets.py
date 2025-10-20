#!/usr/bin/env python3
"""
Extract all GNN datasets into pandas DataFrames for training.
Each dataset produces multiple DataFrames (nodes, edges, tasks, platforms, etc.)
Stores all in a dictionary structure for iterative processing.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')


def extract_dataset_to_dataframes(optimal_result_path: Path) -> Dict[str, pd.DataFrame]:
    """
    Extract a single optimal_result.json file into multiple DataFrames.
    
    Returns:
        Dictionary containing all DataFrames for this dataset
    """
    with open(optimal_result_path, "r") as f:
        result = json.load(f)
    
    # Get dataset ID from path
    dataset_id = optimal_result_path.parent.name
    
    # Get infrastructure and stats
    infra_nodes = result.get("config", {}).get("infrastructure", {}).get("nodes", [])
    stats = result.get("stats", {})
    task_results = stats.get("taskResults", [])
    placement_plan = result.get("sample", {}).get("placement_plan", {})
    
    # ========================================================================
    # 1. NODES DataFrame
    # ========================================================================
    nodes_data = []
    for i, node in enumerate(infra_nodes):
        node_name = node.get("node_name", f"node_{i}")
        node_type = node.get("type", "unknown")
        platforms = node.get("platforms", [])
        network_map = node.get("network_map", {})
        
        platform_counts = {}
        for p in platforms:
            platform_counts[p] = platform_counts.get(p, 0) + 1
        
        nodes_data.append({
            'node_id': i,
            'node_name': node_name,
            'node_type': node_type,
            'is_client': node_name.startswith('client_node'),
            'memory_gb': node.get("memory", 0),
            'n_platforms': len(platforms),
            'n_storage': len(node.get("storage", [])),
            'n_connections': len(network_map),
            'platform_types': ','.join(sorted(set(platforms))),
            'n_rpiCpu': platform_counts.get('rpiCpu', 0),
            'n_xavierCpu': platform_counts.get('xavierCpu', 0),
            'n_xavierGpu': platform_counts.get('xavierGpu', 0),
            'n_xavierDla': platform_counts.get('xavierDla', 0),
            'n_pynqFpga': platform_counts.get('pynqFpga', 0)
        })
    
    df_nodes = pd.DataFrame(nodes_data)
    
    # ========================================================================
    # 2. EDGES DataFrame
    # ========================================================================
    edges_data = []
    network_topology = stats.get("networkTopology", {})
    
    for src_node, connections in network_topology.items():
        if isinstance(connections, dict):
            for dst_node, latency in connections.items():
                edges_data.append({
                    'src_node': src_node,
                    'dst_node': dst_node,
                    'latency': latency,
                    'src_is_client': src_node.startswith('client_node'),
                    'dst_is_client': dst_node.startswith('client_node')
                })
    
    df_edges = pd.DataFrame(edges_data)
    
    # ========================================================================
    # 3. TASKS DataFrame
    # ========================================================================
    tasks_data = []
    for task_result in task_results:
        task_id = task_result.get("taskId")
        placement = placement_plan.get(str(task_id), [None, None])
        
        if isinstance(placement, list) and len(placement) >= 2:
            opt_node_id, opt_platform_id = placement[0], placement[1]
        else:
            opt_node_id, opt_platform_id = None, None
        
        tasks_data.append({
            'task_id': task_id,
            'task_type': task_result.get("taskType", {}).get("name", "unknown"),
            'source_node': task_result.get("sourceNode", ""),
            'dispatched_time': task_result.get("dispatchedTime", 0),
            'scheduled_time': task_result.get("scheduledTime", 0),
            'arrived_time': task_result.get("arrivedTime", 0),
            'started_time': task_result.get("startedTime", 0),
            'done_time': task_result.get("doneTime", 0),
            'optimal_node_id': opt_node_id,
            'optimal_platform_id': opt_platform_id,
            'optimal_platform_type': task_result.get("platform", {}).get("shortName", ""),
            'elapsed_time': task_result.get("elapsedTime", 0),
            'cold_start_time': task_result.get("coldStartTime", 0),
            'execution_time': task_result.get("executionTime", 0),
            'queue_time': task_result.get("queueTime", 0),
            'wait_time': task_result.get("waitTime", 0),
            'network_latency': task_result.get("networkLatency", 0),
            'compute_time': task_result.get("computeTime", 0),
            'communications_time': task_result.get("communicationsTime", 0),
            'cold_started': task_result.get("coldStarted", False),
            'cache_hit': task_result.get("cacheHit", False),
            'local_dependencies': task_result.get("localDependencies", False),
            'penalty': task_result.get("penalty", False),
            'energy': task_result.get("energy", 0)
        })
    
    df_tasks = pd.DataFrame(tasks_data)
    
    # ========================================================================
    # 4. PLATFORMS DataFrame
    # ========================================================================
    platforms_data = []
    node_results = stats.get("nodeResults", [])
    
    # Get replica state
    system_state = stats.get("systemStateResults", [{}])[-1] if stats.get("systemStateResults") else {}
    replicas_by_task = system_state.get("replicas", {})
    
    for node_result in node_results:
        node_id = node_result.get("nodeId")
        node_name = infra_nodes[node_id].get("node_name") if node_id < len(infra_nodes) else f"node_{node_id}"
        
        for plat_result in node_result.get("platformResults", []):
            plat_id = plat_result.get("platformId")
            plat_type_info = plat_result.get("platformType", {})
            plat_type = plat_type_info.get("shortName", "unknown")
            
            # Check replica state
            has_dnn1_replica = False
            has_dnn2_replica = False
            
            for task_type, replica_list in replicas_by_task.items():
                if isinstance(replica_list, list):
                    for replica in replica_list:
                        if isinstance(replica, list) and len(replica) >= 2:
                            if replica[0] == node_name and replica[1] == plat_id:
                                if task_type == "dnn1":
                                    has_dnn1_replica = True
                                elif task_type == "dnn2":
                                    has_dnn2_replica = True
            
            platforms_data.append({
                'platform_id': plat_id,
                'node_id': node_id,
                'node_name': node_name,
                'platform_type': plat_type,
                'hardware': plat_type_info.get("hardware", ""),
                'price': plat_type_info.get("price", 0),
                'idle_energy': plat_type_info.get("idleEnergy", 0),
                'has_dnn1_replica': has_dnn1_replica,
                'has_dnn2_replica': has_dnn2_replica,
                'energy_consumed': plat_result.get("energy", 0),
                'idle_time': plat_result.get("idleTime", 0),
                'idle_proportion': plat_result.get("idleProportion", 0),
                'storage_time': plat_result.get("storageTime", 0)
            })
    
    df_platforms = pd.DataFrame(platforms_data)
    
    # ========================================================================
    # 5. TASK_COMPATIBILITY DataFrame
    # ========================================================================
    task_types = result.get("sim_inputs", {}).get("task_types", {})
    compatibility_data = []
    
    for task_name, task_config in task_types.items():
        supported_platforms = task_config.get("platforms", [])
        memory_reqs = task_config.get("memoryRequirements", {})
        exec_times = task_config.get("executionTime", {})
        cold_start_times = task_config.get("coldStartDuration", {})
        image_sizes = task_config.get("imageSize", {})
        
        for plat_type in supported_platforms:
            compatibility_data.append({
                'task_type': task_name,
                'platform_type': plat_type,
                'memory_required': memory_reqs.get(plat_type, 0),
                'execution_time': exec_times.get(plat_type, 0),
                'cold_start_duration': cold_start_times.get(plat_type, 0),
                'image_size': image_sizes.get(plat_type, 0)
            })
    
    df_compatibility = pd.DataFrame(compatibility_data)
    
    # ========================================================================
    # 6. PLACEMENTS DataFrame (Ground Truth Labels)
    # ========================================================================
    placements_data = []
    for task_id_str, placement in placement_plan.items():
        if isinstance(placement, list) and len(placement) >= 2:
            node_id, platform_id = placement[0], placement[1]
            task_result = next((tr for tr in task_results if tr.get("taskId") == int(task_id_str)), None)
            
            placements_data.append({
                'task_id': int(task_id_str),
                'optimal_node_id': node_id,
                'optimal_platform_id': platform_id,
                'node_name': infra_nodes[node_id].get("node_name") if node_id < len(infra_nodes) else "",
                'platform_type': task_result.get("platform", {}).get("shortName", "") if task_result else "",
                'task_type': task_result.get("taskType", {}).get("name", "") if task_result else "",
                'source_node': task_result.get("sourceNode", "") if task_result else "",
                'is_local': infra_nodes[node_id].get("node_name") == (task_result.get("sourceNode", "") if task_result else "") if node_id < len(infra_nodes) else False,
                'rtt': task_result.get("elapsedTime", 0) if task_result else 0
            })
    
    df_placements = pd.DataFrame(placements_data)
    
    # ========================================================================
    # 7. REPLICAS DataFrame
    # ========================================================================
    replicas_data = []
    for task_type, replica_list in replicas_by_task.items():
        if isinstance(replica_list, list):
            for replica in replica_list:
                if isinstance(replica, list) and len(replica) >= 2:
                    node_name, platform_id = replica[0], replica[1]
                    node_id = next((i for i, n in enumerate(infra_nodes) if n.get("node_name") == node_name), None)
                    
                    replicas_data.append({
                        'task_type': task_type,
                        'node_id': node_id,
                        'node_name': node_name,
                        'platform_id': platform_id,
                        'is_client': node_name.startswith('client_node')
                    })
    
    df_replicas = pd.DataFrame(replicas_data)
    
    # ========================================================================
    # 8. CONFIGURATION DataFrame
    # ========================================================================
    infra_config = result.get("sample", {}).get("infra_config", {})
    config_data = {
        'dataset_id': dataset_id,
        'connection_probability': infra_config.get("network", {}).get("topology", {}).get("connection_probability", 0),
        'replicas_per_client_dnn1': infra_config.get("replicas", {}).get("dnn1", {}).get("per_client", 0),
        'replicas_per_server_dnn1': infra_config.get("replicas", {}).get("dnn1", {}).get("per_server", 0),
        'replicas_per_client_dnn2': infra_config.get("replicas", {}).get("dnn2", {}).get("per_client", 0),
        'replicas_per_server_dnn2': infra_config.get("replicas", {}).get("dnn2", {}).get("per_server", 0),
        'preinit_client_pct': infra_config.get("preinit", {}).get("client_percentage", 0),
        'preinit_server_pct': infra_config.get("preinit", {}).get("server_percentage", 0),
        'prewarm_queue_dnn1': infra_config.get("prewarm", {}).get("dnn1", {}).get("initial_queue", 0),
        'prewarm_queue_dnn2': infra_config.get("prewarm", {}).get("dnn2", {}).get("initial_queue", 0),
        'network_bandwidth': result.get("config", {}).get("infrastructure", {}).get("network", {}).get("bandwidth", 0),
        'total_nodes': len(infra_nodes),
        'total_client_nodes': sum(1 for n in infra_nodes if n.get("node_name", "").startswith("client_node")),
        'total_server_nodes': sum(1 for n in infra_nodes if not n.get("node_name", "").startswith("client_node"))
    }
    
    df_config = pd.DataFrame([config_data])
    
    # ========================================================================
    # 9. SUMMARY_METRICS DataFrame
    # ========================================================================
    # Try to get RTT from best.json
    best_json_path = optimal_result_path.parent / "best.json"
    best_rtt = None
    if best_json_path.exists():
        try:
            with open(best_json_path, "r") as f:
                best_rtt = json.load(f).get("rtt")
        except:
            pass
    
    # Fallback to sum of task elapsed times
    if best_rtt is None and task_results:
        best_rtt = sum(tr.get("elapsedTime", 0) for tr in task_results)
    
    metrics_data = {
        'dataset_id': dataset_id,
        'total_rtt': best_rtt or 0,
        'avg_elapsed_time': stats.get("averageElapsedTime", 0),
        'avg_cold_start_time': stats.get("averageColdStartTime", 0),
        'avg_execution_time': stats.get("averageExecutionTime", 0),
        'avg_wait_time': stats.get("averageWaitTime", 0),
        'avg_queue_time': stats.get("averageQueueTime", 0),
        'avg_compute_time': stats.get("averageComputeTime", 0),
        'avg_communications_time': stats.get("averageCommunicationsTime", 0),
        'avg_network_latency': stats.get("averageNetworkLatency", 0),
        'total_energy': stats.get("energy", 0),
        'reclaimable_energy': stats.get("reclaimableEnergy", 0),
        'penalty_proportion': stats.get("penaltyProportion", 0),
        'cold_start_proportion': stats.get("coldStartProportion", 0),
        'cache_hit_proportion': stats.get("taskCacheHitsProportion", 0),
        'offloading_rate': stats.get("offloadingRate", 0),
        'unused_platforms_pct': stats.get("unusedPlatforms", 0),
        'unused_nodes_pct': stats.get("unusedNodes", 0),
        'end_time': stats.get("endTime", 0),
        'n_tasks': len(task_results)
    }
    
    df_metrics = pd.DataFrame([metrics_data])
    
    # Return all DataFrames for this dataset
    return {
        'nodes': df_nodes,
        'edges': df_edges,
        'tasks': df_tasks,
        'platforms': df_platforms,
        'placements': df_placements,
        'compatibility': df_compatibility,
        'replicas': df_replicas,
        'config': df_config,
        'metrics': df_metrics
    }


def extract_all_datasets(base_dir: Path, max_datasets: int = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Extract all datasets from gnn_datasets directory.
    
    Args:
        base_dir: Path to gnn_datasets directory
        max_datasets: Optional limit on number of datasets to process
    
    Returns:
        Dictionary mapping dataset_id to its DataFrames dictionary
    """
    all_datasets = {}
    dataset_dirs = sorted(base_dir.glob("ds_*"))
    
    if max_datasets:
        dataset_dirs = dataset_dirs[:max_datasets]
    
    print(f"Found {len(dataset_dirs)} dataset directories")
    print(f"Processing datasets...\n")
    
    successful = 0
    skipped = 0
    
    for i, dataset_dir in enumerate(dataset_dirs):
        dataset_id = dataset_dir.name
        optimal_result_path = dataset_dir / "optimal_result.json"
        
        if not optimal_result_path.exists():
            skipped += 1
            if i < 10 or i % 100 == 0:
                print(f"  [{i+1:4d}/{len(dataset_dirs)}] {dataset_id}: SKIP (no optimal_result.json)")
            continue
        
        try:
            dataframes = extract_dataset_to_dataframes(optimal_result_path)
            all_datasets[dataset_id] = dataframes
            successful += 1
            
            if i < 10 or i % 100 == 0:
                rtt = dataframes['metrics']['total_rtt'].values[0]
                n_tasks = dataframes['metrics']['n_tasks'].values[0]
                print(f"  [{i+1:4d}/{len(dataset_dirs)}] {dataset_id}: OK (RTT={rtt:.3f}s, tasks={int(n_tasks)})")
        
        except Exception as e:
            skipped += 1
            if i < 10:
                print(f"  [{i+1:4d}/{len(dataset_dirs)}] {dataset_id}: ERROR - {str(e)}")
    
    print(f"\n{'='*80}")
    print(f"Extraction Complete!")
    print(f"{'='*80}")
    print(f"  Successful: {successful}")
    print(f"  Skipped:    {skipped}")
    print(f"  Total:      {len(dataset_dirs)}")
    
    return all_datasets


def create_consolidated_dataframes(all_datasets: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """
    Consolidate all datasets into single DataFrames with dataset_id as key.
    
    Returns:
        Dictionary of consolidated DataFrames (one per table type)
    """
    print(f"\n{'='*80}")
    print("Creating Consolidated DataFrames...")
    print(f"{'='*80}\n")
    
    consolidated = {}
    
    # Tables that should be concatenated across datasets
    tables_to_consolidate = ['config', 'metrics', 'placements', 'tasks']
    
    for table_name in tables_to_consolidate:
        dfs_to_concat = []
        
        for dataset_id, dataframes in all_datasets.items():
            if table_name in dataframes:
                df = dataframes[table_name].copy()
                # Add dataset_id if not present
                if 'dataset_id' not in df.columns:
                    df.insert(0, 'dataset_id', dataset_id)
                dfs_to_concat.append(df)
        
        if dfs_to_concat:
            consolidated[table_name] = pd.concat(dfs_to_concat, ignore_index=True)
            print(f"  {table_name:<20} : {consolidated[table_name].shape}")
    
    return consolidated


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    BASE_DIR = Path("/root/projects/my-herosim/simulation_data/gnn_datasets")
    
    # Extract all datasets
    print("="*80)
    print("EXTRACTING ALL GNN DATASETS")
    print("="*80)
    print()
    
    all_datasets = extract_all_datasets(BASE_DIR, max_datasets=None)
    
    # Create consolidated DataFrames
    consolidated = create_consolidated_dataframes(all_datasets)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print(f"\n{'='*80}")
    print("FINAL DATASET STRUCTURE")
    print(f"{'='*80}\n")
    
    print("1. PER-DATASET DataFrames (all_datasets dict):")
    print(f"   - Keys: {len(all_datasets)} dataset IDs (ds_0000, ds_0001, ...)")
    print("   - Each dataset contains:")
    print("     * nodes (20 rows) - Infrastructure nodes")
    print("     * edges (varies) - Network connectivity")
    print("     * tasks (5 rows) - Workload tasks with labels")
    print("     * platforms (98 rows) - Platform state")
    print("     * placements (5 rows) - Optimal placement labels")
    print("     * compatibility (6 rows) - Task-platform constraints")
    print("     * replicas (varies) - Warm replica distribution")
    print("     * config (1 row) - Configuration parameters")
    print("     * metrics (1 row) - Performance summary")
    print()
    
    print("2. CONSOLIDATED DataFrames (consolidated dict):")
    for name, df in consolidated.items():
        print(f"   - {name:<20} : {df.shape}")
    print()
    
    print("="*80)
    print("USAGE EXAMPLES")
    print("="*80)
    print("""
# Access a specific dataset:
ds_0049_data = all_datasets['ds_0049']
df_nodes = ds_0049_data['nodes']
df_tasks = ds_0049_data['tasks']

# Access consolidated data across all datasets:
all_configs = consolidated['config']
all_metrics = consolidated['metrics']
all_placements = consolidated['placements']

# Iterate through all datasets:
for dataset_id, dataframes in all_datasets.items():
    print(f"{dataset_id}: RTT = {dataframes['metrics']['total_rtt'].values[0]:.3f}s")

# Find best performing dataset:
best_dataset_id = consolidated['metrics'].loc[consolidated['metrics']['total_rtt'].idxmin(), 'dataset_id']
print(f"Best dataset: {best_dataset_id}")

# Analyze configuration impact:
config_rtt = consolidated['config'].merge(consolidated['metrics'], on='dataset_id')
print(config_rtt[['connection_probability', 'prewarm_queue_dnn1', 'total_rtt']].corr())
""")
    
    print("\n" + "="*80)
    print("DATA READY FOR GNN TRAINING!")
    print("="*80)
    print(f"  Total datasets: {len(all_datasets)}")
    print(f"  Total tasks (training samples): {len(consolidated.get('tasks', pd.DataFrame()))}")
    print(f"  Total placements (labels): {len(consolidated.get('placements', pd.DataFrame()))}")
    print()
    
    # Save summary statistics
    if consolidated.get('metrics') is not None:
        print("Dataset Performance Summary:")
        print(f"  RTT range: {consolidated['metrics']['total_rtt'].min():.3f}s - {consolidated['metrics']['total_rtt'].max():.3f}s")
        print(f"  RTT mean:  {consolidated['metrics']['total_rtt'].mean():.3f}s")
        print(f"  RTT std:   {consolidated['metrics']['total_rtt'].std():.3f}s")

