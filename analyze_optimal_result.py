#!/usr/bin/env python3
"""
Extract all relevant data from optimal_result.json into pandas DataFrames for GNN training.
Creates multiple DataFrames: nodes, edges, tasks, platforms, replicas, and summary metrics.
"""

import json
import pandas as pd
from pathlib import Path

OPT_PATH = Path("/root/projects/my-herosim/simulation_data/gnn_datasets/ds_0049/optimal_result.json")

with open(OPT_PATH, "r") as f:
    result = json.load(f)

print(f"Loaded: {OPT_PATH} ({OPT_PATH.stat().st_size} bytes)\n")

# ============================================================================
# 1. NODES DataFrame - Infrastructure nodes with static features
# ============================================================================
nodes_data = []
infra_nodes = result.get("config", {}).get("infrastructure", {}).get("nodes", [])

for i, node in enumerate(infra_nodes):
    node_name = node.get("node_name", f"node_{i}")
    node_type = node.get("type", "unknown")
    memory = node.get("memory", 0)
    platforms = node.get("platforms", [])
    storage = node.get("storage", [])
    network_map = node.get("network_map", {})
    
    # Count platform types
    platform_counts = {}
    for p in platforms:
        platform_counts[p] = platform_counts.get(p, 0) + 1
    
    nodes_data.append({
        'node_id': i,
        'node_name': node_name,
        'node_type': node_type,
        'is_client': node_name.startswith('client_node'),
        'memory_gb': memory,
        'n_platforms': len(platforms),
        'n_storage': len(storage),
        'n_connections': len(network_map),
        'platform_types': ','.join(sorted(set(platforms))),
        **{f'n_{pt}': platform_counts.get(pt, 0) for pt in ['rpiCpu', 'xavierCpu', 'xavierGpu', 'xavierDla', 'pynqFpga']}
    })

df_nodes = pd.DataFrame(nodes_data)
print("=" * 80)
print("NODES DataFrame")
print("=" * 80)
print(df_nodes.head(10))
print(f"\nShape: {df_nodes.shape}")
print()

# ============================================================================
# 2. EDGES DataFrame - Network topology with latencies
# ============================================================================
edges_data = []
network_topology = result.get("stats", {}).get("networkTopology", {})

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
print("=" * 80)
print("EDGES DataFrame")
print("=" * 80)
print(df_edges.head(10))
print(f"\nShape: {df_edges.shape}")
print(f"Total edges: {len(df_edges)}")
print(f"Average latency: {df_edges['latency'].mean():.4f}s")
print()

# ============================================================================
# 3. TASKS DataFrame - Workload tasks with features and optimal placements
# ============================================================================
tasks_data = []
task_results = result.get("stats", {}).get("taskResults", [])
placement_plan = result.get("sample", {}).get("placement_plan", {})
workload_events = result.get("config", {}).get("workload", {}).get("events", [])

for task_result in task_results:
    task_id = task_result.get("taskId")
    task_type = task_result.get("taskType", {}).get("name", "unknown")
    
    # Optimal placement from placement_plan
    placement = placement_plan.get(str(task_id), [None, None])
    if isinstance(placement, list) and len(placement) >= 2:
        opt_node_id, opt_platform_id = placement[0], placement[1]
    else:
        opt_node_id, opt_platform_id = None, None
    
    # Find corresponding workload event
    source_node = task_result.get("sourceNode", "")
    dispatched_time = task_result.get("dispatchedTime", 0)
    
    tasks_data.append({
        'task_id': task_id,
        'task_type': task_type,
        'source_node': source_node,
        'dispatched_time': dispatched_time,
        'scheduled_time': task_result.get("scheduledTime", 0),
        'arrived_time': task_result.get("arrivedTime", 0),
        'started_time': task_result.get("startedTime", 0),
        'done_time': task_result.get("doneTime", 0),
        # OPTIMAL PLACEMENT (ground truth labels)
        'optimal_node_id': opt_node_id,
        'optimal_platform_id': opt_platform_id,
        'optimal_platform_type': task_result.get("platform", {}).get("shortName", ""),
        # PERFORMANCE METRICS (labels for regression)
        'elapsed_time': task_result.get("elapsedTime", 0),
        'cold_start_time': task_result.get("coldStartTime", 0),
        'execution_time': task_result.get("executionTime", 0),
        'queue_time': task_result.get("queueTime", 0),
        'wait_time': task_result.get("waitTime", 0),
        'network_latency': task_result.get("networkLatency", 0),
        'compute_time': task_result.get("computeTime", 0),
        'communications_time': task_result.get("communicationsTime", 0),
        # STATE INDICATORS
        'cold_started': task_result.get("coldStarted", False),
        'cache_hit': task_result.get("cacheHit", False),
        'local_dependencies': task_result.get("localDependencies", False),
        'penalty': task_result.get("penalty", False),
        'energy': task_result.get("energy", 0)
    })

df_tasks = pd.DataFrame(tasks_data)
print("=" * 80)
print("TASKS DataFrame (GNN Training Labels)")
print("=" * 80)
print(df_tasks)
print(f"\nShape: {df_tasks.shape}")
print(f"\nTotal RTT (sum of elapsed_time): {df_tasks['elapsed_time'].sum():.4f}s")
print(f"Cold start proportion: {df_tasks['cold_started'].mean() * 100:.1f}%")
print()

# ============================================================================
# 4. PLATFORMS DataFrame - All platforms with replica state
# ============================================================================
platforms_data = []
node_results = result.get("stats", {}).get("nodeResults", [])

# Get replica state from system state
system_state = result.get("stats", {}).get("systemStateResults", [{}])[-1]
replicas_by_task = system_state.get("replicas", {})

for node_result in node_results:
    node_id = node_result.get("nodeId")
    node_name = infra_nodes[node_id].get("node_name") if node_id < len(infra_nodes) else f"node_{node_id}"
    
    for plat_result in node_result.get("platformResults", []):
        plat_id = plat_result.get("platformId")
        plat_type_info = plat_result.get("platformType", {})
        plat_type = plat_type_info.get("shortName", "unknown")
        
        # Check which task types have replicas on this platform
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
            # Replica state (features)
            'has_dnn1_replica': has_dnn1_replica,
            'has_dnn2_replica': has_dnn2_replica,
            # Runtime state
            'energy_consumed': plat_result.get("energy", 0),
            'idle_time': plat_result.get("idleTime", 0),
            'idle_proportion': plat_result.get("idleProportion", 0),
            'storage_time': plat_result.get("storageTime", 0)
        })

df_platforms = pd.DataFrame(platforms_data)
print("=" * 80)
print("PLATFORMS DataFrame (Replica State + Runtime)")
print("=" * 80)
print(df_platforms.head(15))
print(f"\nShape: {df_platforms.shape}")
print(f"\nReplica distribution:")
print(f"  Platforms with dnn1 replicas: {df_platforms['has_dnn1_replica'].sum()}")
print(f"  Platforms with dnn2 replicas: {df_platforms['has_dnn2_replica'].sum()}")
print()

# ============================================================================
# 5. TASK_COMPATIBILITY DataFrame - Which platforms support which tasks
# ============================================================================
task_types = result.get("sim_inputs", {}).get("task_types", {})
compatibility_data = []

for task_name, task_config in task_types.items():
    supported_platforms = task_config.get("platforms", [])
    memory_reqs = task_config.get("memoryRequirements", {})
    exec_times = task_config.get("executionTime", {})
    cold_start_times = task_config.get("coldStartDuration", {})
    
    for plat_type in supported_platforms:
        compatibility_data.append({
            'task_type': task_name,
            'platform_type': plat_type,
            'memory_required': memory_reqs.get(plat_type, 0),
            'execution_time': exec_times.get(plat_type, 0),
            'cold_start_duration': cold_start_times.get(plat_type, 0),
            'image_size': task_config.get("imageSize", {}).get(plat_type, 0)
        })

df_compatibility = pd.DataFrame(compatibility_data)
print("=" * 80)
print("TASK_COMPATIBILITY DataFrame (Task-Platform Constraints)")
print("=" * 80)
print(df_compatibility)
print(f"\nShape: {df_compatibility.shape}")
print()

# ============================================================================
# 6. CONFIGURATION DataFrame - Dataset configuration (single row)
# ============================================================================
config_data = {
    'dataset_id': 'ds_0049',
    'connection_probability': result.get("sample", {}).get("infra_config", {}).get("network", {}).get("topology", {}).get("connection_probability", 0),
    'replicas_per_client_dnn1': result.get("sample", {}).get("infra_config", {}).get("replicas", {}).get("dnn1", {}).get("per_client", 0),
    'replicas_per_server_dnn1': result.get("sample", {}).get("infra_config", {}).get("replicas", {}).get("dnn1", {}).get("per_server", 0),
    'replicas_per_client_dnn2': result.get("sample", {}).get("infra_config", {}).get("replicas", {}).get("dnn2", {}).get("per_client", 0),
    'replicas_per_server_dnn2': result.get("sample", {}).get("infra_config", {}).get("replicas", {}).get("dnn2", {}).get("per_server", 0),
    'preinit_client_pct': result.get("sample", {}).get("infra_config", {}).get("preinit", {}).get("client_percentage", 0),
    'preinit_server_pct': result.get("sample", {}).get("infra_config", {}).get("preinit", {}).get("server_percentage", 0),
    'prewarm_queue_dnn1': result.get("sample", {}).get("infra_config", {}).get("prewarm", {}).get("dnn1", {}).get("initial_queue", 0),
    'prewarm_queue_dnn2': result.get("sample", {}).get("infra_config", {}).get("prewarm", {}).get("dnn2", {}).get("initial_queue", 0),
    'network_bandwidth': result.get("config", {}).get("infrastructure", {}).get("network", {}).get("bandwidth", 0),
    'total_nodes': len(infra_nodes),
    'total_client_nodes': sum(1 for n in infra_nodes if n.get("node_name", "").startswith("client_node")),
    'total_server_nodes': sum(1 for n in infra_nodes if not n.get("node_name", "").startswith("client_node")),
}

df_config = pd.DataFrame([config_data])
print("=" * 80)
print("CONFIGURATION DataFrame")
print("=" * 80)
print(df_config.T)
print()

# ============================================================================
# 7. SUMMARY_METRICS DataFrame - Overall performance (single row)
# ============================================================================
stats = result.get("stats", {})
best_info_path = OPT_PATH.parent / "best.json"
best_rtt = None
if best_info_path.exists():
    with open(best_info_path, "r") as f:
        best_rtt = json.load(f).get("rtt")

metrics_data = {
    'total_rtt': best_rtt or df_tasks['elapsed_time'].sum() if not df_tasks.empty else 0,
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
print("=" * 80)
print("SUMMARY_METRICS DataFrame")
print("=" * 80)
print(df_metrics.T)
print()

# ============================================================================
# 8. PLACEMENTS DataFrame - Optimal placement decisions (GNN labels)
# ============================================================================
placements_data = []
placement_plan = result.get("sample", {}).get("placement_plan", {})

for task_id_str, placement in placement_plan.items():
    if isinstance(placement, list) and len(placement) >= 2:
        node_id, platform_id = placement[0], placement[1]
        
        # Find task info
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
print("=" * 80)
print("PLACEMENTS DataFrame (GNN Ground Truth Labels)")
print("=" * 80)
print(df_placements)
print(f"\nShape: {df_placements.shape}")
print(f"Local executions: {df_placements['is_local'].sum()}/{len(df_placements)}")
print()

# ============================================================================
# 9. REPLICAS DataFrame - Initial replica distribution
# ============================================================================
replicas_data = []
for task_type, replica_list in replicas_by_task.items():
    if isinstance(replica_list, list):
        for replica in replica_list:
            if isinstance(replica, list) and len(replica) >= 2:
                node_name, platform_id = replica[0], replica[1]
                
                # Find node_id
                node_id = next((i for i, n in enumerate(infra_nodes) if n.get("node_name") == node_name), None)
                
                replicas_data.append({
                    'task_type': task_type,
                    'node_id': node_id,
                    'node_name': node_name,
                    'platform_id': platform_id,
                    'is_client': node_name.startswith('client_node')
                })

df_replicas = pd.DataFrame(replicas_data)
print("=" * 80)
print("REPLICAS DataFrame (Initial Warm Replica Distribution)")
print("=" * 80)
print(df_replicas.head(15))
print(f"\nShape: {df_replicas.shape}")
print("\nReplica counts by task type:")
print(df_replicas.groupby('task_type').size())
print("\nReplica distribution (client vs server):")
print(df_replicas.groupby(['task_type', 'is_client']).size())
print()

# ============================================================================
# 10. CONSOLIDATED SUMMARY
# ============================================================================
print("=" * 80)
print("CONSOLIDATED SUMMARY FOR GNN TRAINING")
print("=" * 80)
print(f"\n{'Dataset ID:':<30} ds_0049")
print(f"{'File:':<30} {OPT_PATH.name}")
print(f"{'Total RTT (optimal):':<30} {metrics_data['total_rtt']:.4f}s")
print(f"\n{'Infrastructure:':<30}")
print(f"  {'Nodes:':<28} {len(df_nodes)} ({df_config['total_client_nodes'].values[0]} clients, {df_config['total_server_nodes'].values[0]} servers)")
print(f"  {'Platforms:':<28} {len(df_platforms)}")
print(f"  {'Network edges:':<28} {len(df_edges)}")
print(f"  {'Connection probability:':<28} {config_data['connection_probability']}")
print(f"\n{'Workload:':<30}")
print(f"  {'Tasks:':<28} {len(df_tasks)}")
print(f"  {'Task types:':<28} {df_tasks['task_type'].nunique()}")
print(f"  {'Duration:':<28} {df_tasks['done_time'].max():.3f}s")
print(f"\n{'Replica Configuration:':<30}")
print(f"  {'dnn1 replicas:':<28} {len(df_replicas[df_replicas['task_type']=='dnn1'])}")
print(f"  {'dnn2 replicas:':<28} {len(df_replicas[df_replicas['task_type']=='dnn2'])}")
print(f"  {'Prewarm queue (dnn1):':<28} {config_data['prewarm_queue_dnn1']}")
print(f"  {'Prewarm queue (dnn2):':<28} {config_data['prewarm_queue_dnn2']}")
print(f"\n{'Performance Metrics:':<30}")
print(f"  {'Cold start proportion:':<28} {metrics_data['cold_start_proportion']:.1f}%")
print(f"  {'Offloading rate:':<28} {metrics_data['offloading_rate']:.1f}%")
print(f"  {'Cache hit rate:':<28} {metrics_data['cache_hit_proportion']:.1f}%")
print(f"  {'Avg network latency:':<28} {metrics_data['avg_network_latency']:.4f}s")

print("\n" + "=" * 80)
print("DataFrames Available for GNN Training:")
print("=" * 80)
print(f"  df_nodes          : {df_nodes.shape} - Node features")
print(f"  df_edges          : {df_edges.shape} - Network connectivity + latencies")
print(f"  df_tasks          : {df_tasks.shape} - Task features + labels")
print(f"  df_platforms      : {df_platforms.shape} - Platform state + replicas")
print(f"  df_placements     : {df_placements.shape} - Optimal placement labels")
print(f"  df_compatibility  : {df_compatibility.shape} - Task-platform constraints")
print(f"  df_replicas       : {df_replicas.shape} - Warm replica locations")
print(f"  df_config         : {df_config.shape} - Dataset configuration")
print(f"  df_metrics        : {df_metrics.shape} - Summary performance metrics")
print()

# ============================================================================
# 11. GNN TRAINING USAGE EXAMPLE
# ============================================================================
print("=" * 80)
print("GNN TRAINING USAGE")
print("=" * 80)
print("""
# Graph Construction for PyTorch Geometric:
# -----------------------------------------

import torch
from torch_geometric.data import HeteroData

data = HeteroData()

# Node features (20 nodes)
data['node'].x = torch.tensor(df_nodes[['memory_gb', 'n_platforms', 'n_connections', 
                                        'n_rpiCpu', 'n_xavierCpu', 'n_xavierGpu']].values, 
                              dtype=torch.float)

# Edge indices and features (network topology)
src_ids = df_edges['src_node'].map(lambda x: df_nodes[df_nodes['node_name']==x].index[0])
dst_ids = df_edges['dst_node'].map(lambda x: df_nodes[df_nodes['node_name']==x].index[0])
data['node', 'connected_to', 'node'].edge_index = torch.tensor([src_ids.values, dst_ids.values])
data['node', 'connected_to', 'node'].edge_attr = torch.tensor(df_edges[['latency']].values, dtype=torch.float)

# Task features (5 tasks)
data['task'].x = torch.tensor(df_tasks[['dispatched_time', 'task_type']].values)  # Encode task_type

# Platform features (98 platforms)
data['platform'].x = torch.tensor(df_platforms[['has_dnn1_replica', 'has_dnn2_replica', 
                                                 'idle_proportion']].values, dtype=torch.float)

# Task-to-source-node edges
task_src_node_ids = df_tasks['source_node'].map(lambda x: df_nodes[df_nodes['node_name']==x].index[0])
data['task', 'originates_from', 'node'].edge_index = torch.tensor([
    range(len(df_tasks)), task_src_node_ids.values
])

# LABELS (Ground Truth for Supervised Learning):
# -----------------------------------------------

# Classification labels: optimal (node_id, platform_id) per task
y_node = torch.tensor(df_placements['optimal_node_id'].values, dtype=torch.long)
y_platform = torch.tensor(df_placements['optimal_platform_id'].values, dtype=torch.long)

# Regression labels: RTT per task
y_rtt = torch.tensor(df_placements['rtt'].values, dtype=torch.float)

# Total RTT for ranking loss
y_total_rtt = torch.tensor([{total_rtt}], dtype=torch.float)

# Training Loop Example:
# ----------------------

model = YourGNN()
optimizer = torch.optim.Adam(model.parameters())

# Forward pass
pred_placements, pred_rtt = model(data)

# Multi-task loss
loss_placement = F.cross_entropy(pred_placements, y_node)  # Node prediction
loss_rtt = F.mse_loss(pred_rtt, y_rtt)                     # RTT prediction
loss = loss_placement + 0.1 * loss_rtt

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()
""".format(total_rtt=metrics_data['total_rtt']))

print("\n" + "=" * 80)
print("KEY INSIGHTS FOR GNN TRAINING")
print("=" * 80)
print(f"""
1. SUPERVISED LABELS:
   - Optimal placements: df_placements['optimal_node_id', 'optimal_platform_id']
   - Performance targets: df_placements['rtt'] (per-task), df_metrics['total_rtt'] (overall)

2. INPUT FEATURES:
   - Node features: df_nodes (type, memory, platforms, connectivity)
   - Edge features: df_edges (latencies)
   - Task features: df_tasks (type, source, timestamp)
   - Platform state: df_platforms (replicas, utilization)

3. CONSTRAINTS:
   - Compatibility: df_compatibility (which platforms support which tasks)
   - Network: df_edges (only connected nodes can communicate)
   - Replicas: df_replicas (warm replica locations affect cold start costs)

4. LEARNING OBJECTIVES:
   - Minimize total RTT across all tasks
   - Predict optimal placements that balance:
     * Network latency (offload cost)
     * Cold start overhead (replica warmth)
     * Queue contention (platform load)
     * Local vs remote execution tradeoffs

5. DATASET DIVERSITY (from 1000 datasets):
   - Varying network connectivity (0.05 to 0.50)
   - Different replica configurations
   - Multiple pre-warm states (0 to 24 queued tasks)
   → GNN learns robust policies across configurations
""")

print("DataFrames ready for GNN feature engineering and training! ✅")

