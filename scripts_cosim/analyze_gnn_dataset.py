#!/usr/bin/env python3
"""
GNN Dataset Analysis Script
Analyzes the output of a single GNN dataset generation to understand:
1. What data is available at scheduling/inference time
2. Key factors affecting RTT decisions
3. How to compare with HRC scheduler decisions
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def load_infrastructure(path: Path):
    with open(path, 'r') as f:
        return json.load(f)


def load_placements(path: Path):
    placements = []
    with open(path, 'r') as f:
        for line in f:
            placements.append(json.loads(line))
    return placements


def load_workload(path: Path):
    with open(path, 'r') as f:
        return json.load(f)


def load_optimal_result(path: Path):
    with open(path, 'r') as f:
        return json.load(f)


def analyze_dataset(dataset_dir: Path):
    """Analyze a single GNN dataset."""
    
    infra = load_infrastructure(dataset_dir / 'infrastructure.json')
    placements = load_placements(dataset_dir / 'placements' / 'placements.jsonl')
    workload = load_workload(dataset_dir / 'workload.json')
    optimal_result = load_optimal_result(dataset_dir / 'optimal_result.json')
    
    print("=" * 80)
    print("GNN DATASET ANALYSIS")
    print("=" * 80)
    
    # =========================================================================
    # 1. WORKLOAD ANALYSIS
    # =========================================================================
    print("\n[1] WORKLOAD SUMMARY")
    print("-" * 40)
    tasks_by_type = defaultdict(list)
    for i, event in enumerate(workload['events']):
        task_type = list(event['application']['dag'].keys())[0]
        src_node = event['node_name']
        tasks_by_type[task_type].append((i, src_node))
    
    print(f"Total tasks: {len(workload['events'])}")
    for task_type, tasks in tasks_by_type.items():
        print(f"  {task_type}: {len(tasks)} tasks from {[t[1] for t in tasks]}")
    
    # =========================================================================
    # 2. INFRASTRUCTURE ANALYSIS
    # =========================================================================
    print("\n[2] INFRASTRUCTURE SUMMARY")
    print("-" * 40)
    
    # Network topology
    network = infra['network_maps']
    client_nodes = [n for n in network.keys() if n.startswith('client_node')]
    server_nodes = [n for n in network.keys() if not n.startswith('client_node')]
    
    print(f"Client nodes: {len(client_nodes)}")
    print(f"Server nodes: {len(server_nodes)}")
    
    # Calculate connectivity
    avg_connections = sum(len(v) for k, v in network.items() if k.startswith('client_node')) / len(client_nodes)
    print(f"Avg server connections per client: {avg_connections:.1f}")
    
    # Replica placements
    replicas = infra['replica_placements']
    print(f"\nReplica placements:")
    for task_type, r_list in replicas.items():
        by_location = defaultdict(int)
        for r in r_list:
            location = 'client' if r['node_name'].startswith('client_node') else 'server'
            by_location[location] += 1
        print(f"  {task_type}: {len(r_list)} total ({by_location['client']} client, {by_location['server']} server)")
    
    # Queue distributions
    queues = infra['queue_distributions']
    for task_type, q_dict in queues.items():
        non_zero = sum(1 for v in q_dict.values() if v > 0)
        total_queue = sum(q_dict.values())
        print(f"\nQueue distribution ({task_type}): {non_zero}/{len(q_dict)} non-zero, total={total_queue}")
    
    # =========================================================================
    # 3. PLACEMENT ANALYSIS
    # =========================================================================
    print("\n[3] PLACEMENT ANALYSIS")
    print("-" * 40)
    
    rtts = [p['rtt'] for p in placements]
    placements_sorted = sorted(placements, key=lambda x: x['rtt'])
    
    print(f"Total placement combinations: {len(placements)}")
    print(f"RTT range: {min(rtts):.4f}s - {max(rtts):.4f}s")
    print(f"Mean RTT: {sum(rtts)/len(rtts):.4f}s")
    print(f"Max/Min ratio: {max(rtts)/min(rtts):.2f}x")
    
    # Analyze what makes placements optimal vs suboptimal
    best_placements = placements_sorted[:10]
    worst_placements = placements_sorted[-10:]
    
    print("\n[4] OPTIMAL VS SUBOPTIMAL ANALYSIS")
    print("-" * 40)
    
    # Count local vs offload decisions
    def analyze_placement_pattern(placement_list, desc):
        local_count = 0
        offload_count = 0
        for p in placement_list:
            for task_id_str, (node_id, plat_id) in p['placement_plan'].items():
                task_id = int(task_id_str)
                task_src = workload['events'][task_id]['node_name']
                # Determine if local or offload based on node_id
                # node_ids 0-9 are client nodes, 10+ are server nodes
                is_client_node = node_id < 10
                src_node_id = int(task_src.replace('client_node', ''))
                is_local = is_client_node and node_id == src_node_id
                if is_local:
                    local_count += 1
                else:
                    offload_count += 1
        total = local_count + offload_count
        print(f"{desc}: local={local_count}/{total} ({100*local_count/total:.1f}%), offload={offload_count}/{total} ({100*offload_count/total:.1f}%)")
    
    analyze_placement_pattern(best_placements, "Best placements (top 10)")
    analyze_placement_pattern(worst_placements, "Worst placements (bottom 10)")
    
    # =========================================================================
    # 5. KEY DATA FOR GNN DECISIONS
    # =========================================================================
    print("\n[5] KEY DATA FOR GNN DECISIONS (like HRC)")
    print("-" * 40)
    
    task_results = optimal_result['stats']['taskResults']
    
    print("\nData available at scheduling/inference time:")
    print("  - Task type (dnn1/dnn2) and source node")
    print("  - Network topology: latencies from source_node to all servers")
    print("  - Replica placements: which platforms have warm replicas")
    print("  - Queue snapshot: current queue length per replica")
    print("  - Platform types: rpiCpu, xavierCpu, xavierGpu, xavierDla, pynqFpga")
    print("  - Execution times per platform type (from task-types.json)")
    print("  - Cold start duration per platform type")
    
    print("\nKey factors HRC considers (from scheduler.py):")
    print("  1. Penalty: Whether task will miss deadline")
    print("     - current_task_cold_start + current_task_execution + queue_backlog + cold_start + exec_time + network")
    print("  2. Energy consumption: task.type['energy'][platform_type]")
    print("  3. Consolidation: platform_concurrency / target_concurrency")
    
    print("\nSimulation provides (not assumed/approximated):")
    print("  - Actual execution time (simulated, not estimated)")
    print("  - Actual cold start time (simulated when replica not warm)")
    print("  - Actual queue wait time (based on queue state)")
    print("  - Actual network latency (from network_maps)")
    
    # =========================================================================
    # 6. COLD START ANALYSIS
    # =========================================================================
    print("\n[6] COLD START SIMULATION")
    print("-" * 40)
    
    cold_starts = [tr for tr in task_results if tr['coldStarted']]
    cache_hits = [tr for tr in task_results if tr['cacheHit']]
    
    print(f"Cold starts: {len(cold_starts)}/{len(task_results)}")
    print(f"Cache hits: {len(cache_hits)}/{len(task_results)}")
    
    if cold_starts:
        print("\nCold start breakdown:")
        for tr in cold_starts:
            print(f"  Task {tr['taskId']}: cold_start={tr['coldStartTime']:.4f}s")
    else:
        print("\nAll tasks used warm replicas (no cold starts)")
        print("This is expected when preinit.client_percentage and preinit.server_percentage are high")
    
    # Cold start times from task types
    print("\nCold start durations (from task-types.json):")
    for tr in task_results[:1]:  # Just show once
        tt = tr['taskType']
        print(f"  {tt['name']}:")
        for plat, duration in tt['coldStartDuration'].items():
            print(f"    {plat}: {duration:.3f}s")
    
    # =========================================================================
    # 7. QUEUE DISTRIBUTION AT SCHEDULING TIME
    # =========================================================================
    print("\n[7] QUEUE DISTRIBUTION AT SCHEDULING TIME")
    print("-" * 40)
    
    for tr in task_results:
        qs = tr.get('queueSnapshotAtScheduling', {})
        non_zero = {k: v for k, v in qs.items() if v > 0}
        print(f"Task {tr['taskId']} ({tr['taskType']['name']}) from {tr['sourceNode']}:")
        print(f"  Queue snapshot: {len(qs)} replicas visible, {len(non_zero)} with tasks")
        if non_zero:
            for k, v in sorted(non_zero.items()):
                print(f"    {k}: {v}")
    
    # =========================================================================
    # 8. RECOMMENDATIONS FOR GNN TRAINING
    # =========================================================================
    print("\n[8] RECOMMENDATIONS FOR GNN TRAINING")
    print("-" * 40)
    
    print("""
1. GRAPH STRUCTURE:
   - Nodes: Tasks (with features) + Platforms (with features)
   - Edges: Task -> feasible platforms (filtered by network connectivity and replica)
   
2. TASK FEATURES:
   - Task type one-hot: [is_dnn1, is_dnn2]
   - Source node ID (normalized)
   - (Optional) Task deadline/QoS
   
3. PLATFORM FEATURES:
   - Platform type one-hot: [rpiCpu, xavierCpu, xavierGpu, xavierDla, pynqFpga]
   - Is warm replica for task type: [has_dnn1, has_dnn2]
   - Current queue length (normalized)
   - Is local to source node: [0 or 1]
   
4. EDGE FEATURES:
   - Execution time for task on this platform
   - Network latency (0 if local, actual latency if offload)
   - Is warm replica (1 if initialized, 0 otherwise)
   
5. TARGET:
   - Index of optimal platform in feasible platforms list
   - OR RTT for regression + comparison to hash table
   
6. KEY INSIGHT:
   - Brute force gives ground truth: best placement for each scenario
   - 31,236 placements evaluated for 5 tasks in ~100 seconds
   - ~6,247 placements/second throughput
   - This scales well for training data generation
""")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        dataset_dir = Path(sys.argv[1])
    else:
        # Default path
        dataset_dir = Path('/root/projects/my-herosim/simulation_data/gnn_datasets_analysis/ds_analysis_00000')
    
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    analyze_dataset(dataset_dir)
