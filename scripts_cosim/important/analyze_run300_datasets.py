#!/usr/bin/env python3
"""
Analyze run300 GNN datasets for:
- RTT distributions
- Queue distributions
- Network topologies
- Config uniqueness
- GNN training readiness
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set
import hashlib

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration from generate_gnn_datasets_fast.py
CONNECTION_PROBABILITIES = [
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
    0.25, 0.20
]

REPLICA_CONFIGS = [
    (3, 3, 0.0, 0.0), (2, 3, 0.0, 0.0), (1, 3, 0.0, 0.0), (2, 2, 0.0, 0.0), (1, 2, 0.0, 0.0),
    (3, 3, 0.3, 0.5), (2, 3, 0.4, 0.5), (2, 1, 0.5, 0.4),
    (2, 2, 0.5, 0.8), (3, 3, 0.5, 0.7), (2, 3, 0.6, 0.8), (1, 3, 0.3, 0.7),
    (2, 2, 0.6, 0.8), (1, 4, 0.6, 0.6), (2, 4, 0.8, 0.7), (3, 2, 0.7, 0.8),
    (1, 2, 0.5, 0.9), (3, 1, 0.6, 0.8), (1, 1, 0.4, 0.8),
]

QUEUE_DISTRIBUTIONS = [
    ("zero", "constant", 0, 0, 0, 0, 0),
    ("pois2", "poisson", 2, 0, 0, 8, 1),
    ("pois4", "poisson", 4, 0, 0, 12, 1),
    ("pois6", "poisson", 6, 0, 0, 16, 1),
    ("pois8", "poisson", 8, 0, 0, 20, 1),
    ("norm8", "normal", 8, 3, 0, 20, 1),
    ("norm12", "normal", 12, 4, 0, 24, 1),
    ("pois12", "poisson", 12, 0, 0, 24, 1),
    ("norm16", "normal", 16, 5, 0, 32, 1),
]

SEEDS = [101]
NUM_WORKLOAD_TEMPLATES = 10

# Paths
BASE_DIR = Path("/root/projects/my-herosim/simulation_data/artifacts/run300/gnn_datasets")

# Calculate total expected combinations
# Formula: CONNECTION_PROBABILITIES × REPLICA_CONFIGS × SEEDS × QUEUE_DISTRIBUTIONS
# Note: NUM_WORKLOAD_TEMPLATES cycles and does NOT multiply the total
TOTAL_EXPECTED_COMBINATIONS = (
    len(CONNECTION_PROBABILITIES) * 
    len(REPLICA_CONFIGS) * 
    len(SEEDS) * 
    len(QUEUE_DISTRIBUTIONS)
)

# Breakdown for display
COMBINATION_BREAKDOWN = {
    'connection_probabilities': len(CONNECTION_PROBABILITIES),
    'replica_configs': len(REPLICA_CONFIGS),
    'seeds': len(SEEDS),
    'queue_distributions': len(QUEUE_DISTRIBUTIONS),
    'workload_templates': NUM_WORKLOAD_TEMPLATES,  # Note: cycles, not multiplied
}


def extract_infrastructure_info(infra_path: Path) -> Dict[str, Any]:
    """Extract infrastructure info from infrastructure.json."""
    with open(infra_path, 'r') as f:
        infra = json.load(f)
    
    # Extract from network_maps (actual topology)
    network_maps = infra.get('network_maps', {})
    network_edges = 0
    client_nodes = 0
    server_nodes = set()
    
    # Count client nodes and their connections
    for node_name, connections in network_maps.items():
        if node_name.startswith('client_node'):
            client_nodes += 1
            if isinstance(connections, dict):
                network_edges += len(connections)
                # Track which server nodes are connected
                server_nodes.update(connections.keys())
        elif not node_name.startswith('client_node'):
            server_nodes.add(node_name)
    
    # Also extract from nodes if available (fallback)
    nodes = infra.get('nodes', [])
    if nodes and not network_maps:
        for node in nodes:
            if node.get('node_name', '').startswith('client_node'):
                client_nodes += 1
            else:
                server_nodes.add(node.get('node_name', ''))
            network_map = node.get('network_map', {})
            if isinstance(network_map, dict):
                network_edges += len(network_map)
    
    return {
        'num_nodes': len(network_maps) if network_maps else len(nodes),
        'num_client_nodes': client_nodes,
        'num_server_nodes': len(server_nodes),
        'network_edges': network_edges,
        'network_maps': network_maps,
        'replica_placements': infra.get('replica_placements', {}),
        'queue_distributions': infra.get('queue_distributions', {}),
        'metadata': infra.get('metadata', {}),
    }


def extract_best_rtt(best_path: Path) -> float:
    """Extract best RTT."""
    with open(best_path, 'r') as f:
        best = json.load(f)
    return best.get('rtt', float('nan'))


def extract_queue_data(infra_path: Path, system_state_path: Path = None) -> Dict[str, Any]:
    """Extract queue data from infrastructure.json or system_state_captured_unique.json."""
    queue_data = {}
    
    # Try infrastructure.json first
    with open(infra_path, 'r') as f:
        infra = json.load(f)
    queue_distributions = infra.get('queue_distributions', {})
    
    if queue_distributions:
        # Merge queue distributions across task types
        merged_queues = {}
        for task_type, queues in queue_distributions.items():
            for key, queue_length in queues.items():
                if key not in merged_queues:
                    merged_queues[key] = int(queue_length)
                else:
                    merged_queues[key] = max(merged_queues[key], int(queue_length))
        
        queue_data['queue_snapshot'] = merged_queues
        queue_data['total_queue_length'] = sum(merged_queues.values())
        queue_data['max_queue_length'] = max(merged_queues.values()) if merged_queues else 0
        queue_data['num_platforms_with_queues'] = sum(1 for v in merged_queues.values() if v > 0)
    
    # Try system_state_captured_unique.json if available
    if system_state_path and system_state_path.exists():
        try:
            with open(system_state_path, 'r') as f:
                state = json.load(f)
            # This has more detailed queue info
            task_placements = state.get('task_placements', [])
            if task_placements:
                full_queue_snapshot = task_placements[0].get('full_queue_snapshot', {})
                if full_queue_snapshot:
                    queue_data['queue_snapshot_detailed'] = full_queue_snapshot
        except Exception:
            pass
    
    return queue_data


def get_config_hash(infra_path: Path, optimal_result_path: Path = None) -> str:
    """Generate hash of actual config to check uniqueness.
    
    Uses infrastructure.json (network_maps, replica_placements, queue_distributions, metadata)
    and optimal_result.json (config.infrastructure.nodes) to create a unique signature.
    """
    try:
        with open(infra_path, 'r') as f:
            infra = json.load(f)
    except Exception:
        return None
    
    # Extract actual config from infrastructure.json
    network_maps = infra.get('network_maps', {})
    replica_placements = infra.get('replica_placements', {})
    queue_distributions = infra.get('queue_distributions', {})
    metadata = infra.get('metadata', {})
    seed = metadata.get('seed')
    
    # Create network topology signature (sorted by client node, count connections)
    network_signature = {}
    for node_name, connections in sorted(network_maps.items()):
        if node_name.startswith('client_node'):
            if isinstance(connections, dict):
                # Count unique server connections
                server_connections = sorted(connections.keys())
                network_signature[node_name] = len(server_connections)
    
    # Create replica placement signature (count replicas per node/platform type)
    replica_signature = {}
    for task_type, replicas in replica_placements.items():
        if isinstance(replicas, list):
            node_type_counts = defaultdict(int)
            for replica in replicas:
                if isinstance(replica, dict):
                    node_name = replica.get('node_name', '')
                    plat_type = replica.get('platform_type', '')
                    key = f"{node_name}:{plat_type}"
                    node_type_counts[key] += 1
            replica_signature[task_type] = dict(sorted(node_type_counts.items()))
    
    # Create queue distribution signature (sum of queue lengths)
    queue_signature = {}
    for task_type, queues in queue_distributions.items():
        if isinstance(queues, dict):
            total_queues = sum(int(v) for v in queues.values())
            non_zero = sum(1 for v in queues.values() if int(v) > 0)
            queue_signature[task_type] = {'total': total_queues, 'non_zero': non_zero}
    
    # Build config key for hashing
    config_key = {
        'network_signature': network_signature,
        'replica_signature': replica_signature,
        'queue_signature': queue_signature,
        'seed': seed,
    }
    
    config_str = json.dumps(config_key, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def check_gnn_training_requirements(dataset_dir: Path) -> Dict[str, bool]:
    """Check if dataset has all requirements for GNN training."""
    required_files = {
        'optimal_result.json': dataset_dir / 'optimal_result.json',
        'best.json': dataset_dir / 'best.json',
        'infrastructure.json': dataset_dir / 'infrastructure.json',
        'placements.jsonl': dataset_dir / 'placements' / 'placements.jsonl',
        'workload.json': dataset_dir / 'workload.json',
    }
    
    optional_files = {
        'system_state_captured_unique.json': dataset_dir / 'system_state_captured_unique.json',
    }
    
    checks = {}
    for name, path in required_files.items():
        checks[name] = path.exists()
    
    for name, path in optional_files.items():
        checks[name] = path.exists()
    
    return checks


def count_placements(placements_path: Path) -> int:
    """Count number of placements in JSONL file."""
    if not placements_path.exists():
        return 0
    
    count = 0
    try:
        with open(placements_path, 'r') as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception:
        pass
    return count


def analyze_dataset(dataset_dir: Path) -> Dict[str, Any]:
    """Analyze a single dataset."""
    dataset_id = dataset_dir.name
    
    result = {
        'dataset_id': dataset_id,
        'exists': True,
        'config_hash': None,
        'rtt': None,
        'num_placements': 0,
        'network_info': {},
        'queue_info': {},
        'gnn_requirements': {},
        'config': None,
    }
    
    # Check if dataset exists
    if not dataset_dir.exists():
        result['exists'] = False
        return result
    
    # Extract config hash from infrastructure.json and optimal_result.json
    infra_path = dataset_dir / 'infrastructure.json'
    optimal_result_path = dataset_dir / 'optimal_result.json'
    if infra_path.exists():
        result['config_hash'] = get_config_hash(infra_path, optimal_result_path if optimal_result_path.exists() else None)
    
    # Extract RTT
    best_path = dataset_dir / 'best.json'
    if best_path.exists():
        result['rtt'] = extract_best_rtt(best_path)
    
    # Count placements
    placements_path = dataset_dir / 'placements' / 'placements.jsonl'
    result['num_placements'] = count_placements(placements_path)
    
    # Extract infrastructure info
    infra_path = dataset_dir / 'infrastructure.json'
    if infra_path.exists():
        result['network_info'] = extract_infrastructure_info(infra_path)
        
        # Extract queue info
        system_state_path = dataset_dir / 'system_state_captured_unique.json'
        result['queue_info'] = extract_queue_data(infra_path, system_state_path)
    
    # Check GNN training requirements
    result['gnn_requirements'] = check_gnn_training_requirements(dataset_dir)
    
    return result


def main():
    print("="*80)
    print("RUN300 Dataset Analysis")
    print("="*80)
    print()
    
    if not BASE_DIR.exists():
        print(f"ERROR: Base directory not found: {BASE_DIR}")
        sys.exit(1)
    
    # Find all dataset directories
    dataset_dirs = sorted(BASE_DIR.glob("ds_*"))
    print(f"Found {len(dataset_dirs)} datasets in {BASE_DIR}")
    print()
    
    # Show combination breakdown
    print("Expected Total Combinations Breakdown:")
    print(f"  CONNECTION_PROBABILITIES: {COMBINATION_BREAKDOWN['connection_probabilities']} values")
    print(f"    Values: {CONNECTION_PROBABILITIES}")
    print(f"  REPLICA_CONFIGS: {COMBINATION_BREAKDOWN['replica_configs']} configs")
    print(f"    Configs: {COMBINATION_BREAKDOWN['replica_configs']} tuples (per_client, per_server, client_preinit%, server_preinit%)")
    print(f"  QUEUE_DISTRIBUTIONS: {COMBINATION_BREAKDOWN['queue_distributions']} distributions")
    print(f"    Names: {[q[0] for q in QUEUE_DISTRIBUTIONS]}")
    print(f"  SEEDS: {COMBINATION_BREAKDOWN['seeds']} value")
    print(f"    Values: {SEEDS}")
    print(f"  NUM_WORKLOAD_TEMPLATES: {COMBINATION_BREAKDOWN['workload_templates']} (cycles, NOT multiplied)")
    print()
    print("Calculation:")
    print(f"  Total = CONNECTION_PROBABILITIES × REPLICA_CONFIGS × SEEDS × QUEUE_DISTRIBUTIONS")
    print(f"  Total = {COMBINATION_BREAKDOWN['connection_probabilities']} × {COMBINATION_BREAKDOWN['replica_configs']} × {COMBINATION_BREAKDOWN['seeds']} × {COMBINATION_BREAKDOWN['queue_distributions']}")
    print(f"  Total = {TOTAL_EXPECTED_COMBINATIONS}")
    print()
    print(f"Coverage: {len(dataset_dirs)}/{TOTAL_EXPECTED_COMBINATIONS} datasets ({100*len(dataset_dirs)/TOTAL_EXPECTED_COMBINATIONS:.1f}%)")
    print()
    
    # Analyze all datasets
    print("Analyzing datasets...")
    analyses = []
    for dataset_dir in tqdm(dataset_dirs, desc="Processing"):
        analysis = analyze_dataset(dataset_dir)
        analyses.append(analysis)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(analyses)
    
    # 1. RTT Analysis
    print("\n" + "="*80)
    print("1. RTT DISTRIBUTION")
    print("="*80)
    rtts = df['rtt'].dropna()
    if len(rtts) > 0:
        print(f"Datasets with RTT: {len(rtts)}/{len(df)}")
        print(f"Mean RTT: {rtts.mean():.3f}s")
        print(f"Median RTT: {rtts.median():.3f}s")
        print(f"Min RTT: {rtts.min():.3f}s")
        print(f"Max RTT: {rtts.max():.3f}s")
        print(f"Std RTT: {rtts.std():.3f}s")
        print(f"\nRTT Percentiles:")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            print(f"  {p}th percentile: {rtts.quantile(p/100):.3f}s")
    else:
        print("No RTT data found")
    
    # 2. Placement Count Analysis
    print("\n" + "="*80)
    print("2. PLACEMENT COUNT DISTRIBUTION")
    print("="*80)
    placements = df['num_placements']
    valid_placements = placements[placements > 0]
    if len(valid_placements) > 0:
        print(f"Datasets with placements: {len(valid_placements)}/{len(df)}")
        print(f"Mean placements: {valid_placements.mean():.0f}")
        print(f"Median placements: {valid_placements.median():.0f}")
        print(f"Min placements: {valid_placements.min():.0f}")
        print(f"Max placements: {valid_placements.max():.0f}")
        print(f"\nPlacement Percentiles:")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            print(f"  {p}th percentile: {valid_placements.quantile(p/100):.0f}")
    else:
        print("No placement data found")
    
    # 3. Network Topology Analysis
    print("\n" + "="*80)
    print("3. NETWORK TOPOLOGY ANALYSIS")
    print("="*80)
    network_infos = df['network_info'].dropna()
    if len(network_infos) > 0:
        num_nodes = [ni.get('num_nodes', 0) for ni in network_infos if isinstance(ni, dict)]
        network_edges = [ni.get('network_edges', 0) for ni in network_infos if isinstance(ni, dict)]
        client_nodes = [ni.get('num_client_nodes', 0) for ni in network_infos if isinstance(ni, dict)]
        server_nodes = [ni.get('num_server_nodes', 0) for ni in network_infos if isinstance(ni, dict)]
        
        if num_nodes:
            print(f"Datasets with network info: {len(network_infos)}/{len(df)}")
            print(f"Nodes per dataset: mean={np.mean(num_nodes):.1f}, min={min(num_nodes)}, max={max(num_nodes)}")
            print(f"Client nodes per dataset: mean={np.mean(client_nodes):.1f}")
            print(f"Server nodes per dataset: mean={np.mean(server_nodes):.1f}")
            print(f"Network edges per dataset: mean={np.mean(network_edges):.1f}, min={min(network_edges)}, max={max(network_edges)}")
        else:
            print("No valid network info found")
    
    # 4. Queue Analysis
    print("\n" + "="*80)
    print("4. QUEUE DISTRIBUTION ANALYSIS")
    print("="*80)
    queue_infos = df['queue_info'].dropna()
    if len(queue_infos) > 0:
        total_queue_lengths = [qi.get('total_queue_length', 0) for qi in queue_infos]
        max_queue_lengths = [qi.get('max_queue_length', 0) for qi in queue_infos]
        num_platforms_with_queues = [qi.get('num_platforms_with_queues', 0) for qi in queue_infos]
        
        print(f"Datasets with queue info: {len(queue_infos)}/{len(df)}")
        print(f"Total queue length per dataset: mean={np.mean(total_queue_lengths):.1f}, max={max(total_queue_lengths)}")
        print(f"Max queue length per dataset: mean={np.mean(max_queue_lengths):.1f}, max={max(max_queue_lengths)}")
        print(f"Platforms with queues per dataset: mean={np.mean(num_platforms_with_queues):.1f}")
    
    # 5. Config Uniqueness Check
    print("\n" + "="*80)
    print("5. CONFIG UNIQUENESS CHECK")
    print("="*80)
    config_hashes = df['config_hash'].dropna()
    unique_hashes = set(config_hashes)
    print(f"Datasets with config: {len(config_hashes)}/{len(df)}")
    print(f"Unique config hashes: {len(unique_hashes)}")
    print(f"Duplicate configs: {len(config_hashes) - len(unique_hashes)}")
    
    # Find duplicates
    hash_counts = config_hashes.value_counts()
    duplicates = hash_counts[hash_counts > 1]
    if len(duplicates) > 0:
        print(f"\nDuplicate config hashes found:")
        for hash_val, count in duplicates.head(10).items():
            print(f"  Hash {hash_val}: {count} datasets")
            dup_datasets = df[df['config_hash'] == hash_val]['dataset_id'].tolist()
            print(f"    Datasets: {', '.join(dup_datasets[:5])}{'...' if len(dup_datasets) > 5 else ''}")
    else:
        print("✓ All configs are unique!")
    
    # 6. GNN Training Requirements
    print("\n" + "="*80)
    print("6. GNN TRAINING REQUIREMENTS CHECK")
    print("="*80)
    
    required_files = [
        'optimal_result.json', 'best.json', 'infrastructure.json',
        'placements.jsonl', 'workload.json'
    ]
    
    optional_files = ['system_state_captured_unique.json']
    
    for file_name in required_files + optional_files:
        has_file = df['gnn_requirements'].apply(lambda x: x.get(file_name, False) if isinstance(x, dict) else False)
        count = has_file.sum()
        status = "✓" if file_name in optional_files or count == len(df) else "✗"
        print(f"{status} {file_name}: {count}/{len(df)} ({100*count/len(df):.1f}%)")
    
    # Check complete datasets
    complete = df['gnn_requirements'].apply(
        lambda x: all(x.get(f, False) for f in required_files) if isinstance(x, dict) else False
    )
    print(f"\nComplete datasets (all required files): {complete.sum()}/{len(df)} ({100*complete.sum()/len(df):.1f}%)")
    
    # 7. Config Parameter Distribution
    print("\n" + "="*80)
    print("7. CONFIG PARAMETER DISTRIBUTION")
    print("="*80)
    
    # Extract from infrastructure.json metadata and signatures
    network_infos = df['network_info'].dropna()
    if len(network_infos) > 0:
        seeds = []
        network_connectivity = []  # Average connections per client node
        replica_counts = defaultdict(int)
        queue_total_counts = []
        
        for ni in network_infos:
            if not isinstance(ni, dict):
                continue
            
            metadata = ni.get('metadata', {})
            if metadata:
                seed = metadata.get('seed')
                if seed is not None:
                    seeds.append(seed)
            
            network_maps = ni.get('network_maps', {})
            if network_maps:
                # Calculate average connectivity per client node
                client_conn_counts = []
                for node_name, connections in network_maps.items():
                    if node_name.startswith('client_node') and isinstance(connections, dict):
                        client_conn_counts.append(len(connections))
                if client_conn_counts:
                    network_connectivity.append(np.mean(client_conn_counts))
            
            replica_placements = ni.get('replica_placements', {})
            if replica_placements:
                # Count total replicas per task type
                for task_type, replicas in replica_placements.items():
                    if isinstance(replicas, list):
                        replica_counts[f"{task_type}:{len(replicas)}"] += 1
            
            queue_distributions = ni.get('queue_distributions', {})
            if queue_distributions:
                total = 0
                for task_type, queues in queue_distributions.items():
                    if isinstance(queues, dict):
                        total += sum(int(v) for v in queues.values())
                queue_total_counts.append(total)
        
        if seeds:
            print(f"Seeds:")
            seed_counts = pd.Series(seeds).value_counts().sort_index()
            for seed, count in seed_counts.items():
                print(f"  {seed}: {count} datasets")
        
        if network_connectivity:
            print(f"\nNetwork connectivity (avg connections per client node):")
            print(f"  Mean: {np.mean(network_connectivity):.2f}")
            print(f"  Min: {min(network_connectivity):.2f}")
            print(f"  Max: {max(network_connectivity):.2f}")
            print(f"  Std: {np.std(network_connectivity):.2f}")
        
        if replica_counts:
            print(f"\nReplica configurations (task_type:count):")
            for rc, count in sorted(replica_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {rc}: {count} datasets")
        
        if queue_total_counts:
            print(f"\nQueue distributions (total queue length):")
            print(f"  Mean: {np.mean(queue_total_counts):.1f}")
            print(f"  Min: {min(queue_total_counts)}")
            print(f"  Max: {max(queue_total_counts)}")
            print(f"  Zero queues: {sum(1 for q in queue_total_counts if q == 0)} datasets")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total datasets found: {len(df)}")
    print(f"Expected total: {TOTAL_EXPECTED_COMBINATIONS}")
    print(f"Completion: {100*len(df)/TOTAL_EXPECTED_COMBINATIONS:.1f}%")
    print(f"Unique configs: {len(unique_hashes)}")
    print(f"Complete datasets: {complete.sum()}")


if __name__ == "__main__":
    main()
