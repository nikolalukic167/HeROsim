#!/usr/bin/env python3
"""
Verify that each dataset directory contains the expected configuration
based on the grid search order in generate_gnn_datasets_fast.py.

Checks if ds_XXXXX matches the config combination it should have at index XXXXX.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import hashlib

import numpy as np
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration from generate_gnn_datasets_fast.py (same as analysis script)
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


def generate_expected_configs() -> List[Dict[str, Any]]:
    """
    Generate the expected configuration for each dataset index
    based on the grid search order in generate_gnn_datasets_fast.py.
    
    Order: for conn_prob in CONNECTION_PROBABILITIES:
             for replica_cfg in REPLICA_CONFIGS:
               for seed in SEEDS:
                 for queue_dist in QUEUE_DISTRIBUTIONS:
                   dataset_idx += 1
    """
    expected_configs = []
    
    for conn_prob in CONNECTION_PROBABILITIES:
        for replica_cfg in REPLICA_CONFIGS:
            for seed in SEEDS:
                for queue_dist in QUEUE_DISTRIBUTIONS:
                    per_client, per_server, client_preinit_pct, server_preinit_pct = replica_cfg
                    qname, qtype, qp1, qp2, qmin, qmax, qstep = queue_dist
                    
                    expected_configs.append({
                        'connection_probability': conn_prob,
                        'replica_per_client': per_client,
                        'replica_per_server': per_server,
                        'client_preinit_pct': client_preinit_pct,
                        'server_preinit_pct': server_preinit_pct,
                        'seed': seed,
                        'queue_distribution_name': qname,
                        'queue_distribution_type': qtype,
                        'queue_params': (qp1, qp2, qmin, qmax, qstep),
                    })
    
    return expected_configs


def extract_actual_config_from_space(space_path: Path) -> Optional[Dict[str, Any]]:
    """Extract actual config from space_with_network.json."""
    try:
        with open(space_path, 'r') as f:
            config = json.load(f)
    except Exception:
        return None
    
    # Extract from space_with_network.json structure
    network = config.get('network', {})
    topology = network.get('topology', {})
    connection_probability = topology.get('connection_probability')
    seed = topology.get('seed')
    
    replicas = config.get('replicas', {})
    dnn1_replicas = replicas.get('dnn1', {})
    replica_per_client = dnn1_replicas.get('per_client')
    replica_per_server = dnn1_replicas.get('per_server')
    
    preinit = config.get('preinit', {})
    client_preinit_pct = preinit.get('client_percentage')
    server_preinit_pct = preinit.get('server_percentage')
    
    prewarm = config.get('prewarm', {})
    dnn1_prewarm = prewarm.get('dnn1', {})
    queue_params = dnn1_prewarm.get('queue_distribution_params', {})
    queue_type = queue_params.get('type', 'unknown')
    
    # Map queue params to queue distribution name
    queue_name = None
    if queue_type == 'constant':
        if queue_params.get('value', 0) == 0:
            queue_name = 'zero'
    elif queue_type == 'poisson':
        lam = queue_params.get('lambda', 0)
        if lam == 2:
            queue_name = 'pois2'
        elif lam == 4:
            queue_name = 'pois4'
        elif lam == 6:
            queue_name = 'pois6'
        elif lam == 8:
            queue_name = 'pois8'
        elif lam == 12:
            queue_name = 'pois12'
    elif queue_type == 'normal':
        mean = queue_params.get('mean', 0)
        stddev = queue_params.get('stddev', 0)
        if mean == 8 and stddev == 3:
            queue_name = 'norm8'
        elif mean == 12 and stddev == 4:
            queue_name = 'norm12'
        elif mean == 16 and stddev == 5:
            queue_name = 'norm16'
    
    return {
        'seed': seed,
        'connection_probability': connection_probability,
        'replica_per_client': replica_per_client,
        'replica_per_server': replica_per_server,
        'client_preinit_pct': client_preinit_pct,
        'server_preinit_pct': server_preinit_pct,
        'queue_distribution_name': queue_name,
        'queue_distribution_type': queue_type,
        'queue_params': queue_params,
    }


def extract_actual_config_from_optimal_result(optimal_result_path: Path) -> Optional[Dict[str, Any]]:
    """Extract config from optimal_result.json for additional verification."""
    try:
        with open(optimal_result_path, 'r') as f:
            result = json.load(f)
    except Exception:
        return None
    
    # Extract from config.infrastructure.nodes
    config = result.get('config', {})
    infra_config = config.get('infrastructure', {})
    nodes = infra_config.get('nodes', [])
    
    # Count client/server nodes
    client_nodes = 0
    server_nodes = 0
    network_connections = []
    
    for node in nodes:
        node_name = node.get('node_name', '')
        if node_name.startswith('client_node'):
            client_nodes += 1
            network_map = node.get('network_map', {})
            if isinstance(network_map, dict):
                network_connections.append(len(network_map))
        else:
            server_nodes += 1
    
    avg_connectivity = np.mean(network_connections) if network_connections else 0.0
    
    return {
        'num_client_nodes': client_nodes,
        'num_server_nodes': server_nodes,
        'avg_network_connectivity': avg_connectivity,
    }


def compare_configs(expected: Dict[str, Any], actual: Dict[str, Any], dataset_id: str) -> Tuple[bool, List[str]]:
    """Compare expected vs actual config and return (match, issues)."""
    issues = []
    match = True
    
    # Check seed
    exp_seed = expected.get('seed')
    act_seed = actual.get('seed')
    if exp_seed != act_seed:
        issues.append(f"Seed mismatch: expected {exp_seed}, got {act_seed}")
        match = False
    
    # Check connection probability
    exp_conn_prob = expected.get('connection_probability')
    act_conn_prob = actual.get('connection_probability')
    if exp_conn_prob is not None and act_conn_prob is not None:
        # Allow small floating point differences
        if abs(exp_conn_prob - act_conn_prob) > 0.001:
            issues.append(f"Connection probability mismatch: expected {exp_conn_prob}, got {act_conn_prob}")
            match = False
    
    # Check replica config
    exp_per_client = expected.get('replica_per_client')
    act_per_client = actual.get('replica_per_client')
    if exp_per_client is not None and act_per_client is not None and exp_per_client != act_per_client:
        issues.append(f"Replica per_client mismatch: expected {exp_per_client}, got {act_per_client}")
        match = False
    
    exp_per_server = expected.get('replica_per_server')
    act_per_server = actual.get('replica_per_server')
    if exp_per_server is not None and act_per_server is not None and exp_per_server != act_per_server:
        issues.append(f"Replica per_server mismatch: expected {exp_per_server}, got {act_per_server}")
        match = False
    
    # Check preinit percentages
    exp_client_preinit = expected.get('client_preinit_pct')
    act_client_preinit = actual.get('client_preinit_pct')
    if exp_client_preinit is not None and act_client_preinit is not None:
        if abs(exp_client_preinit - act_client_preinit) > 0.001:
            issues.append(f"Client preinit % mismatch: expected {exp_client_preinit}, got {act_client_preinit}")
            match = False
    
    exp_server_preinit = expected.get('server_preinit_pct')
    act_server_preinit = actual.get('server_preinit_pct')
    if exp_server_preinit is not None and act_server_preinit is not None:
        if abs(exp_server_preinit - act_server_preinit) > 0.001:
            issues.append(f"Server preinit % mismatch: expected {exp_server_preinit}, got {act_server_preinit}")
            match = False
    
    # Check queue distribution
    exp_queue_name = expected.get('queue_distribution_name')
    act_queue_name = actual.get('queue_distribution_name')
    if exp_queue_name != act_queue_name:
        issues.append(f"Queue distribution mismatch: expected {exp_queue_name}, got {act_queue_name}")
        match = False
    
    return match, issues


def verify_dataset(dataset_dir: Path, dataset_idx: int, expected_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify a single dataset matches its expected config."""
    dataset_id = dataset_dir.name
    
    result = {
        'dataset_id': dataset_id,
        'dataset_idx': dataset_idx,
        'exists': False,
        'config_match': False,
        'issues': [],
        'expected_config': None,
        'actual_config': None,
    }
    
    if not dataset_dir.exists():
        result['issues'].append("Dataset directory does not exist")
        return result
    
    result['exists'] = True
    
    # Get expected config for this index
    if dataset_idx >= len(expected_configs):
        result['issues'].append(f"Dataset index {dataset_idx} exceeds expected configs ({len(expected_configs)})")
        return result
    
    expected_config = expected_configs[dataset_idx]
    result['expected_config'] = expected_config
    
    # Extract actual config from space_with_network.json (contains the actual config used)
    space_path = dataset_dir / 'space_with_network.json'
    if not space_path.exists():
        result['issues'].append("space_with_network.json not found")
        return result
    
    actual_config = extract_actual_config_from_space(space_path)
    if actual_config is None:
        result['issues'].append("Failed to extract config from space_with_network.json")
        return result
    
    result['actual_config'] = actual_config
    
    # Compare configs
    match, issues = compare_configs(expected_config, actual_config, dataset_id)
    result['config_match'] = match
    result['issues'].extend(issues)
    
    return result


def main():
    print("="*80)
    print("Dataset Configuration Verification")
    print("="*80)
    print()
    
    if not BASE_DIR.exists():
        print(f"ERROR: Base directory not found: {BASE_DIR}")
        sys.exit(1)
    
    # Generate expected configurations
    print("Generating expected configurations...")
    expected_configs = generate_expected_configs()
    print(f"Generated {len(expected_configs)} expected configurations")
    print()
    
    # Find all dataset directories
    dataset_dirs = sorted(BASE_DIR.glob("ds_*"))
    print(f"Found {len(dataset_dirs)} datasets in {BASE_DIR}")
    print()
    
    # Verify each dataset
    print("Verifying datasets...")
    verification_results = []
    
    for dataset_dir in tqdm(dataset_dirs, desc="Verifying"):
        # Extract dataset index from directory name (e.g., ds_00042 -> 42)
        dataset_id = dataset_dir.name
        try:
            dataset_idx = int(dataset_id.split('_')[1])
        except (ValueError, IndexError):
            verification_results.append({
                'dataset_id': dataset_id,
                'dataset_idx': -1,
                'exists': False,
                'config_match': False,
                'issues': [f"Invalid dataset ID format: {dataset_id}"],
            })
            continue
        
        result = verify_dataset(dataset_dir, dataset_idx, expected_configs)
        verification_results.append(result)
    
    # Analyze results
    print()
    print("="*80)
    print("VERIFICATION RESULTS")
    print("="*80)
    
    # Count matches/mismatches
    matches = sum(1 for r in verification_results if r.get('config_match', False))
    mismatches = sum(1 for r in verification_results if r.get('exists', False) and not r.get('config_match', False))
    missing = sum(1 for r in verification_results if not r.get('exists', False))
    out_of_range = sum(1 for r in verification_results if r.get('dataset_idx', -1) >= len(expected_configs))
    
    print(f"Total datasets: {len(verification_results)}")
    print(f"Config matches: {matches}")
    print(f"Config mismatches: {mismatches}")
    print(f"Missing directories: {missing}")
    print(f"Out of range indices: {out_of_range}")
    print()
    
    # Show mismatches
    if mismatches > 0:
        print("="*80)
        print("CONFIG MISMATCHES")
        print("="*80)
        mismatch_count = 0
        for result in verification_results:
            if result.get('exists', False) and not result.get('config_match', False):
                mismatch_count += 1
                if mismatch_count <= 20:  # Show first 20 mismatches
                    print(f"\n{result['dataset_id']} (index {result['dataset_idx']}):")
                    expected = result.get('expected_config', {})
                    actual = result.get('actual_config', {})
                    
                    print(f"  Expected:")
                    print(f"    Seed: {expected.get('seed')}")
                    print(f"    Connection prob: {expected.get('connection_probability')}")
                    print(f"    Replica: ({expected.get('replica_per_client')}, {expected.get('replica_per_server')})")
                    print(f"    Preinit: ({expected.get('client_preinit_pct')}, {expected.get('server_preinit_pct')})")
                    print(f"    Queue: {expected.get('queue_distribution_name')}")
                    
                    print(f"  Actual:")
                    print(f"    Seed: {actual.get('seed')}")
                    print(f"    Connection prob: {actual.get('connection_probability')}")
                    print(f"    Replica: ({actual.get('replica_per_client')}, {actual.get('replica_per_server')})")
                    print(f"    Preinit: ({actual.get('client_preinit_pct')}, {actual.get('server_preinit_pct')})")
                    print(f"    Queue: {actual.get('queue_distribution_name')}")
                    
                    for issue in result.get('issues', []):
                        print(f"    ⚠ {issue}")
                elif mismatch_count == 21:
                    print(f"\n... (showing first 20 of {mismatches} mismatches)")
        print()
    
    # Summary statistics
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total datasets: {len(verification_results)}")
    print(f"Expected total: {len(expected_configs)}")
    print(f"Config matches: {matches} ({100*matches/len(verification_results):.1f}%)")
    print(f"Config mismatches: {mismatches} ({100*mismatches/len(verification_results):.1f}%)")
    
    if matches == len(verification_results):
        print("\n✓ All datasets match their expected configurations!")
    else:
        print(f"\n⚠ Found {mismatches} datasets with config mismatches")


if __name__ == "__main__":
    main()
