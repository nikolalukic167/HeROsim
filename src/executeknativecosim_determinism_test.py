#!/usr/bin/env python3
"""
Determinism test for knative_network simulation.

Runs the same simulation N times and compares outputs to check for non-determinism.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

from src.motivational.constants import KEEP_ALIVE, QUEUE_LENGTH
from src.placement.executor import execute_sim
from src.placement.model import SimulationData, DataclassJSONEncoder

REQUIRED_SIM_FILES = [
    'application-types.json',
    'platform-types.json',
    'qos-types.json',
    'storage-types.json',
    'task-types.json'
]


def load_simulation_inputs(sim_input_path: Path) -> Dict[str, Any]:
    sim_inputs = {}
    for filename in REQUIRED_SIM_FILES:
        file_path = sim_input_path / filename
        with open(file_path, 'r') as f:
            key = filename.replace('.json', '').replace('-', '_')
            sim_inputs[key] = json.load(f)
    return sim_inputs


def prepare_infrastructure_from_dataset(infrastructure_file: Path, space_config: Dict[str, Any]) -> Dict[str, Any]:
    with open(infrastructure_file, 'r') as f:
        infra_data = json.load(f)

    network_maps = infra_data['network_maps']
    deterministic_replica_placements = infra_data['replica_placements']
    deterministic_queue_distributions = infra_data['queue_distributions']

    client_nodes_count = space_config['nodes']['client_nodes']['count']
    server_nodes_count = space_config['nodes']['server_nodes']['count']
    device_types = list(space_config['pci'].keys())

    nodes = []
    for i in range(client_nodes_count):
        device_type = device_types[i % len(device_types)]
        device_specs = space_config['pci'][device_type]['specs']
        node_config = device_specs.copy()
        node_config['node_name'] = f"client_node{i}"
        node_config['type'] = device_type
        node_config['network_map'] = network_maps.get(node_config['node_name'], {})
        nodes.append(node_config)

    for i in range(server_nodes_count):
        device_type = device_types[i % len(device_types)]
        device_specs = space_config['pci'][device_type]['specs']
        node_config = device_specs.copy()
        node_config['node_name'] = f"node{i}"
        node_config['type'] = device_type
        node_config['network_map'] = network_maps.get(node_config['node_name'], {})
        nodes.append(node_config)

    infrastructure_config = {
        "network": {"bandwidth": 1000.0},
        "nodes": nodes,
        "preinitialize_platforms": True,
        "preinit": space_config.get('preinit', {}),
        "replicas": space_config.get('replicas', {}),
        "scheduler": {'batch_size': 1, 'batch_timeout': 0.1},
        "prewarm": space_config.get('prewarm', {}),
        "deterministic_replica_placements": deterministic_replica_placements,
        "deterministic_queue_distributions": deterministic_queue_distributions,
    }

    preinit_config = space_config.get('preinit', {})
    replicas_config = space_config.get('replicas', {})

    all_client_nodes = [n for n in nodes if n['node_name'].startswith('client_node')]
    all_server_nodes = [n for n in nodes if not n['node_name'].startswith('client_node')]

    preinit_clients = preinit_config.get('clients', [])
    preinit_servers = preinit_config.get('servers', [])

    if not preinit_clients and 'client_percentage' in preinit_config:
        k = max(1, int(len(all_client_nodes) * float(preinit_config.get('client_percentage', 0))))
        preinit_clients = [n['node_name'] for n in all_client_nodes[:k]]

    if not preinit_servers and 'server_percentage' in preinit_config:
        k = max(1, int(len(all_server_nodes) * float(preinit_config.get('server_percentage', 0))))
        preinit_servers = [n['node_name'] for n in all_server_nodes[:k]]

    if preinit_clients == "all":
        preinit_clients = [n['node_name'] for n in all_client_nodes]
    if preinit_servers == "all":
        preinit_servers = [n['node_name'] for n in all_server_nodes]

    replica_plan = {
        'preinit_clients': preinit_clients,
        'preinit_servers': preinit_servers,
        'preinit_task_types': list(replicas_config.keys()) if replicas_config else [],
        'replicas_config': replicas_config,
        'prewarm_config': space_config.get('prewarm', {})
    }

    infrastructure_config['replica_plan'] = replica_plan
    return infrastructure_config


def run_single_simulation(dataset_dir: Path, sim_input_path: Path, run_id: int) -> Dict[str, Any]:
    """Run a single simulation and return key results for comparison."""
    
    infrastructure_file = dataset_dir / "infrastructure.json"
    workload_file = dataset_dir / "workload.json"
    space_config_file = dataset_dir / "space_with_network.json"

    with open(space_config_file, 'r') as f:
        space_config = json.load(f)
    with open(workload_file, 'r') as f:
        workload = json.load(f)

    sim_inputs = load_simulation_inputs(sim_input_path)
    infrastructure_config = prepare_infrastructure_from_dataset(infrastructure_file, space_config)

    full_config = {
        "infrastructure": infrastructure_config,
        "workload": workload,
    }

    simulation_data = SimulationData(
        platform_types=sim_inputs['platform_types'],
        storage_types=sim_inputs['storage_types'],
        qos_types=sim_inputs['qos_types'],
        application_types=sim_inputs['application_types'],
        task_types=sim_inputs['task_types'],
    )

    stats = execute_sim(
        simulation_data,
        full_config['infrastructure'],
        'fifo',
        KEEP_ALIVE,
        'fifo',
        QUEUE_LENGTH,
        'kn_network_batch_kn_network_batch',
        full_config['workload'],
        'workload-knative',
    )

    # Extract key results for comparison
    task_results = stats.get('taskResults', [])
    
    # Get placements and RTTs for real tasks only
    placements = []
    rtts = []
    queue_snapshots = []
    temporal_states = []
    
    for tr in task_results:
        if tr.get('taskId') is not None and tr.get('taskId') >= 0:
            placements.append({
                'task_id': tr['taskId'],
                'node': tr.get('executionNode'),
                'platform': tr.get('executionPlatform'),
            })
            rtts.append(tr.get('elapsedTime', 0))
            queue_snapshots.append(tr.get('queueSnapshotAtScheduling', {}))
            temporal_states.append(tr.get('temporalStateAtScheduling', {}))
    
    total_rtt = sum(rtts)
    
    return {
        'run_id': run_id,
        'total_rtt': total_rtt,
        'placements': placements,
        'queue_snapshots': queue_snapshots,
        'temporal_states': temporal_states,
        'individual_rtts': rtts,
    }


def compare_runs(results: list) -> Dict[str, Any]:
    """Compare multiple runs and report differences."""
    if len(results) < 2:
        return {'error': 'Need at least 2 runs to compare'}
    
    base = results[0]
    comparison = {
        'total_runs': len(results),
        'rtt_values': [r['total_rtt'] for r in results],
        'rtt_all_same': len(set(r['total_rtt'] for r in results)) == 1,
        'placement_differences': [],
        'queue_snapshot_differences': [],
        'temporal_state_differences': [],
        'temporal_state_uniformity_stats': {},
        'individual_rtt_differences': [],
    }
    
    # Compare placements
    for i, r in enumerate(results[1:], 1):
        for j, (p1, p2) in enumerate(zip(base['placements'], r['placements'])):
            if p1 != p2:
                comparison['placement_differences'].append({
                    'run': i,
                    'task_id': j,
                    'base': p1,
                    'current': p2,
                })
    
    # Compare queue snapshots
    for i, r in enumerate(results[1:], 1):
        for j, (q1, q2) in enumerate(zip(base['queue_snapshots'], r['queue_snapshots'])):
            if q1 != q2:
                comparison['queue_snapshot_differences'].append({
                    'run': i,
                    'task_id': j,
                    'base': q1,
                    'current': q2,
                })
    
    # Compare temporal states and compute uniformity statistics
    all_temporal_values = {'current_task_remaining': [], 'cold_start_remaining': [], 'comm_remaining': []}
    for i, r in enumerate(results):
        for j, ts in enumerate(r['temporal_states']):
            # Collect all temporal values for uniformity analysis
            for platform_key, state in ts.items():
                if isinstance(state, dict):
                    all_temporal_values['current_task_remaining'].append(state.get('current_task_remaining', 0.0))
                    all_temporal_values['cold_start_remaining'].append(state.get('cold_start_remaining', 0.0))
                    all_temporal_values['comm_remaining'].append(state.get('comm_remaining', 0.0))
            
            # Compare with base run
            if i > 0:
                base_ts = base['temporal_states'][j]
                if base_ts != ts:
                    comparison['temporal_state_differences'].append({
                        'run': i,
                        'task_id': j,
                        'base': base_ts,
                        'current': ts,
                    })
    
    # Compute uniformity statistics
    import statistics
    for key, values in all_temporal_values.items():
        if values:
            non_zero = [v for v in values if v > 0]
            comparison['temporal_state_uniformity_stats'][key] = {
                'count': len(values),
                'non_zero_count': len(non_zero),
                'mean': statistics.mean(values) if values else 0.0,
                'median': statistics.median(values) if values else 0.0,
                'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
                'min': min(values) if values else 0.0,
                'max': max(values) if values else 0.0,
                'mean_non_zero': statistics.mean(non_zero) if non_zero else 0.0,
            }
    
    # Compare individual RTTs
    for i, r in enumerate(results[1:], 1):
        for j, (rtt1, rtt2) in enumerate(zip(base['individual_rtts'], r['individual_rtts'])):
            if abs(rtt1 - rtt2) > 1e-10:
                comparison['individual_rtt_differences'].append({
                    'run': i,
                    'task_id': j,
                    'base': rtt1,
                    'current': rtt2,
                    'diff': rtt2 - rtt1,
                })
    
    comparison['placements_deterministic'] = len(comparison['placement_differences']) == 0
    comparison['queue_snapshots_deterministic'] = len(comparison['queue_snapshot_differences']) == 0
    comparison['temporal_states_deterministic'] = len(comparison['temporal_state_differences']) == 0
    comparison['rtts_deterministic'] = len(comparison['individual_rtt_differences']) == 0
    
    return comparison


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.executeknativecosim_determinism_test <dataset_dir> [num_runs]")
        print("Example: python -m src.executeknativecosim_determinism_test simulation_data/artifacts/run1650/gnn_datasets/ds_00014 5")
        sys.exit(1)
    
    dataset_dir = Path(sys.argv[1])
    num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    sim_input_path = Path("data/nofs-ids")
    
    # Check if system_state_captured_unique.json exists - if not, complete it first
    captured_state_file = dataset_dir / "system_state_captured_unique.json"
    if not captured_state_file.exists():
        print(f"Warning: {captured_state_file} not found. Running baseline capture first...")
        # Import and run the baseline capture
        from src.executeknativecosim import run_knative_baseline_for_dataset
        import logging
        logger = logging.getLogger('simulation')
        logger.setLevel(logging.WARNING)  # Reduce noise
        
        success = run_knative_baseline_for_dataset(dataset_dir, sim_input_path, logger)
        if not success:
            print(f"ERROR: Failed to capture baseline for {dataset_dir.name}")
            sys.exit(1)
        print("Baseline captured. Proceeding with determinism test...")
        print()
    
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    print(f"=== Determinism Test ===")
    print(f"Dataset: {dataset_dir.name}")
    print(f"Number of runs: {num_runs}")
    print()
    
    results = []
    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}...", end=" ", flush=True)
        result = run_single_simulation(dataset_dir, sim_input_path, i)
        results.append(result)
        print(f"RTT: {result['total_rtt']:.6f}s")
    
    print()
    print("=== Comparison Results ===")
    comparison = compare_runs(results)
    
    print(f"RTT values: {comparison['rtt_values']}")
    print(f"All RTTs same: {comparison['rtt_all_same']}")
    print(f"Placements deterministic: {comparison['placements_deterministic']}")
    print(f"Queue snapshots deterministic: {comparison['queue_snapshots_deterministic']}")
    print(f"Temporal states deterministic: {comparison['temporal_states_deterministic']}")
    print(f"Individual RTTs deterministic: {comparison['rtts_deterministic']}")
    
    # Print temporal state uniformity statistics
    if comparison['temporal_state_uniformity_stats']:
        print(f"\n=== Temporal State Uniformity Statistics ===")
        for key, stats in comparison['temporal_state_uniformity_stats'].items():
            print(f"{key}:")
            print(f"  Total samples: {stats['count']}, Non-zero: {stats['non_zero_count']}")
            print(f"  Mean: {stats['mean']:.6f}, Median: {stats['median']:.6f}, StdDev: {stats['stdev']:.6f}")
            print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
            if stats['non_zero_count'] > 0:
                print(f"  Mean (non-zero): {stats['mean_non_zero']:.6f}")
    
    if not comparison['placements_deterministic']:
        print(f"\nPlacement differences ({len(comparison['placement_differences'])}):")
        for diff in comparison['placement_differences'][:5]:
            print(f"  Run {diff['run']}, Task {diff['task_id']}: {diff['base']} -> {diff['current']}")
    
    if not comparison['queue_snapshots_deterministic']:
        print(f"\nQueue snapshot differences ({len(comparison['queue_snapshot_differences'])}):")
        for diff in comparison['queue_snapshot_differences'][:5]:
            print(f"  Run {diff['run']}, Task {diff['task_id']}:")
            print(f"    Base: {diff['base']}")
            print(f"    Current: {diff['current']}")
    
    if not comparison['temporal_states_deterministic']:
        print(f"\nTemporal state differences ({len(comparison['temporal_state_differences'])}):")
        for diff in comparison['temporal_state_differences'][:3]:
            print(f"  Run {diff['run']}, Task {diff['task_id']}:")
            print(f"    Base: {diff['base']}")
            print(f"    Current: {diff['current']}")
    
    if not comparison['rtts_deterministic']:
        print(f"\nIndividual RTT differences ({len(comparison['individual_rtt_differences'])}):")
        for diff in comparison['individual_rtt_differences'][:5]:
            print(f"  Run {diff['run']}, Task {diff['task_id']}: {diff['base']:.6f} -> {diff['current']:.6f} (diff: {diff['diff']:+.6f})")
    
    # Summary
    print()
    if comparison['placements_deterministic'] and comparison['queue_snapshots_deterministic'] and comparison['temporal_states_deterministic'] and comparison['rtts_deterministic']:
        print("✓ SIMULATION IS DETERMINISTIC")
    else:
        print("✗ SIMULATION IS NOT DETERMINISTIC")
        if not comparison['placements_deterministic']:
            print("  - Placements vary between runs")
        if not comparison['queue_snapshots_deterministic']:
            print("  - Queue snapshots vary between runs")
        if not comparison['temporal_states_deterministic']:
            print("  - Temporal states vary between runs")
        if not comparison['rtts_deterministic']:
            print("  - RTTs vary between runs")


if __name__ == "__main__":
    main()

