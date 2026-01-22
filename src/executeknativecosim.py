"""
Knative Baseline Co-simulation Executor

This script runs the knative_network scheduler on existing datasets (ds_*) and captures
the system state with queue occupancy at scheduling time.

Workflow:
1. Load infrastructure.json and workload.json from an existing ds_* directory
2. Run Phase 1 simulation with knative_network scheduler (captures system state)
3. Save system_state_captured.json to the ds_* directory

The saved system state includes queue_occupancy which can later be used to:
- Compare with brute-force optimal placements
- Analyze knative's placement decisions
"""

import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

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


def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger('simulation')
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.propagate = False
    return logger


def load_simulation_inputs(sim_input_path: Path) -> Dict[str, Any]:
    """Load all required simulation input files."""
    sim_inputs = {}

    missing_files = []
    for filename in REQUIRED_SIM_FILES:
        if not (sim_input_path / filename).exists():
            missing_files.append(filename)

    if missing_files:
        raise FileNotFoundError(
            f"Missing required simulation input files: {', '.join(missing_files)}"
        )

    for filename in REQUIRED_SIM_FILES:
        file_path = sim_input_path / filename
        with open(file_path, 'r') as f:
            key = filename.replace('.json', '').replace('-', '_')
            sim_inputs[key] = json.load(f)

    return sim_inputs


def prepare_infrastructure_from_dataset(
        infrastructure_file: Path,
        space_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prepare infrastructure configuration from a dataset's infrastructure.json file.
    """
    with open(infrastructure_file, 'r') as f:
        infra_data = json.load(f)

    required_keys = ['network_maps', 'replica_placements', 'queue_distributions', 'metadata']
    missing_keys = [k for k in required_keys if k not in infra_data]
    if missing_keys:
        raise RuntimeError(
            f"Infrastructure file {infrastructure_file} is missing required keys: {missing_keys}"
        )

    network_maps = infra_data['network_maps']
    deterministic_replica_placements = infra_data['replica_placements']
    deterministic_queue_distributions = infra_data['queue_distributions']

    # Build nodes from space_config
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
        "network": {"bandwidth": 1000.0},  # Default bandwidth
        "nodes": nodes,
        "preinitialize_platforms": True,
        "preinit": space_config.get('preinit', {}),
        "replicas": space_config.get('replicas', {}),
        "scheduler": {'batch_size': 1, 'batch_timeout': 0.1},
        "prewarm": space_config.get('prewarm', {}),
        "deterministic_replica_placements": deterministic_replica_placements,
        "deterministic_queue_distributions": deterministic_queue_distributions,
    }

    # Build replica_plan
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


def execute_simulation(
        config: Dict[str, Any],
        sim_inputs: Dict[str, Any],
        scheduling_strategy: str,
        cache_policy='fifo',
        task_priority='fifo',
        keep_alive=30,
        queue_length=100,
) -> Dict[str, Any]:
    """Execute simulation with full configuration and simulation inputs."""

    simulation_data = SimulationData(
        platform_types=sim_inputs['platform_types'],
        storage_types=sim_inputs['storage_types'],
        qos_types=sim_inputs['qos_types'],
        application_types=sim_inputs['application_types'],
        task_types=sim_inputs['task_types'],
    )

    stats = execute_sim(
        simulation_data,
        config['infrastructure'],
        cache_policy,
        keep_alive,
        task_priority,
        queue_length,
        scheduling_strategy,
        config['workload'],
        'workload-knative',
    )
    return {
        "status": "success",
        "config": config,
        "sim_inputs": sim_inputs,
        "stats": stats
    }


def run_knative_baseline_for_dataset(
        dataset_dir: Path,
        sim_input_path: Path,
        logger: logging.Logger
) -> bool:
    """
    Run knative_network scheduler on a single dataset and save system state.
    
    Returns True if successful, False if skipped or failed.
    """
    logger.info(f"Processing dataset: {dataset_dir.name}")

    # Check required files exist
    infrastructure_file = dataset_dir / "infrastructure.json"
    workload_file = dataset_dir / "workload.json"
    space_config_file = dataset_dir / "space_with_network.json"

    if not infrastructure_file.exists():
        logger.warning(f"Skipping {dataset_dir.name}: infrastructure.json not found")
        return False

    if not workload_file.exists():
        logger.warning(f"Skipping {dataset_dir.name}: workload.json not found")
        return False

    if not space_config_file.exists():
        logger.warning(f"Skipping {dataset_dir.name}: space_with_network.json not found")
        return False

    try:
        # Load simulation inputs
        sim_inputs = load_simulation_inputs(sim_input_path)

        # Load space config
        with open(space_config_file, 'r') as f:
            space_config = json.load(f)

        # Load workload
        with open(workload_file, 'r') as f:
            workload = json.load(f)

        # Prepare infrastructure
        infrastructure_config = prepare_infrastructure_from_dataset(
            infrastructure_file, space_config
        )

        # Combine into full config
        full_config = {
            "infrastructure": infrastructure_config,
            "workload": workload,
        }

        logger.info(f"Running knative_network_batch simulation for {dataset_dir.name}...")
        print(f"  Running knative_network_batch simulation...")

        # Execute simulation with knative_network_batch scheduler
        result = execute_simulation(
            full_config,
            sim_inputs,
            scheduling_strategy='kn_network_batch_kn_network_batch',
            cache_policy='fifo',
            task_priority='fifo',
            keep_alive=KEEP_ALIVE,
            queue_length=QUEUE_LENGTH,
        )

        # Extract system state results
        stats = result.get('stats', {})
        system_state_results = stats.get('systemStateResults', [])

        if not system_state_results:
            logger.warning(f"{dataset_dir.name}: No systemStateResults found")
            return False

        # Get the last system state (after all tasks scheduled)
        final_state = system_state_results[-1]

        # Extract task results for placement info
        task_results = stats.get('taskResults', [])

        # Build task placements with queue snapshot at scheduling time
        task_placements = []
        for tr in task_results:
            if tr.get('taskId') is not None and tr.get('taskId') >= 0:
                task_placements.append({
                    "task_id": tr.get('taskId'),
                    "task_type": tr.get('taskType', {}).get('name', 'unknown'),
                    "source_node": tr.get('sourceNode'),
                    "execution_node": tr.get('executionNode'),
                    "execution_platform": tr.get('executionPlatform'),
                    "elapsed_time": tr.get('elapsedTime'),
                    "queue_time": tr.get('queueTime'),
                    "queue_snapshot_at_scheduling": tr.get('queueSnapshotAtScheduling', {}),
                    "full_queue_snapshot": tr.get('fullQueueSnapshot', {}),
                    "temporal_state_at_scheduling": tr.get('temporalStateAtScheduling', {}),
                })

        # Build captured state with queue occupancy at scheduling time per task
        captured_state = {
            "timestamp": final_state.get('timestamp', 0),
            "replicas": final_state.get('replicas', {}),
            "available_resources": final_state.get('available_resources', {}),
            "scheduler_state": final_state.get('scheduler_state', {}),
            "task_placements": task_placements,
            "total_rtt": sum(
                tr.get('elapsedTime', 0)
                for tr in task_results
                if tr.get('taskId') is not None and tr.get('taskId') >= 0
            ),
        }

        # Save to dataset directory (unique placements version)
        output_file = dataset_dir / "system_state_captured_unique.json"
        with open(output_file, 'w') as f:
            json.dump(captured_state, f, indent=2, cls=DataclassJSONEncoder)

        logger.info(f"✓ Saved {output_file}")
        print(f"  ✓ Saved system_state_captured_unique.json (RTT: {captured_state['total_rtt']:.3f}s)")

        return True

    except Exception as e:
        logger.error(f"Error processing {dataset_dir.name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main entry point for knative baseline co-simulation.
    
    Usage:
        python -m src.executeknativecosim --dataset-dir <path_to_ds_directory>
        python -m src.executeknativecosim --datasets-base <path_to_gnn_datasets>
    """
    # Configuration
    sim_input_path = Path("data/nofs-ids")

    # Parse arguments
    dataset_dir = None
    datasets_base = None

    if '--dataset-dir' in sys.argv:
        idx = sys.argv.index('--dataset-dir')
        if idx + 1 < len(sys.argv):
            dataset_dir = Path(sys.argv[idx + 1])

    if '--datasets-base' in sys.argv:
        idx = sys.argv.index('--datasets-base')
        if idx + 1 < len(sys.argv):
            datasets_base = Path(sys.argv[idx + 1])

    # Setup logging
    logger = setup_logging(Path("."))

    if dataset_dir:
        # Process single dataset
        if not dataset_dir.exists():
            print(f"ERROR: Dataset directory not found: {dataset_dir}")
            sys.exit(1)

        success = run_knative_baseline_for_dataset(dataset_dir, sim_input_path, logger)
        sys.exit(0 if success else 1)

    elif datasets_base:
        # Process all ds_* directories
        if not datasets_base.exists():
            print(f"ERROR: Datasets base directory not found: {datasets_base}")
            sys.exit(1)

        # Find all ds_* directories
        ds_dirs = sorted([
            d for d in datasets_base.iterdir()
            if d.is_dir() and d.name.startswith('ds_')
        ])

        if not ds_dirs:
            print(f"No ds_* directories found in {datasets_base}")
            sys.exit(1)

        print(f"Found {len(ds_dirs)} datasets to process")

        success_count = 0
        skip_count = 0

        for ds_dir in ds_dirs:
            print(f"\n[{ds_dir.name}]")
            if run_knative_baseline_for_dataset(ds_dir, sim_input_path, logger):
                success_count += 1
            else:
                skip_count += 1

        print(f"\n=== Complete ===")
        print(f"Processed: {success_count}")
        print(f"Skipped: {skip_count}")

    else:
        print("Usage:")
        print("  python -m src.executeknativecosim --dataset-dir <path_to_ds_directory>")
        print("  python -m src.executeknativecosim --datasets-base <path_to_gnn_datasets>")
        sys.exit(1)


if __name__ == "__main__":
    main()
