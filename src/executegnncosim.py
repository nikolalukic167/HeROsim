"""
GNN Baseline Co-simulation Executor

This script runs the gnn_cosim scheduler on existing datasets (ds_*) and captures
the system state with queue occupancy at scheduling time.

Workflow:
1. Load infrastructure.json and workload.json from an existing ds_* directory
2. Run Phase 1 simulation with gnn_cosim scheduler (captures system state)
3. Save system_state_gnn.json to the ds_* directory

The saved system state includes queue_occupancy which can later be used to:
- Compare with brute-force optimal placements
- Analyze GNN's placement decisions vs knative baseline
"""

import json
import logging
import os
import sys
import signal
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable

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
        models=None,
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
        'workload-gnn',
        models=models,
    )
    return {
        "status": "success",
        "config": config,
        "sim_inputs": sim_inputs,
        "stats": stats
    }


def run_gnn_baseline_for_dataset(
        dataset_dir: Path,
        sim_input_path: Path,
        logger: logging.Logger,
        gnn_model: Any = None,
        task_types_data: Dict[str, Any] = None,
) -> bool:
    """
    Run gnn_cosim scheduler on a single dataset and save system state.
    
    Returns True if successful, False if skipped or failed.
    """
    logger.info(f"Processing dataset: {dataset_dir.name}")

    # Check required files exist
    infrastructure_file = dataset_dir / "infrastructure.json"
    workload_file = dataset_dir / "workload.json"
    space_config_file = dataset_dir / "space_with_network.json"
    knative_baseline_file = dataset_dir / "system_state_captured_unique.json"

    if not infrastructure_file.exists():
        logger.warning(f"Skipping {dataset_dir.name}: infrastructure.json not found")
        return False

    if not workload_file.exists():
        logger.warning(f"Skipping {dataset_dir.name}: workload.json not found")
        return False

    if not space_config_file.exists():
        logger.warning(f"Skipping {dataset_dir.name}: space_with_network.json not found")
        return False

    if not knative_baseline_file.exists():
        logger.warning(f"Skipping {dataset_dir.name}: system_state_captured_unique.json not found (knative baseline missing)")
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

        logger.info(f"Running gnn_cosim simulation for {dataset_dir.name}...")
        print(f"  Running gnn_cosim simulation...")

        # Prepare models dict for GNN scheduler
        models = {
            'gnn_model': gnn_model,
            'task_types_data': task_types_data,
            'dataset_id': dataset_dir.name,
        }

        # Execute simulation with gnn_cosim scheduler
        result = execute_simulation(
            full_config,
            sim_inputs,
            scheduling_strategy='gnn_cosim_gnn_cosim',
            cache_policy='fifo',
            task_priority='fifo',
            keep_alive=KEEP_ALIVE,
            queue_length=QUEUE_LENGTH,
            models=models,
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
                    "gnn_decision_time": tr.get('gnn_decision_time', 0.0),
                    "queue_snapshot_at_scheduling": tr.get('queueSnapshotAtScheduling', {}),
                    "full_queue_snapshot": tr.get('fullQueueSnapshot', {}),
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
            "scheduler_type": "gnn_cosim",
        }

        # Save to dataset directory (GNN version - different from knative)
        output_file = dataset_dir / "system_state_gnn.json"
        with open(output_file, 'w') as f:
            json.dump(captured_state, f, indent=2, cls=DataclassJSONEncoder)

        logger.info(f"✓ Saved {output_file}")
        print(f"  ✓ Saved system_state_gnn.json (RTT: {captured_state['total_rtt']:.3f}s)")

        return True

    except Exception as e:
        logger.error(f"Error processing {dataset_dir.name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def load_gnn_model(model_path: Path):
    """Load the trained GNN model."""
    import torch
    from src.policy.gnn_cosim.gnn_model import TaskPlacementGNN
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading GNN model from {model_path} on {device}...", flush=True)
        
        # Model architecture must match training
        model = TaskPlacementGNN(
            task_feature_dim=3,
            platform_feature_dim=8,  # With queue_length feature
            embedding_dim=64,
            hidden_dim=64,
            num_layers=3
        )
        
        # Load state dict first, then move to device to avoid CUDA context issues
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        # Clear CUDA cache to avoid memory issues
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        print(f"GNN model loaded successfully ({sum(p.numel() for p in model.parameters()):,} parameters)", flush=True)
        return model, device
    except Exception as e:
        print(f"ERROR loading GNN model: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


def load_task_types_data(sim_input_path: Path) -> Dict[str, Any]:
    """Load task-types.json for feature extraction."""
    task_types_path = sim_input_path / "task-types.json"
    with open(task_types_path, 'r') as f:
        return json.load(f)


class DatasetTimeoutError(Exception):
    """Custom timeout exception for dataset processing."""
    pass


def _timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise DatasetTimeoutError("Operation timed out")


def run_with_timeout(func: Callable, timeout_seconds: float, default_result: Any = None):
    """
    Run a function with a timeout using signal (Unix only).
    
    Returns:
        (result, timed_out): tuple of (function result or default_result, True if timed out)
    """
    if timeout_seconds <= 0:
        # No timeout
        try:
            return func(), False
        except Exception as e:
            raise e
    
    # Set up signal handler
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(timeout_seconds))
    
    try:
        result = func()
        timed_out = False
    except DatasetTimeoutError:
        result = default_result
        timed_out = True
    finally:
        # Restore previous handler and cancel alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    
    return result, timed_out


def main():
    """
    Main entry point for GNN baseline co-simulation.
    
    Usage:
        python -m src.executegnncosim --dataset-dir <path_to_ds_directory>
        python -m src.executegnncosim --datasets-base <path_to_gnn_datasets>
    """
    # Configuration
    sim_input_path = Path("data/nofs-ids")
    model_path = Path("src/notebooks/new/best_gnn_regret_model.pt")

    # Parse arguments
    dataset_dir = None
    datasets_base = None
    timeout_seconds = 0  # 0 means no timeout

    if '--dataset-dir' in sys.argv:
        idx = sys.argv.index('--dataset-dir')
        if idx + 1 < len(sys.argv):
            dataset_dir = Path(sys.argv[idx + 1])

    if '--datasets-base' in sys.argv:
        idx = sys.argv.index('--datasets-base')
        if idx + 1 < len(sys.argv):
            datasets_base = Path(sys.argv[idx + 1])

    if '--timeout' in sys.argv:
        idx = sys.argv.index('--timeout')
        if idx + 1 < len(sys.argv):
            try:
                timeout_seconds = float(sys.argv[idx + 1])
            except ValueError:
                print(f"ERROR: Invalid timeout value: {sys.argv[idx + 1]}")
                sys.exit(1)

    # Setup logging
    logger = setup_logging(Path("."))

    # Load GNN model once at startup
    if not model_path.exists():
        print(f"ERROR: GNN model not found at {model_path}")
        sys.exit(1)
    
    gnn_model, device = load_gnn_model(model_path)
    task_types_data = load_task_types_data(sim_input_path)

    if dataset_dir:
        # Process single dataset
        if not dataset_dir.exists():
            print(f"ERROR: Dataset directory not found: {dataset_dir}")
            sys.exit(1)

        success = run_gnn_baseline_for_dataset(
            dataset_dir, sim_input_path, logger, 
            gnn_model=gnn_model, task_types_data=task_types_data
        )
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
        if timeout_seconds > 0:
            print(f"Timeout per dataset: {timeout_seconds}s")
        print()

        success_count = 0
        skip_count = 0
        timeout_count = 0

        for ds_dir in ds_dirs:
            print(f"[{ds_dir.name}] Processing...", flush=True)
            
            def process_dataset():
                return run_gnn_baseline_for_dataset(
                    ds_dir, sim_input_path, logger,
                    gnn_model=gnn_model, task_types_data=task_types_data
                )
            
            try:
                if timeout_seconds > 0:
                    result, timed_out = run_with_timeout(process_dataset, timeout_seconds, default_result=False)
                    if timed_out:
                        print(f"[{ds_dir.name}] ⏱ Timeout ({timeout_seconds}s exceeded)", flush=True)
                        timeout_count += 1
                    elif result:
                        print(f"[{ds_dir.name}] ✓ Success", flush=True)
                        success_count += 1
                    else:
                        print(f"[{ds_dir.name}] ⚠ Skipped", flush=True)
                        skip_count += 1
                else:
                    # No timeout
                    if process_dataset():
                        print(f"[{ds_dir.name}] ✓ Success", flush=True)
                        success_count += 1
                    else:
                        print(f"[{ds_dir.name}] ⚠ Skipped", flush=True)
                        skip_count += 1
            except Exception as e:
                print(f"[{ds_dir.name}] ✗ Error: {e}", flush=True)
                import traceback
                traceback.print_exc()
                skip_count += 1

        print(f"\n=== Complete ===")
        print(f"Success: {success_count}")
        if timeout_seconds > 0:
            print(f"Timeout: {timeout_count}")
        print(f"Skipped: {skip_count}")
        print(f"Total: {len(ds_dirs)}")

    else:
        print("Usage:")
        print("  python -m src.executegnncosim --dataset-dir <path_to_ds_directory> [--timeout <seconds>]")
        print("  python -m src.executegnncosim --datasets-base <path_to_gnn_datasets> [--timeout <seconds>]")
        print("")
        print("Options:")
        print("  --timeout <seconds>  Timeout per dataset (default: no timeout)")
        sys.exit(1)


if __name__ == "__main__":
    main()

