import json
import logging
import math
import multiprocessing
import os
import pickle
import sys
import time
import hashlib
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional

import concurrent.futures
from pathlib import Path

import numpy as np  # type: ignore[import-not-found]
from itertools import islice

# Try to use orjson for faster JSON serialization (falls back to stdlib json)
try:
    import orjson
    
    def _convert_keys_to_str(obj):
        """Recursively convert dict keys to strings for orjson compatibility."""
        if isinstance(obj, dict):
            return {str(k): _convert_keys_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert_keys_to_str(item) for item in obj]
        elif isinstance(obj, tuple):
            return [_convert_keys_to_str(item) for item in obj]
        return obj
    
    def json_dumps(obj, **kwargs):
        """Fast JSON serialization using orjson."""
        # Fall back to stdlib json if custom encoder is needed
        if 'cls' in kwargs:
            return json.dumps(obj, separators=(',', ':'), **kwargs)
        try:
            return orjson.dumps(_convert_keys_to_str(obj), option=orjson.OPT_SERIALIZE_NUMPY).decode('utf-8')
        except (TypeError, orjson.JSONEncodeError):
            # Fall back if orjson can't handle the object
            return json.dumps(obj, separators=(',', ':'), default=str, **kwargs)
    
    def json_dumps_pretty(obj, **kwargs):
        """Pretty JSON serialization using orjson."""
        # Fall back to stdlib json if custom encoder is needed
        if 'cls' in kwargs:
            return json.dumps(obj, indent=2, **kwargs)
        try:
            return orjson.dumps(_convert_keys_to_str(obj), option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY).decode('utf-8')
        except (TypeError, orjson.JSONEncodeError):
            # Fall back if orjson can't handle the object
            return json.dumps(obj, indent=2, default=str, **kwargs)
    HAS_ORJSON = True
except ImportError:
    def json_dumps(obj, **kwargs):
        """Fallback to stdlib json."""
        return json.dumps(obj, separators=(',', ':'), **kwargs)
    def json_dumps_pretty(obj, **kwargs):
        """Fallback to stdlib json with indent."""
        return json.dumps(obj, indent=2, **kwargs)
    HAS_ORJSON = False

from src.eventgenerator import increase_events_of_app
from src.motivational.constants import KEEP_ALIVE, QUEUE_LENGTH
from src.placement.executor import execute_sim
from src.placement.model import SimulationData, DataclassJSONEncoder

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Global quiet mode flag (set via --quiet command line argument)
QUIET_MODE = False

# Global shared data for worker processes (initialized via _init_worker)
# This avoids pickling large immutable data for every task submission
_worker_shared_data: Dict[str, Any] = {}

REQUIRED_SIM_FILES = [
    'application-types.json',
    'platform-types.json',
    'qos-types.json',
    'storage-types.json',
    'task-types.json'
]


def _init_worker(
    sim_inputs: Dict[str, Any],
    infra_config: Dict[str, Any],
    base_nodes: List[Dict[str, Any]],
    flattened_workloads: Dict[str, Any],
    replica_plan: Dict[str, Any],
    apps: List[str],
    infrastructure_file: Optional[Path],
    sample: np.ndarray,
    mapping: Dict[int, str],
    output_dir: Path,
    quiet: bool = False,
    best_rtt_value: Optional[Any] = None,
    best_rtt_lock: Optional[Any] = None
):
    """
    Initialize worker process with shared immutable data.
    Called once per worker process, not per task.
    
    Args:
        best_rtt_value: Shared multiprocessing.Value for tracking best RTT across workers
        best_rtt_lock: Lock for atomic operations on best_rtt_value
    """
    global _worker_shared_data, QUIET_MODE
    QUIET_MODE = quiet
    _worker_shared_data = {
        'sim_inputs': sim_inputs,
        'infra_config': infra_config,
        'base_nodes': base_nodes,
        'flattened_workloads': flattened_workloads,
        'replica_plan': replica_plan,
        'apps': apps,
        'infrastructure_file': infrastructure_file,
        'sample': sample,
        'mapping': mapping,
        'output_dir': output_dir,
        'best_rtt_value': best_rtt_value,
        'best_rtt_lock': best_rtt_lock,
    }


def _log(msg: str, force: bool = False):
    """Print message unless in quiet mode."""
    if not QUIET_MODE or force:
        print(msg)

def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger('simulation')
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if setup is called multiple times
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        logger.addHandler(ch)

    # Ensure logs don't propagate to root (which can cause duplicates)
    logger.propagate = False

    return logger


def load_simulation_inputs(sim_input_path: Path) -> Dict[str, Any]:
    """Load all required simulation input files."""
    sim_inputs = {}

    # Verify all required files exist
    missing_files = []
    for filename in REQUIRED_SIM_FILES:
        if not (sim_input_path / filename).exists():
            missing_files.append(filename)

    if missing_files:
        raise FileNotFoundError(
            f"Missing required simulation input files: {', '.join(missing_files)}"
        )

    # Load all files
    for filename in REQUIRED_SIM_FILES:
        file_path = sim_input_path / filename
        with open(file_path, 'r') as f:
            # Use filename without extension as key
            key = filename.replace('.json', '').replace('-', '_')
            sim_inputs[key] = json.load(f)

    return sim_inputs


def generate_network_latencies(nodes: List[Dict], config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Generate network latencies between nodes based on configuration.
    Uses connection_probability from config to determine connectivity.
    
    Args:
        nodes: List of node configurations
        config: Configuration containing network latency and topology settings
    
    Returns:
        Dictionary mapping node names to their network maps
    """
    import random
    
    network_config = config.get('network', {})
    latency_config = network_config.get('latency', {})
    topology_config = network_config.get('topology', {})
    
    device_latencies = latency_config.get('device_latencies', {})
    base_latency = latency_config.get('base_latency', 0.1)
    topology_type = topology_config.get('type', 'sparse')
    # Optional reproducibility seed
    seed = topology_config.get('seed')
    if seed is not None:
        try:
            random.seed(int(seed))
            print(f"Seeding network topology RNG with seed={seed}")
        except Exception:
            print(f"Warning: Invalid network topology seed '{seed}', ignoring")
    connection_probability = topology_config.get('connection_probability', 0.85)
    custom_edges = topology_config.get('edges', [])
    
    # Separate clients and servers based on naming convention
    # Client nodes: client_node0, client_node1, etc.
    # Server nodes: node0, node1, etc.
    clients = [node for node in nodes if node['node_name'].startswith('client_node')]
    servers = [node for node in nodes if not node['node_name'].startswith('client_node')]
    
    # Initialize network maps
    network_maps = {node['node_name']: {} for node in nodes}
    
    def generate_latency(device_type1: str, device_type2: str) -> float:
        """Generate latency between two device types."""
        if device_type1 in device_latencies and device_type2 in device_latencies[device_type1]:
            latency_config = device_latencies[device_type1][device_type2]
            min_latency = latency_config.get('min', base_latency)
            max_latency = latency_config.get('max', base_latency)
            return random.uniform(min_latency, max_latency)
        else:
            return base_latency
    
    if topology_type == 'custom' and custom_edges:
        # Use custom topology edges
        print(f"Using custom topology with {len(custom_edges)} edges")
        
        for edge in custom_edges:
            if len(edge) == 2:
                client_name, server_name = edge
                
                # Validate that both nodes exist
                client_node = next((n for n in clients if n['node_name'] == client_name), None)
                server_node = next((n for n in servers if n['node_name'] == server_name), None)
                
                if client_node and server_node:
                    # Generate latency
                    latency = generate_latency(client_node['type'], server_node['type'])
                    
                    # Add bidirectional connection
                    network_maps[client_name][server_name] = latency
                    network_maps[server_name][client_name] = latency
                    print(f"  Custom edge: {client_name} <-> {server_name} (latency: {latency:.3f}s)")
                else:
                    print(f"  Warning: Custom edge {edge} references non-existent nodes")
        
        # Ensure minimum connectivity: each node should have at least one connection
        for node_name, connections in network_maps.items():
            if len(connections) == 0:
                print(f"  Warning: Node {node_name} has no connections from custom topology")
                
    else:
        # Generate connections based on connection probability
        print(f"Using probabilistic topology with connection probability: {connection_probability}")
        
        # Each client can connect to any server with the given probability
        for client in clients:
            client_name = client['node_name']
            client_type = client['type']
            
            for server in servers:
                server_name = server['node_name']
                server_type = server['type']
                
                # Use connection probability to determine if this connection should exist
                if random.random() < connection_probability:
                    # Generate latency
                    latency = generate_latency(client_type, server_type)
                    
                    # Add bidirectional connection
                    network_maps[client_name][server_name] = latency
                    network_maps[server_name][client_name] = latency
    
    # Ensure minimum connectivity: each node should have at least one connection
    # This prevents isolated nodes
    for node_name, connections in network_maps.items():
        if len(connections) == 0:
            # Find a suitable connection partner
            if node_name.startswith('client_node'):
                # Client node needs to connect to a server
                # Find servers that this client hasn't connected to yet
                available_servers = [s for s in servers if s['node_name'] not in connections]
                if available_servers:
                    server = random.choice(available_servers)
                    server_name = server['node_name']
                    server_type = server['type']
                    client_type = next(n['type'] for n in clients if n['node_name'] == node_name)
                    
                    latency = generate_latency(client_type, server_type)
                    network_maps[node_name][server_name] = latency
                    network_maps[server_name][node_name] = latency
            else:
                # Server node needs to connect to a client
                # Find clients that this server hasn't connected to yet
                available_clients = [c for c in clients if c['node_name'] not in connections]
                if available_clients:
                    client = random.choice(available_clients)
                    client_name = client['node_name']
                    client_type = client['type']
                    server_type = next(n['type'] for n in servers if n['node_name'] == node_name)
                    
                    latency = generate_latency(client_type, server_type)
                    network_maps[node_name][client_name] = latency
                    network_maps[client_name][node_name] = latency
    
    # Print statistics
    total_connections = sum(len(connections) for connections in network_maps.values())
    print(f"Network topology generated:")
    print(f"  Total nodes: {len(nodes)} ({len(clients)} clients, {len(servers)} servers)")
    print(f"  Topology type: {topology_type}")
    if topology_type == 'sparse':
        print(f"  Connection probability: {connection_probability}")
    elif topology_type == 'custom':
        print(f"  Custom edges: {len(custom_edges)}")
    print(f"  Total connections: {total_connections / 2}")
    print(f"  Average connections per node: {total_connections / len(nodes):.1f}")
    
    return network_maps


def prepare_workloads(
        sample: np.ndarray,
        mapping: Dict[int, str],
        base_workload: Dict,
        apps: List[str]
) -> Dict[str, Dict]:
    """Prepare workloads based on sample values."""
    reverse_mapping = {name: idx for idx, name in mapping.items()}
    prepared_workloads = {}

    # Process each application
    for app_name in apps:
        # Get the workload factor from sample
        workload_key = f'workload_{app_name}'
        if workload_key in reverse_mapping:
            # logging.warning('read factor from sample, currently set to 1 for debugging purposes')
            factor = sample[int(reverse_mapping[workload_key])]
            # factor = 1
            # Create a deep copy of base workload
            workload_copy = deepcopy(base_workload)
            # Apply increase_events with the factor
            prepared_workloads[app_name] = increase_events_of_app(workload_copy['events'], factor, app_name)

    return prepared_workloads


def load_samples(prefix="lhs_samples"):
    """Load LHS samples and their mapping."""
    samples = np.load(f"{prefix}.npy")
    with open(f"{prefix}_mapping.pkl", 'rb') as f:
        mapping = pickle.load(f)
    return samples, mapping


def load_config(config_file: str) -> dict:
    """Load original configuration file with specs."""
    with open(config_file, 'r') as f:
        return json.load(f)


def create_reverse_mapping(mapping: Dict[int, str]) -> Dict[str, int]:
    """Create reverse mapping from names to indices."""
    return {name: int(idx) for idx, name in mapping.items()}


def calculate_device_counts(cluster_size: int, proportions: Dict[str, float]) -> Dict[str, int]:
    """Calculate number of devices for each type."""
    device_counts = {}
    remaining_size = cluster_size

    # First round up all device counts
    for device, proportion in proportions.items():
        count = math.ceil(cluster_size * proportion)
        device_counts[device] = count
        remaining_size -= count

    # If we allocated too many devices, reduce counts one by one
    # from the device with the smallest proportion
    if remaining_size < 0:
        sorted_devices = sorted(proportions.items(), key=lambda x: x[1])
        idx = 0
        while remaining_size < 0:
            device = sorted_devices[idx][0]
            if device_counts[device] > 1:  # Ensure at least one device remains
                device_counts[device] -= 1
                remaining_size += 1
            idx = (idx + 1) % len(sorted_devices)

    return device_counts


def determine_replica_placement(
        infrastructure: Dict[str, Any],
        simulation_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Determine which replicas should be created for each task type.
    
    This function determines the replica placement plan based on infrastructure
    configuration, which is then passed to simulation.py for execution.
    
    Returns:
        Dictionary with replica placement decisions
    """
    print("\n=== Determining replica placement ===")
    
    # Get configuration
    preinit_config = infrastructure.get('preinit', {})
    replicas_config = infrastructure.get('replicas', {})
    
    # Parse preinit configuration - handle both list and percentage formats
    preinit_clients = preinit_config.get('clients', [])
    preinit_servers = preinit_config.get('servers', [])
    preinit_task_types = preinit_config.get('task_types', [])
    
    # Handle percentage-based configuration
    if not preinit_clients and 'client_percentage' in preinit_config:
        # Get all client nodes from infrastructure
        all_client_nodes = [node for node in infrastructure.get('nodes', []) if node.get('node_name', '').startswith('client_node')]
        k = max(1, int(len(all_client_nodes) * float(preinit_config.get('client_percentage', 0))))
        preinit_clients = [n['node_name'] for n in all_client_nodes[:k]]
        print(f"Converted client_percentage {preinit_config['client_percentage']} to {len(preinit_clients)} clients")
    
    if not preinit_servers and 'server_percentage' in preinit_config:
        # Get all server nodes from infrastructure
        all_server_nodes = [node for node in infrastructure.get('nodes', []) if not node.get('node_name', '').startswith('client_node')]
        k = max(1, int(len(all_server_nodes) * float(preinit_config.get('server_percentage', 0))))
        preinit_servers = [n['node_name'] for n in all_server_nodes[:k]]
        print(f"Converted server_percentage {preinit_config['server_percentage']} to {len(preinit_servers)} servers")
    
    # Handle "all" values - use actual node counts from infrastructure
    all_client_nodes = [node for node in infrastructure.get('nodes', []) if node.get('node_name', '').startswith('client_node')]
    all_server_nodes = [node for node in infrastructure.get('nodes', []) if not node.get('node_name', '').startswith('client_node')]
    
    if preinit_clients == "all":
        preinit_clients = [n['node_name'] for n in all_client_nodes]
    if preinit_servers == "all":
        preinit_servers = [n['node_name'] for n in all_server_nodes]
    if preinit_task_types == "all":
        preinit_task_types = list(simulation_data['task_types'].keys())
    
    print(f"Preinit configuration:")
    print(f"  Clients: {preinit_clients}")
    print(f"  Servers: {preinit_servers}")
    print(f"  Task types: {preinit_task_types}")
    
    # Create replica placement plan
    replica_plan = {
        'preinit_clients': preinit_clients,
        'preinit_servers': preinit_servers,
        'preinit_task_types': preinit_task_types,
        'replicas_config': replicas_config,
        'prewarm_config': infrastructure.get('prewarm', {})
    }
    
    print(f"Replica placement plan created")
    return replica_plan


def prepare_simulation_config(
        sample: np.ndarray,
        mapping: Dict[int, str],
        original_config: Dict[str, Any],
        placement_plan: Optional[Dict[int, Tuple[int, int]]] = None,
        replica_plan: Optional[Dict[str, Any]] = None,
        base_nodes: Optional[List[Dict[str, Any]]] = None,
        infrastructure_file: Optional[Path] = None
) -> Dict[str, Any]:
    """Prepare simulation configuration from a sample."""
    reverse_mapping = create_reverse_mapping(mapping)

    # Extract network bandwidth
    network_bandwidth = sample[reverse_mapping['network_bandwidth']]

    # Get client and server node counts directly from config
    client_nodes_count = original_config['nodes']['client_nodes']['count']
    server_nodes_count = original_config['nodes']['server_nodes']['count']

    # Check if this is a cold start scenario (0% preinit = no pre-created replicas)
    preinit_config = original_config.get('preinit', {})
    client_percentage = preinit_config.get('client_percentage', 0)
    server_percentage = preinit_config.get('server_percentage', 0)
    is_cold_start = (client_percentage == 0 and server_percentage == 0)

    # Prepare simulation configuration
    infrastructure_config = {
        "network": {
            "bandwidth": float(network_bandwidth)
        },
        "nodes": [],
        # PLATFORM PREINITIALIZATION FLAG:
        # This flag enables the simulation to pre-create replicas for each task type
        # instead of starting with zero replicas and waiting for autoscaling.
        # This is useful for testing and debugging to ensure immediate task execution.
        # TODO: Remove this for normal simulation runs where autoscaling should handle replica creation
        "preinitialize_platforms": True,
        # New configuration parameters
        "preinit": original_config.get('preinit', {}),
        "replicas": original_config.get('replicas', {}),
        "scheduler": original_config.get('scheduler', {}),
    }

    # Load pre-generated infrastructure if available
    if infrastructure_file:
        if infrastructure_file.exists():
            print(f"[executecosim] ✓ Infrastructure file provided: {infrastructure_file}")
            print(f"[executecosim] Loading pre-generated deterministic infrastructure...")
            with open(infrastructure_file, 'r') as f:
                infra_data = json.load(f)

            # Validate infrastructure file structure – in co-simulation mode we rely
            # entirely on this file for determinism, so missing keys are fatal.
            required_keys = ['network_maps', 'replica_placements', 'queue_distributions', 'metadata']
            missing_keys = [k for k in required_keys if k not in infra_data]
            if missing_keys:
                raise RuntimeError(
                    f"[executecosim] Infrastructure file {infrastructure_file} is missing "
                    f"required keys for deterministic co-simulation: {missing_keys}"
                )
            print(f"[executecosim] ✓ Infrastructure file structure validated")
            
            network_maps = infra_data['network_maps']
            deterministic_replica_placements = infra_data['replica_placements']
            deterministic_queue_distributions = infra_data['queue_distributions']
            metadata = infra_data.get('metadata', {})
            
            print(f"[executecosim] Infrastructure metadata:")
            print(f"  Seed: {metadata.get('seed', 'N/A')}")
            print(f"  Generation time: {metadata.get('generation_time', 'N/A')}")
            
            # Generate nodes (same order as infrastructure generation)
            device_types = list(original_config['pci'].keys())
            nodes = []
            
            for i in range(client_nodes_count):
                device_type = device_types[i % len(device_types)]
                device_specs = original_config['pci'][device_type]['specs']
                node_config = device_specs.copy()
                node_config['node_name'] = f"client_node{i}"
                node_config['type'] = device_type
                node_config['network_map'] = network_maps.get(node_config['node_name'], {})
                nodes.append(node_config)
            
            for i in range(server_nodes_count):
                device_type = device_types[i % len(device_types)]
                device_specs = original_config['pci'][device_type]['specs']
                node_config = device_specs.copy()
                node_config['node_name'] = f"node{i}"
                node_config['type'] = device_type
                node_config['network_map'] = network_maps.get(node_config['node_name'], {})
                nodes.append(node_config)
            
            infrastructure_config['nodes'] = nodes
            
            # Store deterministic infrastructure data for simulation.py
            infrastructure_config['deterministic_replica_placements'] = deterministic_replica_placements
            infrastructure_config['deterministic_queue_distributions'] = deterministic_queue_distributions
            
            print(f"[executecosim] ✓ Loaded deterministic infrastructure:")
            print(f"  Network maps: {len(network_maps)} nodes")
            print(f"  Replica placements: {sum(len(v) for v in deterministic_replica_placements.values())} total")
            print(f"  Queue distributions: {sum(len(v) for v in deterministic_queue_distributions.values())} platforms")
            
            # Log sample of placements for verification
            for task_type, placements in deterministic_replica_placements.items():
                if placements:
                    sample_placement = placements[0]
                    print(f"  Sample {task_type} placement: {sample_placement['node_name']}:{sample_placement['platform_id']}")
        else:
            print(f"[executecosim] ⚠️  WARNING: Infrastructure file provided but not found: {infrastructure_file}")
            print(f"[executecosim] Falling back to legacy (non-deterministic) infrastructure generation")
            infrastructure_file = None  # Fall through to legacy path
    elif base_nodes is not None and len(base_nodes) > 0:
        # Reuse provided nodes (and their network maps) to keep topology consistent
        print(f"[executecosim] Reusing {len(base_nodes)} base nodes (from previous placement)")
        infrastructure_config['nodes'] = [deepcopy(n) for n in base_nodes]
    else:
        # Legacy path: Generate infrastructure on-the-fly (non-deterministic)
        print(f"[executecosim] ⚠️  Using LEGACY (non-deterministic) infrastructure generation")
        print(f"[executecosim]   No infrastructure file provided - generating network topology on-the-fly")
        
        # Generate client nodes
        device_types = list(original_config['pci'].keys())  # ['rpi', 'xavier', 'pyngFpga']
        for i in range(client_nodes_count):
            device_type = device_types[i % len(device_types)]
            device_specs = original_config['pci'][device_type]['specs']
            node_config = device_specs.copy()
            node_config['node_name'] = f"client_node{i}"
            node_config['type'] = device_type
            infrastructure_config['nodes'].append(node_config)

        # Generate server nodes
        for i in range(server_nodes_count):
            device_type = device_types[i % len(device_types)]
            device_specs = original_config['pci'][device_type]['specs']
            node_config = device_specs.copy()
            node_config['node_name'] = f"node{i}"
            node_config['type'] = device_type
            infrastructure_config['nodes'].append(node_config)
        
        # Generate network maps using configuration-based approach
        print(f"[executecosim]   Generating network topology (may be non-deterministic)...")
        network_maps = generate_network_latencies(infrastructure_config['nodes'], original_config)
            
        # Assign network maps to nodes
        for node in infrastructure_config['nodes']:
            node['network_map'] = network_maps[node['node_name']]
        
        print(f"[executecosim]   Generated network topology with {len(network_maps)} nodes")

    # Debug: Print infrastructure node names
    # print(f"\nInfrastructure nodes created:")
    # for node in infrastructure_config['nodes']:
    #     print(f"  {node['node_name']}: {len(node['network_map'])} network connections")

    # Add placement plan to infrastructure config if provided
    if placement_plan is not None:
        # Make placements available for scheduler
        infrastructure_config['forced_placements'] = placement_plan
        print(f"Added placement plan with {len(placement_plan)} task placements")

    # Pass replica plan if provided (skip for cold start scenarios)
    # For cold start (0% preinit), we don't pre-create replicas, so no replica_plan needed
    if replica_plan is not None and not is_cold_start:
        infrastructure_config['replica_plan'] = replica_plan

    # Add prewarm configuration (reduced for cold start scenarios)
    prewarm_config = original_config.get('prewarm', {})
    # For cold start, use minimal/no prewarming to simulate realistic initial state
    if is_cold_start:
        # Override prewarm config for cold start - no warmup tasks
        prewarm_config = {
            task_type: {
                "distribution": "none",
                "queue_distribution": "statistical",
                "queue_distribution_params": {
                    "type": "constant",
                    "value": 0,  # No initial queue for cold start
                    "min": 0,
                    "max": 0,
                    "step": 0
                }
            }
            for task_type in original_config.get('replicas', {}).keys()
        }
    infrastructure_config['prewarm'] = prewarm_config
    
    # Add replica_plan so simulation.py can pre-create replicas
    if replica_plan is not None:
        infrastructure_config['replica_plan'] = replica_plan

    return infrastructure_config


def execute_simulation(
        config: Dict[str, Any],
        sim_inputs: Dict[str, Any],
        scheduling_strategy: str,
        model_locations: Optional[Dict[str, str]] = None,
        models: Optional[Dict[str, Any]] = None,
        cache_policy='fifo',
        task_priority='fifo',
        keep_alive=30,
        queue_length=100,
        reconcile_interval=1
) -> Dict[str, Any]:
    """Execute simulation with full configuration and simulation inputs."""

    simulation_data = SimulationData(
        platform_types=sim_inputs['platform_types'],
        storage_types=sim_inputs['storage_types'],
        qos_types=sim_inputs['qos_types'],
        application_types=sim_inputs['application_types'],
        task_types=sim_inputs['task_types'],
    )

    stats = execute_sim(simulation_data, config['infrastructure'], cache_policy, keep_alive, task_priority,
                        queue_length,
                        scheduling_strategy, config['workload'], 'workload-mine',
                        model_locations=model_locations, models=models, reconcile_interval=reconcile_interval)
    return {
        "status": "success",
        "config": config,
        "sim_inputs": sim_inputs,
        "stats": stats
    }


def calculate_workload_stats(events: List[Dict]) -> Dict[str, float]:
    """Calculate statistics for the flattened workload."""
    if not events:
        return {
            "average_rps": 0,
            "duration": 0,
            "total_events": 0
        }

    # Get timestamps as integers
    timestamps = [int(event['timestamp']) for event in events]
    min_timestamp = min(timestamps)
    max_timestamp = max(timestamps)

    # Calculate duration in seconds
    duration = max_timestamp - min_timestamp + 1  # +1 to include both start and end second

    # Calculate average RPS
    total_events = len(events)
    average_rps = total_events / duration if duration > 0 else 0

    return {
        "rps": average_rps,
        "duration": duration,
        "total_events": total_events,
        "start_timestamp": min_timestamp,
        "end_timestamp": max_timestamp
    }


def flatten_workloads(workloads: Dict[str, Dict]) -> Dict[str, Any]:
    """Flatten multiple workload events into a single sorted list with statistics."""
    # Collect all events
    all_events = []
    for app_name, workload in workloads.items():
        events = workload
        all_events.extend(events)

    # Sort events by timestamp
    sorted_events = sorted(all_events, key=lambda x: x['timestamp'])

    # Calculate statistics
    stats = calculate_workload_stats(sorted_events)

    return {
        "rps": stats['rps'],
        "duration": stats['duration'],
        "events": sorted_events
    }


def capture_system_state_from_first_task(
        sample: np.ndarray,
        mapping: Dict[int, str],
        infra_config: Dict[str, Any],
        sim_inputs: Dict[str, Any],
        workload_events: List[Dict],
        replica_plan: Dict[str, Any],
        output_dir: Path,
        infrastructure_file: Optional[Path] = None
) -> Optional[Dict[str, List[Tuple[str, int]]]]:
    """
    Capture system state after warmup tasks complete by running simulation with first real task.
    
    This function:
    1. Extracts the first workload event
    2. Creates a workload with only that first event
    3. Sets up infrastructure with forced_placements = {0: (-1, -1)} for auto-resolve
    4. Runs simulation and waits for completion
    5. Extracts final system state from stats['systemStateResults'][-1]
    6. Converts replicas format from [[node_name, platform_id], ...] to [(node_name, platform_id), ...]
    
    Args:
        sample: Sample array for infrastructure configuration
        mapping: Mapping from indices to parameter names
        infra_config: Infrastructure configuration
        sim_inputs: Simulation inputs
        workload_events: Full list of workload events
        replica_plan: Replica placement plan
        output_dir: Output directory for saving the capture result
        infrastructure_file: Optional infrastructure file path
    
    Returns:
        Dict mapping task_type -> list of (node_name, platform_id) tuples, or None if capture fails
    """
    logger = logging.getLogger('simulation')
    logger.info("=== Phase 1: Capturing System State from First Task ===")
    print(f"\n=== Phase 1: Capturing System State from First Task ===")
    
    if not workload_events:
        print("⚠️  No workload events available for state capture")
        return None
    
    logger.info(f"Total workload events: {len(workload_events)}")
    
    # Extract first workload event (use 1 task for state capture)
    # NOTE: We use 1 task but need to ensure replicas are warm before it arrives
    num_events_to_use = 1
    events_to_use = workload_events[:num_events_to_use]
    
    first_event = events_to_use[0]
    first_event_timestamp = first_event.get('timestamp', 0)
    
    logger.info(f"Using first {num_events_to_use} workload event (timestamp: {first_event_timestamp})")
    print(f"Using first {num_events_to_use} workload event")
    print(f"First event node: {first_event.get('node_name', 'unknown')}")
    
    # Create workload with first event
    single_event_workload = {
        "rps": 1.0 / max(1, first_event_timestamp + 1),  # Approximate RPS
        "duration": first_event_timestamp + 1,
        "events": events_to_use
    }
    logger.info(f"Created workload with {len(single_event_workload['events'])} event(s)")
    
    # Prepare infrastructure configuration with auto-resolve for task 0
    # Auto-resolve will find a warm replica that was created during precreate_replicas
    placement_plan = {0: (-1, -1)}
    logger.info(f"Preparing infrastructure configuration with auto-resolve placement for task 0...")
    logger.info("NOTE: Replicas should be pre-created via replica_plan and warmed up before task 0 arrives")
    sim_config = prepare_simulation_config(
        sample,
        mapping,
        infra_config,
        placement_plan=placement_plan,  # Auto-resolve for task 0
        replica_plan=replica_plan,
        infrastructure_file=infrastructure_file
    )
    logger.info("Infrastructure configuration prepared")
    
    # Set batch_size=1 for state capture (only 1 task available)
    sim_config['scheduler'] = {
        'batch_size': 1,
        'batch_timeout': 0.1
    }
    logger.info("Set scheduler batch_size=1 for state capture simulation")
    
    # Verify replica_plan is present (required for warmup tasks)
    if not sim_config.get('replica_plan'):
        logger.warning("WARNING: No replica_plan in sim_config - replicas may not be pre-created!")
        print("⚠️  WARNING: No replica_plan - replicas may not be pre-created")
    else:
        logger.info("✓ replica_plan is present in sim_config")
    
    # Combine infrastructure and workload configurations
    full_config = {
        "infrastructure": sim_config,
        "workload": single_event_workload,
    }
    
    logger.info("Starting state capture simulation with auto-resolve placement for task 0...")
    logger.info("Expected flow: 1) precreate_replicas creates replicas, 2) warmup tasks execute, 3) task 0 arrives and finds warm replica")
    print(f"Running state capture simulation with auto-resolve placement for task 0...")
    
    try:
        # Execute simulation
        cache_policy = 'fifo'
        task_priority = 'fifo'
        keep_alive = KEEP_ALIVE
        queue_length = QUEUE_LENGTH
        scheduling_strategy = 'determined_determined'
        
        logger.info("Executing simulation for state capture...")
        result = execute_simulation(
            full_config,
            sim_inputs,
            scheduling_strategy,
            model_locations={},
            models={},
            cache_policy=cache_policy,
            task_priority=task_priority,
            keep_alive=keep_alive,
            queue_length=queue_length,
        )
        logger.info("Simulation completed, extracting system state...")
        
        # Extract final system state
        stats = result.get('stats', {})
        system_state_results = stats.get('systemStateResults', [])
        
        logger.info(f"Found {len(system_state_results)} system state results")
        if not system_state_results:
            logger.warning("No systemStateResults found in simulation output")
            print("⚠️  No systemStateResults found in simulation output")
            return None
        
        # Get the last system state (after warmup and first task complete)
        final_state = system_state_results[-1]
        final_timestamp = final_state.get('timestamp', 0)
        logger.info(f"Using final system state at timestamp: {final_timestamp}")
        
        replicas_dict = final_state.get('replicas', {})
        
        if not replicas_dict:
            logger.warning("No replicas found in final system state")
            print("⚠️  No replicas found in final system state")
            return None
        
        logger.info(f"Found replicas for {len(replicas_dict)} task types")
        
        # Convert replicas format: {task_type: [[node_name, platform_id], ...]} -> {task_type: [(node_name, platform_id), ...]}
        active_replicas = {}
        for task_type, replica_list in replicas_dict.items():
            active_replicas[task_type] = [
                (replica[0], replica[1])  # Convert list to tuple
                for replica in replica_list
                if len(replica) >= 2
            ]
            logger.info(f"Task type {task_type}: {len(active_replicas[task_type])} active replicas")
        
        print(f"✓ Captured system state with active replicas:")
        for task_type, replicas in active_replicas.items():
            print(f"  {task_type}: {len(replicas)} active replicas")
            for node_name, platform_id in replicas[:3]:  # Show first 3
                print(f"    - {node_name}:{platform_id}")
            if len(replicas) > 3:
                print(f"    ... and {len(replicas) - 3} more")
        
        # Save capture result with specific name (not hash-based)
        logger.info("Saving system state capture result...")
        capture_result = {
            "timestamp": final_state.get('timestamp', 0),
            "active_replicas": active_replicas,
            "system_state": final_state
        }
        capture_file = output_dir / "system_state_capture.json"
        with open(capture_file, 'w') as f:
            json.dump(capture_result, f, indent=2, cls=DataclassJSONEncoder)
        
        logger.info(f"Saved system state capture to: {capture_file}")
        print(f"✓ Saved system state capture to: {capture_file}")
        
        # Save full simulation result with specific name (not hash-based)
        logger.info("Saving full simulation result...")
        result['sample'] = {
            'apps': [],
            'sample': sample.tolist() if hasattr(sample, 'tolist') else sample,
            'mapping': mapping,
            'infra_config': infra_config,
            'sim_inputs': sim_inputs,
            'scheduling_strategy': scheduling_strategy,
            'cache_policy': cache_policy,
            'task_priority': task_priority,
            'keep_alive': keep_alive,
            'queue_length': queue_length,
            'placement_plan': {0: (-1, -1)},  # Auto-resolve placement
        }
        first_run_file = output_dir / "first_task_state_capture_simulation.json"
        with open(first_run_file, 'w') as f:
            json.dump(result, f, indent=2, cls=DataclassJSONEncoder)
        
        logger.info(f"Saved first task simulation result to: {first_run_file}")
        print(f"✓ Saved first task simulation result to: {first_run_file}")
        
        logger.info("=== Phase 1 Complete: System State Captured Successfully ===")
        return active_replicas
        
    except Exception as e:
        logger.error(f"Error during system state capture: {str(e)}")
        logger.exception(e)
        print(f"❌ Error during system state capture: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def generate_brute_force_placement_combinations(
        workload_events: List[Dict],
        infrastructure_config: Dict[str, Any],
        sim_inputs: Dict[str, Any],
        replica_plan: Dict[str, Any],
        active_replicas: Optional[Dict[str, List[Tuple[str, int]]]] = None,
        use_all_replicas: bool = False
) -> List[Dict[int, Tuple[int, int]]]:
    """
    Generate all possible placement combinations for tasks.
    Applies both core filtering rules:
    1. Don't make topologies that offload from client->server that have no network connection
    2. Don't schedule on non-existent replicas for that function
    3. Ensure all tasks have valid, unique replicas (no two tasks share the same replica)
    
    For warm start scenarios, active_replicas from system state capture is used.
    For cold start scenarios (use_all_replicas=True), uses all replicas from
    infrastructure.json's deterministic_replica_placements directly.
    
    Args:
        workload_events: List of workload events with timestamps and node_name
        infrastructure_config: Infrastructure configuration with nodes and network maps
        sim_inputs: Simulation inputs containing task_types and platform_types
        replica_plan: Replica placement plan (used for reference, not for generating placements)
        active_replicas: Dict mapping task_type -> list of (node_name, platform_id) tuples.
                        Used for warm start scenarios.
        use_all_replicas: If True, use all replicas from infrastructure.json directly
                         (cold start mode). If False, use active_replicas from state capture.
    
    Returns:
        List of placement plans, each mapping task_id to (node_id, platform_id) tuple.
        Each placement ensures all tasks have valid, unique replicas.
        Returns empty list if no valid placements can be generated.
    """
    logger = logging.getLogger('simulation')
    logger.info(f"=== Generating Brute Force Placement Combinations (ALL combinations) ===")
    print(f"\n=== Generating Brute Force Placement Combinations (ALL combinations) ===")
    
    # Determine which replica source to use
    det_placements = infrastructure_config.get('deterministic_replica_placements', {})
    
    if use_all_replicas and det_placements:
        logger.info("COLD START MODE: Using ALL replicas from infrastructure.json")
        print(f"✓ COLD START MODE: Using all replicas from deterministic_replica_placements")
        for task_type, placements in det_placements.items():
            logger.info(f"  {task_type}: {len(placements)} replicas from infrastructure")
            print(f"  {task_type}: {len(placements)} replicas from infrastructure")
    elif active_replicas is not None:
        # Check if any task type has 0 replicas - if so, fall back to infrastructure replicas
        missing_replicas = [tt for tt, reps in active_replicas.items() if len(reps) == 0]
        if missing_replicas and det_placements:
            logger.info(f"Active replicas missing for {missing_replicas}, falling back to infrastructure replicas")
            print(f"⚠️  Active replicas missing for {missing_replicas}, using infrastructure replicas (cold start fallback)")
            use_all_replicas = True
            for task_type, placements in det_placements.items():
                logger.info(f"  {task_type}: {len(placements)} replicas from infrastructure")
                print(f"  {task_type}: {len(placements)} replicas from infrastructure")
        else:
            logger.info("Using captured active replicas from post-warmup system state")
            print(f"✓ Using captured active replicas from post-warmup system state")
            logger.info(f"=== Active replicas structure ===")
            for task_type, replicas in active_replicas.items():
                logger.info(f"  {task_type}: {len(replicas)} active replicas")
                print(f"  {task_type}: {len(replicas)} active replicas")
                for idx, replica in enumerate(replicas[:5]):  # Log first 5
                    logger.info(f"    [{idx}] {replica}")
                if len(replicas) > 5:
                    logger.info(f"    ... and {len(replicas) - 5} more")
    else:
        logger.info("Using initial replica plan (no captured state provided)")
        print(f"⚠️  Using initial replica plan (no captured state provided)")
    
    # Get task types and platform types
    task_types = sim_inputs['task_types']
    platform_types = sim_inputs['platform_types']
    
    # Get nodes from infrastructure
    nodes = infrastructure_config['nodes']
    
    # Create node_id mapping
    node_id_map = {node['node_name']: i for i, node in enumerate(nodes)}
    
    # Extract replica configuration
    replicas_config = replica_plan.get('replicas_config', {})
    preinit_clients = replica_plan.get('preinit_clients', [])
    preinit_servers = replica_plan.get('preinit_servers', [])
    
    # Check if we're using deterministic infrastructure
    has_deterministic = 'deterministic_replica_placements' in infrastructure_config
    if has_deterministic:
        print(f"[executecosim] ✓ Using deterministic replica placements from infrastructure.json")
        det_placements = infrastructure_config.get('deterministic_replica_placements', {})
        for task_type, placements in det_placements.items():
            print(f"  {task_type}: {len(placements)} pre-determined placements")
    else:
        print(f"[executecosim] ⚠️  Using replica plan for placement decisions (non-deterministic):")
        print(f"  Preinit clients: {preinit_clients}")
        print(f"  Preinit servers: {preinit_servers}")
        print(f"  Replicas config: {replicas_config}")
    
    # Simulate platform creation (same logic as simulation.py)
    node_platforms = {}  # node_name -> list of platform IDs
    platform_id = 0
    
    logger.info("=== Building node_platforms mapping ===")
    for node in nodes:
        node_name = node['node_name']
        node_platforms[node_name] = []
        
        # Get platforms for this node from infrastructure config
        node_platform_types = node.get('platforms', [])
        logger.info(f"Node {node_name}: {len(node_platform_types)} platforms")
        
        for platform_type_name in node_platform_types:
            node_platforms[node_name].append({
                'platform_id': platform_id,
                'platform_type': platform_type_name
            })
            logger.debug(f"  Added platform_id={platform_id}, type={platform_type_name} to {node_name}")
            platform_id += 1
    
    logger.info(f"Total platforms created: {platform_id}")
    logger.info(f"Node platform mapping: {[(name, len(plats)) for name, plats in node_platforms.items()]}")
    
    # Simulate replica creation
    available_platforms = {}
    
    if use_all_replicas and det_placements:
        # COLD START: Use all replicas from infrastructure.json directly
        print(f"Converting infrastructure replicas to platform_info format (cold start)...")
        logger.info("=== Using deterministic_replica_placements (cold start mode) ===")
        for task_type_name, placements in det_placements.items():
            if task_type_name not in available_platforms:
                available_platforms[task_type_name] = []
            
            for placement in placements:
                node_name = placement['node_name']
                platform_id = placement['platform_id']
                platform_type = placement['platform_type']
                
                if node_name not in node_id_map:
                    logger.warning(f"  Node {node_name} not in node_id_map, skipping")
                    continue
                
                available_platforms[task_type_name].append({
                    'node_name': node_name,
                    'node_id': node_id_map[node_name],
                    'platform_type': platform_type,
                    'platform_id': platform_id
                })
            
            print(f"  {task_type_name}: {len(available_platforms[task_type_name])} replicas available")
        
        print(f"Total: {sum(len(v) for v in available_platforms.values())} replicas from infrastructure")
        logger.info(f"[PLACEMENT] Cold start - available_platforms: {[(k, len(v)) for k, v in available_platforms.items()]}")
    
    elif active_replicas is not None:
        # Use captured active replicas instead of initial replica plan
        print(f"Converting captured active replicas to platform_info format...")
        logger.info("=== Starting replica matching process ===")
        for task_type_name, replica_tuples in active_replicas.items():
            logger.info(f"Processing task_type: {task_type_name} with {len(replica_tuples)} replicas")
            print(f"Processing task_type: {task_type_name} with {len(replica_tuples)} replicas")
            if task_type_name not in available_platforms:
                available_platforms[task_type_name] = []
            
            for replica_idx, (node_name, first_sim_platform_id) in enumerate(replica_tuples):
                logger.info(f"  Replica {replica_idx+1}/{len(replica_tuples)}: {task_type_name} on {node_name}:{first_sim_platform_id}")
                
                # Find node config by name
                node_config = next((n for n in nodes if n['node_name'] == node_name), None)
                if node_config is None:
                    logger.error(f"  ❌ Node {node_name} not found in infrastructure for replica {task_type_name}:{node_name}:{first_sim_platform_id}")
                    print(f"  ❌ Warning: Node {node_name} not found in infrastructure for replica {task_type_name}:{node_name}:{first_sim_platform_id}")
                    continue
                
                # Get platforms for this node in current simulation
                node_platforms_list = node_platforms.get(node_name, [])
                logger.info(f"  Node {node_name} has {len(node_platforms_list)} platforms in current simulation")
                if not node_platforms_list:
                    logger.error(f"  ❌ No platforms found on node {node_name} for replica {task_type_name}:{node_name}:{first_sim_platform_id}")
                    print(f"  ❌ Warning: No platforms found on node {node_name} for replica {task_type_name}:{node_name}:{first_sim_platform_id}")
                    continue
                
                # Log all platforms on this node
                logger.info(f"  Platforms on {node_name}: {[(p['platform_id'], p['platform_type']) for p in node_platforms_list]}")
                
                # Try to find platform by matching platform_id first (if order is consistent)
                platform_info = None
                logger.info(f"  Attempting direct platform_id match: looking for platform_id={first_sim_platform_id}")
                for p_info in node_platforms_list:
                    if p_info['platform_id'] == first_sim_platform_id:
                        platform_info = p_info
                        logger.info(f"  ✓ Direct match found: platform_id={first_sim_platform_id}, type={p_info['platform_type']}")
                        break
                
                # If not found by platform_id, try to find by relative position within node
                if platform_info is None:
                    logger.info(f"  Direct match failed, trying relative position matching...")
                    # Calculate platform_id offset for this node (sum of platforms in previous nodes)
                    node_index = next((i for i, n in enumerate(nodes) if n['node_name'] == node_name), -1)
                    logger.info(f"  Node {node_name} is at index {node_index} in nodes list")
                    
                    if node_index >= 0:
                        platform_id_offset = 0
                        for i in range(node_index):
                            prev_node_name = nodes[i]['node_name']
                            prev_node_platform_count = len(node_platforms.get(prev_node_name, []))
                            platform_id_offset += prev_node_platform_count
                            logger.debug(f"    Previous node {prev_node_name}: {prev_node_platform_count} platforms (offset now: {platform_id_offset})")
                        
                        logger.info(f"  Platform ID offset for node {node_name}: {platform_id_offset}")
                        
                        # Calculate relative position within node
                        relative_position = first_sim_platform_id - platform_id_offset
                        logger.info(f"  Calculated relative position: {first_sim_platform_id} - {platform_id_offset} = {relative_position}")
                        logger.info(f"  Node has {len(node_platforms_list)} platforms, valid range: 0-{len(node_platforms_list)-1}")
                        
                        # If relative position is valid, use it
                        if 0 <= relative_position < len(node_platforms_list):
                            platform_info = node_platforms_list[relative_position]
                            logger.info(f"  ✓ Matched by relative position {relative_position}: platform_id={platform_info['platform_id']}, type={platform_info['platform_type']}")
                        else:
                            logger.warning(f"  Relative position {relative_position} is out of range [0, {len(node_platforms_list)-1}]")
                            # Fallback: find first platform that supports the task type
                            task_type = task_types.get(task_type_name)
                            if task_type:
                                supported_platforms = task_type.get('platforms', [])
                                logger.info(f"  Trying fallback: looking for platform type in {supported_platforms}")
                                for p_info in node_platforms_list:
                                    if p_info['platform_type'] in supported_platforms:
                                        platform_info = p_info
                                        logger.warning(f"  ⚠ Matched by type fallback: platform_id={p_info['platform_id']}, type={p_info['platform_type']}")
                                        break
                    else:
                        logger.error(f"  Could not find node {node_name} in nodes list")
                        # Fallback: find first platform that supports the task type
                        task_type = task_types.get(task_type_name)
                        if task_type:
                            supported_platforms = task_type.get('platforms', [])
                            logger.info(f"  Trying fallback: looking for platform type in {supported_platforms}")
                            for p_info in node_platforms_list:
                                if p_info['platform_type'] in supported_platforms:
                                    platform_info = p_info
                                    logger.warning(f"  ⚠ Matched by type fallback: platform_id={p_info['platform_id']}, type={p_info['platform_type']}")
                                    break
                    
                    if platform_info is None:
                        logger.error(f"  ❌ FAILED: Could not match platform {first_sim_platform_id} on node {node_name} for replica {task_type_name}")
                        logger.error(f"     Node platforms: {[(p['platform_id'], p['platform_type']) for p in node_platforms_list]}")
                        logger.error(f"     Task type: {task_type_name}, Supported platforms: {task_types.get(task_type_name, {}).get('platforms', []) if task_type_name in task_types else 'N/A'}")
                        print(f"  ❌ Error: Could not match platform {first_sim_platform_id} on node {node_name} for replica {task_type_name}")
                        continue
                
                # Create platform_info dict with the matched platform_id from current simulation
                matched_platform_id = platform_info['platform_id']
                matched_platform_type = platform_info['platform_type']
                
                # CRITICAL: Verify that the matched platform actually supports this task type
                task_type = task_types.get(task_type_name)
                if not task_type:
                    logger.error(f"  ❌ Task type {task_type_name} not found in task_types")
                    print(f"  ❌ Error: Task type {task_type_name} not found")
                    continue
                
                supported_platforms = task_type.get('platforms', [])
                if matched_platform_type not in supported_platforms:
                    logger.error(f"  ❌ Platform type {matched_platform_type} does not support task type {task_type_name}")
                    logger.error(f"     Supported platform types for {task_type_name}: {supported_platforms}")
                    logger.error(f"     Matched platform: {node_name}:{matched_platform_id} (type: {matched_platform_type})")
                    print(f"  ❌ Error: Platform type {matched_platform_type} does not support task type {task_type_name}")
                    print(f"     Supported types: {supported_platforms}")
                    continue
                
                logger.info(f"  ✓ Final match: {task_type_name} on {node_name}:{first_sim_platform_id} -> {matched_platform_id} (type: {matched_platform_type}, verified supports {task_type_name})")
                
                # Verify node_name exists in node_id_map (safety check)
                if node_name not in node_id_map:
                    logger.error(f"  ❌ Node {node_name} not found in node_id_map for replica {task_type_name}:{node_name}:{matched_platform_id}")
                    logger.error(f"     Available nodes in node_id_map: {list(node_id_map.keys())[:10]}...")
                    print(f"  ❌ Error: Node {node_name} not found in node_id_map")
                    continue
                
                available_platforms[task_type_name].append({
                    'node_name': node_name,
                    'node_id': node_id_map[node_name],
                    'platform_type': matched_platform_type,
                    'platform_id': matched_platform_id  # Use platform_id from current simulation
                })
            
            logger.info(f"Task type {task_type_name}: matched {len(available_platforms[task_type_name])}/{len(replica_tuples)} replicas")
            print(f"Task type {task_type_name}: matched {len(available_platforms[task_type_name])}/{len(replica_tuples)} replicas")
        
        print(f"Converted {sum(len(v) for v in available_platforms.values())} active replicas to platform_info format")
        
        # CRITICAL: Log summary of available_platforms by task type
        logger.info(f"[PLACEMENT] Summary of available_platforms: {[(k, len(v)) for k, v in available_platforms.items()]}")
        print(f"[PLACEMENT] Summary of available_platforms: {[(k, len(v)) for k, v in available_platforms.items()]}")
    else:
        error_msg = (
            "❌ CRITICAL: active_replicas is None but legacy fallback is disabled. "
            "State capture MUST succeed for co-simulation mode. "
            "Legacy replica configs are inaccurate and will generate invalid placements. "
            "Returning empty list to abort placement generation."
        )
        logger.error(error_msg)
        print(error_msg)
        return []  # Abort - don't generate any placements
    
    # Extract tasks from workload events and apply filtering rules
    tasks = []
    task_id = 0
    # Determinism: count how many tasks the workload requires
    expected_task_count = 0
    
    for event in workload_events:
        application = event['application']
        dag = application.get('dag', {})
        
        # Handle both list and dict DAG formats
        if isinstance(dag, list):
            task_type_names = dag
        elif isinstance(dag, dict):
            task_type_names = list(dag.keys())
        else:
            task_type_names = []
        
        expected_task_count += len(task_type_names)

        for task_type_name in task_type_names:
            if task_type_name not in task_types:
                print(f"Warning: Task type {task_type_name} not found in task types")
                continue
            
            task_type = task_types[task_type_name]
            source_node_name = event['node_name']
            
            # Get available platforms for this task type
            task_platforms = available_platforms.get(task_type_name, [])
            
            # Filter platforms based on network connectivity and client node restrictions
            feasible_platforms = []
            for platform_info in task_platforms:
                node_name = platform_info['node_name']
                
                # Rule 1: Local execution always allowed
                if source_node_name == node_name:
                    feasible_platforms.append(platform_info)
                    continue
                
                # Rule 2: Server nodes with network connectivity are allowed
                # Rule 3: Other client nodes are NOT allowed (must be rejected)
                if node_name.startswith('client_node'):
                    # This is another client node - reject it (matches scheduler logic)
                    logger.debug(f"  Rejecting platform on {node_name} - tasks cannot be placed on other client nodes")
                    continue
                
                # Check network connectivity for server nodes
                node_config = next((n for n in nodes if n['node_name'] == node_name), None)
                if node_config and source_node_name in node_config.get('network_map', {}):
                    # Server node with network connectivity - allow it
                    feasible_platforms.append(platform_info)
                else:
                    # Server node without network connectivity - reject it
                    logger.debug(f"  Rejecting platform on {node_name} - no network connectivity from {source_node_name}")
            
            if feasible_platforms:
                tasks.append({
                    'task_id': task_id,
                    'task_type': task_type_name,
                    'source_node': source_node_name,
                    'feasible_platforms': feasible_platforms
                })
                task_id += 1
            else:
                print(f"❌ Abort: No feasible platforms for workload task index {task_id} ({task_type_name}). Skipping this sample.")
                return []
    
    # Determinism check: ensure we have a placement decision for every workload task
    if len(tasks) != expected_task_count:
        print(f"this is shit and should not happen ❌ Abort: Determinism check failed. Expected {expected_task_count} tasks, built {len(tasks)} tasks. Skipping this sample.")
        return []

    logger.info(f"Found {len(tasks)} tasks with feasible placements")
    print(f"Found {len(tasks)} tasks with feasible placements")
    for task in tasks:
        feasible_replicas = [(p['node_id'], p['platform_id']) for p in task['feasible_platforms']]
        logger.info(f"  Task {task['task_id']} ({task['task_type']}): {len(task['feasible_platforms'])} feasible platforms: {feasible_replicas}")
        print(f"  Task {task['task_id']} ({task['task_type']}): {len(task['feasible_platforms'])} feasible platforms: {feasible_replicas}")
    
    # Analyze overlap between tasks
    logger.info("=== Analyzing replica overlap between tasks ===")
    all_replicas = {}  # replica -> list of task_ids that can use it
    for task in tasks:
        for platform_info in task['feasible_platforms']:
            replica = (platform_info['node_id'], platform_info['platform_id'])
            if replica not in all_replicas:
                all_replicas[replica] = []
            all_replicas[replica].append(task['task_id'])
    
    # Find replicas that are shared by multiple tasks
    shared_replicas = {replica: task_ids for replica, task_ids in all_replicas.items() if len(task_ids) > 1}
    if shared_replicas:
        logger.warning(f"Found {len(shared_replicas)} replicas shared by multiple tasks:")
        for replica, task_ids in shared_replicas.items():
            logger.warning(f"  Replica {replica} is feasible for tasks: {task_ids}")
        print(f"⚠️  Found {len(shared_replicas)} replicas shared by multiple tasks (this can cause uniqueness constraint failures)")
    else:
        logger.info("No replicas are shared between tasks - uniqueness constraint should be satisfiable")
    
    # Check if there are enough unique replicas for all tasks
    unique_replicas_count = len(all_replicas)
    if unique_replicas_count < len(tasks):
        logger.error(f"❌ CRITICAL: Only {unique_replicas_count} unique replicas available for {len(tasks)} tasks")
        logger.error(f"   This makes it IMPOSSIBLE to satisfy the uniqueness constraint")
        print(f"❌ CRITICAL: Only {unique_replicas_count} unique replicas available for {len(tasks)} tasks")
        print(f"   Uniqueness constraint requires {len(tasks)} unique replicas, but only {unique_replicas_count} are available")
    else:
        logger.info(f"✓ {unique_replicas_count} unique replicas available for {len(tasks)} tasks (sufficient for uniqueness constraint)")
    
    # Generate all combinations ensuring each task has a unique replica
    logger.info("Generating placement combinations with unique replica constraint...")
    # Allow skipping datasets with too many combinations (configurable via environment or config)
    skip_threshold = int(os.environ.get('MAX_PLACEMENT_COMBINATIONS_SKIP', '0'))  # 0 = never skip
    combinations = generate_all_combinations_with_unique_replicas(
        tasks, 
        max_combinations_warning=100000,
        skip_if_exceeds=skip_threshold if skip_threshold > 0 else None
    )
    
    if not combinations:
        logger.warning("No placement combinations generated (dataset skipped due to size)")
        print("⚠️  Dataset skipped: too many placement combinations")
    
    logger.info(f"Generated {len(combinations)} valid placement combinations (all tasks have unique replicas)")
    print(f"Generated {len(combinations)} valid placement combinations (all tasks have unique replicas)")
    return combinations


def generate_all_combinations_with_unique_replicas(
    tasks: List[Dict],
    max_combinations_warning: int = 100000,
    skip_if_exceeds: Optional[int] = None
) -> List[Dict[int, Tuple[int, int]]]:
    """
    Generate all possible placement combinations for tasks.
    Ensures that each task gets a unique replica (no two tasks share the same (node_id, platform_id)).
    Only returns placements where ALL tasks have valid replicas.
    
    IMPORTANT: This function does NOT cap combinations because:
    1. Training requires ALL placements to find the true optimal
    2. Capping would create incorrect training labels (suboptimal labeled as optimal)
    3. Negative sampling in StructuredRegretLoss needs all valid placements
    
    Args:
        tasks: List of task dictionaries with feasible_platforms
        max_combinations_warning: Warn if combinations exceed this (default: 100000)
        skip_if_exceeds: If set, return empty list if combinations exceed this threshold (skips dataset)
    
    Returns:
        List of placement plans, each mapping task_id to (node_id, platform_id).
        Returns empty list if skip_if_exceeds is set and threshold is exceeded.
    """
    if not tasks:
        return [{}]
    
    # Calculate total possible combinations (before uniqueness constraint)
    total_possible = 1
    for task in tasks:
        total_possible *= len(task['feasible_platforms'])
    
    print(f"Total possible combinations (before uniqueness constraint): {total_possible}")
    
    # Check if we should skip this dataset
    if skip_if_exceeds is not None and total_possible > skip_if_exceeds:
        print(f"⚠️  SKIPPING DATASET: Combinations ({total_possible:,}) exceed threshold ({skip_if_exceeds:,})")
        print(f"   This dataset would take too long to process. Consider:")
        print(f"   - Reducing number of tasks")
        print(f"   - Reducing feasible platforms per task")
        print(f"   - Increasing skip_if_exceeds threshold")
        return []
    
    # Warn if combinations are very large
    if total_possible > max_combinations_warning:
        print(f"⚠️  WARNING: Large search space detected ({total_possible:,} combinations)")
        print(f"   This dataset may take a long time to process.")
        print(f"   Estimated time: ~{total_possible / 100:.0f} seconds at 100 sim/s")
    
    combinations = []
    used_replicas = set()  # Track (node_id, platform_id) tuples that are already used
    
    # Recursive function to generate combinations with uniqueness constraint
    def generate_recursive(task_index: int, current_placement: Dict[int, Tuple[int, int]], used: set) -> None:
        if task_index >= len(tasks):
            # All tasks have been assigned unique replicas
            # Verify that all tasks are in the placement and all values are valid
            if len(current_placement) != len(tasks):
                return
            
            # Validate all placements have valid integer values
            for task_id, (node_id, platform_id) in current_placement.items():
                if not isinstance(node_id, int) or not isinstance(platform_id, int):
                    return
                if node_id < 0 or platform_id < 0:
                    return
            
            combinations.append(current_placement.copy())
            return
        
        task = tasks[task_index]
        
        # CRITICAL: If this task has no feasible platforms, we cannot generate valid placements
        if not task['feasible_platforms']:
            return  # Backtrack - this path cannot lead to a valid placement
        
        # Try each feasible platform for this task
        found_valid_replica = False
        for platform_info in task['feasible_platforms']:
            # Validate platform_info has valid node_id and platform_id
            node_id = platform_info.get('node_id')
            platform_id = platform_info.get('platform_id')
            
            if node_id is None or platform_id is None:
                continue
            
            if not isinstance(node_id, int) or not isinstance(platform_id, int):
                continue
            
            replica = (node_id, platform_id)
            
            # Skip if this replica is already used by another task
            if replica in used:
                continue
            
            # Add this replica to the current placement and used set
            current_placement[task['task_id']] = replica
            used.add(replica)
            found_valid_replica = True
            
            # Recursively generate placements for remaining tasks
            generate_recursive(task_index + 1, current_placement, used)
            
            # Backtrack: remove this replica before trying the next one
            del current_placement[task['task_id']]
            used.remove(replica)
        
        # If no valid replica was found for this task (all were used), backtrack
        if not found_valid_replica:
            return  # Backtrack - this path cannot lead to a valid placement
    
    # Start recursive generation
    generate_recursive(0, {}, used_replicas)
    
    print(f"Valid combinations after uniqueness constraint: {len(combinations)}")
    return combinations


def process_capture_system_state(args):
    """Process system state capture in a separate process to avoid blocking."""
    (
        sample,
        mapping,
        infra_config,
        sim_inputs,
        workload_events,
        replica_plan,
        output_dir,
        infrastructure_file,
    ) = args

    logger = setup_logging(output_dir)
    logger.info("=== Starting system state capture in separate process ===")
    
    try:
        # Call the actual capture function
        active_replicas = capture_system_state_from_first_task(
            sample,
            mapping,
            infra_config,
            sim_inputs,
            workload_events,
            replica_plan,
            output_dir,
            infrastructure_file=infrastructure_file
        )
        
        logger.info("=== System state capture completed in separate process ===")
        return active_replicas
        
    except Exception as e:
        logger.error(f"Error in system state capture process: {str(e)}")
        logger.exception(e)
        return None


def process_sample_with_placement(args):
    """Process a single sample with a specific placement plan, reusing precomputed per-sample artifacts."""
    (
        i,
        sample,
        placement_plan,
        base_nodes,
        output_dir,
        sim_inputs,
        mapping,
        infra_config,
        flattened_workloads,
        replica_plan,
        apps,
        infrastructure_file,
    ) = args

    logger = setup_logging(output_dir)
    logger.info(f"Processing sample {i + 1} with placement plan {len(placement_plan)} tasks")
    print(f"[executecosim] Sample {i + 1}: launching placement with {len(placement_plan)} tasks")
    if infrastructure_file:
        print(f"[executecosim] Sample {i + 1}: Infrastructure file: {infrastructure_file}")
        if infrastructure_file.exists():
            print(f"[executecosim] Sample {i + 1}: ✓ Infrastructure file exists")
        else:
            print(f"[executecosim] Sample {i + 1}: ⚠️  Infrastructure file missing!")
    else:
        print(f"[executecosim] Sample {i + 1}: ⚠️  No infrastructure file - using non-deterministic mode")

    try:
        # Prepare infrastructure configuration with the specific placement plan
        # Reuse the same node/network topology via base_nodes and keep the same replica plan
        logger.info(f"Sample {i + 1}: Preparing simulation configuration...")
        sim_config = prepare_simulation_config(
            sample,
            mapping,
            infra_config,
            placement_plan,
            replica_plan=replica_plan,
            base_nodes=base_nodes,
            infrastructure_file=infrastructure_file,
        )
        logger.info(f"Sample {i + 1}: Simulation configuration prepared")
        
        # Verify deterministic infrastructure was loaded
        if 'deterministic_replica_placements' in sim_config:
            logger.info(f"Sample {i + 1}: Deterministic infrastructure loaded successfully")
            print(f"[executecosim] Sample {i + 1}: ✓ Deterministic infrastructure loaded successfully")
        else:
            logger.warning(f"Sample {i + 1}: No deterministic infrastructure in config (using legacy mode)")
            print(f"[executecosim] Sample {i + 1}: ⚠️  No deterministic infrastructure in config (using legacy mode)")

        # Combine infrastructure and workload configurations
        full_config = {
            "infrastructure": sim_config,
            "workload": flattened_workloads,
        }

        # Execute simulation with additional inputs
        cache_policy = 'fifo'
        task_priority = 'fifo'
        keep_alive = KEEP_ALIVE
        queue_length = QUEUE_LENGTH
        scheduling_strategy = 'determined_determined'

        logger.info(f"Sample {i + 1}: Starting simulation execution...")
        result = execute_simulation(
            full_config,
            sim_inputs,
            scheduling_strategy,
            model_locations={},
            models={},
            cache_policy=cache_policy,
            task_priority=task_priority,
            keep_alive=keep_alive,
            queue_length=queue_length,
        )
        logger.info(f"Sample {i + 1}: Simulation execution completed")
        result['sample'] = {
            'apps': apps,
            'sample': sample.tolist(),
            'mapping': mapping,
            'infra_config': infra_config,
            'sim_inputs': sim_inputs,
            'scheduling_strategy': scheduling_strategy,
            'cache_policy': cache_policy,
            'task_priority': task_priority,
            'keep_alive': keep_alive,
            'queue_length': queue_length,
            'placement_plan': placement_plan,
        }

        # Calculate RTT directly from stats (avoid file read later)
        stats = result.get('stats', {})
        task_results = stats.get('taskResults', [])
        rtt = 0.0
        counted = False
        for tr in task_results:
            task_id = tr.get('taskId')
            if task_id is not None and task_id >= 0:
                rtt += float(tr.get('elapsedTime', 0))
                counted = True
        rtt_value = float(rtt) if counted else float('inf')

        # Prepare result file path (but don't write yet - will batch write)
        import hashlib, uuid
        placement_key = json.dumps(sorted(placement_plan.items()))
        placement_hash = hashlib.sha1(placement_key.encode('utf-8')).hexdigest()[:16]
        unique_suffix = uuid.uuid4().hex[:8]
        result_file = output_dir / f"simulation_{i + 1}_placement_{placement_hash}_{unique_suffix}.json"

        logger.info(f"Sample {i + 1}: Completed simulation (RTT: {rtt_value:.3f}s)")
        logger.info(f"=== Completed simulation for sample {i + 1} ===")
        # Return both file path and RTT, plus result data for batching
        return result_file, rtt_value, result

    except Exception as e:
        logger.error(f"Sample {i + 1}: Error in simulation: {str(e)}")
        logger.exception(e)
        return None, float('inf'), None


def process_placement_fast(placement_plan: Dict[int, Tuple[int, int]]) -> Tuple[Optional[Path], float, Optional[Dict]]:
    """
    Optimized worker function that uses shared data from _worker_shared_data.
    
    This function is called with ONLY the placement_plan - all other data
    is accessed from the global _worker_shared_data dict initialized by _init_worker.
    This dramatically reduces pickling overhead for each task submission.
    
    OPTIMIZATION: Only writes result file if RTT is better than current best.
    This reduces I/O by 99%+ for large datasets.
    
    Args:
        placement_plan: Dict mapping task_id -> (node_id, platform_id)
    
    Returns:
        Tuple of (result_file_path, rtt_value, placement_plan)
        - result_file_path is None if this result was not written (worse than best)
        - rtt_value is always returned (for tracking and placements.jsonl)
        - placement_plan is always returned (for placements.jsonl)
    """
    global _worker_shared_data, QUIET_MODE
    
    # Access shared data (loaded once per worker process)
    sim_inputs = _worker_shared_data['sim_inputs']
    infra_config = _worker_shared_data['infra_config']
    base_nodes = _worker_shared_data['base_nodes']
    flattened_workloads = _worker_shared_data['flattened_workloads']
    replica_plan = _worker_shared_data['replica_plan']
    apps = _worker_shared_data['apps']
    infrastructure_file = _worker_shared_data['infrastructure_file']
    sample = _worker_shared_data['sample']
    mapping = _worker_shared_data['mapping']
    output_dir = _worker_shared_data['output_dir']
    best_rtt_value = _worker_shared_data.get('best_rtt_value')
    best_rtt_lock = _worker_shared_data.get('best_rtt_lock')
    
    try:
        # Prepare infrastructure configuration with the specific placement plan
        sim_config = prepare_simulation_config(
            sample,
            mapping,
            infra_config,
            placement_plan,
            replica_plan=replica_plan,
            base_nodes=base_nodes,
            infrastructure_file=infrastructure_file,
        )

        # Combine infrastructure and workload configurations
        full_config = {
            "infrastructure": sim_config,
            "workload": flattened_workloads,
        }

        # Execute simulation
        result = execute_simulation(
            full_config,
            sim_inputs,
            'determined_determined',
            model_locations={},
            models={},
            cache_policy='fifo',
            task_priority='fifo',
            keep_alive=KEEP_ALIVE,
            queue_length=QUEUE_LENGTH,
        )
        
        # Minimal result metadata (avoid storing large redundant data)
        result['sample'] = {
            'apps': apps,
            'placement_plan': placement_plan,
        }

        # Calculate RTT FIRST (before file I/O)
        stats = result.get('stats', {})
        task_results = stats.get('taskResults', [])
        rtt = 0.0
        counted = False
        for tr in task_results:
            task_id = tr.get('taskId')
            if task_id is not None and task_id >= 0:
                rtt += float(tr.get('elapsedTime', 0))
                counted = True
        rtt_value = float(rtt) if counted else float('inf')

        # OPTIMIZATION: Only write file if this is better than current best RTT
        # Use lock-free read first to minimize contention, then lock only if needed
        should_write = False
        if best_rtt_value is not None and best_rtt_lock is not None:
            # Lock-free read: check current best without acquiring lock (faster)
            # This avoids serializing all workers when most RTTs are worse
            current_best = best_rtt_value.value
            
            # Only acquire lock if we might have a better result (reduces contention)
            if rtt_value < current_best:
                # Now acquire lock for atomic check-and-update
                with best_rtt_lock:
                    # Re-check after acquiring lock (double-check pattern)
                    # Another worker might have updated it while we waited
                    current_best = best_rtt_value.value
                    if rtt_value < current_best:
                        best_rtt_value.value = rtt_value
                        should_write = True
        else:
            # Fallback: always write if shared state not available (shouldn't happen)
            should_write = True

        result_file = None
        if should_write:
            # Generate unique result file path and write result to disk
            # This avoids passing large result objects through IPC (memory optimization)
            placement_key = json_dumps(sorted(placement_plan.items()))
            placement_hash = hashlib.sha1(placement_key.encode('utf-8')).hexdigest()[:16]
            unique_suffix = uuid.uuid4().hex[:8]
            result_file = output_dir / f"simulation_placement_{placement_hash}_{unique_suffix}.json"
            
            # Write full result to disk (worker-side) to avoid memory accumulation in main process
            with open(result_file, 'w') as f:
                json.dump(result, f, cls=DataclassJSONEncoder)

        # Return file path (None if not written), RTT, and placement plan
        # RTT and placement_plan are always returned (needed for placements.jsonl)
        return result_file, rtt_value, placement_plan

    except Exception as e:
        if not QUIET_MODE:
            print(f"[worker] Error in simulation: {str(e)}")
        return None, float('inf'), None


def execute_brute_force_optimized(
        apps: List[str],
        config_file: str,
        mapping_file: str,
        output_dir: Path,
        sample: np.ndarray,
        sim_input_path: Path,
        workload_base_file: str,
        max_workers: int,
        infrastructure_file: Path,
        quiet: bool = False,
        final_dataset_dir: Optional[Path] = None,
        early_termination_rtt: Optional[float] = None,
        early_termination_pct: Optional[float] = None
) -> List[str]:
    """
    Optimized brute force placement optimization.
    
    Key optimizations:
    1. No sample loop - processes single sample directly
    2. Uses worker initializer to share immutable data once per worker
    3. Minimal per-task data transfer (only placement_plan)
    4. Quiet mode support
    5. Uses orjson for faster serialization when available
    
    Args:
        apps: List of application names
        config_file: Path to infrastructure configuration file
        mapping_file: Path to mapping file
        output_dir: Output directory for results (temporary)
        sample: Single sample array (not array of samples)
        sim_input_path: Path to simulation input files
        workload_base_file: Path to workload base file
        max_workers: Maximum number of parallel workers
        infrastructure_file: Path to infrastructure.json (REQUIRED)
        quiet: If True, suppress per-placement logging
        final_dataset_dir: If provided, write progress/metadata files here instead of output_dir
        early_termination_rtt: If set, stop when RTT <= this value (saves time)
        early_termination_pct: If set, stop after checking this % of placements (0.0-1.0)
    
    Returns:
        List of result file paths
    """
    global QUIET_MODE
    QUIET_MODE = quiet
    
    time_started = time.time()
    
    _log(f"\n=== Starting Optimized Brute Force Placement ===")
    _log(f"Max workers: {max_workers}")
    _log(f"Infrastructure file: {infrastructure_file}")
    _log(f"Using orjson: {HAS_ORJSON}")
    
    logger = logging.getLogger('simulation')
    
    # Load all required data ONCE
    _log("Loading simulation inputs...")
    sim_inputs = load_simulation_inputs(sim_input_path)
    
    _log("Loading mapping file...")
    with open(mapping_file, 'rb') as f:
        mapping = pickle.load(f)
    
    _log("Loading infrastructure config...")
    with open(config_file, 'r') as f:
        infra_config = json.load(f)
    
    _log("Loading workload base...")
    with open(workload_base_file, 'r') as f:
        workload_base = json.load(f)
    
    # Prepare workloads
    workloads = prepare_workloads(sample, mapping, workload_base, apps)
    flattened_workloads = flatten_workloads(workloads)
    _log(f"Prepared {len(flattened_workloads['events'])} workload events")
    
    # Prepare infrastructure configuration
    sim_config = prepare_simulation_config(sample, mapping, infra_config, infrastructure_file=infrastructure_file)
    
    # Generate replica plan
    replica_plan = determine_replica_placement(sim_config, sim_inputs)
    
    base_nodes = sim_config['nodes']
    
    # Phase 1: Capture system state
    _log("\n[Phase 1] Capturing system state...")
    capture_args = (
        sample,
        mapping,
        infra_config,
        sim_inputs,
        flattened_workloads['events'],
        replica_plan,
        output_dir,
        infrastructure_file,
    )
    
    # Run capture in main process (no need for separate executor for single task)
    active_replicas = capture_system_state_from_first_task(
        sample, mapping, infra_config, sim_inputs,
        flattened_workloads['events'], replica_plan, output_dir,
        infrastructure_file=infrastructure_file
    )
    
    if active_replicas is None:
        raise RuntimeError("System state capture FAILED. Cannot proceed with brute-force optimization.")
    
    _log("✓ System state captured successfully")
    
    # Phase 2: Generate placement combinations
    _log("\n[Phase 2] Generating placement combinations...")
    placement_combinations = generate_brute_force_placement_combinations(
        flattened_workloads['events'],
        sim_config,
        sim_inputs,
        replica_plan,
        active_replicas=active_replicas
    )
    
    num_placements = len(placement_combinations)
    _log(f"Generated {num_placements} placement combinations")
    
    if not placement_combinations:
        # Create empty placements.jsonl to indicate infeasible scenario
        placements_file = output_dir / "placements.jsonl"
        placements_file.touch()
        _log("  No valid placement combinations - scenario is infeasible")
        return []  # Return empty list instead of raising exception
    
    # Phase 3: Execute simulations in parallel with worker initializer
    _log(f"\n[Phase 3] Executing {num_placements} simulations with {max_workers} workers...")
    
    best_rtt = float('inf')
    best_file = None  # Path to best result (written by worker, loaded at end if needed)
    rtts = []
    num_written = 0  # Track placements written to disk (streaming)
    
    # Create shared state for best RTT tracking across workers (for I/O optimization)
    manager = multiprocessing.Manager()
    best_rtt_value = manager.Value('d', float('inf'))  # 'd' = double (float)
    best_rtt_lock = manager.Lock()
    
    # Open placements file for streaming writes (avoid memory accumulation)
    placements_file = output_dir / "placements.jsonl"
    placements_fh = open(placements_file, 'w')
    elapsed_time = 0  # Initialize for finally block safety
    
    try:
        # Use worker initializer to share data once per worker (not per task!)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(
                sim_inputs,
                infra_config,
                base_nodes,
                flattened_workloads,
                replica_plan,
                apps,
                infrastructure_file,
                sample,
                mapping,
                output_dir,
                quiet,
                best_rtt_value,
                best_rtt_lock,
            )
        ) as executor:
            # Submit all placement plans - only the small placement_plan dict is pickled per task
            futures = {
                executor.submit(process_placement_fast, plan): idx 
                for idx, plan in enumerate(placement_combinations)
            }
            
            completed = 0
            timeout_per_placement = 2  # 2 seconds per placement (sims take ~10ms, 2s provides 200x safety margin)
            timed_out_count = 0
            
            # Calculate update interval once
            update_interval = max(1, min(1000, num_placements // 100))
            progress_dir = final_dataset_dir if final_dataset_dir else output_dir
            
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                placement_idx = futures[future]
                
                try:
                    # Add timeout to prevent infinite hangs
                    # Workers now return (result_file, rtt, placement_plan) - no large result dict
                    # result_file may be None if worker didn't write (worse than best RTT)
                    result_file, cur_rtt, placement_plan = future.result(timeout=timeout_per_placement)
                    
                    if placement_plan is None:
                        # Error case: placement_plan is None (worker failed)
                        # Still update progress even for None results
                        if completed % update_interval == 0 and progress_dir:
                            try:
                                progress_file = progress_dir / "placement_progress.txt"
                                elapsed = time.time() - time_started
                                rate = completed / elapsed if elapsed > 0 else 0
                                with open(progress_file, 'w') as pf:
                                    pf.write(f"{completed}/{num_placements}\n")
                                    pf.write(f"Rate: {rate:.1f} sim/s\n")
                                    if best_rtt < float('inf'):
                                        pf.write(f"Best RTT: {best_rtt:.3f}s\n")
                            except Exception:
                                pass
                        continue
                    
                    rtts.append(cur_rtt)
                    
                    # Track best result file (only if worker wrote a file)
                    # Note: result_file can be None if worker didn't write (worse RTT)
                    if result_file is not None:
                        # This is guaranteed to be better than previous best (worker checked)
                        # Update our local tracking
                        if cur_rtt < best_rtt:
                            best_rtt = cur_rtt
                            best_file = str(result_file)
                            
                            # Early termination: stop if we found a "good enough" RTT
                            if early_termination_rtt is not None and best_rtt <= early_termination_rtt:
                                if not quiet:
                                    _log(f"  Early termination: Found RTT {best_rtt:.3f}s <= {early_termination_rtt:.3f}s", force=True)
                                logger.info(f"Early termination triggered: RTT {best_rtt:.3f}s <= {early_termination_rtt:.3f}s")
                                # Cancel remaining futures
                                for remaining_future in futures:
                                    if remaining_future != future:
                                        remaining_future.cancel()
                                break
                    else:
                        # Worker didn't write file (worse RTT than best)
                        # Still update best_rtt if needed (worker may have updated shared value)
                        if cur_rtt < best_rtt:
                            best_rtt = cur_rtt
                            # Also check shared value in case another worker updated it
                            with best_rtt_lock:
                                if best_rtt_value.value < best_rtt:
                                    best_rtt = best_rtt_value.value
                    
                    # Stream write placement summary to disk (avoid memory accumulation)
                    # Write to placements.jsonl regardless of whether file was written
                    # This preserves all placement-RTT pairs for RTT hash table
                    summary = {"placement_plan": placement_plan, "rtt": cur_rtt}
                    placements_fh.write(json.dumps(summary, separators=(',', ':')) + '\n')
                    num_written += 1
                    
                    # Flush periodically to ensure data is written
                    if num_written % 1000 == 0:
                        placements_fh.flush()
                    
                except concurrent.futures.TimeoutError:
                    timed_out_count += 1
                    future.cancel()  # Cancel the hung future (doesn't stop running processes but marks as cancelled)
                    if not quiet:
                        _log(f"  Placement {placement_idx} timed out after {timeout_per_placement}s - skipping")
                    logger.warning(f"Placement {placement_idx} timed out after {timeout_per_placement}s")
                    # Continue to progress update below
                except Exception as e:
                    if not quiet:
                        _log(f"  Worker failed for placement {placement_idx}: {e}")
                    logger.warning(f"Worker failed for placement {placement_idx}: {e}")
                    # Continue to progress update below
                
                # Write placement progress (for ALL completions, including timeouts/errors)
                if completed % update_interval == 0 and progress_dir:
                    try:
                        progress_file = progress_dir / "placement_progress.txt"
                        elapsed = time.time() - time_started
                        rate = completed / elapsed if elapsed > 0 else 0
                        with open(progress_file, 'w') as pf:
                            pf.write(f"{completed}/{num_placements}\n")
                            pf.write(f"Rate: {rate:.1f} sim/s\n")
                            if best_rtt < float('inf'):
                                pf.write(f"Best RTT: {best_rtt:.3f}s\n")
                            if timed_out_count > 0:
                                pf.write(f"Timeouts: {timed_out_count}\n")
                    except Exception:
                        pass  # Don't fail on progress file write errors
                
                # Early termination: stop after checking X% of placements
                if early_termination_pct is not None and completed >= int(num_placements * early_termination_pct):
                    if not quiet:
                        _log(f"  Early termination: Checked {completed}/{num_placements} ({100*completed/num_placements:.1f}%) placements", force=True)
                    logger.info(f"Early termination triggered: Checked {100*early_termination_pct:.1f}% of placements")
                    # Cancel remaining futures
                    for remaining_future in futures:
                        if remaining_future != future:
                            remaining_future.cancel()
                    break
                
                # Progress update (every 10% or every 1000)
                if not quiet and (completed % max(1, num_placements // 10) == 0 or completed % 1000 == 0):
                    elapsed = time.time() - time_started
                    rate = completed / elapsed if elapsed > 0 else 0
                    _log(f"  Progress: {completed}/{num_placements} ({100*completed/num_placements:.1f}%) - {rate:.1f} sim/s - best RTT: {best_rtt:.3f}s")
        
        elapsed_time = time.time() - time_started
    
    finally:
        # Close the streaming placements file (ensure cleanup even on error)
        placements_fh.close()
        # Sync final best_rtt from shared value (in case workers updated it)
        with best_rtt_lock:
            if best_rtt_value.value < best_rtt:
                best_rtt = best_rtt_value.value
    
    if timed_out_count > 0:
        _log(f"\n[WARNING] {timed_out_count} placement(s) timed out and were skipped", force=True)
        logger.warning(f"{timed_out_count} placement(s) timed out during execution")
    
    # Write results
    _log(f"\n[Phase 4] Writing results...")
    
    # Count how many result files were actually written (I/O optimization impact)
    result_files_written = len(list(output_dir.glob("simulation_placement_*.json")))
    if num_placements > 0:
        io_reduction_pct = 100.0 * (1.0 - result_files_written / num_placements)
        _log(f"  I/O Optimization: {result_files_written}/{num_placements} result files written ({io_reduction_pct:.1f}% reduction)")
    
    # Placement summaries already written via streaming
    if num_written > 0:
        _log(f"  Saved {num_written} placement summaries (streamed)")
    
    # Find best result file (may need to search if best_file wasn't updated in main thread)
    # Workers write files only when they're better than current best, so we need to find
    # the file that corresponds to the final best_rtt
    result_paths = []
    
    # If we have a best_file, use it (most common case)
    if best_file is not None and Path(best_file).exists():
        optimal_file_path = output_dir / "simulation_1_optimal.json"
        try:
            # Copy the best result file to the canonical optimal path
            import shutil
            shutil.copy2(best_file, optimal_file_path)
            result_paths.append(str(optimal_file_path))
            
            # Write best.json sidecar
            best_info = {"file": os.path.basename(str(optimal_file_path)), "rtt": best_rtt}
            with open(output_dir / "best.json", 'w') as f:
                f.write(json_dumps(best_info))
            _log(f"  Saved optimal result: RTT={best_rtt:.3f}s")
        except Exception as e:
            _log(f"  ERROR: Failed to copy optimal result: {e}")
    elif best_rtt < float('inf') and rtts:
        # Fallback: search for file with matching RTT (shouldn't happen often)
        # This handles edge case where best_file wasn't tracked in main thread
        _log(f"  Searching for best result file (RTT={best_rtt:.3f}s)...")
        # Search through result files to find one with matching RTT
        for result_file in output_dir.glob("simulation_placement_*.json"):
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                stats = result_data.get('stats', {})
                task_results = stats.get('taskResults', [])
                file_rtt = sum(tr.get('elapsedTime', 0) for tr in task_results if tr.get('taskId') is not None and tr.get('taskId') >= 0)
                if abs(file_rtt - best_rtt) < 0.001:  # Float comparison tolerance
                    optimal_file_path = output_dir / "simulation_1_optimal.json"
                    import shutil
                    shutil.copy2(result_file, optimal_file_path)
                    result_paths.append(str(optimal_file_path))
                    best_info = {"file": os.path.basename(str(optimal_file_path)), "rtt": best_rtt}
                    with open(output_dir / "best.json", 'w') as f:
                        f.write(json_dumps(best_info))
                    _log(f"  Saved optimal result: RTT={best_rtt:.3f}s")
                    break
            except Exception:
                continue
    
    # Write final progress
    progress_dir = final_dataset_dir if final_dataset_dir else output_dir
    if progress_dir:
        progress_file = progress_dir / "placement_progress.txt"
        try:
            with open(progress_file, 'w') as pf:
                pf.write(f"{len(rtts)}/{num_placements}\n")
                pf.write(f"Rate: {len(rtts)/elapsed_time:.1f} sim/s\n")
                if rtts:
                    pf.write(f"Best RTT: {best_rtt:.3f}s\n")
                pf.write("Status: COMPLETE\n")
        except Exception:
            pass  # Don't fail on progress file write errors
        
        # Store num_placements in dataset directory for progress.txt logging
        try:
            metadata_file = progress_dir / "placement_metadata.json"
            with open(metadata_file, 'w') as mf:
                json.dump({"num_placements": num_placements, "completed": len(rtts)}, mf)
        except Exception:
            pass
    
    # Summary
    _log(f"\n=== Optimization Complete ===")
    _log(f"Total time: {elapsed_time:.1f}s")
    _log(f"Simulations: {len(rtts)}/{num_placements}")
    _log(f"Rate: {len(rtts)/elapsed_time:.1f} sim/s")
    if rtts:
        _log(f"RTT range: {min(rtts):.3f}s - {max(rtts):.3f}s")
        _log(f"Best RTT: {best_rtt:.3f}s")
    
    return result_paths


def _create_placement_tuple(
    sample_idx: int,
    sample: np.ndarray,
    placement_plan: Dict[int, Tuple[int, int]],
    base_nodes: List[Dict],
    output_dir: Path,
    sim_inputs: Dict[str, Any],
    mapping: Dict[int, str],
    infra_config: Dict[str, Any],
    flattened_workloads: Dict[str, Any],
    replica_plan: Dict[str, Any],
    apps: List[str],
    infrastructure_file: Optional[Path]
) -> Tuple:
    """
    Helper function to create placement tuple for worker submission.
    Extracted to avoid code duplication.
    """
    return (
        sample_idx,
        sample,
        placement_plan,
        base_nodes,
        output_dir,
        sim_inputs,
        mapping,
        infra_config,
        flattened_workloads,
        replica_plan,
        apps,
        infrastructure_file,
    )


def execute_brute_force_placement_optimization(
        apps: List[str],
        config_file: str,
        mapping_file: str,
        output_dir: Path,
        samples: np.ndarray,
        sim_input_path: Path,
        workload_base_file: str,
        max_workers: int,
        infrastructure_file: Optional[Path] = None
) -> List[str]:
    """
    Legacy brute force placement optimization.
    
    NOTE: This is the legacy implementation kept for backwards compatibility.
    For better performance, use execute_brute_force_optimized() which:
    - Uses worker initializer pattern (avoids redundant data pickling)
    - Removes unnecessary sample loop
    - Supports quiet mode
    
    Args:
        apps: List of application names
        config_file: Path to infrastructure configuration file
        mapping_file: Path to mapping file
        output_dir: Output directory for results
        samples: Array of samples (only first sample is used)
        sim_input_path: Path to simulation input files
        workload_base_file: Path to workload base file
        max_workers: Maximum number of parallel workers
    
    Returns:
        List of result file paths
    """
    # NOTE: Legacy implementation processes only first sample
    sample = samples[0]
    sample_idx = 0
    
    print(f"\n=== Starting Brute Force Placement Optimization (Legacy) ===")
    print(f"Max workers: {max_workers}")
    if infrastructure_file:
        print(f"Infrastructure file: {infrastructure_file}")

    logger = logging.getLogger('simulation')
    logger.info("=== Starting Brute Force Placement Optimization (Legacy) ===")
    
    time_started = time.time()
    result_paths = []
    
    try:
        # Load required data
        logger.info("Loading simulation inputs...")
        sim_inputs = load_simulation_inputs(sim_input_path)
        
        logger.info("Loading mapping file...")
        with open(mapping_file, 'rb') as f:
            mapping = pickle.load(f)
        
        logger.info("Loading infrastructure config...")
        with open(config_file, 'r') as f:
            infra_config = json.load(f)
        
        logger.info("Loading workload base...")
        with open(workload_base_file, 'r') as f:
            workload_base = json.load(f)
        
        # Prepare workloads
        logger.info("Preparing workloads...")
        workloads = prepare_workloads(sample, mapping, workload_base, apps)
        flattened_workloads = flatten_workloads(workloads)
        logger.info(f"Prepared {len(flattened_workloads['events'])} workload events")
        
        # Prepare infrastructure configuration
        sim_config = prepare_simulation_config(sample, mapping, infra_config, infrastructure_file=infrastructure_file)
        
        # Generate replica plan
        logger.info("Generating replica plan...")
        replica_plan = determine_replica_placement(sim_config, sim_inputs)
        
        # Phase 1: Capture system state
        logger.info("Phase 1 - Capturing system state...")
        print("[executecosim] Phase 1 - Capturing system state...")
        
        capture_tuple = (
            sample, mapping, infra_config, sim_inputs,
            flattened_workloads['events'], replica_plan, output_dir, infrastructure_file,
        )
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as capture_executor:
            future = capture_executor.submit(process_capture_system_state, capture_tuple)
            try:
                active_replicas = future.result(timeout=3600)
            except Exception as e:
                logger.error(f"System state capture failed: {e}")
                active_replicas = None
        
        if active_replicas is None:
            raise RuntimeError("System state capture FAILED. Cannot proceed.")
        
        print("✓ System state captured successfully")
        
        # Phase 2: Generate placement combinations
        logger.info("Phase 2 - Generating placement combinations...")
        print("[executecosim] Phase 2 - Generating placement combinations...")
        placement_combinations = generate_brute_force_placement_combinations(
            flattened_workloads['events'], sim_config, sim_inputs, replica_plan,
            active_replicas=active_replicas
        )
        
        print(f"Generated {len(placement_combinations)} placement combinations")
        
        if not placement_combinations:
            raise RuntimeError("No valid placement combinations found")
        
        # Phase 3: Execute simulations
        logger.info(f"Phase 3 - Executing {len(placement_combinations)} simulations...")
        print(f"[executecosim] Phase 3 - Executing {len(placement_combinations)} simulations...")
        
        base_nodes = sim_config['nodes']
        best_file: Optional[str] = None
        best_rtt: float = float('inf')
        best_result_data: Optional[Dict[str, Any]] = None
        placement_summaries: List[Dict[str, Any]] = []
        sample_rtts = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            next_placement_idx = 0
            total_completed = 0
            
            # Submit initial batch
            for _ in range(min(max_workers, len(placement_combinations))):
                if next_placement_idx >= len(placement_combinations):
                    break
                placement_plan = placement_combinations[next_placement_idx]
                placement_tuple = _create_placement_tuple(
                    sample_idx, sample, placement_plan, base_nodes, output_dir,
                    sim_inputs, mapping, infra_config, flattened_workloads,
                    replica_plan, apps, infrastructure_file
                )
                future = executor.submit(process_sample_with_placement, placement_tuple)
                futures[future] = next_placement_idx
                next_placement_idx += 1
            
            # Process results and submit new tasks
            while futures:
                done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                
                for future in done:
                    total_completed += 1
                    placement_idx = futures.pop(future)
                    
                    try:
                        result_file, cur_rtt_value, result_data = future.result()
                        
                        # Submit next task
                        if next_placement_idx < len(placement_combinations):
                            placement_plan = placement_combinations[next_placement_idx]
                            placement_tuple = _create_placement_tuple(
                                sample_idx, sample, placement_plan, base_nodes, output_dir,
                                sim_inputs, mapping, infra_config, flattened_workloads,
                                replica_plan, apps, infrastructure_file
                            )
                            new_future = executor.submit(process_sample_with_placement, placement_tuple)
                            futures[new_future] = next_placement_idx
                            next_placement_idx += 1
                        
                        if result_file is None or result_data is None:
                            continue
                        
                        sample_rtts.append(cur_rtt_value)
                        
                        # Track best result
                        if cur_rtt_value < best_rtt:
                            best_rtt = cur_rtt_value
                            best_result_data = result_data
                            best_file = str(result_file)
                        
                        # Keep placement summary
                        placement = result_data.get('sample', {}).get('placement_plan', {})
                        placement_summaries.append({"placement_plan": placement, "rtt": cur_rtt_value})
                        
                    except Exception as e:
                        logger.error(f"Worker failed for placement {placement_idx}: {e}")
                        # Submit next task anyway
                        if next_placement_idx < len(placement_combinations):
                            placement_plan = placement_combinations[next_placement_idx]
                            placement_tuple = _create_placement_tuple(
                                sample_idx, sample, placement_plan, base_nodes, output_dir,
                                sim_inputs, mapping, infra_config, flattened_workloads,
                                replica_plan, apps, infrastructure_file
                            )
                            new_future = executor.submit(process_sample_with_placement, placement_tuple)
                            futures[new_future] = next_placement_idx
                            next_placement_idx += 1
            
            logger.info(f"Completed {total_completed}/{len(placement_combinations)} simulations")
            
            # Write placement summaries
            if placement_summaries:
                placements_file = output_dir / "placements.jsonl"
                with open(placements_file, 'w') as f:
                    for summary in placement_summaries:
                        f.write(json.dumps(summary, separators=(',', ':')) + '\n')
                logger.info(f"Saved {len(placement_summaries)} placement summaries")
            
            # Write best result
            if best_result_data is not None:
                optimal_file_path = output_dir / f"simulation_{sample_idx + 1}_optimal.json"
                with open(optimal_file_path, 'w') as f:
                    json.dump(best_result_data, f, indent=2, cls=DataclassJSONEncoder)
                best_file = str(optimal_file_path)
                logger.info(f"Saved optimal result (RTT: {best_rtt:.3f}s)")
        
        # Record result
        if best_file is not None:
            result_paths.append(best_file)
            best_info = {"file": os.path.basename(best_file), "rtt": best_rtt}
            with open(output_dir / "best.json", 'w') as f:
                json.dump(best_info, f)
            logger.info(f"Saved best.json")
        
        print(f"Completed {len(placement_combinations)} simulations")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.exception(e)
        print(f"Error: {str(e)}")
    
    elapsed_time = time.time() - time_started
    logger.info(f"Total elapsed time: {elapsed_time:.2f} seconds")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"\n=== Brute Force Placement Optimization Complete ===")
    print(f"Total result files generated: {len(result_paths)}")
    
    return result_paths


def main():
    """
    Main entry point for brute-force co-simulation.
    
    Usage:
        python -m src.executecosimulation --brute-force --infrastructure <path> [--quiet]
        
    Arguments:
        --brute-force: Enable brute-force placement optimization
        --infrastructure <path>: Path to infrastructure.json file (required for brute-force)
        --quiet: Suppress per-placement logging (faster execution)
        --legacy: Use legacy (non-optimized) brute-force implementation
    """
    global QUIET_MODE
    
    # Configuration paths
    base_dir = Path("simulation_data")
    sim_input_path = Path("data/nofs-ids")
    samples_file = base_dir / "lhs_samples_simple.npy"
    mapping_file = base_dir / "lhs_samples_simple_mapping.pkl"
    config_file = base_dir / "space_with_network.json"
    # python -m src.generator -d data/nofs-ids --generate-traces --rps 10 --seconds 10
    workload_base_file = "data/nofs-ids/traces/workload-10.json"
    output_dir = base_dir / "initial_results_simple"
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse arguments
    use_brute_force = '--brute-force' in sys.argv
    quiet_mode = '--quiet' in sys.argv
    use_legacy = '--legacy' in sys.argv
    QUIET_MODE = quiet_mode
    
    cpu_count = os.cpu_count()
    max_workers = cpu_count - 1 if cpu_count else 1
    
    if not quiet_mode:
        print(f"CPU count: {cpu_count}, using {max_workers} workers")
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse infrastructure file argument
        infrastructure_file = None
        if '--infrastructure' in sys.argv:
            idx = sys.argv.index('--infrastructure')
            if idx + 1 < len(sys.argv):
                infrastructure_file = Path(sys.argv[idx + 1])
                if not quiet_mode:
                    print(f"[executecosim] Infrastructure file: {infrastructure_file}")
                    if infrastructure_file.exists():
                        print(f"[executecosim] ✓ Infrastructure file exists")
                    else:
                        print(f"[executecosim] ⚠️  WARNING: Infrastructure file does not exist")

        if use_brute_force:
            # Validate infrastructure file
            if infrastructure_file is None or not infrastructure_file.exists():
                raise RuntimeError(
                    "[executecosim] Brute-force co-simulation requires a valid "
                    "--infrastructure JSON file. No usable file was provided."
                )
            
            # Load samples (use first sample only - no sample loop)
            samples = np.load(samples_file)
            sample = samples[0]  # Single sample - no loop needed
            
            # Load config to get app names
            with open(config_file, 'r') as f:
                infra_config = json.load(f)
            apps = list(infra_config['wsc'].keys())
            
            if use_legacy:
                # Use legacy implementation (for comparison/fallback)
                if not quiet_mode:
                    print("[executecosim] Using LEGACY brute-force implementation")
                logger.info("Using legacy brute force placement optimization")
                reactive_results_paths = execute_brute_force_placement_optimization(
                    apps, str(config_file), str(mapping_file), output_dir, samples,
                    sim_input_path, workload_base_file, max_workers,
                    infrastructure_file=infrastructure_file
                )
            else:
                # Use optimized implementation (default)
                if not quiet_mode:
                    print("[executecosim] Using OPTIMIZED brute-force implementation")
                logger.info("Using optimized brute force placement optimization")
                reactive_results_paths = execute_brute_force_optimized(
                    apps, str(config_file), str(mapping_file), output_dir, sample,
                    sim_input_path, workload_base_file, max_workers,
                    infrastructure_file=infrastructure_file,
                    quiet=quiet_mode
                )
            
            logger.info("Completed all simulations")
            logger.info(f'All files can be found under {output_dir}')
        else:
            print("[executecosim] No action specified. Use --brute-force to run optimization.")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
