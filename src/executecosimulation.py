"""
Brute Force Placement Optimization Simulator

This script provides two execution modes:
1. Regular reactive execution (default)
2. Brute force placement optimization

Usage:
    # Regular execution
    python -m src.executecosimulation
    
    # Brute force placement optimization with default 50 combinations per sample
    python -m src.executecosimulation --brute-force
    
    # Brute force placement optimization with custom number of combinations
    python -m src.executecosimulation --brute-force 100

Arguments:
    --brute-force: Enable brute force placement optimization
"""

import json
import logging
import math
import os
import pickle
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional

import concurrent.futures
import json
from pathlib import Path

import numpy as np
import xgboost

# Assuming increase_events is imported from your previous script
from src.eventgenerator import increase_events_of_app
from src.motivational.constants import KEEP_ALIVE, QUEUE_LENGTH
from src.placement.executor import execute_sim
from src.placement.model import SimulationData, DataclassJSONEncoder
from src.train import train_model, save_models

# GNN ModelConfig class (needed for loading saved PyTorch models)
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for the improved GNN model"""
    # Model architecture
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.2
    attention_heads: int = 4
    
    # Training
    learning_rate: float = 0.001
    batch_size: int = 64
    num_epochs: int = 10
    patience: int = 8
    min_delta: float = 1e-4
    
    # Target transformation
    use_log_latency: bool = True
    use_classification: bool = False
    num_latency_buckets: int = 5
    multi_task: bool = False  # Predict latency + energy
    
    # Data processing
    normalize_features: bool = True
    max_graph_size: int = 50  # Limit graph size for memory

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
    
    This function moves the replica placement decision logic from simulation.py
    to executecosimulation.py, while keeping the actual creation in simulation.py.
    
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
    
    # Handle "all" values
    if preinit_clients == "all":
        preinit_clients = [f"client_node{i}" for i in range(10)]  # Default to 10 clients
    if preinit_servers == "all":
        preinit_servers = [f"node{i}" for i in range(10)]  # Default to 10 servers
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
        'replicas_config': replicas_config
    }
    
    print(f"Replica placement plan created")
    return replica_plan


def prepare_simulation_config(
        sample: np.ndarray,
        mapping: Dict[int, str],
        original_config: Dict[str, Any],
        placement_plan: Optional[Dict[int, Tuple[int, int]]] = None,
        replica_plan: Optional[Dict[str, Any]] = None,
        base_nodes: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Prepare simulation configuration from a sample."""
    reverse_mapping = create_reverse_mapping(mapping)

    # Extract network bandwidth
    network_bandwidth = sample[reverse_mapping['network_bandwidth']]

    # Get client and server node counts directly from config
    client_nodes_count = original_config['nodes']['client_nodes']['count']
    server_nodes_count = original_config['nodes']['server_nodes']['count']

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
        "scheduler": original_config.get('scheduler', {})
    }

    if base_nodes is not None and len(base_nodes) > 0:
        # Reuse provided nodes (and their network maps) to keep topology consistent
        infrastructure_config['nodes'] = [deepcopy(n) for n in base_nodes]
    else:
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
        network_maps = generate_network_latencies(infrastructure_config['nodes'], original_config)
            
        # Assign network maps to nodes
        for node in infrastructure_config['nodes']:
            node['network_map'] = network_maps[node['node_name']]

    # Debug: Print infrastructure node names
    print(f"\nInfrastructure nodes created:")
    for node in infrastructure_config['nodes']:
        print(f"  {node['node_name']}: {len(node['network_map'])} network connections")

    # Add placement plan to infrastructure config if provided
    if placement_plan is not None:
        # Make placements available for scheduler
        infrastructure_config['forced_placements'] = placement_plan
        print(f"Added placement plan with {len(placement_plan)} task placements")

    # Pass replica plan if provided
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

    print("execute_simulation")
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


def validate_workload_nodes(workload_events: List[Dict], infrastructure_nodes: List[Dict]) -> None:
    """Validate that all workload events reference valid infrastructure nodes."""
    infrastructure_node_names = {node['node_name'] for node in infrastructure_nodes}
    
    invalid_nodes = set()
    for event in workload_events:
        node_name = event.get('node_name')
        if node_name and node_name not in infrastructure_node_names:
            invalid_nodes.add(node_name)
    
    if invalid_nodes:
        print(f"Warning: Workload references non-existent nodes: {invalid_nodes}")
        print(f"Available infrastructure nodes: {sorted(infrastructure_node_names)}")
        workload_node_names = {event.get('node_name') for event in workload_events if event.get('node_name') is not None}
        # Filter out None values before sorting
        workload_node_names_filtered = {name for name in workload_node_names if name is not None}
        print(f"Workload node names: {sorted(workload_node_names_filtered)}")


def calculate_resource_contention(
        platform_info: Dict[str, Any],
        contention_config: Dict[str, Any]
) -> float:
    """
    Calculate resource contention score for a platform based on configuration.
    
    Args:
        platform_info: Platform information including node and platform details
        contention_config: Resource contention configuration from infrastructure
    
    Returns:
        Contention score between 0.0 (no contention) and 1.0 (high contention)
    """
    if not contention_config.get('simulate_contention', False):
        return 0.0
    
    # Base contention from configuration
    memory_fragmentation = contention_config.get('memory_fragmentation', 0.0)
    platform_utilization = contention_config.get('platform_utilization', 0.0)
    background_tasks = contention_config.get('background_tasks', 0)
    
    # Calculate contention based on platform type and node characteristics
    platform_type = platform_info.get('platform_type', 'unknown')
    node_name = platform_info.get('node_name', 'unknown')
    
    # Different contention levels for different platform types
    platform_contention_multiplier = {
        'rpiCpu': 1.2,      # Raspberry Pi CPUs are more resource-constrained
        'xavierCpu': 1.0,   # Xavier CPUs have moderate contention
        'xavierGpu': 0.8,   # GPUs typically have less contention
        'xavierDla': 0.9,   # DLA has moderate contention
        'pynqFpga': 1.1     # FPGA has slightly higher contention
    }
    
    multiplier = platform_contention_multiplier.get(platform_type, 1.0)
    
    # Calculate total contention score
    contention_score = (
        memory_fragmentation * 0.4 +  # Memory fragmentation impact
        platform_utilization * 0.4 +  # Platform utilization impact
        min(background_tasks * 0.05, 0.2)  # Background tasks impact (capped at 0.2)
    ) * multiplier
    
    # Ensure score is between 0.0 and 1.0
    return min(max(contention_score, 0.0), 1.0)


def is_platform_feasible_with_contention(
        platform_info: Dict[str, Any],
        contention_config: Dict[str, Any],
        max_contention_threshold: float = 0.8
) -> bool:
    """
    Check if a platform is feasible considering resource contention.
    
    Args:
        platform_info: Platform information
        contention_config: Resource contention configuration
        max_contention_threshold: Maximum allowed contention score (0.0-1.0)
    
    Returns:
        True if platform is feasible, False otherwise
    """
    if not contention_config.get('simulate_contention', False):
        return True
    
    contention_score = calculate_resource_contention(platform_info, contention_config)
    return contention_score <= max_contention_threshold


def generate_brute_force_placement_combinations(
        workload_events: List[Dict],
        infrastructure_config: Dict[str, Any],
        sim_inputs: Dict[str, Any],
        replica_plan: Dict[str, Any],
        max_combinations: Optional[int] = 100,
        enable_resource_contention: bool = True,
        max_contention_threshold: float = 0.8
) -> List[Dict[int, Tuple[int, int]]]:
    """
    Generate all possible placement combinations for tasks, limited by max_combinations.
    
    Args:
        workload_events: List of workload events with timestamps and node_name
        infrastructure_config: Infrastructure configuration with nodes and network maps
        sim_inputs: Simulation inputs containing task_types and platform_types
        replica_plan: Replica placement plan
        max_combinations: Maximum number of placement combinations to generate
        enable_resource_contention: Whether to consider resource contention in feasibility
        max_contention_threshold: Maximum allowed contention score (0.0-1.0)
    
    Returns:
        List of placement plans, each mapping task_id to (node_id, platform_id) tuple
    """
    print(f"\n=== Generating Brute Force Placement Combinations (max: {max_combinations if max_combinations is not None else 'ALL'}) ===")
    print(f"Resource contention enabled: {enable_resource_contention}")
    if enable_resource_contention:
        print(f"Max contention threshold: {max_contention_threshold}")
    
    # Get task types and platform types
    task_types = sim_inputs['task_types']
    platform_types = sim_inputs['platform_types']
    
    # Get nodes from infrastructure
    nodes = infrastructure_config['nodes']
    
    # Get resource contention configuration
    contention_config = infrastructure_config.get('startup_simulation', {})
    if enable_resource_contention and contention_config.get('simulate_contention', False):
        print(f"Resource contention config: {contention_config}")
    else:
        print("Resource contention disabled or not configured")
    
    # Create node_id mapping
    node_id_map = {node['node_name']: i for i, node in enumerate(nodes)}
    
    # Extract replica configuration
    replicas_config = replica_plan.get('replicas_config', {})
    preinit_clients = replica_plan.get('preinit_clients', [])
    preinit_servers = replica_plan.get('preinit_servers', [])
    
    print(f"Using replica plan for placement decisions:")
    print(f"  Preinit clients: {preinit_clients}")
    print(f"  Preinit servers: {preinit_servers}")
    print(f"  Replicas config: {replicas_config}")
    
    # Simulate platform creation (same logic as generate_simple_placement_plan)
    node_platforms = {}  # node_name -> list of platform IDs
    global_platform_id = 0
    
    for node in nodes:
        node_name = node['node_name']
        node_platforms[node_name] = []
        
        # Get platforms for this node from infrastructure config
        node_platform_types = node.get('platforms', [])
        
        for platform_type_name in node_platform_types:
            node_platforms[node_name].append({
                'platform_id': global_platform_id,
                'platform_type': platform_type_name
            })
            global_platform_id += 1
    
    # Simulate replica creation
    available_platforms = {}
    
    for task_type_name, task_config in replicas_config.items():
        if task_type_name not in available_platforms:
            available_platforms[task_type_name] = []
        
        # Get supported platforms for this task type
        supported_platforms = task_types[task_type_name].get('platforms', [])
        
        # Create server replicas
        per_server = task_config.get('per_server', 0)
        if per_server > 0:
            for node in nodes:
                node_name = node['node_name']
                if node_name in preinit_servers:
                    suitable_platforms = [
                        p for p in node_platforms[node_name]
                        if p['platform_type'] in supported_platforms
                    ]
                    
                    replicas_created = 0
                    for platform_info in suitable_platforms:
                        if replicas_created >= per_server:
                            break
                        
                        available_platforms[task_type_name].append({
                            'node_name': node_name,
                            'node_id': node_id_map[node_name],
                            'platform_type': platform_info['platform_type'],
                            'platform_id': platform_info['platform_id']
                        })
                        replicas_created += 1
        
        # Create client replicas
        per_client = task_config.get('per_client', 0)
        if per_client > 0:
            for node in nodes:
                node_name = node['node_name']
                if node_name in preinit_clients:
                    suitable_platforms = [
                        p for p in node_platforms[node_name]
                        if p['platform_type'] in supported_platforms
                    ]
                    
                    replicas_created = 0
                    for platform_info in suitable_platforms:
                        if replicas_created >= per_client:
                            break
                        
                        available_platforms[task_type_name].append({
                            'node_name': node_name,
                            'node_id': node_id_map[node_name],
                            'platform_type': platform_info['platform_type'],
                            'platform_id': platform_info['platform_id']
                        })
                        replicas_created += 1
    
    # Extract tasks from workload events
    tasks = []
    task_id = 0
    
    for event in workload_events:
        application = event['application']
        task_type_names = application.get('dag', [])
        
        for task_type_name in task_type_names:
            if task_type_name not in task_types:
                print(f"Warning: Task type {task_type_name} not found in task types")
                continue
            
            task_type = task_types[task_type_name]
            source_node_name = event['node_name']
            
            # Get available platforms for this task type
            task_platforms = available_platforms.get(task_type_name, [])
            
            # Filter platforms based on network connectivity and resource contention
            feasible_platforms = []
            for platform_info in task_platforms:
                node_name = platform_info['node_name']
                
                # Check network connectivity first
                is_network_feasible = False
                if source_node_name == node_name:
                    # Local execution always possible
                    is_network_feasible = True
                else:
                    # Check network connectivity
                    node_config = next((n for n in nodes if n['node_name'] == node_name), None)
                    if node_config and source_node_name in node_config.get('network_map', {}):
                        is_network_feasible = True
                
                # If network is feasible, check resource contention
                if is_network_feasible:
                    if enable_resource_contention:
                        # Add platform type to platform_info for contention calculation
                        platform_info_with_type = platform_info.copy()
                        platform_info_with_type['platform_type'] = platform_info.get('platform_type', 'unknown')
                        platform_info_with_type['node_name'] = node_name
                        
                        if is_platform_feasible_with_contention(platform_info_with_type, contention_config, max_contention_threshold):
                            feasible_platforms.append(platform_info)
                        else:
                            contention_score = calculate_resource_contention(platform_info_with_type, contention_config)
                            print(f"    Platform {platform_info['platform_id']} on {node_name} rejected due to high contention: {contention_score:.3f}")
                    else:
                        feasible_platforms.append(platform_info)
            
            if feasible_platforms:
                tasks.append({
                    'task_id': task_id,
                    'task_type': task_type_name,
                    'source_node': source_node_name,
                    'feasible_platforms': feasible_platforms
                })
                task_id += 1
            else:
                print(f"Warning: No feasible platforms found for task {task_id} ({task_type_name})")
    
    print(f"Found {len(tasks)} tasks with feasible placements")
    for task in tasks:
        print(f"  Task {task['task_id']} ({task['task_type']}): {len(task['feasible_platforms'])} feasible platforms")
    
    combinations = generate_all_combinations(tasks)
    
    print(f"Generated {len(combinations)} placement combinations")
    return combinations


def generate_all_combinations(tasks: List[Dict]) -> List[Dict[int, Tuple[int, int]]]:
    """Generate all possible placement combinations for tasks."""
    if not tasks:
        return [{}]
    
    # Recursive function to generate combinations
    def generate_recursive(task_index: int, current_placement: Dict[int, Tuple[int, int]]) -> List[Dict[int, Tuple[int, int]]]:
        if task_index >= len(tasks):
            return [current_placement.copy()]
        
        task = tasks[task_index]
        combinations = []
        
        for platform_info in task['feasible_platforms']:
            placement = (platform_info['node_id'], platform_info['platform_id'])
            current_placement[task['task_id']] = placement
            combinations.extend(generate_recursive(task_index + 1, current_placement))
            # Clean up: remove the current task's placement before trying the next platform
            del current_placement[task['task_id']]
        
        return combinations
    
    return generate_recursive(0, {})


def filter_valid_placements(
        placement_combinations: List[Dict[int, Tuple[int, int]]],
        infrastructure_config: Dict[str, Any],
        workload_events: List[Dict],
        max_valid_combinations: Optional[int] = 100
) -> List[Dict[int, Tuple[int, int]]]:
    """
    Filter placement combinations to only include those with valid network connectivity.
    
    Args:
        placement_combinations: List of placement plans to filter
        infrastructure_config: Infrastructure configuration with nodes and network maps
        workload_events: List of workload events to get task source nodes
        max_valid_combinations: Maximum number of valid combinations to return
    
    Returns:
        List of valid placement plans that have network connectivity
    """
    print(f"\n=== Filtering Valid Placements ===")
    print(f"Input combinations: {len(placement_combinations)}")
    print(f"Max valid combinations: {max_valid_combinations if max_valid_combinations is not None else 'ALL'}")
    
    # Get nodes from infrastructure
    nodes = infrastructure_config['nodes']
    node_id_map = {i: node for i, node in enumerate(nodes)}
    node_name_map = {node['node_name']: node for node in nodes}
    
    # Create a mapping from task_id to source node name
    task_id = 0
    task_to_source = {}
    for event in workload_events:
        application = event['application']
        task_type_names = application.get('dag', [])
        source_node_name = event['node_name']
        
        for task_type_name in task_type_names:
            task_to_source[task_id] = source_node_name
            task_id += 1
    
    valid_combinations = []
    invalid_combinations = 0
    
    for i, placement_plan in enumerate(placement_combinations):
        if max_valid_combinations is not None and len(valid_combinations) >= max_valid_combinations:
            break
            
        is_valid = True
        
        for task_id, (target_node_id, target_platform_id) in placement_plan.items():
            # Get source node name for this task
            source_node_name = task_to_source.get(task_id, "unknown")
            
            # Check if target node exists
            if target_node_id not in node_id_map:
                is_valid = False
                break
            
            target_node = node_id_map[target_node_id]
            
            # Check network connectivity (must be server->client direction in network_map)
            if source_node_name == target_node['node_name']:
                # Local execution always possible
                continue
            else:
                # Require that the TARGET (server) node has connectivity entry to the SOURCE (client) node
                if target_node['node_name'] in node_name_map and source_node_name in node_name_map:
                    server_node = node_name_map[target_node['node_name']]
                    if source_node_name not in server_node.get('network_map', {}):
                        is_valid = False
                        break
                else:
                    is_valid = False
                    break
        
        if is_valid:
            valid_combinations.append(placement_plan)
        else:
            invalid_combinations += 1
    
    print(f"Valid combinations found: {len(valid_combinations)}")
    print(f"Invalid combinations discarded: {invalid_combinations}")
    
    return valid_combinations


def copy_only_feasible_placements(
        placement_plan: Dict[int, Tuple[int, int]],
        infrastructure_config: Dict[str, Any],
        workload_events: List[Dict]
) -> Dict[int, Tuple[int, int]]:
    """
    Create a copy of a placement plan containing only network-feasible task placements.
    """
    nodes = infrastructure_config['nodes']
    node_id_map = {i: node for i, node in enumerate(nodes)}
    node_name_map = {node['node_name']: node for node in nodes}

    # Map task_id -> source node name in order of events
    task_to_source: Dict[int, str] = {}
    tmp_task_id = 0
    for event in workload_events:
        dag = event.get('application', {}).get('dag', [])
        for _ in dag:
            task_to_source[tmp_task_id] = event['node_name']
            tmp_task_id += 1

    feasible: Dict[int, Tuple[int, int]] = {}
    for t_id, (node_id, platform_id) in placement_plan.items():
        src = task_to_source.get(t_id)
        target_node = node_id_map.get(node_id)
        if src is None or target_node is None:
            continue
        if src == target_node['node_name']:
            feasible[t_id] = (node_id, platform_id)
            continue
        # Require that the TARGET (server) node has connectivity to SOURCE (client)
        server_node = node_name_map.get(target_node['node_name'])
        if server_node and src in server_node.get('network_map', {}):
            feasible[t_id] = (node_id, platform_id)
    return feasible


def process_sample_with_placement(args):
    """Process a single sample with a specific placement plan."""
    i, sample, placement_plan, base_nodes, output_dir, sim_input_path, mapping_file, config_file, workload_base_file, apps = args
    logger = setup_logging(output_dir)
    logger.info(f"Processing sample {i + 1} with placement plan {len(placement_plan)} tasks")

    try:
        sim_inputs = load_simulation_inputs(sim_input_path)

        with open(mapping_file, 'rb') as f:
            mapping = pickle.load(f)

        # Load infrastructure config
        with open(config_file, 'r') as f:
            infra_config = json.load(f)

        with open(workload_base_file, 'r') as f:
            workload_base = json.load(f)

        # Prepare workloads
        workloads = prepare_workloads(sample, mapping, workload_base, apps)
        # Flatten workloads into single sorted list
        flattened_workloads = flatten_workloads(workloads)

        # Prepare infrastructure configuration with the specific placement plan
        # Reuse the same node/network topology via base_nodes
        sim_config = prepare_simulation_config(sample, mapping, infra_config, placement_plan, base_nodes=base_nodes)

        # Validate that workload nodes exist in infrastructure
        validate_workload_nodes(flattened_workloads['events'], sim_config['nodes'])

        # Combine infrastructure and workload configurations
        full_config = {
            "infrastructure": sim_config,
            "workload": flattened_workloads
        }

        # Execute simulation with additional inputs
        cache_policy = 'fifo'
        task_priority = 'fifo'
        keep_alive = KEEP_ALIVE
        queue_length = QUEUE_LENGTH
        scheduling_strategy = 'determined_determined'
        
        result = execute_simulation(full_config, sim_inputs, scheduling_strategy,
                                    model_locations={}, models={},
                                    cache_policy=cache_policy, task_priority=task_priority, keep_alive=keep_alive, queue_length=queue_length)
        result['sample'] = {
            'apps': apps,
            'sample': sample.tolist(),
            'mapping': mapping,
            'infra_config': infra_config,
            'workload_base': workload_base,
            'sim_inputs': sim_inputs,
            'scheduling_strategy': scheduling_strategy,
            'cache_policy': cache_policy,
            'task_priority': task_priority,
            'keep_alive': keep_alive,
            'queue_length': queue_length,
            'placement_plan': placement_plan
        }

        # Save result with collision-resistant placement identifier
        import hashlib, uuid
        placement_key = json.dumps(sorted(placement_plan.items()))
        placement_hash = hashlib.sha1(placement_key.encode('utf-8')).hexdigest()[:16]
        unique_suffix = uuid.uuid4().hex[:8]
        result_file = output_dir / f"simulation_{i + 1}_placement_{placement_hash}_{unique_suffix}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, cls=DataclassJSONEncoder)

        logger.info(f"Completed simulation {i + 1} with placement {placement_hash}_{unique_suffix}")
        return result_file

    except Exception as e:
        logger.error(f"Error in simulation {i + 1}: {str(e)}")
        logger.exception(e)
        return None


def execute_brute_force_placement_optimization(
        apps: List[str],
        config_file: str,
        mapping_file: str,
        output_dir: Path,
        samples: np.ndarray,
        sim_input_path: Path,
        workload_base_file: str,
        max_workers: int,
        max_combinations: int = 100,
        enable_resource_contention: bool = True,
        max_contention_threshold: float = 0.8
) -> List[str]:
    """
    Execute brute force placement optimization by testing different placement combinations.
    
    Args:
        apps: List of application names
        config_file: Path to infrastructure configuration file
        mapping_file: Path to mapping file
        output_dir: Output directory for results
        samples: Array of samples to process
        sim_input_path: Path to simulation input files
        workload_base_file: Path to workload base file
        max_workers: Maximum number of parallel workers
        max_combinations: Maximum number of placement combinations to test
    
    Returns:
        List of result file paths
    """
    print(f"\n=== Starting Brute Force Placement Optimization ===")
    print(f"Max combinations per sample: {max_combinations}")
    print(f"Number of samples: {len(samples)}")
    print(f"Max workers: {max_workers}")
    
    # DEBUG: Calculate total possible placement topologies
    total_possible_topologies = 0
    for sample_idx, sample in enumerate(samples):
        try:
            # Load required data for this sample
            sim_inputs = load_simulation_inputs(sim_input_path)
            
            with open(mapping_file, 'rb') as f:
                mapping = pickle.load(f)
            
            with open(config_file, 'r') as f:
                infra_config = json.load(f)
            
            with open(workload_base_file, 'r') as f:
                workload_base = json.load(f)
            
            # Prepare workloads
            workloads = prepare_workloads(sample, mapping, workload_base, apps)
            flattened_workloads = flatten_workloads(workloads)
            
            # Prepare infrastructure configuration (without placement plan)
            sim_config = prepare_simulation_config(sample, mapping, infra_config)
            
            # Generate replica plan
            replica_plan = determine_replica_placement(sim_config, sim_inputs)
            
            # Calculate possible combinations for this sample
            placement_combinations = generate_brute_force_placement_combinations(
                flattened_workloads['events'],
                sim_config,
                sim_inputs,
                replica_plan,
                max_combinations,
                enable_resource_contention,
                max_contention_threshold
            )
            
            total_possible_topologies += len(placement_combinations)
            
        except Exception as e:
            print(f"Error calculating topologies for sample {sample_idx + 1}: {str(e)}")
            continue
    
    print(f"DEBUG: Total possible placement topologies across all samples: {total_possible_topologies}")
    
    result_paths = []
    
    # Process each sample with multiple placement combinations
    for sample_idx, sample in enumerate(samples):
        print(f"\n--- Processing Sample {sample_idx + 1}/{len(samples)} ---")
        
        try:
            # Load required data for this sample
            sim_inputs = load_simulation_inputs(sim_input_path)
            
            with open(mapping_file, 'rb') as f:
                mapping = pickle.load(f)
            
            with open(config_file, 'r') as f:
                infra_config = json.load(f)
            
            with open(workload_base_file, 'r') as f:
                workload_base = json.load(f)
            
            # Prepare workloads
            workloads = prepare_workloads(sample, mapping, workload_base, apps)
            flattened_workloads = flatten_workloads(workloads)
            
            # Prepare infrastructure configuration (without placement plan)
            sim_config = prepare_simulation_config(sample, mapping, infra_config)
            
            # Generate replica plan
            replica_plan = determine_replica_placement(sim_config, sim_inputs)
            
            # Generate placement combinations
            placement_combinations = generate_brute_force_placement_combinations(
                flattened_workloads['events'],
                sim_config,
                sim_inputs,
                replica_plan,
                max_combinations,
                enable_resource_contention,
                max_contention_threshold
            )
            
            print(f"Generated {len(placement_combinations)} placement combinations for sample {sample_idx + 1}")
            
            # Filter out invalid placements based on network connectivity
            valid_placement_combinations = filter_valid_placements(
                placement_combinations,
                sim_config,
                flattened_workloads['events'],
                max_combinations
            )
            
            print(f"Filtered to {len(valid_placement_combinations)} valid placement combinations for sample {sample_idx + 1}")
            
            if not valid_placement_combinations:
                print(f"⚠️  WARNING: No valid placement combinations found for sample {sample_idx + 1}")
                print(f"   This sample will be skipped")
                continue
            
            # Execute simulations in parallel for this sample
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Create arguments for each valid placement combination
                placement_tuples = []
                for placement_plan in placement_combinations:
                    # Make a per-plan feasible-only copy safeguard
                    feasible_plan = copy_only_feasible_placements(
                        placement_plan,
                        sim_config,
                        flattened_workloads['events']
                    )
                    if not feasible_plan:
                        continue
                    # Reuse the same node topology by passing base_nodes to per-plan config
                    placement_tuples.append(
                        (sample_idx, sample, feasible_plan, sim_config['nodes'], output_dir, sim_input_path, mapping_file, config_file, workload_base_file, apps)
                    )
                
                # Submit all tasks
                future_to_placement = {
                    executor.submit(process_sample_with_placement, placement_tuple): placement_tuple 
                    for placement_tuple in placement_tuples
                }
                
                # Collect results
                for future in concurrent.futures.as_completed(future_to_placement):
                    result_file = future.result()
                    if result_file is not None:
                        result_paths.append(str(result_file))
            
            print(f"Completed {len(valid_placement_combinations)} simulations for sample {sample_idx + 1}")
            
        except Exception as e:
            print(f"Error processing sample {sample_idx + 1}: {str(e)}")
            continue
    
    print(f"\n=== Brute Force Placement Optimization Complete ===")
    print(f"Total result files generated: {len(result_paths)}")
    
    # Analyze results to find the fastest simulation
    print(f"\n=== Analyzing Results for Fastest Simulation ===")
    fastest_simulation = None
    fastest_rtt = float('inf')
    
    for result_file in result_paths:
        try:
            with open(result_file, 'r') as f:
                result_data = json.load(f)
            
            # Extract RTT from simulation stats
            stats = result_data.get('stats', {})
            task_results = stats.get('taskResults', [])
            
            if task_results:
                # Calculate total RTT across all tasks using elapsedTime
                total_rtt = sum(task_result.get('elapsedTime', 0) for task_result in task_results)
                
                # Check if this is the fastest so far
                if total_rtt < fastest_rtt:
                    fastest_rtt = total_rtt
                    fastest_simulation = {
                        'file': result_file,
                        'total_rtt': total_rtt,
                        'placement_plan': result_data.get('sample', {}).get('placement_plan', {}),
                        'sample': result_data.get('sample', {}).get('sample', [])
                    }
                    
        except Exception as e:
            print(f"Error analyzing result file {result_file}: {str(e)}")
            continue
    
    if fastest_simulation:
        print(f"🏆 FASTEST SIMULATION FOUND:")
        print(f"   File: {fastest_simulation['file']}")
        print(f"   Total RTT: {fastest_simulation['total_rtt']:.3f}s")
        print(f"   Placement Plan: {fastest_simulation['placement_plan']}")
        print(f"   Sample: {fastest_simulation['sample']}")
    else:
        print("❌ No valid simulation results found for analysis")
    
    return result_paths


def main():
    # Configuration paths
    base_dir = Path("simulation_data")
    sim_input_path = Path("data/nofs-ids")  # Base path for simulation input files
    samples_file = base_dir / "lhs_samples_simple.npy"
    mapping_file = base_dir / "lhs_samples_simple_mapping.pkl"
    config_file = base_dir / "space_with_network.json"
    # python -m src.generator -d data/nofs-ids --generate-traces --rps 10 --seconds 10
    workload_base_file = "data/nofs-ids/traces/workload-10.json"
    output_dir = base_dir / "initial_results_simple"
    os.makedirs(output_dir, exist_ok=True)
    # todo: max_workers = int(sys.argv[1])
    cpu_count = os.cpu_count()
    print("CPU count: ", cpu_count)
    max_workers = cpu_count - 1 if cpu_count else 1
    # Setup logging
    logger = setup_logging(output_dir)
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting simulation preparation")

        # Load simulation inputs
        logger.info("Loading simulation input files")
        sim_inputs = load_simulation_inputs(sim_input_path)

        # Load samples and mapping
        logger.info("Loading samples and mapping")
        samples = np.load(samples_file)
        with open(mapping_file, 'rb') as f:
            mapping = pickle.load(f)

        # Load infrastructure config
        logger.info("Loading infrastructure configuration")
        with open(config_file, 'r') as f:
            infra_config = json.load(f)

        # Load workload base
        logger.info("Loading workload base")
        with open(workload_base_file, 'r') as f:
            workload_base = json.load(f)

        # Get list of applications from config
        apps = [app for app in infra_config['wsc'].keys()]

        # Process each sample
        # Choose between regular execution and brute force placement optimization
        use_brute_force = len(sys.argv) > 1 and sys.argv[1] == '--brute-force'
        max_combinations: Optional[int] = 50  # Default cap; None means all
        # Contention settings now come from JSON (infra_config['startup_simulation'])
        startup_cfg = infra_config.get('startup_simulation', {})
        enable_resource_contention = bool(startup_cfg.get('simulate_contention', False))
        max_contention_threshold = float(startup_cfg.get('contention_threshold', 0.8))
        
        # Parse command line arguments (only max_combinations)
        if use_brute_force:
            if len(sys.argv) == 2:
                # No number provided → generate ALL combinations (no cap)
                max_combinations = None
            elif len(sys.argv) > 2:
                try:
                    max_combinations = int(sys.argv[2])
                except ValueError:
                    print(f"Warning: Invalid max_combinations value '{sys.argv[2]}', using default {max_combinations}")
        
        if use_brute_force:
            logger.info(f"Using brute force placement optimization with max {max_combinations if max_combinations is not None else 'ALL'} combinations per sample")
            logger.info(f"Resource contention (from JSON): enabled={enable_resource_contention}, threshold={max_contention_threshold}")
            reactive_results_paths = execute_brute_force_placement_optimization(
                apps, str(config_file), str(mapping_file), output_dir, samples,
                sim_input_path, workload_base_file, max_workers, max_combinations,
                enable_resource_contention, max_contention_threshold
            )
        else:
            logger.info("Using regular reactive execution")
            reactive_results_paths = execute_reactive_samples_parallel(apps, str(config_file), str(mapping_file), output_dir, samples,
                                                                       sim_input_path, workload_base_file, max_workers)
        
        # print(reactive_results_paths)
        logger.info("Completed all simulations")

        # logger.info("Training model now...")
        # models, eval_results = train_model(output_dir, samples, include_queue_length=False)
        # model_paths = save_models(models, output_dir)

        # logger.info('Finished model training')
        logger.info(f'All files can be found under {output_dir}')


    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise e


def generate_simple_placement_plan(
        workload_events: List[Dict],
        infrastructure_config: Dict[str, Any],
        sim_inputs: Dict[str, Any],
        replica_plan: Dict[str, Any]
) -> Dict[int, Tuple[int, int]]:
    """
    Generate a simple placement plan for tasks.
    
    Args:
        workload_events: List of workload events with timestamps and node_name
        infrastructure_config: Infrastructure configuration with nodes and network maps
        sim_inputs: Simulation inputs containing task_types and platform_types
    
    Returns:
        Dictionary mapping task_id to (node_id, platform_id) tuple
    """
    print("\n=== Generating Simple Placement Plan ===")
    
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
    
    print(f"Using replica plan for placement decisions:")
    print(f"  Preinit clients: {preinit_clients}")
    print(f"  Preinit servers: {preinit_servers}")
    print(f"  Replicas config: {replicas_config}")
    
    # Simulate the exact platform creation process from simulation.py
    # This matches the platform ID assignment in create_nodes() and precreate_replicas()
    
    # First, simulate platform creation during node creation (like in create_nodes())
    node_platforms = {}  # node_name -> list of platform IDs
    global_platform_id = 0
    
    for node in nodes:
        node_name = node['node_name']
        node_platforms[node_name] = []
        
        # Get platforms for this node from infrastructure config
        # This should match the platforms specified in the node configuration
        node_platform_types = node.get('platforms', [])
        
        for platform_type_name in node_platform_types:
            node_platforms[node_name].append({
                'platform_id': global_platform_id,
                'platform_type': platform_type_name
            })
            global_platform_id += 1
    
    print(f"Simulated platform creation:")
    for node_name, platforms in node_platforms.items():
        print(f"  {node_name}: {len(platforms)} platforms {[p['platform_id'] for p in platforms]}")
    
    # Now simulate replica creation (like in precreate_replicas())
    available_platforms = {}
    # Don't track assigned platforms globally - each task type can use the same platforms
    # This matches the actual simulation behavior where platforms can serve multiple task types
    
    for task_type_name, task_config in replicas_config.items():
        if task_type_name not in available_platforms:
            available_platforms[task_type_name] = []
        
        # Get supported platforms for this task type
        supported_platforms = task_types[task_type_name].get('platforms', [])
        
        # Create server replicas
        per_server = task_config.get('per_server', 0)
        if per_server > 0:
            for node in nodes:
                node_name = node['node_name']
                if node_name in preinit_servers:
                    # Find suitable platforms on this server
                    suitable_platforms = [
                        p for p in node_platforms[node_name]
                        if p['platform_type'] in supported_platforms
                    ]
                    
                    # Create up to per_server replicas on this node
                    replicas_created = 0
                    for platform_info in suitable_platforms:
                        if replicas_created >= per_server:
                            break
                        
                        # Add platform to available platforms for this task type
                        available_platforms[task_type_name].append({
                            'node_name': node_name,
                            'node_id': node_id_map[node_name],
                            'platform_type': platform_info['platform_type'],
                            'platform_id': platform_info['platform_id']
                        })
                        replicas_created += 1
        
        # Create client replicas
        per_client = task_config.get('per_client', 0)
        if per_client > 0:
            for node in nodes:
                node_name = node['node_name']
                if node_name in preinit_clients:
                    # Find suitable platforms on this client
                    suitable_platforms = [
                        p for p in node_platforms[node_name]
                        if p['platform_type'] in supported_platforms
                    ]
                    
                    # Create up to per_client replicas on this node
                    replicas_created = 0
                    for platform_info in suitable_platforms:
                        if replicas_created >= per_client:
                            break
                        
                        # Add platform to available platforms for this task type
                        available_platforms[task_type_name].append({
                            'node_name': node_name,
                            'node_id': node_id_map[node_name],
                            'platform_type': platform_info['platform_type'],
                            'platform_id': platform_info['platform_id']
                        })
                        replicas_created += 1
    
    print(f"Node mapping: {node_id_map}")
    print(f"Available platforms per task type:")
    for task_type, platforms in available_platforms.items():
        print(f"  {task_type}: {len(platforms)} platforms")
    
    # Generate placement plan
    placement_plan = {}
    task_id = 0
    used_platforms = set()  # Track which platforms are already used in this batch
    
    for event in workload_events:
        # Get task type from application DAG
        application = event['application']
        task_type_names = application.get('dag', [])
        
        for task_type_name in task_type_names:
            if task_type_name not in task_types:
                print(f"Warning: Task type {task_type_name} not found in task types")
                continue
            
            task_type = task_types[task_type_name]
            source_node_name = event['node_name']
            
            # Get available platforms for this task type
            task_platforms = available_platforms.get(task_type_name, [])
            print(f"Task {task_id} ({task_type_name}) from {source_node_name} - {len(task_platforms)} available platforms")
            
            # Find the best placement (avoiding already used platforms)
            best_placement = None
            
            # First, try to place on source node (local execution)
            if source_node_name in node_id_map:
                source_node_id = node_id_map[source_node_name]
                
                # Find available platforms on source node
                for platform_info in task_platforms:
                    if platform_info['node_name'] == source_node_name:
                        placement = (platform_info['node_id'], platform_info['platform_id'])
                        
                        # Check if this platform is already used
                        if placement not in used_platforms:
                            best_placement = placement
                            print(f"  -> Local placement: node {platform_info['node_id']}, platform {platform_info['platform_id']} ({platform_info['platform_type']})")
                            break
                        else:
                            print(f"  -> Local platform {platform_info['platform_id']} already used, skipping")
            
            # If no local placement, try server nodes with network connectivity
            if best_placement is None:
                for platform_info in task_platforms:
                    node_name = platform_info['node_name']
                    
                    # Skip client nodes (only consider server nodes for offloading)
                    if node_name.startswith('client_node'):
                        continue
                    
                    # Check network connectivity from source to this node
                    node_config = next((n for n in nodes if n['node_name'] == node_name), None)
                    if node_config and source_node_name in node_config.get('network_map', {}):
                        placement = (platform_info['node_id'], platform_info['platform_id'])
                        
                        # Check if this platform is already used
                        if placement not in used_platforms:
                            best_placement = placement
                            print(f"  -> Server placement: node {platform_info['node_id']}, platform {platform_info['platform_id']} ({platform_info['platform_type']})")
                            break
                        else:
                            print(f"  -> Server platform {platform_info['platform_id']} already used, skipping")
            
            # If still no placement found, use fallback (first available)
            if best_placement is None:
                for platform_info in task_platforms:
                    placement = (platform_info['node_id'], platform_info['platform_id'])
                    
                    # Check if this platform is already used
                    if placement not in used_platforms:
                        best_placement = placement
                        print(f"  -> Fallback placement: node {platform_info['node_id']}, platform {platform_info['platform_id']} ({platform_info['platform_type']})")
                        break
                    else:
                        print(f"  -> Fallback platform {platform_info['platform_id']} already used, skipping")
            
            if best_placement is not None:
                placement_plan[task_id] = best_placement
                used_platforms.add(best_placement)  # Mark this platform as used
                print(f"  Final placement for task {task_id}: {best_placement}")
            else:
                print(f"  ERROR: No placement found for task {task_id} - all compatible platforms are already used")
            
            task_id += 1
    
    print(f"Generated placement plan for {len(placement_plan)} tasks")
    print(f"Placement plan: {placement_plan}")
    print(f"Used platforms: {used_platforms}")
    
    return placement_plan

def process_sample(args):
    i, sample, output_dir, sim_input_path, mapping_file, config_file, workload_base_file, apps = args
    logger = setup_logging(output_dir)
    logger.info(f"Processing sample {i + 1}")

    try:
        sim_inputs = load_simulation_inputs(sim_input_path)
        print("started simulation 4")


        with open(mapping_file, 'rb') as f:
            mapping = pickle.load(f)

        # Load infrastructure config
        with open(config_file, 'r') as f:
            infra_config = json.load(f)

        with open(workload_base_file, 'r') as f:
            workload_base = json.load(f)

        # Prepare workloads
        workloads = prepare_workloads(sample, mapping, workload_base, apps)
        # Flatten workloads into single sorted list
        flattened_workloads = flatten_workloads(workloads)

        # Prepare infrastructure configuration (without placement plan first)
        sim_config = prepare_simulation_config(sample, mapping, infra_config)
        print("started simulation 3")

        # Generate placement plan using replica plan
        replica_plan = determine_replica_placement(sim_config, sim_inputs)
        placement_plan = generate_simple_placement_plan(
            flattened_workloads['events'], 
            sim_config, 
            sim_inputs,
            replica_plan
        )
        
        # Update infrastructure config with placement plan
        sim_config = prepare_simulation_config(sample, mapping, infra_config, placement_plan)

        # Validate that workload nodes exist in infrastructure
        validate_workload_nodes(flattened_workloads['events'], sim_config['nodes'])

        print("started simulation 2")

        # Combine infrastructure and workload configurations
        full_config = {
            "infrastructure": sim_config,
            "workload": flattened_workloads
        }

        # Execute simulation with additional inputs
        cache_policy = 'fifo'
        task_priority = 'fifo'
        keep_alive = KEEP_ALIVE
        queue_length = QUEUE_LENGTH
        # todo: change to gnn_gnn
        scheduling_strategy = 'determined_determined'
        print("started simulation")
        result = execute_simulation(full_config, sim_inputs, scheduling_strategy,
                                    model_locations={}, models={},
                                    cache_policy=cache_policy, task_priority=task_priority, keep_alive=keep_alive, queue_length=queue_length)
        result['sample'] = {
            'apps': apps,
            'sample': sample.tolist(),
            'mapping': mapping,
            'infra_config': infra_config,
            'workload_base': workload_base,
            'sim_inputs': sim_inputs,
            'scheduling_strategy': scheduling_strategy,
            'cache_policy': cache_policy,
            'task_priority': task_priority,
            'keep_alive': keep_alive,
            'queue_length': queue_length
        }

        # Save result
        result_file = output_dir / f"simulation_{i + 1}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, cls=DataclassJSONEncoder)

        logger.info(f"Completed simulation {i + 1}")
        return result_file

    except Exception as e:
        logger.error(f"Error in simulation {i + 1}: {str(e)}")
        logger.exception(e)
        return None

def execute_reactive_samples_parallel(apps, config_file, mapping_file, output_dir, samples, sim_input_path,
                                      workload_base_file, max_workers):
    result_paths = []

    # Use ProcessPoolExecutor for CPU-bound tasks, ThreadPoolExecutor for I/O-bound tasks
    # Adjust max_workers as needed based on your system's capabilities
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of (index, sample) tuples to process
        sample_tuples = [(i, sample, output_dir, sim_input_path, mapping_file, config_file, workload_base_file, apps) for i, sample in enumerate(samples)]
        start_ts = time.time()
        # Submit all tasks and process results as they complete
        future_to_sample = {executor.submit(process_sample, sample_tuple): sample_tuple for sample_tuple in
                            sample_tuples}

        for future in concurrent.futures.as_completed(future_to_sample):
            result_file = future.result()
            if result_file is not None:
                result_paths.append(result_file)
        end_ts = time.time()
        print(f'Duration: {end_ts - start_ts}')
    return result_paths

if __name__ == "__main__":
    main()
