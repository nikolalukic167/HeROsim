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

import numpy as np  # type: ignore[import-not-found]
from itertools import islice

# Assuming increase_events is imported from your previous script
from src.eventgenerator import increase_events_of_app
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
        "scheduler": original_config.get('scheduler', {}),
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
    # print(f"\nInfrastructure nodes created:")
    # for node in infrastructure_config['nodes']:
    #     print(f"  {node['node_name']}: {len(node['network_map'])} network connections")

    # Add placement plan to infrastructure config if provided
    if placement_plan is not None:
        # Make placements available for scheduler
        infrastructure_config['forced_placements'] = placement_plan
        print(f"Added placement plan with {len(placement_plan)} task placements")

    # Pass replica plan if provided
    if replica_plan is not None:
        infrastructure_config['replica_plan'] = replica_plan

    # Add prewarm configuration
    infrastructure_config['prewarm'] = original_config.get('prewarm', {})

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


def generate_brute_force_placement_combinations(
        workload_events: List[Dict],
        infrastructure_config: Dict[str, Any],
        sim_inputs: Dict[str, Any],
        replica_plan: Dict[str, Any],
        max_combinations: Optional[int] = 100
) -> List[Dict[int, Tuple[int, int]]]:
    """
    Generate all possible placement combinations for tasks, limited by max_combinations.
    Applies both core filtering rules:
    1. Don't make topologies that offload from client->server that have no network connection
    2. Don't schedule on non-existent replicas for that function
    
    Args:
        workload_events: List of workload events with timestamps and node_name
        infrastructure_config: Infrastructure configuration with nodes and network maps
        sim_inputs: Simulation inputs containing task_types and platform_types
        replica_plan: Replica placement plan
        max_combinations: Maximum number of placement combinations to generate
    
    Returns:
        List of placement plans, each mapping task_id to (node_id, platform_id) tuple
    """
    print(f"\n=== Generating Brute Force Placement Combinations (max: {max_combinations if max_combinations is not None else 'ALL'}) ===")
    
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
    
    # Simulate platform creation (same logic as simulation.py)
    node_platforms = {}  # node_name -> list of platform IDs
    platform_id = 0
    
    for node in nodes:
        node_name = node['node_name']
        node_platforms[node_name] = []
        
        # Get platforms for this node from infrastructure config
        node_platform_types = node.get('platforms', [])
        
        for platform_type_name in node_platform_types:
            node_platforms[node_name].append({
                'platform_id': platform_id,
                'platform_type': platform_type_name
            })
            platform_id += 1
    
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
            
            # Filter platforms based on network connectivity
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
                
                # Both rules must pass
                if is_network_feasible:
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
                # Abort-on-infeasible-task: skip entire sample to keep runs fully determined
                print(f"❌ Abort: No feasible platforms for workload task index {task_id} ({task_type_name}). Skipping this sample.")
                return []
    
    # Determinism check: ensure we have a placement decision for every workload task
    if len(tasks) != expected_task_count:
        print(f"this is shit and should not happen ❌ Abort: Determinism check failed. Expected {expected_task_count} tasks, built {len(tasks)} tasks. Skipping this sample.")
        return []

    print(f"Found {len(tasks)} tasks with feasible placements")
    for task in tasks:
        print(f"  Task {task['task_id']} ({task['task_type']}): {len(task['feasible_platforms'])} feasible platforms")
    
    # Generate combinations with proper max_combinations enforcement
    combinations = generate_all_combinations_with_limit(tasks, max_combinations)
    
    print(f"Generated {len(combinations)} placement combinations")
    return combinations


def generate_all_combinations_with_limit(tasks: List[Dict], max_combinations: Optional[int] = None) -> List[Dict[int, Tuple[int, int]]]:
    """Generate all possible placement combinations for tasks, with optional limit."""
    if not tasks:
        return [{}]
    
    # Calculate total possible combinations
    total_combinations = 1
    for task in tasks:
        total_combinations *= len(task['feasible_platforms'])
    
    print(f"Total possible combinations: {total_combinations}")
    
    # If no limit or limit is higher than total, generate all
    if max_combinations is None or max_combinations >= total_combinations:
        return generate_all_combinations_recursive(tasks)
    
    # Otherwise, sample combinations up to the limit
    print(f"Sampling {max_combinations} combinations from {total_combinations} possible")
    return generate_sampled_combinations(tasks, max_combinations)


def generate_all_combinations_recursive(tasks: List[Dict]) -> List[Dict[int, Tuple[int, int]]]:
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


def generate_sampled_combinations(tasks: List[Dict], max_combinations: int) -> List[Dict[int, Tuple[int, int]]]:
    """Generate a sampled set of placement combinations up to max_combinations."""
    import random
    
    combinations = []
    seen_combinations = set()
    
    # Generate random combinations until we have enough or run out of unique ones
    while len(combinations) < max_combinations:
        placement = {}
        valid = True
        
        # Generate a random combination
        for task in tasks:
            if not task['feasible_platforms']:
                valid = False
                break
            
            platform_info = random.choice(task['feasible_platforms'])
            placement[task['task_id']] = (platform_info['node_id'], platform_info['platform_id'])
        
        if not valid:
            break
        
        # Check if this combination is unique
        placement_key = tuple(sorted(placement.items()))
        if placement_key not in seen_combinations:
            seen_combinations.add(placement_key)
            combinations.append(placement)
    
    print(f"Generated {len(combinations)} sampled combinations")
    return combinations


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
    ) = args

    logger = setup_logging(output_dir)
    logger.info(f"Processing sample {i + 1} with placement plan {len(placement_plan)} tasks")

    try:
        # Prepare infrastructure configuration with the specific placement plan
        # Reuse the same node/network topology via base_nodes and keep the same replica plan
        sim_config = prepare_simulation_config(
            sample,
            mapping,
            infra_config,
            placement_plan,
            replica_plan=replica_plan,
            base_nodes=base_nodes,
        )

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
        max_combinations: Optional[int] = 100
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
    print(f"Max combinations per sample: {max_combinations if max_combinations is not None else 'ALL'}")
    print(f"Number of samples: {len(samples)}")
    print(f"Max workers: {max_workers}")

    time_started = time.time()
    result_paths = []
    # Track all RTTs across all placement combinations for plotting at the end
    all_rtts = []
    per_sample_rtts = []
    
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
                max_combinations
            )
            
            print(f"Generated {len(placement_combinations)} placement combinations for sample {sample_idx + 1}")
            
            if not placement_combinations:
                print(f"⚠️  WARNING: No valid placement combinations found for sample {sample_idx + 1}")
                print(f"   This sample will be skipped")
                continue
            
            # Execute simulations in parallel for this sample with early stopping
            # Early-stopping config from infra JSON (optional)
            bf_cfg = infra_config.get('brute_force', {}) if isinstance(infra_config, dict) else {}
            early_stop_patience = int(bf_cfg.get('early_stop_patience', 0))  # 0 disables
            early_stop_min_delta = float(bf_cfg.get('early_stop_min_delta', 0.0))
            save_only_optimal = bool(bf_cfg.get('save_only_optimal', True))  # Default: save only optimal

            base_nodes = sim_config['nodes']

            def _total_rtt(path: str) -> float:
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    stats = data.get('stats', {})
                    task_results = stats.get('taskResults', [])
                    if not task_results:
                        return float('inf')
                    return float(sum(tr.get('elapsedTime', 0) for tr in task_results))
                except Exception:
                    return float('inf')

            # Iterate placement combinations in batches to allow early stopping
            placements_iter = iter(placement_combinations)
            best_file: Optional[str] = None
            best_rtt: float = float('inf')
            no_improve_count = 0
            sample_rtts = []

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                while True:
                    batch = list(islice(placements_iter, max_workers))
                    if not batch:
                        break

                    futures = []
                    for placement_plan in batch:
                        placement_tuple = (
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
                        )
                        futures.append(executor.submit(process_sample_with_placement, placement_tuple))

                    # Collect this batch
                    batch_improved = False
                    for future in concurrent.futures.as_completed(futures):
                        result_file = future.result()
                        if result_file is None:
                            continue
                        current_path = str(result_file)
                        cur_rtt_value = _total_rtt(current_path)
                        sample_rtts.append(cur_rtt_value)

                        if cur_rtt_value + early_stop_min_delta < best_rtt:
                            # New best: remove previous best file if save_only_optimal enabled
                            if save_only_optimal and best_file is not None and os.path.exists(best_file):
                                try:
                                    os.remove(best_file)
                                except Exception:
                                    pass
                            best_file = current_path
                            best_rtt = cur_rtt_value
                            batch_improved = True
                        else:
                            # Not better => delete file if save_only_optimal enabled
                            if save_only_optimal:
                                try:
                                    os.remove(current_path)
                                except Exception:
                                    pass

                    if early_stop_patience > 0:
                        if batch_improved:
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                            if no_improve_count >= early_stop_patience:
                                print(f"Early stopping triggered after {no_improve_count} non-improving batches. Best RTT so far: {best_rtt:.3f}s")
                                break

            # Record the surviving best file for this sample
            if best_file is not None:
                result_paths.append(best_file)
                
                # Write best result info to sidecar file for bash script
                best_info = {
                    "file": os.path.basename(best_file),
                    "rtt": best_rtt
                }
                best_json_path = output_dir / "best.json"
                with open(best_json_path, 'w') as f:
                    json.dump(best_info, f)
            
            # Persist per-sample RTTs for final plotting
            if sample_rtts:
                per_sample_rtts.append(sample_rtts)
                all_rtts.extend(sample_rtts)

            if save_only_optimal:
                print(f"Completed {len(placement_combinations)} simulations for sample {sample_idx + 1} (kept 1 best file)")
            else:
                print(f"Completed {len(placement_combinations)} simulations for sample {sample_idx + 1} (kept all {len(placement_combinations)} files)")
            
        except Exception as e:
            print(f"Error processing sample {sample_idx + 1}: {str(e)}")
            continue
    
    print(f"Elapsed time: {time.time() - time_started:.2f} seconds")

    print(f"\n=== Brute Force Placement Optimization Complete ===")
    print(f"Total result files generated: {len(result_paths)}")
    
    """
    # Plot RTT distributions to a PDF
    if all_rtts:
        try:
            import matplotlib  # type: ignore[import-not-found]
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt  # type: ignore[import-not-found]
            from matplotlib.backends.backend_pdf import PdfPages  # type: ignore[import-not-found]

            pdf_path = output_dir / "rtt_summary.pdf"
            with PdfPages(pdf_path) as pdf:
                # Page 1: Overall histogram
                plt.figure(figsize=(10, 6))
                finite_rtts = [r for r in all_rtts if math.isfinite(r)]
                if finite_rtts:
                    plt.hist(finite_rtts, bins=min(50, max(10, len(finite_rtts)//5)), color='#4C78A8', edgecolor='white')
                    plt.title('RTT Histogram (All Placement Runs)')
                    plt.xlabel('Total RTT (s)')
                    plt.ylabel('Count')
                    plt.grid(True, alpha=0.3)
                else:
                    plt.text(0.5, 0.5, 'No finite RTT values to plot', ha='center', va='center')
                    plt.axis('off')
                pdf.savefig(bbox_inches='tight')
                plt.close()

                # Page 2: Per-sample boxplot
                if per_sample_rtts:
                    plt.figure(figsize=(12, 6))
                    data = [
                        [r for r in sample if math.isfinite(r)] or [float('nan')]
                        for sample in per_sample_rtts
                    ]
                    plt.boxplot(data, showfliers=False)
                    plt.title('Per-Sample RTT Distribution (Boxplot)')
                    plt.xlabel('Sample Index')
                    plt.ylabel('Total RTT (s)')
                    plt.grid(True, axis='y', alpha=0.3)
                    pdf.savefig(bbox_inches='tight')
                    plt.close()
            print(f"Saved RTT summary PDF to {pdf_path}")
        except Exception as e:
            print(f"Matplotlib not available or plotting failed, skipping PDF: {str(e)}")
            
    else:
        print("No RTTs collected; skipping PDF generation")
    """

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
            reactive_results_paths = execute_brute_force_placement_optimization(
                apps, str(config_file), str(mapping_file), output_dir, samples,
                sim_input_path, workload_base_file, max_workers, max_combinations
            )

        # print(reactive_results_paths)
        logger.info("Completed all simulations")

        # logger.info('Finished model training')
        logger.info(f'All files can be found under {output_dir}')


    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
