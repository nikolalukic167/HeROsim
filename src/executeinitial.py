import json
import logging
import math
import os
import pickle
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Any, Tuple

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

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: matplotlib not available, visualization will be skipped")


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

    # File handler
    # fh = logging.FileHandler(output_dir / 'simulation.log')
    # fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # logger.addHandler(fh)
    logger.addHandler(ch)

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
    connection_probability = topology_config.get('connection_probability', 0.85)
    
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
    
    # Generate connections based on connection probability
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
    print(f"  Connection probability: {connection_probability}")
    print(f"  Total connections: {total_connections}")
    print(f"  Average connections per node: {total_connections / len(nodes):.1f}")
    print(f"  Expected connections: {len(clients) * len(servers) * connection_probability:.1f}")
    
    return network_maps


def visualize_network_topology(network_maps: Dict[str, Dict[str, float]], output_path: str = "network_topology.pdf"):
    """
    Visualize the network topology with client nodes in one line and server nodes in another line.
    
    Args:
        network_maps: Dictionary mapping node names to their network maps
        output_path: Path to save the PDF visualization
    """
    if not VISUALIZATION_AVAILABLE:
        print("Visualization skipped: matplotlib not available")
        return
    
    # Separate client and server nodes
    client_nodes = [name for name in network_maps.keys() if name.startswith('client_node')]
    server_nodes = [name for name in network_maps.keys() if not name.startswith('client_node')]
    
    # Sort nodes for consistent visualization
    client_nodes.sort()
    server_nodes.sort()
    
    # Create figure with larger size to accommodate all nodes
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    
    # Colors
    client_color = '#FF6B6B'  # Red for clients
    server_color = '#4ECDC4'  # Teal for servers
    connection_color = '#95A5A6'  # Gray for connections
    
    # Node positioning with better spacing
    client_y = 0.85
    server_y = 0.15
    
    # Calculate spacing to ensure good separation between nodes
    # Use a minimum spacing and scale based on available width
    min_spacing = 0.06  # Adjusted for better fit
    available_width = 0.8  # Use 80% of the width
    
    # Calculate spacing for each line
    client_spacing = max(min_spacing, available_width / max(len(client_nodes), 1))
    server_spacing = max(min_spacing, available_width / max(len(server_nodes), 1))
    
    # Use the larger spacing to ensure consistent layout
    node_spacing = max(client_spacing, server_spacing)
    
    # Adjust starting position to center the nodes
    client_start_x = 0.1 + (available_width - (len(client_nodes) - 1) * node_spacing) / 2
    server_start_x = 0.1 + (available_width - (len(server_nodes) - 1) * node_spacing) / 2
    
    # Node circle radius - make it larger to properly contain text
    node_radius = 0.025  # Adjusted for better fit with larger figure
    
    # Draw client nodes as circles
    client_positions = {}
    for i, node_name in enumerate(client_nodes):
        x = client_start_x + i * node_spacing
        client_positions[node_name] = (x, client_y)
        
        # Create node circle
        node_circle = plt.Circle((x, client_y), node_radius, 
                               facecolor=client_color, 
                               edgecolor='black', 
                               linewidth=2)
        ax.add_patch(node_circle)
        
        # Add node label with smaller font to fit in circle
        ax.text(x, client_y, node_name.replace('client_node', 'C'), 
                ha='center', va='center', fontsize=6, fontweight='bold')
    
    # Draw server nodes as circles
    server_positions = {}
    for i, node_name in enumerate(server_nodes):
        x = server_start_x + i * node_spacing
        server_positions[node_name] = (x, server_y)
        
        # Create node circle
        node_circle = plt.Circle((x, server_y), node_radius, 
                               facecolor=server_color, 
                               edgecolor='black', 
                               linewidth=2)
        ax.add_patch(node_circle)
        
        # Add node label with smaller font to fit in circle
        ax.text(x, server_y, node_name.replace('node', 'S'), 
                ha='center', va='center', fontsize=6, fontweight='bold')
    
    # Draw connections
    connection_count = 0
    for node_name, connections in network_maps.items():
        if node_name.startswith('client_node'):
            # Client node connections
            for target, latency in connections.items():
                if target in server_positions:
                    start_pos = client_positions[node_name]
                    end_pos = server_positions[target]
                    
                    # Draw connection line
                    ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                           color=connection_color, alpha=0.3, linewidth=0.5)
                    connection_count += 1
    
    # Add title and statistics with adjusted positioning
    ax.text(0.5, 0.98, f'Network Topology Visualization', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Add legend with smaller font
    client_legend = patches.Patch(color=client_color, label=f'Client Nodes ({len(client_nodes)})')
    server_legend = patches.Patch(color=server_color, label=f'Server Nodes ({len(server_nodes)})')
    ax.legend(handles=[client_legend, server_legend], loc='upper right', fontsize=10)
    
    # Add statistics with smaller font
    stats_text = f'Total Connections: {connection_count}\n'
    stats_text += f'Average Connections per Node: {connection_count / len(network_maps):.1f}'
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Set plot properties with better margins
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Save the plot with better margins
    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    
    print(f"Network topology visualization saved to: {output_path}")


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


def prepare_simulation_config(
        sample: np.ndarray,
        mapping: Dict[int, str],
        original_config: Dict[str, Any]
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
        "nodes": []
    }

    # Generate client nodes
    device_types = list(original_config['pci'].keys())  # ['rpi', 'xavier', 'pyngFpga']
    for i in range(client_nodes_count):
        # Cycle through device types for client nodes
        device_type = device_types[i % len(device_types)]
        device_specs = original_config['pci'][device_type]['specs']
        
        node_config = device_specs.copy()
        node_config['node_name'] = f"client_node{i}"
        node_config['type'] = device_type
        infrastructure_config['nodes'].append(node_config)

    # Generate server nodes
    for i in range(server_nodes_count):
        # Cycle through device types for server nodes
        device_type = device_types[i % len(device_types)]
        device_specs = original_config['pci'][device_type]['specs']
        
        node_config = device_specs.copy()
        node_config['node_name'] = f"node{i}"
        node_config['type'] = device_type
        infrastructure_config['nodes'].append(node_config)
    
    # Generate network maps using configuration-based approach
    network_maps = generate_network_latencies(infrastructure_config['nodes'], original_config)
    
    # Visualize the network topology
    try:
        output_dir = Path("simulation_data")
        output_dir.mkdir(exist_ok=True)
        viz_path = output_dir / "network_topology.pdf"
        visualize_network_topology(network_maps, str(viz_path))
    except Exception as e:
        print(f"Warning: Could not create network topology visualization: {e}")
    
    # Assign network maps to nodes
    for node in infrastructure_config['nodes']:
        node['network_map'] = network_maps[node['node_name']]

    return infrastructure_config


def execute_simulation(
        config: Dict[str, Any],
        sim_inputs: Dict[str, Any],
        scheduling_strategy: str,
        model_locations: Dict[str, str] = None,
        models: Dict[str, xgboost.XGBRegressor] = None,
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


def main():
    # Configuration paths
    base_dir = Path("simulation_data")
    sim_input_path = Path("data/nofs-ids")  # Base path for simulation input files
    samples_file = base_dir / "lhs_samples_simple.npy"
    mapping_file = base_dir / "lhs_samples_simple_mapping.pkl"
    config_file = base_dir / "space_with_network.json"
    # python -m src.generator -d data/nofs-ids --generate-traces --rps 10 --seconds 10
    workload_base_file = "data/nofs-ids/traces/workload-125-250.json"
    output_dir = base_dir / "initial_results_simple"
    os.makedirs(output_dir, exist_ok=True)
    # todo: max_workers = int(sys.argv[1])
    cpu_count = os.cpu_count()
    print("CPU count: ", cpu_count)
    max_workers = cpu_count - 1
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
        # reactive_results_paths = execute_reactive_samples(apps, infra_config, logger, mapping, output_dir, samples,
        #                                                   sim_inputs, workload_base)
        reactive_results_paths = execute_reactive_samples_parallel(apps, config_file, mapping_file, output_dir, samples,
                                                                   sim_input_path, workload_base_file, max_workers)
        # reactive_results_paths = ['simulation_data/initial_results_simple/simulation_{x}.json'  for x in [1,2,3,4,5]]
        # reactive_results_paths = ['simulation_data/initial_results_simple/simulation_{x}.json'  for x in [1]]
        print(reactive_results_paths)
        logger.info("Completed all simulations")

        # logger.info("Training model now...")
        # models, eval_results = train_model(output_dir, samples, include_queue_length=False)
        # model_paths = save_models(models, output_dir)

        # logger.info('Finished model training')
        logger.info(f'All files can be found under {output_dir}')


    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise e


import concurrent.futures
import json
from pathlib import Path


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

        # Prepare infrastructure configuration
        sim_config = prepare_simulation_config(sample, mapping, infra_config)
        print("started simulation 3")

        # Prepare workloads
        workloads = prepare_workloads(sample, mapping, workload_base, apps)
        # Flatten workloads into single sorted list
        flattened_workloads = flatten_workloads(workloads)

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
        scheduling_strategy = 'gnn_gnn'
        print("started simulation")
        result = execute_simulation(full_config, sim_inputs, scheduling_strategy,
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


def process_sample_proactive(args):
    i, sample, output_dir, sim_input_path, mapping_file, config_file, workload_base_file, apps, model_locations = args
    logger = setup_logging(output_dir)
    logger.info(f"Processing sample {i + 1}")

    try:
        sim_inputs = load_simulation_inputs(sim_input_path)

        with open(mapping_file, 'rb') as f:
            mapping = pickle.load(f)

        # Load infrastructure config
        with open(config_file, 'r') as f:
            infra_config = json.load(f)

        with open(workload_base_file, 'r') as f:
            workload_base = json.load(f)

        # Prepare infrastructure configuration
        sim_config = prepare_simulation_config(sample, mapping, infra_config)

        # Prepare workloads
        workloads = prepare_workloads(sample, mapping, workload_base, apps)
        # Flatten workloads into single sorted list
        flattened_workloads = flatten_workloads(workloads)

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
        scheduling_strategy = 'prokn_prokn'
        result = execute_simulation(full_config, sim_inputs, scheduling_strategy, model_locations=model_locations)
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

        # Convert tuple keys to string
        result = convert_tuple_keys_to_str(result)

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


def execute_proactive_samples_parallel(apps, config_file, mapping_file, output_dir, samples, sim_input_path,
                                       workload_base_file, max_workers, model_paths):
    result_paths = []

    # Use ProcessPoolExecutor for CPU-bound tasks, ThreadPoolExecutor for I/O-bound tasks
    # Adjust max_workers as needed based on your system's capabilities
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of (index, sample) tuples to process
        sample_tuples = [
            (i, sample, output_dir, sim_input_path, mapping_file, config_file, workload_base_file, apps, model_paths)
            for i, sample in enumerate(samples)]
        start_ts = time.time()
        # Submit all tasks and process results as they complete
        future_to_sample = {executor.submit(process_sample_proactive, sample_tuple): sample_tuple for sample_tuple in
                            sample_tuples}

        for future in concurrent.futures.as_completed(future_to_sample):
            result_file = future.result()
            if result_file is not None:
                result_paths.append(result_file)
        end_ts = time.time()
        print(f'Duration: {end_ts - start_ts}')
    return result_paths


"""def execute_reactive_samples(apps, infra_config, logger, mapping, output_dir, samples, sim_inputs, workload_base):
    result_paths = []
    for i, sample in enumerate(samples):
        logger.info(f"Processing sample {i + 1}/{len(samples)}")

        # Prepare infrastructure configuration
        sim_config = prepare_simulation_config(sample, mapping, infra_config)

        # Prepare workloads
        workloads = prepare_workloads(sample, mapping, workload_base, apps)
        # Flatten workloads into single sorted list
        flattened_workloads = flatten_workloads(workloads)

        # Combine infrastructure and workload configurations
        full_config = {
            "infrastructure": sim_config,
            "workload": flattened_workloads
        }

        # Execute simulation with additional inputs
        try:
            cache_policy = 'fifo'
            task_priority = 'fifo'
            keep_alive = KEEP_ALIVE
            queue_length = QUEUE_LENGTH
            scheduling_strategy = 'kn_kn'
            result = execute_simulation(full_config, sim_inputs, scheduling_strategy)
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
            result_paths.append(result_file)
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, cls=DataclassJSONEncoder)

            logger.info(f"Completed simulation {i + 1}")

        except Exception as e:
            logger.error(f"Error in simulation {i + 1}: {str(e)}")
            logger.exception(e)
    return result_paths


def execute_proactive_samples(apps, infra_config, logger, mapping, output_dir, samples, sim_inputs, workload_base,
                              models_locations):
    result_paths = []
    for i, sample in enumerate(samples):
        logger.info(f"Processing sample {i + 1}/{len(samples)}")

        # Prepare infrastructure configuration
        sim_config = prepare_simulation_config(sample, mapping, infra_config)

        # Prepare workloads
        workloads = prepare_workloads(sample, mapping, workload_base, apps)
        # Flatten workloads into single sorted list
        flattened_workloads = flatten_workloads(workloads)

        # Combine infrastructure and workload configurations
        full_config = {
            "infrastructure": sim_config,
            "workload": flattened_workloads
        }

        # Execute simulation with additional inputs
        try:
            cache_policy = 'fifo'
            task_priority = 'fifo'
            keep_alive = KEEP_ALIVE
            queue_length = QUEUE_LENGTH
            scheduling_strategy = 'prokn_prokn'
            result = execute_simulation(full_config, sim_inputs, scheduling_strategy, model_locations=models_locations,
                                        cache_policy=cache_policy,
                                        task_priority=task_priority,
                                        keep_alive=keep_alive,
                                        queue_length=queue_length)
            result['sample'] = {
                'apps': apps,
                'sample': sample.tolist(),
                'mapping': mapping,
                'infra_config': infra_config,
                'workload_base': workload_base,
                'models_locations': models_locations,
                'sim_inputs': sim_inputs,
                'scheduling_strategy': scheduling_strategy,
                'cache_policy': cache_policy,
                'task_priority': task_priority,
                'keep_alive': keep_alive,
                'queue_length': queue_length
            }

            # Save result
            result_file = output_dir / f"simulation_{i + 1}.json"
            result_paths.append(result_file)

            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, cls=DataclassJSONEncoder)

            logger.info(f"Completed simulation {i + 1}")

        except Exception as e:
            logger.error(f"Error in simulation {i + 1}: {str(e)}")
            logger.exception(e)
    return result_paths
"""

if __name__ == "__main__":
    main()
