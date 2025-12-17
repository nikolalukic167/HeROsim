"""
Generate deterministic infrastructure configuration for a dataset.

This script pre-generates:
1. Network topology (connections and latencies)
2. Replica placements (which platforms get replicas)
3. Queue distributions (how many warmup tasks per platform)

All randomness is seeded and saved to infrastructure.json
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.utils.distributions import sample_bounded_int, sample_replica_count


def generate_network_topology_deterministic(
    nodes: List[Dict],
    config: Dict[str, Any],
    rng: random.Random
) -> Dict[str, Dict[str, float]]:
    """
    Generate network topology deterministically using seeded RNG.
    
    Args:
        nodes: List of node configurations
        config: Configuration containing network latency and topology settings
        rng: Seeded random number generator
    
    Returns:
        Dictionary mapping node names to their network maps
    """
    network_config = config.get('network', {})
    latency_config = network_config.get('latency', {})
    topology_config = network_config.get('topology', {})
    
    device_latencies = latency_config.get('device_latencies', {})
    base_latency = latency_config.get('base_latency', 0.1)
    topology_type = topology_config.get('type', 'sparse')
    connection_probability = topology_config.get('connection_probability', 0.85)
    custom_edges = topology_config.get('edges', [])
    
    # Separate clients and servers
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
            return rng.uniform(min_latency, max_latency)
        else:
            return base_latency
    
    if topology_type == 'custom' and custom_edges:
        # Use custom topology edges
        for edge in custom_edges:
            if len(edge) == 2:
                client_name, server_name = edge
                
                client_node = next((n for n in clients if n['node_name'] == client_name), None)
                server_node = next((n for n in servers if n['node_name'] == server_name), None)
                
                if client_node and server_node:
                    latency = generate_latency(client_node['type'], server_node['type'])
                    network_maps[client_name][server_name] = latency
                    network_maps[server_name][client_name] = latency
    else:
        # Generate connections based on connection probability
        for client in clients:
            client_name = client['node_name']
            client_type = client['type']
            
            for server in servers:
                server_name = server['node_name']
                server_type = server['type']
                
                if rng.random() < connection_probability:
                    latency = generate_latency(client_type, server_type)
                    network_maps[client_name][server_name] = latency
                    network_maps[server_name][client_name] = latency
    
    # Ensure minimum connectivity
    for node_name, connections in network_maps.items():
        if len(connections) == 0:
            if node_name.startswith('client_node'):
                available_servers = [s for s in servers if s['node_name'] not in connections]
                if available_servers:
                    server = rng.choice(available_servers)
                    server_name = server['node_name']
                    server_type = server['type']
                    client_type = next(n['type'] for n in clients if n['node_name'] == node_name)
                    latency = generate_latency(client_type, server_type)
                    network_maps[node_name][server_name] = latency
                    network_maps[server_name][node_name] = latency
            else:
                available_clients = [c for c in clients if c['node_name'] not in connections]
                if available_clients:
                    client = rng.choice(available_clients)
                    client_name = client['node_name']
                    client_type = client['type']
                    server_type = next(n['type'] for n in servers if n['node_name'] == node_name)
                    latency = generate_latency(client_type, server_type)
                    network_maps[node_name][client_name] = latency
                    network_maps[client_name][node_name] = latency
    
    return network_maps


def generate_replica_placements_deterministic(
    nodes: List[Dict],
    config: Dict[str, Any],
    sim_inputs: Dict[str, Any],
    rng: random.Random
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate replica placements deterministically.
    
    Returns:
        {
            task_type: [
                {"node_name": str, "platform_id": int, "platform_type": str}
            ]
        }
    """
    preinit_config = config.get('preinit', {})
    replicas_config = config.get('replicas', {})
    
    # Get preinit nodes
    all_client_nodes = [node for node in nodes if node.get('node_name', '').startswith('client_node')]
    all_server_nodes = [node for node in nodes if not node.get('node_name', '').startswith('client_node')]
    
    # Handle percentage-based configuration
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
    
    # Create node_id and platform_id mappings (same as simulation.py)
    node_id_map = {node['node_name']: i for i, node in enumerate(nodes)}
    platform_id = 0
    node_platforms = {}  # node_name -> list of (platform_id, platform_type)
    
    for node in nodes:
        node_name = node['node_name']
        node_platforms[node_name] = []
        for platform_type_name in node.get('platforms', []):
            node_platforms[node_name].append({
                'platform_id': platform_id,
                'platform_type': platform_type_name
            })
            platform_id += 1
    
    # Generate replica placements
    replica_placements = {}
    task_types = sim_inputs.get('task_types', {})
    
    assigned_platforms = set()  # Set of (node_name, platform_id) tuples
    
    for task_type_name, replica_config in replicas_config.items():
        if task_type_name not in task_types:
            continue
        
        task_type = task_types[task_type_name]
        supported_platforms = task_type.get('platforms', [])
        
        placements = []
        
        # Create server replicas
        per_server = replica_config.get('per_server', 0)
        if per_server > 0:
            for node in all_server_nodes:
                node_name = node['node_name']
                if node_name in preinit_servers:
                    suitable_platforms = [
                        p for p in node_platforms[node_name]
                        if p['platform_type'] in supported_platforms
                        and (node_name, p['platform_id']) not in assigned_platforms
                    ]
                    
                    replicas_created = 0
                    for platform_info in suitable_platforms:
                        if replicas_created >= per_server:
                            break
                        
                        platform_key = (node_name, platform_info['platform_id'])
                        placements.append({
                            'node_name': node_name,
                            'platform_id': platform_info['platform_id'],
                            'platform_type': platform_info['platform_type']
                        })
                        assigned_platforms.add(platform_key)  # Mark as assigned
                        replicas_created += 1
        
        # Create client replicas
        per_client = replica_config.get('per_client', 0)
        if per_client > 0:
            for node in all_client_nodes:
                node_name = node['node_name']
                if node_name in preinit_clients:
                    suitable_platforms = [
                        p for p in node_platforms[node_name]
                        if p['platform_type'] in supported_platforms
                        and (node_name, p['platform_id']) not in assigned_platforms
                    ]
                    
                    replicas_created = 0
                    for platform_info in suitable_platforms:
                        if replicas_created >= per_client:
                            break
                        
                        platform_key = (node_name, platform_info['platform_id'])
                        placements.append({
                            'node_name': node_name,
                            'platform_id': platform_info['platform_id'],
                            'platform_type': platform_info['platform_type']
                        })
                        assigned_platforms.add(platform_key)  # Mark as assigned
                        replicas_created += 1
        
        replica_placements[task_type_name] = placements
    
    print(f"\n[infra-gen] Replica placement summary:")
    for task_type, placements in replica_placements.items():
        print(f"  {task_type}: {len(placements)} replicas")
    print(f"  Total unique platforms assigned: {len(assigned_platforms)}")
    
    # verify no duplicates
    all_platform_keys = []
    for placements in replica_placements.values():
        for p in placements:
            platform_key = (p['node_name'], p['platform_id'])
            all_platform_keys.append(platform_key)
    
    if len(all_platform_keys) != len(set(all_platform_keys)):
        duplicates = [k for k in all_platform_keys if all_platform_keys.count(k) > 1]
        raise RuntimeError(
            f"CRITICAL: Found duplicate platform assignments: {duplicates}. "
            f"This should not happen - each platform can only be assigned to one task type."
        )
    
    return replica_placements


def generate_queue_distributions_deterministic(
    replica_placements: Dict[str, List[Dict[str, Any]]],
    config: Dict[str, Any],
    rng: random.Random
) -> Dict[str, Dict[str, int]]:
    """
    Generate queue distributions deterministically.
    
    Returns:
        {
            task_type: {
                "node_name:platform_id": queue_length
            }
        }
    """
    queue_distributions = {}
    prewarm_config = config.get('prewarm', {})
    
    for task_type_name, placements in replica_placements.items():
        task_prewarm = prewarm_config.get(task_type_name, {})
        queue_dist = {}
        
        for placement in placements:
            node_name = placement['node_name']
            platform_id = placement['platform_id']
            platform_key = f"{node_name}:{platform_id}"
            
            # Get queue distribution parameters
            initial_queue = task_prewarm.get('initial_queue', 0)
            
            if task_prewarm.get('queue_distribution') == 'statistical':
                q_params = task_prewarm.get('queue_distribution_params') or {}
                if 'min' not in q_params:
                    q_params['min'] = 0
                sampled_q = sample_bounded_int(q_params, rng)
                queue_length = max(0, int(sampled_q))
            else:
                queue_length = initial_queue
            
            queue_dist[platform_key] = queue_length
        
        queue_distributions[task_type_name] = queue_dist
    
    return queue_distributions


def generate_deterministic_infrastructure(
    config_file: str,
    sim_input_path: Path,
    output_file: str,
    seed: int
) -> Dict[str, Any]:
    """
    Generate deterministic infrastructure.
    
    Args:
        config_file: Path to infrastructure configuration file
        sim_input_path: Path to simulation input files directory
        output_file: Path to output infrastructure.json file
        seed: Random seed for deterministic generation
    
    Returns:
        Infrastructure dictionary
    """
    # Load config
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Load simulation inputs
    from src.executecosimulation import load_simulation_inputs
    sim_inputs = load_simulation_inputs(sim_input_path)
    
    # Create seeded RNG
    rng = random.Random(seed)
    
    # Generate nodes (same logic as prepare_simulation_config)
    client_nodes_count = config['nodes']['client_nodes']['count']
    server_nodes_count = config['nodes']['server_nodes']['count']
    
    nodes = []
    device_types = list(config['pci'].keys())
    
    # Generate client nodes
    for i in range(client_nodes_count):
        device_type = device_types[i % len(device_types)]
        device_specs = config['pci'][device_type]['specs']
        node_config = device_specs.copy()
        node_config['node_name'] = f"client_node{i}"
        node_config['type'] = device_type
        nodes.append(node_config)
    
    # Generate server nodes
    for i in range(server_nodes_count):
        device_type = device_types[i % len(device_types)]
        device_specs = config['pci'][device_type]['specs']
        node_config = device_specs.copy()
        node_config['node_name'] = f"node{i}"
        node_config['type'] = device_type
        nodes.append(node_config)
    
    # 1. Generate network topology
    print("[infra-gen] Generating network topology...")
    network_maps = generate_network_topology_deterministic(nodes, config, rng)
    
    # 2. Generate replica placements
    print("[infra-gen] Generating replica placements...")
    replica_placements = generate_replica_placements_deterministic(
        nodes, config, sim_inputs, rng
    )
    
    # 3. Generate queue distributions
    print("[infra-gen] Generating queue distributions...")
    queue_distributions = generate_queue_distributions_deterministic(
        replica_placements, config, rng
    )
    
    infrastructure = {
        "network_maps": network_maps,
        "replica_placements": replica_placements,
        "queue_distributions": queue_distributions,
        "metadata": {
            "seed": seed,
            "config_file": config_file,
            "generation_time": datetime.now().isoformat()
        }
    }
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(infrastructure, f, indent=2)
    
    print(f"[infra-gen] Generated deterministic infrastructure: {output_file}")
    print(f"  Network maps: {len(network_maps)} nodes")
    print(f"  Replica placements: {sum(len(v) for v in replica_placements.values())} total replicas")
    print(f"  Queue distributions: {sum(len(v) for v in queue_distributions.values())} platforms")
    
    return infrastructure


def main():
    """Main entry point for infrastructure generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate deterministic infrastructure configuration"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to infrastructure configuration file'
    )
    parser.add_argument(
        '--sim-input',
        type=str,
        required=True,
        help='Path to simulation input files directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output infrastructure.json file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        required=True,
        help='Random seed for deterministic generation'
    )
    
    args = parser.parse_args()
    
    generate_deterministic_infrastructure(
        args.config,
        Path(args.sim_input),
        args.output,
        args.seed
    )
    print("[infra-gen] Done.")


if __name__ == "__main__":
    main()

