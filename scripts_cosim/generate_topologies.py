#!/usr/bin/env python3
"""
Generate 3 distinct network topologies for GNN training diversity.

Topologies:
1. DENSE: 85% connection probability (many routing options)
2. SPARSE: 40% connection probability (limited routing, more constraints)
3. CLUSTERED: Star topology with specific server assignments

Run: python scripts_cosim/generate_topologies.py
"""

import json
import random
from pathlib import Path

def generate_network_maps(client_nodes, server_nodes, topology_type, base_latency_range=(0.005, 0.02)):
    """Generate network maps for all nodes based on topology type."""
    network_maps = {}
    
    # Initialize all network maps
    for node in client_nodes + server_nodes:
        network_maps[node] = {}
    
    if topology_type == "dense":
        # Dense: 85% connection probability
        prob = 0.85
        for client in client_nodes:
            for server in server_nodes:
                if random.random() < prob:
                    latency = random.uniform(*base_latency_range)
                    network_maps[client][server] = latency
                    network_maps[server][client] = latency
    
    elif topology_type == "sparse":
        # Sparse: 40% connection probability
        prob = 0.40
        for client in client_nodes:
            for server in server_nodes:
                if random.random() < prob:
                    latency = random.uniform(*base_latency_range)
                    network_maps[client][server] = latency
                    network_maps[server][client] = latency
        
        # Ensure each client has at least one server
        for client in client_nodes:
            if not network_maps[client]:
                server = random.choice(server_nodes)
                latency = random.uniform(*base_latency_range)
                network_maps[client][server] = latency
                network_maps[server][client] = latency
    
    elif topology_type == "clustered":
        # Clustered: Each client connects to a specific cluster of servers
        num_clusters = min(3, len(server_nodes))
        cluster_size = max(1, len(server_nodes) // num_clusters)
        
        for i, client in enumerate(client_nodes):
            cluster_idx = i % num_clusters
            cluster_start = cluster_idx * cluster_size
            cluster_end = min(cluster_start + cluster_size + 1, len(server_nodes))
            
            for j in range(cluster_start, cluster_end):
                server = server_nodes[j]
                latency = random.uniform(*base_latency_range)
                network_maps[client][server] = latency
                network_maps[server][client] = latency
    
    return network_maps


def generate_topology_file(output_path, topology_type, seed=42, num_clients=10, num_servers=10):
    """Generate a complete infrastructure file with the specified topology."""
    random.seed(seed)
    
    client_nodes = [f"client_node{i}" for i in range(num_clients)]
    server_nodes = [f"node{i}" for i in range(num_servers)]
    
    network_maps = generate_network_maps(client_nodes, server_nodes, topology_type)
    
    # Generate replica placements (deterministic based on seed)
    replica_placements = {}
    task_types = ["dnn1", "dnn2"]
    platform_id = 0
    
    for task_type in task_types:
        replica_placements[task_type] = []
        for server in server_nodes:
            # Create 2-3 replicas per server for each task type
            num_replicas = random.randint(2, 3)
            for _ in range(num_replicas):
                replica_placements[task_type].append({
                    "node_name": server,
                    "platform_id": platform_id,
                    "queue_snapshot": random.randint(0, 15)  # Initial queue state
                })
                platform_id += 1
    
    # Generate queue distributions (initial state)
    queue_distributions = {}
    for task_type in task_types:
        queue_distributions[task_type] = {}
        for placement in replica_placements[task_type]:
            key = f"{placement['node_name']}:{placement['platform_id']}"
            queue_distributions[task_type][key] = placement['queue_snapshot']
    
    infrastructure = {
        "metadata": {
            "topology_type": topology_type,
            "seed": seed,
            "num_clients": num_clients,
            "num_servers": num_servers,
            "generation_script": "generate_topologies.py"
        },
        "network_maps": network_maps,
        "replica_placements": replica_placements,
        "queue_distributions": queue_distributions
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(infrastructure, f, indent=2)
    
    print(f"Generated {topology_type} topology at {output_path}")
    print(f"  Clients: {num_clients}, Servers: {num_servers}")
    
    # Stats
    total_edges = sum(len(v) for v in network_maps.values()) // 2
    max_edges = num_clients * num_servers
    print(f"  Network edges: {total_edges}/{max_edges} ({100*total_edges/max_edges:.1f}% connectivity)")
    
    return infrastructure


def main():
    base_dir = Path("simulation_data/topologies")
    
    # Generate 3 distinct topologies
    topologies = [
        ("dense", 42),
        ("sparse", 123),
        ("clustered", 456)
    ]
    
    for topology_type, seed in topologies:
        output_path = base_dir / f"infrastructure_{topology_type}.json"
        generate_topology_file(output_path, topology_type, seed=seed)
        print()
    
    print(f"All topologies generated in {base_dir}")
    print("\nUsage in co-simulation:")
    print("  python -m src.executecosimulation --brute-force \\")
    print("    --infrastructure simulation_data/topologies/infrastructure_dense.json")


if __name__ == "__main__":
    main()
