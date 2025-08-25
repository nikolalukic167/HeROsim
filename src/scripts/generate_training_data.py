#!/usr/bin/env python3
"""
Script to generate fully populated training data for GNN scheduling.
This script demonstrates how to create training data with all client-server pairs
and static conditions to ensure consistent, complete graphs.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

from src.generator.traces import generate_exhaustive_training_workload
from src.policy.gnn.training_scheduler import TrainingScheduler
from src.placement.executor import execute_sim
from src.placement.model import SimulationData, SimulationPolicy, PriorityPolicy
from src.placement.model import scheduling_strategies


def create_static_conditions(
    client_nodes: List[str],
    server_nodes: List[str],
    queue_lengths: Dict[int, int] = None,
    memory_usage: Dict[str, float] = None,
    storage_usage: Dict[int, int] = None,
    network_latencies: Dict[Tuple[str, str], float] = None
) -> Dict[str, Any]:
    """
    Create static conditions for consistent training data generation.
    
    Args:
        client_nodes: List of client node names
        server_nodes: List of server node names
        queue_lengths: Platform ID -> queue length mapping
        memory_usage: Node name -> memory usage percentage mapping
        storage_usage: Storage ID -> storage usage bytes mapping
        network_latencies: (client, server) -> latency mapping
    
    Returns:
        Dictionary of static conditions
    """
    if queue_lengths is None:
        # Default: random queue lengths between 0-10
        import random
        queue_lengths = {i: random.randint(0, 10) for i in range(100)}  # Platform IDs
    
    if memory_usage is None:
        # Default: random memory usage between 20-80%
        import random
        memory_usage = {node: random.uniform(0.2, 0.8) for node in client_nodes + server_nodes}
    
    if storage_usage is None:
        # Default: random storage usage
        import random
        storage_usage = {i: random.randint(1000000, 10000000) for i in range(50)}  # Storage IDs
    
    if network_latencies is None:
        # Default: generate network latencies for all client-server pairs
        import random
        network_latencies = {}
        for client in client_nodes:
            for server in server_nodes:
                if client != server:
                    # Random latency between 0.001-0.1 seconds
                    network_latencies[(client, server)] = random.uniform(0.001, 0.1)
    
    return {
        "queue_lengths": queue_lengths,
        "memory_usage": memory_usage,
        "storage_usage": storage_usage,
        "network_latencies": network_latencies
    }


def generate_training_dataset(
    sim_inputs: Dict[str, Any],
    infrastructure_config: Dict[str, Any],
    output_dir: Path,
    num_samples: int = 100,
    rps_per_pair: int = 5,
    duration_time: int = 60
) -> List[Dict[str, Any]]:
    """
    Generate a complete training dataset with fully populated graphs.
    
    Args:
        sim_inputs: Simulation inputs from load_simulation_inputs
        infrastructure_config: Infrastructure configuration
        output_dir: Output directory for training data
        num_samples: Number of training samples to generate
        rps_per_pair: Requests per second per client-server pair
        duration_time: Duration of each simulation in seconds
    
    Returns:
        List of training data samples
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract node information from infrastructure config
    client_nodes = []
    server_nodes = []
    
    # Parse infrastructure config to get node names
    for node_config in infrastructure_config.get("nodes", []):
        node_name = node_config.get("name", f"node_{len(client_nodes + server_nodes)}")
        node_type = node_config.get("type", "server")
        
        if node_type == "client":
            client_nodes.append(node_name)
        else:
            server_nodes.append(node_name)
    
    # If no explicit client nodes, use first few as clients
    if not client_nodes and server_nodes:
        num_clients = min(3, len(server_nodes))
        client_nodes = server_nodes[:num_clients]
        server_nodes = server_nodes[num_clients:]
    
    print(f"Client nodes: {client_nodes}")
    print(f"Server nodes: {server_nodes}")
    
    training_data = []
    
    for sample_idx in range(num_samples):
        print(f"Generating sample {sample_idx + 1}/{num_samples}")
        
        # Create static conditions for this sample
        static_conditions = create_static_conditions(
            client_nodes=client_nodes,
            server_nodes=server_nodes
        )
        
        # Create simulation data
        simulation_data = SimulationData(
            platform_types=sim_inputs['platform_types'],
            storage_types=sim_inputs['storage_types'],
            qos_types=sim_inputs['qos_types'],
            application_types=sim_inputs['application_types'],
            task_types=sim_inputs['task_types'],
        )
        
        # Generate exhaustive workload
        time_series = generate_exhaustive_training_workload(
            data=simulation_data,
            rps_per_pair=rps_per_pair,
            duration_time=duration_time,
            client_nodes=client_nodes,
            server_nodes=server_nodes,
            static_conditions=static_conditions,
            pdf_path=str(output_dir / f"workload_sample_{sample_idx}")
        )
        
        # Create simulation policy with training scheduler
        simulation_policy = SimulationPolicy(
            priority=PriorityPolicy(tasks="fifo"),
            scheduling="training_scheduler",  # Use our custom training scheduler
            cache="fifo",
            keep_alive=30,
            queue_length=100,
            short_name="training_scheduler",
            reconcile_interval=5
        )
        
        # Execute simulation
        try:
            stats = execute_sim(
                simulation_data=simulation_data,
                infrastructure=infrastructure_config,
                cache_policy="fifo",
                keep_alive=30,
                policy="fifo",
                queue_length=100,
                strategy="training_scheduler",
                workload_trace=time_series.to_dict(),
                workload_trace_name=f"training_sample_{sample_idx}",
                reconcile_interval=5
            )
            
            # Extract training data from simulation results
            sample_data = {
                "sample_id": sample_idx,
                "static_conditions": static_conditions,
                "client_nodes": client_nodes,
                "server_nodes": server_nodes,
                "task_results": [],
                "execution_times": {},
                "simulation_stats": stats
            }
            
            # Process task results to extract all execution times
            for task_result in stats.get("taskResults", []):
                # Get the task object to access all_execution_times
                # This would need to be stored during simulation
                task_data = {
                    "task_id": task_result["taskId"],
                    "source_node": task_result["sourceNode"],
                    "execution_node": task_result["executionNode"],
                    "task_type": task_result["taskType"]["name"],
                    "elapsed_time": task_result["elapsedTime"],
                    "execution_time": task_result["executionTime"],
                    "network_latency": task_result["networkLatency"],
                    "queue_time": task_result["queueTime"],
                    "cold_start_time": task_result["coldStartTime"]
                }
                sample_data["task_results"].append(task_data)
            
            training_data.append(sample_data)
            
            # Save individual sample
            sample_file = output_dir / f"training_sample_{sample_idx}.json"
            with open(sample_file, 'w') as f:
                json.dump(sample_data, f, indent=2, default=str)
            
        except Exception as e:
            print(f"Error generating sample {sample_idx}: {e}")
            continue
    
    # Save complete dataset
    dataset_file = output_dir / "complete_training_dataset.json"
    with open(dataset_file, 'w') as f:
        json.dump(training_data, f, indent=2, default=str)
    
    # Save metadata
    metadata = {
        "num_samples": len(training_data),
        "client_nodes": client_nodes,
        "server_nodes": server_nodes,
        "rps_per_pair": rps_per_pair,
        "duration_time": duration_time,
        "total_client_server_pairs": len(client_nodes) * len(server_nodes)
    }
    
    metadata_file = output_dir / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Generated {len(training_data)} training samples")
    print(f"Total client-server pairs: {len(client_nodes) * len(server_nodes)}")
    print(f"Output directory: {output_dir}")
    
    return training_data


def create_gnn_training_graphs(training_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert training data into GNN-compatible graph representations.
    
    Args:
        training_data: List of training data samples
    
    Returns:
        List of graph representations for GNN training
    """
    graphs = []
    
    for sample in training_data:
        client_nodes = sample["client_nodes"]
        server_nodes = sample["server_nodes"]
        static_conditions = sample["static_conditions"]
        
        # Create node features for all nodes
        node_features = {}
        
        # Client node features
        for client in client_nodes:
            node_features[client] = {
                "node_type": "client",
                "memory_usage": static_conditions["memory_usage"].get(client, 0.5),
                "queue_length": 0,  # Clients don't have queues
                "storage_usage": 0.0
            }
        
        # Server node features
        for server in server_nodes:
            node_features[server] = {
                "node_type": "server",
                "memory_usage": static_conditions["memory_usage"].get(server, 0.5),
                "queue_length": sum(static_conditions["queue_lengths"].values()) / len(static_conditions["queue_lengths"]),
                "storage_usage": sum(static_conditions["storage_usage"].values()) / len(static_conditions["storage_usage"])
            }
        
        # Create edge features for all client-server pairs
        edge_features = {}
        for client in client_nodes:
            for server in server_nodes:
                if client != server:
                    edge_key = (client, server)
                    edge_features[edge_key] = {
                        "network_latency": static_conditions["network_latencies"].get(edge_key, 0.01),
                        "execution_times": {}  # Will be populated from task results
                    }
        
        # Populate execution times from task results
        for task_result in sample["task_results"]:
            source = task_result["source_node"]
            execution = task_result["execution_node"]
            task_type = task_result["task_type"]
            
            if (source, execution) in edge_features:
                edge_features[(source, execution)]["execution_times"][task_type] = task_result["elapsed_time"]
        
        # Create graph representation
        graph = {
            "sample_id": sample["sample_id"],
            "node_features": node_features,
            "edge_features": edge_features,
            "static_conditions": static_conditions,
            "task_results": sample["task_results"]
        }
        
        graphs.append(graph)
    
    return graphs


def main():
    """Main function to demonstrate training data generation"""
    # Example usage
    output_dir = Path("training_data_output")
    
    # Example infrastructure config (you would load this from your actual config)
    infrastructure_config = {
        "nodes": [
            {"name": "client_node0", "type": "client"},
            {"name": "client_node1", "type": "client"},
            {"name": "server_node0", "type": "server"},
            {"name": "server_node1", "type": "server"},
            {"name": "server_node2", "type": "server"}
        ]
    }
    
    # Example simulation inputs (you would load this from your actual inputs)
    sim_inputs = {
        "platform_types": {
            "rpiCpu": {"name": "Raspberry Pi CPU", "shortName": "rpiCpu"},
            "xavierCpu": {"name": "Xavier CPU", "shortName": "xavierCpu"},
            "xavierGpu": {"name": "Xavier GPU", "shortName": "xavierGpu"}
        },
        "task_types": {
            "dnn1": {
                "name": "DNN1",
                "platforms": ["rpiCpu", "xavierCpu", "xavierGpu"],
                "executionTime": {"rpiCpu": 2.0, "xavierCpu": 1.0, "xavierGpu": 0.5},
                "coldStartDuration": {"rpiCpu": 1.0, "xavierCpu": 0.5, "xavierGpu": 0.2}
            },
            "dnn2": {
                "name": "DNN2",
                "platforms": ["rpiCpu", "xavierCpu", "xavierGpu"],
                "executionTime": {"rpiCpu": 3.0, "xavierCpu": 1.5, "xavierGpu": 0.8},
                "coldStartDuration": {"rpiCpu": 1.2, "xavierCpu": 0.6, "xavierGpu": 0.3}
            }
        },
        "application_types": {
            "app1": {"name": "Application 1"},
            "app2": {"name": "Application 2"}
        },
        "qos_types": {
            "qos1": {"name": "QoS 1", "maxDurationDeviation": 1.0}
        },
        "storage_types": {
            "local": {"name": "Local Storage", "remote": False},
            "remote": {"name": "Remote Storage", "remote": True}
        }
    }
    
    # Generate training data
    training_data = generate_training_dataset(
        sim_inputs=sim_inputs,
        infrastructure_config=infrastructure_config,
        output_dir=output_dir,
        num_samples=10,  # Small number for demonstration
        rps_per_pair=2,
        duration_time=30
    )
    
    # Convert to GNN graphs
    graphs = create_gnn_training_graphs(training_data)
    
    # Save graphs
    graphs_file = output_dir / "gnn_training_graphs.pkl"
    with open(graphs_file, 'wb') as f:
        pickle.dump(graphs, f)
    
    print(f"Generated {len(graphs)} GNN training graphs")
    print(f"Each graph contains {len(graphs[0]['node_features'])} nodes and {len(graphs[0]['edge_features'])} edges")


if __name__ == "__main__":
    main() 