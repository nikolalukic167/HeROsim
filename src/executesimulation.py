"""
Unified Simulation Executor with Policy Selection

This script runs simulations with different policies (vanilla knative or vanilla gnn)
for real simulation (full workload, no warmup tasks, autoscaling from zero).

Workflow:
1. Load space_with_network.json config file
2. Generate infrastructure (nodes + network topology) deterministically
3. Load workload from file
4. Run simulation with chosen policy (kn_network_kn_network or gnn_gnn)
5. Save simulation results

Usage:
    python -m src.executesimulation --config <space_config.json> --workload <workload.json> --policy knative [--seed <seed>]
    python -m src.executesimulation --config <space_config.json> --workload <workload.json> --policy gnn [--seed <seed>]
    python -m src.executesimulation --config <space_config.json> --workload <workload.json> --policy roundrobin [--seed <seed>]
    python -m src.executesimulation --config <space_config.json> --workload <workload.json> --policy knative_network [--seed <seed>]
    python -m src.executesimulation --config <space_config.json> --workload <workload.json> --policy herocache_network [--seed <seed>]
"""

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.motivational.constants import KEEP_ALIVE, QUEUE_LENGTH, RECONCILE_INTERVAL
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


def generate_network_topology_deterministic(
    nodes: List[Dict],
    config: Dict[str, Any],
    rng: random.Random,
    task_types_data: Optional[Dict[str, Any]] = None
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
    
    # Log network topology configuration
    print(f"\n=== Network Topology Generation ===")
    print(f"Topology type: {topology_type}")
    print(f"Connection probability: {connection_probability} ({connection_probability*100:.1f}%)")
    
    # Separate clients and servers
    clients = [node for node in nodes if node['node_name'].startswith('client_node')]
    servers = [node for node in nodes if not node['node_name'].startswith('client_node')]
    
    print(f"Nodes: {len(clients)} clients, {len(servers)} servers")
    print(f"Total possible client-server pairs: {len(clients) * len(servers)}")
    
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
    
    # Ensure platform-compatibility-aware connectivity for task types (dnn1 and dnn2)
    # Check each client node to ensure it can execute tasks (either locally or remotely)
    if task_types_data:
        # Check both dnn1 and dnn2
        for task_type_name in ['dnn1', 'dnn2']:
            if task_type_name not in task_types_data:
                continue
                
            task_type = task_types_data[task_type_name]
            compatible_platforms = set(task_type.get('platforms', []))
            
            for client in clients:
                client_name = client['node_name']
                client_platforms = set(client.get('platforms', []))
                
                # Check if client has compatible platforms locally
                has_local_support = bool(client_platforms & compatible_platforms)
                
                # Check if client is already connected to a server with compatible platforms
                has_remote_support = False
                for server_name in network_maps[client_name].keys():
                    server = next((s for s in servers if s['node_name'] == server_name), None)
                    if server:
                        server_platforms = set(server.get('platforms', []))
                        if bool(server_platforms & compatible_platforms):
                            has_remote_support = True
                            break
                
                # If client lacks both local and remote support, add connection to a server with support
                if not has_local_support and not has_remote_support:
                    # Find servers with compatible platforms
                    compatible_servers = [
                        s for s in servers
                        if bool(set(s.get('platforms', [])) & compatible_platforms)
                        and s['node_name'] not in network_maps[client_name]
                    ]
                    
                    if compatible_servers:
                        # Connect to a random server with support
                        server = rng.choice(compatible_servers)
                        server_name = server['node_name']
                        server_type = server['type']
                        client_type = client['type']
                        latency = generate_latency(client_type, server_type)
                        network_maps[client_name][server_name] = latency
                        network_maps[server_name][client_name] = latency
                        logging.info(
                            f"Added {task_type_name}-compatibility connection: {client_name} -> {server_name} "
                            f"(client platforms: {client_platforms}, server platforms: {set(server.get('platforms', []))})"
                        )
    
    return network_maps


def prepare_infrastructure_for_real_simulation(
        space_config: Dict[str, Any],
        seed: Optional[int] = None,
        sim_input_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Prepare infrastructure configuration for real simulation (no warmup, autoscaling from zero).
    
    This generates:
    - Nodes (client and server nodes with platforms)
    - Network topology (deterministic with seed)
    
    Does NOT generate:
    - Replica placements (autoscaling handles this)
    - Queue distributions (no warmup tasks)
    
    Args:
        space_config: Configuration from space_with_network.json
        seed: Random seed for deterministic network topology (default: from config or 42)
    
    Returns:
        Infrastructure configuration dictionary
    """
    # Get seed from config or use default
    if seed is None:
        topology_config = space_config.get('network', {}).get('topology', {})
        seed = topology_config.get('seed', 42)
    
    # Create seeded RNG
    rng = random.Random(seed)
    
    # Get node counts
    client_nodes_count = space_config['nodes']['client_nodes']['count']
    server_nodes_count = space_config['nodes']['server_nodes']['count']
    device_types = list(space_config['pci'].keys())
    
    # Generate nodes
    nodes = []
    
    # Generate client nodes
    for i in range(client_nodes_count):
        device_type = device_types[i % len(device_types)]
        device_specs = space_config['pci'][device_type]['specs']
        node_config = device_specs.copy()
        node_config['node_name'] = f"client_node{i}"
        node_config['type'] = device_type
        # network_map will be assigned after topology generation
        nodes.append(node_config)
    
    # Generate server nodes
    for i in range(server_nodes_count):
        device_type = device_types[i % len(device_types)]
        device_specs = space_config['pci'][device_type]['specs']
        node_config = device_specs.copy()
        node_config['node_name'] = f"node{i}"
        node_config['type'] = device_type
        # network_map will be assigned after topology generation
        nodes.append(node_config)
    
    # Load task-types.json for platform compatibility checks
    task_types_data = None
    if sim_input_path is not None:
        task_types_path = sim_input_path / "task-types.json"
        if task_types_path.exists():
            with open(task_types_path, 'r') as f:
                task_types_data = json.load(f)
    
    # Generate network topology deterministically
    network_maps = generate_network_topology_deterministic(nodes, space_config, rng, task_types_data=task_types_data)
    
    # Assign network maps to nodes
    for node in nodes:
        node['network_map'] = network_maps.get(node['node_name'], {})
    
    # Get network bandwidth (default to 1000.0 if not specified)
    network_bandwidth = space_config.get('network', {}).get('bandwidth', 1000.0)
    
    # Build infrastructure configuration for real simulation
    # NO preinitialize_platforms, NO replica_plan, NO deterministic placements
    infrastructure_config = {
        "network": {
            "bandwidth": float(network_bandwidth)
        },
        "nodes": nodes,
        # Real simulation: start with zero replicas, rely on autoscaling
        # No preinitialize_platforms flag
        # No replica_plan
        # No deterministic_replica_placements
        # No deterministic_queue_distributions
    }
    
    return infrastructure_config


def execute_simulation(
        config: Dict[str, Any],
        sim_inputs: Dict[str, Any],
        scheduling_strategy: str,
        cache_policy='fifo',
        task_priority='fifo',
        keep_alive=30,
        queue_length=30,
        models=None,
        reconcile_interval=1,
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
        'workload-simulation',
        models=models,
        reconcile_interval=reconcile_interval,
    )
    return {
        "status": "success",
        "config": config,
        "sim_inputs": sim_inputs,
        "stats": stats
    }


def load_gnn_model(model_path: Path):
    """Load the trained GNN model."""
    import torch
    from src.policy.gnn.gnn_model import TaskPlacementGNN
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading GNN model from {model_path} on {device}...", flush=True)
        
        # Model architecture must match training
        model = TaskPlacementGNN(
            task_feature_dim=3,
            platform_feature_dim=13,
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


def run_simulation(
        config_file: Path,
        workload_file: Path,
        output_file: Path,
        sim_input_path: Path,
        logger: logging.Logger,
        policy: str,
        seed: Optional[int] = None,
        gnn_model: Any = None,
        gnn_device: Any = None,
        task_types_data: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Run simulation with the specified policy.
    
    Args:
        config_file: Path to space_with_network.json config file
        workload_file: Path to workload JSON file
        output_file: Path to save simulation results
        sim_input_path: Path to simulation input files
        logger: Logger instance
        policy: Policy name ('knative', 'gnn', or 'roundrobin')
        seed: Random seed for deterministic network topology (optional)
        gnn_model: GNN model (required for gnn policy)
        task_types_data: Task types data (required for gnn policy)
    
    Returns True if successful, False if failed.
    """
    logger.info(f"Running {policy} simulation")

    # Validate policy
    valid_policies = [
        'knative',
        'gnn',
        'roundrobin',
        'knative_network',
        'herocache_network',
        'herocache_network_batch',
        'random_network',
        'offload_network',
    ]
    if policy not in valid_policies:
        logger.error(
            f"Invalid policy: {policy}. Must be one of: {', '.join(valid_policies)}"
        )
        return False

    # For GNN policy, check if model is provided
    if policy == 'gnn' and (gnn_model is None or task_types_data is None):
        logger.error(f"GNN policy requires gnn_model and task_types_data")
        return False

    # Check required files exist
    if not config_file.exists():
        logger.error(f"Config file not found: {config_file}")
        return False

    if not workload_file.exists():
        logger.error(f"Workload file not found: {workload_file}")
        return False

    try:
        # Load simulation inputs
        sim_inputs = load_simulation_inputs(sim_input_path)

        # Load space config
        with open(config_file, 'r') as f:
            space_config = json.load(f)

        # Load workload
        with open(workload_file, 'r') as f:
            workload = json.load(f)

        # Prepare infrastructure for real simulation
        infrastructure_config = prepare_infrastructure_for_real_simulation(
            space_config, seed=seed, sim_input_path=sim_input_path
        )

        # Combine into full config
        full_config = {
            "infrastructure": infrastructure_config,
            "workload": workload,
        }

        # Determine scheduling strategy
        scheduling_strategy = None
        models = None
        
        if policy == 'knative':
            scheduling_strategy = 'kn_network_kn_network'
            models = None
        elif policy == 'gnn':
            scheduling_strategy = 'gnn_gnn'
            models = {
                'gnn_model': gnn_model,
                'device': gnn_device,
                'task_types_data': task_types_data,
            }
        elif policy == 'roundrobin':
            scheduling_strategy = 'rr_network_rr_network'
            models = None
        elif policy == 'knative_network':
            scheduling_strategy = 'kn_network_kn_network'
            models = None
        elif policy == 'herocache_network':
            scheduling_strategy = 'hrc_network_hrc_network'
            models = None
        elif policy == 'herocache_network_batch':
            scheduling_strategy = 'hrc_network_batch_hrc_network_batch'
            models = None
        elif policy == 'random_network':
            scheduling_strategy = 'rp_network_rp_network'
            models = None
        elif policy == 'offload_network':
            scheduling_strategy = 'offload_network_offload_network'
            models = None
        
        if scheduling_strategy is None:
            logger.error(f"Unknown policy: {policy}")
            return False

        logger.info(f"Running {policy} simulation with strategy {scheduling_strategy}...")
        print(f"  Running {policy} simulation with strategy {scheduling_strategy}...")

        # Execute simulation
        result = execute_simulation(
            full_config,
            sim_inputs,
            scheduling_strategy=scheduling_strategy,
            cache_policy='fifo',
            task_priority='fifo',
            keep_alive=KEEP_ALIVE,
            queue_length=QUEUE_LENGTH,
            models=models,
            reconcile_interval=RECONCILE_INTERVAL,
        )

        # Extract stats
        stats = result.get('stats', {})
        # Use precomputed total_rtt/num_tasks when present (avoids holding full taskResults in memory)
        task_results = stats.get('taskResults', [])
        if stats.get('total_rtt') is not None and stats.get('num_tasks') is not None:
            total_rtt = stats['total_rtt']
            num_tasks = stats['num_tasks']
        else:
            total_rtt = sum(
                tr.get('elapsedTime', 0)
                for tr in task_results
                if tr.get('taskId') is not None and tr.get('taskId') >= 0
            )
            num_tasks = len([tr for tr in task_results if tr.get('taskId') is not None and tr.get('taskId') >= 0])

        # Build result summary
        result_summary = {
            "status": "success",
            "policy": policy,
            "scheduling_strategy": scheduling_strategy,
            "config_file": str(config_file),
            "workload_file": str(workload_file),
            "seed": seed,
            "total_rtt": total_rtt,
            "num_tasks": num_tasks,
            "stats": stats,
        }

        # Save result
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result_summary, f, indent=2, cls=DataclassJSONEncoder)

        logger.info(f"✓ Saved {output_file}")
        print(f"  ✓ Saved {output_file.name} (RTT: {total_rtt:.3f}s)")

        return True

    except Exception as e:
        logger.error(f"Error running simulation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main entry point for unified simulation executor.
    
    Usage:
        python -m src.executesimulation --config <space_config.json> --workload <workload.json> --policy knative [--seed <seed>] [--output <output.json>]
        python -m src.executesimulation --config <space_config.json> --workload <workload.json> --policy gnn [--seed <seed>] [--output <output.json>]
        python -m src.executesimulation --config <space_config.json> --workload <workload.json> --policy roundrobin [--seed <seed>] [--output <output.json>]
        python -m src.executesimulation --config <space_config.json> --workload <workload.json> --policy knative_network [--seed <seed>]
        python -m src.executesimulation --config <space_config.json> --workload <workload.json> --policy herocache_network [--seed <seed>] [--output <output.json>]
        python -m src.executesimulation --config <space_config.json> --workload <workload.json> --policy herocache_network [--seed <seed>] [--output <output.json>]
    """
    # Configuration
    sim_input_path = Path("data/nofs-ids")
    gnn_model_path = Path("models/firm-feather-4.pt")
    default_output_dir = Path("simulation_data/results")

    # Parse arguments
    config_file = None
    workload_file = None
    policy = None
    seed = None
    output_file = None

    if '--config' in sys.argv:
        idx = sys.argv.index('--config')
        if idx + 1 < len(sys.argv):
            config_file = Path(sys.argv[idx + 1])

    if '--workload' in sys.argv:
        idx = sys.argv.index('--workload')
        if idx + 1 < len(sys.argv):
            workload_file = Path(sys.argv[idx + 1])

    if '--policy' in sys.argv:
        idx = sys.argv.index('--policy')
        if idx + 1 < len(sys.argv):
            policy = sys.argv[idx + 1].lower()

    if '--seed' in sys.argv:
        idx = sys.argv.index('--seed')
        if idx + 1 < len(sys.argv):
            try:
                seed = int(sys.argv[idx + 1])
            except ValueError:
                print(f"ERROR: Invalid seed value: {sys.argv[idx + 1]}")
                sys.exit(1)

    if '--output' in sys.argv:
        idx = sys.argv.index('--output')
        if idx + 1 < len(sys.argv):
            output_file = Path(sys.argv[idx + 1])

    # Validate arguments
    if not config_file:
        print("ERROR: --config is required")
        print("Usage: python -m src.executesimulation --config <space_config.json> --workload <workload.json> --policy <knative|gnn> [--seed <seed>] [--output <output.json>]")
        sys.exit(1)

    if not workload_file:
        print("ERROR: --workload is required")
        print(
            "Usage: python -m src.executesimulation "
            "--config <space_config.json> --workload <workload.json> "
            "--policy <knative|gnn|roundrobin|knative_network|herocache_network|"
            "herocache_network_batch|random_network|offload_network> "
            "[--seed <seed>] [--output <output.json>]"
        )
        sys.exit(1)
    
    if not policy:
        print("ERROR: --policy is required")
        print(
            "Usage: python -m src.executesimulation "
            "--config <space_config.json> --workload <workload.json> "
            "--policy <knative|gnn|roundrobin|knative_network|herocache_network|"
            "herocache_network_batch|random_network|offload_network> "
            "[--seed <seed>] [--output <output.json>]"
        )
        sys.exit(1)
    
    cli_valid_policies = [
        'knative',
        'gnn',
        'roundrobin',
        'knative_network',
        'herocache_network',
        'herocache_network_batch',
        'random_network',
        'offload_network',
    ]
    if policy not in cli_valid_policies:
        print(f"ERROR: Invalid policy '{policy}'. Must be one of: {', '.join(cli_valid_policies)}")
        sys.exit(1)

    if not config_file.exists():
        print(f"ERROR: Config file not found: {config_file}")
        sys.exit(1)

    if not workload_file.exists():
        print(f"ERROR: Workload file not found: {workload_file}")
        sys.exit(1)

    # Set default output file if not provided
    if not output_file:
        default_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = default_output_dir / f"simulation_result_{policy}.json"

    # Setup logging
    logger = setup_logging(Path("."))

    # Load GNN model if needed
    gnn_model = None
    gnn_device = None
    task_types_data = None
    if policy == 'gnn':
        if not gnn_model_path.exists():
            print(f"ERROR: GNN model not found at {gnn_model_path}")
            sys.exit(1)
        
        gnn_model, gnn_device = load_gnn_model(gnn_model_path)
        task_types_data = load_task_types_data(sim_input_path)

    # Run simulation
    success = run_simulation(
        config_file, workload_file, output_file, sim_input_path, logger, policy,
        seed=seed, gnn_model=gnn_model, gnn_device=gnn_device, task_types_data=task_types_data
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

