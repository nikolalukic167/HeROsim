#!/usr/bin/env python3
"""
Optimized GNN Dataset Generation Script

This Python script replaces generate_gnn_datasets.sh with significant performance improvements:
1. Eliminates jq overhead (native Python JSON handling)
2. Eliminates subprocess spawning for infrastructure generation
3. Single Python process for all operations
4. Supports --quiet mode for faster execution
5. Uses orjson for faster JSON serialization when available

Usage:
    python scripts_cosim/generate_gnn_datasets_fast.py [--quiet] [--max-datasets N] [--workers N]

Example:
    # Generate up to 100 datasets with quiet mode
    python scripts_cosim/generate_gnn_datasets_fast.py --quiet --max-datasets 100
"""

import argparse
import json
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to use orjson for faster JSON
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
    
    def json_dumps(obj):
        return orjson.dumps(_convert_keys_to_str(obj)).decode('utf-8')
    def json_dumps_pretty(obj):
        return orjson.dumps(_convert_keys_to_str(obj), option=orjson.OPT_INDENT_2).decode('utf-8')
    HAS_ORJSON = True
except ImportError:
    def json_dumps(obj):
        return json.dumps(obj, separators=(',', ':'))
    def json_dumps_pretty(obj):
        return json.dumps(obj, indent=2)
    HAS_ORJSON = False

from src.generate_infrastructure import generate_deterministic_infrastructure
from src.executecosimulation import execute_brute_force_optimized, load_simulation_inputs


# =============================================================================
# CONFIGURATION GRIDS
# =============================================================================

# Connection probabilities for network topology
# NOTE: Start with higher connectivity (0.50+) for reliable placement generation
# Low connectivity (< 0.40) often causes "no feasible placement" failures
CONNECTION_PROBABILITIES = [
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
    # Lower connectivity (more challenging, may have higher failure rate)
    0.40, 0.45, 0.35, 0.30, 0.25, 0.20
]

# Replica configurations: (per_client, per_server, client_preinit_pct, server_preinit_pct)
# NOTE: Configurations with 0% preinit cause "no active replicas" issues because
# replicas are only created during task execution. Start with moderate preinit.
REPLICA_CONFIGS = [
    # Moderate preinit (50-70%) - most reliable
    (2, 2, 0.5, 0.8),
    (3, 3, 0.5, 0.7),
    (2, 3, 0.6, 0.8),
    (1, 3, 0.3, 0.7),
    # High preinit (70-100%)
    (2, 2, 0.6, 0.8),
    (1, 4, 0.6, 0.6),
    (2, 4, 0.8, 0.7),
    (3, 2, 0.7, 0.8),
    (1, 2, 0.5, 0.9),
    (3, 1, 0.6, 0.8),
    (1, 1, 0.4, 0.8),
    # Warm start scenarios (30-50% preinit)
    (3, 3, 0.3, 0.5),
    (2, 3, 0.4, 0.5),
    (2, 1, 0.5, 0.4),
    # Cold start scenarios (0% preinit) - DISABLED: causes no-replica issues
    # These require dynamic autoscaling which doesn't work well with brute-force
    # (3, 3, 0.0, 0.0),
    # (2, 3, 0.0, 0.0),
    # (1, 3, 0.0, 0.0),
]

# Queue distribution configurations: (name, type, param1, param2, min, max, step)
QUEUE_DISTRIBUTIONS = [
    # Zero queues (cold start - most realistic)
    ("zero", "constant", 0, 0, 0, 0, 0),
    # Small queues (realistic initial load)
    ("pois2", "poisson", 2, 0, 0, 8, 1),
    ("pois4", "poisson", 4, 0, 0, 12, 1),
    ("pois6", "poisson", 6, 0, 0, 16, 1),
    # Moderate queues (warm start)
    ("pois8", "poisson", 8, 0, 0, 20, 1),
    ("norm8", "normal", 8, 3, 0, 20, 1),
    ("norm12", "normal", 12, 4, 0, 24, 1),
    # Larger queues (hot start - less common)
    ("pois12", "poisson", 12, 0, 0, 24, 1),
    ("norm16", "normal", 16, 5, 0, 32, 1),
]

# Seeds for deterministic generation
SEEDS = [101]

# Task type ratios: (dnn1%, dnn2%)
TASK_TYPE_RATIOS = [
    (0, 100), (10, 90), (20, 80), (30, 70), (40, 60),
    (50, 50), (60, 40), (70, 30), (80, 20), (90, 10), (100, 0)
]

# Workload parameters
NUM_TASKS = 5
NUM_CLIENT_NODES = 10
NUM_WORKLOAD_TEMPLATES = 10


def log(msg: str, quiet: bool = False, force: bool = False):
    """Print message unless in quiet mode."""
    if not quiet or force:
        print(msg)


def generate_workload_templates(
    base_workload_path: Path,
    output_dir: Path,
    num_templates: int = NUM_WORKLOAD_TEMPLATES,
    quiet: bool = False
) -> List[Path]:
    """
    Generate workload templates with varied task type ratios.
    
    Returns list of paths to generated template files.
    """
    with open(base_workload_path, 'r') as f:
        base_workload = json.load(f)
    
    templates = []
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for template_idx in range(num_templates):
        # Cycle through task type ratios
        dnn1_pct, dnn2_pct = TASK_TYPE_RATIOS[template_idx % len(TASK_TYPE_RATIOS)]
        
        num_dnn1 = NUM_TASKS * dnn1_pct // 100
        num_dnn2 = NUM_TASKS - num_dnn1
        
        # Create task types list
        task_types = ['dnn1'] * num_dnn1 + ['dnn2'] * num_dnn2
        
        # Random client node assignments
        client_nodes = [random.randint(0, NUM_CLIENT_NODES - 1) for _ in range(NUM_TASKS)]
        
        # Create workload
        workload = {
            'rps': base_workload.get('rps', 10),
            'duration': base_workload.get('duration', 10),
            'events': []
        }
        
        base_events = base_workload.get('events', [])
        for idx in range(NUM_TASKS):
            base_event = deepcopy(base_events[idx % len(base_events)])
            task_type = task_types[idx]
            client_node = client_nodes[idx]
            
            base_event['application']['name'] = f"nofs-{task_type}"
            base_event['application']['dag'] = {task_type: []}
            base_event['node_name'] = f"client_node{client_node}"
            
            workload['events'].append(base_event)
        
        # Save template
        template_path = output_dir / f"workload_template_{template_idx}.json"
        with open(template_path, 'w') as f:
            f.write(json_dumps_pretty(workload))
        
        templates.append(template_path)
        
        if not quiet:
            log(f"  Template {template_idx}: {num_dnn1} dnn1 + {num_dnn2} dnn2")
    
    return templates


def create_config_for_iteration(
    base_config: Dict[str, Any],
    connection_prob: float,
    replica_cfg: Tuple[int, int, float, float],
    seed: int,
    queue_dist: Tuple[str, str, int, int, int, int, int]
) -> Dict[str, Any]:
    """
    Create a modified config for a specific iteration.
    
    This replaces the jq-based config modification in the bash script.
    """
    config = deepcopy(base_config)
    
    per_client, per_server, client_pct, server_pct = replica_cfg
    qname, qtype, qp1, qp2, qmin, qmax, qstep = queue_dist
    
    # Network topology
    if 'network' not in config:
        config['network'] = {}
    if 'topology' not in config['network']:
        config['network']['topology'] = {}
    config['network']['topology']['connection_probability'] = connection_prob
    config['network']['topology']['seed'] = seed
    
    # Preinit configuration
    config['preinit'] = {
        'client_percentage': client_pct,
        'server_percentage': server_pct
    }
    
    # Replica configuration
    config['replicas'] = {
        'dnn1': {'per_client': per_client, 'per_server': per_server},
        'dnn2': {'per_client': per_client, 'per_server': per_server}
    }
    
    # Queue distribution parameters
    if qtype == "constant":
        q_params = {'type': 'constant', 'value': qp1, 'min': qmin, 'max': qmax, 'step': qstep}
    elif qtype == "poisson":
        q_params = {'type': 'poisson', 'lambda': qp1, 'min': qmin, 'max': qmax, 'step': qstep}
    elif qtype == "normal":
        stddev = qp2 if qp2 != 0 else 1
        q_params = {'type': 'normal', 'mean': qp1, 'stddev': stddev, 'min': qmin, 'max': qmax, 'step': qstep}
    elif qtype == "uniform":
        q_params = {'type': 'uniform', 'low': qp1, 'high': qp2, 'min': qmin, 'max': qmax, 'step': qstep}
    else:
        q_params = {'type': 'poisson', 'lambda': 4, 'min': qmin, 'max': qmax, 'step': qstep}
    
    config['prewarm'] = {
        'dnn1': {
            'distribution': 'none',
            'queue_distribution': 'statistical',
            'queue_distribution_params': q_params
        },
        'dnn2': {
            'distribution': 'none',
            'queue_distribution': 'statistical',
            'queue_distribution_params': q_params
        }
    }
    
    return config


def generate_single_dataset(
    dataset_id: str,
    output_dir: Path,
    config: Dict[str, Any],
    workload_template: Path,
    sim_input_path: Path,
    samples_file: Path,
    mapping_file: Path,
    seed: int,
    max_workers: int,
    quiet: bool = False
) -> Tuple[bool, float, float]:
    """
    Generate a single GNN dataset.
    
    Returns (success, rtt, duration_seconds)
    """
    start_time = time.time()
    
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config for this dataset
        config_path = output_dir / "space_with_network.json"
        with open(config_path, 'w') as f:
            f.write(json_dumps_pretty(config))
        
        # Copy workload template
        workload_path = output_dir / "workload.json"
        with open(workload_template, 'r') as f:
            workload = json.load(f)
        with open(workload_path, 'w') as f:
            f.write(json_dumps_pretty(workload))
        
        # Copy workload to expected location for simulation
        traces_dir = sim_input_path / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        with open(traces_dir / "workload-10.json", 'w') as f:
            f.write(json_dumps_pretty(workload))
        
        # Generate infrastructure
        infra_file = output_dir / "infrastructure.json"
        log(f"  Generating infrastructure...", quiet)
        generate_deterministic_infrastructure(
            str(config_path),
            sim_input_path,
            str(infra_file),
            seed
        )
        
        # Load sample
        samples = np.load(samples_file)
        sample = samples[0]
        
        # Load apps from config
        apps = list(config['wsc'].keys())
        
        # Create results directory (temporary)
        results_dir = Path("simulation_data/initial_results_simple")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Run optimized brute-force simulation
        log(f"  Running brute-force optimization...", quiet)
        result_paths = execute_brute_force_optimized(
            apps=apps,
            config_file=str(config_path),
            mapping_file=str(mapping_file),
            output_dir=results_dir,
            sample=sample,
            sim_input_path=sim_input_path,
            workload_base_file=str(traces_dir / "workload-10.json"),
            max_workers=max_workers,
            infrastructure_file=infra_file,
            quiet=quiet
        )
        
        # Check for results and copy to dataset directory
        best_json = results_dir / "best.json"
        if best_json.exists():
            # Copy results to dataset directory
            with open(best_json, 'r') as f:
                best_info = json.load(f)
            
            optimal_rtt = best_info.get('rtt', float('inf'))
            optimal_file = best_info.get('file', '')
            
            # Copy best.json
            with open(output_dir / "best.json", 'w') as f:
                f.write(json_dumps(best_info))
            
            # Copy optimal result (use stdlib json to handle numpy types)
            optimal_src = results_dir / optimal_file
            if optimal_src.exists():
                import shutil
                shutil.copy2(optimal_src, output_dir / "optimal_result.json")
            
            # Copy placements
            placements_src = results_dir / "placements.jsonl"
            if placements_src.exists():
                (output_dir / "placements").mkdir(exist_ok=True)
                with open(placements_src, 'r') as f:
                    placements_content = f.read()
                with open(output_dir / "placements" / "placements.jsonl", 'w') as f:
                    f.write(placements_content)
            
            # Clean up results directory
            for f in results_dir.glob("simulation_*.json"):
                f.unlink()
            if best_json.exists():
                best_json.unlink()
            if placements_src.exists():
                placements_src.unlink()
            
            duration = time.time() - start_time
            return True, optimal_rtt, duration
        else:
            duration = time.time() - start_time
            return False, float('inf'), duration
            
    except Exception as e:
        duration = time.time() - start_time
        log(f"  ERROR: {e}", quiet, force=True)
        return False, float('inf'), duration


def main():
    parser = argparse.ArgumentParser(
        description="Optimized GNN Dataset Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress per-placement logging')
    parser.add_argument('--max-datasets', '-n', type=int, default=3000,
                        help='Maximum number of datasets to generate')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip datasets that already exist')
    args = parser.parse_args()
    
    quiet = args.quiet
    max_datasets = args.max_datasets
    cpu_count = os.cpu_count()
    max_workers = args.workers or (cpu_count - 1 if cpu_count and cpu_count > 1 else 1)
    
    # Paths
    base_dir = PROJECT_ROOT / "simulation_data"
    config_path = base_dir / "space_with_network.json"
    output_base = base_dir / "gnn_datasets"
    sim_input_path = PROJECT_ROOT / "data" / "nofs-ids"
    samples_file = base_dir / "lhs_samples_simple.npy"
    mapping_file = base_dir / "lhs_samples_simple_mapping.pkl"
    workload_base_file = sim_input_path / "traces" / "workload-10.json"
    workload_templates_dir = sim_input_path / "traces" / "gnn_templates"
    progress_log = PROJECT_ROOT / "logs" / "progress.txt"
    
    # Create directories
    output_base.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    
    log(f"=== Optimized GNN Dataset Generation ===", quiet)
    log(f"Max datasets: {max_datasets}", quiet)
    log(f"Workers: {max_workers}", quiet)
    log(f"Using orjson: {HAS_ORJSON}", quiet)
    log(f"Quiet mode: {quiet}", quiet)
    
    # Load base config
    with open(config_path, 'r') as f:
        base_config = json.load(f)
    
    # Generate workload templates
    log(f"\nGenerating workload templates...", quiet)
    templates = generate_workload_templates(
        workload_base_file,
        workload_templates_dir,
        NUM_WORKLOAD_TEMPLATES,
        quiet
    )
    log(f"Generated {len(templates)} workload templates", quiet)
    
    # Generate datasets
    log(f"\n=== Starting Dataset Generation ===", quiet)
    
    dataset_idx = 0
    template_idx = 0
    total_time = 0
    successful = 0
    failed = 0
    
    total_combinations = (
        len(CONNECTION_PROBABILITIES) * 
        len(REPLICA_CONFIGS) * 
        len(SEEDS) * 
        len(QUEUE_DISTRIBUTIONS)
    )
    log(f"Total possible combinations: {total_combinations}", quiet)
    
    start_time = time.time()
    
    for conn_prob in CONNECTION_PROBABILITIES:
        for replica_cfg in REPLICA_CONFIGS:
            for seed in SEEDS:
                for queue_dist in QUEUE_DISTRIBUTIONS:
                    if dataset_idx >= max_datasets:
                        break
                    
                    dataset_id = f"ds_{dataset_idx:05d}"
                    output_dir = output_base / dataset_id
                    
                    # Skip if resuming and already exists
                    if args.resume and (output_dir / "best.json").exists():
                        log(f"[{dataset_id}] Skipping (already exists)", quiet)
                        dataset_idx += 1
                        template_idx = (template_idx + 1) % NUM_WORKLOAD_TEMPLATES
                        continue
                    
                    # Get workload template
                    template = templates[template_idx]
                    
                    # Create config for this iteration
                    config = create_config_for_iteration(
                        base_config, conn_prob, replica_cfg, seed, queue_dist
                    )
                    
                    qname = queue_dist[0]
                    per_client, per_server, client_pct, server_pct = replica_cfg
                    
                    log(f"\n[{dataset_id}] conn={conn_prob} rpc={per_client} rps={per_server} "
                        f"cpct={client_pct} spct={server_pct} q={qname}", quiet)
                    
                    # Generate dataset
                    success, rtt, duration = generate_single_dataset(
                        dataset_id=dataset_id,
                        output_dir=output_dir,
                        config=config,
                        workload_template=template,
                        sim_input_path=sim_input_path,
                        samples_file=samples_file,
                        mapping_file=mapping_file,
                        seed=seed,
                        max_workers=max_workers,
                        quiet=quiet
                    )
                    
                    total_time += duration
                    
                    if success:
                        successful += 1
                        log(f"  SUCCESS: RTT={rtt:.3f}s ({duration:.1f}s)", quiet)
                        with open(progress_log, 'a') as f:
                            f.write(f"{dataset_id} SUCCESS {datetime.now().isoformat()} "
                                   f"{duration:.1f}s RTT={rtt:.3f}s q={qname}\n")
                    else:
                        failed += 1
                        log(f"  FAILED ({duration:.1f}s)", quiet)
                        with open(progress_log, 'a') as f:
                            f.write(f"{dataset_id} FAILED {datetime.now().isoformat()} "
                                   f"{duration:.1f}s\n")
                    
                    dataset_idx += 1
                    template_idx = (template_idx + 1) % NUM_WORKLOAD_TEMPLATES
                    
                    # Progress update
                    if dataset_idx % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = dataset_idx / elapsed if elapsed > 0 else 0
                        log(f"\n--- Progress: {dataset_idx}/{max_datasets} "
                            f"({100*dataset_idx/max_datasets:.1f}%) - "
                            f"{rate:.2f} datasets/min ---", quiet)
                
                if dataset_idx >= max_datasets:
                    break
            if dataset_idx >= max_datasets:
                break
        if dataset_idx >= max_datasets:
            break
    
    # Summary
    total_elapsed = time.time() - start_time
    
    log(f"\n=== Generation Complete ===", quiet, force=True)
    log(f"Total datasets: {dataset_idx}", quiet, force=True)
    log(f"Successful: {successful}", quiet, force=True)
    log(f"Failed: {failed}", quiet, force=True)
    log(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)", quiet, force=True)
    log(f"Average time per dataset: {total_elapsed/max(1, dataset_idx):.1f}s", quiet, force=True)
    log(f"Output directory: {output_base}", quiet, force=True)
    log(f"Progress log: {progress_log}", quiet, force=True)


if __name__ == "__main__":
    main()
