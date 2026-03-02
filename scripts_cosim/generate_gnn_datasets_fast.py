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
import signal
import subprocess
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

# Timeout for brute-force simulation (1 hour per dataset)
SIMULATION_TIMEOUT = 900  # seconds


# =============================================================================
# CONFIGURATION GRIDS
# =============================================================================

# Connection probabilities for network topology
# NOTE: With cold start support, lower connectivity should work better now
# since we use all infrastructure replicas instead of captured active ones
CONNECTION_PROBABILITIES = [
    # Standard range - balanced diversity
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
    # Lower connectivity (more challenging but valid scenarios)
    0.25, 0.20
]

# Replica configurations: (per_client, per_server, client_preinit_pct, server_preinit_pct)
# NOTE: Cold start (0% preinit) now supported - uses all infrastructure replicas directly
REPLICA_CONFIGS = [
    # Cold start scenarios (0% preinit - force autoscaling) - MOST REALISTIC
    # Uses all replicas from infrastructure.json directly
    (3, 3, 0.0, 0.0),
    (2, 3, 0.0, 0.0),
    (1, 3, 0.0, 0.0),
    (2, 2, 0.0, 0.0),
    (1, 2, 0.0, 0.0),
    # Warm start scenarios (30-50% preinit)
    (3, 3, 0.3, 0.5),
    (2, 3, 0.4, 0.5),
    (2, 1, 0.5, 0.4),
    # Moderate preinit (50-70%)
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
]

# Queue distribution configurations: (name, type, param1, param2, min, max, step)
# IMPROVED: Added more high-queue scenarios to match real simulation (mean 51.35, max 2944)
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
    # Larger queues (hot start - matches real sim better)
    ("pois12", "poisson", 12, 0, 0, 24, 1),
    ("norm16", "normal", 16, 5, 0, 32, 1),
    # NEW: High-queue scenarios to match real simulation queue buildup
    ("pois20", "poisson", 20, 0, 0, 50, 1),
    ("norm25", "normal", 25, 8, 0, 60, 1),
    ("pois30", "poisson", 30, 0, 0, 80, 1),
    ("norm40", "normal", 40, 12, 0, 100, 1),
]

# Seeds for deterministic generation
SEEDS = [101]

# Task type ratios: (dnn1%, dnn2%)
TASK_TYPE_RATIOS = [
    (0, 100), (50, 50), (100, 0)
]

# Workload parameters (can be overridden via --num-tasks)
NUM_TASKS = 4
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
        
        # Create workload with improved duration for queue accumulation
        workload = {
            'rps': base_workload.get('rps', 1),
            'duration': 1,
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
    queue_dist: Tuple[str, str, int, int, int, int, int],
    batch_size: int = 4
) -> Dict[str, Any]:
    """
    Create a modified config for a specific iteration.
    
    Args:
        batch_size: Batch size for determined scheduler (should match num_tasks)
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
    
    # Set scheduler batch_size to match num_tasks (for determined scheduler)
    # This ensures scheduler processes tasks in batches matching the workload
    if 'scheduler' not in config:
        config['scheduler'] = {}
    config['scheduler']['batch_size'] = batch_size
    config['scheduler']['batch_timeout'] = 0.1
    
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
    quiet: bool = False,
    fast_forward_warmup: bool = True,
    fast_forward_threshold: int = 1,
    allow_non_unique_replicas: bool = True
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
        # NOTE: No timeout wrapper here because execute_brute_force_optimized uses ProcessPoolExecutor internally.
        # If it hangs, the process will need to be killed externally. The simulation itself should complete
        # within reasonable time for most datasets. If it hangs, check for deadlocks in the simulation code.
        log(f"  Running brute-force optimization (max {SIMULATION_TIMEOUT}s expected)...", quiet)
        
        sim_start = time.time()
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
            quiet=quiet,
            final_dataset_dir=output_dir,  # Write progress files to final dataset directory
            fast_forward_warmup=fast_forward_warmup,
            fast_forward_threshold=fast_forward_threshold,
            allow_non_unique_replicas=allow_non_unique_replicas
        )
        sim_duration = time.time() - sim_start
        
        # Warn if simulation took too long (but don't fail - it might be legitimate)
        if sim_duration > SIMULATION_TIMEOUT:
            log(f"  WARNING: Simulation took {sim_duration:.1f}s (exceeded {SIMULATION_TIMEOUT}s threshold)", quiet, force=True)
        
        # Check for results and copy to dataset directory
        best_json = results_dir / "best.json"
        placements_file = results_dir / "placements.jsonl"
        
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
            if placements_file.exists():
                (output_dir / "placements").mkdir(exist_ok=True)
                with open(placements_file, 'r') as f:
                    placements_content = f.read()
                with open(output_dir / "placements" / "placements.jsonl", 'w') as f:
                    f.write(placements_content)
            
            # Copy placement metadata if it exists (will be written by execute_brute_force_optimized)
            metadata_src = results_dir / "placement_metadata.json"
            if metadata_src.exists():
                import shutil
                shutil.copy2(metadata_src, output_dir / "placement_metadata.json")
            
            # Copy placement progress if it exists
            progress_src = results_dir / "placement_progress.txt"
            if progress_src.exists():
                import shutil
                shutil.copy2(progress_src, output_dir / "placement_progress.txt")
            
            # Clean up results directory
            for f in results_dir.glob("simulation_*.json"):
                f.unlink()
            if best_json.exists():
                best_json.unlink()
            if placements_file.exists():
                placements_file.unlink()
            if metadata_src.exists():
                metadata_src.unlink()
            if progress_src.exists():
                progress_src.unlink()
            
            duration = time.time() - start_time
            return 'success', optimal_rtt, duration
        else:
            # No results - check if this was an infeasible scenario (placements.jsonl empty or missing)
            duration = time.time() - start_time
            if placements_file.exists() and placements_file.stat().st_size == 0:
                # Empty placements file = infeasible scenario, skip gracefully
                placements_file.unlink()
                return 'skipped', float('inf'), duration
            else:
                # Some other issue
                return 'failed', float('inf'), duration
            
    except Exception as e:
        duration = time.time() - start_time
        log(f"  ERROR: {e}", quiet, force=True)
        return 'failed', float('inf'), duration


def main():
    parser = argparse.ArgumentParser(
        description="Optimized GNN Dataset Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--quiet', '-q', default=True, action='store_true',
                        help='Suppress per-placement logging (default: True)')
    parser.add_argument('--no-quiet', action='store_false', dest='quiet',
                        help='Disable quiet mode (show per-placement logging)')
    parser.add_argument('--max-datasets', '-n', type=int, default=300000,
                        help='Maximum number of datasets to generate')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip datasets that already exist')
    parser.add_argument('--start-from', type=int, default=0,
                        help='Start from dataset index (e.g., 118 to start from ds_00118)')
    parser.add_argument('--fast-forward-warmup', default=True, action='store_true',
                        help='Enable fast-forward warmup for queues > 1 task (default: True)')
    parser.add_argument('--no-fast-forward-warmup', action='store_false', dest='fast_forward_warmup',
                        help='Disable fast-forward warmup')
    parser.add_argument('--fast-forward-threshold', type=int, default=1,
                        help='Threshold for fast-forward warmup (default: 1)')
    parser.add_argument('--allow-non-unique-replicas', action='store_true',
                        help='Allow multiple tasks to share the same replica')
    parser.add_argument('--num-tasks', type=int, choices=[2, 3, 4], default=4,
                        help='Number of tasks per workload (2, 3 or 4). Sets batch_size accordingly.')
    args = parser.parse_args()
    
    quiet = args.quiet
    # max_datasets is relative to start_from (e.g., --start-from xyz --max-datasets 1 means generate ds_xyz only)
    max_datasets = args.start_from + args.max_datasets
    cpu_count = os.cpu_count()
    max_workers = args.workers or (cpu_count - 1 if cpu_count and cpu_count > 1 else 1)
    
    # Set NUM_TASKS based on argument
    global NUM_TASKS
    NUM_TASKS = args.num_tasks
    batch_size = NUM_TASKS  # Match batch_size to num_tasks
    
    # Paths
    base_dir = PROJECT_ROOT / "simulation_data"
    config_path = base_dir / "space_with_network.json"
    output_base = base_dir / f"gnn_datasets_{NUM_TASKS}tasks"
    sim_input_path = PROJECT_ROOT / "data" / "nofs-ids"
    samples_file = base_dir / "lhs_samples_simple.npy"
    mapping_file = base_dir / "lhs_samples_simple_mapping.pkl"
    workload_base_file = sim_input_path / "traces" / "workload-10.json"
    workload_templates_dir = sim_input_path / "traces" / "gnn_templates"
    progress_log = PROJECT_ROOT / "logs" / f"progress_{NUM_TASKS}tasks.txt"
    
    # Create directories
    output_base.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    
    log(f"=== Optimized GNN Dataset Generation ===", quiet)
    log(f"Num tasks: {NUM_TASKS} (batch_size={batch_size})", quiet)
    log(f"Max datasets: {args.max_datasets} (up to ds_{max_datasets-1:05d})", quiet)
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
    
    dataset_idx = args.start_from
    if dataset_idx > 0:
        log(f"Starting from dataset index: {dataset_idx} (ds_{dataset_idx:05d})", quiet)
    template_idx = 0
    total_time = 0
    successful = 0
    skipped = 0
    failed = 0
    
    total_combinations = (
        len(CONNECTION_PROBABILITIES) * 
        len(REPLICA_CONFIGS) * 
        len(SEEDS) * 
        len(QUEUE_DISTRIBUTIONS)
    )
    log(f"Total possible combinations: {total_combinations}", quiet)
    
    start_time = time.time()
    
    # Calculate starting positions for nested loops
    start_from = args.start_from
    current_idx = 0
    
    for conn_prob in CONNECTION_PROBABILITIES:
        for replica_cfg in REPLICA_CONFIGS:
            for seed in SEEDS:
                for queue_dist in QUEUE_DISTRIBUTIONS:
                    # Skip until we reach the starting index
                    if current_idx < start_from:
                        current_idx += 1
                        template_idx = (template_idx + 1) % NUM_WORKLOAD_TEMPLATES
                        continue
                    
                    if dataset_idx >= max_datasets:
                        break
                    
                    dataset_id = f"ds_{dataset_idx:05d}"
                    output_dir = output_base / dataset_id
                    
                    # Skip if resuming and already exists
                    if args.resume and (output_dir / "best.json").exists():
                        log(f"[{dataset_id}] Skipping (already exists)", quiet)
                        dataset_idx += 1
                        current_idx += 1
                        template_idx = (template_idx + 1) % NUM_WORKLOAD_TEMPLATES
                        continue
                    
                    # Get workload template
                    template = templates[template_idx]
                    
                    # Create config for this iteration (with batch_size matching num_tasks)
                    config = create_config_for_iteration(
                        base_config, conn_prob, replica_cfg, seed, queue_dist, batch_size=batch_size
                    )
                    
                    qname = queue_dist[0]
                    per_client, per_server, client_pct, server_pct = replica_cfg
                    
                    log(f"\n[{dataset_id}] conn={conn_prob} rpc={per_client} rps={per_server} "
                        f"cpct={client_pct} spct={server_pct} q={qname}", quiet)
                    
                    # Generate dataset
                    status, rtt, duration = generate_single_dataset(
                        dataset_id=dataset_id,
                        output_dir=output_dir,
                        config=config,
                        workload_template=template,
                        sim_input_path=sim_input_path,
                        samples_file=samples_file,
                        mapping_file=mapping_file,
                        seed=seed,
                        max_workers=max_workers,
                        quiet=quiet,
                        fast_forward_warmup=args.fast_forward_warmup,
                        fast_forward_threshold=args.fast_forward_threshold,
                        allow_non_unique_replicas=args.allow_non_unique_replicas
                    )
                    
                    total_time += duration
                    
                    # Read placement count from metadata file if it exists
                    placement_count_str = ""
                    metadata_file = output_dir / "placement_metadata.json"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as mf:
                                metadata = json.load(mf)
                                num_placements = metadata.get('num_placements', 0)
                                if num_placements > 0:
                                    placement_count_str = f" placements: {num_placements}"
                        except Exception:
                            pass
                    
                    if status == 'success':
                        successful += 1
                        log(f"  SUCCESS: RTT={rtt:.3f}s ({duration:.1f}s)", quiet)
                        with open(progress_log, 'a') as f:
                            f.write(f"{dataset_id} SUCCESS {datetime.now().isoformat()} "
                                   f"{duration:.1f}s RTT={rtt:.3f}s q={qname}{placement_count_str}\n")
                    elif status == 'skipped':
                        skipped += 1
                        log(f"  SKIPPED: infeasible configuration ({duration:.1f}s)", quiet)
                        with open(progress_log, 'a') as f:
                            f.write(f"{dataset_id} SKIPPED {datetime.now().isoformat()} "
                                   f"{duration:.1f}s (infeasible) q={qname}{placement_count_str}\n")
                    else:
                        failed += 1
                        log(f"  FAILED ({duration:.1f}s)", quiet)
                        with open(progress_log, 'a') as f:
                            f.write(f"{dataset_id} FAILED {datetime.now().isoformat()} "
                                   f"{duration:.1f}s{placement_count_str}\n")
                    
                    dataset_idx += 1
                    current_idx += 1
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
    log(f"Total attempted: {dataset_idx}", quiet, force=True)
    log(f"Successful: {successful}", quiet, force=True)
    log(f"Skipped (infeasible): {skipped}", quiet, force=True)
    log(f"Failed: {failed}", quiet, force=True)
    if successful > 0:
        log(f"Success rate: {100*successful/(successful+failed+skipped):.1f}%", quiet, force=True)
    log(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)", quiet, force=True)
    log(f"Average time per dataset: {total_elapsed/max(1, dataset_idx):.1f}s", quiet, force=True)
    log(f"Output directory: {output_base}", quiet, force=True)
    log(f"Progress log: {progress_log}", quiet, force=True)


if __name__ == "__main__":
    main()
