#!/usr/bin/env python3
"""
Generate Non-Unique Placements for Existing GNN Datasets

This script generates the missing non-unique placements for datasets that were
originally generated with unique replicas only.

For each existing dataset:
1. Load existing infrastructure, workload, config, system state
2. Generate all cartesian (non-unique) placement combinations  
3. Load existing unique placements from placements.jsonl
4. Filter out already-generated placements
5. Run simulations only for new (non-unique) placements
6. Merge results with existing placements.jsonl

Usage:
    python scripts_cosim/generate_non_unique_placements_fast.py [--max-datasets N] [--workers N]

Example:
    # Process first 100 datasets
    python scripts_cosim/generate_non_unique_placements_fast.py --max-datasets 100 --workers 8
"""

import argparse
import concurrent.futures
import json
import logging
import multiprocessing
import os
import pickle
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set

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

from src.executecosimulation import (
    load_simulation_inputs,
    prepare_workloads,
    flatten_workloads,
    prepare_simulation_config,
    determine_replica_placement,
    generate_brute_force_placement_combinations,
    _init_worker,
    process_placement_fast,
    DataclassJSONEncoder,
)

# Global quiet mode flag
QUIET_MODE = False


def log(msg: str, quiet: bool = False, force: bool = False):
    """Print message unless in quiet mode."""
    if not quiet or force:
        print(msg)


def load_existing_placements(dataset_dir: Path) -> Set[Tuple[Tuple[int, int], ...]]:
    """
    Load existing placements from placements.jsonl and return as a set of tuples.
    
    Each placement is converted to a tuple of (node_id, platform_id) tuples,
    sorted by task_id, for easy comparison.
    
    Returns:
        Set of placements where each placement is a tuple of ((node_id, platform_id), ...)
    """
    placements_file = dataset_dir / "placements" / "placements.jsonl"
    existing = set()
    
    if not placements_file.exists():
        return existing
    
    with open(placements_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                p = json.loads(line)
                plan = p.get('placement_plan', {})
                # Convert to sorted tuple of (task_id, (node_id, platform_id))
                # Then extract just the placements sorted by task_id
                sorted_placements = tuple(
                    tuple(plan[str(i)]) for i in range(len(plan))
                )
                existing.add(sorted_placements)
            except Exception:
                continue
    
    return existing


def placement_to_key(plan: Dict[int, Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    """Convert a placement plan to a hashable key for comparison."""
    # Sort by task_id and create tuple of (node_id, platform_id) tuples
    return tuple(plan[i] for i in range(len(plan)))


def filter_new_placements(
    all_placements: List[Dict[int, Tuple[int, int]]],
    existing_keys: Set[Tuple[Tuple[int, int], ...]]
) -> List[Dict[int, Tuple[int, int]]]:
    """
    Filter out placements that already exist.
    
    Returns:
        List of new placements (not in existing_keys)
    """
    new_placements = []
    for plan in all_placements:
        key = placement_to_key(plan)
        if key not in existing_keys:
            new_placements.append(plan)
    return new_placements


def run_simulations_for_placements(
    placement_combinations: List[Dict[int, Tuple[int, int]]],
    sim_inputs: Dict[str, Any],
    infra_config: Dict[str, Any],
    base_nodes: List[Dict],
    flattened_workloads: Dict[str, Any],
    replica_plan: Dict[str, Any],
    apps: List[str],
    infrastructure_file: Path,
    sample: np.ndarray,
    mapping: Dict[int, str],
    output_dir: Path,
    max_workers: int,
    quiet: bool = False,
    progress_dir: Optional[Path] = None
) -> Tuple[List[Dict[str, Any]], float, Optional[str]]:
    """
    Run simulations for a list of placement combinations.
    
    Returns:
        (list of placement results, best_rtt, best_file)
    """
    num_placements = len(placement_combinations)
    if num_placements == 0:
        return [], float('inf'), None
    
    time_started = time.time()
    
    # Create shared state for best RTT tracking
    manager = multiprocessing.Manager()
    best_rtt_value = manager.Value('d', float('inf'))
    best_rtt_lock = manager.Lock()
    
    results = []
    best_rtt = float('inf')
    best_file = None
    
    # Open placements file for streaming writes (will be merged later)
    temp_placements_file = output_dir / "temp_placements.jsonl"
    placements_fh = open(temp_placements_file, 'w')
    num_written = 0
    
    try:
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
            futures = {
                executor.submit(process_placement_fast, plan): idx 
                for idx, plan in enumerate(placement_combinations)
            }
            
            completed = 0
            timeout_per_placement = 2
            timed_out_count = 0
            update_interval = max(1, min(1000, num_placements // 100))
            
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                placement_idx = futures[future]
                
                try:
                    result_file, cur_rtt, placement_plan = future.result(timeout=timeout_per_placement)
                    
                    if placement_plan is None:
                        continue
                    
                    # Track best
                    if result_file is not None and cur_rtt < best_rtt:
                        best_rtt = cur_rtt
                        best_file = str(result_file)
                    
                    # Stream write to temp file
                    summary = {"placement_plan": placement_plan, "rtt": cur_rtt}
                    placements_fh.write(json.dumps(summary, separators=(',', ':')) + '\n')
                    results.append(summary)
                    num_written += 1
                    
                    if num_written % 1000 == 0:
                        placements_fh.flush()
                        
                except concurrent.futures.TimeoutError:
                    timed_out_count += 1
                    future.cancel()
                except Exception as e:
                    if not quiet:
                        log(f"  Worker failed for placement {placement_idx}: {e}")
                
                # Progress update
                if completed % update_interval == 0 and progress_dir:
                    try:
                        progress_file = progress_dir / "non_unique_progress.txt"
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
                        pass
        
        # Final progress
        elapsed = time.time() - time_started
        if progress_dir:
            try:
                progress_file = progress_dir / "non_unique_progress.txt"
                with open(progress_file, 'w') as pf:
                    pf.write(f"{completed}/{num_placements}\n")
                    pf.write(f"Rate: {completed/elapsed:.1f} sim/s\n")
                    if best_rtt < float('inf'):
                        pf.write(f"Best RTT: {best_rtt:.3f}s\n")
                    pf.write(f"Status: COMPLETE\n")
            except Exception:
                pass
                
    finally:
        placements_fh.close()
    
    return results, best_rtt, best_file


def merge_placements(
    dataset_dir: Path,
    new_results: List[Dict[str, Any]],
    new_best_rtt: float,
    output_dir: Path
) -> Tuple[float, int]:
    """
    Merge new placements with existing placements.jsonl.
    Updates best.json if new best RTT is found.
    
    Returns:
        (overall_best_rtt, total_placements)
    """
    placements_dir = dataset_dir / "placements"
    placements_file = placements_dir / "placements.jsonl"
    best_json_file = dataset_dir / "best.json"
    
    # Load existing best RTT
    existing_best_rtt = float('inf')
    if best_json_file.exists():
        with open(best_json_file, 'r') as f:
            best_info = json.load(f)
            existing_best_rtt = best_info.get('rtt', float('inf'))
    
    # Count existing placements
    existing_count = 0
    if placements_file.exists():
        with open(placements_file, 'r') as f:
            for line in f:
                if line.strip():
                    existing_count += 1
    
    # Append new placements
    with open(placements_file, 'a') as f:
        for result in new_results:
            f.write(json.dumps(result, separators=(',', ':')) + '\n')
    
    total_placements = existing_count + len(new_results)
    
    # Update best.json if new best found
    overall_best_rtt = min(existing_best_rtt, new_best_rtt)
    if new_best_rtt < existing_best_rtt:
        # New best found - update best.json
        # Note: We don't have the full result file here, just the RTT
        best_info = {
            "rtt": new_best_rtt,
            "file": "non_unique_optimal.json",
            "updated_by": "generate_non_unique_placements_fast.py"
        }
        with open(best_json_file, 'w') as f:
            json.dump(best_info, f)
    
    # Update metadata
    metadata_file = dataset_dir / "placement_metadata.json"
    metadata = {
        "num_placements": total_placements,
        "completed": total_placements,
        "unique_placements": existing_count,
        "non_unique_placements": len(new_results)
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)
    
    return overall_best_rtt, total_placements


def process_existing_dataset(
    dataset_dir: Path,
    sim_input_path: Path,
    samples_file: Path,
    mapping_file: Path,
    max_workers: int,
    quiet: bool = False
) -> Tuple[str, int, int, float, float]:
    """
    Process an existing dataset to add non-unique placements.
    
    Returns:
        (status, num_existing, num_new, best_rtt, duration)
    """
    start_time = time.time()
    
    try:
        # Verify required files exist
        config_path = dataset_dir / "space_with_network.json"
        infra_file = dataset_dir / "infrastructure.json"
        workload_file = dataset_dir / "workload.json"
        placements_dir = dataset_dir / "placements"
        placements_file = placements_dir / "placements.jsonl"
        
        for f in [config_path, infra_file, workload_file]:
            if not f.exists():
                return 'missing_files', 0, 0, float('inf'), time.time() - start_time
        
        if not placements_file.exists():
            return 'no_placements', 0, 0, float('inf'), time.time() - start_time
        
        # Load existing unique placements
        log(f"  Loading existing placements...", quiet)
        existing_keys = load_existing_placements(dataset_dir)
        num_existing = len(existing_keys)
        log(f"  Found {num_existing} existing unique placements", quiet)
        
        if num_existing == 0:
            return 'no_existing', 0, 0, float('inf'), time.time() - start_time
        
        # Load config and infrastructure
        with open(config_path, 'r') as f:
            infra_config = json.load(f)
        
        with open(workload_file, 'r') as f:
            workload = json.load(f)
        
        # Load simulation inputs
        sim_inputs = load_simulation_inputs(sim_input_path)
        
        # Load mapping
        with open(mapping_file, 'rb') as f:
            mapping = pickle.load(f)
        
        # Load sample
        samples = np.load(samples_file)
        sample = samples[0]
        
        # Get apps from config
        apps = list(infra_config.get('wsc', {}).keys())
        if not apps:
            apps = ['nofs-dnn1', 'nofs-dnn2']
        
        # Prepare workloads
        workloads = prepare_workloads(sample, mapping, workload, apps)
        flattened_workloads = flatten_workloads(workloads)
        
        # Prepare simulation config with infrastructure
        sim_config = prepare_simulation_config(sample, mapping, infra_config, infrastructure_file=infra_file)
        
        # Generate replica plan
        replica_plan = determine_replica_placement(sim_config, sim_inputs)
        
        base_nodes = sim_config['nodes']
        
        # Generate ALL cartesian combinations (non-unique)
        log(f"  Generating all cartesian combinations...", quiet)
        all_placements = generate_brute_force_placement_combinations(
            flattened_workloads['events'],
            sim_config,
            sim_inputs,
            replica_plan,
            active_replicas=None,  # Will use deterministic placements
            use_all_replicas=True,  # Use all from infrastructure
            allow_non_unique_replicas=True  # IMPORTANT: This gives us cartesian product
        )
        
        total_cartesian = len(all_placements)
        log(f"  Generated {total_cartesian} total cartesian combinations", quiet)
        
        # Filter out already-generated placements
        new_placements = filter_new_placements(all_placements, existing_keys)
        num_new = len(new_placements)
        log(f"  Found {num_new} new (non-unique) placements to generate", quiet)
        
        if num_new == 0:
            return 'already_complete', num_existing, 0, float('inf'), time.time() - start_time
        
        # Create temp output directory
        temp_output = Path("simulation_data/temp_non_unique_results")
        temp_output.mkdir(parents=True, exist_ok=True)
        
        # Run simulations for new placements
        log(f"  Running {num_new} simulations...", quiet)
        new_results, best_rtt, best_file = run_simulations_for_placements(
            new_placements,
            sim_inputs,
            infra_config,
            base_nodes,
            flattened_workloads,
            replica_plan,
            apps,
            infra_file,
            sample,
            mapping,
            temp_output,
            max_workers,
            quiet=quiet,
            progress_dir=dataset_dir
        )
        
        # Merge results
        log(f"  Merging results...", quiet)
        overall_best_rtt, total_placements = merge_placements(
            dataset_dir, new_results, best_rtt, temp_output
        )
        
        # Cleanup temp directory
        try:
            temp_placements = temp_output / "temp_placements.jsonl"
            if temp_placements.exists():
                temp_placements.unlink()
            for f in temp_output.glob("simulation_*.json"):
                f.unlink()
        except Exception:
            pass
        
        duration = time.time() - start_time
        return 'success', num_existing, len(new_results), overall_best_rtt, duration
        
    except Exception as e:
        duration = time.time() - start_time
        log(f"  ERROR: {e}", quiet, force=True)
        import traceback
        traceback.print_exc()
        return 'failed', 0, 0, float('inf'), duration


def main():
    parser = argparse.ArgumentParser(
        description="Generate Non-Unique Placements for Existing GNN Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--quiet', '-q', default=True, action='store_true',
                        help='Suppress per-placement logging (default: True)')
    parser.add_argument('--no-quiet', action='store_false', dest='quiet',
                        help='Disable quiet mode')
    parser.add_argument('--max-datasets', '-n', type=int, default=300000,
                        help='Maximum number of datasets to process')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--start-from', type=int, default=0,
                        help='Start from dataset index (e.g., 0 to start from ds_00000)')
    parser.add_argument('--skip-range-start', type=int, default=300000,
                        help='Start of dataset range to skip (default: 300000)')
    parser.add_argument('--skip-range-end', type=int, default=600000,
                        help='End of dataset range to skip (default: 600000)')
    parser.add_argument('--num-tasks', type=int, choices=[2, 3, 4], default=2,
                        help='Number of tasks per workload (determines dataset folder)')
    args = parser.parse_args()
    
    quiet = args.quiet
    max_datasets = args.max_datasets
    cpu_count = os.cpu_count()
    max_workers = args.workers or (cpu_count - 1 if cpu_count and cpu_count > 1 else 1)
    
    # Paths
    base_dir = PROJECT_ROOT / "simulation_data"
    datasets_dir = base_dir / f"gnn_datasets_{args.num_tasks}tasks"
    sim_input_path = PROJECT_ROOT / "data" / "nofs-ids"
    samples_file = base_dir / "lhs_samples_simple.npy"
    mapping_file = base_dir / "lhs_samples_simple_mapping.pkl"
    progress_log = PROJECT_ROOT / "logs" / f"non_unique_progress_{args.num_tasks}tasks.txt"
    
    # Create logs directory
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    
    log(f"=== Generate Non-Unique Placements ===", quiet)
    log(f"Datasets directory: {datasets_dir}", quiet)
    log(f"Max datasets: {max_datasets}", quiet)
    log(f"Workers: {max_workers}", quiet)
    log(f"Skip range: ds_{args.skip_range_start:05d} to ds_{args.skip_range_end:05d}", quiet)
    log(f"Using orjson: {HAS_ORJSON}", quiet)
    
    # Check if datasets directory exists
    if not datasets_dir.exists():
        log(f"ERROR: Datasets directory does not exist: {datasets_dir}", force=True)
        return
    
    # Get list of existing datasets
    dataset_dirs = sorted([
        d for d in datasets_dir.iterdir() 
        if d.is_dir() and d.name.startswith('ds_')
    ])
    
    log(f"Found {len(dataset_dirs)} dataset directories", quiet)
    
    # Process datasets
    start_time = time.time()
    processed = 0
    successful = 0
    skipped = 0
    failed = 0
    total_new_placements = 0
    
    for dataset_dir in dataset_dirs:
        if processed >= max_datasets:
            break
        
        # Extract dataset index
        try:
            ds_idx = int(dataset_dir.name.split('_')[1])
        except (ValueError, IndexError):
            continue
        
        # Skip if before start index
        if ds_idx < args.start_from:
            continue
        
        # Skip datasets in the skip range (missing datasets)
        if args.skip_range_start <= ds_idx <= args.skip_range_end:
            log(f"[{dataset_dir.name}] Skipping (in skip range)", quiet)
            skipped += 1
            processed += 1
            continue
        
        log(f"\n[{dataset_dir.name}] Processing...", quiet)
        
        status, num_existing, num_new, best_rtt, duration = process_existing_dataset(
            dataset_dir,
            sim_input_path,
            samples_file,
            mapping_file,
            max_workers,
            quiet=quiet
        )
        
        processed += 1
        
        if status == 'success':
            successful += 1
            total_new_placements += num_new
            log(f"  SUCCESS: {num_existing} existing + {num_new} new = {num_existing + num_new} total ({duration:.1f}s)", quiet)
            with open(progress_log, 'a') as f:
                f.write(f"{dataset_dir.name} SUCCESS {datetime.now().isoformat()} "
                       f"{duration:.1f}s existing={num_existing} new={num_new} best_rtt={best_rtt:.3f}s\n")
        elif status == 'already_complete':
            log(f"  SKIP: All placements already generated ({num_existing} total)", quiet)
            skipped += 1
        elif status in ['missing_files', 'no_placements', 'no_existing']:
            log(f"  SKIP: {status} ({duration:.1f}s)", quiet)
            skipped += 1
            with open(progress_log, 'a') as f:
                f.write(f"{dataset_dir.name} SKIPPED {datetime.now().isoformat()} {status}\n")
        else:
            failed += 1
            log(f"  FAILED ({duration:.1f}s)", quiet, force=True)
            with open(progress_log, 'a') as f:
                f.write(f"{dataset_dir.name} FAILED {datetime.now().isoformat()} {duration:.1f}s\n")
        
        # Progress update every 10 datasets
        if processed % 10 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            log(f"\n--- Progress: {processed} datasets, {successful} successful, "
                f"{total_new_placements} new placements ({rate:.2f} datasets/min) ---", quiet)
    
    # Summary
    total_elapsed = time.time() - start_time
    
    log(f"\n=== Generation Complete ===", quiet, force=True)
    log(f"Total processed: {processed}", quiet, force=True)
    log(f"Successful: {successful}", quiet, force=True)
    log(f"Skipped: {skipped}", quiet, force=True)
    log(f"Failed: {failed}", quiet, force=True)
    log(f"Total new placements generated: {total_new_placements}", quiet, force=True)
    log(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)", quiet, force=True)
    log(f"Average time per dataset: {total_elapsed/max(1, successful):.1f}s", quiet, force=True)
    log(f"Progress log: {progress_log}", quiet, force=True)


if __name__ == "__main__":
    main()
