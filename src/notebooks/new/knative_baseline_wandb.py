# %%
#!/usr/bin/env python3
"""
Knative Baseline Evaluation - WandB Logging

This script evaluates Knative's placement decisions against brute-force optimal
and logs metrics to WandB for direct comparison with the GNN model.

Key differences from GNN training:
- No learning/training - just evaluation of existing placements
- Uses system_state_captured_unique.json (Knative's decisions)
- Computes the same metrics as GNN evaluation for fair comparison:
  - Regret (absolute and percentage)
  - Accuracy (how often Knative finds optimal)
  - Rank distribution

Metrics that make sense to compare:
1. regret: avg(knative_rtt - optimal_rtt) in seconds
2. regret_pct: avg((knative_rtt - optimal_rtt) / optimal_rtt * 100)
3. accuracy: % of times Knative finds the exact optimal placement
4. rank: average rank of Knative's placement (1 = optimal)
5. percentile_rank: rank / total_placements * 100 (lower is better)
6. placement_found_rate: % of Knative placements found in BF search space
"""

import os
import json
import pickle
import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import statistics
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from tqdm import tqdm
import wandb

# %%
# Configuration
BASE_DIR = Path("/root/projects/my-herosim/simulation_data/artifacts/run2000/gnn_datasets")
CACHE_DIR = Path("/root/projects/my-herosim/simulation_data/artifacts/run2000/graphs_cache_with_queues")

# %%
# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_json(path: Path) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[dict]:
    results = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def build_node_id_to_name_map(space_config: dict) -> Dict[int, str]:
    """Build mapping from node ID to node name based on space config."""
    node_map = {}
    client_count = space_config['nodes']['client_nodes']['count']
    server_count = space_config['nodes']['server_nodes']['count']
    
    for i in range(client_count):
        node_map[i] = f"client_node{i}"
    for i in range(server_count):
        node_map[client_count + i] = f"node{i}"
    
    return node_map


def build_node_name_to_id_map(space_config: dict) -> Dict[str, int]:
    """Build mapping from node name to node ID."""
    node_map = {}
    client_count = space_config['nodes']['client_nodes']['count']
    server_count = space_config['nodes']['server_nodes']['count']
    
    for i in range(client_count):
        node_map[f"client_node{i}"] = i
    for i in range(server_count):
        node_map[f"node{i}"] = client_count + i
    
    return node_map


def extract_knative_placement(captured_state: dict) -> Dict[int, Tuple[str, int]]:
    """Extract placement as {task_id: (node_name, platform_id)}"""
    placement = {}
    for tp in captured_state['task_placements']:
        task_id = tp['task_id']
        node_name = tp['execution_node']
        platform_id = int(tp['execution_platform'])
        placement[task_id] = (node_name, platform_id)
    return placement


def convert_bruteforce_placement(bf_placement: dict, node_map: Dict[int, str]) -> Dict[int, Tuple[str, int]]:
    """Convert brute-force placement to {task_id: (node_name, platform_id)}"""
    result = {}
    for task_id_str, (node_id, platform_id) in bf_placement.items():
        task_id = int(task_id_str)
        node_name = node_map.get(node_id, f"unknown_node_{node_id}")
        result[task_id] = (node_name, platform_id)
    return result


def placement_to_combo_tuple(
    placement: Dict[int, Tuple[str, int]], 
    node_name_to_id: Dict[str, int]
) -> Tuple[Tuple[int, int], ...]:
    """Convert placement dict to sorted combo tuple for hash table lookup."""
    combo_list = []
    for task_id in sorted(placement.keys()):
        node_name, plat_id = placement[task_id]
        node_id = node_name_to_id.get(node_name, -1)
        combo_list.append((node_id, plat_id))
    return tuple(combo_list)


# %%
# ============================================================================
# LOAD RTT HASH TABLE (same as GNN training)
# ============================================================================

def load_rtt_hash_table_from_cache() -> Dict[Tuple[str, Tuple[Tuple[int, int], ...]], float]:
    """Load placement RTT hash table from cache (supports chunked format)."""
    meta_path = CACHE_DIR / "rtt_chunks_meta.json"
    
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        num_chunks = meta['num_chunks']
        total_entries = meta['total_entries']
        
        print(f"Loading RTT hash table from {num_chunks} chunks ({total_entries:,} entries)...")
        
        placement_rtt_hash_table: Dict[Tuple[str, Tuple[Tuple[int, int], ...]], float] = {}
        
        for i in tqdm(range(num_chunks), desc="Loading RTT chunks"):
            chunk_path = CACHE_DIR / f"rtt_chunk_{i}.pkl"
            with open(chunk_path, 'rb') as f:
                chunk = pickle.load(f)
            placement_rtt_hash_table.update(chunk)
        
        print(f"Loaded {len(placement_rtt_hash_table):,} placement RTT entries")
        return placement_rtt_hash_table
    
    # Fall back to single file
    rtt_path = CACHE_DIR / "placement_rtt_hash_table.pkl"
    if not rtt_path.exists():
        raise FileNotFoundError(f"RTT hash table not found at {rtt_path}")
    
    print(f"Loading RTT hash table from: {rtt_path}")
    with open(rtt_path, 'rb') as f:
        placement_rtt_hash_table = pickle.load(f)
    
    print(f"Loaded {len(placement_rtt_hash_table):,} placement RTT entries")
    return placement_rtt_hash_table


def load_optimal_rtt_from_cache() -> Dict[str, float]:
    """Load optimal RTT mapping from cache."""
    optimal_path = CACHE_DIR / "optimal_rtt.pkl"
    if not optimal_path.exists():
        raise FileNotFoundError(f"Optimal RTT cache not found at {optimal_path}")
    
    with open(optimal_path, 'rb') as f:
        optimal_rtt_map = pickle.load(f)
    
    print(f"Loaded optimal RTT for {len(optimal_rtt_map)} datasets")
    return optimal_rtt_map


# %%
# ============================================================================
# EVALUATE SINGLE DATASET
# ============================================================================

def evaluate_knative_dataset(
    dataset_dir: Path,
    placement_rtt_hash_table: Dict,
    optimal_rtt_map: Dict[str, float]
) -> Optional[Dict[str, Any]]:
    """
    Evaluate Knative's placement for a single dataset.
    
    Returns metrics dict or None if dataset can't be evaluated.
    """
    dataset_id = dataset_dir.name
    
    # Required files
    captured_file = dataset_dir / "system_state_captured_unique.json"
    space_file = dataset_dir / "space_with_network.json"
    best_file = dataset_dir / "best.json"  # In root, not placements/
    placements_file = dataset_dir / "placements" / "placements.jsonl"
    
    # Check files exist
    if not captured_file.exists():
        return None
    if not space_file.exists():
        return None
    if not best_file.exists():
        return None
    if not placements_file.exists():
        return None
    
    # Load data
    captured_state = load_json(captured_file)
    space_config = load_json(space_file)
    best_data = load_json(best_file)
    
    # Build mappings
    node_id_to_name = build_node_id_to_name_map(space_config)
    node_name_to_id = build_node_name_to_id_map(space_config)
    
    # Extract Knative placement
    knative_placement = extract_knative_placement(captured_state)
    
    # Get optimal RTT from best.json
    optimal_rtt = best_data.get('rtt')
    if optimal_rtt is None:
        return None
    
    # Convert Knative placement to combo tuple for hash lookup
    knative_combo = placement_to_combo_tuple(knative_placement, node_name_to_id)
    
    # Look up Knative's RTT in hash table
    hash_key = (dataset_id, knative_combo)
    knative_rtt = placement_rtt_hash_table.get(hash_key)
    
    placement_found = knative_rtt is not None
    
    if not placement_found:
        # Placement not in BF search space - can't compute regret
        return {
            'dataset_id': dataset_id,
            'placement_found': False,
            'optimal_rtt': optimal_rtt,
            'knative_rtt': None,
            'regret': None,
            'regret_pct': None,
            'rank': None,
            'total_placements': None,
            'matches_optimal': False,
        }
    
    # Compute regret
    regret = knative_rtt - optimal_rtt
    regret_pct = (regret / optimal_rtt * 100) if optimal_rtt > 0 else 0.0
    
    # Check if Knative matches optimal placement by finding the optimal in placements.jsonl
    all_placements = load_jsonl(placements_file)
    optimal_placement = None
    for p in all_placements:
        if abs(p['rtt'] - optimal_rtt) < 0.0001:  # Match optimal RTT
            optimal_placement = p.get('placement_plan')
            break
    
    matches_optimal = False
    if optimal_placement:
        bf_optimal_placement = convert_bruteforce_placement(optimal_placement, node_id_to_name)
        bf_optimal_combo = placement_to_combo_tuple(bf_optimal_placement, node_name_to_id)
        matches_optimal = (knative_combo == bf_optimal_combo)
    
    # Compute rank by counting how many placements are better
    # Load all placements for this dataset
    all_placements = load_jsonl(placements_file)
    
    # Count placements with lower RTT (strictly better)
    better_count = 0
    for p in all_placements:
        if p['rtt'] < knative_rtt:
            better_count += 1
    
    rank = better_count + 1  # 1 = optimal
    total_placements = len(all_placements)
    
    return {
        'dataset_id': dataset_id,
        'placement_found': True,
        'optimal_rtt': optimal_rtt,
        'knative_rtt': knative_rtt,
        'regret': regret,
        'regret_pct': regret_pct,
        'rank': rank,
        'total_placements': total_placements,
        'matches_optimal': matches_optimal,
    }


# %%
# ============================================================================
# MAIN EVALUATION
# ============================================================================

def compute_epoch_metrics(results: List[Dict]) -> Dict[str, float]:
    """Compute metrics for a batch of results (simulating an epoch)."""
    if not results:
        return {}
    
    found_results = [r for r in results if r['placement_found']]
    
    if not found_results:
        return {
            'val/acc': 0.0,
            'val/regret': 0.0,
            'val/regret_pct': 0.0,
            'val/count_regret': 0,
            'val/hash_hit_rate': 0.0,
        }
    
    # Compute metrics (only for found placements)
    regrets = [r['regret'] for r in found_results]
    regret_pcts = [r['regret_pct'] for r in found_results]
    optimal_matches = [r['matches_optimal'] for r in found_results]
    
    # Accuracy = % that match optimal
    accuracy = sum(optimal_matches) / len(optimal_matches) if optimal_matches else 0.0
    
    # Average regret
    avg_regret = statistics.mean(regrets) if regrets else 0.0
    avg_regret_pct = statistics.mean(regret_pcts) if regret_pcts else 0.0
    
    # Hash hit rate = % of placements found in BF search space
    hash_hit_rate = len(found_results) / len(results) if results else 0.0
    
    return {
        'val/acc': accuracy,
        'val/regret': avg_regret,
        'val/regret_pct': avg_regret_pct,
        'val/count_regret': len(found_results),  # Number of valid regret calculations
        'val/hash_hit_rate': hash_hit_rate,
    }


def main():
    print("=" * 80)
    print("KNATIVE BASELINE EVALUATION (EPOCH-BASED)")
    print("=" * 80)
    print()
    
    # Find all datasets
    ds_dirs = sorted([d for d in BASE_DIR.iterdir() if d.is_dir() and d.name.startswith('ds_')])
    num_datasets = len(ds_dirs)
    
    # Configuration: calculate datasets per epoch for exactly 300 epochs
    NUM_EPOCHS = 300
    DATASETS_PER_EPOCH = math.ceil(num_datasets / NUM_EPOCHS)
    
    print(f"Found {num_datasets} datasets in {BASE_DIR}")
    print(f"Will simulate {NUM_EPOCHS} epochs with ~{DATASETS_PER_EPOCH} datasets per epoch")
    print()
    
    # Load RTT hash table
    placement_rtt_hash_table = load_rtt_hash_table_from_cache()
    optimal_rtt_map = load_optimal_rtt_from_cache()
    
    # ========================================================================
    # WANDB INITIALIZATION
    # ========================================================================
    os.environ['WANDB_API_KEY'] = '85cccc04212d62b698dbc4549b87818a95850133'
    
    wandb.init(
        project="scheduling-gnn-regret-training",  # Same project as GNN
        entity="nikolalukic167-tu-wien",
        name="knative-baseline",
        config={
            "scheduler": "knative",
            "model_type": "heuristic",
            "description": "Knative least-connections scheduling baseline (epoch-simulated)",
            "datasets_per_epoch": DATASETS_PER_EPOCH,
            "base_dir": str(BASE_DIR),
        }
    )
    
    # ========================================================================
    # EVALUATE IN EPOCHS (simulating training)
    # ========================================================================
    all_results = []
    skipped_no_captured = 0
    skipped_no_space = 0
    skipped_no_best = 0
    skipped_no_placements = 0
    
    print(f"Processing {num_datasets} datasets in {NUM_EPOCHS} epochs...")
    print()
    
    for epoch in range(NUM_EPOCHS):
        start_idx = epoch * DATASETS_PER_EPOCH
        end_idx = min(start_idx + DATASETS_PER_EPOCH, num_datasets)
        epoch_ds_dirs = ds_dirs[start_idx:end_idx]
        
        epoch_results = []
        
        for ds_dir in tqdm(epoch_ds_dirs, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            result = evaluate_knative_dataset(ds_dir, placement_rtt_hash_table, optimal_rtt_map)
            if result:
                epoch_results.append(result)
                all_results.append(result)
            else:
                # Debug: check which file is missing
                if not (ds_dir / "system_state_captured_unique.json").exists():
                    skipped_no_captured += 1
                elif not (ds_dir / "space_with_network.json").exists():
                    skipped_no_space += 1
                elif not (ds_dir / "best.json").exists():
                    skipped_no_best += 1
                elif not (ds_dir / "placements" / "placements.jsonl").exists():
                    skipped_no_placements += 1
        
        # Compute and log metrics for this epoch (matching GNN training format)
        epoch_metrics = compute_epoch_metrics(epoch_results)
        
        if epoch_metrics:
            log_dict = {
                "epoch": epoch + 1,
                **epoch_metrics
            }
            wandb.log(log_dict, step=epoch + 1)
            
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}: "
                  f"acc={epoch_metrics['val/acc']:.4f}, "
                  f"regret={epoch_metrics['val/regret']:.6f}, "
                  f"regret_pct={epoch_metrics['val/regret_pct']:.2f}%, "
                  f"hash_hit={epoch_metrics['val/hash_hit_rate']:.2%}")
    
    print()
    print(f"Evaluated {len(all_results)} datasets total")
    print(f"  Skipped (no captured): {skipped_no_captured}")
    print(f"  Skipped (no space): {skipped_no_space}")
    print(f"  Skipped (no best): {skipped_no_best}")
    print(f"  Skipped (no placements): {skipped_no_placements}")
    
    if not all_results:
        print("ERROR: No datasets could be evaluated!")
        wandb.finish()
        return
    
    # Compute aggregate metrics for final summary
    found_results = [r for r in all_results if r['placement_found']]
    not_found_results = [r for r in all_results if not r['placement_found']]
    
    print(f"  Placements found in BF: {len(found_results)} ({len(found_results)/len(all_results)*100:.1f}%)")
    print(f"  Placements NOT found: {len(not_found_results)}")
    
    if not found_results:
        print("ERROR: No valid results to compute metrics!")
        wandb.finish()
        return
    
    # Compute final metrics (only for found placements)
    regrets = [r['regret'] for r in found_results]
    regret_pcts = [r['regret_pct'] for r in found_results]
    ranks = [r['rank'] for r in found_results]
    percentile_ranks = [r['rank'] / r['total_placements'] * 100 for r in found_results]
    optimal_matches = [r['matches_optimal'] for r in found_results]
    
    # Accuracy = % that match optimal
    accuracy = sum(optimal_matches) / len(optimal_matches)
    
    # Stats
    avg_regret = statistics.mean(regrets)
    median_regret = statistics.median(regrets)
    std_regret = statistics.stdev(regrets) if len(regrets) > 1 else 0.0
    min_regret = min(regrets)
    max_regret = max(regrets)
    
    avg_regret_pct = statistics.mean(regret_pcts)
    median_regret_pct = statistics.median(regret_pcts)
    
    avg_rank = statistics.mean(ranks)
    median_rank = statistics.median(ranks)
    
    avg_percentile_rank = statistics.mean(percentile_ranks)
    median_percentile_rank = statistics.median(percentile_ranks)
    
    # Print summary
    print()
    print("=" * 80)
    print("KNATIVE BASELINE METRICS (FINAL)")
    print("=" * 80)
    print(f"Accuracy (exact optimal match): {accuracy*100:.2f}%")
    print()
    print(f"Regret (seconds):")
    print(f"  Mean:   {avg_regret:.4f}s")
    print(f"  Median: {median_regret:.4f}s")
    print(f"  Std:    {std_regret:.4f}s")
    print(f"  Min:    {min_regret:.4f}s")
    print(f"  Max:    {max_regret:.4f}s")
    print()
    print(f"Regret (percentage):")
    print(f"  Mean:   {avg_regret_pct:.2f}%")
    print(f"  Median: {median_regret_pct:.2f}%")
    print()
    print(f"Rank (1 = optimal):")
    print(f"  Mean:   {avg_rank:.1f}")
    print(f"  Median: {median_rank:.1f}")
    print()
    print(f"Percentile rank (lower is better):")
    print(f"  Mean:   {avg_percentile_rank:.2f}%")
    print(f"  Median: {median_percentile_rank:.2f}%")
    
    # ========================================================================
    # LOG FINAL METRICS (matching GNN's final/test format)
    # ========================================================================
    wandb.log({
        "final/test/acc": accuracy,
        "final/test/regret": avg_regret,
        "final/test/regret_pct": avg_regret_pct,
        "final/test/hash_hit_rate": len(found_results) / len(all_results),
    })
    
    # Log data stats
    wandb.log({
        "data/num_datasets_total": len(all_results),
        "data/num_valid": len(found_results),
        "data/num_not_found": len(not_found_results),
    })
    
    # Log per-dataset results as table
    table = wandb.Table(columns=[
        "dataset_id", "placement_found", "optimal_rtt", "knative_rtt", 
        "regret", "regret_pct", "rank", "total_placements", "matches_optimal"
    ])
    
    for r in all_results:
        table.add_data(
            r['dataset_id'],
            r['placement_found'],
            r['optimal_rtt'],
            r.get('knative_rtt'),
            r.get('regret'),
            r.get('regret_pct'),
            r.get('rank'),
            r.get('total_placements'),
            r.get('matches_optimal'),
        )
    
    wandb.log({"results_table": table})
    
    # Log histograms
    if regrets:
        wandb.log({
            "histograms/regret": wandb.Histogram(regrets),
            "histograms/regret_pct": wandb.Histogram(regret_pcts),
            "histograms/rank": wandb.Histogram(ranks),
            "histograms/percentile_rank": wandb.Histogram(percentile_ranks),
        })
    
    # Summary
    wandb.summary["scheduler"] = "knative"
    wandb.summary["accuracy"] = accuracy
    wandb.summary["avg_regret"] = avg_regret
    wandb.summary["avg_regret_pct"] = avg_regret_pct
    wandb.summary["avg_rank"] = avg_rank
    wandb.summary["avg_percentile_rank"] = avg_percentile_rank
    wandb.summary["placement_found_rate"] = len(found_results) / len(all_results)
    
    wandb.finish()
    
    print()
    print("=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"Results logged to WandB project: scheduling-gnn-regret-training")
    print(f"Run name: knative-baseline")
    print(f"Logged {NUM_EPOCHS} epochs for real-time comparison with GNN training")


if __name__ == "__main__":
    main()

