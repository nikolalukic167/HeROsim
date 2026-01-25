#!/usr/bin/env python3
"""
Pre-generate and cache graphs for GNN training (NON-UNIQUE VERSION).

This script builds all graphs and saves them to pickle files for faster training iterations.

NON-UNIQUE PLACEMENTS:
- Supports datasets where multiple tasks can be placed on the same replica
- Creates edges between tasks and all compatible platforms (no uniqueness constraint)
- Compatible with gnn_datasets_2tasks and gnn_datasets_3tasks
- Includes system state, temporal features, queue info, and consolidation metrics
"""

import os
import json
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from tqdm import tqdm
from joblib import Parallel, delayed

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# Configuration
# ============================================================================
# MERGED DATASET SUPPORT: Load from multiple directories with different task counts
# Set MERGE_DATASETS=True to combine 2-task and 3-task datasets
# Set MERGE_DATASETS=False to use single dataset (specify in BASE_DIRS[0])

MERGE_DATASETS = True  # Set to False for single dataset

if MERGE_DATASETS:
    # Load both 2-task and 3-task datasets
    BASE_DIRS = [
        Path("/root/projects/my-herosim/simulation_data/artifacts/run_queue_big/gnn_datasets_2tasks"),
        Path("/root/projects/my-herosim/simulation_data/artifacts/run_queue_big/gnn_datasets_3tasks")
    ]
    CACHE_DIR = BASE_DIRS[0].parent / "graphs_cache_merged_2_3_tasks"
else:
    # Single dataset mode (change path as needed)
    BASE_DIRS = [
        Path("/root/projects/my-herosim/simulation_data/artifacts/run_queue_big/gnn_datasets_2tasks")
    ]
    CACHE_DIR = BASE_DIRS[0].parent / f"graphs_cache_{BASE_DIRS[0].name}"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache file paths
GRAPHS_CACHE_PATH = CACHE_DIR / "graphs.pkl"
DATASET_IDS_CACHE_PATH = CACHE_DIR / "dataset_ids.pkl"
RTT_HASH_CACHE_PATH = CACHE_DIR / "placement_rtt_hash_table.pkl"
PLAT_NODE_MAP_CACHE_PATH = CACHE_DIR / "plat_node_map.pkl"
OPTIMAL_RTT_CACHE_PATH = CACHE_DIR / "optimal_rtt.pkl"
METADATA_CACHE_PATH = CACHE_DIR / "metadata.json"

# Queue normalization constant (queue_length / QUEUE_NORM_FACTOR)
QUEUE_NORM_FACTOR = 50.0

# Version for cache invalidation (increment when graph construction logic changes)
CACHE_VERSION = "4.0"  # Non-unique placements: multiple tasks can be placed on same replica
# - Removed QoS features (qos_deviation, deadline) since co-simulation doesn't capture QoS violations as ground truth
# - Supports datasets where 2+ tasks can be placed on the same (node_id, platform_id)

# ============================================================================
# DATA LOADING (same as main script)
# ============================================================================

def extract_dataset_to_dataframes(optimal_result_path: Path) -> Dict[str, pd.DataFrame]:
    """Extract a single optimal_result.json into DataFrames."""
    with open(optimal_result_path, "r") as f:
        result = json.load(f)
    
    dataset_id = optimal_result_path.parent.name
    infra_nodes = result.get("config", {}).get("infrastructure", {}).get("nodes", [])
    stats = result.get("stats", {})
    task_results = stats.get("taskResults", [])
    placement_plan = result.get("sample", {}).get("placement_plan", {})
    
    # NODES
    nodes_data = []
    for i, node in enumerate(infra_nodes):
        node_name = node.get("node_name", f"node_{i}")
        platforms = node.get("platforms", [])
        network_map = node.get("network_map", {})
        
        nodes_data.append({
            'node_id': i,
            'node_name': node_name,
            'node_type': node.get("type", "unknown"),
            'is_client': node_name.startswith('client_node'),
            'network_map': network_map
        })
    
    df_nodes = pd.DataFrame(nodes_data)
    
    # TASKS
    placement_plan_task_ids = set()
    for k in placement_plan.keys():
        task_id = int(k)
        if task_id >= 0:
            placement_plan_task_ids.add(task_id)
    
    tasks_data = []
    task_ids_seen = []
    
    for task_result in task_results:
        task_id = task_result.get("taskId")
        
        if task_id is None or task_id < 0 or task_id not in placement_plan_task_ids:
            continue
        
        task_ids_seen.append(task_id)
        
        placement = placement_plan.get(str(task_id), [None, None])
        
        if isinstance(placement, list) and len(placement) >= 2:
            opt_node_id, opt_platform_id = placement[0], placement[1]
        else:
            opt_node_id, opt_platform_id = None, None
        
        tasks_data.append({
            'task_id': task_id,
            'task_type': task_result.get("taskType", {}).get("name", "unknown"),
            'source_node': task_result.get("sourceNode", ""),
            'optimal_node_id': opt_node_id,
            'optimal_platform_id': opt_platform_id,
            'elapsed_time': task_result.get("elapsedTime", 0)
        })
    
    tasks_data.sort(key=lambda x: x['task_id'])
    
    if len(task_ids_seen) != len(placement_plan_task_ids):
        print(f"[ERROR] Task filtering: {len(task_ids_seen)} != {len(placement_plan_task_ids)}")
    
    df_tasks = pd.DataFrame(tasks_data)
    
    # PLATFORMS
    platforms_data = []
    node_results = stats.get("nodeResults", [])
    system_state = stats.get("systemStateResults", [{}])[-1] if stats.get("systemStateResults") else {}
    replicas_by_task = system_state.get("replicas", {})
    
    for node_result in node_results:
        node_id = node_result.get("nodeId")
        node_name = infra_nodes[node_id].get("node_name") if node_id < len(infra_nodes) else f"node_{node_id}"
        
        for plat_result in node_result.get("platformResults", []):
            plat_id = plat_result.get("platformId")
            plat_type = plat_result.get("platformType", {}).get("shortName", "unknown")
            
            has_dnn1_replica = False
            has_dnn2_replica = False
            
            for task_type, replica_list in replicas_by_task.items():
                if isinstance(replica_list, list):
                    for replica in replica_list:
                        if isinstance(replica, list) and len(replica) >= 2:
                            if replica[0] == node_name and replica[1] == plat_id:
                                if task_type == "dnn1":
                                    has_dnn1_replica = True
                                elif task_type == "dnn2":
                                    has_dnn2_replica = True
            
            platforms_data.append({
                'platform_id': plat_id,
                'node_id': node_id,
                'node_name': node_name,
                'platform_type': plat_type,
                'has_dnn1_replica': has_dnn1_replica,
                'has_dnn2_replica': has_dnn2_replica
            })
    
    df_platforms = pd.DataFrame(platforms_data)
    
    # METRICS
    best_json_path = optimal_result_path.parent / "best.json"
    best_rtt = None
    if best_json_path.exists():
        with open(best_json_path, "r") as f:
            best_rtt = json.load(f).get("rtt")
    if best_rtt is None:
        best_rtt = sum(tr.get("elapsedTime", 0) for tr in task_results)
    
    df_metrics = pd.DataFrame([{'dataset_id': dataset_id, 'total_rtt': best_rtt}])
    
    return {
        'nodes': df_nodes,
        'tasks': df_tasks,
        'platforms': df_platforms,
        'metrics': df_metrics
    }


def load_extended_state_data(dataset_dir: Path) -> Dict[str, Any]:
    """
    Load extended state data from system_state_captured_unique.json and infrastructure.json.
    For run_non_unique datasets, also tries to load from infrastructure.json queue_distributions.
    Returns dict with:
    - queue_snapshot: Dict mapping "node_name:platform_id" -> queue_length
    - temporal_state: Dict mapping "node_name:platform_id" -> {current_task_remaining, cold_start_remaining, comm_remaining}
    Note: QoS data removed since co-simulation doesn't capture QoS violations as ground truth.
    """
    result = {
        'queue_snapshot': {},
        'temporal_state': {}
    }
    
    # Load queue snapshot from system_state_captured_unique.json (if available)
    ssc_path = dataset_dir / "system_state_captured_unique.json"
    if ssc_path.exists():
        try:
            with open(ssc_path, 'r') as f:
                data = json.load(f)
            
            task_placements = data.get('task_placements', [])
            if task_placements:
                # Use first task's full_queue_snapshot (same for all tasks in batch)
                full_queue_snapshot = task_placements[0].get('full_queue_snapshot', {})
                result['queue_snapshot'] = {k: int(v) for k, v in full_queue_snapshot.items()}
                
                # Extract temporal state from task placements
                # Each task has temporal_state_at_scheduling for its valid replica platforms
                # Merge across all tasks to get complete platform coverage
                merged_temporal_state = {}
                for tp in task_placements:
                    temp_state = tp.get('temporal_state_at_scheduling', {})
                    if isinstance(temp_state, dict):
                        # Merge temporal state (later tasks may overwrite earlier ones for same platform)
                        # This is fine since all tasks in batch see same snapshot, just filtered differently
                        for platform_key, state_dict in temp_state.items():
                            if isinstance(state_dict, dict):
                                # Convert values to float (they should already be floats in JSON)
                                merged_temporal_state[platform_key] = {
                                    'current_task_remaining': float(state_dict.get('current_task_remaining', 0.0)),
                                    'cold_start_remaining': float(state_dict.get('cold_start_remaining', 0.0)),
                                    'comm_remaining': float(state_dict.get('comm_remaining', 0.0))
                                }
                
                if merged_temporal_state:
                    result['temporal_state'] = merged_temporal_state
        except Exception as e:
            print(f"[WARN] Failed to load extended state from {ssc_path}: {e}")
    
    # Fallback: Load queue data from infrastructure.json (run_non_unique format)
    if not result['queue_snapshot']:
        infra_path = dataset_dir / "infrastructure.json"
        if infra_path.exists():
            try:
                with open(infra_path, 'r') as f:
                    infra_data = json.load(f)
                
                # Load queue_distributions: task_type -> { "node_name:platform_id": queue_length }
                queue_distributions = infra_data.get('queue_distributions', {})
                
                # Merge queue distributions across all task types into single queue_snapshot
                # If same platform appears in multiple task types, take the maximum queue length
                merged_queues = {}
                for task_type, queues in queue_distributions.items():
                    for key, queue_length in queues.items():
                        # key is "node_name:platform_id"
                        if key not in merged_queues:
                            merged_queues[key] = int(queue_length)
                        else:
                            # Take maximum if platform appears for multiple task types
                            merged_queues[key] = max(merged_queues[key], int(queue_length))
                
                result['queue_snapshot'] = merged_queues
            except Exception as e:
                print(f"[WARN] Failed to load queue data from {infra_path}: {e}")
    
    # Note: QoS data loading removed since co-simulation doesn't capture QoS violations as ground truth
    
    return result


def load_all_datasets(base_dirs: List[Path], require_queue_data: bool = True) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load all datasets from multiple gnn_datasets directories (supports merging).
    
    Args:
        base_dirs: List of Paths to gnn_datasets directories (can be single or multiple)
        require_queue_data: If True, skip datasets without system_state_captured_unique.json
    """
    all_datasets = {}
    skipped_no_queue = 0
    
    for base_dir in base_dirs:
        if not base_dir.exists():
            print(f"Warning: Directory {base_dir} does not exist, skipping")
            continue
        
        dataset_dirs = sorted(base_dir.glob("ds_*"))
        print(f"Loading {len(dataset_dirs)} datasets from {base_dir.name}...")
        start_time = time.perf_counter()
        
        for dataset_dir in tqdm(dataset_dirs, desc=f"Loading {base_dir.name}", unit="dataset"):
            optimal_result_path = dataset_dir / "optimal_result.json"
            if not optimal_result_path.exists():
                continue
            
            # Load extended state data (queue, temporal)
            extended_state = load_extended_state_data(dataset_dir)
            
            # Skip if queue data is required but not available
            if require_queue_data and not extended_state.get('queue_snapshot'):
                skipped_no_queue += 1
                continue
            
            try:
                dataframes = extract_dataset_to_dataframes(optimal_result_path)
                # Use unique key: base_dir_name/dataset_name to avoid collisions
                unique_key = f"{base_dir.name}/{dataset_dir.name}"
                all_datasets[unique_key] = {
                    **dataframes,
                    'dataset_dir': dataset_dir,
                    'source_dir': base_dir.name,  # Track which directory this came from
                    'num_tasks': len(dataframes['tasks']),  # Track task count
                    'queue_snapshot': extended_state.get('queue_snapshot', {}),
                    'temporal_state': extended_state.get('temporal_state', {})
                }
            except Exception as e:
                tqdm.write(f"  Error loading {dataset_dir.name}: {e}")
        
        elapsed = time.perf_counter() - start_time
        print(f"  Loaded {len([k for k in all_datasets if k.startswith(base_dir.name)])} datasets from {base_dir.name} in {elapsed:.2f}s")
    
    print(f"\nTotal datasets loaded: {len(all_datasets)}")
    if skipped_no_queue > 0:
        print(f"  Skipped {skipped_no_queue} datasets without queue data")
    
    # Print task count distribution
    task_counts = {}
    for ds_dict in all_datasets.values():
        n_tasks = ds_dict['num_tasks']
        task_counts[n_tasks] = task_counts.get(n_tasks, 0) + 1
    print(f"\nTask count distribution:")
    for n_tasks in sorted(task_counts.keys()):
        print(f"  {n_tasks} tasks: {task_counts[n_tasks]} datasets")
    
    return all_datasets


# ============================================================================
# RTT HASH TABLE BUILDING (Parallel + Chunked Saving)
# ============================================================================

def _parse_jsonl_file_to_dict(jsonl_path: Path) -> Dict[Tuple[str, Tuple[Tuple[int, int], ...]], float]:
    """Parse a single JSONL file and return dict of (dataset_id, combo) -> rtt."""
    results = {}
    try:
        # Use unique dataset_id: source_dir/ds_name to avoid collisions
        ds_name = jsonl_path.parent.parent.name
        source_dir = jsonl_path.parent.parent.parent.name
        dataset_id = f"{source_dir}/{ds_name}"
        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    placement_plan = data.get("placement_plan", {})
                    rtt_val = data.get("rtt")
                    
                    if not placement_plan or rtt_val is None:
                        continue
                    
                    sorted_tasks = sorted(placement_plan.keys(), key=lambda x: int(x))
                    combo: Tuple[Tuple[int, int], ...] = tuple(
                        (int(placement_plan[task][0]), int(placement_plan[task][1]))
                        for task in sorted_tasks
                        if isinstance(placement_plan[task], list) and len(placement_plan[task]) >= 2
                    )
                    
                    if len(combo) == 0:
                        continue
                    
                    key = (dataset_id, combo)
                    if key not in results:
                        results[key] = float(rtt_val)
                        
                except (json.JSONDecodeError, ValueError, KeyError, IndexError):
                    continue
    except Exception:
        pass
    
    return results
    

def build_and_save_rtt_hash_table_chunked(
    base_dirs: List[Path], 
    cache_dir: Path,
    n_jobs: int = 24,
    chunk_size: int = 5_000_000
) -> int:
    """
    Build RTT hash table in parallel from multiple directories and save in chunks to avoid OOM.
    
    Args:
        base_dirs: List of paths to gnn_datasets directories
        cache_dir: Path to save cache files
        n_jobs: Number of parallel jobs
        chunk_size: Number of entries per chunk file
    
    Returns the total number of entries saved.
    """
    # Collect all JSONL files from all base directories
    all_jsonl_files = []
    for base_dir in base_dirs:
        if base_dir.exists():
            files = sorted(base_dir.glob("ds_*/placements/placements.jsonl"))
            all_jsonl_files.extend(files)
            print(f"Found {len(files)} JSONL files in {base_dir.name}")
    
    print(f"Building placement RTT hash table from {len(all_jsonl_files)} JSONL files using n_jobs={n_jobs}...")
    start_time = time.perf_counter()
    
    # Parse all files in parallel - each returns a small dict
    parsed_dicts: List[Dict] = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_parse_jsonl_file_to_dict)(Path(p)) for p in tqdm(all_jsonl_files, desc="Parsing JSONL files")
    )
    
    parse_time = time.perf_counter() - start_time
    print(f"Parsed {len(all_jsonl_files)} files in {parse_time:.2f}s")
    
    # Merge and save in chunks
    print("Merging results and saving in chunks...")
    merge_start = time.perf_counter()
    
    chunk_idx = 0
    current_chunk: Dict[Tuple[str, Tuple[Tuple[int, int], ...]], float] = {}
    total_entries = 0
    num_duplicates = 0
    
    for parsed_dict in tqdm(parsed_dicts, desc="Merging"):
        for key, rtt in parsed_dict.items():
            if key not in current_chunk:
                current_chunk[key] = rtt
                total_entries += 1
            else:
                num_duplicates += 1
            
            # Save chunk when it reaches chunk_size
            if len(current_chunk) >= chunk_size:
                chunk_path = cache_dir / f"rtt_chunk_{chunk_idx}.pkl"
                with open(chunk_path, 'wb') as f:
                    pickle.dump(current_chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"  Saved chunk {chunk_idx} ({len(current_chunk):,} entries) to {chunk_path}")
                chunk_idx += 1
                current_chunk = {}
        
        # Clear parsed dict to free memory
        parsed_dict.clear()
    
    # Save remaining entries
    if current_chunk:
        chunk_path = cache_dir / f"rtt_chunk_{chunk_idx}.pkl"
        with open(chunk_path, 'wb') as f:
            pickle.dump(current_chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved chunk {chunk_idx} ({len(current_chunk):,} entries) to {chunk_path}")
        chunk_idx += 1
    
    # Save metadata about chunks
    chunk_meta = {
        'num_chunks': chunk_idx,
        'total_entries': total_entries,
        'chunk_size': chunk_size,
    }
    meta_path = cache_dir / "rtt_chunks_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(chunk_meta, f)
    
    merge_time = time.perf_counter() - merge_start
    total_time = time.perf_counter() - start_time
    
    print(f"\nSaved {total_entries:,} entries in {chunk_idx} chunks")
    print(f"Timing: parse={parse_time:.2f}s, merge+save={merge_time:.2f}s, total={total_time:.2f}s")
    if num_duplicates > 0:
        print(f"Note: Found {num_duplicates:,} duplicate keys (kept first occurrence)")
    
    return total_entries


def load_rtt_hash_table_chunked(cache_dir: Path) -> Dict[Tuple[str, Tuple[Tuple[int, int], ...]], float]:
    """Load RTT hash table from chunks."""
    meta_path = cache_dir / "rtt_chunks_meta.json"
    
    if not meta_path.exists():
        # Fall back to single file if it exists
        single_path = cache_dir / "placement_rtt_hash_table.pkl"
        if single_path.exists():
            print(f"Loading RTT hash table from single file: {single_path}")
            with open(single_path, 'rb') as f:
                return pickle.load(f)
        raise FileNotFoundError(f"No RTT hash table found in {cache_dir}")
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    num_chunks = meta['num_chunks']
    total_entries = meta['total_entries']
    
    print(f"Loading RTT hash table from {num_chunks} chunks ({total_entries:,} entries)...")
    
    placement_rtt_map: Dict[Tuple[str, Tuple[Tuple[int, int], ...]], float] = {}
    
    for i in tqdm(range(num_chunks), desc="Loading chunks"):
        chunk_path = cache_dir / f"rtt_chunk_{i}.pkl"
        with open(chunk_path, 'rb') as f:
            chunk = pickle.load(f)
        placement_rtt_map.update(chunk)
    
    print(f"Loaded {len(placement_rtt_map):,} entries")
    return placement_rtt_map


# ============================================================================
# GRAPH CONSTRUCTION (same as main script)
# ============================================================================

TASK_PLATFORM_COMPATIBILITY = {
    'dnn1': ['rpiCpu', 'xavierGpu', 'xavierCpu', 'pynqFpga'],
    'dnn2': ['rpiCpu', 'xavierGpu', 'xavierCpu']
}

def build_graph(
    df_nodes, 
    df_tasks, 
    df_platforms, 
    queue_snapshot: Optional[Dict[str, int]] = None,
    temporal_state: Optional[Dict[str, Dict[str, float]]] = None
) -> Data:
    """
    Build a bipartite graph with tasks and platforms as nodes.
    
    Args:
        df_nodes: DataFrame with node information
        df_tasks: DataFrame with task information
        df_platforms: DataFrame with platform information
        queue_snapshot: Dict mapping "node_name:platform_id" -> queue_length (from full_queue_snapshot)
        temporal_state: Dict mapping "node_name:platform_id" -> {current_task_remaining, ...}
    """
    
    # Load priors (task-types) used for edge features
    _cached = globals().get("_CACHED_TASK_PRIORS", None)
    if _cached is None:
        try:
            with open("/root/projects/my-herosim/data/nofs-ids/task-types.json", "r") as f:
                globals()["_CACHED_TASK_PRIORS"] = json.load(f)
        except Exception:
            globals()["_CACHED_TASK_PRIORS"] = {}
    _CACHED_TASK_PRIORS = globals()["_CACHED_TASK_PRIORS"]
    
    # Basic sizes / offsets
    n_tasks = len(df_tasks)
    n_platforms = len(df_platforms)
    task_offset = 0
    platform_offset = n_tasks
    
    # Precompute lookups
    first_idx_per_name = (
        df_nodes.reset_index()[['index', 'node_name']]
        .groupby('node_name', as_index=True)['index']
        .first()
        .to_dict()
    )
    
    plat_pos_by_id = {row.platform_id: i for i, row in enumerate(df_platforms.itertuples(index=False))}
    
    plats_by_node = {}
    node_names_arr = df_platforms['node_name'].to_numpy()
    for pos, name in enumerate(node_names_arr):
        plats_by_node.setdefault(name, []).append(pos)
    
    network_map_by_node = {row.node_name: row.network_map for row in df_nodes.itertuples(index=False)}
    
    plat_types_by_pos = df_platforms['platform_type'].to_numpy()
    plat_node_by_pos = df_platforms['node_name'].to_numpy()
    plat_ids_arr = df_platforms['platform_id'].to_numpy()
    
    # TASK FEATURES (3 dims: 2 type + 1 source)
    # Note: QoS features removed since co-simulation doesn't capture QoS violations as ground truth
    task_types_vocab = np.array(['dnn1', 'dnn2'])
    task_type_arr = df_tasks['task_type'].to_numpy()
    task_onehot = (task_type_arr[:, None] == task_types_vocab[None, :]).astype(float)
    
    src_names = df_tasks['source_node'].to_numpy()
    src_idx = np.fromiter((first_idx_per_name.get(n, 0) for n in src_names),
                          dtype=np.float64, count=n_tasks)
    src_norm = (src_idx / max(len(df_nodes), 1)).reshape(-1, 1)
    
    task_features = np.concatenate([task_onehot, src_norm], axis=1)
    task_features_tensor = torch.from_numpy(task_features).to(torch.float32)
    
    # PLATFORM FEATURES (13 dims: 5 type + 2 replica + 1 queue + 3 temporal + 2 consolidation)
    platform_types_vocab = np.array(['rpiCpu','xavierCpu','xavierGpu','xavierDla','pynqFpga'])
    plat_type_arr = df_platforms['platform_type'].to_numpy()
    plat_onehot = (plat_type_arr[:, None] == platform_types_vocab[None, :]).astype(float)
    
    has_dnn1_arr = df_platforms['has_dnn1_replica'].to_numpy(dtype=bool)
    has_dnn2_arr = df_platforms['has_dnn2_replica'].to_numpy(dtype=bool)
    
    has_dnn1 = has_dnn1_arr.astype(float).reshape(-1, 1)
    has_dnn2 = has_dnn2_arr.astype(float).reshape(-1, 1)
    
    # QUEUE LENGTH FEATURE (normalized by QUEUE_NORM_FACTOR)
    queue_lengths = np.zeros(n_platforms, dtype=np.float64)
    if queue_snapshot:
        for pos in range(n_platforms):
            node_name = str(plat_node_by_pos[pos])
            plat_id = int(plat_ids_arr[pos])
            key = f"{node_name}:{plat_id}"
            queue_lengths[pos] = queue_snapshot.get(key, 0)
    
    # Normalize queue lengths
    queue_lengths_norm = (queue_lengths / QUEUE_NORM_FACTOR).reshape(-1, 1)
    
    # TEMPORAL STATE FEATURES (current task remaining times)
    # Since we don't have exact temporal state, we approximate:
    # - If queue > 0: platform is busy, estimate remaining time
    # - Otherwise: platform is idle
    current_task_remaining = np.zeros(n_platforms, dtype=np.float64)
    cold_start_remaining = np.zeros(n_platforms, dtype=np.float64)
    comm_remaining = np.zeros(n_platforms, dtype=np.float64)
    
    if temporal_state:
        for pos in range(n_platforms):
            node_name = str(plat_node_by_pos[pos])
            plat_id = int(plat_ids_arr[pos])
            key = f"{node_name}:{plat_id}"
            temp_state = temporal_state.get(key, {})
            current_task_remaining[pos] = temp_state.get('current_task_remaining', 0.0)
            cold_start_remaining[pos] = temp_state.get('cold_start_remaining', 0.0)
            comm_remaining[pos] = temp_state.get('comm_remaining', 0.0)
    else:
        # Approximate: if queue > 0, estimate some remaining time
        for pos in range(n_platforms):
            if queue_lengths[pos] > 0:
                # Estimate: average execution time for platform type
                plat_type = str(plat_types_by_pos[pos])
                # Get average exec time across task types for this platform
                avg_exec = 0.0
                count = 0
                for task_type in task_types_vocab:
                    task_priors = _CACHED_TASK_PRIORS.get(str(task_type), {})
                    exec_map = task_priors.get("executionTime", {})
                    if isinstance(exec_map, dict):
                        exec_time = exec_map.get(plat_type, 0.0)
                        if exec_time > 0:
                            avg_exec += exec_time
                            count += 1
                if count > 0:
                    current_task_remaining[pos] = avg_exec / count
                    # Cold start typically much shorter than execution for warm platforms
                    cold_start_remaining[pos] = current_task_remaining[pos] * 0.1
                    comm_remaining[pos] = current_task_remaining[pos] * 0.05
    
    # Normalize temporal features (assume max ~10s)
    current_task_remaining_norm = (current_task_remaining / 10.0).reshape(-1, 1)
    cold_start_remaining_norm = (cold_start_remaining / 10.0).reshape(-1, 1)
    comm_remaining_norm = (comm_remaining / 10.0).reshape(-1, 1)
    
    # CONSOLIDATION METRICS (target concurrency and usage ratio)
    # Calculate target concurrency per platform (similar to HRC logic)
    # Baseline: fastest platform for each task type
    target_concurrencies = np.zeros(n_platforms, dtype=np.float64)
    usage_ratios = np.zeros(n_platforms, dtype=np.float64)
    
    # For each platform, calculate target concurrency based on task types it supports
    for pos in range(n_platforms):
        plat_type = str(plat_types_by_pos[pos])
        # Find which task types can run on this platform
        supported_task_types = []
        for task_type in task_types_vocab:
            task_priors = _CACHED_TASK_PRIORS.get(str(task_type), {})
            platforms = task_priors.get("platforms", [])
            if plat_type in platforms:
                supported_task_types.append(str(task_type))
        
        # Calculate target concurrency: average of baseline concurrency for supported task types
        # HRC uses baseline platform (fastest) as reference
        baseline_concurrency = 5.0  # Default target (can be tuned)
        if supported_task_types:
            # Find fastest platform for each supported task type
            min_exec_times = []
            for task_type in supported_task_types:
                task_priors = _CACHED_TASK_PRIORS.get(task_type, {})
                exec_map = task_priors.get("executionTime", {})
                if isinstance(exec_map, dict) and exec_map:
                    min_exec = min(exec_map.values())
                    min_exec_times.append(min_exec)
            
            if min_exec_times:
                # Target concurrency inversely related to execution time
                avg_min_exec = np.mean(min_exec_times)
                exec_map_this = _CACHED_TASK_PRIORS.get(supported_task_types[0], {}).get("executionTime", {})
                this_exec = exec_map_this.get(plat_type, avg_min_exec) if isinstance(exec_map_this, dict) else avg_min_exec
                if this_exec > 0:
                    target_concurrencies[pos] = baseline_concurrency * (avg_min_exec / this_exec)
                else:
                    target_concurrencies[pos] = baseline_concurrency
            else:
                target_concurrencies[pos] = baseline_concurrency
        else:
            target_concurrencies[pos] = baseline_concurrency
        
        # Usage ratio: queue_length / target_concurrency
        if target_concurrencies[pos] > 0:
            usage_ratios[pos] = queue_lengths[pos] / target_concurrencies[pos]
        else:
            usage_ratios[pos] = 0.0
    
    # Normalize consolidation metrics
    target_concurrency_norm = (target_concurrencies / 20.0).reshape(-1, 1)  # Assume max ~20
    usage_ratio_norm = (usage_ratios / 5.0).reshape(-1, 1)  # Assume max usage ratio ~5
    
    # Concatenate all platform features
    platform_features = np.concatenate([
        plat_onehot,  # 5 dims
        has_dnn1, has_dnn2,  # 2 dims
        queue_lengths_norm,  # 1 dim
        current_task_remaining_norm, cold_start_remaining_norm, comm_remaining_norm,  # 3 dims
        target_concurrency_norm, usage_ratio_norm  # 2 dims
    ], axis=1)
    platform_features_tensor = torch.from_numpy(platform_features).to(torch.float32)
    
    # Cache feasible platforms per source node
    feasible_plats_cache = {}
    def feasible_platform_positions(src_node_name: str) -> np.ndarray:
        """Get network-feasible platform positions."""
        hit = feasible_plats_cache.get(src_node_name)
        if hit is not None:
            return hit
        nm = network_map_by_node.get(src_node_name, {})
        feasible_nodes = [src_node_name, *nm.keys()] if isinstance(nm, dict) else [src_node_name]
        out = []
        for node in feasible_nodes:
            out.extend(plats_by_node.get(node, ()))
        arr = np.fromiter(out, dtype=np.int64, count=len(out)) if out else np.empty(0, dtype=np.int64)
        feasible_plats_cache[src_node_name] = arr
        return arr
    
    # Compatibility filtering
    allowed_types_dnn1 = np.array(TASK_PLATFORM_COMPATIBILITY.get('dnn1', []))
    allowed_types_dnn2 = np.array(TASK_PLATFORM_COMPATIBILITY.get('dnn2', []))
    
    plat_type_compat_dnn1 = np.isin(plat_type_arr, allowed_types_dnn1)
    plat_type_compat_dnn2 = np.isin(plat_type_arr, allowed_types_dnn2)
    
    def filter_compatible_platforms(
        network_feasible_plats: np.ndarray,
        task_type: str
    ) -> np.ndarray:
        """Filter platforms by compatibility rules."""
        if network_feasible_plats.size == 0:
            return network_feasible_plats
        
        if task_type == 'dnn1':
            type_mask = plat_type_compat_dnn1
        elif task_type == 'dnn2':
            type_mask = plat_type_compat_dnn2
        else:
            return np.empty(0, dtype=np.int64)
        
        compatible_mask = type_mask[network_feasible_plats]
        return network_feasible_plats[compatible_mask]
    
    # EDGES + LABELS
    edge_src, edge_dst = [], []
    edge_attrs = []
    y_list = []
    
    # NEW: Build per-task mapping from logit index -> (node_id, platform_id)
    # This is needed for StructuredRegretLoss to look up RTT in hash table
    # task_logit_to_placement[task_idx][logit_idx] = (node_id, platform_id)
    task_logit_to_placement: Dict[int, List[Tuple[int, int]]] = {}
    
    # Build node_name -> node_id mapping
    node_name_to_id = {row.node_name: row.node_id for row in df_nodes.itertuples(index=False)}
    
    optimal_platform_ids = df_tasks['optimal_platform_id'].to_numpy()
    task_types_arr = df_tasks['task_type'].to_numpy()
    
    for t_pos, (src_name, opt_pid, task_type) in enumerate(zip(src_names, optimal_platform_ids, task_types_arr)):
        network_feas_plats = feasible_platform_positions(src_name)
        compat_plats = filter_compatible_platforms(network_feas_plats, task_type)
        
        if compat_plats.size:
            task_node_idx = task_offset + t_pos
            edge_src.extend([task_node_idx] * compat_plats.size)
            dst_list = (platform_offset + compat_plats).tolist()
            edge_dst.extend(dst_list)
            
            # Build logit_idx -> (node_id, platform_id) mapping for this task
            task_logit_to_placement[t_pos] = []
            
            task_type = str(task_type)
            task_priors = _CACHED_TASK_PRIORS.get(task_type, {})
            exec_map = task_priors.get("executionTime", {})
            src_nm = network_map_by_node.get(src_name, {})
            for logit_idx, plat_pos in enumerate(compat_plats.tolist()):
                plat_type = str(plat_types_by_pos[plat_pos])
                plat_node_name = str(plat_node_by_pos[plat_pos])
                plat_id = int(plat_ids_arr[plat_pos])
                node_id = node_name_to_id.get(plat_node_name, -1)
                
                # Store mapping: logit_idx -> (node_id, platform_id)
                task_logit_to_placement[t_pos].append((node_id, plat_id))
                
                exec_time = float(exec_map.get(plat_type, 0.0)) if isinstance(exec_map, dict) else 0.0
                
                # Network latency
                lat_entry = src_nm.get(plat_node_name, {}) if isinstance(src_nm, dict) else {}
                if isinstance(lat_entry, dict):
                    latency = float(lat_entry.get('latency', 0.0))
                else:
                    try:
                        latency = float(lat_entry)
                    except Exception:
                        latency = 0.0
                
                # Warm replica flag
                if task_type == 'dnn1':
                    is_warm = float(has_dnn1_arr[plat_pos])
                elif task_type == 'dnn2':
                    is_warm = float(has_dnn2_arr[plat_pos])
                else:
                    is_warm = 0.0
                
                # Energy consumption (from task-types.json)
                energy = 0.0
                energy_map = task_priors.get("energy", {})
                if isinstance(energy_map, dict):
                    energy = float(energy_map.get(plat_type, 0.0))
                
                # Communication time (storage read + write)
                # Estimate from state sizes and typical storage throughput
                comm_time = 0.0
                state_size_map = task_priors.get("stateSize", {})
                if isinstance(state_size_map, dict):
                    # Use first application type's state size (approximation)
                    app_state = list(state_size_map.values())[0] if state_size_map else {}
                    if isinstance(app_state, dict):
                        input_size = app_state.get("input", 0)  # bytes
                        output_size = app_state.get("output", 0)  # bytes
                        # Typical storage: 100 MB/s throughput, 1ms latency
                        storage_throughput = 100.0 * 1024 * 1024  # bytes/s
                        storage_latency = 0.001  # seconds
                        read_time = (input_size / storage_throughput) + storage_latency
                        write_time = (output_size / storage_throughput) + storage_latency
                        comm_time = read_time + write_time
                
                # Edge attributes: [exec_time, latency, is_warm, energy, comm_time] (5 dims)
                # Note: penalty_score removed since co-simulation doesn't capture QoS violations as ground truth
                edge_attrs.append([exec_time, latency, is_warm, energy, comm_time])
            
            opt_pos = plat_pos_by_id.get(opt_pid, None)
            if opt_pos is None:
                y_list.append(-1)
            else:
                matches = np.nonzero(compat_plats == opt_pos)[0]
                if matches.size:
                    y_list.append(int(matches[0]))
                else:
                    y_list.append(-1)
        else:
            y_list.append(-1)
    
    # Stack edges
    if edge_src:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr_tensor = torch.tensor(edge_attrs, dtype=torch.float32) if edge_attrs else torch.empty((0, 5), dtype=torch.float32)
        num_nodes = n_tasks + n_platforms
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        if edge_attr_tensor.numel() > 0:
            # For undirected edges, duplicate edge attributes
            edge_attr_tensor = torch.cat([edge_attr_tensor, edge_attr_tensor.clone()], dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.empty((0, 5), dtype=torch.float32)
    
    y = torch.tensor(y_list, dtype=torch.long)
    
    # Create PyG Data
    task_idx_to_task_id = {i: row.task_id for i, row in enumerate(df_tasks.itertuples(index=False))}
    
    data = Data(
        edge_index=edge_index,
        y=y,
        n_tasks=n_tasks,
        n_platforms=n_platforms,
        task_features=task_features_tensor,
        platform_features=platform_features_tensor,
    )
    data.edge_attr = edge_attr_tensor
    data._plat_pos_by_id = plat_pos_by_id
    data._task_idx_to_task_id = task_idx_to_task_id
    # NEW: Per-task mapping from logit index -> (node_id, platform_id) for regret loss
    data._task_logit_to_placement = task_logit_to_placement
    
    return data


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    script_start_time = time.perf_counter()
    
    print("="*80)
    print("PRE-GENERATING GRAPH CACHE" + (" (MERGED DATASETS)" if MERGE_DATASETS else ""))
    print("="*80)
    if MERGE_DATASETS:
        print(f"Merging datasets from {len(BASE_DIRS)} directories:")
        for bd in BASE_DIRS:
            print(f"  - {bd}")
    else:
        print(f"Loading from: {BASE_DIRS[0]}")
    print()
    
    # Load all datasets
    print("Step 1: Loading datasets...")
    step1_start = time.perf_counter()
    all_datasets = load_all_datasets(BASE_DIRS)
    step1_time = time.perf_counter() - step1_start
    
    if len(all_datasets) == 0:
        print("ERROR: No datasets loaded!")
        sys.exit(1)
    
    # Build helper maps for validation regret computation
    print("\nStep 2: Building helper maps (platform->node mapping and optimal RTT)...")
    step2_start = time.perf_counter()
    
    # Build DATA_PLAT_NODE_MAP: dataset_id -> { platform_id -> node_id }
    plat_node_map = {
        ds_id: {int(row.platform_id): int(row.node_id) for row in ds_dict['platforms'].itertuples(index=False)}
        for ds_id, ds_dict in all_datasets.items()
    }
    
    # Build DATA_OPTIMAL_RTT: dataset_id -> optimal RTT (best.json)
    optimal_rtt_map = {
        ds_id: float(ds_dict['metrics']['total_rtt'].iloc[0]) if 'metrics' in ds_dict and not ds_dict['metrics'].empty else 0.0
        for ds_id, ds_dict in all_datasets.items()
    }
    
    step2_time = time.perf_counter() - step2_start
    print(f"Built helper maps for {len(plat_node_map)} datasets")
    
    # Build RTT hash table (parallel + chunked saving)
    print("\nStep 3: Building and saving RTT hash table in chunks...")
    step3_start = time.perf_counter()
    num_rtt_entries = build_and_save_rtt_hash_table_chunked(BASE_DIRS, CACHE_DIR, n_jobs=24)
    step3_time = time.perf_counter() - step3_start
    
    # Build graphs
    print("\nStep 4: Building graphs...")
    step4_start = time.perf_counter()
    graphs = []
    dataset_ids = []
    
    for dataset_id, dataset_dict in tqdm(all_datasets.items(), desc="Building graphs", unit="dataset"):
        try:
            graph = build_graph(
                dataset_dict['nodes'],
                dataset_dict['tasks'],
                dataset_dict['platforms'],
                queue_snapshot=dataset_dict.get('queue_snapshot', {}),
                temporal_state=dataset_dict.get('temporal_state', {})
            )
            graph.dataset_id = dataset_id
            graphs.append(graph)
            dataset_ids.append(dataset_id)
        except Exception as e:
            tqdm.write(f"  Error building graph for {dataset_id}: {e}")
    
    step4_time = time.perf_counter() - step4_start
    print(f"\nBuilt {len(graphs)} graphs in {step4_time:.2f}s")
    
    # Compute statistics (including task count distribution)
    stats_start = time.perf_counter()
    ys = np.concatenate([g.y.numpy() for g in graphs])
    
    # Task count distribution
    task_count_dist = {}
    for g in graphs:
        n = int(g.n_tasks)
        task_count_dist[n] = task_count_dist.get(n, 0) + 1
    
    stats_time = time.perf_counter() - stats_start
    
    print(f"\nStatistics:")
    print(f"  Total graphs: {len(graphs)}")
    print(f"  Task count distribution:")
    for n_tasks in sorted(task_count_dist.keys()):
        print(f"    {n_tasks} tasks: {task_count_dist[n_tasks]} graphs ({task_count_dist[n_tasks]/len(graphs)*100:.1f}%)")
    print(f"  Valid labels: {np.sum(ys >= 0)} / {len(ys)} ({np.sum(ys >= 0)/len(ys)*100:.1f}%)")
    print(f"  Graphs with no edges: {sum([g.edge_index.numel() == 0 for g in graphs])} / {len(graphs)}")
    print(f"  Avg edges per graph: {np.mean([g.edge_index.size(1) for g in graphs]):.1f}")
    print(f"  Avg valid tasks per graph: {np.mean([(g.y >= 0).sum().item() for g in graphs]):.2f}")
    print(f"  Statistics computed in {stats_time:.2f}s")
    
    # Save to cache
    print("\nStep 5: Saving to cache...")
    step5_start = time.perf_counter()
    
    # Save graphs
    save_start = time.perf_counter()
    with open(GRAPHS_CACHE_PATH, 'wb') as f:
        pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    graphs_save_time = time.perf_counter() - save_start
    print(f"  Saved {len(graphs)} graphs to {GRAPHS_CACHE_PATH} ({graphs_save_time:.2f}s)")
    
    # Save dataset IDs
    save_start = time.perf_counter()
    with open(DATASET_IDS_CACHE_PATH, 'wb') as f:
        pickle.dump(dataset_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
    ids_save_time = time.perf_counter() - save_start
    print(f"  Saved dataset IDs to {DATASET_IDS_CACHE_PATH} ({ids_save_time:.2f}s)")
    
    # RTT hash table already saved in chunks during Step 3
    print(f"  RTT hash table already saved in chunks ({num_rtt_entries:,} entries)")
    
    # Save helper maps
    save_start = time.perf_counter()
    with open(PLAT_NODE_MAP_CACHE_PATH, 'wb') as f:
        pickle.dump(plat_node_map, f, protocol=pickle.HIGHEST_PROTOCOL)
    plat_node_save_time = time.perf_counter() - save_start
    print(f"  Saved platform->node mapping ({len(plat_node_map)} datasets) to {PLAT_NODE_MAP_CACHE_PATH} ({plat_node_save_time:.2f}s)")
    
    save_start = time.perf_counter()
    with open(OPTIMAL_RTT_CACHE_PATH, 'wb') as f:
        pickle.dump(optimal_rtt_map, f, protocol=pickle.HIGHEST_PROTOCOL)
    optimal_rtt_save_time = time.perf_counter() - save_start
    print(f"  Saved optimal RTT mapping ({len(optimal_rtt_map)} datasets) to {OPTIMAL_RTT_CACHE_PATH} ({optimal_rtt_save_time:.2f}s)")
    
    # Save metadata
    save_start = time.perf_counter()
    metadata = {
        'version': CACHE_VERSION,
        'merged_datasets': MERGE_DATASETS,
        'base_dirs': [str(bd) for bd in BASE_DIRS],
        'num_graphs': len(graphs),
        'num_datasets': len(all_datasets),
        'num_rtt_entries': num_rtt_entries,
        'dataset_ids': dataset_ids,
        'statistics': {
            'valid_labels': int(np.sum(ys >= 0)),
            'total_labels': len(ys),
            'graphs_with_no_edges': int(sum([g.edge_index.numel() == 0 for g in graphs])),
            'avg_edges': float(np.mean([g.edge_index.size(1) for g in graphs])),
            'avg_valid_tasks': float(np.mean([(g.y >= 0).sum().item() for g in graphs])),
            'task_count_distribution': {str(k): v for k, v in task_count_dist.items()},
        },
        'timing': {
            'step1_load_datasets': step1_time,
            'step2_build_helper_maps': step2_time,
            'step3_build_rtt_hash': step3_time,
            'step4_build_graphs': step4_time,
            'step5_save_cache': time.perf_counter() - step5_start,
            'total_time': time.perf_counter() - script_start_time,
        }
    }
    
    with open(METADATA_CACHE_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    metadata_save_time = time.perf_counter() - save_start
    print(f"  Saved metadata to {METADATA_CACHE_PATH} ({metadata_save_time:.2f}s)")
    
    step5_time = time.perf_counter() - step5_start
    total_time = time.perf_counter() - script_start_time
    
    # Compute file sizes
    graphs_size = GRAPHS_CACHE_PATH.stat().st_size / (1024 * 1024)  # MB
    
    # Sum up all RTT chunk sizes
    rtt_size = 0.0
    for chunk_file in CACHE_DIR.glob("rtt_chunk_*.pkl"):
        rtt_size += chunk_file.stat().st_size / (1024 * 1024)
    
    plat_node_size = PLAT_NODE_MAP_CACHE_PATH.stat().st_size / (1024 * 1024)  # MB
    optimal_rtt_size = OPTIMAL_RTT_CACHE_PATH.stat().st_size / (1024 * 1024)  # MB
    
    print("\n" + "="*80)
    print("CACHE GENERATION COMPLETE!")
    print("="*80)
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Graphs cache: {GRAPHS_CACHE_PATH} ({graphs_size:.2f} MB)")
    print(f"RTT hash cache: {len(list(CACHE_DIR.glob('rtt_chunk_*.pkl')))} chunks ({rtt_size:.2f} MB total)")
    print(f"Platform->node mapping cache: {PLAT_NODE_MAP_CACHE_PATH} ({plat_node_size:.2f} MB)")
    print(f"Optimal RTT cache: {OPTIMAL_RTT_CACHE_PATH} ({optimal_rtt_size:.2f} MB)")
    print(f"Total cache size: {graphs_size + rtt_size + plat_node_size + optimal_rtt_size:.2f} MB")
    print(f"Cache version: {CACHE_VERSION}")
    print()
    print("Timing Summary:")
    print(f"  Step 1 - Load datasets:        {step1_time:7.2f}s")
    print(f"  Step 2 - Build helper maps:    {step2_time:7.2f}s")
    print(f"  Step 3 - Build RTT hash:       {step3_time:7.2f}s")
    print(f"  Step 4 - Build graphs:         {step4_time:7.2f}s")
    print(f"  Step 5 - Save cache:           {step5_time:7.2f}s")
    print(f"  Total time:                    {total_time:7.2f}s")
    print()


if __name__ == "__main__":
    main()

