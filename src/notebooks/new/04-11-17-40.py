# %%
#!/usr/bin/env python3
"""
GNN for Task-to-Platform Placement Prediction
Train a Graph Isomorphism Network (GIN) to predict optimal task placements.
"""

import os
import json
import random
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.nn.models import GIN
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from joblib import Parallel, delayed
import wandb


random.seed(42); 
np.random.seed(42); 
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %%
# Configuration
# /artifacts/run10_all
BASE_DIR = Path("/root/projects/my-herosim/simulation_data/artifacts/run9_all/gnn_datasets")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters intended for grid search
# 16, 32, 64
EMBEDDING_DIM = 32

# 32, 64, 128
HIDDEN_DIM = 32

# 0.005, 0.001, 0.0005 
LEARNING_RATE = 0.001

# 16, 32
BATCH_SIZE = 16

# 3, 5
NUM_GIN_LAYERS = 3

# don't grid search
WEIGHT_DECAY = 1e-3
EPOCHS = 300

# %%
# ============================================================================
# DATA LOADING (reuse extraction logic)
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
    # Filter: Only include tasks with positive IDs that appear in placement_plan
    # These are the tasks we'll have at inference time in the real scheduler
    placement_plan_task_ids = set()
    for k in placement_plan.keys():
        task_id = int(k)
        if task_id >= 0:
            placement_plan_task_ids.add(task_id)
    
    tasks_data = []
    task_ids_seen = []
    
    for task_result in task_results:
        task_id = task_result.get("taskId")
        
        # Filter: Only include tasks with positive IDs that are in placement_plan
        if task_id is None or task_id < 0 or task_id not in placement_plan_task_ids:
            continue  # Skip negative IDs (cosimulation logic) and tasks not in placement plan
        
        task_ids_seen.append(task_id)
        
        # Use task ID to look up placement in placement_plan
        placement = placement_plan.get(str(task_id), [None, None])
        
        if isinstance(placement, list) and len(placement) >= 2:
            opt_node_id, opt_platform_id = placement[0], placement[1]
        else:
            opt_node_id, opt_platform_id = None, None
        
        tasks_data.append({
            'task_id': task_id,  # Use actual task ID from JSON (matches placement_plan keys)
            'task_type': task_result.get("taskType", {}).get("name", "unknown"),
            'source_node': task_result.get("sourceNode", ""),
            'optimal_node_id': opt_node_id,
            'optimal_platform_id': opt_platform_id,
            'elapsed_time': task_result.get("elapsedTime", 0)
        })
    
    # Sort tasks by task_id to ensure consistent ordering (0, 1, 2, 3, 4...)
    tasks_data.sort(key=lambda x: x['task_id'])
    
    # Debug: Log task filtering results
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
            
            # Check replica state
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
                'has_dnn1_replica': has_dnn1_replica, # has_dnn1_replica: bool
                'has_dnn2_replica': has_dnn2_replica # has_dnn2_replica: bool
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

def _load_single_summary(summary_file: Path) -> Tuple[Optional[Dict], List[str]]:
    """Load a single summary file and return (summary, warnings)."""
    warnings_list = []
    try:
        with open(summary_file, "r") as f:
            summary = json.load(f)
            # Validate summary structure
            if 'placement_plan' not in summary:
                warnings_list.append(f"{summary_file.name} missing 'placement_plan' key")
            if 'rtt' not in summary:
                warnings_list.append(f"{summary_file.name} missing 'rtt' key")
            return summary, warnings_list
    except Exception as e:
        warnings_list.append(f"Failed to load {summary_file.name}: {e}")
        return None, warnings_list

def load_placement_summaries(dataset_dir: Path) -> List[Dict]:
    """Load all placement summaries from placements/ folder (parallelized)."""
    placements_dir = dataset_dir / "placements"
    if not placements_dir.exists():
        print(f"  [DEBUG] No placements/ directory found in {dataset_dir.name}")
        return []
    
    summary_files = list(placements_dir.glob("placement_summary_*.json"))
    if len(summary_files) == 0:
        print(f"  [DEBUG] No placement_summary_*.json files found in {dataset_dir.name}/placements/")
        return []
    
    # Parallel loading of JSON files
    if len(summary_files) > 1:
        results = Parallel(n_jobs=-1, backend='threading')(
            delayed(_load_single_summary)(f) for f in summary_files
        )
    else:
        # Single file - no need for parallel overhead
        results = [_load_single_summary(summary_files[0])]
    
    # Process results and collect warnings
    summaries = []
    all_warnings = []
    for result, warnings in results:
        all_warnings.extend(warnings)
        if result is not None:
            summaries.append(result)
    
    # Print warnings in batch (only if there are any)
    if all_warnings:
        for warning in all_warnings:
            print(f"  [WARNING] {warning}")
    
    print(f"  [DEBUG] Loaded {len(summaries)} placement summaries from {dataset_dir.name}")
    return summaries

def load_all_datasets(base_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load all datasets from gnn_datasets directory, including placement summaries."""
    all_datasets = {}
    dataset_dirs = sorted(base_dir.glob("ds_*"))
    
    print(f"Loading {len(dataset_dirs)} datasets...")
    total_summaries = 0
    datasets_with_summaries = 0
    
    for dataset_dir in dataset_dirs:
        optimal_result_path = dataset_dir / "optimal_result.json"
        if not optimal_result_path.exists():
            continue
        
        try:
            dataframes = extract_dataset_to_dataframes(optimal_result_path)
            # Load placement summaries
            # placement_summaries = load_placement_summaries(dataset_dir)
            placement_summaries = []
            if len(placement_summaries) > 0:
                datasets_with_summaries += 1
                total_summaries += len(placement_summaries)
            
            # Store both dataframes and placement summaries
            all_datasets[dataset_dir.name] = {
                **dataframes,
                '_placement_summaries': placement_summaries,
                'dataset_dir': dataset_dir
            }
        except Exception as e:
            print(f"  Error loading {dataset_dir.name}: {e}")
    
    print(f"Loaded {len(all_datasets)} datasets successfully")
    print(f"[DEBUG] Placement summaries: {total_summaries} total across {datasets_with_summaries}/{len(all_datasets)} datasets\n")
    return all_datasets

# %%
# ============================================================================
# GRAPH CONSTRUCTION (sped up; same functionality)
# ============================================================================

# Hardcoded compatibility: task types -> allowed platform types
TASK_PLATFORM_COMPATIBILITY = {
    'dnn1': ['rpiCpu', 'xavierGpu', 'xavierCpu'],
    'dnn2': ['rpiCpu', 'xavierGpu', 'xavierCpu']
}

def build_graph(df_nodes, df_tasks, df_platforms) -> Data:
    """
    Build a bipartite graph with tasks and platforms as nodes.
    Edges connect tasks to feasible platforms based on network connectivity AND compatibility.

      - Task features: one-hot(task_type in ['dnn1','dnn2']) + normalized source node index
      - Platform features: one-hot(platform_type in ['rpiCpu','xavierCpu','xavierGpu','xavierDla','pynqFpga'])
                           + has_dnn1_replica + has_dnn2_replica
      - Edge features (added):
               1) executionTime(task_type, platform_type)
               2) network latency from source node to platform's node
      - Edges: task -> compatible platforms that are:
               1) Network feasible (source node + its network neighbors)
               2) Compatible platform type (from TASK_PLATFORM_COMPATIBILITY)
               3) Have replica for the task type (has_dnn1_replica for dnn1, has_dnn2_replica for dnn2)
      - Reverse edges are added (undirected graph for GIN)
      - Labels y: for each task, index of optimal platform within *its own* compatible list; -1 if no compatible platforms
    """

    # ---------------------------------------------
    # Load priors (task-types) used for edge features
    # ---------------------------------------------
    _cached = globals().get("_CACHED_TASK_PRIORS", None)
    if _cached is None:
        try:
            with open("/root/projects/my-herosim/data/nofs-ids/task-types.json", "r") as f:
                globals()["_CACHED_TASK_PRIORS"] = json.load(f)
        except Exception:
            globals()["_CACHED_TASK_PRIORS"] = {}
    _CACHED_TASK_PRIORS = globals()["_CACHED_TASK_PRIORS"]

    # ---------------------------------------------
    # Basic sizes / offsets
    # ---------------------------------------------
    n_tasks = len(df_tasks)
    n_platforms = len(df_platforms)
    task_offset = 0
    platform_offset = n_tasks

    # ---------------------------------------------
    # Precompute lookups (no per-row DataFrame scans)
    # ---------------------------------------------
    # node_name -> FIRST matching *index label* in df_nodes (preserves original behavior)
    # If names are unique, this is just a simple mapping; if not, we take the first.
    # (Using groupby-first avoids .loc in a loop.)
    first_idx_per_name = (
        df_nodes.reset_index()[['index', 'node_name']]
        .groupby('node_name', as_index=True)['index']
        .first()
        .to_dict()
    )

    # platform_id -> positional index (0..P-1)
    plat_pos_by_id = {row.platform_id: i for i, row in enumerate(df_platforms.itertuples(index=False))}

    # node_name -> list of platform *positions* on that node
    plats_by_node = {}
    node_names_arr = df_platforms['node_name'].to_numpy()
    for pos, name in enumerate(node_names_arr):
        plats_by_node.setdefault(name, []).append(pos)

    # node_name -> network_map dict (fast direct access)
    network_map_by_node = {row.node_name: row.network_map for row in df_nodes.itertuples(index=False)}

    # ---------------------------------------------
    # Platform lookup arrays by position for edge feature construction
    # ---------------------------------------------
    plat_types_by_pos = df_platforms['platform_type'].to_numpy()
    plat_node_by_pos = df_platforms['node_name'].to_numpy()

    # ---------------------------------------------
    # TASK FEATURES (vectorized)
    # [task_type_onehot (2), source_node_id_norm (1)]
    # ---------------------------------------------
    task_types_vocab = np.array(['dnn1', 'dnn2'])
    task_type_arr = df_tasks['task_type'].to_numpy()
    task_onehot = (task_type_arr[:, None] == task_types_vocab[None, :]).astype(float)

    src_names = df_tasks['source_node'].to_numpy()
    # map to first index label; default 0 if missing
    src_idx = np.fromiter((first_idx_per_name.get(n, 0) for n in src_names),
                          dtype=np.float64, count=n_tasks)
    src_norm = (src_idx / max(len(df_nodes), 1)).reshape(-1, 1)

    task_features = np.concatenate([task_onehot, src_norm], axis=1)
    task_features_tensor = torch.from_numpy(task_features).to(torch.float32)

    # ---------------------------------------------
    # PLATFORM FEATURES (vectorized)
    # [platform_type_onehot (5), has_dnn1, has_dnn2]
    # ---------------------------------------------

    platform_types_vocab = np.array(['rpiCpu','xavierCpu','xavierGpu','xavierDla','pynqFpga'])
    plat_type_arr = df_platforms['platform_type'].to_numpy()
    plat_onehot = (plat_type_arr[:, None] == platform_types_vocab[None, :]).astype(float)

    has_dnn1_arr = df_platforms['has_dnn1_replica'].to_numpy(dtype=bool)
    has_dnn2_arr = df_platforms['has_dnn2_replica'].to_numpy(dtype=bool)
    
    # Keep float versions for features
    has_dnn1 = has_dnn1_arr.astype(float).reshape(-1, 1)
    has_dnn2 = has_dnn2_arr.astype(float).reshape(-1, 1)

    platform_features = np.concatenate([plat_onehot, has_dnn1, has_dnn2], axis=1)
    platform_features_tensor = torch.from_numpy(platform_features).to(torch.float32)

    # ---------------------------------------------
    # Cache feasible platforms per source node (avoid repeated filtering)
    # ---------------------------------------------
    feasible_plats_cache = {}
    def feasible_platform_positions(src_node_name: str) -> np.ndarray:
        """Get network-feasible platform positions (source node + network neighbors)."""
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

    # ---------------------------------------------
    # Compatibility filtering (vectorized)
    # ---------------------------------------------
    # Pre-compute allowed platform types as numpy arrays for vectorized operations
    allowed_types_dnn1 = np.array(TASK_PLATFORM_COMPATIBILITY.get('dnn1', []))
    allowed_types_dnn2 = np.array(TASK_PLATFORM_COMPATIBILITY.get('dnn2', []))
    
    # Pre-compute platform type compatibility masks (boolean arrays) using vectorized operations
    plat_type_compat_dnn1 = np.isin(plat_type_arr, allowed_types_dnn1)
    plat_type_compat_dnn2 = np.isin(plat_type_arr, allowed_types_dnn2)
    
    def filter_compatible_platforms(
        network_feasible_plats: np.ndarray,
        task_type: str
    ) -> np.ndarray:
        """
        Filter platforms by compatibility rules (vectorized):
        1. Platform type must be in TASK_PLATFORM_COMPATIBILITY[task_type]
        2. Platform must have the appropriate replica (has_dnn1_replica for dnn1, has_dnn2_replica for dnn2)
        
        Args:
            network_feasible_plats: Array of platform positions (0..P-1) that are network-feasible
            task_type: Task type ('dnn1' or 'dnn2')
            
        Returns:
            Array of compatible platform positions
        """
        if network_feasible_plats.size == 0:
            return network_feasible_plats
        
        # Get pre-computed compatibility masks
        if task_type == 'dnn1':
            type_mask = plat_type_compat_dnn1
            # Do NOT gate by warm replicas anymore
            # replica_mask = has_dnn1_arr
        elif task_type == 'dnn2':
            type_mask = plat_type_compat_dnn2
            # replica_mask = has_dnn2_arr
        else:
            # Unknown task type -> no compatible platforms
            return np.empty(0, dtype=np.int64)
        
        # Vectorized filtering: platform type compatible (no warm requirement)
        # Apply mask to the network-feasible platforms only
        compatible_mask = type_mask[network_feasible_plats]
        
        return network_feasible_plats[compatible_mask]

    # ---------------------------------------------
    # EDGES + LABELS
    # ---------------------------------------------
    edge_src, edge_dst = [], []
    edge_attrs = []  # per-edge features: [exec_time, latency, is_warm_for_task]
    y_list = []

    optimal_platform_ids = df_tasks['optimal_platform_id'].to_numpy()
    task_types_arr = df_tasks['task_type'].to_numpy()

    for t_pos, (src_name, opt_pid, task_type) in enumerate(zip(src_names, optimal_platform_ids, task_types_arr)):
        # Step 1: Get network-feasible platforms
        network_feas_plats = feasible_platform_positions(src_name)  # platform positions (0..P-1)
        
        # Step 2: Filter by compatibility (platform type + replica) - vectorized
        compat_plats = filter_compatible_platforms(network_feas_plats, task_type)
        
        if compat_plats.size:
            task_node_idx = task_offset + t_pos
            # Edges: task -> (n_tasks + platform_pos)
            edge_src.extend([task_node_idx] * compat_plats.size)
            dst_list = (platform_offset + compat_plats).tolist()
            edge_dst.extend(dst_list)

            # Edge features for each compatible platform
            task_type = str(task_type)
            task_priors = _CACHED_TASK_PRIORS.get(task_type, {})
            exec_map = task_priors.get("executionTime", {})
            src_nm = network_map_by_node.get(src_name, {})
            for plat_pos in compat_plats.tolist():
                plat_type = str(plat_types_by_pos[plat_pos])
                plat_node_name = str(plat_node_by_pos[plat_pos])
                # 1) execution time prior (seconds)
                exec_time = float(exec_map.get(plat_type, 0.0)) if isinstance(exec_map, dict) else 0.0
                # 2) network latency from source to platform node (seconds)
                lat_entry = src_nm.get(plat_node_name, {}) if isinstance(src_nm, dict) else {}
                # accept numeric or dict with 'latency'
                if isinstance(lat_entry, dict):
                    latency = float(lat_entry.get('latency', 0.0))
                else:
                    try:
                        latency = float(lat_entry)
                    except Exception:
                        latency = 0.0
                # 3) is warm for task: replica present for that task type at this platform
                if task_type == 'dnn1':
                    is_warm = float(has_dnn1_arr[plat_pos])
                elif task_type == 'dnn2':
                    is_warm = float(has_dnn2_arr[plat_pos])
                else:
                    is_warm = 0.0
                edge_attrs.append([exec_time, latency, is_warm])

            # Label: index of optimal platform within this task's COMPATIBLE list (-1 if not found)
            opt_pos = plat_pos_by_id.get(opt_pid, None)
            if opt_pos is None:
                # Optimal platform not in platform list -> invalid label
                y_list.append(-1)
            else:
                # Check if optimal platform is in the compatible list
                matches = np.nonzero(compat_plats == opt_pos)[0]
                if matches.size:
                    y_list.append(int(matches[0]))  # Index within compatible platforms
                else:
                    # Optimal platform exists but not in compatible set -> invalid label
                    y_list.append(-1)
        else:
            # No compatible platforms → no edges for this task; invalid label
            y_list.append(-1)

    # Stack edges
    if edge_src:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr_tensor = torch.tensor(edge_attrs, dtype=torch.float32) if edge_attrs else torch.empty((0, 3), dtype=torch.float32)
        # Add reverse edges: make undirected for GIN message passing
        num_nodes = n_tasks + n_platforms
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        # Duplicate edge_attr for reverse edges to keep alignment; reverse attrs won't be used in scoring
        if edge_attr_tensor.numel() > 0:
            edge_attr_tensor = torch.cat([edge_attr_tensor, torch.zeros_like(edge_attr_tensor)], dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.empty((0, 3), dtype=torch.float32)

    y = torch.tensor(y_list, dtype=torch.long)

    # ---------------------------------------------
    # Create PyG Data
    # ---------------------------------------------
    # Store task index -> actual task ID mapping (critical for matching with placement summaries)
    task_idx_to_task_id = {i: row.task_id for i, row in enumerate(df_tasks.itertuples(index=False))}
    
    data = Data(
        edge_index=edge_index,
        y=y,
        n_tasks=n_tasks,
        n_platforms=n_platforms,
        task_features=task_features_tensor,             # dim=3
        platform_features=platform_features_tensor,     # dim=7
    )
    # Attach edge attributes (exec_time, latency)
    data.edge_attr = edge_attr_tensor
    # Store platform position mapping as private attribute (PyG will skip during batching)
    data._plat_pos_by_id = plat_pos_by_id
    # Store task index -> task ID mapping (needed for matching predictions with placement summaries)
    data._task_idx_to_task_id = task_idx_to_task_id

    return data

# %%
# ============================================================================
# GNN MODEL
# ============================================================================

class TaskEncoder(nn.Module):
    """2-layer MLP encoder for task features with LayerNorm for training stability."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class PlatformEncoder(nn.Module):
    """2-layer MLP encoder for platform features with LayerNorm for training stability."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x



class EdgeScorer(nn.Module):
    """2-layer MLP to score task-platform edges with optional edge attributes."""
    def __init__(self, embedding_dim, hidden_dim, edge_dim=0):
        super().__init__()
        # Input: concatenation of task, platform embeddings and edge attrs
        in_dim = 2 * embedding_dim + (edge_dim if edge_dim else 0)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, e_task, e_platform, e_attr=None):
        # Concatenate task and platform embeddings (+ edge attrs if provided)
        x = torch.cat([e_task, e_platform] + ([e_attr] if e_attr is not None else []), dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                              # (E, 1)
        return x.squeeze(-1)                         # (E,)

class TaskPlacementGNN(nn.Module):
    """
    1. Encode task and platform features separately
    2. GIN to produce node embeddings
    3. Edge MLP to score task-platform compatibility
    4. Masked softmax to predict placement probabilities
    """
    def __init__(self, task_feature_dim, platform_feature_dim, embedding_dim=64, hidden_dim=128, num_layers=3):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Separate encoders for tasks and platforms
        self.task_encoder = TaskEncoder(task_feature_dim, hidden_dim, embedding_dim)
        self.platform_encoder = PlatformEncoder(platform_feature_dim, hidden_dim, embedding_dim)
        
        # GIN model for message passing
        self.gin = GIN(
            in_channels=embedding_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            out_channels=embedding_dim
        )
        
        # Edge scoring MLP (edge_dim=3: exec_time, latency, is_warm)
        self.edge_scorer = EdgeScorer(embedding_dim, hidden_dim, edge_dim=3)

    def forward(self, data):
        n_tasks = data.n_tasks
        n_platforms = data.n_platforms

        # Check for NaN/Inf in input features (prevent propagation)
        if torch.isnan(data.task_features).any() or torch.isinf(data.task_features).any():
            print(f"[WARNING] Task features contain NaN/Inf")
            data.task_features = torch.nan_to_num(data.task_features, nan=0.0, posinf=1e6, neginf=-1e6)
        if torch.isnan(data.platform_features).any() or torch.isinf(data.platform_features).any():
            print(f"[WARNING] Platform features contain NaN/Inf")
            data.platform_features = torch.nan_to_num(data.platform_features, nan=0.0, posinf=1e6, neginf=-1e6)

        # 1) Encode features
        task_embeddings = self.task_encoder(data.task_features)        # (T, D)
        platform_embeddings = self.platform_encoder(data.platform_features)  # (P, D)
        
        # Check for NaN/Inf after encoding
        if torch.isnan(task_embeddings).any() or torch.isinf(task_embeddings).any():
            print(f"[WARNING] Task embeddings contain NaN/Inf after encoding")
            task_embeddings = torch.nan_to_num(task_embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
        if torch.isnan(platform_embeddings).any() or torch.isinf(platform_embeddings).any():
            print(f"[WARNING] Platform embeddings contain NaN/Inf after encoding")
            platform_embeddings = torch.nan_to_num(platform_embeddings, nan=0.0, posinf=1e6, neginf=-1e6)

        # 2) Message passing on concatenated nodes
        x = torch.cat([task_embeddings, platform_embeddings], dim=0)   # (T+P, D)
        x = self.gin(x, data.edge_index)
        
        # Check for NaN/Inf after GIN
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[WARNING] Node embeddings contain NaN/Inf after GIN, clamping")
            x = torch.clamp(x, min=-50.0, max=50.0)
        
        task_emb = x[:n_tasks]        # (T, D)
        platform_emb = x[n_tasks:]    # (P, D)

        # 3) Score all edges in one shot
        ei = data.edge_index                                             # (2, E)
        if ei.numel() == 0:
            # No edges in this graph: return empty logits per task
            return [torch.empty(0, device=x.device) for _ in range(n_tasks)]

        ti = ei[0]                                                        # (E,) task indices [0..T-1]
        pj = ei[1] - n_tasks                                              # (E,) platform indices [0..P-1]
        valid = (pj >= 0) & (pj < n_platforms)
        ti = ti[valid]
        pj = pj[valid]
        if ti.numel() == 0:
            return [torch.empty(0, device=x.device) for _ in range(n_tasks)]

        e_task = task_emb[ti]                # (E_valid, D)
        e_platform = platform_emb[pj]        # (E_valid, D)
        # Select aligned edge attributes; reverse edges are filtered out by 'valid'
        e_attr = None
        if hasattr(data, 'edge_attr') and data.edge_attr.numel() > 0:
            try:
                e_attr = data.edge_attr[valid]
            except Exception:
                e_attr = None
        edge_scores = self.edge_scorer(e_task, e_platform, e_attr)   # (E_valid,)
        
        # Check for NaN/Inf in edge scores and clamp
        if torch.isnan(edge_scores).any() or torch.isinf(edge_scores).any():
            print(f"[WARNING] Edge scores contain NaN/Inf, clamping")
            edge_scores = torch.clamp(edge_scores, min=-50.0, max=50.0)  # Prevent extreme values

        # 4) Split scores per task
        logits_per_task = []
        for t in range(n_tasks):
            mask_t = (ti == t)
            logits_t = edge_scores[mask_t]   # (K_t,)
            # Clamp logits to prevent NaN/Inf in softmax
            if logits_t.numel() > 0:
                logits_t = torch.clamp(logits_t, min=-50.0, max=50.0)
            logits_per_task.append(logits_t)

        return logits_per_task

# %%
# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def loss_original_ce(logits_per_task, data, device):
    """
    Original cross-entropy loss with one-hot labels (optimal placement).
    L_t = -log(π_t,p*(t))
    """
    loss_total = torch.zeros(1, device=device)
    valid_tasks = 0
    
    for task_idx, logits_t in enumerate(logits_per_task):
        if logits_t.numel() == 0:
            continue
        
        logits = logits_t.unsqueeze(0)  # (1, K)
        target = data.y[task_idx].long()
        if target.ndim == 0:
            target = target.unsqueeze(0)
        
        if target.item() < 0 or target.item() >= logits.size(1):
            continue
        
        loss_total = loss_total + F.cross_entropy(logits, target)
        valid_tasks += 1
    
    if valid_tasks == 0:
        return torch.zeros(1, device=device), 0
    
    return loss_total / valid_tasks, valid_tasks


def loss_expected_rtt(logits_per_task, data, device, temperature=1.0):
    """
    Loss Path A: Expected RTT loss over all placements (OPTIMIZED).
    L = E_{π ~ q_θ}[RTT(π)] = Σ_π q_θ(π) * RTT(π)
    where q_θ(π) = Π_t p_θ(π(t)|t) (independent per-task assumption)
    """
    dataset_id = getattr(data, 'dataset_id', None)
    if dataset_id is None:
        return torch.zeros(1, device=device), 0
    
    # Use preprocessed placement data (much faster than parsing every time)
    preprocessed = _PREPROCESSED_PLACEMENT_DATA.get(dataset_id)
    if preprocessed is None:
        return torch.zeros(1, device=device), 0
    
    placement_rtt_map = preprocessed['placement_rtt_map']
    if len(placement_rtt_map) == 0:
        return torch.zeros(1, device=device), 0
    
    n_tasks = data.n_tasks
    
    # Cache task->platform mappings (computed once per graph)
    if not hasattr(data, '_task_to_platforms_cached'):
        ei = data.edge_index
        task_to_platforms = {}
        task_to_edge_indices = {}
        
        if ei.numel() > 0:
            ti = ei[0]
            pj = ei[1] - n_tasks
            valid = (pj >= 0) & (pj < data.n_platforms)
            ti_valid = ti[valid]
            pj_valid = pj[valid]
            
            for edge_idx in range(len(ti_valid)):
                t = int(ti_valid[edge_idx].item())
                p = int(pj_valid[edge_idx].item())
                if t not in task_to_platforms:
                    task_to_platforms[t] = []
                    task_to_edge_indices[t] = []
                task_to_platforms[t].append(p)
                task_to_edge_indices[t].append(edge_idx)
        
        data._task_to_platforms_cached = task_to_platforms
        data._task_to_edge_indices_cached = task_to_edge_indices
    else:
        task_to_platforms = data._task_to_platforms_cached
        task_to_edge_indices = data._task_to_edge_indices_cached
    
    # Compute per-task probabilities from logits
    probs_per_task = []
    for t in range(n_tasks):
        if t in task_to_edge_indices:
            edge_indices = task_to_edge_indices[t]
            logits_t = logits_per_task[t]  # (K_t,)
            probs_t = F.softmax(logits_t / temperature, dim=0)  # (K_t,)
            probs_per_task.append({
                'platforms': task_to_platforms[t],
                'probs': probs_t
            })
        else:
            probs_per_task.append(None)
    
    # Compute joint probability for each placement: q_θ(π) = Π_t p_θ(π(t)|t)
    expected_rtt = torch.zeros(1, device=device)
    total_weight = 0.0
    
    for placement_key, rtt in placement_rtt_map.items():
        # Compute joint probability
        log_prob_sum = torch.zeros(1, device=device)
        valid_placement = True
        
        for task_id, plat_pos in placement_key:
            if task_id >= n_tasks or probs_per_task[task_id] is None:
                valid_placement = False
                break
            
            plat_idx_in_task = None
            for idx, p in enumerate(probs_per_task[task_id]['platforms']):
                if p == plat_pos:
                    plat_idx_in_task = idx
                    break
            
            if plat_idx_in_task is None:
                valid_placement = False
                break
            
            prob = probs_per_task[task_id]['probs'][plat_idx_in_task]
            log_prob_sum = log_prob_sum + torch.log(prob + 1e-10)
        
        if valid_placement:
            prob_joint = torch.exp(log_prob_sum)
            expected_rtt = expected_rtt + prob_joint * rtt
            total_weight = total_weight + prob_joint.item()
    
    if total_weight > 0:
        return expected_rtt / total_weight, 1
    
    # Log if no valid placements found
    if hasattr(data, '_debug_logged_expected_rtt_weight') and not data._debug_logged_expected_rtt_weight:
        print(f"[DEBUG loss_expected_rtt] total_weight=0, no valid placements matched")
        print(f"  - placement_rtt_map size: {len(placement_rtt_map)}")
        print(f"  - n_tasks: {n_tasks}, probs_per_task valid: {sum(1 for p in probs_per_task if p is not None)}")
        data._debug_logged_expected_rtt_weight = True
    return torch.zeros(1, device=device), 0

# %%
# ============================================================================
# PLACEMENT SUMMARIES LOOKUP (global dictionary approach)
# ============================================================================
# Strategy: Store placement summaries in global dictionaries keyed by dataset_id.
# Each graph stores its dataset_id as a regular attribute (preserved by PyG).
# During training/evaluation, we look up placement summaries from the global
# dictionaries using the dataset_id and attach them to the data object.
# This avoids issues with custom attributes being lost during batching.
# ============================================================================

# Global mappings to store placement summaries and platform mappings by dataset_id
# This is the single source of truth - graphs will look up from here
_PLACEMENT_SUMMARIES_BY_DATASET_ID = {}
_PLAT_POS_BY_ID_BY_DATASET_ID = {}
_TASK_IDX_TO_TASK_ID_BY_DATASET_ID = {}  # Task index -> actual task ID mapping per dataset

# Pre-processed placement data for fast lookups (built once, reused many times)
# Key: dataset_id -> Value: {
#   'placement_rtt_map': {tuple(sorted([(task_id, plat_pos), ...])): rtt, ...},
#   'placement_key_to_summary': {tuple(...): summary_index, ...},
#   'optimal_rtt': float,
#   'max_rtt': float,
#   'sorted_by_rtt': [(summary_index, rtt), ...],
#   'task_to_platform_rtt': {task_id: {plat_pos: min_rtt, ...}, ...}
# }
_PREPROCESSED_PLACEMENT_DATA = {}

def get_placement_summaries_for_graph(data):
    """
    Get placement summaries for a graph by looking up its dataset_id.
    Returns empty list if not found or dataset_id is missing.
    """
    dataset_id = getattr(data, 'dataset_id', None)
    if dataset_id is None:
        return []
    return _PLACEMENT_SUMMARIES_BY_DATASET_ID.get(dataset_id, [])

def get_plat_pos_by_id_for_graph(data):
    """
    Get platform position mapping for a graph by looking up its dataset_id.
    Returns empty dict if not found or dataset_id is missing.
    """
    dataset_id = getattr(data, 'dataset_id', None)
    if dataset_id is None:
        return {}
    return _PLAT_POS_BY_ID_BY_DATASET_ID.get(dataset_id, {})

def preprocess_placement_data(dataset_id, placement_summaries, plat_pos_by_id):
    """
    Pre-process placement summaries into fast lookup structures.
    This is done once when graphs are built, then reused during training.
    """
    if dataset_id in _PREPROCESSED_PLACEMENT_DATA:
        return _PREPROCESSED_PLACEMENT_DATA[dataset_id]
    
    if len(placement_summaries) == 0:
        return None
    
    placement_rtt_map = {}
    placement_key_to_summary_idx = {}
    all_rtts = []
    task_to_platform_rtt = {}  # task_id -> {plat_pos: min_rtt}
    
    for summary_idx, summary in enumerate(placement_summaries):
        placement_plan = summary.get('placement_plan', {})
        rtt = summary.get('rtt', float('inf'))
        all_rtts.append(rtt)
        
        # Convert placement plan to tuple key: ((task_id, plat_pos), ...)
        key_parts = []
        for task_id_str, placement in placement_plan.items():
            task_id = int(task_id_str)
            if isinstance(placement, list) and len(placement) >= 2:
                _, plat_id = int(placement[0]), int(placement[1])
                plat_pos = plat_pos_by_id.get(plat_id, None)
                if plat_pos is not None:
                    key_parts.append((task_id, plat_pos))
                    
                    # Build task->platform->min_rtt mapping for soft-label loss
                    if task_id not in task_to_platform_rtt:
                        task_to_platform_rtt[task_id] = {}
                    if plat_pos not in task_to_platform_rtt[task_id]:
                        task_to_platform_rtt[task_id][plat_pos] = rtt
                    else:
                        task_to_platform_rtt[task_id][plat_pos] = min(
                            task_to_platform_rtt[task_id][plat_pos], rtt
                        )
        
        if key_parts:
            key = tuple(sorted(key_parts))
            # Keep minimum RTT if same placement appears multiple times
            if key not in placement_rtt_map or rtt < placement_rtt_map[key]:
                placement_rtt_map[key] = rtt
                placement_key_to_summary_idx[key] = summary_idx
    
    if len(placement_rtt_map) == 0:
        return None
    
    optimal_rtt = min(all_rtts) if all_rtts else float('inf')
    max_rtt = max(all_rtts) if all_rtts else float('inf')
    
    # Sort summaries by RTT for top-k lookup
    sorted_by_rtt = sorted(
        [(idx, s.get('rtt', float('inf'))) for idx, s in enumerate(placement_summaries)],
        key=lambda x: x[1]
    )
    
    preprocessed = {
        'placement_rtt_map': placement_rtt_map,
        'placement_key_to_summary_idx': placement_key_to_summary_idx,
        'optimal_rtt': optimal_rtt,
        'max_rtt': max_rtt,
        'sorted_by_rtt': sorted_by_rtt,
        'task_to_platform_rtt': task_to_platform_rtt,
        # 'all_rtts': all_rtts
    }
    
    _PREPROCESSED_PLACEMENT_DATA[dataset_id] = preprocessed
    return preprocessed

# %%
# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, train_loader, optimizer, device, epoch_num, is_last_epoch=False):
    model.train()
    # loss accross all graphs in the batch
    running_ce = 0.0
    running_expected_rtt = 0.0
    n_steps = 0
    
    # Track dataset_ids processed in this epoch
    dataset_ids_processed = set()

    for batch in tqdm(train_loader, desc=f"Epoch {epoch_num:3d} [Train]", leave=is_last_epoch):
        optimizer.zero_grad() # reset gradients
        graphs_in_batch = batch.to_data_list()

        loss_ce_total = torch.zeros(1, device=device)
        loss_expected_rtt_total = torch.zeros(1, device=device)
        n_graphs = 0

        for data in graphs_in_batch:
            data = data.to(device)
            
            # Attach placement summaries and platform mapping from global lookup
            dataset_id = getattr(data, 'dataset_id', None)
            if dataset_id:
                dataset_ids_processed.add(dataset_id)
                data._placement_summaries = get_placement_summaries_for_graph(data)
                data._plat_pos_by_id = get_plat_pos_by_id_for_graph(data)
                # Restore task index -> task ID mapping if not present
                if not hasattr(data, '_task_idx_to_task_id'):
                    task_idx_map = _TASK_IDX_TO_TASK_ID_BY_DATASET_ID.get(dataset_id, {})
                    if task_idx_map:
                        data._task_idx_to_task_id = task_idx_map
                    else:
                        # Fallback: assume indices match IDs
                        data._task_idx_to_task_id = {i: i for i in range(data.n_tasks)}
            
            logits_per_task = model(data)

            # Debug: Check placement summaries in batched data
            placement_summaries_attr = getattr(data, '_placement_summaries', None)
            has_summaries = placement_summaries_attr is not None and len(placement_summaries_attr) > 0
            
            # Original CE loss
            loss_ce, valid_ce = loss_original_ce(logits_per_task, data, device)
            if valid_ce > 0:
                # Check for NaN/Inf before adding
                if torch.isnan(loss_ce) or torch.isinf(loss_ce):
                    print(f"[WARNING] NaN/Inf loss detected: loss_ce={loss_ce}, valid_ce={valid_ce}")
                    # Check logits for NaN/Inf
                    for t, logits_t in enumerate(logits_per_task):
                        if logits_t.numel() > 0:
                            if torch.isnan(logits_t).any() or torch.isinf(logits_t).any():
                                print(f"  Task {t} logits contain NaN/Inf: min={logits_t.min()}, max={logits_t.max()}")
                else:
                    loss_ce_total = loss_ce_total + loss_ce
                    n_graphs += 1
            
            # Expected RTT loss (only if placement summaries available)
            if False:
            # if has_summaries:
                loss_er, valid_er = loss_expected_rtt(logits_per_task, data, device)
                if valid_er > 0:
                    loss_expected_rtt_total = loss_expected_rtt_total + loss_er

        if n_graphs == 0:
            # nothing usable in this batch; skip backward to avoid NaNs
            continue

        # Average losses
        loss_ce_avg = loss_ce_total / n_graphs
        
        # Check for NaN/Inf before combining losses
        if torch.isnan(loss_ce_avg) or torch.isinf(loss_ce_avg):
            print(f"[WARNING] Skipping backward pass: loss_ce_avg is NaN/Inf")
            continue
        
        # Use CE loss for backprop (primary loss)
        loss = loss_ce_avg
        
        # Optional: combine with RTT losses (weighted)
        # Only add RTT losses if they're valid (not NaN/Inf)
        if loss_expected_rtt_total.item() > 0 and not (torch.isnan(loss_expected_rtt_total) or torch.isinf(loss_expected_rtt_total)):
            loss = loss + 0.1 * loss_expected_rtt_total / n_graphs

        # Final check before backward
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[WARNING] Skipping backward pass: combined loss is NaN/Inf")
            continue

        # backpropagate loss
        loss.backward() # compute gradients 

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step() # update weights

        running_ce += loss_ce_avg.item()
        running_expected_rtt += loss_expected_rtt_total.item() / max(n_graphs, 1)
        n_steps += 1
    
    # Log dataset_ids processed in this epoch (every epoch)
    print(f"\n[Epoch {epoch_num}] Processed {len(dataset_ids_processed)} unique dataset_ids:")

    return {
        'ce': running_ce / max(1, n_steps),
        'expected_rtt': running_expected_rtt / max(1, n_steps),
        'dataset_ids': sorted(dataset_ids_processed)
    }

@torch.no_grad()
def decode_predicted_placement(logits_per_task, data):
    """
    Decode model predictions into a placement plan (same format as placement_summaries).
    Returns: Dict[int, List[int, int]] mapping task_id -> [node_id, platform_id]
    Uses actual task IDs from JSON, not task indices.
    """
    n_tasks = data.n_tasks
    plat_pos_by_id = getattr(data, '_plat_pos_by_id', {})
    
    # Get task index -> actual task ID mapping (critical for matching)
    task_idx_to_task_id = getattr(data, '_task_idx_to_task_id', {i: i for i in range(n_tasks)})
    
    # Reverse mapping: plat_pos -> platform_id
    plat_id_by_pos = {pos: plat_id for plat_id, pos in plat_pos_by_id.items()}
    
    # Get task->platform edge mappings
    ei = data.edge_index
    task_to_platforms = {}
    
    if ei.numel() > 0:
        ti = ei[0]
        pj = ei[1] - n_tasks
        valid = (pj >= 0) & (pj < data.n_platforms)
        ti_valid = ti[valid]
        pj_valid = pj[valid]
        
        for edge_idx in range(len(ti_valid)):
            t = int(ti_valid[edge_idx].item())
            p = int(pj_valid[edge_idx].item())
            if t not in task_to_platforms:
                task_to_platforms[t] = []
            task_to_platforms[t].append(p)
    
    # Decode greedy placement: argmax per task
    predicted_placement = {}
    
    for task_idx in range(n_tasks):
        if task_idx not in task_to_platforms or len(logits_per_task[task_idx]) == 0:
            continue
        
        # Get argmax platform position for this task
        logits_t = logits_per_task[task_idx]
        
        # Check for NaN/Inf before argmax
        if torch.isnan(logits_t).any() or torch.isinf(logits_t).any():
            continue
        
        pred_idx = logits_t.argmax().item()
        
        if pred_idx < len(task_to_platforms[task_idx]):
            plat_pos = task_to_platforms[task_idx][pred_idx]
            plat_id = plat_id_by_pos.get(plat_pos, None)
            
            if plat_id is not None:
                # Use actual task ID (from JSON), not task index
                actual_task_id = task_idx_to_task_id.get(task_idx, task_idx)
                predicted_placement[actual_task_id] = [None, plat_id]  # node_id will be filled from placement_summaries
    
    return predicted_placement


@torch.no_grad()
def compute_rtt_metrics(logits_per_task, data, device):
    """
    Compute RTT-based accuracy metrics using placement summaries (OPTIMIZED).
    Returns dict with: regret, predicted_rtt, optimal_rtt
    """
    dataset_id = getattr(data, 'dataset_id', None)
    if dataset_id is None:
        return {
            'regret': 0.0,
            'predicted_rtt': float('inf'),
            'optimal_rtt': float('inf')
        }
    
    # Use preprocessed placement data
    preprocessed = _PREPROCESSED_PLACEMENT_DATA.get(dataset_id)
    if preprocessed is None:
        return {
            'regret': 0.0,
            'predicted_rtt': float('inf'),
            'optimal_rtt': float('inf')
        }
    
    placement_rtt_map = preprocessed['placement_rtt_map']
    optimal_rtt = preprocessed['optimal_rtt']
    placement_summaries = getattr(data, '_placement_summaries', [])
    
    # Decode predicted placement
    predicted_placement = decode_predicted_placement(logits_per_task, data)
    
    # Find predicted placement's RTT using O(1) lookup
    predicted_rtt = float('inf')
            
    if predicted_placement:
        # Get platform position mapping (needed to convert plat_id to plat_pos)
        plat_pos_by_id = getattr(data, '_plat_pos_by_id', {})
        
        # Get set of task IDs that appear in placement summaries (for filtering)
        task_ids_in_summaries = set()
        for key in placement_rtt_map.keys():
            task_ids_in_summaries.update([t for t, _ in key])
        
        # Create key for matching (use platform POSITIONS, not IDs, to match summary keys)
        # Only include tasks that exist in placement summaries
        pred_key_parts = []
        for t, p in predicted_placement.items():
            if t in task_ids_in_summaries and p[1] is not None:
                plat_id = int(p[1])
                plat_pos = plat_pos_by_id.get(plat_id, None)
                if plat_pos is not None:
                    pred_key_parts.append((int(t), plat_pos))
        pred_key_parts = sorted(pred_key_parts)
        
        if len(pred_key_parts) == 0:
            predicted_rtt = float('inf')
        else:
            # O(1) lookup instead of O(n) linear search
            pred_key = tuple(pred_key_parts)
            if pred_key in placement_rtt_map:
                predicted_rtt = placement_rtt_map[pred_key]
            else:
                # Debug: Log if no match found (only once per evaluation)
                if not hasattr(compute_rtt_metrics, '_debug_logged_no_match'):
                    print(f"[DEBUG compute_rtt_metrics] Predicted placement not found in {len(placement_rtt_map)} placement keys")
                    print(f"  Predicted key (filtered to tasks in summaries): {pred_key_parts}")
                    print(f"  Task IDs in summaries: {sorted(task_ids_in_summaries)[:10]}... (showing first 10)")
                    print(f"  Task IDs in predicted (all): {sorted([int(t) for t in predicted_placement.keys()])[:10]}... (showing first 10)")
                    print(f"  Sample keys (first 3): {list(placement_rtt_map.keys())[:3]}")
                    compute_rtt_metrics._debug_logged_no_match = True
    
    # Compute regret (difference from optimal)
    if predicted_rtt == float('inf') or optimal_rtt == float('inf'):
        print(f"[DEBUG compute_rtt_metrics] Predicted RTT or Optimal RTT is inf")
        regret = float('inf')
    else:
        regret = predicted_rtt - optimal_rtt

    # Convert inf to NaN for logging
    return {
        'regret': regret if regret != float('inf') else float('nan'),
        'predicted_rtt': predicted_rtt if predicted_rtt != float('inf') else float('nan'),
        'optimal_rtt': optimal_rtt if optimal_rtt != float('inf') else float('nan'),
    }


@torch.no_grad()
def evaluate(model, loader, device, is_last_epoch=False):
    model.eval()
    total_loss_ce, total_loss_expected_rtt = 0.0, 0.0
    correct, total = 0, 0
    n_graphs = 0

    # RTT-based metrics accumulators
    total_regret = 0.0
    total_predicted_rtt = 0.0
    total_optimal_rtt = 0.0
    n_rtt_evaluations = 0
    
    # Track dataset_ids processed
    dataset_ids_processed = set()

    for batch in tqdm(loader, desc="Evaluating", leave=is_last_epoch):
        graphs_in_batch = batch.to_data_list()
        
        for data in graphs_in_batch:
            data = data.to(device)
            
            # Attach placement summaries and platform mapping from global lookup
            dataset_id = getattr(data, 'dataset_id', None)
            if dataset_id:
                dataset_ids_processed.add(dataset_id)
                data._placement_summaries = get_placement_summaries_for_graph(data)
                data._plat_pos_by_id = get_plat_pos_by_id_for_graph(data)
                # Restore task index -> task ID mapping if not present
                if not hasattr(data, '_task_idx_to_task_id'):
                    task_idx_map = _TASK_IDX_TO_TASK_ID_BY_DATASET_ID.get(dataset_id, {})
                    if task_idx_map:
                        data._task_idx_to_task_id = task_idx_map
                    else:
                        # Fallback: assume indices match IDs
                        data._task_idx_to_task_id = {i: i for i in range(data.n_tasks)}
            
            logits_per_task = model(data)

            # Original CE loss
            loss_ce, valid_ce = loss_original_ce(logits_per_task, data, device)
            if valid_ce > 0:
                total_loss_ce += loss_ce.item() * valid_ce
                total += valid_ce
                n_graphs += 1
                
                # Compute accuracy
                for task_idx, task_logits in enumerate(logits_per_task):
                    if task_logits.numel() == 0:
                        continue

                    logits = task_logits.unsqueeze(0)        # (1, K)

                    target = data.y[task_idx].long()
                    if target.ndim == 0:
                        target = target.unsqueeze(0)         # (1,)
                    if target.item() < 0 or target.item() >= logits.size(1):
                        continue

                    pred = logits.argmax(dim=1).item()       # int
                    correct += int(pred == target.item())
            
            # Expected RTT loss (only if placement summaries available)
            """
            placement_summaries_check = getattr(data, '_placement_summaries', None)
            if placement_summaries_check is not None and len(placement_summaries_check) > 0:
                loss_er, valid_er = loss_expected_rtt(logits_per_task, data, device)
                if valid_er > 0:
                    total_loss_expected_rtt += loss_er.item()

                # Compute RTT-based metrics
                rtt_metrics = compute_rtt_metrics(logits_per_task, data, device)
                    total_regret += rtt_metrics['regret']
                    total_predicted_rtt += rtt_metrics['predicted_rtt']
                    total_optimal_rtt += rtt_metrics['optimal_rtt']
                    n_rtt_evaluations += 1
            """

    avg_loss_ce = total_loss_ce / total if total else 0.0
    # avg_loss_expected_rtt = total_loss_expected_rtt / max(n_graphs, 1)
    acc = correct / total if total else 0.0
    
    # RTT-based metrics (only computed when placement summaries available)
    avg_regret = total_regret / n_rtt_evaluations if n_rtt_evaluations > 0 else 0.0
    avg_predicted_rtt = total_predicted_rtt / n_rtt_evaluations if n_rtt_evaluations > 0 else 0.0
    avg_optimal_rtt = total_optimal_rtt / n_rtt_evaluations if n_rtt_evaluations > 0 else 0.0
    
    # Log dataset_ids processed during evaluation (every evaluation)
    print(f"\n[Evaluation] Processed {len(dataset_ids_processed)} unique dataset_ids:")
    
    return {
        'ce': avg_loss_ce,
        # 'expected_rtt': avg_loss_expected_rtt,
        'acc': acc,
        'regret': avg_regret,
        'predicted_rtt': avg_predicted_rtt,
        'optimal_rtt': avg_optimal_rtt,
        'n_rtt_evaluations': n_rtt_evaluations,
        'dataset_ids': sorted(dataset_ids_processed)
    }


# %%
# ========================================================================
# Load all datasets
# ========================================================================
all_datasets = load_all_datasets(BASE_DIR)

if len(all_datasets) == 0:
    print("ERROR: No datasets loaded!")
    exit(1)

# ========================================================================
# Build graphs for all datasets
# ========================================================================
print("Building graphs...")
graphs = []
dataset_ids = []


# Use tqdm to show progress
graphs_with_summaries = 0
total_graph_summaries = 0
for dataset_id, dataset_dict in tqdm(all_datasets.items(), desc="Building graphs", unit="dataset"):
    try:
        graph = build_graph(
            dataset_dict['nodes'],
            dataset_dict['tasks'],
            dataset_dict['platforms']
        )
        # Prefer the internal key used during loading, fallback to old key name
        placement_summaries = dataset_dict.get('_placement_summaries', dataset_dict.get('placement_summaries', []))
        
        # Store dataset_id as a regular attribute (PyG will preserve it during batching)
        # Using a string representation that can be stored in the Data object
        graph.dataset_id = dataset_id
        
        if len(placement_summaries) > 0:
            graphs_with_summaries += 1
            total_graph_summaries += len(placement_summaries)
        
        # Store in global lookup dictionaries (single source of truth)
        if dataset_id:
            _PLACEMENT_SUMMARIES_BY_DATASET_ID[dataset_id] = placement_summaries
            if hasattr(graph, '_plat_pos_by_id'):
                _PLAT_POS_BY_ID_BY_DATASET_ID[dataset_id] = graph._plat_pos_by_id
                # Store task index -> task ID mapping for this dataset
                if hasattr(graph, '_task_idx_to_task_id'):
                    _TASK_IDX_TO_TASK_ID_BY_DATASET_ID[dataset_id] = graph._task_idx_to_task_id
                # Preprocess placement data for fast lookups (done once per dataset)
                if len(placement_summaries) > 0:
                    preprocess_placement_data(dataset_id, placement_summaries, graph._plat_pos_by_id)

        graphs.append(graph)
        dataset_ids.append(dataset_id)

    except Exception as e:
        tqdm.write(f"  Error building graph for {dataset_id}: {e}")


ys = np.concatenate([g.y.numpy() for g in graphs])
print("Valid labels:", np.sum(ys >= 0), "/", len(ys))
print(sum([g.edge_index.numel() == 0 for g in graphs]), "/", len(graphs))

print("Avg edges:", np.mean([g.edge_index.size(1) for g in graphs]))
print("Avg valid tasks:", np.mean([(g.y >= 0).sum().item() for g in graphs]))


print(f"\nBuilt {len(graphs)} graphs")
print(f"[DEBUG] Graphs with placement summaries: {graphs_with_summaries}/{len(graphs)} (total summaries: {total_graph_summaries})")
print(f"[DEBUG] Global lookup populated: {len(_PLACEMENT_SUMMARIES_BY_DATASET_ID)} dataset_ids\n")

# ========================================================================
# Train/Val/Test Split (80/20)
# ========================================================================
train_graphs, test_graphs, train_ids, test_ids = train_test_split(
    graphs, dataset_ids, test_size=0.2, random_state=42
)

print("Dataset split:")
print(f"  Train: {len(train_graphs)} datasets")
print(f"  Test:  {len(test_graphs)} datasets\n")

# %%
os.environ['WANDB_API_KEY'] = '85cccc04212d62b698dbc4549b87818a95850133'

wandb.init(
    project="scheduling-gnn-rtt-metrics",
    entity="nikolalukic167-tu-wien",
    config={
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "lr": LEARNING_RATE,
        "epochs": EPOCHS,
        "device": DEVICE,
    }
)

# %%
# ========================================================================
# Initialize model
# ========================================================================
# Task features: 2 (task types) + 1 (source node ID) = 3
# Platform features: 5 (platform types) + 2 (replica flags) = 7 (old)
task_feature_dim = 3
platform_feature_dim = 7

model = TaskPlacementGNN(
    task_feature_dim=task_feature_dim,
    platform_feature_dim=platform_feature_dim,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_GIN_LAYERS
).to(DEVICE)

# Weight initialization for training stability
def init_weights(m):
    """Initialize weights using Xavier uniform and zeros for bias."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

model.apply(init_weights)

# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

# ========================================================================
# Training loop
# ========================================================================
print("="*80)
print("TRAINING")
print("="*80)
print()

wandb.watch(model, log="gradients", log_freq=100)  # now that model exists

best_val_acc = 0

# NOTE: Using num_workers=0 to avoid multiprocessing issues
# Placement summaries are now looked up from global dictionaries using dataset_id
train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_graphs,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

for epoch in range(EPOCHS):
    is_last_epoch = (epoch == EPOCHS - 1)
    
    # Train
    train_losses = train_epoch(model, train_loader, optimizer, DEVICE, epoch, is_last_epoch=is_last_epoch)
    
    # Evaluate
    val_metrics = evaluate(model, test_loader, DEVICE, is_last_epoch=is_last_epoch)
    
    # Wandb logging - core metrics focused on placement performance
    log_dict = {
        "epoch": epoch,
        "train/loss_ce": train_losses['ce'],  # Keep for debugging/overfitting detection
        "val/loss_ce": val_metrics['ce'],     # Keep for debugging/overfitting detection
        "val/acc": val_metrics['acc'],         # Classification accuracy (task-platform matching)
        "lr": optimizer.param_groups[0]["lr"],
    }
    
    # Add RTT-based metrics
    if val_metrics.get('n_rtt_evaluations', 0) > 0:
        log_dict.update({
            # Primary metrics: RTT difference
            "val/regret": val_metrics['regret'],                    # Absolute RTT gap (primary metric)
            "val/predicted_rtt": val_metrics['predicted_rtt'],      # Your prediction's RTT
            "val/optimal_rtt": val_metrics['optimal_rtt'],          # Best available RTT
        })
    
    wandb.log(log_dict, step=epoch)
    
    if val_metrics['acc'] > best_val_acc:
        best_val_acc = val_metrics['acc']
        # Save best model
        torch.save(model.state_dict(), 'best_gnn_placement_model.pt')
    
    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train CE: {train_losses['ce']:.4f} | "
              f"Val CE: {val_metrics['ce']:.4f} | "
              f"Val Acc: {val_metrics['acc']*100:.2f}%", end="")
        
        if val_metrics.get('n_rtt_evaluations', 0) > 0:
            print(f" | Regret: {val_metrics['regret']:.2f}s | "
                  f"Predicted RTT: {val_metrics['predicted_rtt']:.2f}s | Optimal RTT: {val_metrics['optimal_rtt']:.2f}s")
        else:
            print()

print()
print(f"Best validation accuracy: {best_val_acc*100:.2f}%")

# ========================================================================
# Final Evaluation
# ========================================================================
print()
print("="*80)
print("FINAL EVALUATION")
print("="*80)

# Load best model
model.load_state_dict(torch.load('best_gnn_placement_model.pt'))

train_loader_eval = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
test_loader_eval  = DataLoader(test_graphs,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

train_metrics = evaluate(model, train_loader_eval, DEVICE, is_last_epoch=True)
test_metrics  = evaluate(model, test_loader_eval,  DEVICE, is_last_epoch=True)

# ========================================================================
# WANDB
# ========================================================================

# Log simple counts
wandb.log({
    "data/num_datasets_total": len(graphs),
    "data/num_train": len(train_graphs),
    "data/num_test":  len(test_graphs),
})

# Optionally: store the list of dataset IDs for traceability
wandb.summary["train_dataset_ids"] = train_ids
wandb.summary["test_dataset_ids"]  = test_ids

wandb.summary["best_val_acc"] = best_val_acc

artifact = wandb.Artifact("placement-gnn", type="model")
artifact.add_file("best_gnn_placement_model.pt")
wandb.log_artifact(artifact)

wandb.finish()

# ========================================================================
# local logging
# ========================================================================

print(f"\nTrain: CE={train_metrics['ce']:.4f}, Acc={train_metrics['acc']*100:.2f}%")
if train_metrics.get('n_rtt_evaluations', 0) > 0:
    print(f"       Regret: {train_metrics['regret']:.2f}s | ")
    print(f"       Predicted RTT: {train_metrics['predicted_rtt']:.2f}s | Optimal RTT: {train_metrics['optimal_rtt']:.2f}s")

print(f"Test:  CE={test_metrics['ce']:.4f}, Acc={test_metrics['acc']*100:.2f}%")
if test_metrics.get('n_rtt_evaluations', 0) > 0:
    print(f"       Regret: {test_metrics['regret']:.2f}s | ")
    print(f"       Predicted RTT: {test_metrics['predicted_rtt']:.2f}s | Optimal RTT: {test_metrics['optimal_rtt']:.2f}s")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"Model saved to: best_gnn_placement_model.pt")
print(f"Best validation accuracy: {best_val_acc*100:.2f}%")