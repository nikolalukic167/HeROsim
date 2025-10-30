# %%
#!/usr/bin/env python3
"""
GNN for Task-to-Platform Placement Prediction
Train a Graph Isomorphism Network (GIN) to predict optimal task placements.
"""

import os
import json
import random
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
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
BASE_DIR = Path("/root/projects/my-herosim/simulation_data/artifacts/run7/gnn_datasets")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters intended for grid search
# 16, 32, 64
EMBEDDING_DIM = 32

# 32, 64, 128
HIDDEN_DIM = 128

# 0.005, 0.001, 0.0005 
LEARNING_RATE = 0.001

# 16, 32
BATCH_SIZE = 16

# 3, 5
NUM_GIN_LAYERS = 3

# don't grid search
WEIGHT_DECAY = 1e-3
EPOCHS = 100

# %%
# ============================================================================
# DATA LOADING (reuse extraction logic)
# ============================================================================

def extract_dataset_to_dataframes_v1(optimal_result_path: Path) -> Dict[str, pd.DataFrame]:
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
    tasks_data = []
    for task_result in task_results:
        task_id = task_result.get("taskId")
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


def load_all_datasets_v1(base_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load all datasets from gnn_datasets directory."""
    all_datasets = {}
    dataset_dirs = sorted(base_dir.glob("ds_*"))
    
    print(f"Loading {len(dataset_dirs)} datasets...")
    
    for dataset_dir in dataset_dirs:
        optimal_result_path = dataset_dir / "optimal_result.json"
        if not optimal_result_path.exists():
            continue
        
        try:
            dataframes = extract_dataset_to_dataframes_v1(optimal_result_path)
            all_datasets[dataset_dir.name] = dataframes
        except Exception as e:
            print(f"  Error loading {dataset_dir.name}: {e}")
    
    print(f"Loaded {len(all_datasets)} datasets successfully\n")
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
      - Edges: task -> compatible platforms that are:
               1) Network feasible (source node + its network neighbors)
               2) Compatible platform type (from TASK_PLATFORM_COMPATIBILITY)
               3) Have replica for the task type (has_dnn1_replica for dnn1, has_dnn2_replica for dnn2)
      - Reverse edges are added (undirected graph for GIN)
      - Labels y: for each task, index of optimal platform within *its own* compatible list; -1 if no compatible platforms
    """

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
            replica_mask = has_dnn1_arr
        elif task_type == 'dnn2':
            type_mask = plat_type_compat_dnn2
            replica_mask = has_dnn2_arr
        else:
            # Unknown task type -> no compatible platforms
            return np.empty(0, dtype=np.int64)
        
        # Vectorized filtering: platform type compatible AND has replica
        # Apply mask to the network-feasible platforms only
        compatible_mask = type_mask[network_feasible_plats] & replica_mask[network_feasible_plats]
        
        return network_feasible_plats[compatible_mask]

    # ---------------------------------------------
    # EDGES + LABELS
    # ---------------------------------------------
    edge_src, edge_dst = [], []
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
            edge_dst.extend((platform_offset + compat_plats).tolist())

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
        # Add reverse edges: make undirected for GIN message passing
        num_nodes = n_tasks + n_platforms
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    y = torch.tensor(y_list, dtype=torch.long)

    # ---------------------------------------------
    # Create PyG Data
    # ---------------------------------------------
    data = Data(
        edge_index=edge_index,
        y=y,
        n_tasks=n_tasks,
        n_platforms=n_platforms,
        task_features=task_features_tensor,             # dim=3
        platform_features=platform_features_tensor,     # dim=7
    )
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
    """2-layer MLP to score task-platform edges."""
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        # Input: concatenation of task and platform embeddings
        self.fc1 = nn.Linear(2 * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, e_task, e_platform):
        # Concatenate task and platform embeddings
        # x = torch.cat([e_task, e_platform], dim=-1)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        # return x.squeeze(-1)
        # e_task: (E, D) or (1, D)
        # e_platform: (E, D) or (1, D)
        x = torch.cat([e_task, e_platform], dim=-1)  # (E, 2D)
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
        
        # Edge scoring MLP
        self.edge_scorer = EdgeScorer(embedding_dim, hidden_dim)

    def forward(self, data):
        n_tasks = data.n_tasks
        n_platforms = data.n_platforms

        # 1) Encode features
        task_embeddings = self.task_encoder(data.task_features)        # (T, D)
        platform_embeddings = self.platform_encoder(data.platform_features)  # (P, D)

        # 2) Message passing on concatenated nodes
        x = torch.cat([task_embeddings, platform_embeddings], dim=0)   # (T+P, D)
        x = self.gin(x, data.edge_index)

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
        edge_scores = self.edge_scorer(e_task, e_platform)   # (E_valid,)

        # 4) Split scores per task
        logits_per_task = []
        for t in range(n_tasks):
            mask_t = (ti == t)
            logits_t = edge_scores[mask_t]   # (K_t,)
            logits_per_task.append(logits_t)

        return logits_per_task

"""
    def forward(self, data):
        n_tasks = data.n_tasks
        n_platforms = data.n_platforms
        
        # Encode task and platform features separately (they have different input dims)
        task_embeddings = self.task_encoder(data.task_features)
        platform_embeddings = self.platform_encoder(data.platform_features)
        
        # Combine for GIN (now both have same embedding_dim)
        x = torch.cat([task_embeddings, platform_embeddings], dim=0)
        
        # Apply GIN for message passing
        x = self.gin(x, data.edge_index)
        
        # Split back into task and platform embeddings
        task_emb = x[:n_tasks]
        platform_emb = x[n_tasks:]
        
        # Score all edges
        edge_scores = []
        for i in range(data.edge_index.size(1)):
            task_idx = int(data.edge_index[0, i])
            platform_idx = int(data.edge_index[1, i]) - n_tasks
            if 0 <= platform_idx < n_platforms:
                t = task_emb[task_idx].unsqueeze(0)      # (1, D)
                p = platform_emb[platform_idx].unsqueeze(0)  # (1, D)
                score = self.edge_scorer(t, p)           # returns (1,)
                edge_scores.append(score.squeeze(0))     # scalar tensor
        edge_scores = (torch.stack(edge_scores)
                    if len(edge_scores) > 0
                    else torch.empty(0, device=task_emb.device))
        
        edge_scores = torch.stack(edge_scores)
        
        # Compute masked softmax per task
        # For each task, normalize over its feasible platforms
        logits_per_task = []
        
        for task_idx in range(n_tasks):
            task_edges_mask = (data.edge_index[0] == task_idx)
            task_edge_scores = edge_scores[task_edges_mask]
            
            # Softmax over this task's feasible platforms
            task_probs = F.softmax(task_edge_scores, dim=0)
            logits_per_task.append(task_edge_scores)
        
        return logits_per_task
"""

# %%
# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, train_loader, optimizer, device, epoch_num):
    model.train()
    # loss accross all graphs in the batch
    running = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch_num:3d} [Train]", leave=False):
        optimizer.zero_grad() # reset gradients
        graphs_in_batch = batch.to_data_list()

        loss_total = torch.zeros(1, device=device)
        n_graphs = 0

        for data in graphs_in_batch:
            data = data.to(device)
            logits_per_task = model(data)

            loss_g = torch.zeros(1, device=device)
            valid_tasks = 0

            for task_idx, logits_t in enumerate(logits_per_task):
                if logits_t.numel() == 0:
                    continue  # no feasible platforms

                logits = logits_t.unsqueeze(0)  # (1, K)
                target = data.y[task_idx].long()
                if target.ndim == 0:
                    target = target.unsqueeze(0)  # (1,)

                if target.item() < 0 or target.item() >= logits.size(1):
                    continue  # label not in feasible set

                loss_g = loss_g + F.cross_entropy(logits, target)
                valid_tasks += 1

            if valid_tasks == 0:
                continue  # skip this graph for loss/step

            loss_g = loss_g / valid_tasks
            loss_total = loss_total + loss_g
            n_graphs += 1

        if n_graphs == 0:
            # nothing usable in this batch; skip backward to avoid NaNs
            continue

        loss = loss_total / n_graphs

        # backpropagate loss
        loss.backward() # compute gradients 

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step() # update weights

        running += loss.item()

    return running / max(1, len(train_loader))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        for data in batch.to_data_list():
            data = data.to(device)
            logits_per_task = model(data)

            for task_idx, task_logits in enumerate(logits_per_task):
                if task_logits.numel() == 0:
                    continue

                logits = task_logits.unsqueeze(0)        # (1, K)

                target = data.y[task_idx].long()
                if target.ndim == 0:
                    target = target.unsqueeze(0)         # (1,)
                # TODO: skip invalid labels (-1) and out-of-range labels 
                if target.item() < 0 or target.item() >= logits.size(1):
                    continue

                total_loss += F.cross_entropy(logits, target).item()
                pred = logits.argmax(dim=1).item()       # int
                correct += int(pred == target.item())
                total += 1

    avg_loss = total_loss / total if total else 0.0
    acc = correct / total if total else 0.0
    
    return avg_loss, acc

# %%
# ========================================================================
# Load all datasets
# ========================================================================
all_datasets = load_all_datasets_v1(BASE_DIR)

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
for dataset_id, dataframes in tqdm(all_datasets.items(), desc="Building graphs", unit="dataset"):
    try:
        graph = build_graph(
            dataframes['nodes'],
            dataframes['tasks'],
            dataframes['platforms']
        )
        graphs.append(graph)
        dataset_ids.append(dataset_id)
    except Exception as e:
        tqdm.write(f"  Error building graph for {dataset_id}: {e}")

print(f"\nBuilt {len(graphs)} graphs\n")

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
    project="Scheduling-GNN",
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

train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_graphs,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

for epoch in range(EPOCHS):
    # Train
    train_loss = train_epoch(model, train_loader, optimizer, DEVICE, epoch)
    
    # Evaluate
    val_loss, val_acc = evaluate(model, test_loader, DEVICE)
    
    # Wandb logging
    wandb.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "val/loss":   val_loss,
        "val/acc":    val_acc,
        "lr":         optimizer.param_groups[0]["lr"],
    }, step=epoch)

    #if epoch % 5 == 0:
      #  for name, p in model.named_parameters():
        #    wandb.log({f"hist/{name}": wandb.Histogram(p.detach().cpu().numpy())})
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # Save best model
        torch.save(model.state_dict(), 'best_gnn_placement_model.pt')
    
    # Print progress
    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

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

train_loader_eval = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader_eval  = DataLoader(test_graphs,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

train_loss, train_acc = evaluate(model, train_loader_eval, DEVICE)
test_loss,  test_acc  = evaluate(model, test_loader_eval,  DEVICE)

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
wandb.finish()

# ========================================================================
# local logging
# ========================================================================

print(f"\nTrain: Loss={train_loss:.4f}, Accuracy={train_acc*100:.2f}%")
print(f"Test:  Loss={test_loss:.4f}, Accuracy={test_acc*100:.2f}%")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"Model saved to: best_gnn_placement_model.pt")
print(f"Best validation accuracy: {best_val_acc*100:.2f}%")

# %%
artifact = wandb.Artifact("placement-gnn", type="model")
artifact.add_file("best_gnn_placement_model.pt")
# with open("splits.json","w") as f: json.dump({"train": train_ids, "test": test_ids}, f)
# artifact.add_file("splits.json")
wandb.log_artifact(artifact)


