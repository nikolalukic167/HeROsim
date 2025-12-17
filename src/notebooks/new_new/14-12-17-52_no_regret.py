# %%
#!/usr/bin/env python3
"""
GNN for Task-to-Platform Placement Prediction
Train a Graph Isomorphism Network (GIN) to predict optimal task placements.
"""

import os
import json
import pickle
import random
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict
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
CACHE_DIR = Path("/root/projects/my-herosim/simulation_data/artifacts/run1100/graphs_cache_old")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cache file paths
GRAPHS_CACHE_PATH = CACHE_DIR / "graphs.pkl"
DATASET_IDS_CACHE_PATH = CACHE_DIR / "dataset_ids.pkl"

# Hyperparameters intended for grid search
# 16, 32, 64
EMBEDDING_DIM = 64

# 32, 64, 128
HIDDEN_DIM = 64

# 0.005, 0.001, 0.0005 
LEARNING_RATE = 0.001

# 16, 32
BATCH_SIZE = 16

# 3, 4
NUM_GIN_LAYERS = 3

# don't grid search
WEIGHT_DECAY = 1e-3
EPOCHS = 300

# %%
# ============================================================================
# CACHE LOADING
# ============================================================================

def load_graphs_from_cache() -> Tuple[List[Data], List[str]]:
    """Load graphs and dataset IDs from cache."""
    if not GRAPHS_CACHE_PATH.exists():
        raise FileNotFoundError(f"Graphs cache not found at {GRAPHS_CACHE_PATH}. Run prepare_graphs_cache.py first.")
    if not DATASET_IDS_CACHE_PATH.exists():
        raise FileNotFoundError(f"Dataset IDs cache not found at {DATASET_IDS_CACHE_PATH}. Run prepare_graphs_cache.py first.")
    
    print(f"Loading graphs from cache: {GRAPHS_CACHE_PATH}")
    with open(GRAPHS_CACHE_PATH, 'rb') as f:
        graphs = pickle.load(f)
    
    print(f"Loading dataset IDs from cache: {DATASET_IDS_CACHE_PATH}")
    with open(DATASET_IDS_CACHE_PATH, 'rb') as f:
        dataset_ids = pickle.load(f)
    
    print(f"Loaded {len(graphs)} graphs with {len(dataset_ids)} dataset IDs")
    return graphs, dataset_ids








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
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class PlatformEncoder(nn.Module):
    """2-layer MLP encoder for platform features with LayerNorm for training stability."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x



class EdgeScorer(nn.Module):
    """2-layer MLP to score task-platform edges with optional edge attributes."""
    def __init__(self, embedding_dim, hidden_dim, edge_dim=0):
        super().__init__()
        # Input: concatenation of task, platform embeddings and edge attrs
        in_dim = 2 * embedding_dim + (edge_dim if edge_dim else 0)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, e_task, e_platform, e_attr=None):
        # Concatenate task and platform embeddings (+ edge attrs if provided)
        x = torch.cat([e_task, e_platform] + ([e_attr] if e_attr is not None else []), dim=-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
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
        self.post_gin_dropout = nn.Dropout(p=0.2)
        
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
        x = self.post_gin_dropout(x)
        
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


# %%
# ============================================================================
# CUSTOM COLLATE FUNCTION (preserves custom attributes during batching)
# ============================================================================

def custom_collate_fn(batch):
    """
    Custom collate function that preserves _plat_pos_by_id, _task_idx_to_task_id, and dataset_id
    attributes during batching.
    """
    import sys
    print(f"[DEBUG collate] custom_collate_fn called with {len(batch)} graphs", file=sys.stderr, flush=True)
    
    from torch_geometric.data import Batch
    
    # Extract custom attributes before batching
    custom_attrs = []
    for i, data in enumerate(batch):
        attrs = {}
        if hasattr(data, '_plat_pos_by_id'):
            attrs['_plat_pos_by_id'] = data._plat_pos_by_id
        if hasattr(data, '_task_idx_to_task_id'):
            attrs['_task_idx_to_task_id'] = data._task_idx_to_task_id
        if hasattr(data, 'dataset_id'):
            attrs['dataset_id'] = data.dataset_id
        custom_attrs.append(attrs)
        
        # Debug: Check first graph in batch
        if i == 0:
            plat_pos_size = len(getattr(data, '_plat_pos_by_id', {}))
            print(f"[DEBUG collate] First graph in batch: _plat_pos_by_id size={plat_pos_size}", file=sys.stderr, flush=True)
    
    # Create the batch using PyG's default collate
    batch_obj = Batch.from_data_list(batch)
    
    # Store custom attributes in the batch (will be restored after unbatching)
    batch_obj._custom_attrs = custom_attrs
    
    # Debug: Check if custom_attrs were stored
    if len(custom_attrs) > 0:
        first_attrs_size = len(custom_attrs[0].get('_plat_pos_by_id', {}))
        print(f"[DEBUG collate] Stored custom_attrs: first graph _plat_pos_by_id size={first_attrs_size}", file=sys.stderr, flush=True)
        print(f"[DEBUG collate] batch_obj has _custom_attrs: {hasattr(batch_obj, '_custom_attrs')}", file=sys.stderr, flush=True)
    
    return batch_obj


def restore_custom_attrs(batch, graphs):
    """
    Restore custom attributes from global dictionary using dataset_id.
    PyG's DataLoader doesn't preserve custom attributes, so we use a global dict.
    """
    global GRAPH_CUSTOM_ATTRS
    
    # Restore custom attributes from global dictionary using dataset_id
    for graph in graphs:
        dataset_id = getattr(graph, 'dataset_id', None)
        if dataset_id and dataset_id in GRAPH_CUSTOM_ATTRS:
            attrs = GRAPH_CUSTOM_ATTRS[dataset_id]
            graph._plat_pos_by_id = attrs.get('_plat_pos_by_id', {})
            graph._plat_id_to_node_id = attrs.get('_plat_id_to_node_id', {})
            graph._task_idx_to_task_id = attrs.get('_task_idx_to_task_id', {})
            graph.dataset_id = attrs.get('dataset_id', dataset_id)
    
    return graphs


# %%
# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, train_loader, optimizer, device, epoch_num, is_last_epoch=False):
    model.train()
    # loss accross all graphs in the batch
    running_ce = 0.0
    n_steps = 0
    
    # Track dataset_ids processed in this epoch
    dataset_ids_processed = set()

    for batch in tqdm(train_loader, desc=f"Epoch {epoch_num:3d} [Train]", leave=is_last_epoch):
        optimizer.zero_grad() # reset gradients
        graphs_in_batch = batch.to_data_list()
        
        # Restore custom attributes that were lost during batching
        graphs_in_batch = restore_custom_attrs(batch, graphs_in_batch)

        loss_ce_total = torch.zeros(1, device=device)
        n_graphs = 0

        for data in graphs_in_batch:
            data = data.to(device)

            dataset_id = getattr(data, 'dataset_id', None)
            if dataset_id:
                dataset_ids_processed.add(dataset_id)
                        
            logits_per_task = model(data)

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
        n_steps += 1
    
    # Log dataset_ids processed in this epoch (every epoch)
    print(f"\n[Epoch {epoch_num}] Processed {len(dataset_ids_processed)} unique dataset_ids:")

    return {
        'ce': running_ce / max(1, n_steps),
        'dataset_ids': sorted(dataset_ids_processed)
    }

@torch.no_grad()
def decode_predicted_placement(logits_per_task, data):
    """
    Decode model predictions into a placement plan.
    Returns: Dict[int, List[int, int]] mapping task_id -> [node_id, platform_id]
    Uses actual task IDs from JSON, not task indices.
    
    Args:
        logits_per_task: List of logits tensors, one per task
        data: Graph data object
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
                predicted_placement[actual_task_id] = [None, plat_id]
    
    return predicted_placement




@torch.no_grad()
def evaluate(model, loader, device, is_last_epoch=False):
    model.eval()
    total_loss_ce = 0.0
    total_valid_tasks_for_loss = 0  # Total valid tasks for loss calculation
    correct_graphs = 0  # Count graphs where ALL tasks are correct
    total_graphs = 0    # Total number of graphs evaluated
    total_tasks_correct = 0  # For debugging: total tasks correct across all graphs
    total_tasks = 0     # For debugging: total tasks across all graphs
    n_graphs = 0
    dataset_ids_processed = set()

    for batch in tqdm(loader, desc="Evaluating", leave=is_last_epoch):
        graphs_in_batch = batch.to_data_list()
        
        # Restore custom attributes that were lost during batching
        graphs_in_batch = restore_custom_attrs(batch, graphs_in_batch)
        
        for data in graphs_in_batch:
            # Preserve custom attributes before device transfer
            plat_pos_by_id_orig = getattr(data, '_plat_pos_by_id', {})
            task_idx_to_task_id_orig = getattr(data, '_task_idx_to_task_id', {})
            dataset_id_orig = getattr(data, 'dataset_id', None)
            
            data = data.to(device)
            
            # Restore custom attributes after device transfer (PyG might not preserve them)
            if plat_pos_by_id_orig:
                data._plat_pos_by_id = plat_pos_by_id_orig
            if task_idx_to_task_id_orig:
                data._task_idx_to_task_id = task_idx_to_task_id_orig
            if dataset_id_orig:
                data.dataset_id = dataset_id_orig

            dataset_id = getattr(data, 'dataset_id', None)
            if dataset_id:
                dataset_ids_processed.add(dataset_id)
            
            logits_per_task = model(data)

            # Original CE loss
            loss_ce, valid_ce = loss_original_ce(logits_per_task, data, device)
            if valid_ce > 0:
                total_loss_ce += loss_ce.item() * valid_ce
                total_valid_tasks_for_loss += valid_ce
                n_graphs += 1
                total_graphs += 1

                # Compute accuracy: graph is correct only if ALL tasks are correct
                graph_all_correct = True
                graph_valid_tasks = 0
                
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
                    is_correct = int(pred == target.item())
                    total_tasks_correct += is_correct
                    total_tasks += 1
                    graph_valid_tasks += 1
                    
                    # If any task is wrong, the graph is not fully correct
                    if not is_correct:
                        graph_all_correct = False
                
                # Only count graph as correct if ALL valid tasks are correct AND exactly 5 tasks
                # (per simulation logic: exactly 5 tasks need to be placed)
                if graph_all_correct and graph_valid_tasks == 5:
                    correct_graphs += 1

    # Calculate average loss per task (for backward compatibility)
    avg_loss_ce = total_loss_ce / total_valid_tasks_for_loss if total_valid_tasks_for_loss > 0 else 0.0
    
    # Accuracy: fraction of graphs where ALL tasks are correctly placed
    acc = correct_graphs / total_graphs if total_graphs > 0 else 0.0
    
    # Log dataset_ids processed during evaluation (every evaluation)
    print(f"\n[Evaluation] Processed {len(dataset_ids_processed)} unique dataset_ids:")
    
    # Debug: Summary of accuracy calculation
    print(f"[DEBUG] Accuracy calculation summary:")
    print(f"  Total graphs processed: {total_graphs}")
    print(f"  Graphs with ALL tasks correct: {correct_graphs} ({correct_graphs/total_graphs*100:.1f}%)")
    print(f"  Total tasks processed: {total_tasks}")
    print(f"  Total tasks correct: {total_tasks_correct} ({total_tasks_correct/total_tasks*100:.1f}% per-task accuracy)")
    
    task_acc = total_tasks_correct / total_tasks if total_tasks > 0 else 0.0
    
    return {
        'ce': avg_loss_ce,
        'acc': acc,
        'task_acc': task_acc,
        'dataset_ids': sorted(dataset_ids_processed),
    }


# %%
# ========================================================================
# Load graphs from cache
# ========================================================================
graphs, dataset_ids = load_graphs_from_cache()

if len(graphs) == 0:
    print("ERROR: No graphs loaded from cache!")
    exit(1)

# Compute statistics
ys = np.concatenate([g.y.numpy() for g in graphs])
print("Valid labels:", np.sum(ys >= 0), "/", len(ys))
print(sum([g.edge_index.numel() == 0 for g in graphs]), "/", len(graphs))

print("Avg edges:", np.mean([g.edge_index.size(1) for g in graphs]))
print("Avg valid tasks:", np.mean([(g.y >= 0).sum().item() for g in graphs]))


print(f"\nLoaded {len(graphs)} graphs from cache")

# Store custom attributes in global dictionary keyed by dataset_id
# This is needed because PyG's DataLoader doesn't preserve custom attributes during batching
GRAPH_CUSTOM_ATTRS = {}  # dataset_id -> {_plat_pos_by_id, _task_idx_to_task_id, dataset_id}

for graph in graphs:
    dataset_id = getattr(graph, 'dataset_id', None)
    if dataset_id:
        GRAPH_CUSTOM_ATTRS[dataset_id] = {
            '_plat_pos_by_id': getattr(graph, '_plat_pos_by_id', {}),
            '_task_idx_to_task_id': getattr(graph, '_task_idx_to_task_id', {}),
            'dataset_id': dataset_id
        }

# ========================================================================
# Train/Val/Test Split (80/10/10)
# ========================================================================
# First split: 80% train, 20% temp (val+test)
train_graphs, temp_graphs, train_ids, temp_ids = train_test_split(
    graphs, dataset_ids, test_size=0.2, random_state=42
)

# Second split: split temp (20%) into 50% val and 50% test (10% each overall)
val_graphs, test_graphs, val_ids, test_ids = train_test_split(
    temp_graphs, temp_ids, test_size=0.5, random_state=42
)

print("Dataset split:")
print(f"  Train: {len(train_graphs)} datasets ({len(train_graphs)/len(graphs)*100:.1f}%)")
print(f"  Val:   {len(val_graphs)} datasets ({len(val_graphs)/len(graphs)*100:.1f}%)")
print(f"  Test:  {len(test_graphs)} datasets ({len(test_graphs)/len(graphs)*100:.1f}%)\n")

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
# Note: PyG's DataLoader doesn't preserve custom attributes during batching,
# so we restore them from GRAPH_CUSTOM_ATTRS dictionary after unbatching
train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_graphs,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_graphs,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

for epoch in range(EPOCHS):
    is_last_epoch = (epoch == EPOCHS - 1)
    
    # Train
    train_losses = train_epoch(model, train_loader, optimizer, DEVICE, epoch, is_last_epoch=is_last_epoch)
    
    # Evaluate on validation set
    val_metrics = evaluate(model, val_loader, DEVICE, is_last_epoch=is_last_epoch)
    
    # Wandb logging - core metrics
    log_dict = {
        "epoch": epoch,
        "train/loss_ce": train_losses['ce'],
        "val/loss_ce": val_metrics['ce'],
        "val/acc": val_metrics['acc'],              # Graphs with ALL tasks correct
        "val/task_acc": val_metrics['task_acc'],    # Per-task accuracy
        "lr": optimizer.param_groups[0]["lr"],
    }
    
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
val_loader_eval   = DataLoader(val_graphs,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
test_loader_eval  = DataLoader(test_graphs,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

train_metrics = evaluate(model, train_loader_eval, DEVICE, is_last_epoch=True)
val_metrics_final = evaluate(model, val_loader_eval, DEVICE, is_last_epoch=True)
test_metrics  = evaluate(model, test_loader_eval,  DEVICE, is_last_epoch=True)

# ========================================================================
# WANDB
# ========================================================================

# Log simple counts
wandb.log({
    "data/num_datasets_total": len(graphs),
    "data/num_train": len(train_graphs),
    "data/num_val": len(val_graphs),
    "data/num_test":  len(test_graphs),
})

# Log final evaluation metrics
wandb.log({
    "final/train/loss_ce": train_metrics['ce'],
    "final/train/acc": train_metrics['acc'],
    "final/train/task_acc": train_metrics['task_acc'],
    "final/val/loss_ce": val_metrics_final['ce'],
    "final/val/acc": val_metrics_final['acc'],
    "final/val/task_acc": val_metrics_final['task_acc'],
    "final/test/loss_ce": test_metrics['ce'],
    "final/test/acc": test_metrics['acc'],
    "final/test/task_acc": test_metrics['task_acc'],
})

# Optionally: store the list of dataset IDs for traceability
wandb.summary["train_dataset_ids"] = train_ids
wandb.summary["val_dataset_ids"] = val_ids
wandb.summary["test_dataset_ids"]  = test_ids

wandb.summary["best_val_acc"] = best_val_acc
wandb.summary["final_test_acc"] = test_metrics['acc']
wandb.summary["final_test_task_acc"] = test_metrics['task_acc']

artifact = wandb.Artifact("placement-gnn", type="model")
artifact.add_file("best_gnn_placement_model.pt")
wandb.log_artifact(artifact)

wandb.finish()

# ========================================================================
# local logging
# ========================================================================

print(f"\nTrain: CE={train_metrics['ce']:.4f}, Acc={train_metrics['acc']*100:.2f}%")
print(f"Val:   CE={val_metrics_final['ce']:.4f}, Acc={val_metrics_final['acc']*100:.2f}%")
print(f"Test:  CE={test_metrics['ce']:.4f}, Acc={test_metrics['acc']*100:.2f}%")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"Model saved to: best_gnn_placement_model.pt")
print(f"Best validation accuracy: {best_val_acc*100:.2f}%")