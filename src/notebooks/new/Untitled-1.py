# %%
#!/usr/bin/env python3
"""
GNN for Task-to-Platform Placement Prediction
Train a Graph Isomorphism Network (GIN) to predict optimal task placements.
"""

import json
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
from torch_geometric.nn import GIN
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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


def load_all_datasets(base_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load all datasets from gnn_datasets directory."""
    all_datasets = {}
    dataset_dirs = sorted(base_dir.glob("ds_*"))
    
    print(f"Loading {len(dataset_dirs)} datasets...")
    
    for dataset_dir in dataset_dirs:
        optimal_result_path = dataset_dir / "optimal_result.json"
        if not optimal_result_path.exists():
            continue
        
        try:
            dataframes = extract_dataset_to_dataframes(optimal_result_path)
            all_datasets[dataset_dir.name] = dataframes
        except Exception as e:
            print(f"  Error loading {dataset_dir.name}: {e}")
    
    print(f"Loaded {len(all_datasets)} datasets successfully\n")
    return all_datasets

# %%
# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_graph(df_nodes, df_tasks, df_platforms) -> Data:
    """
    Build a bipartite graph with tasks and platforms as nodes.
    Edges connect tasks to feasible platforms based on network connectivity.
    
    TODO: Add task-platform compatibility filtering (e.g., dnn1 can only run on certain platform types)
    """
    
    # Create node ID mappings
    # Task nodes: 0 to n_tasks-1
    # Platform nodes: n_tasks to n_tasks+n_platforms-1
    n_tasks = len(df_tasks)
    n_platforms = len(df_platforms)
    
    task_offset = 0
    platform_offset = n_tasks
    
    # ========================================================================
    # TASK FEATURES: [task_type_onehot, source_node_id]
    # ========================================================================
    task_types = ['dnn1', 'dnn2']
    task_features = []
    
    for _, task in df_tasks.iterrows():
        # One-hot encode task type
        task_type_onehot = [1.0 if task['task_type'] == tt else 0.0 for tt in task_types]
        
        # Source node ID (normalized to [0, 1])
        source_node_id = df_nodes[df_nodes['node_name'] == task['source_node']].index[0] if len(df_nodes[df_nodes['node_name'] == task['source_node']]) > 0 else 0
        source_node_id_norm = source_node_id / len(df_nodes)
        
        task_features.append(task_type_onehot + [source_node_id_norm])
    
    task_features = torch.tensor(task_features, dtype=torch.float)
    
    # ========================================================================
    # PLATFORM FEATURES: [platform_type_onehot, has_dnn1_replica, has_dnn2_replica]
    # ========================================================================
    platform_types = ['rpiCpu', 'xavierCpu', 'xavierGpu', 'xavierDla', 'pynqFpga']
    platform_features = []
    
    for _, platform in df_platforms.iterrows():
        # One-hot encode platform type
        plat_type_onehot = [1.0 if platform['platform_type'] == pt else 0.0 for pt in platform_types]
        
        # Replica flags
        has_dnn1 = 1.0 if platform['has_dnn1_replica'] else 0.0
        has_dnn2 = 1.0 if platform['has_dnn2_replica'] else 0.0
        
        platform_features.append(plat_type_onehot + [has_dnn1, has_dnn2])
    
    platform_features = torch.tensor(platform_features, dtype=torch.float)
    
    # Store raw features separately (they have different dimensions)
    # Tasks: dim=3, Platforms: dim=7
    # We'll encode them separately in the model, so don't concatenate here
    task_features_tensor = task_features
    platform_features_tensor = platform_features
    
    # ========================================================================
    # EDGES: Task → Feasible Platforms (based on network connectivity)
    # ========================================================================
    edge_index = []
    edge_labels = []  # For tracking which edges are optimal
    
    for task_idx, task in df_tasks.iterrows():
        source_node_name = task['source_node']
        task_type = task['task_type']
        
        # Get source node's network map
        source_node_row = df_nodes[df_nodes['node_name'] == source_node_name]
        if len(source_node_row) == 0:
            continue
        
        network_map = source_node_row.iloc[0]['network_map']
        
        # Feasible nodes: source node + connected server nodes
        feasible_node_names = [source_node_name] + list(network_map.keys())
        
        # Find all platforms on feasible nodes
        # TODO: Add compatibility filtering here (task_type → supported platform types)
        feasible_platforms = df_platforms[df_platforms['node_name'].isin(feasible_node_names)]
        
        # Create edges from task to each feasible platform
        task_node_idx = task_offset + task_idx
        
        for _, platform in feasible_platforms.iterrows():
            platform_idx = platform_offset + platform.name  # platform.name is the DataFrame index
            
            edge_index.append([task_node_idx, platform_idx])
            
            # Mark if this is the optimal edge (for ground truth)
            is_optimal = (platform['platform_id'] == task['optimal_platform_id'])
            edge_labels.append(is_optimal)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # ========================================================================
    # LABELS: For each task, which platform is optimal?
    # ========================================================================
    # We need to convert optimal_platform_id to the index in our edge list
    y = []
    
    for task_idx, task in df_tasks.iterrows():
        task_node_idx = task_offset + task_idx
        optimal_platform_id = task['optimal_platform_id']
        
        # Find which edge index corresponds to this task's optimal platform
        task_edges_mask = (edge_index[0] == task_node_idx)
        task_edge_indices = torch.where(task_edges_mask)[0]
        
        # Find the optimal edge among this task's edges
        optimal_edge_idx = None
        for edge_idx in task_edge_indices:
            platform_node_idx = edge_index[1, edge_idx].item()
            platform_df_idx = platform_node_idx - platform_offset
            
            if platform_df_idx >= 0 and platform_df_idx < len(df_platforms):
                platform_id = df_platforms.iloc[platform_df_idx]['platform_id']
                if platform_id == optimal_platform_id:
                    # Label is the index within this task's feasible platforms
                    optimal_edge_idx = (task_edge_indices == edge_idx).nonzero(as_tuple=True)[0].item()
                    break
        
        y.append(optimal_edge_idx if optimal_edge_idx is not None else 0)
    
    y = torch.tensor(y, dtype=torch.long)
    
    # ========================================================================
    # CREATE PyG Data object
    # ========================================================================
    data = Data(
        edge_index=edge_index,
        y=y,
        n_tasks=n_tasks,
        n_platforms=n_platforms,
        # Store task and platform features separately (different dimensions)
        task_features=task_features_tensor,
        platform_features=platform_features_tensor
    )
    
    return data

# %%
# ============================================================================
# GNN MODEL
# ============================================================================

class TaskEncoder(nn.Module):
    """2-layer MLP encoder for task features."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PlatformEncoder(nn.Module):
    """2-layer MLP encoder for platform features."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
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
    GNN for task-to-platform placement prediction.
    
    Architecture:
    1. Encode task and platform features separately
    2. GIN to produce node embeddings
    3. Edge MLP to score task-platform compatibility
    4. Masked softmax to predict placement probabilities
    """
    def __init__(self, task_feature_dim, platform_feature_dim, embedding_dim=64, hidden_dim=128):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Separate encoders for tasks and platforms
        self.task_encoder = TaskEncoder(task_feature_dim, hidden_dim, embedding_dim)
        self.platform_encoder = PlatformEncoder(platform_feature_dim, hidden_dim, embedding_dim)
        
        # GIN model for message passing
        self.gin = GIN(
            in_channels=embedding_dim,
            hidden_channels=hidden_dim,
            num_layers=3,
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
    running = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch_num:3d} [Train]", leave=False):
        optimizer.zero_grad()
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
        loss.backward()
        optimizer.step()

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
# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

print("="*80)
print("GNN TASK PLACEMENT TRAINING")
print("="*80)
print()

# Configuration
BASE_DIR = Path("/root/projects/my-herosim/simulation_data/gnn_datasets")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
EMBEDDING_DIM = 64 # 64
HIDDEN_DIM = 128 # 128
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 16

print(f"Device: {DEVICE}")
print(f"Embedding dim: {EMBEDDING_DIM}")
print(f"Hidden dim: {HIDDEN_DIM}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Epochs: {EPOCHS}")
print()

# %%
from tqdm import tqdm

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
# DEBUG = False
# if DEBUG:
#     print("DEBUG MODE")
#     EPOCHS = 5
#     train_graphs = train_graphs[:5]

# %%
#import wandb

# Safely end the current run if any (no-op if none)
#wandb.finish()          

# Extra cleanup: close sockets/threads and reset global state
#wandb.teardown()

# %%
import os
import wandb
os.environ['WANDB_API_KEY'] = '85cccc04212d62b698dbc4549b87818a95850133'

# --- wandb init ---
# Prefer env var WANDB_API_KEY set outside the script (e.g., export WANDB_API_KEY=...).
# If you MUST do it in code (not recommended), use wandb.login(key=...)
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
from torch_geometric.loader import DataLoader

# ========================================================================
# Initialize model
# ========================================================================
# Task features: 2 (task types) + 1 (source node ID) = 3
# Platform features: 5 (platform types) + 2 (replica flags) = 7
task_feature_dim = 3
platform_feature_dim = 7

model = TaskPlacementGNN(
    task_feature_dim=task_feature_dim,
    platform_feature_dim=platform_feature_dim,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
# ============================================================================
# EXPORT AND ANALYZE WANDB RUNS
# ============================================================================

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize wandb API
api = wandb.Api()

# Fetch all runs from the project
entity = "nikolalukic167-tu-wien"
project = "Scheduling-GNN"
runs = api.runs(f"{entity}/{project}")

print(f"Found {len(runs)} runs in project {project}\n")

# Extract run data
runs_data = []
for run in runs:
    # Get summary metrics - handle case where it might be a string
    summary = run.summary._json_dict
    if not isinstance(summary, dict):
        summary = {}
    
    # Handle run.config which might be a string or dict
    if isinstance(run.config, dict):
        config = {k: v for k, v in run.config.items() if not k.startswith('_')}
    else:
        config = {}
    
    runs_data.append({
        'run_id': run.id,
        'run_name': run.name,
        'state': run.state,
        'created_at': run.created_at,
        'best_val_acc': summary.get('best_val_acc', None),
        'final_val_acc': summary.get('val/acc', None),
        'final_val_loss': summary.get('val/loss', None),
        'final_train_loss': summary.get('train/loss', None),
        'embedding_dim': config.get('embedding_dim', None),
        'hidden_dim': config.get('hidden_dim', None),
        'lr': config.get('lr', None),
        'epochs': config.get('epochs', None),
        'device': config.get('device', None),
        'num_train': summary.get('data/num_train', None),
        'num_test': summary.get('data/num_test', None),
    })

# Create DataFrame
df_runs = pd.DataFrame(runs_data)

# Sort by best validation accuracy
df_runs = df_runs.sort_values('best_val_acc', ascending=False)

print("="*80)
print("RUN SUMMARY")
print("="*80)
print(df_runs[['run_name', 'best_val_acc', 'embedding_dim', 'hidden_dim', 'lr', 'state']].to_string(index=False))
print()

# Statistics
print("="*80)
print("STATISTICS")
print("="*80)
print(f"Total runs: {len(df_runs)}")
print(f"Completed runs: {len(df_runs[df_runs['state'] == 'finished'])}")
print(f"Best accuracy: {df_runs['best_val_acc'].max()*100:.2f}%")
print(f"Mean accuracy: {df_runs['best_val_acc'].mean()*100:.2f}%")
print(f"Std accuracy: {df_runs['best_val_acc'].std()*100:.2f}%")
print()

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Best val acc distribution
axes[0, 0].hist(df_runs['best_val_acc'].dropna() * 100, bins=20, edgecolor='black')
axes[0, 0].set_xlabel('Best Validation Accuracy (%)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Best Validation Accuracy')
axes[0, 0].axvline(df_runs['best_val_acc'].mean() * 100, color='red', linestyle='--', label='Mean')
axes[0, 0].legend()

# 2. Accuracy vs Embedding Dim
axes[0, 1].scatter(df_runs['embedding_dim'], df_runs['best_val_acc'] * 100, alpha=0.6)
axes[0, 1].set_xlabel('Embedding Dimension')
axes[0, 1].set_ylabel('Best Validation Accuracy (%)')
axes[0, 1].set_title('Accuracy vs Embedding Dimension')

# 3. Accuracy vs Hidden Dim
axes[1, 0].scatter(df_runs['hidden_dim'], df_runs['best_val_acc'] * 100, alpha=0.6)
axes[1, 0].set_xlabel('Hidden Dimension')
axes[1, 0].set_ylabel('Best Validation Accuracy (%)')
axes[1, 0].set_title('Accuracy vs Hidden Dimension')

# 4. Accuracy vs Learning Rate
axes[1, 1].scatter(df_runs['lr'], df_runs['best_val_acc'] * 100, alpha=0.6)
axes[1, 1].set_xlabel('Learning Rate')
axes[1, 1].set_ylabel('Best Validation Accuracy (%)')
axes[1, 1].set_title('Accuracy vs Learning Rate')
axes[1, 1].set_xscale('log')

plt.tight_layout()
plt.savefig('wandb_runs_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved to: wandb_runs_analysis.png")

# Export to CSV
df_runs.to_csv('wandb_runs_export.csv', index=False)
print("Data exported to: wandb_runs_export.csv")

# Display top 5 runs
print("\n" + "="*80)
print("TOP 5 RUNS")
print("="*80)
print(df_runs.head(5)[['run_name', 'best_val_acc', 'embedding_dim', 'hidden_dim', 'lr', 'epochs']].to_string(index=False))



