"""
Copyright 2024 b<>com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from timeit import default_timer
from typing import Generator, Set, Tuple, TYPE_CHECKING, List, Dict, Any, Optional, cast

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GIN

from src.placement.model import SystemState
from src.placement.scheduler import Scheduler


# ============================================================================
# GNN MODEL CLASSES (from notebook)
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
        self.fc1 = nn.Linear(2 * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, e_task, e_platform):
        x = torch.cat([e_task, e_platform], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)


class TaskPlacementGNN(nn.Module):
    """GNN for task-to-platform placement prediction."""
    def __init__(self, task_feature_dim, platform_feature_dim, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.task_encoder = TaskEncoder(task_feature_dim, hidden_dim, embedding_dim)
        self.platform_encoder = PlatformEncoder(platform_feature_dim, hidden_dim, embedding_dim)
        self.gin = GIN(
            in_channels=embedding_dim,
            hidden_channels=hidden_dim,
            num_layers=3,
            out_channels=embedding_dim
        )
        self.edge_scorer = EdgeScorer(embedding_dim, hidden_dim)

    def forward(self, data):
        n_tasks = data.n_tasks
        n_platforms = data.n_platforms

        # Encode features
        task_embeddings = self.task_encoder(data.task_features)
        platform_embeddings = self.platform_encoder(data.platform_features)

        # Message passing
        x = torch.cat([task_embeddings, platform_embeddings], dim=0)
        x = self.gin(x, data.edge_index)

        task_emb = x[:n_tasks]
        platform_emb = x[n_tasks:]

        # Score all edges
        ei = data.edge_index
        if ei.numel() == 0:
            return [torch.empty(0, device=x.device) for _ in range(n_tasks)]

        ti = ei[0]
        pj = ei[1] - n_tasks
        valid = (pj >= 0) & (pj < n_platforms)
        ti = ti[valid]
        pj = pj[valid]
        if ti.numel() == 0:
            return [torch.empty(0, device=x.device) for _ in range(n_tasks)]

        e_task = task_emb[ti]
        e_platform = platform_emb[pj]
        edge_scores = self.edge_scorer(e_task, e_platform)

        # Split scores per task
        logits_per_task = []
        for t in range(n_tasks):
            mask_t = (ti == t)
            logits_t = edge_scores[mask_t]
            logits_per_task.append(logits_t)

        return logits_per_task


# ============================================================================
# SCHEDULER
# ============================================================================

class EvaluatorScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Get batch size from policy or use default
        self.batch_size = getattr(self.policy, 'batch_size', 5)
        self.batch_timeout = getattr(self.policy, 'batch_timeout', 0.1)
        
        # Load GNN model
        self._load_gnn_model()
        
        # Task and platform type lists (from training)
        self.task_types = ['dnn1', 'dnn2']
        self.platform_types = ['rpiCpu', 'xavierCpu', 'xavierGpu', 'xavierDla', 'pynqFpga']
    
    def _load_gnn_model(self):
        """Load the trained GNN model from disk."""
        model_path = Path('/root/projects/my-herosim/src/policy/evaluator/best_gnn_placement_model.pt')
        
        if not model_path.exists():
            print(f"ERROR: GNN model not found at {model_path}")
            sys.exit(1)
        
        # Initialize model with same architecture as training
        task_feature_dim = 3  # 2 (task type one-hot) + 1 (source node ID)
        platform_feature_dim = 7  # 5 (platform type one-hot) + 2 (replica flags)
        
        self.gnn_model = TaskPlacementGNN(
            task_feature_dim=task_feature_dim,
            platform_feature_dim=platform_feature_dim,
            embedding_dim=64,
            hidden_dim=128
        )
        
        # Load trained weights
        self.gnn_model.load_state_dict(torch.load(model_path))
        self.gnn_model.eval()
        
        print(f"✓ GNN model loaded from {model_path}")

    def scheduler_process(self) -> Generator:
        # keep this for the simpy generator 
        if False:
            yield

        """Override to process multiple tasks simultaneously in batches"""
        print(
            f"[ {self.env.now} ] Evaluator Scheduler started with policy"
            f" {self.policy} (batch_size={self.batch_size})"
        )

        while True:
            # Collect a batch of tasks
            batch_tasks = yield self.env.process(self._collect_task_batch())
            
            if not batch_tasks:
                # No tasks available, wait a bit and try again
                yield self.env.timeout(0.1)
                continue

            print(f"[ {self.env.now} ] DEBUG: Processing batch of {len(batch_tasks)} tasks simultaneously")

            # Process all tasks in the batch together
            yield self.env.process(self._process_task_batch(batch_tasks))

    def _collect_task_batch(self) -> Generator[Any, Any, List[Task]]:
        """Collect a batch of tasks that are ready for scheduling"""
        batch = []
        
        print(f"[ {self.env.now} ] DEBUG: Starting batch collection (size={self.batch_size})")
        
        # Try to get up to batch_size tasks
        for i in range(self.batch_size):
            try:
                # Try to get a task (this will block until a task is available)
                task: Task = yield self.tasks.get(
                    lambda queued_task: all(
                        dependency.finished for dependency in queued_task.dependencies
                    )
                )
                batch.append(task)
                print(f"[ {self.env.now} ] DEBUG: Added task {task.id} to batch (size={len(batch)})")
            except:
                # No more tasks available
                print(f"[ {self.env.now} ] DEBUG: No more tasks available after {len(batch)} tasks")
                break
        
        print(f"[ {self.env.now} ] DEBUG: Batch collection complete, returning {len(batch)} tasks")
        return batch

    def _process_task_batch(self, batch_tasks: List[Task]) -> Generator:
        """Process multiple tasks simultaneously in a single operation"""
        print(f"[ {self.env.now} ] DEBUG: Processing {len(batch_tasks)} tasks in batch")
        # DEBUG: dump current per-platform queue status including internal (prewarm) tasks
        try:
            for node in self.nodes.items:
                for plat in node.platforms.items:
                    q_len = len(plat.queue.items)
                    if q_len > 0:
                        internal = sum(1 for t in plat.queue.items if getattr(t, 'is_internal', False))
                        print(f"[ {self.env.now} ] DEBUG: Platform {plat.id}@{node.node_name} q_len={q_len} internal={internal}")
        except Exception:
            pass
        
        # Get system state once for all tasks
        system_state: SystemState = yield self.mutex.get()
        replicas: Dict[str, Set[Tuple[Node, Platform]]] = system_state.replicas
        
        # Process all tasks in the batch
        for task in batch_tasks:
            task_replicas = replicas[task.type["name"]]

            # Scaling from zero must be forced
            if not task_replicas:
                logging.warning(
                    f"[ {self.env.now} ] Scheduler did not find available replica for"
                    f" {task}"
                )

                # Put task back in queue
                task.postponed_count += 1
                yield self.tasks.put(task)

                # Request a new replica from the Autoscaler
                stop = yield self.env.process(
                    self.autoscaler.create_first_replica(system_state, task.type)
                )

                # Next event
                self.env.step()
                continue

            # Measure wall-clock time for the scheduling decision
            start = default_timer()

            # Schedule task according to policy
            placement_result = yield self.env.process(
                self.placement(system_state, task)
            )

            # Check if placement was successful
            if placement_result is None:
                # No valid replicas available - postpone task and request scaling
                task.postponed_count += 1
                yield self.tasks.put(task)
                
                # Request a new replica from the Autoscaler
                stop = yield self.env.process(
                    self.autoscaler.create_first_replica(system_state, task.type)
                )
                
                # Next event
                self.env.step()
                continue

            sched_node, sched_platform = placement_result

            # Update node
            node: Node = yield self.nodes.get(lambda node: node.id == sched_node.id)
            task.node = node
            node.unused = False
            
            # Update platform
            platform: Platform = yield node.platforms.get(
                lambda platform: platform.id == sched_platform.id
            )
            task.platform = platform

            # contention-based pre-exec delay removed; rely on queues and warmth only

            # End wall-clock time measurement
            end = default_timer()
            elapsed_clock_time = end - start
            node.wall_clock_scheduling_time += elapsed_clock_time

            # Queue the task for execution
            yield platform.queue.put(task)
            yield task.scheduled.succeed()

            # Release platform
            yield node.platforms.put(platform)

            # Node is released
            yield self.nodes.put(node)
            
            # print(f"[ {self.env.now} ] DEBUG: Completed task {task.id} in batch")

        # Release mutex after processing entire batch
        yield self.mutex.put(system_state)
        
        print(f"[ {self.env.now} ] DEBUG: Batch processing complete for {len(batch_tasks)} tasks")




    def placement(self, system_state: SystemState, task: Task) -> Generator[Any, Any, Optional[Tuple[Node, Platform]]]:
        """Use GNN to predict optimal placement for a single task."""
        # Scheduling functions called in a Simpy Process must be Generators
        # No-op as per https://stackoverflow.com/a/68628599/9568489
        if False:
            yield
        
        replicas: Set[Tuple[Node, Platform]] = system_state.replicas[task.type["name"]]
        valid_replicas = self._get_valid_replicas(replicas, task)
        
        if not valid_replicas:
            print(f"[ {self.env.now} ] ERROR: No valid replicas for task {task.id}")
            return None
        
        try:
            # Build graph for this single task
            graph_data = self._build_graph_for_tasks([task], [valid_replicas], system_state)
            
            # Run GNN inference
            with torch.no_grad():
                logits_per_task = self.gnn_model(graph_data)
            
            # Extract prediction for this task
            if len(logits_per_task) == 0 or logits_per_task[0].numel() == 0:
                raise ValueError("No logits returned from GNN")
            
            task_logits = logits_per_task[0]
            best_platform_idx = task_logits.argmax().item()
            
            # Map back to actual Node/Platform tuple
            selected_replica = valid_replicas[best_platform_idx]
            
            print(f"[ {self.env.now} ] GNN placed task {task.id} on {selected_replica[0].node_name}:{selected_replica[1].id}")
            return selected_replica
            
        except Exception as e:
            # Fallback to Least Connected
            print(f"[ {self.env.now} ] GNN failed for task {task.id}: {e}, using Least Connected")
            bounded_concurrency = min(valid_replicas, key=lambda couple: len(couple[1].queue.items))
            return bounded_concurrency
    
    def _build_graph_for_tasks(
        self, 
        tasks: List[Task], 
        valid_replicas_per_task: List[List[Tuple[Node, Platform]]], 
        system_state: SystemState
    ) -> Data:
        """
        Build PyG Data object for batch inference.
        
        Args:
            tasks: List of tasks to schedule
            valid_replicas_per_task: For each task, list of valid (Node, Platform) tuples
            system_state: Current system state
        
        Returns:
            PyG Data object with task and platform features + edges
        """
        n_tasks = len(tasks)
        
        # Collect all unique platforms across all tasks
        all_platforms = []
        platform_to_idx = {}
        for replicas in valid_replicas_per_task:
            for node, platform in replicas:
                key = (node.id, platform.id)
                if key not in platform_to_idx:
                    platform_to_idx[key] = len(all_platforms)
                    all_platforms.append((node, platform))
        
        n_platforms = len(all_platforms)
        
        # Build node name to ID mapping for source node normalization
        all_nodes = list(set(node for node, _ in all_platforms))
        node_name_to_id = {node.node_name: i for i, node in enumerate(all_nodes)}
        max_node_id = len(all_nodes)
        
        # Build task features: [task_type_onehot (2), source_node_id_norm (1)]
        task_features = []
        for task in tasks:
            task_type_name = task.type['name']
            task_type_onehot = [1.0 if task_type_name == tt else 0.0 for tt in self.task_types]
            
            source_node_id = node_name_to_id.get(task.node_name, 0)
            source_node_id_norm = source_node_id / max(1, max_node_id)
            
            task_features.append(task_type_onehot + [source_node_id_norm])
        
        task_features = torch.tensor(task_features, dtype=torch.float)
        
        # Build platform features: [platform_type_onehot (5), has_dnn1_replica (1), has_dnn2_replica (1)]
        platform_features = []
        for node, platform in all_platforms:
            platform_type = platform.type['shortName']
            plat_type_onehot = [1.0 if platform_type == pt else 0.0 for pt in self.platform_types]
            
            # Check replica state for this platform
            has_dnn1 = 1.0 if (node, platform) in system_state.replicas.get('dnn1', set()) else 0.0
            has_dnn2 = 1.0 if (node, platform) in system_state.replicas.get('dnn2', set()) else 0.0
            
            platform_features.append(plat_type_onehot + [has_dnn1, has_dnn2])
        
        platform_features = torch.tensor(platform_features, dtype=torch.float)
        
        # Build edge index: tasks (0..n_tasks-1) -> platforms (n_tasks..n_tasks+n_platforms-1)
        edge_index = []
        for task_idx, replicas in enumerate(valid_replicas_per_task):
            for node, platform in replicas:
                key = (node.id, platform.id)
                platform_idx = platform_to_idx[key]
                edge_index.append([task_idx, n_tasks + platform_idx])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Create PyG Data object
        data = Data(
            edge_index=edge_index,
            n_tasks=n_tasks,
            n_platforms=n_platforms,
            task_features=task_features,
            platform_features=platform_features
        )
        
        return data

    def _get_valid_replicas(self, replicas: Set[Tuple[Node, Platform]], task: Task) -> List[Tuple[Node, Platform]]:
        """Get valid replicas: task's source node + server nodes with network connectivity"""
        # Debug header
        try:
            task_type_name = task.type["name"]
        except Exception:
            task_type_name = "unknown"
        print(
            f"[ {self.env.now} ] DEBUG: _get_valid_replicas task={task.id} src={task.node_name} type={task_type_name} candidates={len(replicas)}"
        )

        valid_replicas = []
        kept_local = 0
        kept_server_connected = 0
        skipped_client_other = 0
        skipped_no_connectivity = 0

        for node, platform in replicas:
            # Include if it's the task's source node (local execution)
            if node.node_name == task.node_name:
                valid_replicas.append((node, platform))
                kept_local += 1
            # Include if it's a server node AND has network connectivity to task source
            elif not node.node_name.startswith('client_node'):
                # Check if this node has network connectivity to the task's source node
                if hasattr(node, 'network_map') and task.node_name in node.network_map:
                    valid_replicas.append((node, platform))
                    kept_server_connected += 1
                else:
                    skipped_no_connectivity += 1
            else:
                skipped_client_other += 1
        
        # Never fall back to client nodes - only allow source node or connected server nodes
        if not valid_replicas:
            # If no valid replicas, only allow local execution on source node
            source_replicas = [(node, platform) for node, platform in replicas if node.node_name == task.node_name]
            if source_replicas:
                print(
                    f"[ {self.env.now} ] DEBUG: _get_valid_replicas fallback to local-only: {len(source_replicas)}"
                )
                return source_replicas
            else:
                print(
                    f"[ {self.env.now} ] DEBUG: _get_valid_replicas no valid replicas (kept_local={kept_local}, kept_server_connected={kept_server_connected}, skipped_client_other={skipped_client_other}, skipped_no_connectivity={skipped_no_connectivity})"
                )
                # Last resort: return empty list (will cause scaling from zero)
                return []
        
        # Debug footer with a small sample of chosen nodes
        chosen_nodes = [n.node_name for (n, _) in valid_replicas]
        sample = chosen_nodes[:5]
        print(
            f"[ {self.env.now} ] DEBUG: _get_valid_replicas selected={len(valid_replicas)} (local={kept_local}, server_connected={kept_server_connected}, skipped_client_other={skipped_client_other}, skipped_no_connectivity={skipped_no_connectivity}) sample={sample}"
        )

        return valid_replicas
