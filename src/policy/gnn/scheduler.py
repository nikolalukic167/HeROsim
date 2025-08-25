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

import sys
import os
from typing import Set, Tuple, TYPE_CHECKING, Generator, Dict, List, Any, Optional
import logging
import time
from collections import defaultdict
import random
import json
import pickle
from datetime import datetime
import numpy as np

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task

from src.placement.model import SystemState
from src.placement.scheduler import Scheduler

# GNN-specific imports
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, GATConv
from torch_geometric.data import Data


# Model configuration class (matching notebook)
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for the WRR-based GNN model"""
    # Model architecture
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.2
    attention_heads: int = 4
    
    # Training
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 50
    patience: int = 15
    min_delta: float = 1e-4
    
    # WRR-specific configuration
    use_composite_score: bool = True
    score_normalization: str = 'sigmoid'
    
    # Composite score weights
    weight_latency: float = 0.4
    weight_queue: float = 0.25
    weight_cold_start: float = 0.15
    weight_energy: float = 0.1
    weight_utilization: float = 0.1
    
    # Data processing
    normalize_features: bool = True
    num_physical_nodes: int = 10

class GNNScheduler(Scheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.experiment_mode = "knative"  # Options: "local", "offload", "wrr-gnn", "lowest-time-on-platform", "knative", "gnn"
        self.system_state_results: List[Dict[str, Any]] = []

        # WRR-specific configuration
        self.gnn_update_interval = 10.0  # Update GNN weights every 10 seconds
        self.weight_cache_timeout = 30.0  # Cache timeout for weights
        self.last_gnn_update = 0.0
        self.replica_weights = {}  # Cache for replica weights
        self.wrr_counters = defaultdict(int)  # Round-robin counters

        # GNN model components
        self.gnn_model = None
        if self.experiment_mode == "wrr-gnn" or self.experiment_mode == "gnn":
            self._load_gnn_components()

        # Simple replica change tracking
        self.last_replica_change_time = 0.0
        self.current_replicas = {}  # Track current replicas per task type

    def _load_gnn_components(self):
        """Load WRR GNN model"""
        try:
            # Load WRR GNN model
            model_path = '/root/projects/my-herosim/src/notebooks/models/wrr_gnn_final.pt'
            if os.path.exists(model_path):
                # Note: Using weights_only=False because checkpoint contains ModelConfig class
                # This is safe since we trust our own model file
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                self.gnn_model = self._build_wrr_gnn_model(checkpoint)
                print("WRR GNN model loaded successfully")
            else:
                print(f"Warning: WRR GNN model not found at {model_path}")
        except Exception as e:
            print(f"Warning: Failed to load WRR GNN model: {e}")

    def _build_wrr_gnn_model(self, checkpoint):
        """Build WRR GNN model from checkpoint"""
        # Use the config from the checkpoint (saved during training)
        config = checkpoint.get('config', None)
        if config is None:
            # Fallback to default config if not in checkpoint
            config = ModelConfig()
        
        class WRRNodeBasedGNN(torch.nn.Module):
            """Node-based GNN that predicts WRR composite scores for physical nodes"""
            
            def __init__(self, node_feature_dim=25, edge_feature_dim=2):  # Updated to 25 features including client node flag
                super().__init__()
                self.config = config
                self.node_feature_dim = node_feature_dim
                self.edge_feature_dim = edge_feature_dim
                
                # Input encoders
                self.node_encoder = torch.nn.Sequential(
                    torch.nn.Linear(node_feature_dim, config.hidden_dim),
                    torch.nn.BatchNorm1d(config.hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(config.dropout)
                )
                
                self.edge_encoder = torch.nn.Sequential(
                    torch.nn.Linear(edge_feature_dim, config.hidden_dim),
                    torch.nn.BatchNorm1d(config.hidden_dim),
                    torch.nn.ReLU()
                )
                
                # Graph convolution layers
                self.conv_layers = torch.nn.ModuleList()
                self.batch_norms = torch.nn.ModuleList()
                
                for i in range(config.num_layers):
                    self.conv_layers.append(
                        GATConv(
                            config.hidden_dim, 
                            config.hidden_dim // config.attention_heads,
                            heads=config.attention_heads, 
                            dropout=config.dropout,
                            edge_dim=config.hidden_dim
                        )
                    )
                    self.batch_norms.append(BatchNorm(config.hidden_dim))
                
                # Output layers - predict WRR composite scores
                self.wrr_score_predictor = torch.nn.Sequential(
                    torch.nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(config.dropout),
                    torch.nn.Linear(config.hidden_dim // 2, 1),
                    torch.nn.Sigmoid() if config.score_normalization == 'sigmoid' else torch.nn.Identity()
                )
            
            def forward(self, x, edge_index, edge_attr):
                # Encode inputs
                x = self.node_encoder(x)
                edge_attr = self.edge_encoder(edge_attr)
                
                # Apply graph convolutions with residual connections
                for i, (conv, norm) in enumerate(zip(self.conv_layers, self.batch_norms)):
                    x_new = conv(x, edge_index, edge_attr)
                    x_new = norm(x_new)
                    x_new = F.relu(x_new)
                    x_new = F.dropout(x_new, p=self.config.dropout, training=self.training)
                    
                    # Residual connection
                    if x.size(-1) == x_new.size(-1):
                        x = x + x_new
                    else:
                        x = x_new
                
                # Predict WRR composite scores
                wrr_score_pred = self.wrr_score_predictor(x)
                
                return {
                    'wrr_score': wrr_score_pred.squeeze(-1)
                }
  
        model = WRRNodeBasedGNN()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def _should_update_gnn_weights(self) -> bool:
        """Check if GNN weights should be updated"""
        current_time = time.time()
        time_since_last_update = current_time - self.last_gnn_update
        
        # Update if interval has passed or weights are expired
        return (time_since_last_update >= self.gnn_update_interval or 
                time_since_last_update >= self.weight_cache_timeout or
                not self.replica_weights)

    def _update_gnn_weights(self, system_state: SystemState, task: Task):
        """Update replica weights using GNN predictions"""
        try:
            print(f"[WRR-GNN] Updating replica weights using GNN...")
            
            # Extract features for all physical nodes
            result = self._extract_wrr_gnn_features(system_state, task)
            if result[0] is None:
                print(f"[WRR-GNN] Failed to extract features for weight update")
                return
            
            x, edge_index, edge_attr, node_mapping, construction_time = result
            
            # Run GNN inference
            inference_start = time.time()
            with torch.no_grad():
                outputs = self.gnn_model(x, edge_index, edge_attr)
                composite_scores = outputs['wrr_score']
            inference_time = time.time() - inference_start
            
            # Convert scores to weights and cache them
            self.replica_weights.clear()
            
            # Normalize scores to create weights (higher score = higher weight)
            if len(composite_scores.shape) == 0:
                composite_scores = composite_scores.unsqueeze(0)
            
            scores_np = composite_scores.cpu().numpy()
            # Ensure weights are positive and sum to 1
            weights = np.exp(scores_np - np.max(scores_np))  # Softmax-like normalization
            weights = weights / np.sum(weights)
            
            # Cache weights for each node
            for i, (node_name, _) in enumerate(node_mapping):
                self.replica_weights[node_name] = float(weights[i])
            
            self.last_gnn_update = time.time()
            
            print(f"[WRR-GNN] Updated weights for {len(node_mapping)} nodes")
            print(f"[WRR-GNN] Weight range: {weights.min():.4f} - {weights.max():.4f}")
            print(f"[WRR-GNN] Inference time: {inference_time:.4f}s")
            
        except Exception as e:
            print(f"[WRR-GNN] Error updating weights: {e}")

    def _select_replica_weighted_round_robin(self, replicas: Set[Tuple[Node, Platform]], task: Task) -> Optional[Tuple[Node, Platform]]:
        """Select replica using weighted round-robin based on cached weights"""
        if not replicas:
            return None
        
        # Group replicas by node
        node_replicas = defaultdict(list)
        for node, platform in replicas:
            node_replicas[node.node_name].append((node, platform))
        
        # Get weights for available nodes
        weighted_nodes = []
        total_weight = 0.0
        
        for node_name, replica_list in node_replicas.items():
            weight = self.replica_weights.get(node_name, 1.0)  # Default weight if not cached
            if weight > 0:
                weighted_nodes.append((node_name, replica_list, weight))
                total_weight += weight
        
        if not weighted_nodes:
            # Fallback to random selection
            return random.choice(list(replicas))
        
        # Weighted round-robin selection
        # Use accumulative probability distribution
        random_value = random.random() * total_weight
        accumulative_weight = 0.0
        
        for node_name, replica_list, weight in weighted_nodes:
            accumulative_weight += weight
            if random_value <= accumulative_weight:
                # Selected this node, now pick replica with shortest queue
                best_replica = min(replica_list, key=lambda x: len(x[1].queue.items))
                
                # print(f"[WRR-GNN] Selected node {node_name} (weight: {weight:.4f}, queue: {len(best_replica[1].queue.items)})")
                return best_replica
        
        # Fallback (should not reach here)
        return weighted_nodes[-1][1][0]  # Return first replica of last node

    def _is_client_node(self, node_name: str) -> bool:
        """Check if a node is a client node based on its name."""
        return node_name.startswith('client_node')

    def _extract_wrr_gnn_features(self, system_state: SystemState, task: Task) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List, float]:
        """Extract features for WRR GNN using physical node approach with sparse connectivity"""
        construction_start = time.time()
        
        try:
            # Get all physical nodes from the scheduler's nodes FilterStore (all nodes in system)
            all_nodes = list(self.nodes.items)
            node_names = [node.node_name for node in all_nodes]
            
            # Sort node names to ensure consistent ordering
            node_names.sort()
            
            node_features = []
            node_mapping = []
            
            # Extract features for each physical node
            for node_name in node_names:
                features = self._extract_physical_node_features_from_system(
                    node_name, task, system_state
                )
                if features is not None:
                    node_features.append(features)
                    node_mapping.append((node_name, None))  # No specific platform mapping
                else:
                    # Use default features if extraction fails
                    node_features.append([0.0] * 25)  # 25 default features including client node flag
                    node_mapping.append((node_name, None))

            if not node_features:
                print(f"[WRR-GNN] Failed to extract features for any physical node")
                return None, None, None, [], 0.0

            # Create sparse network topology edges based on actual connectivity
            edge_index = []
            edge_features = []
            
            for i in range(len(node_names)):
                for j in range(len(node_names)):
                    if i != j:
                        node_i_name = node_names[i]
                        node_j_name = node_names[j]
                        
                        # Find the actual node objects
                        node_i = None
                        for node in all_nodes:
                            if node.node_name == node_i_name:
                                node_i = node
                                break
                        
                        if node_i and node_i.network_map:
                            # Check if there's a network connection between these nodes
                            if node_j_name in node_i.network_map:
                                # Connection exists - use actual latency
                                network_latency = node_i.network_map[node_j_name]
                                edge_index.append([i, j])
                                edge_features.append([
                                    network_latency,
                                    1.0 if node_i_name == node_j_name else 0.0  # Locality indicator
                                ])

            # Convert to tensors
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.tensor(edge_features, dtype=torch.float) if edge_features else torch.empty((0, 2), dtype=torch.float)

            construction_time = time.time() - construction_start
            
            return x, edge_index, edge_attr, node_mapping, construction_time

        except Exception as e:
            print(f"[WRR-GNN] Error in WRR feature extraction: {e}")
            construction_time = time.time() - construction_start
            return None, None, None, [], construction_time

    def _extract_physical_node_features_from_system(self, node_name: str, task: Task, 
                                                   system_state: SystemState) -> Optional[List[float]]:
        """Extract features for a physical node using system state data"""
        try:
            # Find the node in all nodes (from scheduler's nodes FilterStore)
            target_node = None
            for node in self.nodes.items:
                if node.node_name == node_name:
                    target_node = node
                    break
            
            if not target_node:
                print(f"[WRR-GNN] Node {node_name} not found in system state")
                return None
            
            # Get task type data from simulation data
            task_type_name = task.type["name"]
            if task_type_name not in self.data.task_types:
                print(f"[WRR-GNN] Task type {task_type_name} not found in task types data")
                return None
            
            task_type_data = self.data.task_types[task_type_name]
            
            # Constants
            PLATFORM_TYPES = ['rpiCpu', 'xavierCpu', 'xavierGpu', 'xavierDla', 'pynqFpga']
            HARDWARE_TYPES = ['cpu', 'gpu', 'dla', 'fpga']
            TASK_TYPES = ['dnn1', 'dnn2']
            
            # 1. NODE MEMORY CHARACTERISTICS (2 features)
            node_memory = float(target_node.memory)
            node_memory_tier = 0 if node_memory <= 1 else (1 if node_memory <= 8 else 2)
            
            # 2. PLATFORM COUNTS (5 features)
            platform_counts = defaultdict(int)
            for platform in target_node.platforms.items:
                platform_counts[platform.type["shortName"]] += 1
            
            rpi_cpu_count = platform_counts.get('rpiCpu', 0)
            xavier_cpu_count = platform_counts.get('xavierCpu', 0)
            xavier_gpu_count = platform_counts.get('xavierGpu', 0)
            xavier_dla_count = platform_counts.get('xavierDla', 0)
            pynq_fpga_count = platform_counts.get('pynqFpga', 0)
            
            # 3. TASK EXECUTION PERFORMANCE (6 features)
            available_platforms = [p for p in task_type_data['platforms'] if platform_counts.get(p, 0) > 0]
            if available_platforms:
                execution_times = [task_type_data['executionTime'][p] for p in available_platforms]
                min_execution_time = min(execution_times)
                max_execution_time = max(execution_times)
                avg_execution_time = sum(execution_times) / len(execution_times)
                
                cold_starts = [task_type_data['coldStartDuration'][p] for p in available_platforms]
                min_cold_start = min(cold_starts)
                max_cold_start = max(cold_starts)
                avg_cold_start = sum(cold_starts) / len(cold_starts)
            else:
                # No compatible platforms - use penalty values
                min_execution_time = max_execution_time = avg_execution_time = 10.0
                min_cold_start = max_cold_start = avg_cold_start = 10.0
            
            # 4. TASK I/O CHARACTERISTICS (4 features)
            app_type = 'nofs-' + task_type_name
            state_size_data = task_type_data['stateSize'].get(app_type, {})
            task_input_size = state_size_data.get('input', 153600) / 1000  # Convert to KB
            task_output_size = state_size_data.get('output', 8000) / 1000
            
            # Storage performance (simplified)
            storage_read_speed = 294.0  # Default eMMC read speed
            storage_write_speed = 82.0  # Default eMMC write speed
            
            # 5. TASK-PLATFORM MEMORY COMPATIBILITY (2 features)
            if available_platforms:
                memory_reqs = [task_type_data['memoryRequirements'][p] for p in available_platforms]
                min_memory_required = min(memory_reqs)
                max_memory_required = max(memory_reqs)
            else:
                min_memory_required = max_memory_required = 1.0
            
            # 6. CATEGORICAL ENCODINGS (6 features)
            # Task type encoding (2 features)
            task_type_encoding = [1 if task_type_name == 'dnn1' else 0, 1 if task_type_name == 'dnn2' else 0]
            
            # Node type encoding (4 features) - including client node flag
            is_client_node = self._is_client_node(node_name)
            node_type_encoding = [
                1 if 'rpi' in node_name.lower() else 0,
                1 if 'xavier' in node_name.lower() or any('xavier' in p for p in platform_counts.keys()) else 0,
                1 if 'pynq' in node_name.lower() else 0,
                1 if is_client_node else 0  # Client node flag
            ]
            
            # COMBINE ALL FEATURES (25 total)
            node_features = [
                # Memory characteristics (2) 
                node_memory, node_memory_tier,
                # Platform counts (5)
                rpi_cpu_count, xavier_cpu_count, xavier_gpu_count, xavier_dla_count, pynq_fpga_count,
                # Execution performance (6)
                min_execution_time, max_execution_time, avg_execution_time,
                min_cold_start, max_cold_start, avg_cold_start,
                # I/O characteristics (4)
                task_input_size, task_output_size, storage_read_speed, storage_write_speed,
                # Memory compatibility (2)
                min_memory_required, max_memory_required,
                # Categorical encodings (5)
                *task_type_encoding, *node_type_encoding
            ]
            
            return node_features

        except Exception as e:
            print(f"[WRR-GNN] Failed to extract features for {node_name}: {e}")
            return None

    def _knative_fallback(self, replicas: Set[Tuple[Node, Platform]], task: Task) -> Optional[Tuple[Node, Platform]]:
        """Simple knative-style fallback: pick replica with shortest queue"""
        if replicas:
            best_replica = min(replicas, key=lambda couple: len(couple[1].queue.items))
            task.execution_node = best_replica[0].node_name
            task.execution_platform = str(best_replica[1].id)
            print(f"[FALLBACK] Task {task.id} placed on node {task.execution_node} (shortest queue)")
            return best_replica
        # return None
    
    def _filter_replicas_by_connectivity(self, replicas: Set[Tuple[Node, Platform]], task: Task, 
                                       network_topology: Dict[str, Dict[str, float]]) -> Set[Tuple[Node, Platform]]:
        if not replicas:
            return set()

        source_node = task.node_name
        reachable_replicas = set()

        source_node = task.node_name
        reachable_replicas = set()

        for node, platform in replicas:
            target_node = node.node_name
            
            # Skip if same node (local execution)
            if source_node == target_node:
                reachable_replicas.add((node, platform))
                continue
            
            # Prevent offloading between client nodes
            if self._is_client_node(source_node) and self._is_client_node(target_node):
                # print(f"[R] Skipping replica on {target_node} - client nodes cannot offload to other client nodes")
                continue
            
            # Check if there's a network path from source to target
            if source_node in network_topology and target_node in network_topology[source_node]:
                # Network connection exists
                reachable_replicas.add((node, platform))
            # else:
                # No network connection - skip this replica
                # print(f"[R] Skipping replica on {target_node} - no network path from {source_node}")

        return reachable_replicas

    def placement(self, system_state: SystemState, task: Task) -> Generator[Any, Any, Optional[Tuple[Node, Platform]]]:
        """Place a task on a suitable platform using the specified experiment mode"""
        if False:
            yield

        replicas: Set[Tuple[Node, Platform]] = system_state.replicas[task.type["name"]]
        
        # Simple replica change tracking
        task_type = task.type["name"]
        current_replica_set = frozenset((node.node_name, platform.id) for node, platform in replicas)
        
        if task_type in self.current_replicas and self.current_replicas[task_type] != current_replica_set:
            # Replicas changed
            time_since_change = time.time() - self.last_replica_change_time
            print(f"[REPLICA-CHANGE] {task_type} replicas changed after {time_since_change:.4f}s")
            print(f"[REPLICA-CHANGE] Old: {len(self.current_replicas[task_type])} replicas, New: {len(current_replica_set)} replicas")
            self.last_replica_change_time = time.time()
        
        self.current_replicas[task_type] = current_replica_set
        
        # Get network topology from all nodes in the system (not just available resources)
        network_topology = {}
        for node in self.nodes.items:
            network_topology[node.node_name] = node.network_map
        
        # Filter replicas based on network connectivity and disallow client-to-client offloading
        time_now = time.time()
        reachable_replicas = self._filter_replicas_by_connectivity(replicas, task, network_topology)
        print("time needed to filter replicas: ", time.time() - time_now)
        
        if not reachable_replicas:
            print(f"[R] No reachable replicas for task {task.id} from {task.node_name}")
            # Try to create a local replica as fallback
            try:
                replica_result = yield self.env.process(
                    self.autoscaler.create_first_replica_on_node(system_state, task.type, task.node_name)
                )
                if not isinstance(replica_result, StopIteration):
                    task.execution_node = replica_result[0].node_name
                    task.execution_platform = str(replica_result[1].id)
                    print(f"[R] Created local replica for task {task.id}")
                    return replica_result
            except Exception as e:
                print(f"[R] Failed to create local replica: {e}")
                # Final fallback - use any available replica
                if replicas:
                    any_replica = next(iter(replicas))
                    task.execution_node = any_replica[0].node_name
                    task.execution_platform = str(any_replica[1].id)
                    print(f"[R] Using any available replica as fallback")
                    return any_replica
                else:
                    raise Exception("No replicas available and failed to create new replica")
        
        # Use filtered replicas for placement decisions
        replicas = reachable_replicas

        if self.experiment_mode == "local":
            local_replicas = []
            for node, platform in replicas:
                if node.node_name == task.node_name:
                    local_replicas.append((node, platform))

            if local_replicas:
                best_local_replica = min(local_replicas, key=lambda x: len(x[1].queue.items))
                task.execution_node = best_local_replica[0].node_name
                task.execution_platform = str(best_local_replica[1].id)
                print(f"[LOCAL] Task {task.id} executing locally on node {task.execution_node}")
                return best_local_replica

            # No local replica exists, Try to create a local replica
            try:
                replica_result = yield self.env.process(
                    self.autoscaler.create_first_replica_on_node(system_state, task.type, task.node_name)
                )
                if isinstance(replica_result, StopIteration):
                    print(f"[ERROR LOCAL] Failed to create local replica: {replica_result}")
                else:
                    task.execution_node = replica_result[0].node_name
                    task.execution_platform = str(replica_result[1].id)
                    print(f"[LOCAL] Task {task.id} ({task.type['name']}) executing on newly created local replica on node {task.execution_node}")
                    return replica_result
            except Exception as e:
                print(f"[LOCAL] Exception during local replica creation: {e}")
                # Fallback to any available platform
                return self._knative_fallback(replicas, task)

        elif self.experiment_mode == "offload":
            # todo: implement offload only logic 
            if replicas:
                best_replica = min(replicas, key=lambda couple: len(couple[1].queue.items))
                task.execution_node = best_replica[0].node_name
                task.execution_platform = str(best_replica[1].id)
                print(f"[OFFLOAD] Task {task.id} offloaded to node {task.execution_node}")
                return best_replica
        
        elif self.experiment_mode == "wrr-gnn":
            # WRR-GNN based placement
            if not self.gnn_model:
                print(f"[WRR-GNN] Model not available, falling back to knative")
                return self._knative_fallback(replicas, task)
            
            try:
                decision_start = time.time()
                
                # Update GNN weights periodically
                if self._should_update_gnn_weights():
                    self._update_gnn_weights(system_state, task)
                
                # Use weighted round-robin selection based on cached weights
                selected_replica = self._select_replica_weighted_round_robin(replicas, task)
                
                if selected_replica:
                    task.execution_node = selected_replica[0].node_name
                    task.execution_platform = str(selected_replica[1].id)
                    
                    decision_time = time.time() - decision_start
                    task.gnn_decision_time = decision_time
                    
                    # Simulate the decision time (minimal since we're using cached weights)
                    yield self.env.timeout(decision_time)
                    
                    # print(f"[WRR-GNN] Task {task.id} placed on node {task.execution_node} (decision time: {decision_time:.4f}s)")
                    return selected_replica
                else:
                    print(f"[WRR-GNN] No replica selected, trying to create new replica")
                    # Try to create a new replica
                    try:
                        replica_result = yield self.env.process(
                            self.autoscaler.create_first_replica(system_state, task.type)
                        )
                        if not isinstance(replica_result, StopIteration):
                            task.execution_node = replica_result[0].node_name
                            task.execution_platform = str(replica_result[1].id)
                            print(f"[WRR-GNN] Created new replica for task {task.id}")
                            return replica_result
                    except Exception as e:
                        print(f"[WRR-GNN] Failed to create new replica: {e}")
                        return self._knative_fallback(replicas, task)
                        
            except Exception as e:
                print(f"[WRR-GNN] Error during WRR placement: {e}")
                return self._knative_fallback(replicas, task)

        elif self.experiment_mode == "knative":
            # Knative-style placement: pick the replica with the shortest queue
            if replicas:
                best_replica = min(replicas, key=lambda couple: len(couple[1].queue.items))
                task.execution_node = best_replica[0].node_name
                task.execution_platform = str(best_replica[1].id)  # Use string platform ID
                print(f"[KNATIVE] Task {task.id} placed on node {task.execution_node} platform {task.execution_platform}")
                return best_replica

        # elif self.experiment_mode == "lowest-time-on-platform":
            # todo:

        # If no placement found, try to create new replica
        # todo: scale dynamically based on average queue length
        try:
            replica_result = yield self.env.process(
                self.autoscaler.create_first_replica(system_state, task.type)
            )
            if isinstance(replica_result, StopIteration):
                print(f"Failed to create replica: {replica_result}")
                # Final fallback: use any available platform
                fallback_result = self._knative_fallback(replicas, task)
                if fallback_result:
                    return fallback_result
                else:
                    print(f"No replicas available and failed to create new replica")
                    # As last resort, pick any available replica
                    if replicas:
                        any_replica = next(iter(replicas))
                        task.execution_node = any_replica[0].node_name
                        task.execution_platform = str(any_replica[1].id)
                        print(f"[FINAL] Using any available replica as last resort")
                        return any_replica
                    else:
                        raise Exception("No replicas available and failed to create new replica")
            print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX: \n   {replica_result}")
            return replica_result
        except Exception as e:
            print(f"Exception during replica creation: {e}")
            # Final fallback: use any available platform
            fallback_result = self._knative_fallback(replicas, task)
            if fallback_result:
                return fallback_result
            else:
                print(f"No replicas available and exception during creation: {e}")
                # As last resort, pick any available replica
                if replicas:
                    any_replica = next(iter(replicas))
                    task.execution_node = any_replica[0].node_name
                    task.execution_platform = str(any_replica[1].id)
                    print(f"[FINAL] Using any available replica as last resort after exception")
                    return any_replica
                else:
                    raise Exception(f"No replicas available and exception during creation: {e}")
