"""
GNN-based Scheduler for Task-to-Platform Placement (NON-UNIQUE VERSION)

This scheduler uses a trained GNN model to make placement decisions.
It processes batches of 2-3 tasks together (model training range) and
uses a per-task greedy decoder that allows non-unique placements
(multiple tasks can be placed on the same replica).

Fallback to shortest-queue for:
- Single task batches (model not trained on 1 task)
- Large batches > 3 tasks (model not trained on 4+ tasks)
"""

from __future__ import annotations

import logging
import json
from timeit import default_timer
from typing import Generator, List, Optional, Set, Tuple, TYPE_CHECKING, Dict, Any

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task

from src.placement.model import SystemState
from src.placement.scheduler import Scheduler

# Task-platform compatibility (same as training)
TASK_PLATFORM_COMPATIBILITY = {
    'dnn1': ['rpiCpu', 'xavierGpu', 'xavierCpu', 'pynqFpga'],
    'dnn2': ['rpiCpu', 'xavierGpu', 'xavierCpu']
}

# Queue normalization constant (same as training)
QUEUE_NORM_FACTOR = 50.0

# GNN model training range: 2-3 tasks
# Use fallback (shortest queue) for batches outside this range
MIN_BATCH_SIZE_FOR_GNN = 2  # Model trained on 2-3 tasks
MAX_BATCH_SIZE_FOR_GNN = 3  # Model not trained on 4+ tasks


class GNNScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Batch processing: model trained on 2-3 tasks
        # Batches of 1 or 4+ will use shortest-queue fallback
        self.batch_size = 3  # Max batch size (model trained on 2-3 tasks)
        
        # Timeout analysis (in simulation seconds):
        # - 0.01s (10ms): Very responsive, may get single tasks
        # - 0.05s (50ms): Good balance for batching 2-3 tasks
        # - 0.1s (100ms): Better batching but adds latency
        # Looking at task arrival patterns (~10-50ms apart), 50ms should collect 2-3 tasks
        self.batch_timeout = 0.05  # 50ms to collect tasks for GNN batching
        
        # GNN model will be set via models dict from orchestrator
        self.gnn_model = None
        self.device = None
        self.task_types_data = None
        self.dataset_id = None
        self.models_dict = None  # Store full models dict
        
        # === GNN Configuration ===
        # Use pure GNN with simple fallback (no soft blending)
        # Adaptive queue normalization will be calculated per batch
        
        # Stats tracking for debugging
        self.gnn_pure_decisions = 0
        self.fallback_decisions = 0

    def set_models(self, models: dict):
        """
        Set GNN models from orchestrator.
        
        Expected models dict structure:
        {
            'gnn_model': trained PyTorch model,
            'device': torch device (cuda/cpu),
            'task_types_data': task type metadata for feature computation,
            'dataset_id': optional dataset identifier
        }
        """
        print(f"[GNN Scheduler] set_models called with keys: {list(models.keys()) if models else 'None'}", flush=True)
        self.models_dict = models
        
        if models is None:
            print("[GNN Scheduler] WARNING: set_models called with None", flush=True)
            return
        
        if 'gnn_model' in models:
            self.gnn_model = models['gnn_model']
            print(f"[GNN Scheduler] Model loaded: {type(self.gnn_model).__name__}", flush=True)
        else:
            print("[GNN Scheduler] WARNING: 'gnn_model' not found in models dict", flush=True)
        
        if 'device' in models:
            self.device = models['device']
            print(f"[GNN Scheduler] Device: {self.device}", flush=True)
        else:
            self.device = torch.device('cpu')
            print("[GNN Scheduler] No device specified, using CPU", flush=True)
        
        if 'task_types_data' in models:
            self.task_types_data = models['task_types_data']
            print(f"[GNN Scheduler] Task types data loaded: {list(self.task_types_data.keys()) if self.task_types_data else 'None'}", flush=True)
        
        if 'dataset_id' in models:
            self.dataset_id = models['dataset_id']
        
        # Put model in eval mode
        if self.gnn_model is not None:
            self.gnn_model.eval()
            print("[GNN Scheduler] Model set to eval mode", flush=True)
        
        print(f"[GNN Scheduler] After set_models: gnn_model is None = {self.gnn_model is None}", flush=True)

    def scheduler_process(self) -> Generator:
        if False:
            yield

        logging.info(
            f"[ {self.env.now} ] GNN Scheduler started with policy {self.policy}"
            f" (batch_size={self.batch_size}, gnn_range=[{MIN_BATCH_SIZE_FOR_GNN},{MAX_BATCH_SIZE_FOR_GNN}])"
        )

        while True:
            batch_tasks = yield self.env.process(self._collect_task_batch())
            
            if not batch_tasks:
                yield self.env.timeout(0.01)
                continue

            yield self.env.process(self._process_task_batch(batch_tasks))

    def _collect_task_batch(self) -> Generator[Any, Any, List[Task]]:
        """
        Collect tasks into a batch using timeout-based waiting.
        
        Strategy:
        1. Wait for at least one task (blocking)
        2. Wait for batch_timeout duration to collect more tasks
        3. Return whatever was collected (min 1, max batch_size)
        
        This ensures GNN gets batches of 2-3 tasks (its training range).
        """
        batch: List[Task] = []
        
        def task_filter(queued_task):
            return all(dependency.finished for dependency in queued_task.dependencies)
        
        # First task: wait indefinitely (blocking is expected)
        task: Task = yield self.tasks.get(task_filter)
        batch.append(task)
        
        # Wait for batch_timeout to collect more tasks
        # Use small increments to be responsive while still batching
        timeout_remaining = self.batch_timeout
        poll_interval = 0.001  # 1ms polling interval
        
        while len(batch) < self.batch_size and timeout_remaining > 0:
            # Check if there are any ready tasks in the queue
            ready_tasks = [t for t in self.tasks.items if task_filter(t)]
            
            if ready_tasks:
                # Get the ready task immediately
                task = yield self.tasks.get(task_filter)
                batch.append(task)
            else:
                # Wait a small interval for more tasks to arrive
                wait_time = min(poll_interval, timeout_remaining)
                yield self.env.timeout(wait_time)
                timeout_remaining -= wait_time
        
        # Log batch size for debugging
        if len(batch) >= MIN_BATCH_SIZE_FOR_GNN:
            logging.debug(f"[ {self.env.now} ] GNN: Collected batch of {len(batch)} tasks (will use GNN)")
        else:
            logging.debug(f"[ {self.env.now} ] GNN: Collected batch of {len(batch)} tasks (will use fallback)")
        
        return batch

    def _process_task_batch(self, batch_tasks: List[Task]) -> Generator:
        """
        Process a batch of tasks using GNN placement.
        
        Optimized version: single mutex acquisition per batch, no proactive replica creation.
        """
        batch_start = default_timer()
        batch_size = len(batch_tasks)
        
        # Get system state once for the entire batch
        system_state: SystemState = yield self.mutex.get()
        
        # Capture queue snapshot for GNN inference
        queue_snapshot = self._capture_batch_queue_snapshot(system_state, batch_tasks)
        
        # Skip GNN for batches outside training range [2, 3]
        if batch_size < MIN_BATCH_SIZE_FOR_GNN or batch_size > MAX_BATCH_SIZE_FOR_GNN:
            placements = None  # Will trigger fallback to shortest queue
            inference_time = 0.0
            logging.info(f"[ {self.env.now} ] GNN: Batch size {batch_size} outside GNN range [{MIN_BATCH_SIZE_FOR_GNN},{MAX_BATCH_SIZE_FOR_GNN}], using fallback")
        else:
            # Build graph and run GNN inference
            inference_start = default_timer()
            placements = self._gnn_inference(batch_tasks, system_state, queue_snapshot)
            inference_time = default_timer() - inference_start
            if placements:
                logging.info(f"[ {self.env.now} ] GNN: Batch of {batch_size} tasks, GNN returned {len(placements)} placements in {inference_time*1000:.2f}ms")
            else:
                logging.info(f"[ {self.env.now} ] GNN: Batch of {batch_size} tasks, GNN inference failed (model not loaded?)")
        
        # Release mutex before processing tasks (allows monitor/autoscaler to run)
        yield self.mutex.put(system_state)
        
        # Process each task in batch
        for task_idx, task in enumerate(batch_tasks):
            task_start = default_timer()
            
            # Get fresh system state for this task
            current_system_state: SystemState = yield self.mutex.get()
            
            task_replicas = current_system_state.replicas.get(task.type["name"], set())
            valid_replicas = self._get_valid_replicas(task_replicas, task)

            # If no valid replicas, request autoscaling (reactive, like knative_network)
            if not valid_replicas:
                logging.warning(
                    f"[ {self.env.now} ] GNN Scheduler: no network-accessible replica for {task}"
                )
                
                task.postponed_count += 1
                yield self.tasks.put(task)

                # Request replica from autoscaler
                stop = yield self.env.process(
                    self.autoscaler.create_first_replica(
                        current_system_state, 
                        task.type,
                        source_node_name=task.node_name
                    )
                )
                
                yield self.mutex.put(current_system_state)
                continue

            # Select placement using GNN with fallback to shortest queue
            target_node, target_platform = self._select_placement_pure_gnn(
                task, task_idx, placements, valid_replicas
            )

            # Fallback to shortest queue if GNN placement is invalid
            if target_node is None or target_platform is None:
                target_node, target_platform = min(
                    valid_replicas, key=lambda couple: len(couple[1].queue.items)
                )
                self.fallback_decisions += 1

            task.execution_node = target_node.node_name
            task.execution_platform = str(target_platform.id)
            task.gnn_decision_time = inference_time / batch_size  # Amortized

            # Update node
            node: Node = yield self.nodes.get(lambda node: node.id == target_node.id)
            task.node = node
            node.unused = False
            
            # Update platform
            platform: Platform = yield node.platforms.get(lambda platform: platform.id == target_platform.id)
            task.platform = platform

            # End wall-clock time measurement
            task_end = default_timer()
            elapsed_clock_time = task_end - task_start
            node.wall_clock_scheduling_time += elapsed_clock_time

            # Put task in platform queue
            yield platform.queue.put(task)
            yield task.scheduled.succeed()

            yield node.platforms.put(platform)
            yield self.nodes.put(node)
            
            yield self.mutex.put(current_system_state)

    def _gnn_inference(
        self,
        batch_tasks: List[Task],
        system_state: SystemState,
        queue_snapshot: Dict[str, int]
    ) -> Optional[Dict[int, Tuple[int, int]]]:
        """
        Run GNN inference on the batch of tasks.
        
        Returns: Dict mapping task_idx -> (node_id, platform_id)
        """
        if self.gnn_model is None:
            print("[GNN] Model not loaded, using fallback")
            return None
        
        try:
            # Build graph from current system state
            graph, task_logit_to_placement = self._build_inference_graph(
                batch_tasks, system_state, queue_snapshot
            )
            
            if graph is None:
                return None
            
            # Move to device
            graph = graph.to(self.device)
            
            # Run inference
            with torch.no_grad():
                logits_per_task = self.gnn_model(graph)
            
            # Decode placements using greedy decoder
            placements = self._decode_placements(
                logits_per_task, task_logit_to_placement, len(batch_tasks)
            )
            
            return placements
            
        except Exception as e:
            print(f"[GNN] Inference error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_adaptive_queue_norm(self, queue_snapshot: Dict[str, int]) -> float:
        """
        Calculate adaptive queue normalization factor using 90th percentile.
        
        This adapts to the actual queue distribution in the system, which is critical
        for high-load scenarios where queues can be much larger than training data.
        
        Args:
            queue_snapshot: Dictionary mapping "node:platform" -> queue length
            
        Returns:
            Normalization factor (defaults to QUEUE_NORM_FACTOR if no queues present)
        """
        if not queue_snapshot:
            return QUEUE_NORM_FACTOR
        
        queue_values = list(queue_snapshot.values())
        if not queue_values:
            return QUEUE_NORM_FACTOR
        
        # Calculate 90th percentile for robustness (less sensitive to outliers than max)
        queue_values_sorted = sorted(queue_values)
        percentile_idx = int(len(queue_values_sorted) * 0.9)
        percentile_90 = queue_values_sorted[percentile_idx] if percentile_idx < len(queue_values_sorted) else queue_values_sorted[-1]
        
        # Use 90th percentile as normalization factor, with minimum of 1.0
        # This ensures that even in high-load scenarios, queues are normalized appropriately
        adaptive_factor = max(1.0, percentile_90)
        
        # Cap at reasonable maximum to avoid over-normalization
        # If queues are extremely large (e.g., 1000+), we still want some signal
        adaptive_factor = min(adaptive_factor, 100.0)
        
        return adaptive_factor

    def _build_inference_graph(
        self,
        batch_tasks: List[Task],
        system_state: SystemState,
        queue_snapshot: Dict[str, int]
    ) -> Tuple[Optional[Data], Optional[Dict[int, List[Tuple[int, int]]]]]:
        """
        Build a PyG graph from current system state for GNN inference.
        
        Returns: (graph, task_logit_to_placement mapping)
        """
        # Calculate adaptive queue normalization factor for this batch
        # adaptive_queue_norm = self._calculate_adaptive_queue_norm(queue_snapshot)
        adaptive_queue_norm = QUEUE_NORM_FACTOR
        # if adaptive_queue_norm != QUEUE_NORM_FACTOR:
        #     print(f"[GNN] Using adaptive queue norm factor: {adaptive_queue_norm:.2f} (90th percentile), fixed factor: {QUEUE_NORM_FACTOR}")
        
        n_tasks = len(batch_tasks)
        
        # Collect all platforms from nodes
        all_nodes = list(self.nodes.items)
        platforms_info = []  # List of (node, platform, node_id, plat_id, plat_type, node_name)
        
        for node in all_nodes:
            for platform in node.platforms.items:
                platforms_info.append((
                    node, platform, node.id, platform.id,
                    platform.type["shortName"], node.node_name
                ))
        
        n_platforms = len(platforms_info)
        if n_platforms == 0:
            return None, None
        
        # Build node_name -> node_id mapping
        node_name_to_id = {node.node_name: node.id for node in all_nodes}
        
        # Build platform position lookup
        plat_pos_by_key = {}  # (node_id, plat_id) -> position in platforms_info
        for pos, (node, plat, node_id, plat_id, plat_type, node_name) in enumerate(platforms_info):
            plat_pos_by_key[(node_id, plat_id)] = pos
        
        # Task features: [task_type_onehot(2), source_node_normalized(1)]
        task_types_vocab = ['dnn1', 'dnn2']
        task_features = []
        for task in batch_tasks:
            task_type = task.type["name"]
            onehot = [1.0 if task_type == t else 0.0 for t in task_types_vocab]
            src_node_idx = node_name_to_id.get(task.node_name, 0)
            src_norm = src_node_idx / max(len(all_nodes), 1)
            task_features.append(onehot + [src_norm])
        
        task_features_tensor = torch.tensor(task_features, dtype=torch.float32)
        
        # Platform features:
        # [type_onehot(5), has_dnn1(1), has_dnn2(1), queue(1),
        #  current_task_remaining(1), cold_start_remaining(1), comm_remaining(1),
        #  target_concurrency(1), usage_ratio(1)]
        platform_types_vocab = ['rpiCpu', 'xavierCpu', 'xavierGpu', 'xavierDla', 'pynqFpga']
        
        # Build replica lookup
        dnn1_replicas = set()
        dnn2_replicas = set()
        for node, plat in system_state.replicas.get('dnn1', set()):
            dnn1_replicas.add((node.id, plat.id))
        for node, plat in system_state.replicas.get('dnn2', set()):
            dnn2_replicas.add((node.id, plat.id))
        
        platform_features = []
        for node, plat, node_id, plat_id, plat_type, node_name in platforms_info:
            # Type one-hot
            onehot = [1.0 if plat_type == t else 0.0 for t in platform_types_vocab]
            # Replica flags
            has_dnn1 = 1.0 if (node_id, plat_id) in dnn1_replicas else 0.0
            has_dnn2 = 1.0 if (node_id, plat_id) in dnn2_replicas else 0.0
            # Queue length (normalized using adaptive factor)
            queue_key = f"{node_name}:{plat_id}"
            queue_len_raw = queue_snapshot.get(queue_key, 0)
            queue_len = queue_len_raw / adaptive_queue_norm

            # Temporal state (approximate if not available)
            current_task_remaining = 0.0
            cold_start_remaining = 0.0
            comm_remaining = 0.0
            if queue_len_raw > 0:
                avg_exec = 0.0
                count = 0
                if self.task_types_data:
                    for task_type_name, task_priors in self.task_types_data.items():
                        exec_map = task_priors.get("executionTime", {})
                        if isinstance(exec_map, dict):
                            exec_time = exec_map.get(plat_type, 0.0)
                            if exec_time > 0:
                                avg_exec += exec_time
                                count += 1
                if count > 0:
                    current_task_remaining = avg_exec / count
                    cold_start_remaining = current_task_remaining * 0.1
                    comm_remaining = current_task_remaining * 0.05

            # Normalize temporal features (assume max ~10s)
            current_task_remaining_norm = current_task_remaining / 10.0
            cold_start_remaining_norm = cold_start_remaining / 10.0
            comm_remaining_norm = comm_remaining / 10.0

            # Target concurrency and usage ratio (approximate, training parity)
            baseline_concurrency = 5.0
            target_concurrency = baseline_concurrency
            if self.task_types_data:
                supported_task_types = [
                    task_type_name
                    for task_type_name, task_priors in self.task_types_data.items()
                    if plat_type in task_priors.get("platforms", [])
                ]
                min_exec_times = []
                for task_type_name in supported_task_types:
                    task_priors = self.task_types_data.get(task_type_name, {})
                    exec_map = task_priors.get("executionTime", {})
                    if isinstance(exec_map, dict) and exec_map:
                        min_exec_times.append(min(exec_map.values()))
                if min_exec_times:
                    avg_min_exec = sum(min_exec_times) / len(min_exec_times)
                    exec_map_this = self.task_types_data.get(supported_task_types[0], {}).get("executionTime", {})
                    exec_time_this = exec_map_this.get(plat_type, avg_min_exec) if isinstance(exec_map_this, dict) else avg_min_exec
                    if exec_time_this > 0:
                        target_concurrency = max(1.0, avg_min_exec / exec_time_this * baseline_concurrency)

            usage_ratio = (queue_len_raw / target_concurrency) if target_concurrency > 0 else 0.0
            target_concurrency_norm = target_concurrency / 20.0
            usage_ratio_norm = usage_ratio / 5.0

            platform_features.append(
                onehot
                + [has_dnn1, has_dnn2, queue_len]
                + [current_task_remaining_norm, cold_start_remaining_norm, comm_remaining_norm]
                + [target_concurrency_norm, usage_ratio_norm]
            )
        
        platform_features_tensor = torch.tensor(platform_features, dtype=torch.float32)
        
        # Build edges: task -> compatible platforms
        task_offset = 0
        platform_offset = n_tasks
        
        edge_src, edge_dst = [], []
        edge_attrs = []
        task_logit_to_placement: Dict[int, List[Tuple[int, int]]] = {}
        
        # Build network map lookup
        network_maps = {}
        for node in all_nodes:
            if hasattr(node, 'network_map'):
                network_maps[node.node_name] = node.network_map
        
        for t_idx, task in enumerate(batch_tasks):
            task_type = task.type["name"]
            source_node = task.node_name
            compatible_types = TASK_PLATFORM_COMPATIBILITY.get(task_type, [])
            
            task_logit_to_placement[t_idx] = []
            
            for pos, (node, plat, node_id, plat_id, plat_type, node_name) in enumerate(platforms_info):
                # Check compatibility
                if plat_type not in compatible_types:
                    continue
                
                # Check network feasibility
                is_local = (source_node == node_name)
                is_server = not node_name.startswith('client_node')
                
                if not is_local:
                    if not is_server:
                        continue  # Can't place on other client nodes
                    # Check network connectivity
                    if node_name not in network_maps:
                        continue
                    if source_node not in network_maps[node_name]:
                        continue
                
                # Add edge
                task_node_idx = task_offset + t_idx
                plat_node_idx = platform_offset + pos
                edge_src.append(task_node_idx)
                edge_dst.append(plat_node_idx)
                
                # Edge attributes: [exec_time, latency, is_warm, energy, comm_time]
                exec_time = 0.0
                if self.task_types_data and task_type in self.task_types_data:
                    exec_time = self.task_types_data[task_type].get("executionTime", {}).get(plat_type, 0.0)
                
                latency = 0.0
                if not is_local and node_name in network_maps:
                    lat_entry = network_maps[node_name].get(source_node, {})
                    if isinstance(lat_entry, dict):
                        latency = lat_entry.get('latency', 0.0)
                    else:
                        try:
                            latency = float(lat_entry)
                        except:
                            latency = 0.0
                
                is_warm = 0.0
                if task_type == 'dnn1' and (node_id, plat_id) in dnn1_replicas:
                    is_warm = 1.0
                elif task_type == 'dnn2' and (node_id, plat_id) in dnn2_replicas:
                    is_warm = 1.0
                
                energy = 0.0
                if self.task_types_data and task_type in self.task_types_data:
                    energy_map = self.task_types_data[task_type].get("energy", {})
                    if isinstance(energy_map, dict):
                        energy = float(energy_map.get(plat_type, 0.0))

                comm_time = 0.0
                if self.task_types_data and task_type in self.task_types_data:
                    state_size_map = self.task_types_data[task_type].get("stateSize", {})
                    if isinstance(state_size_map, dict) and state_size_map:
                        app_state = next(iter(state_size_map.values()))
                        if isinstance(app_state, dict):
                            input_size = app_state.get("input", 0)
                            output_size = app_state.get("output", 0)
                            storage_throughput = 100.0 * 1024 * 1024  # bytes/s
                            storage_latency = 0.001  # seconds
                            read_time = (input_size / storage_throughput) + storage_latency
                            write_time = (output_size / storage_throughput) + storage_latency
                            comm_time = read_time + write_time

                edge_attrs.append([exec_time, latency, is_warm, energy, comm_time])
                
                # Store mapping for decoding
                task_logit_to_placement[t_idx].append((node_id, plat_id))
        
        if not edge_src:
            return None, None
        
        # Build edge tensors
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32) if edge_attrs else torch.empty((0, 5), dtype=torch.float32)
        
        # Make undirected
        num_nodes = n_tasks + n_platforms
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        if edge_attr.numel() > 0:
            edge_attr = torch.cat([edge_attr, edge_attr.clone()], dim=0)
        
        # Create PyG Data object
        data = Data(
            edge_index=edge_index,
            n_tasks=n_tasks,
            n_platforms=n_platforms,
            task_features=task_features_tensor,
            platform_features=platform_features_tensor,
        )
        data.edge_attr = edge_attr
        
        return data, task_logit_to_placement

    def _decode_placements(
        self,
        logits_per_task: List[torch.Tensor],
        task_logit_to_placement: Dict[int, List[Tuple[int, int]]],
        n_tasks: int
    ) -> Dict[int, Tuple[int, int]]:
        """
        Per-task greedy decoder (NON-UNIQUE version):
        For each task, independently select the highest-scoring platform.
        
        Multiple tasks CAN be placed on the same replica (non-unique placements).
        This matches the training decoder in 24-01-15-16_non_unique.py.
        
        Returns: Dict mapping task_idx -> (node_id, platform_id)
        """
        placements = {}
        
        for t_idx in range(n_tasks):
            if t_idx not in task_logit_to_placement:
                continue
            
            logits_t = logits_per_task[t_idx]
            if logits_t.numel() == 0:
                continue
            
            # Pick highest scoring platform for this task (greedy per-task)
            best_logit_idx = logits_t.argmax().item()
            
            if best_logit_idx >= len(task_logit_to_placement[t_idx]):
                continue
            
            node_id, plat_id = task_logit_to_placement[t_idx][best_logit_idx]
            placements[t_idx] = (node_id, plat_id)
        
        return placements

    def _capture_batch_queue_snapshot(self, system_state: SystemState, batch_tasks: List[Task]) -> Dict[str, int]:
        queue_snapshot = {}
        task_types = set(task.type["name"] for task in batch_tasks)
        for task_type in task_types:
            replicas = system_state.replicas.get(task_type, set())
            for node, platform in replicas:
                key = f"{node.node_name}:{platform.id}"
                if key not in queue_snapshot:
                    queue_snapshot[key] = len(platform.queue.items)
        return queue_snapshot

    def _capture_full_queue_snapshot(self) -> Dict[str, int]:
        queue_snapshot = {}
        for node in self.nodes.items:
            for platform in node.platforms.items:
                key = f"{node.node_name}:{platform.id}"
                queue_snapshot[key] = len(platform.queue.items)
        return queue_snapshot

    def placement(self, system_state: SystemState, task: Task) -> Generator:
        if False:
            yield
        return None

    def _get_valid_replicas(self, replicas: Set[Tuple[Node, Platform]], task: Task) -> List[Tuple[Node, Platform]]:
        """Get valid replicas: task's source node + server nodes with network connectivity.
        
        Matches herocache_network logic: only servers (non-client nodes) can receive remote tasks.
        """
        valid_replicas = []
        
        # Find source node to check its network_map
        source_node = None
        for n in self.nodes.items:
            if n.node_name == task.node_name:
                source_node = n
                break
        
        for node, platform in replicas:
            # Include if it's the task's source node (local execution)
            if node.node_name == task.node_name:
                valid_replicas.append((node, platform))
            # Include if it's a server node AND has network connectivity to task source
            elif not node.node_name.startswith('client_node'):
                # Check if source node has network connectivity to this server
                if source_node is not None and hasattr(source_node, 'network_map'):
                    if node.node_name in source_node.network_map:
                        valid_replicas.append((node, platform))
                # Fallback: check bidirectional connectivity
                elif hasattr(node, 'network_map') and task.node_name in node.network_map:
                    valid_replicas.append((node, platform))
        
        return valid_replicas

    def _capture_temporal_state_snapshot(self) -> Dict[str, Dict[str, float]]:
        """
        Capture temporal state (remaining times) for all platforms.
        
        Returns: Dict mapping "node_name:platform_id" -> {
            "current_task_remaining": float,
            "cold_start_remaining": float,
            "comm_remaining": float
        }
        """
        temporal_state = {}
        
        for node in self.nodes.items:
            # Get node storage for communication time calculation
            node_storage = None
            for storage in node.storage.items:
                if not storage.type.get("remote", False):
                    node_storage = storage
                    break
            
            for platform in node.platforms.items:
                key = f"{node.node_name}:{platform.id}"
                
                # Initialize with zeros
                current_task_remaining = 0.0
                cold_start_remaining = 0.0
                comm_remaining = 0.0
                
                if platform.current_task is not None:
                    current_task = platform.current_task
                    now = self.env.now
                    
                    # Current task cold start remaining
                    if current_task.cold_started and not hasattr(current_task, "started_time"):
                        # Task is still in cold start
                        cold_start_duration = current_task.type["coldStartDuration"][platform.type["shortName"]]
                        elapsed_cold_start = now - current_task.arrived_time
                        cold_start_remaining = max(0.0, cold_start_duration - elapsed_cold_start)
                    
                    # Current task execution remaining
                    if hasattr(current_task, "started_time") and current_task.started_time is not None:
                        # Task has started executing
                        exec_duration = current_task.type["executionTime"][platform.type["shortName"]]
                        elapsed_exec = now - current_task.started_time
                        current_task_remaining = max(0.0, exec_duration - elapsed_exec)
                        
                        # Communications remaining (estimate based on output state size)
                        if node_storage and current_task.application:
                            state_size_map = current_task.type.get("stateSize", {})
                            app_name = current_task.application.type.get("name", "")
                            if isinstance(state_size_map, dict) and app_name in state_size_map:
                                output_size = state_size_map[app_name].get("output", 0)
                                if isinstance(output_size, (int, float)) and output_size > 0:
                                    throughput = node_storage.type.get("throughput", {}).get("write", 100.0 * 1024 * 1024)  # bytes/s
                                    latency = node_storage.type.get("latency", {}).get("write", 0.001)  # seconds
                                    comm_remaining = (output_size / throughput) + latency
                
                temporal_state[key] = {
                    "current_task_remaining": current_task_remaining,
                    "cold_start_remaining": cold_start_remaining,
                    "comm_remaining": comm_remaining,
                }
        
        return temporal_state

    def _select_placement_pure_gnn(
        self,
        task: Task,
        task_idx: int,
        placements: Optional[Dict[int, Tuple[int, int]]],
        available_replicas: List[Tuple[Node, Platform]]
    ) -> Tuple[Node, Platform]:
        """
        Select placement using pure GNN decision with shortest-queue fallback.
        This is the original logic, preserved when soft blending is disabled.
        """
        target_node, target_platform = None, None
        
        # Get GNN's placement decision
        gnn_placement = placements.get(task_idx) if placements else None
        
        if gnn_placement:
            target_node_id, target_plat_id = gnn_placement
            # Find the actual node/platform objects
            for node, plat in available_replicas:
                if node.id == target_node_id and plat.id == target_plat_id:
                    target_node, target_platform = node, plat
                    self.gnn_pure_decisions += 1
                    break
        
        # Fallback to shortest queue if GNN placement is invalid
        if target_node is None or target_platform is None:
            print(f"[ {self.env.now} ] GNN: Fallback to shortest queue for task {task.id}")
            target_node, target_platform = min(
                available_replicas, key=lambda couple: len(couple[1].queue.items)
            )
            self.fallback_decisions += 1
        
        return target_node, target_platform
