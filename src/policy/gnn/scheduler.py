"""
GNN-based Scheduler for Task-to-Platform Placement

This scheduler uses a trained GNN model to make placement decisions.
It processes all tasks in a batch together (single forward pass) and
uses a greedy decoder to ensure unique placements.
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
QUEUE_NORM_FACTOR = 10.0


class GNNScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Batch processing now works with batch_size > 1
        # We pre-scan the batch and create needed replicas upfront before placement
        self.batch_size = 5
        self.batch_timeout = 0.01
        
        # GNN model will be set via models dict from orchestrator
        self.gnn_model = None
        self.device = None
        self.task_types_data = None
        self.dataset_id = None
        
        # === GNN Configuration ===
        # Use pure GNN with simple fallback (no soft blending)
        # Adaptive queue normalization will be calculated per batch
        
        # Stats tracking for debugging
        self.gnn_pure_decisions = 0
        self.fallback_decisions = 0

    def set_models(self, models: Dict[str, Any]):
        """Set the GNN model and related data from executor."""
        if models:
            self.gnn_model = models.get('gnn_model')
            self.task_types_data = models.get('task_types_data')
            self.dataset_id = models.get('dataset_id')
            if self.gnn_model is not None:
                self.device = next(self.gnn_model.parameters()).device
                print(f"[GNN Scheduler] Model loaded on {self.device}")
    
    def configure_scheduler(self, scheduler_config: Dict[str, Any]):
        """
        Configure scheduler parameters from scheduler config.
        
        Example config:
        {
            "batch_size": 5,
            "batch_timeout": 0.01
        }
        """
        if scheduler_config:
            self.batch_size = scheduler_config.get('batch_size', self.batch_size)
            self.batch_timeout = scheduler_config.get('batch_timeout', self.batch_timeout)
                
            print(f"[GNN Scheduler] Configured: batch_size={self.batch_size}, batch_timeout={self.batch_timeout}")

    def scheduler_process(self) -> Generator:
        if False:
            yield

        print(
            f"[ {self.env.now} ] GNN Scheduler started with policy"
            f" {self.policy} (batch_size={self.batch_size})"
        )

        while True:
            batch_tasks = yield self.env.process(self._collect_task_batch())
            
            if not batch_tasks:
                yield self.env.timeout(0.1)
                continue

            print(f"[ {self.env.now} ] GNN: Processing batch of {len(batch_tasks)} tasks")
            yield self.env.process(self._process_task_batch(batch_tasks))

    def _collect_task_batch(self) -> Generator[Any, Any, List[Task]]:
        """
        Collect tasks into a batch with timeout to avoid blocking.
        
        Strategy:
        1. Wait for at least one task (blocking)
        2. Poll for more tasks until timeout
        3. Return whatever was collected (even partial batches)
        
        NOTE: We use a polling approach instead of yield get|timeout to avoid
        the "dangling get" problem where unfulfilled get events consume future tasks.
        """
        batch: List[Task] = []
        
        def task_filter(queued_task):
            return all(dependency.finished for dependency in queued_task.dependencies)
        
        # First task: wait indefinitely (blocking is expected)
        task: Task = yield self.tasks.get(task_filter)
        batch.append(task)
        
        # Set deadline for collecting rest of batch
        batch_deadline = self.env.now + self.batch_timeout
        
        # Collect remaining tasks using polling (avoids dangling get events)
        while len(batch) < self.batch_size:
            remaining_time = batch_deadline - self.env.now
            if remaining_time <= 0:
                break
            
            # Check if there are any ready tasks in the queue
            ready_tasks = [t for t in self.tasks.items if task_filter(t)]
            
            if ready_tasks:
                # Get the first ready task immediately
                task = yield self.tasks.get(task_filter)
                batch.append(task)
            else:
                # No ready tasks - wait a small step and check again
                # Use small timeout to avoid busy-waiting
                step_time = min(0.01, remaining_time)
                yield self.env.timeout(step_time)
        
        return batch

    def _ensure_replicas_for_batch(
        self, 
        batch_tasks: List[Task], 
        system_state: SystemState
    ) -> Generator[Any, Any, Dict[str, int]]:
        """
        Pre-scan batch and ensure enough replicas exist for all task types.
        
        This prevents the cascade problem where each task in a batch creates
        a new replica because previous replicas are already used.
        
        IMPORTANT: After creating replicas, we wait for them to initialize.
        Otherwise tasks get placed on uninitialized replicas and hang.
        
        Returns: Dict mapping task_type -> number of replicas created
        """
        from collections import Counter
        
        # Count how many tasks of each type are in the batch
        task_type_counts: Dict[str, int] = Counter(task.type["name"] for task in batch_tasks)
        replicas_created: Dict[str, int] = {}
        
        # Track newly created replicas so we can wait for initialization
        new_replicas: List[Tuple[Node, Platform]] = []
        
        for task_type_name, needed_count in task_type_counts.items():
            current_replicas = system_state.replicas.get(task_type_name, set())
            # Track which replicas existed before we started
            existing_replica_ids = {(n.id, p.id) for n, p in current_replicas}
            current_count = len(current_replicas)
            
            # Calculate how many additional replicas we need
            # We want at least as many replicas as tasks in the batch
            deficit = needed_count - current_count
            
            if deficit <= 0:
                replicas_created[task_type_name] = 0
                continue
            
            print(f"[ {self.env.now} ] GNN: Pre-creating {deficit} replica(s) for {task_type_name} "
                  f"(have {current_count}, need {needed_count} for batch)")
            
            # Get task type data
            task_type = self.data.task_types.get(task_type_name)
            if not task_type:
                logging.warning(f"[ {self.env.now} ] Unknown task type: {task_type_name}")
                replicas_created[task_type_name] = 0
                continue
            
            # Find source nodes from tasks of this type (for network connectivity)
            source_nodes = set(
                task.node_name for task in batch_tasks 
                if task.type["name"] == task_type_name
            )
            
            created = 0
            for _ in range(deficit):
                # Try to create replica with connectivity to at least one source node
                replica_created = False
                for source_node in source_nodes:
                    stop = yield self.env.process(
                        self.autoscaler.create_first_replica(
                            system_state,
                            task_type,
                            source_node_name=source_node
                        )
                    )
                    
                    if not isinstance(stop, StopIteration):
                        created += 1
                        replica_created = True
                        break
                
                if not replica_created:
                    # Couldn't create with any source node, try without constraint
                    stop = yield self.env.process(
                        self.autoscaler.create_first_replica(
                            system_state,
                            task_type
                        )
                    )
                    if not isinstance(stop, StopIteration):
                        created += 1
                    else:
                        logging.warning(
                            f"[ {self.env.now} ] GNN: Could not create more replicas for {task_type_name} "
                            f"(created {created}/{deficit})"
                        )
                        break
            
            # Find newly created replicas (those not in existing_replica_ids)
            for node, plat in system_state.replicas.get(task_type_name, set()):
                if (node.id, plat.id) not in existing_replica_ids:
                    new_replicas.append((node, plat))
            
            replicas_created[task_type_name] = created
            if created > 0:
                print(f"[ {self.env.now} ] GNN: Pre-created {created} replica(s) for {task_type_name}")
        
        # Don't block waiting for initialization - let tasks use replicas when ready
        # This allows batch processing to continue while replicas initialize in background
        # Tasks will automatically use initialized replicas when available
        if new_replicas:
            print(f"[ {self.env.now} ] GNN: Created {len(new_replicas)} replica(s), allowing initialization in background")
        
        return replicas_created

    def _process_task_batch(self, batch_tasks: List[Task]) -> Generator:
        print(f"[ {self.env.now} ] GNN: Processing {len(batch_tasks)} tasks in batch")
        
        system_state: SystemState = yield self.mutex.get()
        
        # Pre-scan: ensure enough replicas exist for all task types in batch
        # This prevents cascade replica creation during batch processing
        replicas_created = yield self.env.process(
            self._ensure_replicas_for_batch(batch_tasks, system_state)
        )
        if any(count > 0 for count in replicas_created.values()):
            print(f"[ {self.env.now} ] GNN: Pre-created replicas: {replicas_created}")
        
        # Capture queue snapshots
        full_queue_snapshot = self._capture_full_queue_snapshot()
        batch_queue_snapshot = self._capture_batch_queue_snapshot(system_state, batch_tasks)
        
        # Calculate adaptive queue normalization based on current max queue length
        adaptive_queue_norm = self._calculate_adaptive_queue_norm(full_queue_snapshot)
        
        # Build graph and run GNN inference (always use GNN)
        inference_start = default_timer()
        
        placements = self._gnn_inference(
            batch_tasks, system_state, full_queue_snapshot, adaptive_queue_norm
        )
        
        inference_time = default_timer() - inference_start
        print(f"[ {self.env.now} ] GNN inference time: {inference_time*1000:.2f}ms (adaptive_queue_norm={adaptive_queue_norm:.1f})")
        
        # Track used platforms within this batch
        used_platforms: Set[Tuple[int, int]] = set()
        
        for task_idx, task in enumerate(batch_tasks):
            task_replicas = system_state.replicas.get(task.type["name"], set())

            if not task_replicas:
                logging.warning(f"[ {self.env.now} ] Scheduler did not find available replica for {task}")
                task.postponed_count += 1
                
                # Limit postponements to prevent infinite loops
                if task.postponed_count > 9999999:
                    logging.error(f"[ {self.env.now} ] Task {task.id} postponed too many times, marking as failed")
                    # Mark task as failed and trigger all events so task_process can complete
                    task.finished = True
                    task.postponed_count = 9999999  # Set to unreal number to indicate permanent failure
                    # Trigger all events in sequence so task_process can progress to done
                    if not task.scheduled.triggered:
                        task.scheduled.succeed()
                    if not task.arrived.triggered:
                        task.arrived.succeed()
                    if not task.started.triggered:
                        task.started.succeed()
                    if not task.done.triggered:
                        task.done.succeed()
                    continue
                
                yield self.tasks.put(task)
                stop = yield self.env.process(
                    self.autoscaler.create_first_replica(
                        system_state, 
                        task.type,
                        source_node_name=task.node_name  # Pass source node to check connectivity
                    )
                )
                
                # Check if replica creation succeeded
                if isinstance(stop, StopIteration):
                    logging.warning(f"[ {self.env.now} ] Failed to create replica for {task.type['name']}")
                    # If replica creation failed and we've tried many times, mark as failed
                    if task.postponed_count > 9999999:
                        logging.error(f"[ {self.env.now} ] Task {task.id} cannot create replica after {task.postponed_count} attempts, marking as failed")
                        task.finished = True
                        task.postponed_count = 9999999  # Set to unreal number to indicate permanent failure
                        # Trigger all events in sequence so task_process can progress to done
                        if not task.scheduled.triggered:
                            task.scheduled.succeed()
                        if not task.arrived.triggered:
                            task.arrived.succeed()
                        if not task.started.triggered:
                            task.started.succeed()
                        if not task.done.triggered:
                            task.done.succeed()
                        continue
                
                self.env.step()
                continue

            start = default_timer()

            valid_replicas = self._get_valid_replicas(task_replicas, task)
            
            if not valid_replicas:
                # Diagnostic: log why no valid replicas
                total_replicas = len(task_replicas)
                logging.warning(
                    f"[ {self.env.now} ] GNN ERROR: No valid replicas for task {task.id} "
                    f"({task.type['name']}) from {task.node_name}. "
                    f"Total replicas for {task.type['name']}: {total_replicas}"
                )
                task.postponed_count += 1
                
                # Limit postponements to prevent infinite loops
                if task.postponed_count > 9999999:
                    logging.error(f"[ {self.env.now} ] Task {task.id} postponed too many times, marking as failed")
                    # Mark task as failed and trigger all events so task_process can complete
                    task.finished = True
                    task.postponed_count = 9999999  # Set to unreal number to indicate permanent failure
                    # Trigger all events in sequence so task_process can progress to done
                    if not task.scheduled.triggered:
                        task.scheduled.succeed()
                    if not task.arrived.triggered:
                        task.arrived.succeed()
                    if not task.started.triggered:
                        task.started.succeed()
                    if not task.done.triggered:
                        task.done.succeed()
                    continue
                
                # Try creating a replica on a node with network connectivity
                # This helps when replicas exist but none are reachable from the task's source node
                yield self.tasks.put(task)
                stop = yield self.env.process(
                    self.autoscaler.create_first_replica(
                        system_state, 
                        task.type,
                        source_node_name=task.node_name  # Pass source node to check connectivity
                    )
                )
                
                # Check if replica creation succeeded
                if isinstance(stop, StopIteration):
                    logging.warning(f"[ {self.env.now} ] Failed to create replica for {task.type['name']}")
                    # If replica creation failed and we've tried many times, mark as failed
                    if task.postponed_count > 999999:
                        logging.error(f"[ {self.env.now} ] Task {task.id} cannot create replica after {task.postponed_count} attempts, marking as failed")
                        task.finished = True
                        task.postponed_count = 999999  # Set to unreal number to indicate permanent failure
                        # Trigger all events in sequence so task_process can progress to done
                        if not task.scheduled.triggered:
                            task.scheduled.succeed()
                        if not task.arrived.triggered:
                            task.arrived.succeed()
                        if not task.started.triggered:
                            task.started.succeed()
                        if not task.done.triggered:
                            task.done.succeed()
                        continue
                
                self.env.step()
                continue

            # Filter out already-used platforms
            available_replicas = [
                (node, plat) for node, plat in valid_replicas 
                if (node.id, plat.id) not in used_platforms
            ]
            
            # Prefer initialized replicas - they can process tasks immediately
            # But still allow placement on uninitialized replicas (they'll wait for init)
            initialized_replicas = [
                (node, plat) for node, plat in available_replicas 
                if plat.initialized.triggered
            ]
            
            # Use initialized replicas if available, otherwise use any available
            if initialized_replicas:
                available_replicas = initialized_replicas
            
            if not available_replicas:
                print(f"[ {self.env.now} ] GNN ERROR: No unique replicas left for task {task.id}")
                task.postponed_count += 1
                
                # Limit postponements to prevent infinite loops
                if task.postponed_count > 9999999:
                    logging.error(f"[ {self.env.now} ] Task {task.id} postponed too many times, marking as failed")
                    # Mark task as failed and trigger all events so task_process can complete
                    task.finished = True
                    task.postponed_count = 9999999  # Set to unreal number to indicate permanent failure
                    # Trigger all events in sequence so task_process can progress to done
                    if not task.scheduled.triggered:
                        task.scheduled.succeed()
                    if not task.arrived.triggered:
                        task.arrived.succeed()
                    if not task.started.triggered:
                        task.started.succeed()
                    if not task.done.triggered:
                        task.done.succeed()
                    continue
                
                # All valid replicas are used in this batch - try creating another replica
                yield self.tasks.put(task)
                stop = yield self.env.process(
                    self.autoscaler.create_first_replica(
                        system_state, 
                        task.type,
                        source_node_name=task.node_name  # Pass source node to check connectivity
                    )
                )
                
                # Check if replica creation succeeded
                if isinstance(stop, StopIteration):
                    logging.warning(f"[ {self.env.now} ] Failed to create replica for {task.type['name']}")
                    # If replica creation failed and we've tried many times, mark as failed
                    if task.postponed_count > 999999:
                        logging.error(f"[ {self.env.now} ] Task {task.id} cannot create replica after {task.postponed_count} attempts, marking as failed")
                        task.finished = True
                        task.postponed_count = 999999  # Set to unreal number to indicate permanent failure
                        # Trigger all events in sequence so task_process can progress to done
                        if not task.scheduled.triggered:
                            task.scheduled.succeed()
                        if not task.arrived.triggered:
                            task.arrived.succeed()
                        if not task.started.triggered:
                            task.started.succeed()
                        if not task.done.triggered:
                            task.done.succeed()
                        continue
                
                self.env.step()
                continue

            # Store queue snapshots on task
            task.queue_snapshot_at_scheduling = {
                f"{node.node_name}:{plat.id}": batch_queue_snapshot.get(f"{node.node_name}:{plat.id}", 0)
                for node, plat in available_replicas
            }
            task.full_queue_snapshot = full_queue_snapshot

            # Select placement using pure GNN with simple fallback
            target_node, target_platform = self._select_placement_pure_gnn(
                task, task_idx, placements, available_replicas
            )

            # Safety check: fallback to shortest queue if GNN placement is invalid
            if target_node is None or target_platform is None:
                print(f"[ {self.env.now} ] GNN: Fallback to shortest queue for task {task.id} (GNN returned None)")
                target_node, target_platform = min(
                    available_replicas, key=lambda couple: len(couple[1].queue.items)
                )
                self.fallback_decisions += 1

            # Mark this platform as used
            used_platforms.add((target_node.id, target_platform.id))

            task.execution_node = target_node.node_name
            task.execution_platform = str(target_platform.id)
            task.gnn_decision_time = inference_time / len(batch_tasks)  # Amortized

            node: Node = yield self.nodes.get(lambda node: node.id == target_node.id)
            task.node = node
            node.unused = False
            
            platform: Platform = yield node.platforms.get(lambda platform: platform.id == target_platform.id)
            task.platform = platform

            end = default_timer()
            elapsed_clock_time = end - start
            node.wall_clock_scheduling_time += elapsed_clock_time

            yield platform.queue.put(task)
            yield task.scheduled.succeed()

            yield node.platforms.put(platform)
            yield self.nodes.put(node)

            print(f"GNN placed task {task.id} on ({target_node.node_name}, platform {target_platform.id})")

        yield self.mutex.put(system_state)
        print(f"[ {self.env.now} ] GNN: Batch processing complete for {len(batch_tasks)} tasks")

    def _gnn_inference(
        self,
        batch_tasks: List[Task],
        system_state: SystemState,
        queue_snapshot: Dict[str, int],
        queue_norm: float
    ) -> Optional[Dict[int, Tuple[int, int]]]:
        """
        Run GNN inference on the batch of tasks.
        
        Args:
            queue_norm: Adaptive queue normalization factor for this batch
        
        Returns: Dict mapping task_idx -> (node_id, platform_id)
        """
        if self.gnn_model is None:
            print("[GNN] Model not loaded, using fallback")
            return None
        
        try:
            # Build graph from current system state
            graph, task_logit_to_placement = self._build_inference_graph(
                batch_tasks, system_state, queue_snapshot, queue_norm
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

    def _build_inference_graph(
        self,
        batch_tasks: List[Task],
        system_state: SystemState,
        queue_snapshot: Dict[str, int],
        queue_norm: float
    ) -> Tuple[Optional[Data], Optional[Dict[int, List[Tuple[int, int]]]]]:
        """
        Build a PyG graph from current system state for GNN inference.
        
        Returns: (graph, task_logit_to_placement mapping)
        """
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
        
        # Platform features: [type_onehot(5), has_dnn1_replica(1), has_dnn2_replica(1), queue_length(1)]
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
            # Queue length (normalized using adaptive normalization)
            queue_key = f"{node_name}:{plat_id}"
            queue_len = queue_snapshot.get(queue_key, 0) / queue_norm
            
            platform_features.append(onehot + [has_dnn1, has_dnn2, queue_len])
        
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
                
                # Edge attributes: [exec_time, latency, is_warm]
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
                
                edge_attrs.append([exec_time, latency, is_warm])
                
                # Store mapping for decoding
                task_logit_to_placement[t_idx].append((node_id, plat_id))
        
        if not edge_src:
            return None, None
        
        # Build edge tensors
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32) if edge_attrs else torch.empty((0, 3), dtype=torch.float32)
        
        # Make undirected
        num_nodes = n_tasks + n_platforms
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        if edge_attr.numel() > 0:
            edge_attr = torch.cat([edge_attr, torch.zeros_like(edge_attr)], dim=0)
        
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
        Greedy decoder: select edges in descending score order,
        enforcing uniqueness (each platform used at most once).
        
        Returns: Dict mapping task_idx -> (node_id, platform_id)
        """
        # Collect all (score, task_idx, logit_idx) candidates
        candidates = []
        for t in range(n_tasks):
            if t not in task_logit_to_placement:
                continue
            logits_t = logits_per_task[t]
            for logit_idx in range(logits_t.numel()):
                score = float(logits_t[logit_idx].item())
                candidates.append((score, t, logit_idx))
        
        if not candidates:
            return {}
        
        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        assigned_tasks = set()
        used_platforms = set()
        placements = {}
        
        for score, t_idx, logit_idx in candidates:
            if t_idx in assigned_tasks:
                continue
            
            if logit_idx >= len(task_logit_to_placement[t_idx]):
                continue
            
            node_id, plat_id = task_logit_to_placement[t_idx][logit_idx]
            
            if (node_id, plat_id) in used_platforms:
                continue
            
            assigned_tasks.add(t_idx)
            used_platforms.add((node_id, plat_id))
            placements[t_idx] = (node_id, plat_id)
            
            if len(assigned_tasks) == n_tasks:
                break
        
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

    def _calculate_adaptive_queue_norm(self, queue_snapshot: Dict[str, int]) -> float:
        """
        Calculate adaptive queue normalization based on current max queue length.
        
        This ensures queue normalization matches the actual queue lengths in the system,
        preventing the normalization from being too small (causing queue to dominate)
        or too large (causing queue to be ignored).
        
        Args:
            queue_snapshot: Dictionary mapping platform keys to queue lengths
            
        Returns:
            Adaptive normalization factor (max_queue * 1.1, minimum 10.0)
        """
        if not queue_snapshot:
            return 10.0  # Default fallback
        
        max_queue = max(queue_snapshot.values())
        # Use max queue + 10% buffer, with minimum of 10.0
        # This ensures normalization scales with actual load
        adaptive_norm = max(max_queue * 1.1, 10.0)
        return adaptive_norm

    def placement(self, system_state: SystemState, task: Task) -> Generator:
        if False:
            yield
        return None

    def _get_valid_replicas(self, replicas: Set[Tuple[Node, Platform]], task: Task) -> List[Tuple[Node, Platform]]:
        valid_replicas = []
        for node, platform in replicas:
            if node.node_name == task.node_name:
                valid_replicas.append((node, platform))
            elif not node.node_name.startswith('client_node'):
                if hasattr(node, 'network_map') and task.node_name in node.network_map:
                    valid_replicas.append((node, platform))
        return valid_replicas


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
