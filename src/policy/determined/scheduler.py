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
from timeit import default_timer
from typing import Generator, Set, Tuple, TYPE_CHECKING, List, Dict, Any, Optional, cast

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task

from src.placement.model import SystemState
from src.placement.scheduler import Scheduler
from src.policy.state_capture import StateCaptureHelper


class DeterminedScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        logger = logging.getLogger('simulation')
        logger.info("DeterminedScheduler: Starting initialization")
        super().__init__(*args, **kwargs)
        # Default batch size (can be overridden by orchestrator via infrastructure config)
        self.batch_size = 3  # Default: process 2 tasks at once
        # State capture helper (initialized lazily when env/nodes are available)
        self._state_capture: Optional[StateCaptureHelper] = None

    def scheduler_process(self) -> Generator:
        # keep this for the simpy generator 
        if False:
            yield

        """Override to process multiple tasks simultaneously in batches"""
        logger = logging.getLogger('simulation')
        logger.info(f"DeterminedScheduler: scheduler_process started at time {self.env.now}")
        print(
            f"[ {self.env.now} ] Determined Scheduler started with policy"
            f" {self.policy} (batch_size={self.batch_size})"
        )

        while True:
            logger.info(f"DeterminedScheduler: Starting batch collection at time {self.env.now}")
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
        logger = logging.getLogger('simulation')
        batch = []
        
        logger.info(f"DeterminedScheduler: _collect_task_batch starting (batch_size={self.batch_size})")
        print(f"[ {self.env.now} ] DEBUG: Starting batch collection (size={self.batch_size})")
        
        # Try to get up to batch_size tasks
        for i in range(self.batch_size):
            try:
                logger.info(f"DeterminedScheduler: Attempting to get task {i+1}/{self.batch_size}")
                # Try to get a task (this will block until a task is available)
                task: Task = yield self.tasks.get(
                    lambda queued_task: all(
                        dependency.finished for dependency in queued_task.dependencies
                    )
                )
                batch.append(task)
                logger.info(f"DeterminedScheduler: Added task {task.id} to batch (size={len(batch)})")
                print(f"[ {self.env.now} ] DEBUG: Added task {task.id} to batch (size={len(batch)})")
            except:
                # No more tasks available
                # logger.info(f"DeterminedScheduler: No more tasks available after {len(batch)} tasks")
                # print(f"[ {self.env.now} ] DEBUG: No more tasks available after {len(batch)} tasks")
                break
        
        logger.info(f"DeterminedScheduler: Batch collection complete, returning {len(batch)} tasks")
        # print(f"[ {self.env.now} ] DEBUG: Batch collection complete, returning {len(batch)} tasks")
        return batch

    def _process_task_batch(self, batch_tasks: List[Task]) -> Generator:
        """Process multiple tasks simultaneously in a single operation"""
        print(f"[ {self.env.now} ] DEBUG: Processing {len(batch_tasks)} tasks in batch")
        
        # Get system state once for all tasks
        system_state: SystemState = yield self.mutex.get()
        replicas: Dict[str, Set[Tuple[Node, Platform]]] = system_state.replicas
        
        # Capture queue snapshot ONCE for the entire batch (before any placements)
        batch_queue_snapshot = self._capture_batch_queue_snapshot(system_state, batch_tasks)
        
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

            # Set queue snapshot for this task (from the batch snapshot, filtered to valid replicas)
            task_replicas = replicas.get(task.type["name"], set())
            valid_replicas = self._get_valid_replicas(task_replicas, task)
            task.queue_snapshot_at_scheduling = {
                f"{node.node_name}:{plat.id}": batch_queue_snapshot.get(f"{node.node_name}:{plat.id}", 0)
                for node, plat in valid_replicas
            }
            
            # Capture full queue snapshot and temporal state for this task
            task.full_queue_snapshot = self._capture_full_queue_snapshot()
            valid_replicas_set = set(valid_replicas)
            task.temporal_state_at_scheduling = self.state_capture.capture_temporal_state_for_replicas(valid_replicas_set)

            # Update node
            node: Node = yield self.nodes.get(lambda node: node.id == sched_node.id)
            task.node = node
            node.unused = False
            
            # Update platform
            platform: Platform = yield node.platforms.get(lambda platform: platform.id == sched_platform.id)
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

    def _capture_batch_queue_snapshot(self, system_state: SystemState, batch_tasks: List[Task]) -> Dict[str, int]:
        """Capture queue lengths for all platforms across all task types in the batch.
        This is done ONCE before any placements so all tasks see the same queue state."""
        queue_snapshot = {}
        
        # Get all unique task types in the batch
        task_types = set(task.type["name"] for task in batch_tasks)
        
        # Capture queue lengths for all replicas of all task types
        for task_type in task_types:
            replicas = system_state.replicas.get(task_type, set())
            for node, platform in replicas:
                key = f"{node.node_name}:{platform.id}"
                if key not in queue_snapshot:
                    queue_snapshot[key] = len(platform.queue.items)
        
        return queue_snapshot
    
    def _capture_full_queue_snapshot(self) -> Dict[str, int]:
        """Capture queue lengths for ALL platforms across all nodes."""
        queue_snapshot = {}
        for node in self.nodes.items:
            for platform in node.platforms.items:
                key = f"{node.node_name}:{platform.id}"
                queue_snapshot[key] = len(platform.queue.items)
        return queue_snapshot



    def placement(self, system_state: SystemState, task: Task) -> Generator[Any, Any, Optional[Tuple[Node, Platform]]]:
        # Scheduling functions called in a Simpy Process must be Generators
        # No-op as per https://stackoverflow.com/a/68628599/9568489
        if False:
            yield

        # Check for forced placements first
        if self.forced_placements and task.id in self.forced_placements:
            forced_node_id, forced_platform_id = self.forced_placements[task.id]
            
            # Special marker (-1, -1) means auto-resolve to an available replica
            if forced_node_id == -1 and forced_platform_id == -1:
                logger = logging.getLogger('simulation')
                logger.info(f"Auto-resolving forced placement for task {task.id} ({task.type['name']}) at time {self.env.now}")
                print(f"[ {self.env.now} ] DEBUG: Auto-resolving forced placement for task {task.id} ({task.type['name']})")
                # Get available replicas and auto-select one
                replicas_for_task = system_state.replicas.get(task.type["name"], set())
                logger.info(f"Found {len(replicas_for_task)} total replicas for task type {task.type['name']}")
                if not replicas_for_task:
                    logger.error(f"No replicas available for task {task.id} ({task.type['name']})")
                    print(f"[ {self.env.now} ] ERROR: No replicas available for task {task.id} ({task.type['name']})")
                    return None
                
                # Get valid replicas (respecting network connectivity)
                logger.info(f"Getting valid replicas for task {task.id} from source node {task.node_name}")
                valid_replicas = self._get_valid_replicas(replicas_for_task, task)
                logger.info(f"Found {len(valid_replicas)} valid replicas for task {task.id}")
                if not valid_replicas:
                    logger.error(f"No valid replicas for task {task.id} ({task.type['name']}) from source {task.node_name}")
                    print(f"[ {self.env.now} ] ERROR: No valid replicas for task {task.id} ({task.type['name']})")
                    return None
                
                # Select least loaded replica
                target_node, target_platform = min(
                    valid_replicas, key=lambda couple: len(couple[1].queue.items)
                )
                
                print(f"[ {self.env.now} ] DEBUG: Auto-resolved to node {target_node.id}, platform {target_platform.id}")
                task.execution_node = target_node.node_name
                task.execution_platform = str(target_platform.id)
                return (target_node, target_platform)
            
            # Normal forced placement
            print(f"[ {self.env.now} ] DEBUG: Using forced placement for task {task.id}: node {forced_node_id}, platform {forced_platform_id}")
            
            # Find the node and platform by ID
            target_node = None
            target_platform = None
            
            for node in self.nodes.items:
                if node.id == forced_node_id:
                    target_node = node
                    for platform in node.platforms.items:
                        if platform.id == forced_platform_id:
                            target_platform = platform
                            break
                    break

            # Safety 1: the forced (node, platform) must exist in the replica set
            if target_node is not None and target_platform is not None:
                replicas_for_task = system_state.replicas.get(task.type["name"], set())
                if (target_node, target_platform) not in replicas_for_task:
                    # Collect some info information to help diagnose infra / placement mismatches
                    replica_ids = [(n.id, p.id) for (n, p) in replicas_for_task]
                    replica_names = [(n.node_name, p.id) for (n, p) in replicas_for_task]
                    print(
                        f"[ {self.env.now} ] ERROR: Forced placement for task {task.id} "
                        f"({task.type['name']}) points to (node_id={forced_node_id}, "
                        f"platform_id={forced_platform_id}), which is not an active replica "
                        f"for this task type. Active replica ids: {replica_ids}"
                    )
                    print(
                        f"[ {self.env.now} ] ERROR DETAILS: "
                        f"Requested: node_id={forced_node_id}, platform_id={forced_platform_id}, "
                        f"node_name={target_node.node_name if target_node else 'None'}; "
                        f"Active replicas (node_name:platform_id): {replica_names}"
                    )
                    # CRITICAL: Log full system state for dnn2
                    if task.type["name"] == "dnn2":
                        print(f"[ {self.env.now} ] [DEBUG] All replicas in system_state.replicas: {[(task_type, [(n.node_name, n.id, p.id) for (n, p) in replicas]) for task_type, replicas in system_state.replicas.items()]}")
                    raise RuntimeError(
                        f"Invalid forced placement for task {task.id}: "
                        f"(node_id={forced_node_id}, platform_id={forced_platform_id}) "
                        f"is not in replicas for task type '{task.type['name']}'"
                    )

                # Safety 2: platform must be initialized, otherwise the platform_process
                # will block forever on `yield self.initialized`.
                if not target_platform.initialized.triggered:
                    print(
                        f"[ {self.env.now} ] ERROR: Forced placement for task {task.id} "
                        f"({task.type['name']}) targets platform {forced_platform_id} on "
                        f"node {forced_node_id}, but that platform has not been initialized."
                    )
                    raise RuntimeError(
                        f"Forced placement for task {task.id} targets an uninitialized "
                        f"platform (node_id={forced_node_id}, platform_id={forced_platform_id})"
                    )

                # All checks passed – this is a valid deterministic replica
                task.execution_node = target_node.node_name
                task.execution_platform = str(target_platform.id)
                return (target_node, target_platform)
            else:
                print(f"[ {self.env.now} ] ERROR: Forced placement not found: node {forced_node_id}, platform {forced_platform_id}")
                # CRITICAL: Do not fall back to normal placement - this would bypass the brute-force optimization
                # Instead, return None to indicate placement failure, which should trigger proper error handling
                print(f"[ {self.env.now} ] ERROR: Forced placement failed - simulation should abort to prevent invalid results")
                return None
        
        # No forced placement - this should not happen in brute-force mode
        import sys
        print(f"[ {self.env.now} ] ERROR: No forced placement found for task {task.id}")
        sys.exit(1)
        """
        replicas: Set[Tuple[Node, Platform]] = system_state.replicas[task.type["name"]]

        # Get valid replicas: task's source node + server nodes
        valid_replicas = self._get_valid_replicas(replicas, task)

        # Check if we have any valid replicas
        if not valid_replicas:
            # No valid replicas available - this should trigger scaling from zero
            # Return None to indicate no placement possible
            print(f"[ {self.env.now} ] ERROR: No valid replicas for task {task.id} ({task.type['name']})")
            return None

        # Least Connected
        bounded_concurrency = min(
            valid_replicas, key=lambda couple: len(couple[1].queue.items)
        )

        print(f"task: {task.id}")
        print(f"bounded_concurrency: {bounded_concurrency}")

        return bounded_concurrency
        """

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

    # ==================== State Capture Methods ====================
    
    @property
    def state_capture(self) -> StateCaptureHelper:
        """Lazy initialization of state capture helper."""
        if self._state_capture is None:
            self._state_capture = StateCaptureHelper(self.env, self.nodes)
        return self._state_capture
    
    def enable_state_capture(self, output_path: str):
        """Enable state capture and set output path."""
        self.state_capture.enable_capture(output_path)
    
    def disable_state_capture(self):
        """Disable state capture."""
        self.state_capture.disable_capture()
    
    def capture_task_placement(
        self,
        task: 'Task',
        execution_node: str,
        execution_platform: str,
        elapsed_time: float,
        valid_replicas: List[Tuple['Node', 'Platform']]
    ) -> Dict[str, Any]:
        """
        Capture a task placement decision with full state information.
        
        Args:
            task: The task being placed
            execution_node: Node where task will execute
            execution_platform: Platform ID where task will execute
            elapsed_time: Wall-clock time for scheduling decision
            valid_replicas: Set of valid replicas for this task
            
        Returns:
            Dict with placement information
        """
        # Calculate queue time
        queue_time = self.env.now - task.arrived_time if hasattr(task, 'arrived_time') else 0.0
        
        # Capture queue snapshots
        valid_replicas_set = set(valid_replicas)
        queue_snapshot_at_scheduling = self.state_capture.capture_queue_snapshot_for_replicas(valid_replicas_set)
        full_queue_snapshot = self.state_capture.capture_full_queue_snapshot()
        
        # Capture temporal state
        temporal_state_at_scheduling = self.state_capture.capture_temporal_state_for_replicas(valid_replicas_set)
        
        return self.state_capture.capture_task_placement(
            task=task,
            execution_node=execution_node,
            execution_platform=execution_platform,
            elapsed_time=elapsed_time,
            queue_time=queue_time,
            queue_snapshot_at_scheduling=queue_snapshot_at_scheduling,
            full_queue_snapshot=full_queue_snapshot,
            temporal_state_at_scheduling=temporal_state_at_scheduling,
        )
    
    def save_captured_state(self, system_state: 'SystemState', total_rtt: float = 0.0, output_path: Optional[str] = None):
        """Save captured state to JSON file."""
        self.state_capture.save_captured_state(system_state, total_rtt, output_path)
    
    def get_captured_state(self, system_state: 'SystemState', total_rtt: float = 0.0) -> Dict[str, Any]:
        """Get captured state as dictionary."""
        return self.state_capture.get_captured_state(system_state, total_rtt)
    
    def reset_state_capture(self):
        """Reset captured placements for a new simulation run."""
        self.state_capture.reset()
