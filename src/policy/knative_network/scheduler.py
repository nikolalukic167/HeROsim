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
from typing import Generator, List, Set, Tuple, TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task

from src.placement.model import SystemState
from src.placement.scheduler import Scheduler


class KnativeNetworkScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_batch_size = 5  # Maximum batch size (safety limit)

    def scheduler_process(self) -> Generator:
        if False:
            yield

        print(
            f"[ {self.env.now} ] KnativeNetwork Scheduler started with policy"
            f" {self.policy}"
        )
        print(f"[ {self.env.now} ] SCHEDULER: Process started")

        while True:
            print(f"[ {self.env.now} ] SCHEDULER: Starting batch collection")
            batch_tasks = yield self.env.process(self._collect_dynamic_batch())
            
            if not batch_tasks:
                print(f"[ {self.env.now} ] SCHEDULER: No tasks collected, waiting")
                yield self.env.timeout(0.1)
                continue

            print(f"[ {self.env.now} ] SCHEDULER: Collected batch of {len(batch_tasks)} tasks")
            yield self.env.process(self._process_task_batch(batch_tasks))
            print(f"[ {self.env.now} ] SCHEDULER: Batch processing complete")

    def _collect_dynamic_batch(self) -> Generator[Any, Any, List[Task]]:
        """Collect tasks dynamically based on available unique replicas.
        
        Collects tasks until:
        - We reach max_batch_size, OR
        - We encounter a task type that has no replicas, OR
        - We encounter a task type where we've already batched all available unique replicas
        """
        batch = []
        system_state: SystemState = yield self.mutex.get()
        print(f"[ {self.env.now} ] BATCH_COLLECT: Started, tasks queue size: {len(self.tasks.items)}")
        
        # Track how many tasks of each type we've collected (to ensure we don't exceed unique replica count)
        task_type_counts: Dict[str, int] = {}
        
        try:
            while len(batch) < self.max_batch_size:
                # Try to get a task
                print(f"[ {self.env.now} ] BATCH_COLLECT: Attempting to get task (batch size: {len(batch)})")
                task: Task = yield self.tasks.get(
                    lambda queued_task: all(
                        dependency.finished for dependency in queued_task.dependencies
                    )
                )
                print(f"[ {self.env.now} ] BATCH_COLLECT: Got task {task.id} ({task.type['name']})")
                
                task_type = task.type["name"]
                task_replicas = system_state.replicas.get(task_type, set())
                
                # If no replicas, add to batch (limit 1 per task type) so autoscaler can create first replica
                if not task_replicas:
                    # Check if we already have a task of this type in the batch
                    if any(t.type["name"] == task_type for t in batch):
                        # Already have one of this type, put this back and return current batch
                        print(f"[ {self.env.now} ] BATCH_COLLECT: Already have task type {task_type} in batch, returning batch of {len(batch)}")
                        yield self.tasks.put(task)
                        yield self.mutex.put(system_state)
                        return batch
                    # Add to batch so autoscaler can handle it
                    batch.append(task)
                    print(f"[ {self.env.now} ] BATCH_COLLECT: Added task {task.id} (no replicas) to batch for autoscaler (size: {len(batch)})")
                    continue
                
                # Get valid replicas for this task
                valid_replicas = self._get_valid_replicas(task_replicas, task)
                
                # If no valid replicas, add to batch (limit 1 per task type) so autoscaler can create replica
                if not valid_replicas:
                    # Check if we already have a task of this type in the batch
                    if any(t.type["name"] == task_type for t in batch):
                        # Already have one of this type, put this back and return current batch
                        print(f"[ {self.env.now} ] BATCH_COLLECT: Already have task type {task_type} in batch, returning batch of {len(batch)}")
                        yield self.tasks.put(task)
                        yield self.mutex.put(system_state)
                        return batch
                    # Add to batch so autoscaler can handle it
                    batch.append(task)
                    print(f"[ {self.env.now} ] BATCH_COLLECT: Added task {task.id} (no valid replicas) to batch for autoscaler (size: {len(batch)})")
                    continue
                
                # Count unique replicas for this task type
                unique_replica_count = len(valid_replicas)
                
                # Check how many tasks of this type we've already collected
                existing_count = task_type_counts.get(task_type, 0)
                
                # If we've already collected enough tasks of this type to use all unique replicas, stop
                if existing_count >= unique_replica_count:
                    print(f"[ {self.env.now} ] BATCH_COLLECT: Task type {task_type} limit reached ({existing_count}/{unique_replica_count}), returning batch of {len(batch)}")
                    # Put task back and return current batch
                    yield self.tasks.put(task)
                    yield self.mutex.put(system_state)
                    return batch
                
                # Add task to batch and update count
                batch.append(task)
                task_type_counts[task_type] = existing_count + 1
                print(f"[ {self.env.now} ] BATCH_COLLECT: Added task {task.id} to batch (size: {len(batch)}, type {task_type}: {task_type_counts[task_type]}/{unique_replica_count})")
                
        except Exception as e:
            # No more tasks available
            print(f"[ {self.env.now} ] BATCH_COLLECT: Exception occurred: {e}, returning batch of {len(batch)}")
        
        print(f"[ {self.env.now} ] BATCH_COLLECT: Completed, returning batch of {len(batch)} tasks")
        yield self.mutex.put(system_state)
        return batch

    def _process_task_batch(self, batch_tasks: List[Task]) -> Generator:
        """Process a batch of tasks, ensuring unique replica placements"""
        print(f"[ {self.env.now} ] BATCH_PROCESS: Starting to process {len(batch_tasks)} tasks")
        system_state: SystemState = yield self.mutex.get()
        
        full_queue_snapshot = self._capture_full_queue_snapshot()
        
        # Track used platforms within this batch to ensure unique placements
        used_platforms: Set[Tuple[int, int]] = set()
        
        scheduled_count = 0
        postponed_count = 0
        failed_count = 0
        
        for task_idx, task in enumerate(batch_tasks):
            print(f"[ {self.env.now} ] BATCH_PROCESS: Processing task {task.id} ({task.type['name']})")
            task_replicas = system_state.replicas.get(task.type["name"], set())

            if not task_replicas:
                task.postponed_count += 1
                
                # Check if hardware is available before postponing
                # If no hardware is available, fail immediately instead of postponing 100 times
                hardware_available = self._check_hardware_available(system_state, task.type, task.node_name)
                
                if not hardware_available:
                    logging.error(
                        f"[ {self.env.now} ] TASK FAILURE: Task {task.id} ({task.type['name']}) from {task.node_name} "
                        f"cannot be scheduled - no available hardware. Marking as failed immediately."
                    )
                    task.failed = True
                    task.finished = True
                    task.failure_reason = f"No available hardware for {task.type['name']} from {task.node_name}"
                    if not task.scheduled.triggered:
                        task.scheduled.succeed()
                    if not task.arrived.triggered:
                        task.arrived.succeed()
                    if not task.started.triggered:
                        task.started.succeed()
                    if not task.done.triggered:
                        task.done.succeed()
                    failed_count += 1
                    continue
                
                if task.postponed_count >= 100:
                    logging.error(
                        f"[ {self.env.now} ] TASK FAILURE: Task {task.id} ({task.type['name']}) from {task.node_name} "
                        f"postponed {task.postponed_count} times. Marking as failed."
                    )
                    task.failed = True
                    task.finished = True
                    task.failure_reason = f"Unable to create replica after {task.postponed_count} attempts"
                    if not task.scheduled.triggered:
                        task.scheduled.succeed()
                    if not task.arrived.triggered:
                        task.arrived.succeed()
                    if not task.started.triggered:
                        task.started.succeed()
                    if not task.done.triggered:
                        task.done.succeed()
                    failed_count += 1
                    continue
                
                postponed_count += 1
                yield self.mutex.put(system_state)
                yield self.tasks.put(task)
                # Strategy 2: Non-blocking autoscaling - fire and forget
                self.env.process(
                    self.autoscaler.create_first_replica(system_state, task.type, source_node_name=task.node_name)
                )
                # Strategy 3: State refresh - re-acquire mutex to get fresh state, then continue processing batch
                system_state = yield self.mutex.get()
                # With non-blocking autoscaling, we can continue processing the rest of the batch
                continue

            start = default_timer()
            valid_replicas = self._get_valid_replicas(task_replicas, task)
            
            if not valid_replicas:
                task.postponed_count += 1
                
                # Check if hardware is available before postponing
                # If no hardware is available, fail immediately instead of postponing 100 times
                hardware_available = self._check_hardware_available(system_state, task.type, task.node_name)
                
                if not hardware_available:
                    logging.error(
                        f"[ {self.env.now} ] TASK FAILURE: Task {task.id} ({task.type['name']}) from {task.node_name} "
                        f"cannot be scheduled - no available hardware. Marking as failed immediately."
                    )
                    task.failed = True
                    task.finished = True
                    task.failure_reason = f"No available hardware for {task.type['name']} from {task.node_name}"
                    if not task.scheduled.triggered:
                        task.scheduled.succeed()
                    if not task.arrived.triggered:
                        task.arrived.succeed()
                    if not task.started.triggered:
                        task.started.succeed()
                    if not task.done.triggered:
                        task.done.succeed()
                    failed_count += 1
                    continue
                
                if task.postponed_count >= 100:
                    logging.error(
                        f"[ {self.env.now} ] TASK FAILURE: Task {task.id} ({task.type['name']}) from {task.node_name} "
                        f"postponed {task.postponed_count} times. Marking as failed."
                    )
                    task.failed = True
                    task.finished = True
                    task.failure_reason = f"No valid replicas found after {task.postponed_count} attempts"
                    if not task.scheduled.triggered:
                        task.scheduled.succeed()
                    if not task.arrived.triggered:
                        task.arrived.succeed()
                    if not task.started.triggered:
                        task.started.succeed()
                    if not task.done.triggered:
                        task.done.succeed()
                    failed_count += 1
                    continue
                
                postponed_count += 1
                yield self.mutex.put(system_state)
                yield self.tasks.put(task)
                # Strategy 2: Non-blocking autoscaling - fire and forget
                self.env.process(
                    self.autoscaler.create_first_replica(system_state, task.type, source_node_name=task.node_name)
                )
                # Strategy 3: State refresh - re-acquire mutex to get fresh state, then continue processing batch
                system_state = yield self.mutex.get()
                # With non-blocking autoscaling, we can continue processing the rest of the batch
                continue

            # Filter out already-used platforms AND uninitialized platforms to ensure unique placements
            # CRITICAL: Only schedule on initialized platforms to prevent deadlock
            available_replicas = [
                (node, plat) for node, plat in valid_replicas 
                if (node.id, plat.id) not in used_platforms
                and plat.initialized.triggered  # Only schedule on initialized platforms
            ]
            
            if not available_replicas:
                task.postponed_count += 1
                
                if task.postponed_count >= 100:
                    logging.error(
                        f"[ {self.env.now} ] TASK FAILURE: Task {task.id} ({task.type['name']}) from {task.node_name} "
                        f"postponed {task.postponed_count} times. Marking as failed."
                    )
                    task.failed = True
                    task.finished = True
                    task.failure_reason = f"No unique replicas available after {task.postponed_count} attempts"
                    if not task.scheduled.triggered:
                        task.scheduled.succeed()
                    if not task.arrived.triggered:
                        task.arrived.succeed()
                    if not task.started.triggered:
                        task.started.succeed()
                    if not task.done.triggered:
                        task.done.succeed()
                    failed_count += 1
                    continue
                
                postponed_count += 1
                yield self.mutex.put(system_state)
                yield self.tasks.put(task)
                # Strategy 2: Non-blocking autoscaling - fire and forget
                self.env.process(
                    self.autoscaler.create_first_replica(system_state, task.type, source_node_name=task.node_name)
                )
                # Strategy 3: State refresh - re-acquire mutex to get fresh state, then continue processing batch
                system_state = yield self.mutex.get()
                # With non-blocking autoscaling, we can continue processing the rest of the batch
                continue

            task.queue_snapshot_at_scheduling = {
                f"{node.node_name}:{plat.id}": full_queue_snapshot.get(f"{node.node_name}:{plat.id}", 0)
                for node, plat in available_replicas
            }
            task.full_queue_snapshot = full_queue_snapshot

            # Least Connected (shortest queue) among available replicas
            target_node, target_platform = min(
                available_replicas, key=lambda couple: len(couple[1].queue.items)
            )

            # Mark this platform as used
            used_platforms.add((target_node.id, target_platform.id))

            task.execution_node = target_node.node_name
            task.execution_platform = str(target_platform.id)

            node: Node = yield self.nodes.get(lambda node: node.id == target_node.id)
            task.node = node
            node.unused = False
            
            platform: Platform = yield node.platforms.get(lambda platform: platform.id == target_platform.id)
            task.platform = platform

            end = default_timer()
            elapsed_clock_time = end - start
            node.wall_clock_scheduling_time += elapsed_clock_time

            print(f"[ {self.env.now} ] BATCH_PROCESS: Scheduling task {task.id} on {target_node.node_name}:{target_platform.id} (platform initialized: {target_platform.initialized.triggered}, queue size: {len(target_platform.queue.items)})")
            yield platform.queue.put(task)
            yield task.scheduled.succeed()
            scheduled_count += 1
            
            # Strategy 1: Dynamic Load State - update queue snapshot after scheduling
            queue_key = f"{target_node.node_name}:{target_platform.id}"
            full_queue_snapshot[queue_key] = full_queue_snapshot.get(queue_key, 0) + 1
            
            print(f"[ {self.env.now} ] BATCH_PROCESS: Task {task.id} scheduled on {target_node.node_name}:{target_platform.id} (queue size after: {len(platform.queue.items)})")
            print(f"[ {self.env.now} ] BATCH_PROCESS: Task {task.id} scheduled event triggered, platform queue size: {len(platform.queue.items)}")

            yield node.platforms.put(platform)
            yield self.nodes.put(node)

            print(f"task: {task.id}")
            print(f"bounded_concurrency: ({target_node}, {target_platform})")

        print(f"[ {self.env.now} ] BATCH_PROCESS: Complete - {scheduled_count} scheduled, {postponed_count} postponed, {failed_count} failed")
        yield self.mutex.put(system_state)

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

    def _check_hardware_available(self, system_state: SystemState, task_type: dict, source_node_name: str) -> bool:
        """Check if hardware is available for creating a replica for the given task type from the source node.
        
        Returns True if hardware is available, False otherwise.
        This is a synchronous check that doesn't require autoscaling.
        """
        # Find source node to check its network_map
        source_node = None
        for node in system_state.available_resources.keys():
            if node.node_name == source_node_name:
                source_node = node
                break
        
        # Check if any compatible hardware exists on reachable nodes
        for node, platforms in system_state.available_resources.items():
            # Check network connectivity if source_node_name is provided
            if source_node_name:
                is_reachable = False
                
                # Local placement: same node as source (always valid)
                if node.node_name == source_node_name:
                    is_reachable = True
                # Remote placement: check if source can reach target node
                elif source_node is not None:
                    # Check if target node is in source's network_map
                    if node.node_name in source_node.network_map:
                        is_reachable = True
                    # For client nodes: can only use local resources or servers they can reach
                    elif source_node_name.startswith('client_node'):
                        if not node.node_name.startswith('client_node'):
                            # Server, but not in network_map - skip
                            is_reachable = False
                        else:
                            # Another client - clients can't reach each other
                            is_reachable = False
                    # For server-to-server: servers can reach each other
                    elif not source_node_name.startswith('client_node') and not node.node_name.startswith('client_node'):
                        is_reachable = True
                    else:
                        is_reachable = False
                else:
                    # Source node not found - be conservative
                    if source_node_name.startswith('client_node'):
                        is_reachable = False
                    elif not node.node_name.startswith('client_node'):
                        is_reachable = True
                    else:
                        is_reachable = False
                
                if not is_reachable:
                    continue
            
            # Check if any platform on this node is compatible with the task type
            for platform in platforms:
                if platform.type["shortName"] in task_type["platforms"]:
                    # Found compatible hardware on a reachable node
                    return True
        
        return False

    def _get_valid_replicas(self, replicas: Set[Tuple[Node, Platform]], task: Task) -> List[Tuple[Node, Platform]]:
        valid_replicas = []
        
        # Find source node to check its network_map
        source_node = None
        for n in self.nodes.items:
            if n.node_name == task.node_name:
                source_node = n
                break
        
        for node, platform in replicas:
            # Local placement: same node as source (always valid)
            if node.node_name == task.node_name:
                valid_replicas.append((node, platform))
            # Remote placement: check network connectivity
            elif not node.node_name.startswith('client_node'):
                # For a task to be placed on a server, the client must be able to reach the server
                # Check if server name is in client's network_map (client can reach server)
                if source_node is not None and hasattr(source_node, 'network_map'):
                    if node.node_name in source_node.network_map:
                        valid_replicas.append((node, platform))
                # Fallback: check if client name is in server's network_map (bidirectional check)
                elif hasattr(node, 'network_map') and task.node_name in node.network_map:
                    valid_replicas.append((node, platform))
        return valid_replicas
