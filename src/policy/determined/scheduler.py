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
from typing import Generator, Set, Tuple, TYPE_CHECKING, List, Dict, Any, Optional

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task

from src.placement.model import SystemState
from src.placement.scheduler import Scheduler


class DeterminedScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Get batch size from policy or use default
        self.batch_size = getattr(self.policy, 'batch_size', 5)  # Process 10tasks at once by default
        self.batch_timeout = getattr(self.policy, 'batch_timeout', 0.1)  # Wait up to 0.1 seconds to fill batch
        
        # Get forced placements from infrastructure config if available
        self.forced_placements = getattr(self.policy, 'forced_placements', None)
        self.forced_placements_sequence = getattr(self.policy, 'forced_placements_sequence', None)
        self._forced_index = 0
        if self.forced_placements:
            print(f"[ {self.env.now} ] DeterminedScheduler initialized with {len(self.forced_placements)} forced placements")

    def scheduler_process(self) -> Generator:
        # keep this for the simpy generator 
        if False:
            yield

        """Override to process multiple tasks simultaneously in batches"""
        print(
            f"[ {self.env.now} ] Determined Scheduler started with policy"
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
            
            print(f"[ {self.env.now} ] DEBUG: Completed task {task.id} in batch")

        # Release mutex after processing entire batch
        yield self.mutex.put(system_state)
        
        print(f"[ {self.env.now} ] DEBUG: Batch processing complete for {len(batch_tasks)} tasks")




    def placement(self, system_state: SystemState, task: Task) -> Generator[Any, Any, Optional[Tuple[Node, Platform]]]:
        # Scheduling functions called in a Simpy Process must be Generators
        # No-op as per https://stackoverflow.com/a/68628599/9568489
        if False:
            yield

        # Check for forced placements first
        if self.forced_placements and task.id in self.forced_placements:
            forced_node_id, forced_platform_id = self.forced_placements[task.id]
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
            
            if target_node is not None and target_platform is not None:
                print(f"[ {self.env.now} ] DEBUG: Found forced placement: {target_node.node_name}:{target_platform.id}")
                return (target_node, target_platform)
            else:
                print(f"[ {self.env.now} ] ERROR: Forced placement not found: node {forced_node_id}, platform {forced_platform_id}")
                # Fall back to normal placement

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
