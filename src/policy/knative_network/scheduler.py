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
from typing import Generator, List, Optional, Set, Tuple, TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task

from src.placement.model import SystemState
from src.placement.scheduler import Scheduler


class KnativeNetworkScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = 5
        self.batch_timeout = 0.1

    def scheduler_process(self) -> Generator:
        if False:
            yield

        print(
            f"[ {self.env.now} ] KnativeNetwork Scheduler started with policy"
            f" {self.policy} (batch_size={self.batch_size})"
        )

        while True:
            batch_tasks = yield self.env.process(self._collect_task_batch())
            
            if not batch_tasks:
                yield self.env.timeout(0.1)
                continue

            print(f"[ {self.env.now} ] DEBUG: Processing batch of {len(batch_tasks)} tasks simultaneously")
            yield self.env.process(self._process_task_batch(batch_tasks))

    def _collect_task_batch(self) -> Generator[Any, Any, List[Task]]:
        batch = []
        for i in range(self.batch_size):
            try:
                task: Task = yield self.tasks.get(
                    lambda queued_task: all(
                        dependency.finished for dependency in queued_task.dependencies
                    )
                )
                batch.append(task)
            except:
                break
        return batch

    def _process_task_batch(self, batch_tasks: List[Task]) -> Generator:
        print(f"[ {self.env.now} ] DEBUG: Processing {len(batch_tasks)} tasks in batch")
        
        system_state: SystemState = yield self.mutex.get()
        
        full_queue_snapshot = self._capture_full_queue_snapshot()
        batch_queue_snapshot = self._capture_batch_queue_snapshot(system_state, batch_tasks)
        
        # Track used platforms within this batch to ensure unique placements
        # Key: (node_id, platform_id) tuple - same as BF search space
        used_platforms: Set[Tuple[int, int]] = set()
        
        for task in batch_tasks:
            task_replicas = system_state.replicas.get(task.type["name"], set())

            if not task_replicas:
                logging.warning(f"[ {self.env.now} ] Scheduler did not find available replica for {task}")
                task.postponed_count += 1
                yield self.tasks.put(task)
                stop = yield self.env.process(
                    self.autoscaler.create_first_replica(system_state, task.type)
                )
                self.env.step()
                continue

            start = default_timer()

            valid_replicas = self._get_valid_replicas(task_replicas, task)
            
            if not valid_replicas:
                print(f"[ {self.env.now} ] ERROR: No valid replicas for task {task.id} ({task.type['name']})")
                task.postponed_count += 1
                yield self.tasks.put(task)
                stop = yield self.env.process(
                    self.autoscaler.create_first_replica(system_state, task.type)
                )
                self.env.step()
                continue

            # Filter out already-used platforms to ensure unique placements
            # Use (node.id, platform.id) tuple to match BF search space exactly
            available_replicas = [
                (node, plat) for node, plat in valid_replicas 
                if (node.id, plat.id) not in used_platforms
            ]
            
            # If all valid replicas are used, this is an error - BF would not generate this case
            # Do NOT fall back to reuse - that creates placements outside BF search space
            if not available_replicas:
                print(f"[ {self.env.now} ] ERROR: No unique replicas left for task {task.id} ({task.type['name']}) - all {len(valid_replicas)} valid replicas already used in batch")
                task.postponed_count += 1
                yield self.tasks.put(task)
                stop = yield self.env.process(
                    self.autoscaler.create_first_replica(system_state, task.type)
                )
                self.env.step()
                continue

            task.queue_snapshot_at_scheduling = {
                f"{node.node_name}:{plat.id}": batch_queue_snapshot.get(f"{node.node_name}:{plat.id}", 0)
                for node, plat in available_replicas
            }
            task.full_queue_snapshot = full_queue_snapshot

            # Least Connected (shortest queue) among available replicas
            target_node, target_platform = min(
                available_replicas, key=lambda couple: len(couple[1].queue.items)
            )

            # Mark this platform as used (node_id, platform_id) - matches BF search space
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

            yield platform.queue.put(task)
            yield task.scheduled.succeed()

            yield node.platforms.put(platform)
            yield self.nodes.put(node)

            print(f"task: {task.id}")
            print(f"bounded_concurrency: ({target_node}, {target_platform})")

        yield self.mutex.put(system_state)
        print(f"[ {self.env.now} ] DEBUG: Batch processing complete for {len(batch_tasks)} tasks")

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
        valid_replicas = []
        for node, platform in replicas:
            if node.node_name == task.node_name:
                valid_replicas.append((node, platform))
            elif not node.node_name.startswith('client_node'):
                if hasattr(node, 'network_map') and task.node_name in node.network_map:
                    valid_replicas.append((node, platform))
        return valid_replicas
