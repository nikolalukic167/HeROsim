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


class MultiLoopScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Get batch size from policy or use default
        self.batch_size = getattr(self.policy, 'batch_size', 5)  # Process 5 tasks at once by default
        self.batch_timeout = getattr(self.policy, 'batch_timeout', 0.1)  # Wait up to 0.1 seconds to fill batch

    def scheduler_process(self) -> Generator:
        # keep this for the simpy generator 
        if False:
            yield

        """Override to process multiple tasks simultaneously in batches"""
        print(
            f"[ {self.env.now} ] MultiLoop Scheduler started with policy"
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
            (sched_node, sched_platform) = yield self.env.process(
                self.placement(system_state, task)
            )

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





    def placement(self, system_state: SystemState, task: Task) -> Generator[Any, Any, Tuple[Node, Platform]]:
        # Scheduling functions called in a Simpy Process must be Generators
        # No-op as per https://stackoverflow.com/a/68628599/9568489
        if False:
            yield

        replicas: Set[Tuple[Node, Platform]] = system_state.replicas[task.type["name"]]

        # Least Connected
        bounded_concurrency = min(
            replicas, key=lambda couple: len(couple[1].queue.items)
        )

        print(f"task: {task.id}")
        print(f"bounded_concurrency: {bounded_concurrency}")

        return bounded_concurrency
