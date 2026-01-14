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
from typing import Generator, Set, Tuple, List, TYPE_CHECKING


if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task

from src.placement.model import SystemState

from src.placement.scheduler import Scheduler


class KnativeScheduler(Scheduler):
    def scheduler_process(self):
        """Override to check for valid network-connected replicas."""
        logging.info(
            f"[ {self.env.now} ] Orchestrator Scheduler started with policy"
            f" {self.policy}"
        )

        while True:
            # Get task with satisfied dependencies
            task: Task = yield self.tasks.get(
                lambda queued_task: all(
                    dependency.finished for dependency in queued_task.dependencies
                )
            )

            logging.info(f"[ {self.env.now} ] Scheduler woken up")

            # Get available replicas
            system_state: SystemState = yield self.mutex.get()
            replicas: Set[Tuple[Node, Platform]] = system_state.replicas[task.type["name"]]

            # Filter replicas based on network connectivity
            valid_replicas = self._get_valid_replicas(replicas, task)

            # If no valid replicas (either no replicas or none are network-reachable), request autoscaling
            if not valid_replicas:
                logging.warning(
                    f"[ {self.env.now} ] Scheduler did not find network-accessible replica for"
                    f" {task} (total replicas: {len(replicas)})"
                )

                # Put task back in queue
                task.postponed_count += 1
                yield self.tasks.put(task)

                # Request a new replica from the Autoscaler
                stop = yield self.env.process(
                    self.autoscaler.create_first_replica(system_state, task.type, source_node_name=task.node_name)
                )

                # Release mutex
                yield self.mutex.put(system_state)

                # Next step
                continue

            # Use parent's placement method which will call our placement() method
            from timeit import default_timer
            start = default_timer()

            # Schedule tasks according to policy
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
            # Update state
            yield self.mutex.put(system_state)

            # End wall-clock time measurement
            end = default_timer()
            elapsed_clock_time = end - start
            node.wall_clock_scheduling_time += elapsed_clock_time

            # Put task in platform queue
            yield platform.queue.put(task)
            yield task.scheduled.succeed()

            yield node.platforms.put(platform)
            yield self.nodes.put(node)

    def placement(self, system_state: SystemState, task: Task) -> Generator:
        # Scheduling functions called in a Simpy Process must be Generators
        # No-op as per https://stackoverflow.com/a/68628599/9568489
        if False:
            yield

        replicas: Set[Tuple[Node, Platform]] = system_state.replicas[task.type["name"]]

        # Filter replicas based on network connectivity
        valid_replicas = self._get_valid_replicas(replicas, task)

        # This should never be empty here since scheduler_process checks first
        if not valid_replicas:
            raise ValueError(f"No valid replicas for task {task.id}")

        # DEBUG: dump current per-platform queue status including internal (prewarm) tasks
        """
        try:
            for node, plat in valid_replicas:
                q_len = len(plat.queue.items)
                if q_len > 0:
                    internal = sum(1 for t in plat.queue.items if getattr(t, 'is_internal', False))
                    print(f"[ {self.env.now} ] DEBUG: Platform {plat.id}@{node.node_name} q_len={q_len} internal={internal}")
        except Exception:
            pass
        """

        # Prefer initialized replicas - they can process tasks immediately
        # Uninitialized replicas are still pulling images (30-78s)
        initialized_replicas = [r for r in valid_replicas if r[1].initialized.triggered]
        
        # Use initialized replicas if available, otherwise fall back to all valid replicas
        # (scheduler_process already limits queue depth on uninitialized replicas)
        candidates = initialized_replicas if initialized_replicas else valid_replicas
        
        # Least Connected (shortest queue) among candidates
        bounded_concurrency = min(
            candidates, key=lambda couple: len(couple[1].queue.items)
        )

        # print(f"task: {task.id}")
        # print(f"bounded_concurrency: {bounded_concurrency}")

        return bounded_concurrency

    def _get_valid_replicas(self, replicas: Set[Tuple[Node, Platform]], task: Task) -> List[Tuple[Node, Platform]]:
        """Filter replicas based on network connectivity."""
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
            else:
                # Check if target node is in source's network_map
                if source_node is not None and hasattr(source_node, 'network_map'):
                    if node.node_name in source_node.network_map:
                        valid_replicas.append((node, platform))
                # Fallback: check bidirectional connectivity
                elif hasattr(node, 'network_map') and task.node_name in node.network_map:
                    valid_replicas.append((node, platform))
        return valid_replicas
