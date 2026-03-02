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
import random

from typing import List, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task

from src.placement.model import SystemState

from src.placement.scheduler import Scheduler


class RandomScheduler(Scheduler):
    def placement(self, system_state: SystemState, task: Task):  # type: ignore[override]
        # Scheduling functions called in a Simpy Process must be Generators
        # No-op as per https://stackoverflow.com/a/68628599/9568489
        if False:
            yield

        replicas: Set[Tuple[Node, Platform]] = system_state.replicas[task.type["name"]]

        # Select random platform
        found = random.randint(0, len(replicas) - 1)
        random_couple = list(replicas)[found]

        return random_couple


class RandomNetworkScheduler(Scheduler):
    def scheduler_process(self):
        """Override to filter by network connectivity and request autoscaling when no reachable replicas."""
        logging.info(
            f"[ {self.env.now} ] Orchestrator Scheduler started with policy"
            f" {self.policy}"
        )

        while True:
            task = yield self.tasks.get(
                lambda queued_task: all(
                    dependency.finished for dependency in queued_task.dependencies
                )
            )
            assert task is not None

            logging.info(f"[ {self.env.now} ] Scheduler woken up")

            system_state = yield self.mutex.get()
            assert system_state is not None
            replicas: Set[Tuple[Node, Platform]] = system_state.replicas[task.type["name"]]

            valid_replicas = self._get_valid_replicas(replicas, task)

            if not valid_replicas:
                logging.warning(
                    f"[ {self.env.now} ] Scheduler did not find network-accessible replica for"
                    f" task {task.id} (total replicas: {len(replicas)})"
                )
                task.postponed_count += 1
                yield self.tasks.put(task)
                # KnativeNetworkAutoscaler (used with rp_network) supports source_node_name
                yield self.env.process(
                    self.autoscaler.create_first_replica(
                        system_state, task.type, source_node_name=task.node_name  # type: ignore[call-arg]
                    )
                )
                yield self.mutex.put(system_state)
                continue

            (sched_node, sched_platform) = yield self.env.process(
                self.placement(system_state, task)
            )
            node = yield self.nodes.get(lambda n: n.id == sched_node.id)
            assert node is not None
            task.node = node
            node.unused = False
            platform = yield node.platforms.get(
                lambda p: p.id == sched_platform.id
            )
            assert platform is not None
            task.platform = platform
            yield self.mutex.put(system_state)
            yield platform.queue.put(task)
            yield task.scheduled.succeed()
            yield node.platforms.put(platform)
            yield self.nodes.put(node)

    def _get_valid_replicas(
        self, replicas: Set[Tuple[Node, Platform]], task: Task
    ) -> List[Tuple[Node, Platform]]:
        """Filter replicas by network connectivity from task's source node."""
        valid = []
        source_node = None
        for n in (self.nodes.items or []):
            if n.node_name == task.node_name:
                source_node = n
                break
        for node, platform in replicas:
            if node.node_name == task.node_name:
                valid.append((node, platform))
            else:
                if source_node is not None and getattr(source_node, "network_map", None):
                    if node.node_name in source_node.network_map:
                        valid.append((node, platform))
                elif getattr(node, "network_map", None) and task.node_name in node.network_map:
                    valid.append((node, platform))
        return valid

    def placement(self, system_state: SystemState, task: Task):  # type: ignore[override]
        # Scheduling functions called in a Simpy Process must be Generators
        if False:
            yield

        replicas: Set[Tuple[Node, Platform]] = system_state.replicas[task.type["name"]]
        connected_replicas = self._get_valid_replicas(replicas, task)

        if not connected_replicas:
            raise ValueError(f"No network-connected replicas for task {task.id}")

        return random.choice(connected_replicas)
