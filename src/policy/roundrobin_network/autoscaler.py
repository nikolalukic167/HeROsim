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
import math

from typing import Set, Tuple, TYPE_CHECKING, List

from src.placement.model import SchedulerState

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task

from src.placement.model import (
    DurationSecond,
    PlatformVector,
    SchedulerState,
    SizeGigabyte,
    SpeedMBps,
    SystemState,
    TaskType, TimeSeries,
)

from src.placement.autoscaler import Autoscaler


class RoundRobinNetworkAutoscaler(Autoscaler):

    def scaling_level(self, system_state: SystemState, task_type: TaskType):
        # Scheduling functions called in a Simpy Process must be Generators
        # No-op as per https://stackoverflow.com/a/68628599/9568489
        if False:
            yield

        # Use knative-style autoscaling based on queue lengths
        state: SchedulerState = system_state.scheduler_state
        target_concurrencies: PlatformVector = state.target_concurrencies[
            task_type["name"]
        ]
        function_replicas: Set[Tuple[Node, Platform]] = system_state.replicas[
            task_type["name"]
        ]

        # Calculate average queue length across all replicas
        queue_lengths = [
            len(platform.queue.items) for node, platform in function_replicas
        ]
        avg_queue_length = sum(queue_lengths) / len(queue_lengths) if queue_lengths else 0.0

        platform_count = len(self.data.platform_types)
        replica_count = len(function_replicas)

        # Result > 0 means scaling up
        # Result < 0 means scaling down
        # Result == 0 means current scaling level is adequate
        # Use average target concurrency across platforms
        avg_target = sum(target_concurrencies.values()) / len(target_concurrencies) if target_concurrencies else self.policy.queue_length
        
        concurrency_results: PlatformVector = {
            platform_type["shortName"]: (
                math.ceil(avg_queue_length / avg_target) - replica_count
            )
            for platform_type in self.data.platform_types.values()
        }

        return concurrency_results

    def create_first_replica(self, system_state: SystemState, task_type: TaskType):
        # RoundRobinNetwork will allocate a new replica
        available_hardware: Set[str] = set()
        for _, platforms in system_state.available_resources.items():
            for platform in platforms:
                if (
                    platform.type["shortName"]
                    in task_type["platforms"]
                ):
                    available_hardware.add(platform.type["shortName"])

        stop = None
        # FIXME: What if no available hardware?
        for platform_name in available_hardware:
            stop = yield self.env.process(
                self.scale_up(
                    1,
                    system_state,
                    task_type["name"],
                    self.data.platform_types[platform_name]["shortName"],
                )
            )

            if not isinstance(stop, StopIteration):
                # Resource found, stop iterating
                break

        return stop

    def create_replica(
        self, couples_suitable: Set[Tuple[Node, Platform]], task_type: TaskType
    ):
        # Scaling functions that do not yield values must still be Generators
        # No-op as per https://stackoverflow.com/a/68628599/9568489
        if False:
            yield

        # RoundRobinNetwork selects a replica on the most available node (cf. ENSURE)
        available_couple = max(
            couples_suitable,
            key=lambda couple: couple[0].available_platforms,
        )

        return available_couple

    def initialize_replica(
        self,
        new_replica: Tuple[Node, Platform],
        function_replicas: Set[Tuple[Node, Platform]],
        task_type: TaskType,
        system_state: SystemState,
    ):
        node: Node = new_replica[0]
        platform: Platform = new_replica[1]

        # Check node RAM cache
        warm_function: bool = (
            platform.previous_task is not None
            and platform.previous_task.type["name"] == task_type["name"]
        )

        # Initialize image retrieval duration
        retrieval_duration: DurationSecond = 0.0

        # TODO: Retrieve image if function not in RAM cache nor in disk cache
        # FIXME: Should be factored in superclass
        if not warm_function:
            logging.info(
                f"[ {self.env.now} ] 💾 {node} needs to pull image for {task_type}"
            )

            # Update image retrieval duration
            retrieval_size: SizeGigabyte = task_type["imageSize"][
                platform.type["shortName"]
            ]
            # Depends on storage performance
            # FIXME: What's the policy for storage selection?
            node_storage = yield node.storage.get(
                lambda storage: not storage.type["remote"]
            )
            # Depends on network link speed
            retrieval_speed: SpeedMBps = min(
                node_storage.type["throughput"]["write"], node.network["bandwidth"]
            )
            retrieval_duration += (
                retrieval_size / (retrieval_speed / 1024)
                + node_storage.type["latency"]["write"]
            )

            # TODO: Update disk usage
            stored = node_storage.store_function(platform.type["shortName"], task_type)

            if not stored:
                logging.error(
                    f"[ {self.env.now} ] 💾 {node_storage} has no available capacity to"
                    f" cache image for {self}"
                )

            # Release storage
            yield node.storage.put(node_storage)

        # Update state - initialize scheduled_count for round-robin
        state: SchedulerState = system_state.scheduler_state
        if not hasattr(state, 'scheduled_count'):
            state.scheduled_count = {task_type["name"]: {} for task_type in self.data.task_types}
        if task_type["name"] not in state.scheduled_count:
            state.scheduled_count[task_type["name"]] = {}
        state.scheduled_count[task_type["name"]][
            (new_replica[0].id, new_replica[1].id)
        ] = 0

        # FIXME: Retrieve function image
        yield self.env.timeout(retrieval_duration)

        # FIXME: Update platform time spent on storage
        platform.storage_time += retrieval_duration

        # FIXME: Double initialize bug...
        try:
            # Set platform to ready state
            yield platform.initialized.succeed()
        except RuntimeError:
            pass

        # Statistics (Node)
        node.cache_hits += 0

    def remove_replica(
        self,
        function_replicas: Set[Tuple[Node, Platform]],
        task_type: TaskType,
        system_state: SystemState,
    ):
        # Scaling functions that do not yield values must still be Generators
        # No-op as per https://stackoverflow.com/a/68628599/9568489
        if False:
            yield

        # Sort function replicas by in-flight requests count
        sorted_replicas = sorted(
            function_replicas, key=lambda couple: len(couple[1].queue.items)
        )

        # Mark replica for removal if its task queue is empty
        # Return None if no replica can be removed
        removed_couple = next(
            (
                replica
                for replica in sorted_replicas
                if not replica[1].queue.items
                and not replica[1].current_task
                and (self.env.now - replica[1].idle_since) > self.policy.keep_alive
            ),
            None,
        )

        if removed_couple:
            # Update state - remove scheduled_count entry
            state: SchedulerState = system_state.scheduler_state
            if hasattr(state, 'scheduled_count') and task_type["name"] in state.scheduled_count:
                try:
                    del state.scheduled_count[task_type["name"]][
                        (removed_couple[0].id, removed_couple[1].id)
                    ]
                except KeyError:
                    pass

        return removed_couple
