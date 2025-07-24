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
from abc import abstractmethod
from collections import defaultdict
from typing import Dict, Generator, List, Set, Tuple, TYPE_CHECKING, Optional, Union, Any

from simpy.core import Environment, SimTime
from simpy.events import Process
from simpy.resources.store import Store

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform

from src.placement.model import (
    PlatformVector,
    ScaleEvent,
    SimulationData,
    SimulationPolicy,
    SystemState,
    TaskType, SystemEvent,
)

logger = logging.getLogger(__name__)


class Autoscaler:
    def __init__(
            self,
            env: Environment,
            mutex: Store,
            data: SimulationData,
            policy: SimulationPolicy,
    ):
        self.env = env
        self.mutex = mutex
        self.data = data
        self.policy = policy
        self.reconcile_interval = policy.reconcile_interval

        self.scale_events: List[ScaleEvent] = []
        self.system_status_events: List[SystemEvent] = []
        self.run: Process

    def autoscaler_process(self):
        logging.info(
            f"[ {self.env.now} ] Orchestrator Autoscaler started with policy"
            f" {self.policy}"
        )

        last_force_scale_up: Dict[str, SimTime] = {
            function_name: 0.0 for function_name in self.data.task_types
        }

        while True:
            # Per-function scaling decision
            system_state: SystemState = yield self.mutex.get()
            replicas: Dict[str, Set[Tuple[Node, Platform]]] = system_state.replicas

            for function_name, function_replicas in replicas.items():
                force_scale_up = True

                scaling_difference: PlatformVector[float] = yield self.env.process(
                    self.scaling_level(
                        system_state, self.data.task_types[function_name]
                    )
                )


                for hardware_target, hardware_scaling in scaling_difference.items():
                    if hardware_scaling < 0:
                        # Scale down
                        count = abs(math.floor(hardware_scaling))
                        # logging.error(f"[ {self.env.now} ] Scaling down {function_name} by {count} (currently {len(function_replicas)})")
                        stop = yield self.env.process(
                            self.scale_down(
                                count, system_state, function_name, hardware_target
                            )
                        )
                        # Do not force scale up
                        force_scale_up = False

                    elif hardware_scaling > 0:
                        # Scale up
                        count = abs(math.ceil(hardware_scaling))
                        # logging.error(f"[ {self.env.now} ] Scaling up {function_name} by {count} (currently {len(function_replicas)})")
                        stop = yield self.env.process(
                            self.scale_up(
                                count, system_state, function_name, hardware_target
                            )
                        )
                        # Successfully scaled up on hardware target
                        if not isinstance(stop, StopIteration):
                            force_scale_up = False

                    else:
                        # Correct scaling level, do nothing
                        force_scale_up = False
                        # pass

                # Force scale up on any hardware type if necessary
                if force_scale_up and (
                        (self.env.now - last_force_scale_up[function_name])
                        > self.policy.keep_alive
                ):
                    stop = yield self.env.process(
                        self.create_first_replica(
                            system_state, self.data.task_types[function_name]
                        )
                    )
                    last_force_scale_up[function_name] = self.env.now

            self.log_system_status(replicas)

            # Release mutex
            yield self.mutex.put(system_state)

            # Next event
            self.env.step()

            # Wake Autoscaler up once per second
            yield self.env.timeout(self.reconcile_interval)

    def scale_up(
            self,
            count: int,
            system_state: SystemState,
            function_name: str,
            hardware_target: str,
    ) -> Generator:
        # Get current function replicas
        function_replicas = system_state.replicas[function_name]

        # Scale up by `count` replicas
        for _ in range(count):
            replicas_count = len(function_replicas)
            # Filter out nodes by task requirements
            couples_suitable: Set[Tuple[Node, Platform]] = set()

            available_resources: Dict[Node, Set[Platform]] = (
                system_state.available_resources
            )
            for node, platforms in available_resources.items():
                for platform in platforms:
                    if (
                            hardware_target != "any"
                            and platform.type["shortName"] != hardware_target
                    ):
                        continue
                    if (
                            platform.type["shortName"]
                            not in self.data.task_types[function_name]["platforms"]
                    ):
                        continue
                    if (
                            node.memory
                            < self.data.task_types[function_name]["memoryRequirements"][
                        platform.type["shortName"]
                    ]
                    ):
                        continue
                    couples_suitable.add((node, platform))

            # No suitable resources for replica creation
            if not couples_suitable:
                # logging.error(state.average_hardware_contention[function_name])
                # Next step
                return StopIteration(
                    f"Autoscaler could not create a {hardware_target} replica for"
                    f" {function_name} (currently {replicas_count} replica)"
                )

            logging.info(
                f"[ {self.env.now} ] Autoscaler scaling up {function_name} (currently"
                f" {replicas_count})"
            )

            # Resources selection (Node, Platform)
            new_replica: Tuple[Node, Platform]
            new_replica = yield self.env.process(
                self.create_replica(
                    couples_suitable, self.data.task_types[function_name]
                )
            )

            logging.info(f"[ {self.env.now} ] {new_replica}")

            try:
                # Remove selected platform from available resources on the node
                available_resources[new_replica[0]].remove(new_replica[1])

                # Update node availability
                new_replica[0].available_platforms -= 1

                # Allocate task memory requirements from node's available memory
                new_replica[0].available_memory -= self.data.task_types[function_name][
                    "memoryRequirements"
                ][new_replica[1].type["shortName"]]

                # Add function replica to the pool, so it can be considered by the Scheduler
                function_replicas.add(new_replica)

                # Initialize replica (pull image)
                # It will be available for task execution when function image is pulled
                self.env.process(
                    self.initialize_replica(
                        new_replica,
                        function_replicas,
                        self.data.task_types[function_name],
                        system_state,
                    )
                )

                # Statistics
                new_replica[1].last_allocated = self.env.now

                event: ScaleEvent = {
                    "name": function_name,
                    "timestamp": self.env.now,
                    "action": "up",
                    "count": len(function_replicas),
                    "average_queue_length": sum(
                        [len(replica[1].queue.items) for replica in function_replicas]
                    ) / len(function_replicas),
                }
                self.scale_events.append(event)
            except KeyError:
                """
                logging.error(
                    f"[ {self.env.now} ] Autoscaler tried to scale up "
                    f"{function_name}, but {new_replica} was already allocated"
                )

                logging.error(
                    f"[ {self.env.now} ] Last allocation time: "
                    f"{new_replica[1].last_allocated} "
                    " -- Last removal time: "
                    f"{new_replica[1].last_removed}"
                )

                logging.error(
                    f"[ {self.env.now} ] {system_state.available_resources}"
                )
                logging.error(
                    f"{new_replica[1].initialized} // {new_replica[0].available_platforms}"
                )
                """
                pass

    def scale_down(
            self,
            count: int,
            system_state: SystemState,
            function_name: str,
            hardware_target: str,
    ) -> Generator[Any, Any, Optional[Union[StopIteration, Tuple[Node, Platform]]]]:
        """Scale down replicas for a given function"""
        # Get current function replicas
        function_replicas = system_state.replicas[function_name]

        # Filter replicas according to hardware target
        suitable_replicas = set(
            filter(
                lambda replica: (replica[1].type["shortName"] == hardware_target or hardware_target == 'any'),
                function_replicas,
            )
        )

        # Scale down
        for _ in range(count):
            replicas_count = len(function_replicas)
            print(f"[ {self.env.now} ] Attempting to scale down {function_name} (currently {replicas_count} replicas)")

            """
            # Check if we need to scale down based on queue length
            avg_queue_length = sum(len(replica[1].queue.items) for replica in function_replicas) / len(function_replicas)
            if avg_queue_length > self.policy.queue_length * 0.5:  # Only scale down if queue is less than 50% full
                print(f"Cannot scale down {function_name} due to high queue utilization ({avg_queue_length:.2f})")
                # return None

            # Check if any tasks are currently running on replicas
            tasks_running = any(len(replica[1].queue.items) > 0 for replica in function_replicas)
            if tasks_running:
                print(f"Cannot scale down {function_name} while tasks are running")
                # return None
            """

            removed_replica: Tuple[Node, Platform]
            removed_replica = yield self.env.process(
                self.remove_replica(
                    suitable_replicas, self.data.task_types[function_name], system_state
                )
            )

            # Could not scale down (tasks in queue on all replicas)
            # if not removed_replica:
                # print(f"Autoscaler could not scale down {function_name} (currently {replicas_count})")
                # return None

            logging.info(
                f"[ {self.env.now} ] Autoscaler scaling down {function_name} (currently"
                f" {replicas_count})"
            )

            logging.info(f"[ {self.env.now} ] {removed_replica}")
            print(f"[ {self.env.now} ] removed: {removed_replica}")

            try:
                # Remove replica from function replicas
                # FIXME: Sometimes raises KeyError ... (double remove)
                function_replicas.remove(removed_replica)

                # Reset platform to uninitialized state
                removed_replica[1].initialized = removed_replica[1].env.event()

                # Release replica into available resources
                available_resources: Dict[Node, Set[Platform]] = (
                    system_state.available_resources
                )
                available_resources[removed_replica[0]].add(removed_replica[1])

                # Update node availability
                removed_replica[0].available_platforms += 1

                # Reclaim node memory
                removed_replica[0].available_memory += self.data.task_types[
                    function_name
                ]["memoryRequirements"][removed_replica[1].type["shortName"]]

                # Statistics
                removed_replica[1].last_removed = self.env.now

                event: ScaleEvent = {
                    "name": function_name,
                    "timestamp": self.env.now,
                    "action": "down",
                    "count": len(function_replicas),
                    "average_queue_length": (
                        sum(
                            [
                                len(replica[1].queue.items)
                                for replica in function_replicas
                            ]
                        )
                        / len(function_replicas)
                        if function_replicas
                        else 0.0
                    ),
                    "platform_type": removed_replica[1].type["shortName"]
                }
                self.scale_events.append(event)
                return removed_replica
            except KeyError:
                logging.debug(
                    f"[ {self.env.now} ] Replica {removed_replica} was already removed"
                )
                return None

    @abstractmethod
    def scaling_level(
            self, system_state: SystemState, task_type: TaskType
    ) -> Generator[Any, Any, PlatformVector[float]]:
        """Determine the scaling level for a task type"""
        pass

    @abstractmethod
    def create_first_replica(
            self, system_state: SystemState, task_type: TaskType
    ) -> Generator:
        pass

    @abstractmethod
    def create_first_replica_on_node(
            self, system_state: SystemState, task_type: TaskType, node_name: str
    ) -> Generator:
        pass

    @abstractmethod
    def create_replica(
            self, couples_suitable: Set[Tuple[Node, Platform]], task_type: TaskType
    ) -> Generator:
        pass

    @abstractmethod
    def initialize_replica(
            self,
            new_replica: Tuple[Node, Platform],
            function_replicas: Set[Tuple[Node, Platform]],
            task_type: TaskType,
            state: SystemState,
    ) -> Generator:
        pass

    @abstractmethod
    def remove_replica(
            self,
            couples_suitable: Set[Tuple[Node, Platform]],
            task_type: TaskType,
            state: SystemState,
    ) -> Generator:
        pass

    def log_system_status(self, replicas: Dict[str, Set[Tuple[Node, Platform]]]):
        for function_name, function_replicas in replicas.items():
            count_by_platform_type = defaultdict(int)

            for _, platform in function_replicas:
                count_by_platform_type[platform.type['shortName']] += 1

            event: ScaleEvent = {
                "name": function_name,
                "timestamp": self.env.now,
                "count": len(function_replicas),
                "average_queue_length": (
                    sum(
                        [
                            len(replica[1].queue.items)
                            for replica in function_replicas
                        ]
                    )
                    / len(function_replicas)
                    if function_replicas
                    else 0.0
                ),
            }

            for platform_type, count in count_by_platform_type.items():
                event[platform_type] = count
            self.system_status_events.append(event)
