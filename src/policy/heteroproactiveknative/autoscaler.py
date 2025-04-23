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
from collections import defaultdict
from typing import Set, Tuple, TYPE_CHECKING, Dict

import numpy as np
import xgboost
from simpy import Environment, Store

from src.motivational.constants import PREDICTION_WINDOW_SIZE
from src.motivationalhetero.encoders import get_platform_type_encoder, PLATFORM_TYPES
from src.policy.heteroproactiveknative.model import HeteroProactiveKnativeSystemState
from src.policy.knative.model import KnativeSchedulerState, KnativeSystemState
from src.policy.knative.util import count_events_in_windows_ts

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform

from src.placement.model import (
    DurationSecond,
    SchedulerState,
    SizeGigabyte,
    SpeedMBps,
    SystemState,
    TaskType, SimulationData, SimulationPolicy,
)

from src.placement.autoscaler import Autoscaler

logger = logging.getLogger(__name__)


# def manage_platforms(platforms, available_platform_types):
#     # Sort the list of tuples based on modify_replicas (ascending)
#     sorted_platforms = sorted(platforms, key=lambda x: x[1])
#
#     # Start as many platforms as possible based on availability
#     started_platforms = {}
#     remaining_workload = {}
#
#     for platform_type, needed_replicas in sorted_platforms:
#         available_replicas = available_platform_types.get(platform_type, 0)
#         if available_replicas >= needed_replicas:
#             started_platforms[platform_type] = needed_replicas
#             available_platform_types[platform_type] -= needed_replicas
#         else:
#             started_platforms[platform_type] = available_replicas
#             remaining_workload[platform_type] = needed_replicas - available_replicas
#             available_platform_types[platform_type] = 0
#
#     # Calculate total platforms needed for 100% workload
#     total_platforms_needed = {}
#     for platform_type, replicas in platforms:
#         total_platforms_needed[platform_type] = replicas
#
#     # Calculate percentage of workload satisfied
#     workload_satisfied = {}
#     for platform_type in total_platforms_needed:
#         if platform_type in started_platforms:
#             workload_satisfied[platform_type] = (started_platforms[platform_type] /
#                                                  total_platforms_needed[platform_type]) * 100
#         else:
#             workload_satisfied[platform_type] = 0
#
#     return sorted_platforms, started_platforms, remaining_workload, workload_satisfied


def select_platforms_for_100_percent(workload_satisfied, platforms):
    # Convert workload_satisfied dictionary to a list of tuples
    workload_list = [(platform_type, percentage) for platform_type, percentage in workload_satisfied.items()]

    # Sort the workload list by percentages in descending order
    sorted_workload = sorted(workload_list, key=lambda x: x[1], reverse=True)

    # Select platform types until the sum of percentages reaches or exceeds 100
    selected_platforms = []
    total_percentage = 0

    for platform_type, percentage in sorted_workload:
        if total_percentage < 100:
            selected_platforms.append((platform_type, percentage))
            total_percentage += percentage
        else:
            break

    # Create a dictionary with platform_type as key and the number of replicas as value
    replicas_needed = {}

    # Find the original number of replicas for each platform type
    platform_replicas = {platform_type: replicas for platform_type, replicas in platforms.items()}

    # Calculate the number of replicas needed for each selected platform
    for platform_type, percentage in selected_platforms:
        # Get the original number of replicas needed for this platform type
        original_replicas = platform_replicas.get(platform_type, 0)

        # Calculate the number of replicas based on the percentage contribution
        replicas_needed[platform_type] = original_replicas

    return selected_platforms, total_percentage, replicas_needed


def manage_platforms(platforms, available_platform_types, running_platforms=None):
    """
    Manages platform allocation considering available platforms, running platforms,
    and minimizing changes while ensuring 100% workload coverage.

    Args:
        platforms: List of tuples (platform_type, required_replicas) for 100% workload
        available_platform_types: Dict of {platform_type: available_count}
        running_platforms: Dict of {platform_type: currently_running_count}, defaults to empty dict

    Returns:
        Dict with comprehensive platform allocation information
    """
    # Initialize running_platforms if not provided
    if running_platforms is None:
        running_platforms = {}

    # Sort the list of tuples based on required_replicas (ascending)
    sorted_platforms = sorted(platforms, key=lambda x: x[1])

    # Create a dictionary of required replicas for each platform type
    required_replicas = {platform_type: replicas for platform_type, replicas in platforms}

    # Start as many platforms as possible based on availability
    started_platforms = {}
    remaining_workload = {}

    # First consider platforms that are already running
    for platform_type, needed_replicas in sorted_platforms:
        running_replicas = running_platforms.get(platform_type, 0)
        available_replicas = available_platform_types.get(platform_type, 0)

        # Calculate how many more we need to start
        additional_needed = max(0, needed_replicas - running_replicas)

        if available_replicas >= additional_needed:
            # We can start all the additional replicas we need
            started_platforms[platform_type] = running_replicas + additional_needed
            available_platform_types[platform_type] -= additional_needed
        else:
            # Start as many as we can
            started_platforms[platform_type] = running_replicas + available_replicas
            remaining_workload[platform_type] = needed_replicas - (running_replicas + available_replicas)
            available_platform_types[platform_type] = 0

    # Calculate percentage of workload satisfied for each platform type
    workload_satisfied = {}
    for platform_type, needed_replicas in platforms:
        started = started_platforms.get(platform_type, 0)
        workload_satisfied[platform_type] = (started / needed_replicas) * 100 if needed_replicas > 0 else 0

    # Sort platforms by workload percentage (descending) for 100% selection
    sorted_workload = sorted(workload_satisfied.items(), key=lambda x: x[1], reverse=True)

    # Select platform types until the sum of percentages reaches or exceeds 100
    selected_platforms = []
    total_percentage = 0

    for platform_type, percentage in sorted_workload:
        if total_percentage < 100:
            selected_platforms.append((platform_type, percentage))
            total_percentage += percentage
        else:
            break

    # Create a dictionary with platform_type as key and the number of replicas as value
    replicas_needed = {}

    # Calculate the number of replicas needed for each selected platform
    for platform_type, _ in selected_platforms:
        # Get the original number of replicas needed for this platform type
        replicas_needed[platform_type] = required_replicas.get(platform_type, 0)

    # Get the list of selected platform types
    selected_platform_types = [platform_type for platform_type, _ in selected_platforms]

    # Calculate changes needed (positive = add, negative = remove) ONLY for selected platforms
    changes_needed = {}

    # For platforms in selected_platforms, calculate changes needed
    for platform in selected_platform_types:
        required = replicas_needed.get(platform, 0)
        running = running_platforms.get(platform, 0)
        changes_needed[platform] = required - running

    # For platforms that are running but not selected, they should be removed
    for platform in running_platforms:
        if platform not in selected_platform_types:
            changes_needed[platform] = -running_platforms[platform]

    # Calculate platforms to add and remove
    platforms_to_add = {p: c for p, c in changes_needed.items() if c > 0}
    platforms_to_remove = {p: -c for p, c in changes_needed.items() if c < 0}

    # Calculate total changes
    total_changes = sum(abs(change) for change in changes_needed.values())

    return {
        "sorted_platforms": sorted_platforms,
        "started_platforms": started_platforms,
        "remaining_workload": remaining_workload,
        "workload_satisfied": workload_satisfied,
        "selected_platforms": selected_platforms,
        "total_percentage": total_percentage,
        "replicas_needed": replicas_needed,
        "changes_needed": changes_needed,
        "platforms_to_add": platforms_to_add,
        "platforms_to_remove": platforms_to_remove,
        "total_changes": total_changes
    }


class HeteroProactiveKnativeAutoscaler(Autoscaler):

    def __init__(
            self,
            env: Environment,
            mutex: Store,
            data: SimulationData,
            policy: SimulationPolicy,
            models: Dict[str, xgboost.XGBRegressor],
            encoder=None
    ):
        super().__init__(env, mutex, data, policy)
        if encoder is None:
            encoder = get_platform_type_encoder()
        self.encoder = encoder
        self.models = models

    def scaling_level(self, system_state: HeteroProactiveKnativeSystemState, task_type: TaskType):
        # Scheduling functions called in a Simpy Process must be Generators
        # No-op as per https://stackoverflow.com/a/68628599/9568489
        if False:
            yield
        if self.models.get(task_type['name']) is None:
            return {"any": 0}
        look_forward_size = PREDICTION_WINDOW_SIZE
        events = \
            count_events_in_windows_ts(self.env.now, system_state.time_series, task_type['name'], look_forward_size,
                                       look_forward_size)
        if events is None:
            print(f'FN {task_type["name"]} has no events')
            return {"any": 0}

        events = events[0] / look_forward_size
        no_replicas_by_platform_type = {}
        for platform_type in PLATFORM_TYPES:
            platform_encoded = self.encoder.transform([[platform_type]])
            X_events = np.array([events])
            X = np.column_stack((X_events, np.tile(platform_encoded, (len(X_events), 1))))
            no_replicas = self.models[task_type['name']].predict(X)[0]
            no_replicas_by_platform_type[platform_type] = math.ceil(no_replicas)
            print(f"{platform_type} - {X} - {no_replicas_by_platform_type}")

        function_replicas: Set[Tuple[Node, Platform]] = system_state.replicas[
            task_type["name"]
        ]

        function_replica_count_by_platform = defaultdict(int)
        for _, platform in function_replicas:
            function_replica_count_by_platform[platform.type['shortName']] += 1

        modify_replicas_by_platform_type = defaultdict(int)
        modify_replicas_by_platform_type_sorted = []
        for platform_type in PLATFORM_TYPES:
            current_no_replicas = function_replica_count_by_platform[platform_type]
            predicted_no_replicas = no_replicas_by_platform_type[platform_type]
            modify_replicas = predicted_no_replicas - current_no_replicas
            modify_replicas_by_platform_type[platform_type] = modify_replicas
            modify_replicas_by_platform_type_sorted.append((platform_type, no_replicas_by_platform_type[platform_type]))

        running_platforms = [x[1] for x in system_state.replicas[task_type["name"]]]
        running_platform_count = defaultdict(int)
        for platform in running_platforms:
            running_platform_count[platform.type['shortName']] += 1
        print(running_platform_count)
        a = manage_platforms(modify_replicas_by_platform_type_sorted, system_state.available_platform_types,
                             running_platform_count)
        # selected_platforms, total_percentage, replicas_needed = select_platforms_for_100_percent(workload_satisfied, started_platforms)
        print(a['changes_needed'])
        return a['changes_needed']

    def create_first_replica(self, system_state: SystemState, task_type: TaskType):
        # Knative will allocate a new CPU replica
        available_hardware: Set[str] = set()
        for _, platforms in system_state.available_resources.items():
            for platform in platforms:
                if (
                        # platform.type["hardware"] == "cpu"
                        # and platform.type["shortName"] in task_type["platforms"]
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

        """
        # Knative only allocates CPUs
        filtered_couples = set(filter(
            lambda couple: couple[1].type["hardware"] == "cpu",
            couples_suitable
        ))
        """

        # Knative selects a replica on the most available node (cf. ENSURE)
        available_couple = max(
            # filtered_couples, key=lambda couple: couple[0].available_platforms
            couples_suitable,
            key=lambda couple: couple[0].available_platforms,
        )

        return available_couple

    def initialize_replica(
            self,
            new_replica: Tuple[Node, Platform],
            function_replicas: Set[Tuple[Node, Platform]],
            task_type: TaskType,
            system_state: KnativeSystemState,
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

            # print(f"retrieval size = {retrieval_size}")
            # print(f"retrieval speed = {retrieval_speed}")

            # TODO: Update disk usage
            stored = node_storage.store_function(platform.type["shortName"], task_type)

            if not stored:
                logging.error(
                    f"[ {self.env.now} ] 💾 {node_storage} has no available capacity to"
                    f" cache image for {self}"
                )

            # Release storage
            yield node.storage.put(node_storage)

        # print(f"retrieval duration = {retrieval_duration}")

        # Update state
        # FIXME: Move to state update methods
        state: KnativeSchedulerState = system_state.scheduler_state
        # Knative policy
        state.average_contention[task_type["name"]][
            (new_replica[0].id, new_replica[1].id)
        ] = 1.0

        # FIXME: Retrieve function image
        yield self.env.timeout(retrieval_duration)

        # FIXME: Update platform time spent on storage
        platform.storage_time += retrieval_duration

        # FIXME: Double initialize bug...
        try:
            # Set platform to ready state
            yield platform.initialized.succeed()
        except RuntimeError:
            """
            logging.error(
                f"[ {self.env.now} ] Autoscaler tried to initialize "
                f"{new_replica[1]} ({new_replica[0]}) but it was already initialized."
            )

            logging.error(
                f"[ {self.env.now} ] Last allocation time: "
                f"{new_replica[1].last_allocated} "
                " -- Last removal time: "
                f"{new_replica[1].last_removed}"
            )
            """
            pass

        # Statistics (Node)
        node.cache_hits += 0

    def remove_replica(
            self,
            function_replicas: Set[Tuple[Node, Platform]],
            task_type: TaskType,
            system_state: KnativeSystemState,
    ):
        # Scaling functions that do not yield values must still be Generators
        # No-op as per https://stackoverflow.com/a/68628599/9568489&
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
            # Update state
            # FIXME: Move to state update methods
            state: SchedulerState = system_state.scheduler_state
            try:
                # Knative policy
                del state.average_contention[task_type["name"]][
                    (removed_couple[0].id, removed_couple[1].id)
                ]
            except KeyError:
                """
                logging.error(
                    f"[ {self.env.now} ] Autoscaler tried to scale down "
                    f"{task_type['name']}, but {removed_couple[1]} was already removed"
                )
                """
                pass

        return removed_couple
