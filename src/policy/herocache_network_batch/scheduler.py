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
from collections import Counter
from timeit import default_timer

from typing import Any, Dict, Generator, List, Optional, Set, Tuple, TYPE_CHECKING

from src.policy.herocache.model import HRCSystemState

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Storage, Task

from src.placement.model import SystemState
from src.placement.scheduler import Scheduler


class HRCScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = 5  # Batch size for efficient processing
        self.batch_timeout = 0.1  # Timeout for collecting batch

    def scheduler_process(self) -> Generator:
        if False:
            yield

        print(
            f"[ {self.env.now} ] HRC Network Scheduler started with policy"
            f" {self.policy} (batch_size={self.batch_size})"
        )

        while True:
            batch_tasks = yield self.env.process(self._collect_task_batch())
            
            if not batch_tasks:
                yield self.env.timeout(0.1)
                continue

            print(f"[ {self.env.now} ] HRC Network: Processing batch of {len(batch_tasks)} tasks")
            yield self.env.process(self._process_task_batch(batch_tasks))

    def _collect_task_batch(self) -> Generator[Any, Any, List[Task]]:
        """
        Collect tasks into a batch with timeout to avoid blocking.
        
        Uses polling-based approach to avoid dangling get events that can
        cause tasks to be silently consumed.
        """
        batch: List[Task] = []
        
        def task_filter(queued_task):
            return all(dependency.finished for dependency in queued_task.dependencies)
        
        # First task: wait indefinitely (blocking is expected)
        task: Task = yield self.tasks.get(task_filter)
        batch.append(task)
        
        # Set deadline for collecting rest of batch
        batch_deadline = self.env.now + self.batch_timeout
        
        # Collect remaining tasks using polling (avoids dangling get events)
        while len(batch) < self.batch_size:
            remaining_time = batch_deadline - self.env.now
            if remaining_time <= 0:
                break
            
            # Check if there are any ready tasks in the queue
            ready_tasks = [t for t in self.tasks.items if task_filter(t)]
            
            if ready_tasks:
                # Get the first ready task immediately
                task = yield self.tasks.get(task_filter)
                batch.append(task)
            else:
                # No ready tasks - wait a small step and check again
                step_time = min(0.01, remaining_time)
                yield self.env.timeout(step_time)
        
        return batch

    def _process_task_batch(self, batch_tasks: List[Task]) -> Generator:
        """Process a batch of tasks using HRC placement strategy."""
        print(f"[ {self.env.now} ] HRC Network: Processing {len(batch_tasks)} tasks in batch")
        
        # Track scheduling time for the batch
        batch_start = default_timer()
        
        system_state: Optional[SystemState] = yield self.mutex.get()
        if system_state is None:
            logging.error(f"[ {self.env.now} ] HRC Network: Failed to get system state")
            yield self.mutex.put(None)
            return
        hrc_system_state: HRCSystemState = system_state  # type: ignore
        
        for task in batch_tasks:
            task_start = default_timer()
            replicas: Set[Tuple[Node, Platform]] = hrc_system_state.replicas[task.type["name"]]

            # Filter replicas based on network connectivity
            valid_replicas = self._get_valid_replicas(replicas, task)

            # If no valid replicas (either no replicas or none are network-reachable), request autoscaling
            if not valid_replicas:
                logging.warning(
                    f"[ {self.env.now} ] HRC Network Scheduler did not find network-accessible replica for"
                    f" {task} (total replicas: {len(replicas)})"
                )

                # Put task back in queue
                task.postponed_count += 1
                yield self.tasks.put(task)

                # Request a new replica from the Autoscaler
                # Note: HRC autoscaler doesn't support source_node_name yet, but we can add it later
                stop = yield self.env.process(
                    self.autoscaler.create_first_replica(system_state, task.type)
                )

                # Continue to next task in batch
                continue

            # Use parent's placement method which will call our placement() method
            # Schedule tasks according to policy
            placement_result: Tuple[Node, Platform] = yield self.env.process(
                self.placement(system_state, task)
            )
            sched_node, sched_platform = placement_result

            # Update node
            node: Optional[Node] = yield self.nodes.get(lambda node: node.id == sched_node.id)
            if node is None:
                logging.error(f"[ {self.env.now} ] HRC Network: Failed to get node {sched_node.id}")
                continue
            task.node = node
            node.unused = False
            # Update platform
            platform: Optional[Platform] = yield node.platforms.get(
                lambda platform: platform.id == sched_platform.id
            )
            if platform is None:
                logging.error(f"[ {self.env.now} ] HRC Network: Failed to get platform {sched_platform.id}")
                yield self.nodes.put(node)
                continue
            task.platform = platform

            # End wall-clock time measurement
            task_end = default_timer()
            elapsed_clock_time = task_end - task_start
            node.wall_clock_scheduling_time += elapsed_clock_time

            # Put task in platform queue
            yield platform.queue.put(task)
            yield task.scheduled.succeed()

            yield node.platforms.put(platform)
            yield self.nodes.put(node)

        yield self.mutex.put(system_state)
        
        batch_end = default_timer()
        batch_time = (batch_end - batch_start) * 1000  # ms
        print(f"[ {self.env.now} ] HRC Network: Batch processing complete for {len(batch_tasks)} tasks "
              f"(scheduling time: {batch_time:.2f}ms)")

    def placement(self, system_state: SystemState, task: Task) -> Generator[Any, Any, Tuple[Node, Platform]]:  # type: ignore[override]
        # Scheduling functions called in a Simpy Process must be Generators
        # No-op as per https://stackoverflow.com/a/68628599/9568489
        if False:
            yield

        # Cast to HRCSystemState for type safety
        hrc_system_state: HRCSystemState = system_state  # type: ignore
        
        replicas: Set[Tuple[Node, Platform]] = hrc_system_state.replicas[task.type["name"]]

        # Filter replicas based on network connectivity
        valid_replicas = self._get_valid_replicas(replicas, task)
        
        # This should never be empty here since scheduler_process checks first
        if not valid_replicas:
            raise ValueError(f"No valid replicas with network connectivity for task {task.id}")

        logging.warning(f"replicas = {replicas}, valid_replicas = {valid_replicas}")

        scores: Dict[str, Dict[Tuple[Node, Platform], float]] = {
            "penalty": {},
            "energy_consumption": {},
            "consolidation": {},
        }

        for node, platform in valid_replicas:
            # FIXME: Get node-local storage (which one?)
            some_node_storage: Optional[Storage] = yield node.storage.get(
                lambda storage: not storage.type["remote"]
            )
            
            if some_node_storage is None:
                raise ValueError(f"No local storage available on node {node.node_name}")
            # Current task cold start
            current_task_cold_start = (
                platform.current_task.type["coldStartDuration"][
                    platform.type["shortName"]
                ]
                - (self.env.now - platform.current_task.arrived_time)
                if platform.current_task
                and platform.current_task.cold_started
                and not hasattr(platform.current_task, "started_time")
                else 0
            )
            # Current task execution time
            current_task_execution_time = (
                platform.current_task.type["executionTime"][platform.type["shortName"]]
                - (self.env.now - (getattr(platform.current_task, "started_time", None) or 0))
                if platform.current_task
                else 0
            )
            # FIXME: We would need task storage to be fixed before task execution
            # Current task communications time
            current_task_communications_time = (
                platform.current_task.type["stateSize"][
                    platform.current_task.application.type["name"]
                ]["output"]
                / (some_node_storage.type["throughput"]["write"] * 1024 * 1024)
                + some_node_storage.type["latency"]["write"]
                if platform.current_task
                else 0
            )
            # Platform task queue (QoS weighted)
            platform_task_queue = sum(
                queued_task.type["executionTime"][platform.type["shortName"]]
                for queued_task in platform.queue.items
                if queued_task.application.qos["maxDurationDeviation"]
                >= task.application.qos["maxDurationDeviation"]
            )
            # Next task cold start
            next_task_cold_start = (
                task.type["coldStartDuration"][platform.type["shortName"]]
                if not platform.current_task and not platform.previous_task
                else 0
            )
            # Next task execution time
            next_task_execution_time = task.type["executionTime"][
                platform.type["shortName"]
            ]
            # Next task communications time
            # FIXME: We would need task storage to be fixed before task scheduling
            next_task_communications_time = (
                task.type["stateSize"][task.application.type["name"]]["input"]
                / (some_node_storage.type["throughput"]["read"] * 1024 * 1024)
                + some_node_storage.type["latency"]["read"]
            ) + (
                task.type["stateSize"][task.application.type["name"]]["output"]
                / (some_node_storage.type["throughput"]["write"] * 1024 * 1024)
                + some_node_storage.type["latency"]["write"]
            )
            
            # Network latency for offloaded tasks
            next_task_network_latency = 0.0
            if node.node_name != task.node_name:
                # Task is being offloaded - add network latency
                if hasattr(node, 'network_map') and task.node_name in node.network_map:
                    next_task_network_latency = node.network_map[task.node_name]
                # Fallback: find source node and check its network_map
                else:
                    source_node = None
                    for n in self.nodes.items:
                        if n.node_name == task.node_name:
                            source_node = n
                            break
                    if source_node is not None and hasattr(source_node, 'network_map'):
                        if node.node_name in source_node.network_map:
                            next_task_network_latency = source_node.network_map[node.node_name]
            
            yield node.storage.put(some_node_storage)
            # Task deadline
            task_deadline = (
                max(task.type["executionTime"].values())
                * task.application.qos["maxDurationDeviation"]
            )

            scores["penalty"][(node, platform)] = (
                current_task_cold_start
                + current_task_execution_time
                + current_task_communications_time
                + platform_task_queue
                + next_task_cold_start
                + next_task_execution_time
                + next_task_communications_time
                + next_task_network_latency
            ) > task_deadline

            """
            # FIXME: PENALTY PER APPLICATION
            remaining_tasks = list(task.application.tasks)
            remaining_tasks.remove(task)
            """

            scores["energy_consumption"][(node, platform)] = task.type["energy"][
                platform.type["shortName"]
            ]

            # Platform-level consolidation
            concurrency_target: float = (
                hrc_system_state.scheduler_state.target_concurrencies[task.type["name"]][
                    platform.type["shortName"]
                ]
            )
            platform_concurrency = len(platform.queue.items)
            platform_usage_ratio = platform_concurrency / concurrency_target

            scores["consolidation"][(node, platform)] = (
                math.exp(platform_usage_ratio)
                if platform_usage_ratio <= self.policy.queue_length
                else math.exp(self.policy.queue_length)
            )

            """
            scores["consolidation"][(node, platform)] = (
                1
                - platform_usage_ratio
                + 2 * (math.trunc(platform_usage_ratio)) * platform_usage_ratio
            )
            """

        # logging.error(f"scores = {scores}")

        # Weights?
        weights: Dict[str, float] = {
            "penalty": 2 / 3,
            "energy_consumption": 0.5 / 6,
            "consolidation": 1.5 / 6,
        }

        # Normalize scores?
        normalized_scores: Dict[str, Dict[Tuple[Node, Platform], float]] = dict(scores)
        for metric, values in scores.items():
            t_min = 1
            t_max = 100

            v_min = min(scores[metric].values())
            v_max = max(scores[metric].values())

            # TODO if v_min == v_max...
            denominator = v_max - v_min
            denominator_safe = denominator if denominator != 0.0 else 0.01

            for couple, value in values.items():
                normalized_scores[metric][couple] = (
                    # FIXME
                    ((value - v_min) / denominator_safe) * (t_max - t_min)
                    + t_min
                )

        # logging.error(f"{self.env.now} normalized_scores = {normalized_scores}")
        # pprint.pprint(normalized_scores)

        consolidated_scores: Dict[Tuple[Node, Platform], float] = {}

        for metric, values in normalized_scores.items():
            for couple, value in values.items():
                if couple not in consolidated_scores:
                    consolidated_scores[couple] = 0

                consolidated_scores[couple] += weights[metric] * value

        # logging.error(f"consolidated_scores = {consolidated_scores}")

        selected = min(consolidated_scores, key=consolidated_scores.__getitem__)

        # logging.error(f"selected = {selected}")

        return selected

    def _get_valid_replicas(self, replicas: Set[Tuple[Node, Platform]], task: Task) -> List[Tuple[Node, Platform]]:
        """Get valid replicas: task's source node + server nodes with network connectivity"""
        valid_replicas = []
        
        # Find source node to check its network_map
        source_node = None
        for n in self.nodes.items:
            if n.node_name == task.node_name:
                source_node = n
                break
        
        for node, platform in replicas:
            # Include if it's the task's source node (local execution)
            if node.node_name == task.node_name:
                valid_replicas.append((node, platform))
            # Include if it's a server node AND has network connectivity to task source
            elif not node.node_name.startswith('client_node'):
                # Check if this node has network connectivity to the task's source node
                if source_node is not None and hasattr(source_node, 'network_map'):
                    if node.node_name in source_node.network_map:
                        valid_replicas.append((node, platform))
                # Fallback: check bidirectional connectivity
                elif hasattr(node, 'network_map') and task.node_name in node.network_map:
                    valid_replicas.append((node, platform))
        
        return valid_replicas
