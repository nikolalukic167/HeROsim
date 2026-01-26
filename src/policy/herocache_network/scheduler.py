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

from typing import Any, Dict, Generator, List, Optional, Set, Tuple, TYPE_CHECKING

from src.policy.herocache.model import HRCSystemState

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Storage, Task

from src.placement.model import SystemState
from src.placement.scheduler import Scheduler
from src.policy.state_capture import StateCaptureHelper


class HRCScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # State capture helper (initialized lazily when env/nodes are available)
        self._state_capture: Optional[StateCaptureHelper] = None
    
    def scheduler_process(self):
        """
        Process tasks ONE BY ONE (like original herocache).
        Removed batching to avoid mutex bottleneck.
        """
        logging.info(
            f"[ {self.env.now} ] HRC Network Scheduler started with policy"
            f" {self.policy}"
        )

        while True:
            # Get task with satisfied dependencies (same as base scheduler)
            task: Task = yield self.tasks.get(
                lambda queued_task: all(
                    dependency.finished for dependency in queued_task.dependencies
                )
            )

            logging.info(f"[ {self.env.now} ] Scheduler woken up")

            # Get available replicas
            system_state: SystemState = yield self.mutex.get()
            hrc_system_state: HRCSystemState = system_state  # type: ignore
            replicas: Set[Tuple[Node, Platform]] = hrc_system_state.replicas[task.type["name"]]

            # Filter replicas based on network connectivity
            valid_replicas = self._get_valid_replicas(replicas, task)

            # If no valid replicas, request autoscaling
            if not valid_replicas:
                logging.warning(
                    f"[ {self.env.now} ] HRC Network Scheduler did not find network-accessible replica for"
                    f" {task} (total replicas: {len(replicas)})"
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

                # Release mutex
                yield self.mutex.put(system_state)

                # Next step
                continue

            # Capture state BEFORE placement decision (for analysis)
            task.queue_snapshot_at_scheduling = self._capture_queue_snapshot_for_replicas(valid_replicas)
            task.full_queue_snapshot = self._capture_full_queue_snapshot()
            task.temporal_state_at_scheduling = self._capture_temporal_state_for_replicas(valid_replicas)

            # Measure wall-clock time for the scheduling decision
            from timeit import default_timer
            start = default_timer()

            # Schedule tasks according to policy
            (sched_node, sched_platform) = yield self.env.process(
                self.placement(system_state, task)
            )

            # Store execution node/platform on task
            task.execution_node = sched_node.node_name
            task.execution_platform = str(sched_platform.id)

            # Update node
            node: Node = yield self.nodes.get(lambda node: node.id == sched_node.id)
            task.node = node
            node.unused = False
            # Update platform
            platform: Platform = yield node.platforms.get(
                lambda platform: platform.id == sched_platform.id
            )
            task.platform = platform
            # Update state - release mutex IMMEDIATELY after placement
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

    # ==================== State Capture Methods ====================
    
    @property
    def state_capture(self) -> StateCaptureHelper:
        """Lazy initialization of state capture helper."""
        if self._state_capture is None:
            self._state_capture = StateCaptureHelper(self.env, self.nodes)
        return self._state_capture
    
    def enable_state_capture(self, output_path: str):
        """Enable state capture and set output path."""
        self.state_capture.enable_capture(output_path)
    
    def disable_state_capture(self):
        """Disable state capture."""
        self.state_capture.disable_capture()
    
    def capture_task_placement(
        self,
        task: 'Task',
        execution_node: str,
        execution_platform: str,
        elapsed_time: float,
        valid_replicas: List[Tuple['Node', 'Platform']]
    ) -> Dict[str, Any]:
        """
        Capture a task placement decision with full state information.
        
        Args:
            task: The task being placed
            execution_node: Node where task will execute
            execution_platform: Platform ID where task will execute
            elapsed_time: Wall-clock time for scheduling decision
            valid_replicas: Set of valid replicas for this task
            
        Returns:
            Dict with placement information
        """
        # Calculate queue time
        queue_time = self.env.now - task.arrived_time if hasattr(task, 'arrived_time') else 0.0
        
        # Capture queue snapshots
        valid_replicas_set = set(valid_replicas)
        queue_snapshot_at_scheduling = self.state_capture.capture_queue_snapshot_for_replicas(valid_replicas_set)
        full_queue_snapshot = self.state_capture.capture_full_queue_snapshot()
        
        # Capture temporal state
        temporal_state_at_scheduling = self.state_capture.capture_temporal_state_for_replicas(valid_replicas_set)
        
        return self.state_capture.capture_task_placement(
            task=task,
            execution_node=execution_node,
            execution_platform=execution_platform,
            elapsed_time=elapsed_time,
            queue_time=queue_time,
            queue_snapshot_at_scheduling=queue_snapshot_at_scheduling,
            full_queue_snapshot=full_queue_snapshot,
            temporal_state_at_scheduling=temporal_state_at_scheduling,
        )
    
    def save_captured_state(self, system_state: 'SystemState', total_rtt: float = 0.0, output_path: Optional[str] = None):
        """Save captured state to JSON file."""
        self.state_capture.save_captured_state(system_state, total_rtt, output_path)
    
    def get_captured_state(self, system_state: 'SystemState', total_rtt: float = 0.0) -> Dict[str, Any]:
        """Get captured state as dictionary."""
        return self.state_capture.get_captured_state(system_state, total_rtt)
    
    def reset_state_capture(self):
        """Reset captured placements for a new simulation run."""
        self.state_capture.reset()

    # ==================== Direct State Capture (for task results) ====================
    
    def _capture_queue_snapshot_for_replicas(self, replicas: List[Tuple['Node', 'Platform']]) -> Dict[str, int]:
        """Capture queue lengths for a specific set of replicas."""
        queue_snapshot = {}
        for node, platform in replicas:
            key = f"{node.node_name}:{platform.id}"
            queue_snapshot[key] = len(platform.queue.items)
        return queue_snapshot
    
    def _capture_full_queue_snapshot(self) -> Dict[str, int]:
        """Capture queue lengths for ALL platforms in the system."""
        queue_snapshot = {}
        for node in self.nodes.items:
            for platform in node.platforms.items:
                key = f"{node.node_name}:{platform.id}"
                queue_snapshot[key] = len(platform.queue.items)
        return queue_snapshot
    
    def _capture_temporal_state_for_replicas(self, replicas: List[Tuple['Node', 'Platform']]) -> Dict[str, Dict[str, float]]:
        """Capture temporal state (remaining times) for a set of replicas."""
        temporal_state = {}
        now = self.env.now
        
        for node, platform in replicas:
            key = f"{node.node_name}:{platform.id}"
            
            current_task_remaining = 0.0
            cold_start_remaining = 0.0
            comm_remaining = 0.0
            
            if platform.current_task is not None:
                current_task = platform.current_task
                
                # Check if task is in cold start phase
                if current_task.cold_started and not hasattr(current_task, "started_time"):
                    cold_start_duration = current_task.type["coldStartDuration"].get(
                        platform.type["shortName"], 0.0
                    )
                    elapsed_cold_start = now - current_task.arrived_time
                    cold_start_remaining = max(0.0, cold_start_duration - elapsed_cold_start)
                
                # Check if task is executing
                if hasattr(current_task, "started_time") and current_task.started_time is not None:
                    exec_duration = current_task.type["executionTime"].get(
                        platform.type["shortName"], 0.0
                    )
                    elapsed_exec = now - current_task.started_time
                    current_task_remaining = max(0.0, exec_duration - elapsed_exec)
                    
                    # Estimate communication remaining
                    if current_task.application:
                        state_size_map = current_task.type.get("stateSize", {})
                        app_name = current_task.application.type.get("name", "")
                        if isinstance(state_size_map, dict) and app_name in state_size_map:
                            output_size = state_size_map[app_name].get("output", 0)
                            if isinstance(output_size, (int, float)) and output_size > 0:
                                throughput = 100.0 * 1024 * 1024  # 100 MB/s
                                latency = 0.001  # 1ms
                                comm_remaining = (output_size / throughput) + latency
            
            temporal_state[key] = {
                "current_task_remaining": current_task_remaining,
                "cold_start_remaining": cold_start_remaining,
                "comm_remaining": comm_remaining,
            }
        
        return temporal_state
