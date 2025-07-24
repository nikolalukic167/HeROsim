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

from typing import Set, Tuple, TYPE_CHECKING, List, Dict, Generator

from src.policy.gnn.model import GNNSystemState, GNNSchedulerState

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


class GNNAutoscaler(Autoscaler):

    def scaling_level(self, system_state: GNNSystemState, task_type: TaskType):
        # Scheduling functions called in a Simpy Process must be Generators
        # No-op as per https://stackoverflow.com/a/68628599/9568489
        if False:
            yield

        # Knative default values (cf. https://knative.dev/docs/serving/autoscaling/concurrency/)
        # Lambda is 1 (cf. https://notes.crmarsh.com/isolates-microvms-and-webassembly)
        state: GNNSchedulerState = system_state.scheduler_state
        target_concurrencies: PlatformVector = state.target_concurrencies[
            task_type["name"]
        ]
        function_concurrencies = state.average_contention[task_type["name"]].values()
        function_replicas: Set[Tuple[Node, Platform]] = system_state.replicas[
            task_type["name"]
        ]

        """
        target_concurrencies: PlatformVector = {
            platform: self.policy.queue_length if platform == baseline_platform else 0
            for platform in self.data.platform_types
        }
        """

        platform_count = len(
            [
                platform
                for platform in self.data.platform_types.values()
                # if platform["hardware"] == "cpu"
            ]
        )

        # Per-function concurrency level
        # Knative only allocates CPUs (baseline platform)
        in_system_concurrencies: PlatformVector = {
            platform_type["shortName"]: (
                0.0
                if not function_concurrencies
                else sum(function_concurrencies) / platform_count
                # if platform_type["hardware"] == "cpu"
                # else 0.0
            )
            for platform_type in self.data.platform_types.values()
        }
        #
        replica_count = len(function_replicas)
        # if task_type['name'] == 'rf' and replica_count < 1:
        #     return {"any": 50}
        # elif task_type['name'] == 'rf' and replica_count > 1:
        #     return {"any": 0}

        # Result > 0 means scaling up
        # Result < 0 means scaling down
        # Result == 0 means current scaling level is adequate
        concurrency_results: PlatformVector = {
            platform_type["shortName"]: (
                math.ceil(
                    in_system_concurrencies[platform_type["shortName"]]
                    / target_concurrencies[platform_type["shortName"]]
                )
                - replica_count
                # if platform_type["hardware"] == "cpu"
                # else 0
            )
            for platform_type in self.data.platform_types.values()
        }
        """
        logging.error(f"[ {self.env.now} ] ===")
        logging.error(f"[ {self.env.now} ] {task_type['name']} {in_system_concurrencies}")
        logging.error(f"[ {self.env.now} ] {task_type['name']} {function_replicas}")
        logging.error(f"[ {self.env.now} ] {task_type['name']} {target_concurrencies}")
        logging.error(f"[ {self.env.now} ] {task_type['name']} {concurrency_results}")
        logging.error(f"[ {self.env.now} ] ===")
        """
        return concurrency_results

    def create_first_replica_og2(self, system_state: GNNSystemState, task_type: TaskType):
        if False:
            yield
        
        # Get available nodes and platforms
        available_nodes = list(system_state.available_resources.keys())
        
        # Find suitable node-platform couples
        suitable_couples = []
        for node in available_nodes:
            # Get platforms directly from the set
            platforms = system_state.available_resources[node]
            for platform in platforms:
                # Check if platform type is supported by the task
                if platform.type["shortName"] in task_type["platforms"]:
                    # Check if node has enough memory
                    if node.memory >= task_type["memoryRequirements"][platform.type["shortName"]]:
                        suitable_couples.append((node, platform))

        if not suitable_couples:
            print(f"[GNN-AS] No suitable couples available for creating replica for {task_type['name']}")
            # Instead of returning None, return a StopIteration with a descriptive message
            yield StopIteration(f"Could not create replica for {task_type['name']}: No suitable node-platform couples found")
            return

        # Create replica on the first suitable couple
        new_replica = suitable_couples[0]

        try:
            # Remove platform from available resources
            system_state.available_resources[new_replica[0]].remove(new_replica[1])

            # Update node availability
            new_replica[0].available_platforms -= 1

            # Allocate memory
            memory_req = task_type["memoryRequirements"][new_replica[1].type["shortName"]]
            new_replica[0].available_memory -= memory_req

            # Add to replicas pool
            system_state.replicas[task_type["name"]].add(new_replica)

            # Initialize replica
            self.env.process(
                self.initialize_replica(
                    new_replica,
                    system_state.replicas[task_type["name"]],
                    task_type,
                    system_state,
                )
            )

            # Update statistics
            new_replica[1].last_allocated = self.env.now

            yield new_replica
            return

        except Exception as e:
            print(f"[GNN-AS] Error creating replica for {task_type['name']}: {str(e)}")
            # Return StopIteration with error message
            yield StopIteration(f"Failed to create replica for {task_type['name']}: {str(e)}")
            return

    def create_first_replica_og(self, system_state: SystemState, task_type: TaskType):
        # Knative will allocate a new CPU replica
        available_hardware: Set[str] = set()
        for _, platforms in system_state.available_resources.items():
            for platform in platforms:
                if (
                    platform.type["shortName"]
                    in task_type["platforms"]
                ):
                    available_hardware.add(platform.type["shortName"])
        
        # print(f"[GNN-AS] Available hardware types for task {task_type['name']}: {available_hardware}")

        stop = None
        # FIXME: What if no available hardware?
        if not available_hardware:
            print(f"[GNN-AS] No available hardware for task {task_type['name']}")
            return StopIteration(f"Could not create replica: No suitable platforms available")
            
        for platform_name in available_hardware:
            # print(f"[GNN-AS] Attempting to scale up {task_type['name']} on platform {platform_name}")
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
                # print(f"[GNN-AS] Successfully created first replica for {task_type['name']} on {platform_name}")
                break
            else:
                print(f"[GNN-AS] Failed to scale up on {platform_name}: {stop}")

        return stop

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

    def create_first_replica_on_node(self, system_state: SystemState, task_type: TaskType, node_name: str):
        # print(f"[GNN-AS] Starting create_first_replica_on_node for task {task_type['name']} on node {node_name}")
        # Generator functions in simpy need to yield if they might take time
        if False:
            yield
        
        # Find available platforms on the specified node
        available_hardware: Set[Tuple[Node, Platform]] = set()
        
        for node, platforms in system_state.available_resources.items():
            if node.node_name == node_name:
                # print(f"[GNN-AS] Found target node {node_name}, checking platforms")
                for platform in platforms:
                    if (platform.type["shortName"] in task_type["platforms"] and
                        node.memory >= task_type["memoryRequirements"][platform.type["shortName"]]):
                        # print(f"[GNN-AS] Found suitable platform {platform.type['shortName']} on node {node_name}")
                        available_hardware.add((node, platform))
                    # else:
                        # print(f"[GNN-AS] Platform {platform.type['shortName']} doesn't meet requirements for {task_type['name']} on node {node_name}")
                break  # Found the node we want, no need to continue searching
        
        if not available_hardware:
            # No suitable platforms on the specified node
            # print(f"[GNN-AS] Could not create replica for {task_type['name']} on node {node_name}: No suitable platforms available")
            return StopIteration(
                f"Could not create replica for {task_type['name']} on node {node_name}: "
                f"No suitable platforms available"
            )

        # Create replica using the first suitable platform found
        new_replica = next(iter(available_hardware))
        # print(f"[GNN-AS] Selected platform {new_replica[1].type['shortName']} on node {new_replica[0].node_name} for {task_type['name']}")
        
        try:
            # Remove selected platform from available resources on the node
            system_state.available_resources[new_replica[0]].remove(new_replica[1])
            # print(f"[GNN-AS] Removed platform from available resources on node {new_replica[0].node_name}")

            # Update node availability
            new_replica[0].available_platforms -= 1
            # print(f"[GNN-AS] Updated node {new_replica[0].node_name} availability: remaining platforms = {new_replica[0].available_platforms}")

            # Allocate task memory requirements from node's available memory
            memory_req = task_type["memoryRequirements"][new_replica[1].type["shortName"]]
            new_replica[0].available_memory -= memory_req
            # print(f"[GNN-AS] Allocated {memory_req} memory on node {new_replica[0].node_name}, remaining: {new_replica[0].available_memory}")

            # Add function replica to the pool
            system_state.replicas[task_type["name"]].add(new_replica)
            # print(f"[GNN-AS] Added replica to system state pool for {task_type['name']}")

            # Initialize replica (pull image)
            self.env.process(
                self.initialize_replica(
                    new_replica,
                    system_state.replicas[task_type["name"]],
                    task_type,
                    system_state,
                )
            )
            # print(f"[GNN-AS] Started initialization process for replica")

            # Statistics
            new_replica[1].last_allocated = self.env.now
            # print(f"[GNN-AS] Updated statistics, replica created at time {self.env.now}")

            logging.info(f"Successfully created replica for {task_type['name']} on node {node_name}")
            return new_replica

        except KeyError as e:
            print(f"[GNN-AS] Failed to create replica for {task_type['name']} on node {node_name}: {str(e)}")
            logging.error(
                f"[ {self.env.now} ] Failed to create replica for {task_type['name']} "
                f"on node {node_name}: {str(e)}"
            )
            return StopIteration(f"Failed to create replica: {str(e)}")

    def create_replica(
        self, couples_suitable: Set[Tuple[Node, Platform]], task_type: TaskType
    ):
        # print(f"[GNN-AS] Creating replica for {task_type['name']} from {len(couples_suitable)} suitable options")
        # Scaling functions that do not yield values must still be Generators
        # No-op as per https://stackoverflow.com/a/68628599/9568489
        if False:
            yield

        if not couples_suitable:
            print(f"[GNN-AS] No suitable couples available for creating replica")
            return None
            
        # Knative selects a replica on the most available node (cf. ENSURE)
        try:
            available_couple = max(
                couples_suitable,
                key=lambda couple: couple[0].available_platforms,
            )
            # print(f"[GNN-AS] Selected node {available_couple[0].node_name} with {available_couple[0].available_platforms} available platforms")
            # print(f"[GNN-AS] Platform type: {available_couple[1].type['shortName']}")
            
            return available_couple
        except ValueError as e:
            print(f"[GNN-AS] Error selecting node: {e}")
            return None

    def initialize_replica(
        self,
        new_replica: Tuple[Node, Platform],
        function_replicas: Set[Tuple[Node, Platform]],
        task_type: TaskType,
        system_state: GNNSystemState,
    ):
        # print(f"[GNN-AS] Initializing replica for {task_type['name']} on node {new_replica[0].node_name}")
        node: Node = new_replica[0]
        platform: Platform = new_replica[1]

        # Check node RAM cache
        warm_function: bool = (
            platform.previous_task is not None
            and platform.previous_task.type["name"] == task_type["name"]
        )
        
        if warm_function:
            # print(f"[GNN-AS] Warm function found in RAM cache for {task_type['name']} on node {node.node_name}")
            pass
        else:
            # print(f"[GNN-AS] Cold start: function {task_type['name']} not in RAM cache on node {node.node_name}")
            pass

        # Initialize image retrieval duration
        retrieval_duration: DurationSecond = 0.0

        # TODO: Retrieve image if function not in RAM cache nor in disk cache
        # FIXME: Should be factored in superclass
        if not warm_function:
            # print(f"[GNN-AS] Node {node.node_name} needs to pull image for {task_type['name']}")
            logging.info(
                f"[ {self.env.now} ] 💾 {node} needs to pull image for {task_type}"
            )

            # Update image retrieval duration
            retrieval_size: SizeGigabyte = task_type["imageSize"][
                platform.type["shortName"]
            ]
            # Depends on storage performance
            # FIXME: What's the policy for storage selection?
            # print(f"[GNN-AS] Getting storage for node {node.node_name}")
            node_storage = yield node.storage.get(
                lambda storage: not storage.type["remote"]
            )
            # print(f"[GNN-AS] Selected storage: {node_storage.type['name']}")
            
            # Depends on network link speed
            retrieval_speed: SpeedMBps = min(
                node_storage.type["throughput"]["write"], node.network["bandwidth"]
            )
            retrieval_duration += (
                retrieval_size / (retrieval_speed / 1024)
                + node_storage.type["latency"]["write"]
            )
            
            # print(f"[GNN-AS] Image size: {retrieval_size}GB, retrieval speed: {retrieval_speed}MB/s")
            # print(f"[GNN-AS] Estimated retrieval duration: {retrieval_duration} seconds")

            # TODO: Update disk usage
            stored = node_storage.store_function(platform.type["shortName"], task_type)

            if stored:
                # print(f"[GNN-AS] Successfully stored function image in node cache")
                pass
            else:
                print(f"[GNN-AS] Failed to store function image - insufficient capacity")
                logging.error(
                    f"[ {self.env.now} ] 💾 {node_storage} has no available capacity to"
                    f" cache image for {self}"
                )

            # Release storage
            # print(f"[GNN-AS] Releasing node storage")
            yield node.storage.put(node_storage)

        # Update state
        # FIXME: Move to state update methods
        # print(f"[GNN-AS] Updating scheduler state")
        state: GNNSchedulerState = system_state.scheduler_state
        # Knative policy
        state.average_contention[task_type["name"]][
            (new_replica[0].id, new_replica[1].id)
        ] = 1.0

        # FIXME: Retrieve function image
        # print(f"[GNN-AS] Waiting for image retrieval: {retrieval_duration} seconds")
        yield self.env.timeout(retrieval_duration)

        # FIXME: Update platform time spent on storage
        platform.storage_time += retrieval_duration
        # print(f"[GNN-AS] Updated platform storage time: {platform.storage_time}")

        # FIXME: Double initialize bug...
        try:
            # Set platform to ready state
            # print(f"[GNN-AS] Setting platform to initialized state")
            yield platform.initialized.succeed()
            # print(f"[GNN-AS] Platform successfully initialized")
        except RuntimeError:
            # print(f"[GNN-AS] Platform was already initialized (expected in some cases)")
            pass

        # Statistics (Node)
        node.cache_hits += 0
        # print(f"[GNN-AS] Replica initialization completed for {task_type['name']} on node {node.node_name}")

    def remove_replica(
        self,
        function_replicas: Set[Tuple[Node, Platform]],
        task_type: TaskType,
        system_state: GNNSystemState,
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

    def scale_up_alt(
        self,
        target_replicas: int,
        system_state: GNNSystemState,
        function_name: str,
        platform_name: str,
        **_,
    ) -> Generator:
        # print(f"[GNN-AS] Scale up request: {function_name} on {platform_name}, target: {target_replicas} replicas")
        # Determine how many replicas to add
        current_replicas = len(system_state.replicas[function_name])
        # print(f"[GNN-AS] Current replicas: {current_replicas}, target: {target_replicas}")
        replicas_delta = target_replicas - current_replicas

        if replicas_delta <= 0:
            # print(f"[GNN-AS] No scaling needed, current replicas ({current_replicas}) >= target ({target_replicas})")
            # No need to scale up
            return None

        # Get task type
        task_type = self.data.task_types[function_name]

        # Filter out nodes by task requirements
        couples_suitable: Set[Tuple[Node, Platform]] = set()
        # print(f"[GNN-AS] Finding suitable nodes and platforms for {function_name}")

        available_resources: Dict[Node, Set[Platform]] = system_state.available_resources
        for node, platforms in available_resources.items():
            for platform in platforms:
                # Skip if the platform is not of the right type
                if platform_name != "any" and platform.type["shortName"] != platform_name:
                    continue
                # Skip if task can't run on this platform
                if platform.type["shortName"] not in task_type["platforms"]:
                    continue
                # Skip if not enough memory
                if (
                    node.memory
                    < task_type["memoryRequirements"][platform.type["shortName"]]
                ):
                    continue
                
                # print(f"[GNN-AS] Found suitable node {node.node_name} with platform {platform.type['shortName']}")
                couples_suitable.add((node, platform))

        if not couples_suitable:
            print(f"[GNN-AS] No suitable nodes/platforms found for scaling {function_name}")
            # No suitable resources found
            return StopIteration(
                f"Could not find suitable resources to scale up {function_name}"
            )

        # Create replicas
        # print(f"[GNN-AS] Creating {replicas_delta} replicas for {function_name}")
        replicas_added = 0
        for _ in range(replicas_delta):
            # Get a replica
            try:
                new_replica = yield self.env.process(
                    self.create_replica(couples_suitable, task_type)
                )
                # print(f"[GNN-AS] Got new replica on node {new_replica[0].node_name}")
            except Exception as e:
                print(f"[GNN-AS] Failed to create replica: {e}")
                continue

            # Remove platform from available resources
            try:
                system_state.available_resources[new_replica[0]].remove(new_replica[1])
                # print(f"[GNN-AS] Removed platform from available resources")
            except KeyError:
                print(f"[GNN-AS] Platform not found in available resources")
                continue
            except ValueError:
                print(f"[GNN-AS] Platform not found in node's platform list")
                continue

            # Update node availability
            new_replica[0].available_platforms -= 1
            # print(f"[GNN-AS] Updated node availability: {new_replica[0].available_platforms} platforms left")

            # Allocate task memory requirements from node's available memory
            memory_req = task_type["memoryRequirements"][new_replica[1].type["shortName"]]
            new_replica[0].available_memory -= memory_req
            # print(f"[GNN-AS] Allocated {memory_req} memory, {new_replica[0].available_memory} left")

            # Add function replica to the pool
            system_state.replicas[function_name].add(new_replica)
            # print(f"[GNN-AS] Added replica to pool for {function_name}")

            # Remove combination from suitable couples
            try:
                couples_suitable.remove(new_replica)
                # print(f"[GNN-AS] Removed combination from suitable couples")
            except KeyError:
                # print(f"[GNN-AS] Combination not found in suitable couples")
                pass

            # Statistics
            new_replica[1].last_allocated = self.env.now

            # Initialize replica
            self.env.process(
                self.initialize_replica(
                    new_replica,
                    system_state.replicas[function_name],
                    task_type,
                    system_state,
                )
            )

            replicas_added += 1
            # print(f"[GNN-AS] Successfully initiated replica {replicas_added}/{replicas_delta}")

        return replicas_added