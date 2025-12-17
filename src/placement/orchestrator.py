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

import dataclasses
import logging
import statistics
from abc import abstractmethod
from collections import defaultdict
from graphlib import TopologicalSorter
from typing import Dict, Generator, List, Tuple, Type

from simpy.core import Environment, SimTime
from simpy.events import Event, Process
from simpy.resources.store import Store, FilterStore

from src.placement.autoscaler import Autoscaler
from src.placement.infrastructure import Application, Task
from src.placement.model import (
    ApplicationResult,
    MomentSecond,
    NodeResult,
    PlatformResult,
    SimulationData,
    SimulationPolicy,
    SystemState,
    TaskResult,
    SimulationStats,
    TimeSeries,
    WorkloadEvent,
    SystemStateResult,
)
from src.placement.scheduler import Scheduler


def check_serializable(obj, path=""):
    """Recursively check if an object is JSON serializable and log any issues."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, (str, int, float, bool, type(None))):
                logging.error(f"Non-serializable key at {path}.{key}: {type(key)}")
            check_serializable(value, f"{path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            check_serializable(item, f"{path}[{i}]")
    elif isinstance(obj, (str, int, float, bool, type(None))):
        pass  # These are serializable
    else:
        logging.error(f"Non-serializable value at {path}: {type(obj)} = {obj}")


# todo: add network latency statistics
class Orchestrator:
    def __init__(
            self,
            env: Environment,
            data: SimulationData,
            policy: SimulationPolicy,
            autoscaler: Type[Autoscaler],
            scheduler: Type[Scheduler],
            time_series: TimeSeries,
            nodes: FilterStore,
            end_event: Event,
            trace_file: str,
            models=None,
            device_type_mapping=None,
            initial_replicas=None
    ):
        self.env = env
        self.mutex = Store(env, capacity=1)
        self.data = data
        self.policy = policy

        self.time_series = time_series
        self.nodes = nodes
        self.initial_replicas = initial_replicas or {}

        self.gateway: Process
        self.monitor: Process
        if models is not None and isinstance(models, dict) and len(models) > 0:
            self.autoscaler = autoscaler(self.env, self.mutex, self.data, self.policy, models)
        elif models is not None and device_type_mapping is not None:
            self.autoscaler = autoscaler(self.env, self.mutex, self.data, self.policy, models, device_type_mapping)
        else:
            self.autoscaler = autoscaler(self.env, self.mutex, self.data, self.policy)
        self.scheduler = scheduler(
            self.env, self.mutex, self.data, self.policy, self.autoscaler, self.nodes
        )
        self.initializer = env.process(self.initializer_process())

        self.end_event = end_event
        self.end_time: SimTime

        self.application_archive: List[Application] = []
        self.task_archive: List[Task] = []
        self.trace_file = trace_file
        self.system_state_results: List[SystemStateResult] = []  # Store system state snapshots
        
        # Set orchestrator reference on all nodes for system state capture
        for node in self.nodes.items:
            node.orchestrator_ref = self

    def stats(self) -> SimulationStats:
        logger = logging.getLogger('simulation')
        
        try:
            application_results: List[ApplicationResult] = [
                application.result() for application in self.application_archive
                if not all(getattr(task, 'is_internal', False) for task in application.tasks)
            ]
            task_results: List[TaskResult] = [
                task.result() for task in self.task_archive
                # if not getattr(task, 'is_internal', False)
            ]
            node_results: List[NodeResult] = [
                node.result() for node in self.nodes.items
            ]
            platform_results: List[PlatformResult] = [
                platform.result()
                for node in self.nodes.items
                for platform in node.platforms.items
            ]
        except KeyError as e:
            raise e

        # Unused platforms (% of platform count)
        unused_platforms = len(
            [
                platform_result
                for platform_result in platform_results
                if platform_result["idleProportion"] == 100
            ]
        ) / len(platform_results)

        # Unused nodes (% of node count)
        unused_nodes = len(
            [node_result for node_result in node_results if node_result["unused"]]
        ) / len(node_results)

        # Average resource occupation time
        resources_occupation: Dict[int, float] = {}
        for platform_result in sorted(
                platform_results, key=lambda result: result["platformId"]
        ):
            """
            logging.error(
                f"{platform_result['platformId']}"
                f" ({platform_result['platformType']['hardware']})"
                f" -- {round(100 - platform_result['idleProportion'], 2)}%"
            )
            """
            resources_occupation[platform_result["platformId"]] = (
                    100 - platform_result["idleProportion"]
            )

        average_occupation = sum(resources_occupation.values()) / len(
            resources_occupation
        )

        # print("Scheduling times on each node:")
        # print(
        #     *(node_result["schedulingTime"] for node_result in node_results), sep="\n"
        # )

        average_elapsed_time = sum(
            task_result["elapsedTime"] for task_result in task_results
        ) / len(task_results)
        average_pull_time = sum(
            task_result["pullTime"] for task_result in task_results
        ) / len(task_results)
        average_cold_start_time = sum(
            task_result["coldStartTime"] for task_result in task_results
        ) / len(task_results)
        average_execution_time = sum(
            task_result["executionTime"] for task_result in task_results
        ) / len(task_results)
        average_wait_time = sum(
            task_result["waitTime"] for task_result in task_results
        ) / len(task_results)
        average_queue_time = sum(
            task_result["queueTime"] for task_result in task_results
        ) / len(task_results)
        average_initialization_time = sum(
            task_result["initializationTime"] for task_result in task_results
        ) / len(task_results)
        average_compute_time = sum(
            task_result["computeTime"] for task_result in task_results
        ) / len(task_results)
        average_communications_time = sum(
            task_result["communicationsTime"] for task_result in task_results
        ) / len(task_results)

        energy_total = sum(task_result["energy"] for task_result in task_results) + sum(
            platform_result["energyIdle"] for platform_result in platform_results
        )

        unused_nodes_idle_energy = [
            node_result["energyIdle"]
            for node_result in node_results
            if node_result["unused"]
        ]
        unused_platforms_idle_energy = [
            sum(unused_platforms.values())
            for unused_platforms in unused_nodes_idle_energy
        ]
        reclaimable_energy = sum(unused_platforms_idle_energy)

        penalty_proportion = sum(
            application_result["penalty"] for application_result in application_results
        ) / len(application_results)

        local_dependencies_proportion = sum(
            task_result["localDependencies"] for task_result in task_results
        ) / len(task_results)
        local_communications_proportion = sum(
            task_result["localCommunications"] for task_result in task_results
        ) / len(task_results)

        cold_start_proportion = sum(
            task_result["coldStarted"] for task_result in task_results
        ) / len(task_results)
        node_cache_hits_proportion = sum(
            node_result["cacheHits"] for node_result in node_results
        ) / len(task_results)
        task_cache_hit_proportion = sum(
            task_result["cacheHit"] for task_result in task_results
        ) / len(task_results)

        # Compute quantiles with defensive checks
        logger.info(f"[STATS] Computing task_response_time_quantiles from {len(task_results)} task results")
        task_elapsed_times = [task["elapsedTime"] for task in task_results]
        logger.info(f"[STATS] Task elapsed times: count={len(task_elapsed_times)}, values={task_elapsed_times[:10] if len(task_elapsed_times) > 0 else 'empty'}")
        
        if len(task_elapsed_times) < 2:
            logger.warning(f"[STATS] Cannot compute quantiles: need at least 2 data points, got {len(task_elapsed_times)}")
            logger.warning(f"[STATS] Using fallback: single value or empty list")
            if len(task_elapsed_times) == 1:
                # Single value: return list with that value repeated
                task_response_time_quantiles = [task_elapsed_times[0]] * 100
                logger.info(f"[STATS] Using single value {task_elapsed_times[0]} for all quantiles")
            else:
                # Empty list: return list of zeros
                task_response_time_quantiles = [0.0] * 100
                logger.warning(f"[STATS] No task results available, using zeros for quantiles")
        else:
            try:
                task_response_time_quantiles = statistics.quantiles(task_elapsed_times, n=100)
                logger.info(f"[STATS] Successfully computed task_response_time_quantiles: {len(task_response_time_quantiles)} quantiles")
            except Exception as e:
                logger.error(f"[STATS] Error computing task_response_time_quantiles: {e}")
                logger.error(f"[STATS] Task elapsed times: {task_elapsed_times}")
                raise
        
        logger.info(f"[STATS] Computing application_response_time_quantiles from {len(application_results)} application results")
        application_elapsed_times = [application["elapsedTime"] for application in application_results]
        logger.info(f"[STATS] Application elapsed times: count={len(application_elapsed_times)}, values={application_elapsed_times[:10] if len(application_elapsed_times) > 0 else 'empty'}")
        
        if len(application_elapsed_times) < 2:
            logger.warning(f"[STATS] Cannot compute quantiles: need at least 2 data points, got {len(application_elapsed_times)}")
            logger.warning(f"[STATS] Using fallback: single value or empty list")
            if len(application_elapsed_times) == 1:
                # Single value: return list with that value repeated
                application_response_time_quantiles = [application_elapsed_times[0]] * 100
                logger.info(f"[STATS] Using single value {application_elapsed_times[0]} for all quantiles")
            else:
                # Empty list: return list of zeros
                application_response_time_quantiles = [0.0] * 100
                logger.warning(f"[STATS] No application results available, using zeros for quantiles")
        else:
            try:
                application_response_time_quantiles = statistics.quantiles(application_elapsed_times, n=100)
                logger.info(f"[STATS] Successfully computed application_response_time_quantiles: {len(application_response_time_quantiles)} quantiles")
            except Exception as e:
                logger.error(f"[STATS] Error computing application_response_time_quantiles: {e}")
                logger.error(f"[STATS] Application elapsed times: {application_elapsed_times}")
                raise

        # Sort task results by arrival time
        # Filter out non-penalty tasks
        penalty_distribution_over_time: List[Tuple[MomentSecond, float]] = []
        applications_count = 0
        distribution = 0
        for application_result in sorted(
                application_results, key=lambda app_res: app_res["dispatchedTime"]
        ):
            applications_count += 1
            if application_result["penalty"]:
                distribution += 1
                penalty_distribution_over_time.append(
                    (
                        application_result["dispatchedTime"],
                        distribution / applications_count,
                    )
                )

        # Calculate network statistics
        average_network_latency = sum(
            task_result["networkLatency"] for task_result in task_results
        ) / len(task_results)

        # Calculate per-node-pair latencies
        node_pair_latencies = defaultdict(list)
        for task_result in task_results:
            if task_result["sourceNode"] != task_result["executionNode"]:
                pair = (task_result["sourceNode"], task_result["executionNode"])
                node_pair_latencies[pair].append(task_result["networkLatency"])

        # don't touch this, this works and is used in the notebook
        average_node_pair_latencies = {
            f"{pair[0]}->{pair[1]}": sum(latencies)/len(latencies)
            for pair, latencies in node_pair_latencies.items()
        }

        # Extract network topology from nodes
        network_topology = {}
        for node in self.nodes.items:
            network_topology[node.node_name] = node.network_map

        result = {
            "policy": dataclasses.asdict(self.policy),
            "endTime": self.end_time,
            'traceFile': self.trace_file,
            "unusedPlatforms": unused_platforms * 100,
            "unusedNodes": unused_nodes * 100,
            "averageOccupation": average_occupation,
            "averageElapsedTime": average_elapsed_time,
            "averagePullTime": average_pull_time,
            "averageColdStartTime": average_cold_start_time,
            "averageExecutionTime": average_execution_time,
            "averageWaitTime": average_wait_time,
            "averageQueueTime": average_queue_time,
            "averageInitializationTime": average_initialization_time,
            "averageComputeTime": average_compute_time,
            "averageCommunicationsTime": average_communications_time,
            "penaltyProportion": penalty_proportion * 100,
            "localDependenciesProportion": local_dependencies_proportion * 100,
            "localCommunicationsProportion": local_communications_proportion * 100,
            "nodeCacheHitsProportion": node_cache_hits_proportion * 100,
            "taskCacheHitsProportion": task_cache_hit_proportion * 100,
            "coldStartProportion": cold_start_proportion * 100,
            "taskResponseTimeDistribution": task_response_time_quantiles,
            "applicationResponseTimeDistribution": application_response_time_quantiles,
            "penaltyDistributionOverTime": penalty_distribution_over_time,
            "energy": energy_total,
            "reclaimableEnergy": reclaimable_energy,
            "applicationResults": application_results,
            "nodeResults": node_results,
            "taskResults": task_results,
            "scaleEvents": self.autoscaler.scale_events,
            "systemEvents": self.autoscaler.system_status_events,
            "averageNetworkLatency": average_network_latency,
            "nodePairLatencies": average_node_pair_latencies,
            "networkTopology": network_topology,
            "offloadingRate": len([
                task for task in task_results
                if task["sourceNode"] != task["executionNode"]
            ]) / len(task_results) * 100,
            "systemStateResults": self.system_state_results,
        }
        
        # Debug: Check for non-serializable types
        logging.info("Checking for non-serializable types in stats...")
        check_serializable(result, "stats")
        
        return result

    def create_application(
            self, env: Environment, app_id: int, task_id: int, event: WorkloadEvent
    ) -> Application:
        application_type = event["application"]
        qos_type = event["qos"]

        application_tasks: List[Task] = []
        application = Application(
            id=app_id,
            dispatched_time=event["timestamp"],
            application_type=application_type,
            qos_type=qos_type,
            tasks=application_tasks,
        )

        # TODO: Traverse application DAG
        sorter = TopologicalSorter(application_type["dag"])
        ordered = tuple(sorter.static_order())

        # TODO: Create dependencies
        dependencies: Dict[str, List[Task]] = {}
        function_tasks: Dict[str, Task] = {}

        for function_name in ordered:
            if function_name not in dependencies:
                dependencies[function_name] = []

            function_task = Task(
                env=env,
                task_id=task_id,
                task_type=self.data.task_types[function_name],
                application=application,
                dependencies=dependencies[function_name],
                policy=self.policy,
                node_name=event["node_name"]
            )

            task_id += 1
            application_tasks.append(function_task)
            function_tasks[function_name] = function_task

        for function_name in ordered:
            predecessors = application_type["dag"][function_name]

            for predecessor_name in predecessors:
                dependencies[function_name].append(function_tasks[predecessor_name])

        return application

    @abstractmethod
    def initialize_state(self) -> SystemState:
        pass

    def initializer_process(self) -> Generator:
        # Initialize shared data structures according to simulation policy
        system_state: SystemState = self.initialize_state()
        # Putting it all together...
        yield self.mutex.put(system_state)

        # Register any precreated warmup tasks so they can appear in logs/stats
        # NOTE: Warmup tasks are ONLY created by executecosimulation.py (co-simulation mode)
        # executeinitial.py does NOT create warmup tasks, so this code should find nothing
        # when running from executeinitial.py
        try:
            warmup_task_count = 0
            for node in self.nodes.items:
                for plat in node.platforms.items:
                    if hasattr(plat, '_warmup_tasks') and plat._warmup_tasks:
                        for t in plat._warmup_tasks:
                            # Archive task and its pseudo-application
                            # Warmup tasks are marked with is_internal=True and excluded from completion wait
                            if t.application not in self.application_archive:
                                self.application_archive.append(t.application)
                            if t not in self.task_archive:
                                self.task_archive.append(t)
                                warmup_task_count += 1
            if warmup_task_count > 0:
                logging.info(f"Registered {warmup_task_count} warmup tasks (co-simulation mode)")
                print(f"Registered {warmup_task_count} warmup tasks (co-simulation mode)")
        except Exception:
            pass

        # Begin orchestration
        self.gateway = self.env.process(self.gateway_process())
        self.monitor = self.env.process(self.monitor_process())
        self.autoscaler.run = self.env.process(self.autoscaler.autoscaler_process())
        self.scheduler.run = self.env.process(self.scheduler.scheduler_process())

    @abstractmethod
    def monitor_process(self) -> Generator:
        pass

    def workflow_process(self, task: Task) -> Generator:
        # Find next task in the application
        task_dag = task.application.type["dag"]
        sorter = TopologicalSorter(task_dag)
        ordered = tuple(sorter.static_order())
        current_index = ordered.index(task.type["name"])

        # If current task is the last task of the application, clear application data
        # FIXME
        # first_task = task.application.tasks[0]
        # first_task.storage["input"].remove_data(first_task)
        if current_index == len(ordered) - 1:
            for application_task in task.application.tasks:
                application_task.storage["input"]
                output_storage = application_task.storage["output"]

                if output_storage:
                    output_storage.remove_data(application_task)

            return

        # Else, schedule next task to be run after current task finishes its execution
        next_task_name = ordered[current_index + 1]
        next_task = next(
            filter(
                lambda app_task: app_task.type["name"] == next_task_name,
                task.application.tasks,
            )
        )

        # Wait for current task execution
        yield task.done

        # Dispatch next task
        yield next_task.dispatched.succeed()

        # Put next task in scheduler queue
        yield self.scheduler.tasks.put(next_task)

        # Monitor workflow execution
        self.env.process(self.workflow_process(next_task))

    def gateway_process(self) -> Generator:
        print(f"[ {self.env.now} ] API Gateway started with {len(self.time_series.events)} events")

        app_id = 0
        task_id = 0

        # TODO: maybe pop more tasks at once here?
        events_processed = 0
        while self.time_series.events:
            events_processed += 1
            remaining = len(self.time_series.events)
            print(f"[ {self.env.now} ] Gateway: Processing event {events_processed} ({remaining} remaining)")
            # Process workload events (FIFO)
            workload_event: WorkloadEvent = self.time_series.events.pop(0)

            # Timeout until event timestamp
            time_until_next_event = workload_event["timestamp"] - self.env.now
            yield self.env.timeout(time_until_next_event)

            # Create the application according to the event properties
            app = self.create_application(
                env=self.env,
                app_id=app_id,
                task_id=task_id,
                event=workload_event,
            )

            # Increment application and task IDs
            app_id += 1
            task_id += len(app.tasks)

            # Tasks are stored in an archive for further analysis
            self.application_archive.append(app)
            self.task_archive.extend(app.tasks)

            # Start counting first task time from here
            first_task: Task = app.tasks[0]
            yield first_task.dispatched.succeed()

            # Subsequent tasks in application DAG will be dispatched later
            # workflow_process waits for task completion before dispatching next task
            self.env.process(self.workflow_process(first_task))

            # Tasks are stored in a queue to be scheduled on execution platforms
            # See scheduler_process()
            yield self.scheduler.tasks.put(first_task)

        # All workload events have been processed - gateway is done
        print(f"[ {self.env.now} ] Gateway: All {len(self.task_archive)} tasks from {len(self.application_archive)} applications have been dispatched")
        logging.info(f"[ {self.env.now} ] Gateway: All workload events processed, waiting for task completion")
        
        # Simulation ends when:
        #  - all platforms are released
        #  - all tasks are done
        yield self.env.all_of([task.done for task in self.task_archive])
        
        # End simulation
        # Capture final system state
        system_state = yield self.mutex.get()
        self.system_state_results.append(system_state.result(self.env.now))
        yield self.mutex.put(system_state)
        self.end_time = self.env.now
        yield self.end_event.succeed()
