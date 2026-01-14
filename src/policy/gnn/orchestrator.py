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

import logging
import math
from typing import TYPE_CHECKING, Dict, Set, Tuple, List, Generator

from simpy.core import Environment, SimTime
from simpy.events import Event, Process
from simpy.resources.store import FilterStore, Store

from src.policy.gnn.model import KnativeSchedulerState, KnativeSystemState
from src.placement.infrastructure import Application, Task
from src.placement.model import (
    SimulationData,
    SimulationPolicy,
    TimeSeries,
    SystemStateResult,
)

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform

from src.placement.orchestrator import Orchestrator


class GNNOrchestrator(Orchestrator):
    def __init__(
            self,
            env: Environment,
            data: SimulationData,
            policy: SimulationPolicy,
            autoscaler,
            scheduler,
            time_series: TimeSeries,
            nodes: FilterStore,
            end_event: Event,
            trace_file: str,
            models=None,
            device_type_mapping=None,
            initial_replicas=None
    ):
        """Override __init__ to handle models specially for GNN scheduler."""
        self.env = env
        self.mutex = Store(env, capacity=1)
        self.data = data
        self.policy = policy

        self.time_series = time_series
        self.nodes = nodes
        self.initial_replicas = initial_replicas or {}
        self.models = models  # Store models for scheduler

        self.gateway: Process
        self.monitor: Process
        
        # Don't pass models to autoscaler - it doesn't need them
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
        self.system_state_results: List[SystemStateResult] = []
        
        # Set orchestrator reference on all nodes for system state capture
        for node in self.nodes.items:
            node.orchestrator_ref = self
        
        # Log initialization start
        logging.info("[GNN Orchestrator] Initialization starting")
        print("[GNN Orchestrator] Initialization starting")

    def initialize_state(self) -> KnativeSystemState:
        # Initialize scheduler state
        scheduler_state = KnativeSchedulerState(
            average_contention={task_type: {} for task_type in self.data.task_types},
            panic_contention={task_type: {} for task_type in self.data.task_types},
            target_concurrencies={
                task_type: {
                    platform_type["shortName"]: self.policy.queue_length
                    for platform_type in self.data.platform_types.values()
                }
                for task_type in self.data.task_types
            },
        )
        # Initialize available resources to all Tuple[Node, Platform]
        available_resources: Dict[Node, Set[Platform]] = {
            node: {platform for platform in set(node.platforms.items)}
            for node in set(self.nodes.items)
        }
        # Initialize function replicas to empty sets
        replicas: Dict[str, Set[Tuple[Node, Platform]]] = {
            task_type: set() for task_type in self.data.task_types
        }
        
        # Seed initial replicas if provided (from precreate_replicas in co-simulation mode)
        if self.initial_replicas:
            logging.info(f"GNNOrchestrator: Using {len(self.initial_replicas)} pre-seeded replica sets")
            print(f"\n=== GNN Orchestrator: Using {len(self.initial_replicas)} pre-seeded replica sets ===")
            for task_type, replica_set in self.initial_replicas.items():
                if task_type in replicas:
                    replicas[task_type] = replica_set.copy()
                    print(f"  {task_type}: {len(replica_set)} replicas")
                    
                    # Remove these platforms from available resources since they're now allocated
                    for node, platform in replica_set:
                        if node in available_resources and platform in available_resources[node]:
                            available_resources[node].remove(platform)
                            node.available_platforms -= 1
                            # Allocate memory for this replica
                            memory_required = self.data.task_types[task_type]["memoryRequirements"][platform.type["shortName"]]
                            node.available_memory -= memory_required
                            
                            # Initialize average_contention for this replica to prevent KeyError
                            scheduler_state.average_contention[task_type][(node.id, platform.id)] = 0.0
            print("=== Initial replicas integrated ===\n")
        
        system_state = KnativeSystemState(
            scheduler_state=scheduler_state,
            available_resources=available_resources,
            replicas=replicas,
            tasks=self.task_archive,
            time_series=self.time_series
        )

        # Pass models to scheduler if available
        try:
            if hasattr(self, 'models') and self.models and hasattr(self.scheduler, 'set_models'):
                logging.info("[GNN Orchestrator] Passing models to scheduler...")
                self.scheduler.set_models(self.models)
                print("[GNN Orchestrator] Models passed to scheduler")
                logging.info("[GNN Orchestrator] Models passed to scheduler successfully")
            elif not self.models:
                logging.warning("[GNN Orchestrator] No models provided - scheduler will use fallback")
                print("[GNN Orchestrator] WARNING: No models provided - scheduler will use fallback")
            else:
                logging.warning(f"[GNN Orchestrator] Models check failed: hasattr={hasattr(self, 'models')}, models={self.models}, has_set_models={hasattr(self.scheduler, 'set_models')}")
        except Exception as e:
            logging.error(f"[GNN Orchestrator] Error passing models to scheduler: {e}")
            print(f"[GNN Orchestrator] ERROR: Failed to pass models to scheduler: {e}")
            import traceback
            traceback.print_exc()
            # Continue anyway - scheduler should handle missing models gracefully
        
        logging.info("[GNN Orchestrator] State initialization complete")
        return system_state

    def initializer_process(self) -> Generator:
        """Override to add error handling for state initialization."""
        try:
            logging.info("[GNN Orchestrator] Starting initializer process...")
            print("[GNN Orchestrator] Starting initializer process...")
            
            # Initialize shared data structures according to simulation policy
            system_state: KnativeSystemState = self.initialize_state()
            
            # Putting it all together...
            yield self.mutex.put(system_state)
            logging.info("[GNN Orchestrator] System state put in mutex")

            # Register any precreated warmup tasks so they can appear in logs/stats
            try:
                warmup_task_count = 0
                for node in self.nodes.items:
                    for plat in node.platforms.items:
                        if hasattr(plat, '_warmup_tasks') and plat._warmup_tasks:
                            for t in plat._warmup_tasks:
                                if t.application not in self.application_archive:
                                    self.application_archive.append(t.application)
                                if t not in self.task_archive:
                                    self.task_archive.append(t)
                                    warmup_task_count += 1
                if warmup_task_count > 0:
                    logging.info(f"Registered {warmup_task_count} warmup tasks (GNN mode)")
                    print(f"Registered {warmup_task_count} warmup tasks (GNN mode)")
            except Exception as e:
                logging.warning(f"[GNN Orchestrator] Error registering warmup tasks: {e}")

            # Begin orchestration
            logging.info("[GNN Orchestrator] Starting gateway, monitor, autoscaler, and scheduler processes...")
            self.gateway = self.env.process(self.gateway_process())
            self.monitor = self.env.process(self.monitor_process())
            self.autoscaler.run = self.env.process(self.autoscaler.autoscaler_process())
            self.scheduler.run = self.env.process(self.scheduler.scheduler_process())
            logging.info("[GNN Orchestrator] All processes started successfully")
            print("[GNN Orchestrator] All processes started successfully")
            
        except Exception as e:
            logging.error(f"[GNN Orchestrator] CRITICAL ERROR in initializer_process: {e}")
            print(f"[GNN Orchestrator] CRITICAL ERROR in initializer_process: {e}")
            import traceback
            traceback.print_exc()
            # Re-raise to let SimPy handle it
            raise

    def monitor_process(self):
        logging.info(f"[ {self.env.now} ] GNN Orchestrator Monitor started")

        # Initialize time-window average
        latest_window_start = self.env.now

        while True:
            # Step
            step = math.floor(self.env.now - latest_window_start) + 1

            system_state: KnativeSystemState = yield self.mutex.get()
            replicas: Dict[str, Set[Tuple[Node, Platform]]] = system_state.replicas
            state: KnativeSchedulerState = system_state.scheduler_state

            # Clear average using time-window bounds if necessary
            if step == 7:
                # Store averages at the granularity of replicas
                for function_name, function_replicas in replicas.items():
                    for node, platform in function_replicas:
                        state.average_contention[function_name][
                            (node.id, platform.id)
                        ] = len(platform.queue.items)

                # Update tick time
                latest_window_start = self.env.now
            else:
                # Update contention rolling means
                for function_name, function_replicas in replicas.items():
                    for node, platform in function_replicas:
                        value = (
                            state.average_contention[function_name][
                                (node.id, platform.id)
                            ]
                            * (step - 1)
                            + len(platform.queue.items)
                        ) / step
                        state.average_contention[function_name][
                            (node.id, platform.id)
                        ] = value

            yield self.mutex.put(system_state)

            # Wake Monitor up once per second
            yield self.env.timeout(1)
