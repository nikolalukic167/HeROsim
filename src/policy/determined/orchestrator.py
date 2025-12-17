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
import sys
from typing import TYPE_CHECKING, Dict, Set, Tuple

from src.policy.determined.model import DeterminedSchedulerState, DeterminedSystemState

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform

from src.placement.orchestrator import Orchestrator


class DeterminedOrchestrator(Orchestrator):
    def __init__(self, *args, **kwargs):
        logger = logging.getLogger('simulation')
        logger.info("DeterminedOrchestrator: Starting initialization")
        # Extract infrastructure config if provided
        self.infrastructure = kwargs.pop('infrastructure', None) if 'infrastructure' in kwargs else None
        logger.info("DeterminedOrchestrator: Calling super().__init__")
        super().__init__(*args, **kwargs)
        logger.info("DeterminedOrchestrator: super().__init__ completed")
        
        # Pass forced placements to scheduler if available
        if self.infrastructure and 'forced_placements' in self.infrastructure:
            logger.info(f"DeterminedOrchestrator: Passing {len(self.infrastructure['forced_placements'])} forced placements to scheduler")
            self.scheduler.forced_placements = self.infrastructure['forced_placements']
            # CRITICAL: Log forced placements for debugging dnn2 issue
            if self.infrastructure.get('forced_placements'):
                print(f"[ORCHESTRATOR] Setting forced_placements: {self.infrastructure['forced_placements']}")
                logger = logging.getLogger('simulation')
                logger.info(f"[ORCHESTRATOR] Forced placements: {self.infrastructure['forced_placements']}")
            # print(f"[ {self.env.now} ] DeterminedOrchestrator: Passed {len(self.infrastructure['forced_placements'])} forced placements to scheduler")
        else:
            logger.error("DeterminedOrchestrator: No forced placements found in infrastructure")
            print(f"[ {self.env.now} ] DeterminedOrchestrator: No forced placements found in infrastructure")
            sys.exit(1)
        
        # Pass scheduler config (batch_size, batch_timeout) if available
        if self.infrastructure and 'scheduler' in self.infrastructure:
            scheduler_config = self.infrastructure['scheduler']
            if 'batch_size' in scheduler_config:
                self.scheduler.batch_size = scheduler_config['batch_size']
                logger.info(f"DeterminedOrchestrator: Set scheduler batch_size={self.scheduler.batch_size}")
            if 'batch_timeout' in scheduler_config:
                self.scheduler.batch_timeout = scheduler_config['batch_timeout']
                logger.info(f"DeterminedOrchestrator: Set scheduler batch_timeout={self.scheduler.batch_timeout}")
        
        logger.info("DeterminedOrchestrator: Initialization completed")
    
    def initialize_state(self) -> DeterminedSystemState:
        logger = logging.getLogger('simulation')
        logger.info("DeterminedOrchestrator: initialize_state called")
        # Initialize scheduler state
        logger.info("DeterminedOrchestrator: Creating DeterminedSchedulerState")
        scheduler_state = DeterminedSchedulerState(
            average_contention={task_type: {} for task_type in self.data.task_types},
            panic_contention={task_type: {} for task_type in self.data.task_types},
            target_concurrencies={
                task_type: {
                    # platform_type["shortName"]: self.policy.queue_length if platform_type["hardware"] == "cpu" else 0.0
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
        
        # Todo: remove this after testing
        # Seed initial replicas if provided
        logger.info(f"DeterminedOrchestrator: Checking initial_replicas (count: {len(self.initial_replicas) if self.initial_replicas else 0})")
        if self.initial_replicas:
            logger.info(f"DeterminedOrchestrator: Using {len(self.initial_replicas)} pre-seeded replica sets")
            print(f"\n=== Using {len(self.initial_replicas)} pre-seeded replica sets ===")
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
                            # print(f"    Allocated {node.node_name}:{platform.id} for {task_type} (memory: {memory_required}GB)")
                            
                            # Initialize average_contention for this replica to prevent KeyError
                            scheduler_state.average_contention[task_type][(node.id, platform.id)] = 0.0
                            # print(f"    Initialized contention tracking for {node.node_name}:{platform.id}")
            print("=== Initial replicas integrated ===\n")
        
        logger.info("DeterminedOrchestrator: Creating DeterminedSystemState")
        system_state = DeterminedSystemState(
            scheduler_state=scheduler_state,
            available_resources=available_resources,
            replicas=replicas,
            tasks=self.task_archive,
            time_series=self.time_series
        )
        logger.info("DeterminedOrchestrator: initialize_state completed")
        return system_state

    def monitor_process(self):
        # TODO: State initialization and update methods should be made abstract
        # and moved to policy package
        logger = logging.getLogger('simulation')
        logger.info(f"DeterminedOrchestrator: monitor_process started at time {self.env.now}")
        logging.info(f"[ {self.env.now} ] Orchestrator Monitor started")

        # Initialize time-window average
        latest_window_start = self.env.now
        logger.info("DeterminedOrchestrator: monitor_process entering main loop")

        while True:
            # Step
            step = math.floor(self.env.now - latest_window_start) + 1

            system_state: DeterminedSystemState = yield self.mutex.get()
            replicas: Dict[str, Set[Tuple[Node, Platform]]] = system_state.replicas
            state: DeterminedSchedulerState = system_state.scheduler_state

            # Clear average using time-window bounds if necessary
            # FIXME: Implement panic mode (60- vs 6-second time windows)
            if step == 7:
                # Store averages at the granularity of replicas
                for function_name, function_replicas in replicas.items():
                    # Accumulators
                    for node, platform in function_replicas:
                        # Knative policy
                        state.average_contention[function_name][
                            (node.id, platform.id)
                        ] = len(platform.queue.items)

                # Update tick time
                latest_window_start = self.env.now
            else:
                # Update contention rolling means
                for function_name, function_replicas in replicas.items():
                    for node, platform in function_replicas:
                        # Knative policy
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
