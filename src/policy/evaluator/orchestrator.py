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

from src.policy.evaluator.model import EvaluatorSchedulerState, EvaluatorSystemState

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform

from src.placement.orchestrator import Orchestrator


class EvaluatorOrchestrator(Orchestrator):
    def __init__(self, *args, **kwargs):
        # Extract infrastructure config if provided
        self.infrastructure = kwargs.pop('infrastructure', None) if 'infrastructure' in kwargs else None
        super().__init__(*args, **kwargs)
    
    def initialize_state(self) -> EvaluatorSystemState:
        # Initialize scheduler state
        scheduler_state = EvaluatorSchedulerState(
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
        if self.initial_replicas:
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
        
        system_state = EvaluatorSystemState(
            scheduler_state=scheduler_state,
            available_resources=available_resources,
            replicas=replicas,
            tasks=self.task_archive,
            time_series=self.time_series
        )

        return system_state

    def monitor_process(self):
        # TODO: State initialization and update methods should be made abstract
        # and moved to policy package
        logging.info(f"[ {self.env.now} ] Orchestrator Monitor started")

        # Initialize time-window average
        latest_window_start = self.env.now

        while True:
            # Step
            step = math.floor(self.env.now - latest_window_start) + 1

            system_state: EvaluatorSystemState = yield self.mutex.get()
            replicas: Dict[str, Set[Tuple[Node, Platform]]] = system_state.replicas
            state: EvaluatorSchedulerState = system_state.scheduler_state

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
