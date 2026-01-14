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
from typing import TYPE_CHECKING, Dict, Set, Tuple

from src.placement.model import SystemState, SchedulerState, PlatformVector

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform

from src.placement.orchestrator import Orchestrator


class RoundRobinNetworkOrchestrator(Orchestrator):
    def initialize_state(self) -> SystemState:
        # Initialize scheduler state with scheduled_count for round-robin
        scheduler_state = SchedulerState(
            target_concurrencies={
                task_type: {
                    platform_type["shortName"]: self.policy.queue_length
                    for platform_type in self.data.platform_types.values()
                }
                for task_type in self.data.task_types
            },
        )
        # Initialize scheduled_count for round-robin scheduling
        scheduler_state.scheduled_count = {
            task_type: {} for task_type in self.data.task_types
        }
        
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
            logging.info(f"RoundRobinNetworkOrchestrator: Using {len(self.initial_replicas)} pre-seeded replica sets")
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
                            
                            # Initialize scheduled_count for this replica
                            scheduler_state.scheduled_count[task_type][(node.id, platform.id)] = 0
            print("=== Initial replicas integrated ===\n")
        
        system_state = SystemState(
            scheduler_state=scheduler_state,
            available_resources=available_resources,
            replicas=replicas,
        )

        return system_state

    def monitor_process(self):
        # Round-robin doesn't need special monitoring like knative
        # Just use the base monitor process
        logging.info(f"[ {self.env.now} ] Orchestrator Monitor started")
        while True:
            yield self.env.timeout(1)
