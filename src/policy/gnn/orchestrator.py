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
from typing import TYPE_CHECKING, Dict, Set, Tuple

from src.policy.gnn.model import KnativeSchedulerState, KnativeSystemState

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform

from src.placement.orchestrator import Orchestrator


class GNNOrchestrator(Orchestrator):
    """GNN Orchestrator - simplified to match knative_network structure.
    
    The GNN model is passed via the models parameter and forwarded to the scheduler.
    """
    
    def __init__(self, *args, models=None, **kwargs):
        """Initialize orchestrator with optional GNN models."""
        # Remove unsupported kwargs before calling parent
        kwargs.pop('initial_replicas', None)
        kwargs.pop('scheduler_config', None)
        kwargs.pop('device_type_mapping', None)
        
        # Store models temporarily - will be overwritten by parent's __init__
        _models = models
        print(f"[GNN Orchestrator] __init__ called with models={models is not None}", flush=True)
        if models:
            print(f"[GNN Orchestrator] models type: {type(models)}, keys: {list(models.keys()) if isinstance(models, dict) else 'N/A'}", flush=True)
        
        # Call parent init (which sets self.models = None since models not in kwargs)
        super().__init__(*args, **kwargs)
        
        # Re-set self.models after parent init
        self.models = _models
        print(f"[GNN Orchestrator] After super().__init__, self.models restored: {self.models is not None}", flush=True)
    
    def initialize_state(self) -> KnativeSystemState:
        """Initialize system state - matches knative_network."""
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
        system_state = KnativeSystemState(
            scheduler_state=scheduler_state,
            available_resources=available_resources,
            replicas=replicas,
            tasks=self.task_archive,
            time_series=self.time_series
        )

        # Pass models to scheduler if available
        print(f"[GNN Orchestrator] initialize_state called, models={self.models is not None}", flush=True)
        print(f"[GNN Orchestrator] scheduler has set_models: {hasattr(self.scheduler, 'set_models')}", flush=True)
        if self.models:
            print(f"[GNN Orchestrator] models keys: {list(self.models.keys()) if isinstance(self.models, dict) else 'not a dict'}", flush=True)
            if hasattr(self.scheduler, 'set_models'):
                self.scheduler.set_models(self.models)
                print("[GNN Orchestrator] Models passed to scheduler", flush=True)
            else:
                print("[GNN Orchestrator] WARNING: scheduler doesn't have set_models method!", flush=True)
        else:
            print("[GNN Orchestrator] WARNING: self.models is None or empty!", flush=True)

        return system_state

    def monitor_process(self):
        """Monitor process - matches knative_network (simple direct queue count)."""
        logging.info(f"[ {self.env.now} ] GNN Orchestrator Monitor started")

        while True:
            system_state: KnativeSystemState = yield self.mutex.get()
            replicas: Dict[str, Set[Tuple[Node, Platform]]] = system_state.replicas
            state: KnativeSchedulerState = system_state.scheduler_state

            # Count queue depth for autoscaling (direct assignment like knative_network)
            for function_name, function_replicas in replicas.items():
                for node, platform in function_replicas:
                    state.average_contention[function_name][
                        (node.id, platform.id)
                    ] = len(platform.queue.items)

            yield self.mutex.put(system_state)

            # Wake Monitor up once per second
            yield self.env.timeout(1)
