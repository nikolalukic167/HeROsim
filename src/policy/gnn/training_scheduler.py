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
import time
from typing import Set, Tuple, TYPE_CHECKING, Generator, Dict, List, Any, Optional
from collections import defaultdict

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task

from src.placement.model import SystemState
from src.placement.scheduler import Scheduler


class TrainingScheduler(Scheduler):
    """
    Scheduler designed for generating training data with predefined client-server pairs.
    This scheduler makes static conditions and forces specific offloading decisions
    to generate fully populated graphs for GNN training.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_mode = True
        self.static_conditions = {}
        self.force_offloading = True  # Always force offloading for training
        
        # Track all execution times for each client-server pair
        self.execution_times: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
        
        # Static system conditions for consistent training data
        self.static_queue_lengths = {}
        self.static_memory_usage = {}
        self.static_storage_usage = {}
        self.static_network_latencies = {}

    def set_static_conditions(self, conditions: Dict[str, Any]):
        """Set static conditions for consistent training data generation"""
        self.static_conditions = conditions
        
        # Extract specific static values
        self.static_queue_lengths = conditions.get('queue_lengths', {})
        self.static_memory_usage = conditions.get('memory_usage', {})
        self.static_storage_usage = conditions.get('storage_usage', {})
        self.static_network_latencies = conditions.get('network_latencies', {})

    def _apply_static_conditions(self, system_state: SystemState):
        """Apply static conditions to make dynamic variables constant"""
        # Apply static queue lengths to all platforms
        for node in self.nodes.items:
            for platform in node.platforms.items:
                if platform.id in self.static_queue_lengths:
                    # Clear existing queue and add static number of dummy tasks
                    platform.queue.items.clear()
                    static_length = self.static_queue_lengths[platform.id]
                    for _ in range(static_length):
                        # Add dummy task to maintain queue length
                        platform.queue.items.append(None)  # Placeholder
        
        # Apply static memory usage
        for node in self.nodes.items:
            if node.node_name in self.static_memory_usage:
                node.memory_usage = self.static_memory_usage[node.node_name]
        
        # Apply static storage usage
        for node in self.nodes.items:
            for storage in node.storage.items:
                if storage.id in self.static_storage_usage:
                    storage.used = self.static_storage_usage[storage.id]

    def _find_target_server(self, task: Task) -> Optional[Tuple[Node, Platform]]:
        """Find the target server based on workload event information"""
        # Check if task has target server information from workload
        if hasattr(task, 'target_server') and task.target_server:
            target_server_name = task.target_server
            
            # Find the target server node
            for node in self.nodes.items:
                if node.node_name == target_server_name:
                    # Find a suitable platform on this node
                    for platform in node.platforms.items:
                        if platform.type["shortName"] in task.type["platforms"]:
                            return node, platform
            
            logging.warning(f"Target server {target_server_name} not found for task {task.id}")
        
        return None

    def _force_offload_to_target(self, task: Task, system_state: SystemState) -> Optional[Tuple[Node, Platform]]:
        """Force offloading to the predefined target server"""
        target = self._find_target_server(task)
        if target:
            target_node, target_platform = target
            
            # Check if target platform is available
            if target_platform in system_state.replicas.get(task.type["name"], set()):
                logging.info(f"[TRAINING] Forcing task {task.id} from {task.node_name} to {target_node.node_name}")
                return target_node, target_platform
        
        return None

    def _generate_all_execution_times(self, task: Task, system_state: SystemState) -> Dict[Tuple[str, str], float]:
        """Generate execution times for all possible client-server pairs"""
        execution_times = {}
        
        # Get all nodes that could be clients and servers
        all_nodes = list(self.nodes.items)
        
        for client_node in all_nodes:
            for server_node in all_nodes:
                if client_node.node_name == server_node.node_name:
                    continue  # Skip self-execution
                
                # Calculate execution time for this client-server pair
                execution_time = self._calculate_execution_time(
                    task, client_node, server_node, system_state
                )
                
                pair_key = (client_node.node_name, server_node.node_name)
                execution_times[pair_key] = execution_time
                
                # Store for training data
                self.execution_times[(client_node.node_name, server_node.node_name, task.type["name"])].append(execution_time)
        
        return execution_times

    def _calculate_execution_time(self, task: Task, client_node: Node, server_node: Node, system_state: SystemState) -> float:
        """Calculate execution time for a specific client-server pair"""
        # Base execution time from task type and platform
        base_execution_time = 0.0
        
        # Find best platform on server node
        for platform in server_node.platforms.items:
            if platform.type["shortName"] in task.type["platforms"]:
                base_execution_time = task.type["executionTime"][platform.type["shortName"]]
                break
        
        # Network latency
        network_latency = 0.0
        if client_node.node_name != server_node.node_name:
            # Use static network latency if available
            pair_key = (client_node.node_name, server_node.node_name)
            if pair_key in self.static_network_latencies:
                network_latency = self.static_network_latencies[pair_key]
            elif server_node.node_name in client_node.network_map:
                network_latency = client_node.network_map[server_node.node_name]
        
        # Queue time (use static queue length)
        queue_time = 0.0
        for platform in server_node.platforms.items:
            if platform.type["shortName"] in task.type["platforms"]:
                static_queue_length = self.static_queue_lengths.get(platform.id, 0)
                queue_time = static_queue_length * 0.1  # Assume 0.1s per queued task
                break
        
        # Cold start time
        cold_start_time = 0.0
        for platform in server_node.platforms.items:
            if platform.type["shortName"] in task.type["platforms"]:
                cold_start_time = task.type["coldStartDuration"][platform.type["shortName"]]
                break
        
        # Total execution time
        total_time = base_execution_time + network_latency + queue_time + cold_start_time
        
        return total_time

    def placement(self, system_state: SystemState, task: Task) -> Generator[Any, Any, Optional[Tuple[Node, Platform]]]:
        """Placement method for training data generation"""
        # Apply static conditions to make system state consistent
        self._apply_static_conditions(system_state)
        
        # Generate execution times for all client-server pairs
        all_execution_times = self._generate_all_execution_times(task, system_state)
        
        # Store the execution times in the task for later use in training data
        task.all_execution_times = all_execution_times
        
        # Force offloading to target server if specified
        if self.force_offloading:
            target = self._force_offload_to_target(task, system_state)
            if target:
                target_node, target_platform = target
                task.execution_node = target_node.node_name
                task.execution_platform = str(target_platform.id)
                return target
        
        # Fallback: use any available replica
        replicas: Set[Tuple[Node, Platform]] = system_state.replicas[task.type["name"]]
        if replicas:
            # Pick first available replica
            selected_replica = next(iter(replicas))
            task.execution_node = selected_replica[0].node_name
            task.execution_platform = str(selected_replica[1].id)
            return selected_replica
        
        # No replicas available
        logging.error(f"No replicas available for task {task.id}")
        return None

    def get_training_data(self) -> Dict[str, Any]:
        """Get collected training data"""
        return {
            "execution_times": dict(self.execution_times),
            "static_conditions": self.static_conditions,
            "total_pairs": len(self.execution_times)
        } 