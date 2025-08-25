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
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Any
from enum import Enum

from src.placement.model import SystemState, SchedulerState


class TaskStatus(Enum):
    """Task execution status for microstate tracking"""
    COLD = "cold"
    WARM = "warm"
    RUNNING = "running"
    QUEUED = "queued"


@dataclass
class PlatformMicroState:
    """Complete microstate of a platform for deterministic simulation"""
    # Current task state
    current_task_id: Optional[int]
    current_task_remaining_time: float
    current_task_type: Optional[str]
    
    # Queue state (ordered list with full task information)
    queue_tasks: List[Tuple[int, str, TaskStatus, float]]  # (task_id, task_type, status, estimated_time)
    
    # Warm start state
    previous_task_type: Optional[str]
    is_initialized: bool
    idle_since: float
    
    # Resource contention
    queue_length: int
    memory_usage_percent: float
    storage_usage_percent: float
    
    # Cache state
    function_cache: Set[str]  # Set of cached function types
    last_cache_update: float
    
    # Autoscaler state
    last_allocated: Optional[float]
    last_removed: Optional[float]
    
    def __post_init__(self):
        """Ensure queue_length matches actual queue size"""
        self.queue_length = len(self.queue_tasks)


@dataclass
class StorageMicroState:
    """Complete microstate of storage for deterministic simulation"""
    # Usage metrics
    usage_percentage: float
    cache_volume_gb: float
    data_volume_gb: float
    
    # Cache content (FIFO order matters for eviction)
    functions_cache: List[str]  # Ordered list of cached functions
    data_store: Dict[str, Any]  # Data content
    
    # I/O state
    concurrent_io_operations: int
    io_queue: List[Tuple[str, str, float]]  # (operation_type, task_id, start_time)
    
    # Network state
    network_transfers: List[Tuple[str, str, float, float]]  # (source, dest, start_time, remaining_bytes)


@dataclass
class NodeMicroState:
    """Complete microstate of a node for deterministic simulation"""
    # Memory state
    available_memory_gb: float
    total_memory_gb: float
    memory_usage_percent: float
    
    # Network topology (static)
    network_map: Dict[str, float]  # node_name -> latency
    
    # In-flight transfers
    in_flight_transfers: List[Tuple[str, str, float, float]]  # (source, dest, start_time, remaining_bytes)
    
    # Local dependencies
    local_dependencies_count: int
    cache_hits_count: int
    
    # Resource contention
    cpu_usage_percent: float
    io_wait_percent: float


@dataclass
class TrainingSystemState(SystemState):
    """Enhanced system state for GNN training with frozen dynamic variables"""
    
    # Original system state components
    scheduler_state: SchedulerState
    available_resources: Dict["Node", Set["Platform"]]
    replicas: Dict[str, Set[Tuple["Node", "Platform"]]]
    
    # Training-specific frozen microstates
    platform_microstates: Dict[int, PlatformMicroState]  # platform_id -> microstate
    storage_microstates: Dict[int, StorageMicroState]    # storage_id -> microstate
    node_microstates: Dict[int, NodeMicroState]          # node_id -> microstate
    
    # Global simulation state
    simulation_time: float
    random_seed: int
    autoscaler_disabled: bool
    
    # Task batch information
    current_batch: List["Task"]
    batch_timestamp: float
    
    def get_frozen_features(self) -> Dict[str, Any]:
        """Extract frozen features for GNN training"""
        features = {
            "platform_states": {},
            "storage_states": {},
            "node_states": {},
            "global_state": {
                "simulation_time": self.simulation_time,
                "random_seed": self.random_seed,
                "batch_size": len(self.current_batch)
            }
        }
        
        # Extract platform features
        for platform_id, microstate in self.platform_microstates.items():
            features["platform_states"][platform_id] = {
                "queue_length": microstate.queue_length,
                "memory_usage": microstate.memory_usage_percent,
                "storage_usage": microstate.storage_usage_percent,
                "is_initialized": microstate.is_initialized,
                "idle_since": microstate.idle_since,
                "cached_functions_count": len(microstate.function_cache),
                "current_task_type": microstate.current_task_type,
                "previous_task_type": microstate.previous_task_type
            }
        
        # Extract storage features
        for storage_id, microstate in self.storage_microstates.items():
            features["storage_states"][storage_id] = {
                "usage_percentage": microstate.usage_percentage,
                "cache_volume": microstate.cache_volume_gb,
                "data_volume": microstate.data_volume_gb,
                "concurrent_io": microstate.concurrent_io_operations,
                "cached_functions_count": len(microstate.functions_cache)
            }
        
        # Extract node features
        for node_id, microstate in self.node_microstates.items():
            features["node_states"][node_id] = {
                "available_memory": microstate.available_memory_gb,
                "memory_usage_percent": microstate.memory_usage_percent,
                "cpu_usage": microstate.cpu_usage_percent,
                "io_wait": microstate.io_wait_percent,
                "local_dependencies": microstate.local_dependencies_count,
                "cache_hits": microstate.cache_hits_count
            }
        
        return features
    
    def is_deterministic(self) -> bool:
        """Check if this state is sufficient for deterministic simulation"""
        # All critical microstate components must be present
        if not self.platform_microstates or not self.storage_microstates or not self.node_microstates:
            return False
        
        # Check that all platforms have complete microstate
        for platform_id, microstate in self.platform_microstates.items():
            if (microstate.current_task_id is None and microstate.queue_tasks) or \
               (microstate.current_task_id is not None and not microstate.is_initialized):
                return False
        
        return True
    
    def clone_for_training(self) -> "TrainingSystemState":
        """Create a deep copy for training iterations"""
        # This would need to be implemented based on your deep copy requirements
        # For now, return a reference (you'll need to implement proper cloning)
        return self


@dataclass
class TrainingTask:
    """Task representation for training data generation"""
    task_id: int
    task_type: str
    client_node: str
    memory_requirements: float
    storage_requirements: float
    dependencies: List[int]
    
    # Precomputed execution characteristics
    execution_time_per_platform: Dict[int, float]  # platform_id -> execution_time
    cold_start_time_per_platform: Dict[int, float]  # platform_id -> cold_start_time
    communication_time_per_platform: Dict[int, float]  # platform_id -> communication_time
    
    def get_total_time_for_platform(self, platform_id: int, include_network: bool = True) -> float:
        """Calculate total RTT for a specific platform"""
        total_time = (
            self.execution_time_per_platform.get(platform_id, 0.0) +
            self.cold_start_time_per_platform.get(platform_id, 0.0) +
            self.communication_time_per_platform.get(platform_id, 0.0)
        )
        return total_time


@dataclass
class TrainingBatch:
    """Batch of tasks for training data generation"""
    batch_id: str
    timestamp: float
    tasks: List[TrainingTask]
    system_state: TrainingSystemState
    
    def get_batch_features(self) -> Dict[str, Any]:
        """Extract features for the entire batch"""
        return {
            "batch_size": len(self.tasks),
            "task_types": [task.task_type for task in self.tasks],
            "client_nodes": list(set(task.client_node for task in self.tasks)),
            "system_state": self.system_state.get_frozen_features()
        }

