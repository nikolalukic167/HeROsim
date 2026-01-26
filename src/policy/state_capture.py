"""
Shared state capture utilities for all schedulers.

This module provides helper methods to capture system state, queue snapshots,
and temporal state in a format compatible with the GNN training data
(system_state_captured_unique.json).
"""

from __future__ import annotations

import json
from typing import Dict, List, Set, Tuple, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task
    from src.placement.model import SystemState


class StateCaptureHelper:
    """
    Helper class for capturing system state during simulation.
    
    Can be used by any scheduler to capture:
    - Full queue snapshot (all platforms)
    - Temporal state (remaining times for current tasks)
    - Replica state (which platforms have which task types)
    - Task placement decisions
    """
    
    def __init__(self, env, nodes):
        """
        Initialize the state capture helper.
        
        Args:
            env: SimPy environment
            nodes: FilterStore of nodes
        """
        self.env = env
        self.nodes = nodes
        self.captured_placements: List[Dict[str, Any]] = []
        self.capture_enabled = False
        self.output_path: Optional[str] = None
    
    def enable_capture(self, output_path: str):
        """Enable state capture and set output path."""
        self.capture_enabled = True
        self.output_path = output_path
        self.captured_placements = []
    
    def disable_capture(self):
        """Disable state capture."""
        self.capture_enabled = False
    
    def capture_full_queue_snapshot(self) -> Dict[str, int]:
        """
        Capture queue lengths for ALL platforms in the system.
        
        Returns:
            Dict mapping "node_name:platform_id" -> queue_length
        """
        queue_snapshot = {}
        for node in self.nodes.items:
            for platform in node.platforms.items:
                key = f"{node.node_name}:{platform.id}"
                queue_snapshot[key] = len(platform.queue.items)
        return queue_snapshot
    
    def capture_queue_snapshot_for_replicas(
        self,
        replicas: Set[Tuple['Node', 'Platform']]
    ) -> Dict[str, int]:
        """
        Capture queue lengths for a specific set of replicas.
        
        Args:
            replicas: Set of (node, platform) tuples
            
        Returns:
            Dict mapping "node_name:platform_id" -> queue_length
        """
        queue_snapshot = {}
        for node, platform in replicas:
            key = f"{node.node_name}:{platform.id}"
            queue_snapshot[key] = len(platform.queue.items)
        return queue_snapshot
    
    def capture_temporal_state_for_replicas(
        self,
        replicas: Set[Tuple['Node', 'Platform']]
    ) -> Dict[str, Dict[str, float]]:
        """
        Capture temporal state (remaining times) for a set of replicas.
        
        Args:
            replicas: Set of (node, platform) tuples
            
        Returns:
            Dict mapping "node_name:platform_id" -> {
                "current_task_remaining": float,
                "cold_start_remaining": float,
                "comm_remaining": float
            }
        """
        temporal_state = {}
        now = self.env.now
        
        for node, platform in replicas:
            key = f"{node.node_name}:{platform.id}"
            
            # Initialize with zeros
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
                    
                    # Estimate communication remaining (output write time)
                    if current_task.application:
                        state_size_map = current_task.type.get("stateSize", {})
                        app_name = current_task.application.type.get("name", "")
                        if isinstance(state_size_map, dict) and app_name in state_size_map:
                            output_size = state_size_map[app_name].get("output", 0)
                            if isinstance(output_size, (int, float)) and output_size > 0:
                                # Default storage parameters
                                throughput = 100.0 * 1024 * 1024  # 100 MB/s
                                latency = 0.001  # 1ms
                                comm_remaining = (output_size / throughput) + latency
            
            temporal_state[key] = {
                "current_task_remaining": current_task_remaining,
                "cold_start_remaining": cold_start_remaining,
                "comm_remaining": comm_remaining,
            }
        
        return temporal_state
    
    def capture_full_temporal_state(self) -> Dict[str, Dict[str, float]]:
        """
        Capture temporal state for ALL platforms in the system.
        
        Returns:
            Dict mapping "node_name:platform_id" -> temporal state dict
        """
        all_replicas = set()
        for node in self.nodes.items:
            for platform in node.platforms.items:
                all_replicas.add((node, platform))
        return self.capture_temporal_state_for_replicas(all_replicas)
    
    def capture_replicas_state(
        self,
        system_state: 'SystemState'
    ) -> Dict[str, List[List[Any]]]:
        """
        Capture current replica assignments per task type.
        
        Args:
            system_state: Current system state with replicas
            
        Returns:
            Dict mapping task_type -> list of [node_name, platform_id]
        """
        replicas_state = {}
        for task_type, replicas in system_state.replicas.items():
            replicas_state[task_type] = [
                [node.node_name, platform.id]
                for node, platform in replicas
            ]
        return replicas_state
    
    def capture_available_resources(self) -> Dict[str, List[int]]:
        """
        Capture available (idle) platform IDs per node.
        
        Returns:
            Dict mapping node_name -> list of available platform IDs
        """
        available = {}
        for node in self.nodes.items:
            idle_platforms = []
            for platform in node.platforms.items:
                # Platform is available if it has no current task and empty queue
                if platform.current_task is None and len(platform.queue.items) == 0:
                    idle_platforms.append(platform.id)
            available[node.node_name] = idle_platforms
        return available
    
    def capture_task_placement(
        self,
        task: 'Task',
        execution_node: str,
        execution_platform: str,
        elapsed_time: float,
        queue_time: float,
        queue_snapshot_at_scheduling: Dict[str, int],
        full_queue_snapshot: Dict[str, int],
        temporal_state_at_scheduling: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Capture a single task placement decision.
        
        Args:
            task: The task being placed
            execution_node: Node where task will execute
            execution_platform: Platform ID where task will execute
            elapsed_time: Wall-clock time for scheduling decision
            queue_time: Time task spent in queue before scheduling
            queue_snapshot_at_scheduling: Queue lengths for valid replicas
            full_queue_snapshot: Queue lengths for ALL platforms
            temporal_state_at_scheduling: Temporal state for valid replicas
            
        Returns:
            Dict with all placement information
        """
        placement = {
            "task_id": task.id,
            "task_type": task.type["name"],
            "source_node": task.node_name,
            "execution_node": execution_node,
            "execution_platform": execution_platform,
            "elapsed_time": elapsed_time,
            "queue_time": queue_time,
            "queue_snapshot_at_scheduling": queue_snapshot_at_scheduling,
            "full_queue_snapshot": full_queue_snapshot,
            "temporal_state_at_scheduling": temporal_state_at_scheduling,
        }
        
        if self.capture_enabled:
            self.captured_placements.append(placement)
        
        return placement
    
    def get_captured_state(
        self,
        system_state: 'SystemState',
        total_rtt: float = 0.0
    ) -> Dict[str, Any]:
        """
        Get the full captured state in the format expected by prepare_graphs_cache.py.
        
        Args:
            system_state: Current system state
            total_rtt: Total round-trip time for all tasks
            
        Returns:
            Dict with full system state capture
        """
        return {
            "timestamp": self.env.now,
            "replicas": self.capture_replicas_state(system_state),
            "available_resources": self.capture_available_resources(),
            "scheduler_state": self._capture_scheduler_state(system_state),
            "task_placements": self.captured_placements,
            "total_rtt": total_rtt,
            "num_tasks": len(self.captured_placements),
        }
    
    def _capture_scheduler_state(
        self,
        system_state: 'SystemState'
    ) -> Dict[str, Any]:
        """
        Capture scheduler-specific state (target concurrencies, contention).
        
        Args:
            system_state: Current system state
            
        Returns:
            Dict with scheduler state
        """
        scheduler_state = {
            "target_concurrencies": {},
            "average_contention": {},
            "panic_contention": {},
        }
        
        # Try to get scheduler state if it exists (HRC-style)
        if hasattr(system_state, 'scheduler_state'):
            ss = system_state.scheduler_state
            if hasattr(ss, 'target_concurrencies'):
                scheduler_state["target_concurrencies"] = dict(ss.target_concurrencies)
            if hasattr(ss, 'average_contention'):
                # Convert tuple keys to string keys
                avg_cont = {}
                for task_type, contention_dict in ss.average_contention.items():
                    avg_cont[task_type] = {}
                    for key, value in contention_dict.items():
                        if isinstance(key, tuple):
                            str_key = f"{key[0]}_{key[1]}"
                        else:
                            str_key = str(key)
                        avg_cont[task_type][str_key] = value
                scheduler_state["average_contention"] = avg_cont
            if hasattr(ss, 'panic_contention'):
                scheduler_state["panic_contention"] = dict(ss.panic_contention)
        
        return scheduler_state
    
    def save_captured_state(
        self,
        system_state: 'SystemState',
        total_rtt: float = 0.0,
        output_path: Optional[str] = None
    ):
        """
        Save captured state to a JSON file.
        
        Args:
            system_state: Current system state
            total_rtt: Total round-trip time
            output_path: Path to save file (uses self.output_path if not provided)
        """
        path = output_path or self.output_path
        if not path:
            return
        
        state = self.get_captured_state(system_state, total_rtt)
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def reset(self):
        """Reset captured placements for a new simulation run."""
        self.captured_placements = []
