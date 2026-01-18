"""
Knative-style Batched Scheduler with Network Awareness

This scheduler batches tasks and places them using shortest-queue (least connections)
strategy, similar to Knative's load balancing but with batching for efficiency.
"""

from __future__ import annotations

import logging
from collections import Counter
from timeit import default_timer
from typing import Generator, List, Set, Tuple, TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task

from src.placement.model import SystemState
from src.placement.scheduler import Scheduler


class KnativeNetworkScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = 5  # Batch size for efficient processing
        self.batch_timeout = 0.1  # Timeout for collecting batch

    def scheduler_process(self) -> Generator:
        if False:
            yield

        print(
            f"[ {self.env.now} ] KnativeNetwork Scheduler started with policy"
            f" {self.policy} (batch_size={self.batch_size})"
        )

        while True:
            batch_tasks = yield self.env.process(self._collect_task_batch())
            
            if not batch_tasks:
                yield self.env.timeout(0.1)
                continue

            print(f"[ {self.env.now} ] Knative: Processing batch of {len(batch_tasks)} tasks")
            yield self.env.process(self._process_task_batch(batch_tasks))

    def _collect_task_batch(self) -> Generator[Any, Any, List[Task]]:
        """
        Collect tasks into a batch with timeout to avoid blocking.
        
        Uses polling-based approach to avoid dangling get events that can
        cause tasks to be silently consumed.
        """
        batch: List[Task] = []
        
        def task_filter(queued_task):
            return all(dependency.finished for dependency in queued_task.dependencies)
        
        # First task: wait indefinitely (blocking is expected)
        task: Task = yield self.tasks.get(task_filter)
        batch.append(task)
        
        # Set deadline for collecting rest of batch
        batch_deadline = self.env.now + self.batch_timeout
        
        # Collect remaining tasks using polling (avoids dangling get events)
        while len(batch) < self.batch_size:
            remaining_time = batch_deadline - self.env.now
            if remaining_time <= 0:
                break
            
            # Check if there are any ready tasks in the queue
            ready_tasks = [t for t in self.tasks.items if task_filter(t)]
            
            if ready_tasks:
                # Get the first ready task immediately
                task = yield self.tasks.get(task_filter)
                batch.append(task)
            else:
                # No ready tasks - wait a small step and check again
                step_time = min(0.01, remaining_time)
                yield self.env.timeout(step_time)
        
        return batch

    def _ensure_replicas_for_batch(
        self, 
        batch_tasks: List[Task], 
        system_state: SystemState
    ) -> Generator[Any, Any, Dict[str, int]]:
        """
        Pre-scan batch and ensure enough replicas exist for all task types.
        Waits for initialization to complete before returning.
        """
        # Count how many tasks of each type are in the batch
        task_type_counts: Dict[str, int] = Counter(task.type["name"] for task in batch_tasks)
        replicas_created: Dict[str, int] = {}
        
        # Track newly created replicas so we can wait for initialization
        new_replicas: List[Tuple[Node, Platform]] = []
        
        for task_type_name, needed_count in task_type_counts.items():
            current_replicas = system_state.replicas.get(task_type_name, set())
            existing_replica_ids = {(n.id, p.id) for n, p in current_replicas}
            current_count = len(current_replicas)
            
            # Calculate how many additional replicas we need
            deficit = needed_count - current_count
            
            if deficit <= 0:
                replicas_created[task_type_name] = 0
                continue
            
            print(f"[ {self.env.now} ] Knative: Pre-creating {deficit} replica(s) for {task_type_name} "
                  f"(have {current_count}, need {needed_count} for batch)")
            
            # Get task type data
            task_type = self.data.task_types.get(task_type_name)
            if not task_type:
                logging.warning(f"[ {self.env.now} ] Unknown task type: {task_type_name}")
                replicas_created[task_type_name] = 0
                continue
            
            # Find source nodes from tasks of this type
            source_nodes = set(
                task.node_name for task in batch_tasks 
                if task.type["name"] == task_type_name
            )
            
            created = 0
            for _ in range(deficit):
                replica_created = False
                for source_node in source_nodes:
                    stop = yield self.env.process(
                        self.autoscaler.create_first_replica(
                            system_state,
                            task_type,
                            source_node_name=source_node
                        )
                    )
                    
                    if not isinstance(stop, StopIteration):
                        created += 1
                        replica_created = True
                        break
                
                if not replica_created:
                    stop = yield self.env.process(
                        self.autoscaler.create_first_replica(system_state, task_type)
                    )
                    if not isinstance(stop, StopIteration):
                        created += 1
                    else:
                        logging.warning(
                            f"[ {self.env.now} ] Knative: Could not create more replicas for {task_type_name} "
                            f"(created {created}/{deficit})"
                        )
                        break
            
            # Find newly created replicas
            for node, plat in system_state.replicas.get(task_type_name, set()):
                if (node.id, plat.id) not in existing_replica_ids:
                    new_replicas.append((node, plat))
            
            replicas_created[task_type_name] = created
            if created > 0:
                print(f"[ {self.env.now} ] Knative: Pre-created {created} replica(s) for {task_type_name}")
        
        # Don't block waiting for initialization - let tasks use replicas when ready
        # This allows batch processing to continue while replicas initialize in background
        if new_replicas:
            print(f"[ {self.env.now} ] Knative: Created {len(new_replicas)} replica(s), allowing initialization in background")
        
        return replicas_created

    def _process_task_batch(self, batch_tasks: List[Task]) -> Generator:
        """Process a batch using shortest-queue placement strategy."""
        print(f"[ {self.env.now} ] Knative: Processing {len(batch_tasks)} tasks in batch")
        
        # Track scheduling time for the batch
        batch_start = default_timer()
        
        system_state: SystemState = yield self.mutex.get()
        
        # Pre-create replicas and wait for initialization
        replicas_created = yield self.env.process(
            self._ensure_replicas_for_batch(batch_tasks, system_state)
        )
        if any(count > 0 for count in replicas_created.values()):
            print(f"[ {self.env.now} ] Knative: Pre-created replicas: {replicas_created}")
        
        # Capture queue snapshot for load balancing
        full_queue_snapshot = self._capture_full_queue_snapshot()
        
        # Track used platforms within this batch
        used_platforms: Set[Tuple[int, int]] = set()
        
        for task in batch_tasks:
            task_start = default_timer()
            task_replicas = system_state.replicas.get(task.type["name"], set())

            if not task_replicas:
                logging.warning(f"[ {self.env.now} ] Knative: No replicas for {task}")
                task.postponed_count += 1
                
                if task.postponed_count > 9999999:
                    self._mark_task_failed(task, "No replicas available")
                    continue
                
                yield self.tasks.put(task)
                stop = yield self.env.process(
                    self.autoscaler.create_first_replica(
                        system_state, task.type, source_node_name=task.node_name
                    )
                )
                
                # Check if replica creation succeeded
                if isinstance(stop, StopIteration):
                    logging.warning(f"[ {self.env.now} ] Knative: Failed to create replica for {task.type['name']}")
                
                # CRITICAL: Force simulation time to advance
                self.env.step()
                continue

            valid_replicas = self._get_valid_replicas(task_replicas, task)
            
            if not valid_replicas:
                task.postponed_count += 1
                
                if task.postponed_count > 9999999:
                    self._mark_task_failed(task, "No valid replicas with network connectivity")
                    continue
                
                yield self.tasks.put(task)
                stop = yield self.env.process(
                    self.autoscaler.create_first_replica(
                        system_state, task.type, source_node_name=task.node_name
                    )
                )
                
                if isinstance(stop, StopIteration):
                    logging.warning(f"[ {self.env.now} ] Knative: Failed to create replica for {task.type['name']}")
                
                # CRITICAL: Force simulation time to advance
                self.env.step()
                continue

            # Filter: unused AND initialized replicas
            available_replicas = [
                (node, plat) for node, plat in valid_replicas 
                if (node.id, plat.id) not in used_platforms
                and plat.initialized.triggered
            ]
            
            if not available_replicas:
                task.postponed_count += 1
                
                if task.postponed_count > 9999999:
                    self._mark_task_failed(task, "No unique initialized replicas available")
                    continue
                
                yield self.tasks.put(task)
                stop = yield self.env.process(
                    self.autoscaler.create_first_replica(
                        system_state, task.type, source_node_name=task.node_name
                    )
                )
                
                if isinstance(stop, StopIteration):
                    logging.warning(f"[ {self.env.now} ] Knative: Failed to create replica for {task.type['name']}")
                
                # CRITICAL: Force simulation time to advance
                self.env.step()
                continue

            # Store queue snapshot on task
            task.queue_snapshot_at_scheduling = {
                f"{node.node_name}:{plat.id}": full_queue_snapshot.get(f"{node.node_name}:{plat.id}", 0)
                for node, plat in available_replicas
            }
            task.full_queue_snapshot = full_queue_snapshot

            # Knative-style: Shortest queue (least connections) placement
            target_node, target_platform = min(
                available_replicas, key=lambda couple: len(couple[1].queue.items)
            )

            # Mark this platform as used
            used_platforms.add((target_node.id, target_platform.id))

            task.execution_node = target_node.node_name
            task.execution_platform = str(target_platform.id)
            
            # Track scheduling decision time (amortized over batch)
            task_end = default_timer()
            task.knative_decision_time = task_end - task_start

            node: Node = yield self.nodes.get(lambda node: node.id == target_node.id)
            task.node = node
            node.unused = False
            
            platform: Platform = yield node.platforms.get(lambda platform: platform.id == target_platform.id)
            task.platform = platform

            end = default_timer()
            elapsed_clock_time = end - task_start
            node.wall_clock_scheduling_time += elapsed_clock_time

            yield platform.queue.put(task)
            yield task.scheduled.succeed()

            yield node.platforms.put(platform)
            yield self.nodes.put(node)

            print(f"Knative placed task {task.id} on ({target_node.node_name}, platform {target_platform.id}) "
                  f"[queue: {len(target_platform.queue.items)}]")

        yield self.mutex.put(system_state)
        
        batch_end = default_timer()
        batch_time = (batch_end - batch_start) * 1000  # ms
        print(f"[ {self.env.now} ] Knative: Batch processing complete for {len(batch_tasks)} tasks "
              f"(scheduling time: {batch_time:.2f}ms)")

    def _mark_task_failed(self, task: Task, reason: str):
        """Mark a task as failed and trigger all events."""
        logging.error(f"[ {self.env.now} ] Task {task.id} failed: {reason}")
        task.finished = True
        task.failed = True
        task.failure_reason = reason
        if not task.scheduled.triggered:
            task.scheduled.succeed()
        if not task.arrived.triggered:
            task.arrived.succeed()
        if not task.started.triggered:
            task.started.succeed()
        if not task.done.triggered:
            task.done.succeed()

    def _capture_full_queue_snapshot(self) -> Dict[str, int]:
        queue_snapshot = {}
        for node in self.nodes.items:
            for platform in node.platforms.items:
                key = f"{node.node_name}:{platform.id}"
                queue_snapshot[key] = len(platform.queue.items)
        return queue_snapshot

    def placement(self, system_state: SystemState, task: Task) -> Generator:
        if False:
            yield
        return None

    def _get_valid_replicas(self, replicas: Set[Tuple[Node, Platform]], task: Task) -> List[Tuple[Node, Platform]]:
        """Get replicas that have network connectivity to the task's source node."""
        valid_replicas = []
        for node, platform in replicas:
            if node.node_name == task.node_name:
                valid_replicas.append((node, platform))
            elif not node.node_name.startswith('client_node'):
                if hasattr(node, 'network_map') and task.node_name in node.network_map:
                    valid_replicas.append((node, platform))
        return valid_replicas
