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

import json
import logging
import os
import sys

from datetime import datetime
from typing import Dict, Tuple, Type, Set, Any, List, Optional

from src.placement.infrastructure import Node, Platform, Storage, Application, Task

from simpy.core import Environment  # type: ignore[import-not-found]
from simpy.resources.store import FilterStore  # type: ignore[import-not-found]

from src.placement.model import (
    DataclassJSONEncoder,
    Infrastructure,
    SimulationData,
    SimulationPolicy,
    TimeSeries,
    SimulationStats,
    ApplicationType,
    QoSType,
)

from src.placement.orchestrator import Orchestrator

from src.placement.autoscaler import Autoscaler
from src.placement.scheduler import Scheduler
from src.policy.gnn.autoscaler import GNNAutoscaler
from src.policy.gnn.orchestrator import GNNOrchestrator
from src.policy.gnn.scheduler import GNNScheduler

from src.policy.herofake.orchestrator import HROOrchestrator
from src.policy.herofake.autoscaler import HROAutoscaler
from src.policy.herofake.scheduler import HROScheduler

from src.policy.herocache.orchestrator import HRCOrchestrator
from src.policy.herocache.autoscaler import HRCAutoscaler
from src.policy.herocache.scheduler import HRCScheduler
from src.policy.heteroproactiveknative.autoscaler import HeteroProactiveKnativeAutoscaler
from src.policy.heteroproactiveknative.orchestrator import HeteroProactiveKnativeOrchestrator
from src.policy.heteroproactiveknative.scheduler import HeteroProactiveKnativeScheduler

from src.policy.knative.orchestrator import KnativeOrchestrator
from src.policy.knative.autoscaler import KnativeAutoscaler
from src.policy.knative.scheduler import KnativeScheduler
from src.policy.knative_network.orchestrator import KnativeNetworkOrchestrator as KnativeNetworkOrchestrator
from src.policy.knative_network.autoscaler import KnativeAutoscaler as KnativeNetworkAutoscaler
from src.policy.knative_network.scheduler import KnativeNetworkScheduler as KnativeNetworkScheduler
from src.policy.proactiveknative.autoscaler import ProactiveKnativeAutoscaler
from src.policy.proactiveknative.orchestrator import ProactiveKnativeOrchestrator
from src.policy.proactiveknative.scheduler import ProactiveKnativeScheduler

from src.policy.random.scheduler import RandomScheduler

from src.policy.bpff.scheduler import BPFFScheduler

from src.policy.multiloop.orchestrator import MultiLoopOrchestrator
from src.policy.multiloop.autoscaler import MultiLoopAutoscaler
from src.policy.multiloop.scheduler import MultiLoopScheduler
from src.policy.determined.orchestrator import DeterminedOrchestrator
from src.policy.determined.autoscaler import DeterminedAutoscaler
from src.policy.determined.scheduler import DeterminedScheduler
from src.policy.evaluator.orchestrator import EvaluatorOrchestrator
from src.policy.evaluator.autoscaler import EvaluatorAutoscaler
from src.policy.evaluator.scheduler import EvaluatorScheduler

from src.policy.knative_network.orchestrator import KnativeNetworkOrchestrator
from src.policy.knative_network.autoscaler import KnativeAutoscaler as KnativeNetworkAutoscaler
from src.policy.knative_network.scheduler import KnativeNetworkScheduler

from src.utils.distributions import sample_bounded_int, sample_replica_count


def create_nodes(
        env: Environment,
        simulation_data: SimulationData,
        simulation_policy: SimulationPolicy,
        infrastructure: Infrastructure,
) -> FilterStore:
    node_id = 0
    platform_id = 0
    storage_id = 0

    nodes_store = FilterStore(env)

    for node in infrastructure["nodes"]:
        platforms_store = FilterStore(env)
        storage_store = FilterStore(env)

        # Initialize node
        current_node = Node(
            env=env,
            node_id=node_id,
            memory=node["memory"],
            platforms=platforms_store,
            storage=storage_store,
            network_map=node["network_map"],
            network=infrastructure["network"],
            policy=simulation_policy,
            data=simulation_data,
            node_type=node["type"],
            node_name=node["node_name"]
        )
        nodes_store.put(current_node)

        for name in node["platforms"]:
            plat = Platform(
                env=env,
                platform_id=platform_id,
                platform_type=simulation_data.platform_types[name],
                node=current_node,
            )
            platforms_store.put(plat)
            platform_id += 1

        current_node.available_platforms = len(platforms_store.items)

        for name in node["storage"]:
            storage_store.put(
                Storage(
                    env=env,
                    storage_id=storage_id,
                    storage_type=simulation_data.storage_types[name],
                    node=current_node,
                )
            )

            storage_id += 1

        node_id += 1

    return nodes_store


def precreate_replicas(
        nodes: FilterStore,
        simulation_data: SimulationData,
        replica_plan: Dict[str, Any] | None = None,
        env: Environment | None = None,
        simulation_policy: SimulationPolicy | None = None,
        seed: int | None = None,
        deterministic_placements: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        deterministic_queues: Optional[Dict[str, Dict[str, int]]] = None
) -> Dict[str, Set[Tuple["Node", "Platform"]]]:
    """
    EXECUTE REPLICA CREATION:
    Create replicas for each task type based on the provided replica plan.
    This function focuses on execution, not decision-making.
    
    Args:
        nodes: All nodes in the simulation
        simulation_data: Task type and platform information
        replica_plan: Replica placement plan from executecosimulation.py (required)
    
    PURPOSE:
    - Executes the replica creation plan determined by executecosimulation.py
    - Ensures immediate task execution without waiting for autoscaling
    - Creates replicas according to the provided specifications
    - Creates warmup tasks for prewarmed replicas (co-simulation mode only)
    """
    print("\n=== Executing replica creation ===")
    
    # Replica plan is REQUIRED for this function (co-simulation mode only)
    # This function should NOT be called from executeinitial.py
    if not replica_plan:
        raise ValueError(
            "replica_plan is required for precreate_replicas. "
            "This function is only used by executecosimulation.py (co-simulation mode). "
            "executeinitial.py should not call this function."
        )
    
    preinit_clients = replica_plan['preinit_clients']
    preinit_servers = replica_plan['preinit_servers']
    preinit_task_types = replica_plan['preinit_task_types']
    replicas_config = replica_plan['replicas_config']
    prewarm_config = replica_plan.get('prewarm_config', {})
    print("Using replica placement plan from executecosimulation.py (co-simulation mode)")
    
    # Get all nodes and their platforms
    all_nodes = list(nodes.items)
    server_nodes = [node for node in all_nodes if not node.node_name.startswith('client_node')]
    client_nodes = [node for node in all_nodes if node.node_name.startswith('client_node')]
    
    """
    print(f"Available nodes:")
    print(f"  Server nodes: {[n.node_name for n in server_nodes]}")
    print(f"  Client nodes: {[n.node_name for n in client_nodes]}")
    print(f"Replica plan:")
    print(f"  preinit_servers: {preinit_servers}")
    print(f"  preinit_clients: {preinit_clients}")
    """
    
    # Track which platforms have been assigned to avoid double-booking
    assigned_platforms = set()
    initial_replicas = {}
    
    # Use deterministic placements if provided
    if deterministic_placements:
        print("Using deterministic replica placements from infrastructure.json")
        
        # Create node and platform lookup maps
        node_map = {node.node_name: node for node in all_nodes}
        
        for task_type_name, placements in deterministic_placements.items():
            initial_replicas[task_type_name] = set()
            
            for placement in placements:
                node_name = placement['node_name']
                platform_id = placement['platform_id']
                
                # Find node and platform
                node = node_map.get(node_name)
                if not node:
                    continue
                
                # Find platform by ID
                platform = None
                for p in node.platforms.items:
                    if p.id == platform_id:
                        platform = p
                        break
                
                if platform and (node, platform) not in assigned_platforms:
                    replica = (node, platform)
                    initial_replicas[task_type_name].add(replica)
                    assigned_platforms.add(replica)
                    
                    # Mark platform as initialized and warm
                    platform.initialized.succeed()
                    platform.previous_task = type('Task', (), {'type': {'name': task_type_name}})()
                    
                    # Use deterministic queue length if provided
                    if env and simulation_policy and deterministic_queues:
                        queue_key = f"{node_name}:{platform_id}"
                        queue_length = deterministic_queues.get(task_type_name, {}).get(queue_key, 0)
                        
                        if queue_length > 0:
                            warmup_tasks = create_warmup_tasks(
                                env, platform, task_type_name, simulation_data,
                                simulation_policy, queue_length
                            )
                            for warmup_task in warmup_tasks:
                                platform.queue.put(warmup_task)
                            print(f"    Enqueued {len(warmup_tasks)} warmup tasks to {node_name}:{platform_id}")
        
        print(f"\n=== Replica creation complete (deterministic) ===")
        for task_type, replicas in initial_replicas.items():
            print(f"{task_type}: {len(replicas)} replicas")
        print(f"Total unique platforms assigned: {len(assigned_platforms)}")
        
        return initial_replicas
    
    # Legacy mode: use random sampling
    # Use seed for deterministic RNG (same seed as network topology)
    import random
    if seed is not None:
        rng = random.Random(seed)
        print(f"Using seeded RNG (seed={seed}) for replica and queue distributions")
    else:
        rng = random.Random()
        print("Warning: No seed provided for RNG - replica and queue distributions will be non-deterministic")
    
    # Create replicas for each task type according to configuration
    for task_type_name, replica_config in replicas_config.items():
        print(f"\nTask type: {task_type_name}")
        print(f"  Replica config: {replica_config}")
        
        # Get supported platforms for this task type
        task_type = simulation_data.task_types[task_type_name]
        supported_platforms = task_type["platforms"]
        print(f"  Supported platforms: {supported_platforms}")
        
        # Initialize replica set for this task type
        initial_replicas[task_type_name] = set()
        
        # Create server replicas
        per_server = replica_config.get('per_server', 0)
        if per_server > 0:
            print(f"  Creating {per_server} replicas per server")
            
            for node in server_nodes:
                if node.node_name in preinit_servers:
                    # Allow statistical override per node (if configured)
                    task_prewarm_cfg = prewarm_config.get(task_type_name, {}) if prewarm_config else {}
                    per_node_target = per_server
                    if task_prewarm_cfg.get('distribution') == 'statistical':
                        rep_dist = task_prewarm_cfg.get('replica_distribution') or {}
                        sampled = sample_replica_count('server', rep_dist, rng)
                        # preserve at least 0, and don't exceed number of suitable platforms
                        per_node_target = max(0, int(sampled))
                    # Find suitable unassigned platforms on this server
                    suitable_platforms = [
                        platform for platform in node.platforms.items
                        if (platform.type["shortName"] in supported_platforms and 
                            (node, platform) not in assigned_platforms)
                    ]
                    
                    # Create up to per_server replicas on this node
                    replicas_created = 0
                    for platform in suitable_platforms:
                        if replicas_created >= per_node_target:
                            break
                        
                        # Create replica
                        replica = (node, platform)
                        initial_replicas[task_type_name].add(replica)
                        assigned_platforms.add(replica)
                        
                        # Mark platform as initialized and warm for this task type
                        platform.initialized.succeed()
                        platform.previous_task = type('Task', (), {'type': {'name': task_type_name}})()
                        
                        # Create warmup tasks if configured and environment/policy available
                        if env and simulation_policy and prewarm_config:
                            task_prewarm = prewarm_config.get(task_type_name, {})
                            initial_queue = task_prewarm.get('initial_queue', 0)
                            # Statistical queue distribution support
                            if task_prewarm.get('queue_distribution') == 'statistical':
                                q_params = task_prewarm.get('queue_distribution_params') or {}
                                # default clamp: non-negative small cap to avoid huge queues
                                if 'min' not in q_params:
                                    q_params['min'] = 0
                                sampled_q = sample_bounded_int(q_params, rng)
                                initial_queue = max(0, int(sampled_q))
                            if initial_queue > 0:
                                warmup_tasks = create_warmup_tasks(
                                    env, platform, task_type_name, simulation_data, 
                                    simulation_policy, initial_queue
                                )
                                # Enqueue warmup tasks to the platform
                                for warmup_task in warmup_tasks:
                                    platform.queue.put(warmup_task)
                                print(f"    Enqueued {len(warmup_tasks)} warmup tasks to {node.node_name}:{platform.id}")
                        
                        # print(f"    Created replica on {node.node_name} ({platform.type['shortName']}) - Platform {platform.id}")
                        replicas_created += 1
        
        # Create client replicas (if requested)
        per_client = replica_config.get('per_client', 0)
        if per_client > 0:
            print(f"  Creating {per_client} replicas per client")
            
            for node in client_nodes:
                if node.node_name in preinit_clients:
                    # Allow statistical override per node (if configured)
                    task_prewarm_cfg = prewarm_config.get(task_type_name, {}) if prewarm_config else {}
                    per_node_target = per_client
                    if task_prewarm_cfg.get('distribution') == 'statistical':
                        rep_dist = task_prewarm_cfg.get('replica_distribution') or {}
                        sampled = sample_replica_count('client', rep_dist, rng)
                        per_node_target = max(0, int(sampled))
                    # Find suitable unassigned platforms on this client
                    suitable_platforms = [
                        platform for platform in node.platforms.items
                        if (platform.type["shortName"] in supported_platforms and 
                            (node, platform) not in assigned_platforms)
                    ]
                    
                    # Create up to per_client replicas on this node
                    replicas_created = 0
                    for platform in suitable_platforms:
                        if replicas_created >= per_node_target:
                            break
                        
                        # Create replica
                        replica = (node, platform)
                        initial_replicas[task_type_name].add(replica)
                        assigned_platforms.add(replica)
                        
                        # Mark platform as initialized and warm for this task type
                        platform.initialized.succeed()
                        platform.previous_task = type('Task', (), {'type': {'name': task_type_name}})()
                        
                        # Create warmup tasks if configured and environment/policy available
                        if env and simulation_policy and prewarm_config:
                            task_prewarm = prewarm_config.get(task_type_name, {})
                            initial_queue = task_prewarm.get('initial_queue', 0)
                            if task_prewarm.get('queue_distribution') == 'statistical':
                                q_params = task_prewarm.get('queue_distribution_params') or {}
                                if 'min' not in q_params:
                                    q_params['min'] = 0
                                sampled_q = sample_bounded_int(q_params, rng)
                                initial_queue = max(0, int(sampled_q))
                            if initial_queue > 0:
                                warmup_tasks = create_warmup_tasks(
                                    env, platform, task_type_name, simulation_data, 
                                    simulation_policy, initial_queue
                                )
                                # Enqueue warmup tasks to the platform
                                for warmup_task in warmup_tasks:
                                    platform.queue.put(warmup_task)
                                print(f"    Enqueued {len(warmup_tasks)} warmup tasks to {node.node_name}:{platform.id}")
                        
                        # print(f"    Created replica on {node.node_name} ({platform.type['shortName']}) - Platform {platform.id}")
                        replicas_created += 1
        
        # print(f"  Total replicas created: {len(initial_replicas[task_type_name])}")
    
    print(f"\n=== Replica creation complete ===")
    for task_type, replicas in initial_replicas.items():
        print(f"{task_type}: {len(replicas)} replicas")
        # for replica in replicas:
        #     node, platform = replica
        #     print(f"  - {node.node_name}:{platform.id} ({platform.type['shortName']})")
    print(f"Total unique platforms assigned: {len(assigned_platforms)}")
    
    return initial_replicas


def create_warmup_tasks(
        env: Environment,
        platform: Platform,
        task_type_name: str,
        simulation_data: SimulationData,
        simulation_policy: SimulationPolicy,
        count: int
) -> List[Task]:
    """
    Create warmup tasks for a platform to prefill its queue.
    
    Args:
        env: SimPy environment
        platform: Platform to create warmup tasks for
        task_type_name: Name of the task type
        simulation_data: Simulation data containing task types, application types, QoS types
        simulation_policy: Simulation policy
        count: Number of warmup tasks to create
    
    Returns:
        List of created warmup tasks (marked with is_internal=True)
    """
    if count <= 0:
        return []
    
    warmup_tasks = []
    task_type = simulation_data.task_types[task_type_name]
    
    # Find an application type that uses this task type
    application_type = None
    for app_type in simulation_data.application_types.values():
        if task_type_name in app_type.get('dag', {}):
            application_type = app_type
            break
    
    if not application_type:
        # Fallback to first application type if none found
        application_type = list(simulation_data.application_types.values())[0]
    
    # Use medium QoS as default
    qos_type = simulation_data.qos_types.get('medium', list(simulation_data.qos_types.values())[0])
    
    for i in range(count):
        # Create a lightweight application for the warmup task
        warmup_app = Application(
            id=-1000 - i,  # Negative ID to distinguish from real applications
            dispatched_time=0.0,
            application_type=application_type,
            qos_type=qos_type,
            tasks=[]  # Will be set after task creation
        )
        
        # Create the warmup task
        warmup_task = Task(
            env=env,
            task_id=-1000 - i,  # Negative ID to distinguish from real tasks
            task_type=task_type,
            application=warmup_app,
            dependencies=[],
            policy=simulation_policy,
            node_name=platform.node.node_name
        )
        
        # Mark as internal warmup task
        setattr(warmup_task, 'is_internal', True)
        
        # Set up the task for execution
        warmup_task.node = platform.node
        warmup_task.platform = platform
        
        # Trigger events to allow task_process to advance
        warmup_task.dispatched.succeed()
        warmup_task.scheduled.succeed()
        
        # Add to application's task list
        warmup_app.tasks = [warmup_task]
        
        # Attach to platform for orchestrator to discover later
        if not hasattr(platform, '_warmup_tasks'):
            setattr(platform, '_warmup_tasks', [])  # type: ignore[attr-defined]
        platform._warmup_tasks.append(warmup_task)  # type: ignore[attr-defined]

        warmup_tasks.append(warmup_task)
    
    return warmup_tasks


def start_simulation(
        simulation_data: SimulationData,
        simulation_policy: SimulationPolicy,
        infrastructure: Infrastructure,
        time_series: TimeSeries,
        trace_file: str,
        models = None
) -> SimulationStats | None:
    # Logger
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.ERROR)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s [%(funcName)18s() ] %(message)s",
        handlers=[console_handler],
    )

    logger = logging.getLogger('simulation')

    # Simulation
    env = Environment()
    finished = env.event()

    # Initialize infrastructure
    nodes: FilterStore = create_nodes(
        env=env,
        simulation_data=simulation_data,
        simulation_policy=simulation_policy,
        infrastructure=infrastructure,
    )
    print(f"[simulation] Created {len(nodes.items)} nodes for simulation '{simulation_policy.scheduling}'")

    # Pre-create replicas for each task type based on configuration
    # NOTE: This is ONLY used by executecosimulation.py (co-simulation mode)
    # executeinitial.py does NOT provide replica_plan and should not preinitialize platforms
    initial_replicas = {}
    if infrastructure.get("preinitialize_platforms", False):
        # Replica plan is only provided by executecosimulation.py (co-simulation mode)
        replica_plan = infrastructure.get('replica_plan')
        if replica_plan is not None:
            # Get deterministic placements and queues if available
            deterministic_placements = infrastructure.get('deterministic_replica_placements')
            deterministic_queues = infrastructure.get('deterministic_queue_distributions')
            
            # Get seed from infrastructure (same seed used for network topology)
            seed = None
            network_config = infrastructure.get('network', {})
            topology_config = network_config.get('topology', {})
            if topology_config and 'seed' in topology_config:
                seed = topology_config['seed']
            
            # Only create replicas and warmup tasks when replica_plan exists (co-sim mode)
            initial_replicas = precreate_replicas(
                nodes, simulation_data, replica_plan, env, simulation_policy,
                seed=seed,
                deterministic_placements=deterministic_placements,
                deterministic_queues=deterministic_queues
            )
        else:
            # preinitialize_platforms=True but no replica_plan - skip replica creation
            # This should not happen, but handle gracefully for executeinitial.py mode
            print("Warning: preinitialize_platforms=True but no replica_plan provided. Skipping replica precreation.")

    policies: Dict[
        str, Tuple[Type[Orchestrator], Type[Autoscaler], Type[Scheduler]]
    ] = {
        "hro_hro": (HROOrchestrator, HROAutoscaler, HROScheduler),
        "hro_hrc": (HROOrchestrator, HROAutoscaler, HRCScheduler),
        "hro_kn": (HROOrchestrator, HROAutoscaler, KnativeScheduler),
        "hro_rp": (HROOrchestrator, HROAutoscaler, RandomScheduler),
        "hro_bpff": (HROOrchestrator, HROAutoscaler, BPFFScheduler),
        "hrc_hrc": (HRCOrchestrator, HRCAutoscaler, HRCScheduler),
        "hrc_hro": (HRCOrchestrator, HRCAutoscaler, HROScheduler),
        "hrc_kn": (HRCOrchestrator, HRCAutoscaler, KnativeScheduler),
        "hrc_rp": (HRCOrchestrator, HRCAutoscaler, RandomScheduler),
        "hrc_bpff": (HRCOrchestrator, HRCAutoscaler, BPFFScheduler),
        "kn_kn": (KnativeOrchestrator, KnativeAutoscaler, KnativeScheduler),
        "kn_hro": (KnativeOrchestrator, KnativeAutoscaler, HROScheduler),
        "kn_hrc": (KnativeOrchestrator, KnativeAutoscaler, HRCScheduler),
        "kn_rp": (KnativeOrchestrator, KnativeAutoscaler, RandomScheduler),
        "kn_bpff": (KnativeOrchestrator, KnativeAutoscaler, BPFFScheduler),
        "prokn_prokn": (ProactiveKnativeOrchestrator, ProactiveKnativeAutoscaler, ProactiveKnativeScheduler),
        "prohetkn_prohetkn": (HeteroProactiveKnativeOrchestrator, HeteroProactiveKnativeAutoscaler, HeteroProactiveKnativeScheduler),
        "gnn_gnn": (GNNOrchestrator, GNNAutoscaler, GNNScheduler),
        "multiloop_multiloop": (MultiLoopOrchestrator, MultiLoopAutoscaler, MultiLoopScheduler),
        "determined_determined": (DeterminedOrchestrator, DeterminedAutoscaler, DeterminedScheduler),
        "evaluator_evaluator": (EvaluatorOrchestrator, EvaluatorAutoscaler, EvaluatorScheduler),
        "kn_network_kn_network": (KnativeNetworkOrchestrator, KnativeNetworkAutoscaler, KnativeNetworkScheduler),
    }

    # Retrieve relevant Autoscaler and Scheduler classes
    # Both will be instantiated by the Orchestrator
    orchestrator_type, autoscaler_type, scheduler_type = policies[
        simulation_policy.scheduling
    ]

    # Prepare orchestrator arguments
    orchestrator_args = {
        'env': env,
        'data': simulation_data,
        'policy': simulation_policy,
        'autoscaler': autoscaler_type,
        'scheduler': scheduler_type,
        'time_series': time_series,
        'nodes': nodes,
        'end_event': finished,
        'trace_file': str(trace_file),
        'models': models,
        'initial_replicas': initial_replicas  # Pass initial replicas to orchestrator
    }
    
    # Add infrastructure config for determined orchestrator
    if orchestrator_type.__name__ == 'DeterminedOrchestrator':
        orchestrator_args['infrastructure'] = infrastructure
    
    orchestrator = orchestrator_type(**orchestrator_args)
    env.run(until=finished)
    logging.info(f"[ {orchestrator.end_time} ] ✨ Simulation finished")

    # Statistics
    stats = orchestrator.stats()

    logger.info("start_simulation: Simulation completed")
    return stats
