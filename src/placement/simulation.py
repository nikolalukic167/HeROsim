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
from typing import Dict, Tuple, Type, Set, Any, List

from src.placement.infrastructure import Node, Platform, Storage

from simpy.core import Environment
from simpy.resources.store import FilterStore

from src.placement.model import (
    DataclassJSONEncoder,
    Infrastructure,
    SimulationData,
    SimulationPolicy,
    TimeSeries,
    SimulationStats,
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
            platforms_store.put(
                Platform(
                    env=env,
                    platform_id=platform_id,
                    platform_type=simulation_data.platform_types[name],
                    node=current_node,
                )
            )

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
        replica_plan: Dict[str, Any] | None = None
) -> Dict[str, Set[Tuple["Node", "Platform"]]]:
    """
    EXECUTE REPLICA CREATION:
    Create replicas for each task type based on the provided replica plan.
    This function focuses on execution, not decision-making.
    
    Args:
        nodes: All nodes in the simulation
        simulation_data: Task type and platform information
        replica_plan: Replica placement plan from executecosimulation.py (preferred)
    
    PURPOSE:
    - Executes the replica creation plan determined by executecosimulation.py
    - Ensures immediate task execution without waiting for autoscaling
    - Creates replicas according to the provided specifications
    """
    print("\n=== Executing replica creation ===")
    
    # Use replica_plan if provided, otherwise fall back to infrastructure config
    if replica_plan:
        preinit_clients = replica_plan['preinit_clients']
        preinit_servers = replica_plan['preinit_servers']
        preinit_task_types = replica_plan['preinit_task_types']
        replicas_config = replica_plan['replicas_config']
        print("Using replica placement plan from executecosimulation.py")
    else:
        print("this should not happen in simulation.py")
        sys.exit(1)
    
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
                    # Find suitable unassigned platforms on this server
                    suitable_platforms = [
                        platform for platform in node.platforms.items
                        if (platform.type["shortName"] in supported_platforms and 
                            (node, platform) not in assigned_platforms)
                    ]
                    
                    # Create up to per_server replicas on this node
                    replicas_created = 0
                    for platform in suitable_platforms:
                        if replicas_created >= per_server:
                            break
                        
                        # Create replica
                        replica = (node, platform)
                        initial_replicas[task_type_name].add(replica)
                        assigned_platforms.add(replica)
                        
                        # Mark platform as initialized and warm for this task type
                        platform.initialized.succeed()
                        platform.previous_task = type('Task', (), {'type': {'name': task_type_name}})()
                        
                        # print(f"    Created replica on {node.node_name} ({platform.type['shortName']}) - Platform {platform.id}")
                        replicas_created += 1
        
        # Create client replicas (if requested)
        per_client = replica_config.get('per_client', 0)
        if per_client > 0:
            print(f"  Creating {per_client} replicas per client")
            
            for node in client_nodes:
                if node.node_name in preinit_clients:
                    # Find suitable unassigned platforms on this client
                    suitable_platforms = [
                        platform for platform in node.platforms.items
                        if (platform.type["shortName"] in supported_platforms and 
                            (node, platform) not in assigned_platforms)
                    ]
                    
                    # Create up to per_client replicas on this node
                    replicas_created = 0
                    for platform in suitable_platforms:
                        if replicas_created >= per_client:
                            break
                        
                        # Create replica
                        replica = (node, platform)
                        initial_replicas[task_type_name].add(replica)
                        assigned_platforms.add(replica)
                        
                        # Mark platform as initialized and warm for this task type
                        platform.initialized.succeed()
                        platform.previous_task = type('Task', (), {'type': {'name': task_type_name}})()
                        
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

    # Pre-create replicas for each task type based on configuration
    initial_replicas = {}
    if infrastructure.get("preinitialize_platforms", False):
        # Accept optional replica plan embedded by executeinitial.py
        replica_plan = infrastructure.get('replica_plan')
        initial_replicas = precreate_replicas(nodes, simulation_data, replica_plan)

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
        "determined_determined": (DeterminedOrchestrator, DeterminedAutoscaler, DeterminedScheduler)
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

    # with open(os.path.join("result", f"{simulation_time}.json"), "w") as outfile:
    #     json.dump(stats, outfile, indent=2, cls=DataclassJSONEncoder)

    # logging.warning(f"Total time: {env.now / 3600} hours")
    # logging.warning(f"Average elapsed time: {stats['averageElapsedTime']}")
    # logging.warning(f"Average compute time: {stats['averageComputeTime']}")
    # logging.warning(f"Total energy: {stats['energy']}")
    # logging.warning(f"Penalty proportion: {stats['penaltyProportion']}")
    # logging.warning(f"Cold start proportion: {stats['coldStartProportion']}")
    return stats
