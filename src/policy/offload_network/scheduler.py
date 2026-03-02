"""
Network-aware offload scheduler.

This scheduler behaves like knative_network but *never* executes tasks
on the client node where they arrive. Instead, it always offloads to
remote server nodes that are reachable in the network topology, and
selects among them using the same shortest-queue logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Set, Tuple

from src.policy.knative_network.scheduler import KnativeScheduler

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task


class OffloadNetworkScheduler(KnativeScheduler):
    """
    Network-aware offload scheduler:
    - Never executes on the source (client) node.
    - Only uses remote server nodes (node names not starting with 'client_node').
    - Still respects network connectivity (source->target in network_map).
    - Among valid replicas, uses the same shortest-queue logic as knative_network.
    """

    def _get_valid_replicas(
        self,
        replicas: Set[Tuple["Node", "Platform"]],
        task: "Task",
    ) -> List[Tuple["Node", "Platform"]]:
        valid_replicas: List[Tuple["Node", "Platform"]] = []

        # Find source node to check its network_map
        source_node = None
        for n in self.nodes.items:
            if n.node_name == task.node_name:
                source_node = n
                break

        for node, platform in replicas:
            # Skip local placement: we never execute on the source node
            if node.node_name == task.node_name:
                continue

            # Only offload to server nodes (exclude client_node*)
            if node.node_name.startswith("client_node"):
                continue

            # Remote placement: check network connectivity (same logic as knative_network)
            if source_node is not None and hasattr(source_node, "network_map"):
                if node.node_name in source_node.network_map:
                    valid_replicas.append((node, platform))
            elif hasattr(node, "network_map") and task.node_name in node.network_map:
                valid_replicas.append((node, platform))

        return valid_replicas

