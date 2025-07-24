# SystemState Structure in GNN Scheduler

## Overview

The `SystemState` is the main data structure passed to the GNN scheduler's `placement()` method. It contains all the information about the current state of the distributed system.

## Structure Definition

```python
@dataclass
class SystemState:
    scheduler_state: SchedulerState
    available_resources: Dict["Node", Set["Platform"]]
    replicas: Dict[str, Set[Tuple["Node", "Platform"]]]
```

## Detailed Breakdown

### 1. scheduler_state: SchedulerState
```
SchedulerState {
    target_concurrencies: Dict[str, PlatformVector]
}
```

**Purpose**: Contains scheduler-specific state information
- `target_concurrencies`: Maps task types to target concurrency levels for each platform type
- Example: `{'dnn1': {'rpiCpu': 2, 'xavierCpu': 3, 'xavierGpu': 1}}`

### 2. available_resources: Dict[Node, Set[Platform]]
```
Node1 → {Platform1, Platform2, Platform3}
Node2 → {Platform4, Platform5}
client_node_1 → {Platform6}
```

**Purpose**: Shows which platforms are currently free/available on each node
- Key: Node object
- Value: Set of available Platform objects
- Used to determine immediate resource availability

### 3. replicas: Dict[str, Set[Tuple[Node, Platform]]]
```
'dnn1' → {
    (Node1, Platform1),
    (Node1, Platform2),
    (Node2, Platform3)
}
'dnn2' → {
    (Node2, Platform3),
    (client_node_1, Platform6)
}
```

**Purpose**: Maps task types to all available (Node, Platform) pairs that can execute that task type
- Key: Task type name (string)
- Value: Set of (Node, Platform) tuples
- This is the primary data used by the GNN scheduler for placement decisions

## Data Flow in GNN Scheduler

```
Task Arrives
    ↓
1. Get replicas for task type
   replicas[task.type["name"]]
    ↓
2. Filter by network connectivity
   _filter_replicas_by_connectivity()
    ↓
3. Extract features for each physical node
   _extract_physical_node_features_from_system()
    ↓
4. Run GNN inference
   gnn_model(x, edge_index, edge_attr)
    ↓
5. Convert scores to weights
   _update_gnn_weights()
    ↓
6. Weighted Round-Robin selection
   _select_replica_weighted_round_robin()
    ↓
7. Place task on selected replica
```

## Key Components Used by GNN

### Node Structure
```python
Node {
    node_name: str                    # e.g., "node_1", "client_node_1"
    memory: SizeGigabyte             # Available memory
    platforms: List[Platform]        # All platforms on this node
    network_map: Dict[str, SpeedMBps] # Network connectivity to other nodes
    storage: List[Storage]           # Storage resources
}
```

### Platform Structure
```python
Platform {
    id: int                          # Unique platform ID
    type: PlatformType              # Platform type (rpiCpu, xavierGpu, etc.)
    queue: Queue[Task]              # Current task queue
    current_task: Optional[Task]    # Currently executing task
    energy_consumption: EnergykWh   # Energy usage
}
```

### TaskType Structure
```python
TaskType {
    name: str                        # Task type name (e.g., "dnn1", "dnn2")
    platforms: List[str]            # Compatible platform types
    memoryRequirements: PlatformVector
    executionTime: PlatformVector
    coldStartDuration: PlatformVector
    energy: PlatformVector
    stateSize: Dict[str, IOVector]
}
```

## Example SystemState Instance

```python
system_state = SystemState(
    scheduler_state=SchedulerState(
        target_concurrencies={
            'dnn1': {'rpiCpu': 2, 'xavierCpu': 3, 'xavierGpu': 1},
            'dnn2': {'rpiCpu': 1, 'xavierCpu': 2, 'xavierGpu': 2}
        }
    ),
    available_resources={
        node_1: {platform_1, platform_2},
        node_2: {platform_3},
        client_node_1: {platform_4}
    },
    replicas={
        'dnn1': {
            (node_1, platform_1),
            (node_1, platform_2),
            (node_2, platform_3)
        },
        'dnn2': {
            (node_2, platform_3),
            (client_node_1, platform_4)
        }
    }
)
```

## GNN Feature Extraction Process

The GNN scheduler extracts features from the SystemState as follows:

1. **Get all physical nodes**: `self.nodes.items` (all nodes in system)
2. **For each node, extract 25 features**:
   - Memory characteristics (2 features)
   - Platform counts (5 features)
   - Task execution performance (6 features)
   - I/O characteristics (4 features)
   - Memory compatibility (2 features)
   - Categorical encodings (6 features)

3. **Create sparse network topology**:
   - Use `node.network_map` to determine connectivity
   - Create edges only between connected nodes
   - Edge features: [network_latency, locality_indicator]

## Key Methods in scheduler.py

- `placement()`: Main entry point for task placement
- `_filter_replicas_by_connectivity()`: Filters replicas based on network topology
- `_extract_wrr_gnn_features()`: Extracts features for GNN inference
- `_update_gnn_weights()`: Updates replica weights using GNN predictions
- `_select_replica_weighted_round_robin()`: Final replica selection

## Important Insights

1. **replicas** contains all available (Node, Platform) pairs for each task type
2. **available_resources** shows which platforms are currently free
3. **GNN operates on physical nodes**, not individual platforms
4. **Network topology** comes from `node.network_map` for connectivity filtering
5. **Client nodes** have special handling (no client-to-client offloading)
6. **Feature extraction** happens at the node level, not platform level

## Serialization

The SystemState has a `result()` method that serializes it to a dictionary format:

```python
def result(self, timestamp: MomentSecond = 0.0) -> SystemStateResult:
    return {
        "timestamp": timestamp,
        "scheduler_state": scheduler_state_dict,
        "available_resources": {
            node.node_name: [platform.id for platform in platforms]
            for node, platforms in self.available_resources.items()
        },
        "replicas": {
            task_type: [
                [node.node_name, platform.id]
                for node, platform in replica_set
            ]
            for task_type, replica_set in self.replicas.items()
        }
    }
```

This serialized format is used for logging and analysis purposes. 