# GNN Scheduler Data Access Fix

## Problem

The GNN scheduler was incorrectly loading data from the `src/notebooks/data/infrastructure/` directory instead of using the proper simulation data that flows through the system. This was problematic because:

1. **Wrong Data Source**: Loading static files instead of using dynamic simulation data
2. **Hardcoded Paths**: Dependencies on specific file locations
3. **Inconsistent Data**: Not using the actual infrastructure configuration for the current simulation run
4. **Poor Architecture**: Violating the proper data flow patterns in the system

## Solution

Fixed the GNN scheduler to use the proper simulation data that flows through the system:

### 1. **Removed Hardcoded File Loading**

**Before (WRONG):**
```python
# Load static infrastructure data
infrastructure_path = "src/notebooks/data/infrastructure/infrastructure.json"
platform_types_path = "src/notebooks/data/infrastructure/platform-types.json"
task_types_path = "src/notebooks/data/infrastructure/task-types.json"

# Load static data
with open(infrastructure_path, 'r') as f:
    infrastructure_data = json.load(f)
with open(platform_types_path, 'r') as f:
    platform_types_data = json.load(f)
with open(task_types_path, 'r') as f:
    task_types_data = json.load(f)
```

**After (CORRECT):**
```python
# Get task type data from simulation data
task_type_name = task.type["name"]
if task_type_name not in self.data.task_types:
    print(f"[WRR-GNN] Task type {task_type_name} not found in task types data")
    return None

task_type_data = self.data.task_types[task_type_name]
```

### 2. **Proper Data Access Patterns**

The scheduler now uses:
- `self.data.task_types` - Task type definitions from simulation
- `self.data.platform_types` - Platform type definitions from simulation
- `self.nodes.items` - ALL nodes in the system (including those without available platforms)
- `node.network_map` - Network topology from actual Node objects

### 3. **Network Topology Access**

**Before (WRONG):**
```python
# Get network latency from infrastructure
network_latency = 0.1  # Default local latency
for node_data in infrastructure_data['nodes']:
    if node_data['node_name'] == node_i_name:
        network_map = node_data.get('network_map', {})
        network_latency = network_map.get(node_j_name, 100.0) / 1000.0
        break
```

**After (CORRECT):**
```python
# Find the actual node objects
node_i = None
for node in all_nodes:
    if node.node_name == node_i_name:
        node_i = node
        break

if node_i and node_i.network_map:
    # Check if there's a network connection between these nodes
    if node_j_name in node_i.network_map:
        # Connection exists - use actual latency
        network_latency = node_i.network_map[node_j_name]
```

### 4. **Connectivity Filtering**

Added a new method to filter replicas based on network connectivity:

```python
def _filter_replicas_by_connectivity(self, replicas: Set[Tuple[Node, Platform]], task: Task, 
                                   network_topology: Dict[str, Dict[str, float]]) -> Set[Tuple[Node, Platform]]:
    """
    Filter replicas based on network connectivity and offloading constraints.
    """
    source_node = task.node_name
    reachable_replicas = set()
    
    for node, platform in replicas:
        target_node = node.node_name
        
        # Skip if same node (local execution)
        if source_node == target_node:
            reachable_replicas.add((node, platform))
            continue
        
        # Check if there's a network path from source to target
        if source_node in network_topology and target_node in network_topology[source_node]:
            reachable_replicas.add((node, platform))
    
    return reachable_replicas
```

### 5. **Sparse Graph Construction**

Updated the GNN feature extraction to use sparse connectivity:

```python
# Create sparse network topology edges based on actual connectivity
edge_index = []
edge_features = []

for i in range(len(node_names)):
    for j in range(len(node_names)):
        if i != j:
            # Only create edge if connection exists
            if node_j_name in node_i.network_map:
                edge_index.append([i, j])
                edge_features.append([network_latency, locality_indicator])
```

## Data Flow in the System

### Proper Data Flow:
1. **Simulation Inputs** → `load_simulation_inputs()` → `SimulationData`
2. **Infrastructure Config** → `prepare_simulation_config()` → Infrastructure with network maps
3. **Node Creation** → `create_nodes()` → Node objects with `network_map`
4. **Scheduler Access** → `self.data` + `system_state` + `self.nodes`

### What the Scheduler Now Uses:
- **Task Types**: `self.data.task_types[task_type_name]`
- **Platform Types**: `self.data.platform_types[platform_name]`
- **Network Topology**: `node.network_map` from actual Node objects
- **All Nodes**: `self.nodes.items` (complete system topology)
- **Current Replicas**: `system_state.replicas[task_type_name]`

## Benefits

### 1. **Correct Data Usage**
- Uses actual simulation data instead of static files
- Respects the current infrastructure configuration
- Properly handles dynamic network topologies

### 2. **Better Architecture**
- Follows proper data flow patterns
- No hardcoded file dependencies
- Consistent with other schedulers in the system

### 3. **Network Awareness**
- Filters replicas based on actual network connectivity
- Only considers reachable nodes for offloading
- Handles sparse network topologies correctly
- **Complete topology access**: Uses all nodes, not just those with available platforms

### 4. **Improved Performance**
- Sparse graph construction for GNN
- Only creates edges for existing connections
- More efficient feature extraction

### 5. **Robust Error Handling**
- Graceful fallback when no reachable replicas exist
- Proper error messages for debugging
- Local replica creation as fallback

## Integration with Sparse Network Topology

The GNN scheduler now properly integrates with the sparse network topology system:

1. **Complete Network Topology Extraction**: Gets network maps from ALL Node objects via `self.nodes.items`
2. **Connectivity Filtering**: Only considers reachable replicas based on complete topology
3. **Sparse Graph Construction**: Only creates edges for existing connections
4. **Fallback Handling**: Creates local replicas when no reachable ones exist
5. **Full System Coverage**: Includes nodes without available platforms for complete connectivity analysis

## Testing

The fix was verified with tests that confirm:
- ✓ Proper data access patterns
- ✓ Complete network topology access (all nodes)
- ✓ Connectivity filtering with full topology
- ✓ Sparse graph construction
- ✓ Error handling
- ✓ GNN feature extraction from all nodes

## Conclusion

The GNN scheduler now properly uses simulation data instead of loading static files from the notebooks directory. This ensures:

1. **Correctness**: Uses actual simulation configuration
2. **Consistency**: Follows proper data flow patterns
3. **Flexibility**: Works with any infrastructure configuration
4. **Network Awareness**: Respects network connectivity constraints
5. **Performance**: Efficient sparse graph processing

The scheduler is now properly integrated with the sparse network topology system and will work correctly with the configurable network latency generation from `space_with_network.json`. It has access to the complete network topology from all nodes in the system, ensuring accurate connectivity filtering and GNN feature extraction. 