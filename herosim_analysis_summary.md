# Herosim Physical Node and Platform Utilization Analysis Summary

## Overview
This analysis examined how the herosim simulation framework models physical node utilization and resource management in edge computing environments. The focus was on understanding CPU/GPU utilization, storage fullness, memory management, and how these metrics could be encoded as feature vectors for GNN training.

## Key Findings

### 1. Platform Utilization Modeling

**Queue-based Execution Model:**
- Each platform has a `Store(env)` queue for pending tasks
- FIFO (First-In-First-Out) task selection
- Tracks current executing task (`current_task`)
- Calculates idle time: `idle_time = env.now - load_time`
- **CPU utilization = 100 - idle_proportion**

**Platform Types Supported:**
- `rpiCpu`: Raspberry Pi 4 Broadcom BCM2711 (CPU)
- `xavierCpu`: Nvidia Xavier Carmel Arm v8.2 (CPU)
- `xavierGpu`: Nvidia Xavier Volta (GPU)
- `xavierDla`: Nvidia Xavier DLA (Deep Learning Accelerator)
- `pynqFpga`: Xilinx Artix-7 (FPGA)

### 2. Resource Sharing Architecture

**Memory Sharing:**
- All platforms on a node share the same memory pool
- No per-platform memory allocation
- Node-level memory capacity (e.g., 8GB for rpi, 32GB for xavier)
- Memory requirements checked at task execution time

**Storage Sharing:**
- Multiple storage devices per node (e.g., eMMC, Flash Card, Remote Storage)
- All platforms share the same storage devices
- Storage selection based on dependency locality (local vs remote)
- Storage types have different characteristics:
  - **eMMC**: 32GB capacity, 294 MB/s read, 82 MB/s write
  - **Flash Card**: 64GB capacity, 100 MB/s read, 40 MB/s write
  - **Remote Storage**: Unlimited capacity, 108 MB/s read/write

### 3. Utilization Tracking

**What's Currently Tracked:**
- Platform idle proportion (inverse of CPU utilization)
- Storage usage percentage
- Energy consumption (idle and active)
- Queue lengths (in code but not exposed in results)
- Memory requirements per task type

**What's Missing for Real-time GNN Training:**
- Timestamp-based snapshots
- Real-time utilization percentages
- Platform queue length in result objects
- Memory utilization as percentage

### 4. Feature Vectors for GNN Training

**Physical Node Feature Vector (25 features):**
```python
node_features = [
    # Memory characteristics (2)
    node_memory, node_memory_tier,
    
    # Platform counts (5)
    rpi_cpu_count, xavier_cpu_count, xavier_gpu_count, xavier_dla_count, pynq_fpga_count,
    
    # Task execution performance (6)
    min_execution_time, max_execution_time, avg_execution_time,
    min_cold_start, max_cold_start, avg_cold_start,
    
    # I/O characteristics (4)
    task_input_size, task_output_size, storage_read_speed, storage_write_speed,
    
    # Memory compatibility (2)
    min_memory_required, max_memory_required,
    
    # Categorical encodings (6)
    task_type_dnn1, task_type_dnn2, is_rpi_node, is_xavier_node, is_pynq_node, is_client_node
]
```

**Platform Feature Vector (15 features):**
```python
platform_features = [
    # Platform characteristics (4)
    platform_id, platform_type_encoding, hardware_type_encoding, idle_energy,
    
    # Current utilization (4)
    queue_length, has_current_task, idle_proportion, load_time,
    
    # Performance metrics (4)
    tasks_count, local_dependencies_ratio, cache_hits, storage_time,
    
    # Task-specific performance (3)
    task_execution_time, task_cold_start_duration, task_memory_requirement
]
```

### 5. Task Execution Characteristics

**Task Types:**
- `dnn1`, `dnn2`: Deep neural network tasks
- Each task type has platform-specific characteristics:
  - Memory requirements
  - Execution times
  - Cold start durations
  - Energy consumption
  - I/O sizes (input/output)

**Execution Flow:**
1. Task arrives at platform queue
2. Platform processes task (FIFO)
3. Cold start if needed
4. Input data retrieval from storage
5. Task execution
6. Output data storage
7. Platform becomes idle

### 6. Resource Contention Modeling

**Storage Contention:**
- Multiple platforms compete for storage devices
- Storage selection based on dependency locality
- Cache eviction when storage is full
- I/O operations block until storage is available

**Memory Contention:**
- Shared memory pool across all platforms
- Memory requirements checked before task execution
- No explicit memory allocation/deallocation

**Network Effects:**
- Network latency affects data transfer
- Bandwidth limitations impact I/O performance
- Remote vs local storage selection

### 7. Recommendations for GNN Training

**Data Collection Modifications:**
1. Add real-time snapshot collection
2. Expose queue lengths in platform results
3. Calculate memory utilization percentages
4. Store snapshots at regular intervals

**Feature Engineering:**
1. Focus on node-level resource utilization
2. Model resource contention patterns
3. Consider storage hierarchy (local vs remote)
4. Include network topology information

**Training Considerations:**
1. Use node-level features for resource sharing
2. Model platform competition for shared resources
3. Consider temporal aspects of utilization
4. Include energy efficiency metrics

## Code Structure Insights

**Key Classes:**
- `Platform`: Manages individual execution resources
- `Node`: Contains multiple platforms and shared resources
- `Storage`: Manages storage devices and caching
- `Task`: Represents computational workloads

**Key Methods:**
- `Platform.platform_process()`: Main execution loop
- `Platform.result()`: Utilization statistics
- `Storage.get_usage()`: Storage utilization calculation
- `Node.result()`: Aggregated node statistics

## Limitations and Gaps

1. **No real-time monitoring**: Utilization tracked only at simulation end
2. **Missing queue metrics**: Queue lengths not exposed in results
3. **No memory utilization**: Memory usage not tracked as percentage
4. **Limited temporal data**: No timestamp-based snapshots
5. **Simplified resource allocation**: No explicit resource reservation

## GNN-Based Scheduling Analysis

### Current GNN Implementation Issues

**Problematic Graph Structure:**
- **Artificial edges**: Fully connected graph with no meaningful relationships
- **Poor performance**: R² ≈ -0.17 (worse than predicting mean)
- **No message passing**: Edges don't represent real system interactions
- **Arbitrary connectivity**: Every node connects to every other node

**Example of Current Problematic Approach:**
```python
# Current implementation - PROBLEMATIC
for i in range(len(PHYSICAL_NODES)):
    for j in range(len(PHYSICAL_NODES)):
        if i != j:
            edge_index.append([i, j])  # Artificial connection
            edge_attr.append([network_latency, locality_indicator])
```

### Storage-Platform Relationship Analysis

**Critical Storage Influences on Task Execution:**

1. **Function Caching Impact:**
   - **Cached functions**: Execute immediately (warm start)
   - **Non-cached functions**: Require cold start with significant delay
   - **Storage capacity**: Limits which functions can be cached
   - **Eviction policies**: Affect cache hit rates

2. **Storage Performance Characteristics:**
   - **eMMC**: 294 MB/s read, 82 MB/s write, 4.52ms latency
   - **FlashCard**: 100 MB/s read, 40 MB/s write, 1.22ms latency
   - **Remote Storage**: 108 MB/s read/write, 15ms latency

3. **Data Locality Effects:**
   - **Local storage**: Fast access, no network overhead
   - **Remote storage**: Network latency + storage latency
   - **Local dependencies**: Tasks can read from previous task's output directly

4. **Storage Contention:**
   - **High usage**: Slower I/O performance, increased latency
   - **Write amplification**: Multiple platforms writing simultaneously
   - **Cache pressure**: Limited space forces frequent evictions

### Recommended Graph Structure for GNN

**Meaningful Platform Relationships:**

1. **Resource Contention Edges:**
```python
def create_contention_edges(platforms: List[Platform]) -> Tuple[List, List]:
    """Create edges only between platforms that actually compete for resources"""
    edge_index = []
    edge_features = []
    
    for i, platform1 in enumerate(platforms):
        for j, platform2 in enumerate(platforms):
            if i != j and should_create_contention_edge(platform1, platform2):
                edge_index.append([i, j])
                edge_features.append([
                    calculate_memory_contention(platform1, platform2),
                    calculate_hardware_contention(platform1, platform2),
                    calculate_storage_contention(platform1, platform2),
                    calculate_thermal_interference(platform1, platform2)
                ])
    
    return edge_index, edge_features
```

2. **Storage Access Edges:**
```python
def create_storage_edges(platforms: List[Platform], storages: List[Storage]) -> Tuple[List, List]:
    """Create edges between platforms and storage devices they access"""
    edge_index = []
    edge_features = []
    
    for i, platform in enumerate(platforms):
        for j, storage in enumerate(storages):
            if platform.node.id == storage.node.id:  # Platform can access this storage
                edge_index.append([i, j])
                edge_features.append([
                    storage.type['throughput']['read'],
                    storage.type['latency']['read'],
                    storage.get_usage(),
                    calculate_cache_hit_probability(platform, storage)
                ])
    
    return edge_index, edge_features
```

3. **Hierarchical Edges:**
```python
def create_hierarchy_edges(platforms: List[Platform]) -> Tuple[List, List]:
    """Create edges within hardware type groups"""
    # Group platforms by hardware type
    hardware_groups = defaultdict(list)
    for i, platform in enumerate(platforms):
        hardware_groups[platform.type['hardware']].append(i)
    
    # Create edges within same hardware type
    for hardware_type, platform_indices in hardware_groups.items():
        for i in platform_indices:
            for j in platform_indices:
                if i != j:
                    edge_index.append([i, j])
                    edge_features.append([
                        1.0,  # Same hardware type
                        get_hardware_priority(hardware_type),
                        0.0,  # Not cross-hardware
                        1.0   # Hierarchical connection
                    ])
```

### Graph Prediction vs Node Prediction

**Recommended Approach: Graph Prediction**

**Scenario 1: Full Topology Graph (NOT RECOMMENDED)**
- **Input**: All nodes (clients + servers) with network connections
- **Output**: E2E time for each client→server path
- **Problems**: Limited meaningful relationships, heterogeneous nodes, sparse connections

**Scenario 2: Client-Centric Server Graph (RECOMMENDED)**
- **Input**: All platforms across all server nodes with resource contention edges
- **Output**: Best platform(s) for the task
- **Advantages**: Meaningful relationships, effective message passing, homogeneous nodes

**Implementation Structure:**
```python
def create_server_infrastructure_graph(all_server_nodes: List[Node]) -> Data:
    """Create a single graph with all server platforms as nodes"""
    
    # Collect all platforms from all server nodes
    all_platforms = []
    for node in all_server_nodes:
        all_platforms.extend(node.platforms.items)
    
    # Create meaningful edges
    edge_index = []
    edge_features = []
    
    # 1. Resource contention edges (only between competing platforms)
    contention_edges, contention_features = create_contention_edges(all_platforms)
    edge_index.extend(contention_edges)
    edge_features.extend(contention_features)
    
    # 2. Storage access edges (platforms to storage)
    storage_edges, storage_features = create_storage_edges(all_platforms, get_all_storage(all_server_nodes))
    edge_index.extend(storage_edges)
    edge_features.extend(storage_features)
    
    # 3. Hierarchical edges (within hardware types)
    hierarchy_edges, hierarchy_features = create_hierarchy_edges(all_platforms)
    edge_index.extend(hierarchy_edges)
    edge_features.extend(hierarchy_features)
    
    return Data(
        x=torch.tensor(extract_platform_features(all_platforms)),
        edge_index=torch.tensor(edge_index).t(),
        edge_attr=torch.tensor(edge_features)
    )
```

### Effective Message Passing Scenarios

**Real Relationships for Message Passing:**

1. **Memory Contention:**
```python
# Platform A → Platform B: "I'm using X% of memory, you should consider this"
def calculate_memory_contention(platform1: Platform, platform2: Platform) -> float:
    if platform1.node.id != platform2.node.id:
        return 0.0
    
    memory_usage1 = get_platform_memory_usage(platform1)
    memory_usage2 = get_platform_memory_usage(platform2)
    total_memory = platform1.node.memory
    
    return (memory_usage1 + memory_usage2) / total_memory
```

2. **Storage I/O Contention:**
```python
# Platform A → Platform B: "I'm doing heavy I/O, expect delays"
def calculate_storage_contention(platform1: Platform, platform2: Platform) -> float:
    shared_storage = get_shared_storage(platform1, platform2)
    if not shared_storage:
        return 0.0
    
    io_load1 = get_platform_io_load(platform1)
    io_load2 = get_platform_io_load(platform2)
    
    return (io_load1 + io_load2) / shared_storage.type['iops']['read']
```

3. **Thermal Interference:**
```python
# Platform A → Platform B: "I'm running hot, you might get throttled"
def calculate_thermal_interference(platform1: Platform, platform2: Platform) -> float:
    if platform1.node.id != platform2.node.id:
        return 0.0
    
    temp1 = get_platform_temperature(platform1)
    temp2 = get_platform_temperature(platform2)
    
    return max(0, (temp1 + temp2 - 70) / 30)  # Normalize around 70°C
```

### Key Insights for GNN Implementation

1. **Only create edges between platforms that have meaningful relationships:**
   - Competing for same resources (memory, storage, thermal budget)
   - Sharing infrastructure (same node, same storage device)
   - Similar characteristics (same hardware type, similar performance)

2. **Message passing should represent real system interactions:**
   - Resource contention patterns
   - Load balancing strategies
   - Cache locality effects
   - Thermal interference
   - Storage I/O bottlenecks

3. **Avoid artificial relationships:**
   - Don't connect unrelated platforms
   - Don't create fully connected graphs
   - Don't ignore the actual system architecture

## Storage and Memory as GNN Nodes

### Why Storage and Memory Should Be Nodes

**Benefits of Including Storage/Memory as Nodes:**

1. **Explicit Resource Modeling:**
   - Storage devices become first-class citizens in the graph
   - Memory pools are explicitly represented
   - Resource contention is naturally modeled through edges

2. **Better Message Passing:**
   - Platforms can directly "communicate" with storage devices
   - Memory usage patterns are learned through graph convolutions
   - Storage performance impacts are captured in node embeddings

3. **More Realistic Architecture:**
   - Reflects actual system topology
   - Storage and memory are physical resources that platforms access
   - Network effects between resources are naturally captured

### Heterogeneous Graph Structure

**Node Types in the GNN:**

1. **Platform Nodes:**
   - Execution resources (CPU, GPU, DLA, FPGA)
   - Features: utilization, queue length, hardware type, energy consumption

2. **Storage Nodes:**
   - Storage devices (eMMC, Flash Card, Remote Storage)
   - Features: capacity, usage percentage, throughput, latency, cache hit rate

3. **Memory Nodes:**
   - Memory pools per node
   - Features: total capacity, used memory, available memory, memory pressure

**Edge Types:**

1. **Platform → Storage Edges:**
   - Access relationships
   - Features: access frequency, I/O load, cache hit probability

2. **Platform → Memory Edges:**
   - Memory allocation relationships
   - Features: memory usage, allocation pressure

3. **Platform → Platform Edges:**
   - Resource contention relationships
   - Features: shared resource usage, interference patterns

### Implementation Example

```python
def create_heterogeneous_infrastructure_graph(all_nodes: List[Node]) -> HeteroData:
    """Create a heterogeneous graph with platforms, storage, and memory as nodes"""
    
    # Collect all entities
    platforms = []
    storages = []
    memory_pools = []
    
    for node in all_nodes:
        platforms.extend(node.platforms.items)
        storages.extend(node.storage.items)
        memory_pools.append({
            'node_id': node.id,
            'capacity': node.memory,
            'used': calculate_used_memory(node),
            'available': node.memory - calculate_used_memory(node)
        })
    
    # Create node features
    platform_features = extract_platform_features(platforms)
    storage_features = extract_storage_features(storages)
    memory_features = extract_memory_features(memory_pools)
    
    # Create edge indices and features
    edge_dict = {}
    
    # Platform → Storage edges
    platform_storage_edges = []
    platform_storage_features = []
    for i, platform in enumerate(platforms):
        for j, storage in enumerate(storages):
            if platform.node.id == storage.node.id:
                platform_storage_edges.append([i, j])
                platform_storage_features.append([
                    storage.type['throughput']['read'],
                    storage.type['latency']['read'],
                    storage.get_usage(),
                    calculate_cache_hit_probability(platform, storage)
                ])
    
    edge_dict[('platform', 'accesses', 'storage')] = (
        torch.tensor(platform_storage_edges).t(),
        torch.tensor(platform_storage_features)
    )
    
    # Platform → Memory edges
    platform_memory_edges = []
    platform_memory_features = []
    for i, platform in enumerate(platforms):
        for j, memory in enumerate(memory_pools):
            if platform.node.id == memory['node_id']:
                platform_memory_edges.append([i, j])
                platform_memory_features.append([
                    get_platform_memory_usage(platform),
                    memory['used'] / memory['capacity'],
                    memory['available'] / memory['capacity']
                ])
    
    edge_dict[('platform', 'uses', 'memory')] = (
        torch.tensor(platform_memory_edges).t(),
        torch.tensor(platform_memory_features)
    )
    
    # Platform → Platform edges (resource contention)
    platform_platform_edges = []
    platform_platform_features = []
    for i, platform1 in enumerate(platforms):
        for j, platform2 in enumerate(platforms):
            if i != j and platform1.node.id == platform2.node.id:
                platform_platform_edges.append([i, j])
                platform_platform_features.append([
                    calculate_memory_contention(platform1, platform2),
                    calculate_storage_contention(platform1, platform2),
                    calculate_thermal_interference(platform1, platform2)
                ])
    
    edge_dict[('platform', 'contends', 'platform')] = (
        torch.tensor(platform_platform_edges).t(),
        torch.tensor(platform_platform_features)
    )
    
    return HeteroData(
        x_dict={
            'platform': torch.tensor(platform_features),
            'storage': torch.tensor(storage_features),
            'memory': torch.tensor(memory_features)
        },
        edge_index_dict=edge_dict
    )
```

### Storage Node Features

```python
def extract_storage_features(storages: List[Storage]) -> List[List[float]]:
    """Extract features for storage nodes"""
    features = []
    
    for storage in storages:
        storage_features = [
            # Capacity and usage
            storage.type['capacity'],                    # Total capacity in GB
            storage.get_usage() * 100,                   # Usage percentage
            storage.get_cache_volume(),                  # Cache volume used
            storage.get_data_volume(),                   # Data volume used
            
            # Performance characteristics
            storage.type['throughput']['read'],          # Read throughput MB/s
            storage.type['throughput']['write'],         # Write throughput MB/s
            storage.type['latency']['read'],             # Read latency in seconds
            storage.type['latency']['write'],            # Write latency in seconds
            
            # I/O statistics
            storage.writes,                              # Total bytes written
            storage.erases,                              # Total bytes erased
            len(storage.functions_cache),                # Number of cached functions
            len(storage.data_store),                     # Number of stored data items
            
            # Categorical features
            1.0 if storage.type['remote'] else 0.0,      # Is remote storage
            1.0 if storage.type['hardware'] == 'mmc' else 0.0,  # Is eMMC
            1.0 if storage.type['hardware'] == 'flash' else 0.0,  # Is Flash
            1.0 if storage.type['hardware'] == 'ssd' else 0.0,   # Is SSD
        ]
        features.append(storage_features)
    
    return features
```

### Memory Node Features

```python
def extract_memory_features(memory_pools: List[Dict]) -> List[List[float]]:
    """Extract features for memory nodes"""
    features = []
    
    for memory in memory_pools:
        memory_features = [
            # Capacity and usage
            memory['capacity'],                          # Total capacity in GB
            memory['used'],                              # Used memory in GB
            memory['available'],                         # Available memory in GB
            memory['used'] / memory['capacity'],         # Usage ratio
            
            # Memory pressure indicators
            max(0, (memory['used'] / memory['capacity']) - 0.8),  # High usage penalty
            memory['available'] / memory['capacity'],    # Available ratio
            
            # Memory tier classification
            1.0 if memory['capacity'] <= 1 else 0.0,     # Low memory tier
            1.0 if 1 < memory['capacity'] <= 8 else 0.0, # Medium memory tier
            1.0 if memory['capacity'] > 8 else 0.0,      # High memory tier
        ]
        features.append(memory_features)
    
    return features
```

### Benefits of This Approach

1. **Natural Resource Modeling:**
   - Storage and memory become explicit entities
   - Resource contention is modeled through graph structure
   - Performance bottlenecks are naturally captured

2. **Better Predictions:**
   - GNN can learn storage access patterns
   - Memory pressure effects are captured
   - Cache locality is modeled through edges

3. **Scalable Architecture:**
   - Easy to add new resource types
   - Heterogeneous graph handles different node types
   - Message passing between different resource types

### Message Passing Between Resource Types

```python
# Storage → Platform: "I'm at 85% capacity, expect slower I/O"
# Memory → Platform: "I'm under pressure, allocation might fail"
# Platform → Storage: "I need to write 100MB, expect delays"
# Platform → Memory: "I need 2GB, check availability"
```

This approach creates a much more realistic and effective GNN for edge computing scheduling, where resource contention and performance bottlenecks are critical factors in decision-making.

## Conclusion

Herosim provides a comprehensive simulation of edge computing environments with realistic resource sharing and contention modeling. While it tracks utilization metrics, modifications are needed to support real-time GNN training with the specific format requested ("75% CPU utilization at timestamp T"). The shared resource model accurately reflects real-world edge computing scenarios where hardware resources are shared across multiple execution units.

**For GNN-based scheduling, the key insight is that meaningful graph structures require real relationships between platforms, not artificial connections. The recommended approach is to model all server platforms as nodes in a single graph with edges representing resource contention, storage sharing, and hardware hierarchy relationships.**

**Including storage and memory as explicit nodes in the GNN creates a more realistic and effective model that naturally captures resource contention patterns and performance bottlenecks through the graph structure itself.** 