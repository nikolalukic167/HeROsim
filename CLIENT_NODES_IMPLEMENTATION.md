# Client Nodes Implementation

This document describes the implementation of client node functionality to prevent network connections and offloading between client nodes.

## Overview

Client nodes are distinguished from server nodes by their naming convention and have special restrictions:
- **Client nodes**: Named with `client_node` prefix (e.g., `client_node0`, `client_node1`)
- **Server nodes**: Named with `node` prefix (e.g., `node0`, `node1`)
- **Restriction**: Client nodes cannot have network connections to other client nodes
- **Allowed**: Client nodes can connect to server nodes, and server nodes can connect to each other

## Implementation Details

### 1. Node Naming in `src/generator/traces.py`

**File**: `src/generator/traces.py`
**Function**: `generate_time_series()`

**Changes**:
- Modified node name assignment to use `client_node` prefix instead of `node`
- Client nodes are now named: `client_node0`, `client_node1`, ..., `client_node{n_clients-1}`

```python
# Before
node_name = f"node{client_ids[i] % n_clients}"

# After  
node_name = f"client_node{client_ids[i] % n_clients}"
```

### 2. Network Topology Generation in `src/executeinitial.py`

**File**: `src/executeinitial.py`
**Function**: `generate_network_latencies()`

**Changes**:
- Added logic to prevent network connections between client nodes
- Client nodes can still connect to server nodes and vice versa

```python
# Prevent connections between client nodes
if node_name.startswith('client_node') and other_node_name.startswith('client_node'):
    can_connect = False
```

### 3. Scheduler Filtering in `src/policy/gnn/scheduler.py`

**File**: `src/policy/gnn/scheduler.py`
**Functions**: 
- `_is_client_node()` - Helper function to detect client nodes
- `_filter_replicas_by_connectivity()` - Filter replicas based on connectivity rules

**Changes**:
- Added `_is_client_node()` helper function
- Modified replica filtering to prevent offloading between client nodes
- Added client node flag to GNN feature extraction (25 features instead of 24)

```python
def _is_client_node(self, node_name: str) -> bool:
    """Check if a node is a client node based on its name."""
    return node_name.startswith('client_node')

# In _filter_replicas_by_connectivity:
if self._is_client_node(source_node) and self._is_client_node(target_node):
    print(f"[GNN] Skipping replica on {target_node} - client nodes cannot offload to other client nodes")
    continue
```

### 4. GNN Feature Extraction

**Changes**:
- Added client node flag as the 25th feature in node feature extraction
- Updated feature count from 24 to 25 features
- Updated model configuration to handle 25 features

```python
# Node type encoding (4 features) - including client node flag
is_client_node = self._is_client_node(node_name)
node_type_encoding = [
    1 if 'rpi' in node_name.lower() else 0,
    1 if 'xavier' in node_name.lower() or any('xavier' in p for p in platform_counts.keys()) else 0,
    1 if 'pynq' in node_name.lower() else 0,
    1 if is_client_node else 0  # Client node flag
]
```

## Network Topology Rules

1. **Client → Client**: ❌ **BLOCKED** - No network connections allowed
2. **Client → Server**: ✅ **ALLOWED** - Client nodes can connect to server nodes
3. **Server → Client**: ✅ **ALLOWED** - Server nodes can connect to client nodes  
4. **Server → Server**: ✅ **ALLOWED** - Server nodes can connect to each other

## Offloading Rules

1. **Client → Client**: ❌ **BLOCKED** - Tasks cannot be offloaded between client nodes
2. **Client → Server**: ✅ **ALLOWED** - Tasks can be offloaded from client to server nodes
3. **Server → Client**: ✅ **ALLOWED** - Tasks can be offloaded from server to client nodes
4. **Server → Server**: ✅ **ALLOWED** - Tasks can be offloaded between server nodes

## Configuration

The number of client nodes is controlled by the `n_clients` parameter in `generate_time_series()`:
- Default: 30 client nodes
- Client nodes: `client_node0` to `client_node29`
- Server nodes: `node30` and beyond (depending on cluster size)

## Testing

The implementation has been tested to verify:
- ✅ Client nodes are properly marked with `client_node` prefix
- ✅ Network connections between client nodes are prevented
- ✅ Client nodes can connect to server nodes
- ✅ Scheduler correctly filters out client-to-client offloading
- ✅ GNN feature extraction includes client node flag

## Impact

This implementation ensures that:
1. **Security**: Client nodes cannot communicate directly with each other
2. **Resource Isolation**: Client workloads are isolated from each other
3. **Proper Offloading**: Tasks can only be offloaded to appropriate server nodes
4. **GNN Awareness**: The GNN model is aware of client vs server node distinctions 