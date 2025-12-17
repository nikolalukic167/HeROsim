# Determinism Analysis: Pipeline Non-Determinism Sources and Solution

## Current Pipeline Flow

```
generate_gnn_datasets.sh
  ↓ (for each dataset)
  ├─ Generates config with network seed, replica config, queue distribution params
  ├─ Calls executecosimulation.py
  │   ↓
  │   ├─ prepare_simulation_config() → generate_network_latencies()
  │   │   └─ Uses random.random() with seed (but seed may be consumed differently)
  │   │
  │   ├─ determine_replica_placement() → Creates replica plan
  │   │
  │   ├─ generate_brute_force_placement_combinations() → Creates placement plans
  │   │
  │   └─ For each placement plan:
  │       └─ process_sample_with_placement()
  │           └─ Calls simulation.py:start_simulation()
  │               ├─ create_nodes()
  │               ├─ precreate_replicas()
  │               │   ├─ Uses random.Random() → sample_replica_count()
  │               │   └─ Uses random.Random() → sample_bounded_int() for queues
  │               └─ Orchestrator runs simulation
```

## Non-Determinism Sources Identified

### 1. Network Topology Generation (executecosimulation.py:81-226)
**Location**: `generate_network_latencies()`
- **Issue**: Uses `random.random()` to decide connections and `random.uniform()` for latencies
- **Seed**: Seed is set from config, but:
  - If called multiple times, RNG state changes
  - Different simulation runs may consume seed differently
- **Impact**: Different network topologies across runs

### 2. Replica Placement (simulation.py:236-243, 298-303)
**Location**: `precreate_replicas()` → `sample_replica_count()`
- **Issue**: Creates `rng = random.Random()` without seed
- **Impact**: Different replica counts per node across runs
- **Evidence**: Analysis shows negative task counts vary (161-214)

### 3. Queue Distribution (simulation.py:270-276)
**Location**: `precreate_replicas()` → `sample_bounded_int()`
- **Issue**: Creates `rng = random.Random()` without seed
- **Impact**: Different queue lengths per platform across runs
- **Evidence**: Analysis shows warmup task times vary significantly

### 4. Warmup Task Execution Order
**Location**: SimPy event queue
- **Issue**: Even with same queue lengths, execution order can vary
- **Impact**: Different arrival/done times for warmup tasks

## Solution: Pre-Generate Deterministic Infrastructure

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Phase 1: Infrastructure Generation (ONCE per dataset) │
│  ┌───────────────────────────────────────────────────┐  │
│  │ generate_infrastructure.py                        │  │
│  │  - Generate network topology (with seed)          │  │
│  │  - Generate replica placements (with seed)       │  │
│  │  - Generate queue distributions (with seed)      │  │
│  │  - Save to infrastructure.json                   │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  Phase 2: Simulation Execution (MANY times per dataset) │
│  ┌───────────────────────────────────────────────────┐  │
│  │ executecosimulation.py                            │  │
│  │  - Load infrastructure.json                       │  │
│  │  - Use pre-generated network, replicas, queues   │  │
│  │  - Run simulations with different placements     │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Implementation Plan

#### Step 1: Create Infrastructure Generator Script

**File**: `src/generate_infrastructure.py`

```python
"""
Generate deterministic infrastructure configuration for a dataset.

This script pre-generates:
1. Network topology (connections and latencies)
2. Replica placements (which platforms get replicas)
3. Queue distributions (how many warmup tasks per platform)

All randomness is seeded and saved to infrastructure.json
"""

def generate_deterministic_infrastructure(
    config_file: str,
    output_file: str,
    seed: int
) -> Dict:
    """
    Generate deterministic infrastructure.
    
    Returns:
        {
            "network_maps": {node_name: {other_node: latency}},
            "replica_placements": {
                task_type: [
                    {"node_name": str, "platform_id": int, "replica_id": int}
                ]
            },
            "queue_distributions": {
                task_type: {
                    "node_name:platform_id": queue_length
                }
            },
            "metadata": {
                "seed": seed,
                "config_file": config_file,
                "generation_time": timestamp
            }
        }
    """
    import random
    rng = random.Random(seed)
    
    # 1. Generate network topology
    network_maps = generate_network_topology(config, rng)
    
    # 2. Generate replica placements
    replica_placements = generate_replica_placements(config, rng)
    
    # 3. Generate queue distributions
    queue_distributions = generate_queue_distributions(config, rng)
    
    return {
        "network_maps": network_maps,
        "replica_placements": replica_placements,
        "queue_distributions": queue_distributions,
        "metadata": {...}
    }
```

#### Step 2: Modify executecosimulation.py

**Changes**:
1. Check if `infrastructure.json` exists in dataset directory
2. If exists, load it and use pre-generated values
3. If not, generate it first (or fail with error)

```python
def prepare_simulation_config(
    sample: np.ndarray,
    mapping: Dict[int, str],
    original_config: Dict[str, Any],
    placement_plan: Optional[Dict[int, Tuple[int, int]]] = None,
    replica_plan: Optional[Dict[str, Any]] = None,
    base_nodes: Optional[List[Dict[str, Any]]] = None,
    infrastructure_file: Optional[Path] = None  # NEW
) -> Dict[str, Any]:
    """
    Prepare simulation configuration.
    
    If infrastructure_file is provided, load pre-generated infrastructure.
    Otherwise, generate on-the-fly (legacy mode).
    """
    if infrastructure_file and infrastructure_file.exists():
        # Load pre-generated infrastructure
        with open(infrastructure_file, 'r') as f:
            infra_data = json.load(f)
        
        # Use pre-generated network maps
        network_maps = infra_data['network_maps']
        
        # Use pre-generated replica placements
        replica_placements = infra_data['replica_placements']
        
        # Use pre-generated queue distributions
        queue_distributions = infra_data['queue_distributions']
        
        # Build nodes with pre-generated network maps
        nodes = build_nodes_from_infrastructure(original_config, network_maps)
        
        # Override replica_plan with pre-generated placements
        replica_plan = build_replica_plan_from_infrastructure(
            replica_placements, queue_distributions
        )
    else:
        # Legacy mode: generate on-the-fly
        # ... existing code ...
```

#### Step 3: Modify simulation.py

**Changes**:
1. Accept pre-generated replica plan with exact placements
2. Accept pre-generated queue distributions
3. Use deterministic values instead of random sampling

```python
def precreate_replicas(
    nodes: FilterStore,
    simulation_data: SimulationData,
    replica_plan: Dict[str, Any] | None = None,
    env: Environment | None = None,
    simulation_policy: SimulationPolicy | None = None,
    deterministic_placements: Optional[Dict] = None,  # NEW
    deterministic_queues: Optional[Dict] = None  # NEW
) -> Dict[str, Set[Tuple["Node", "Platform"]]]:
    """
    Create replicas using deterministic placements if provided.
    """
    if deterministic_placements:
        # Use pre-generated replica placements
        for task_type, placements in deterministic_placements.items():
            for placement in placements:
                node_name = placement['node_name']
                platform_id = placement['platform_id']
                # Find node and platform
                node = find_node(nodes, node_name)
                platform = find_platform(node, platform_id)
                # Create replica deterministically
                ...
    else:
        # Legacy mode: use random sampling
        # ... existing code ...
    
    if deterministic_queues:
        # Use pre-generated queue lengths
        for task_type, queues in deterministic_queues.items():
            for platform_key, queue_length in queues.items():
                node_name, platform_id = parse_platform_key(platform_key)
                # Set queue length deterministically
                ...
    else:
        # Legacy mode: use random sampling
        # ... existing code ...
```

#### Step 4: Modify generate_gnn_datasets.sh

**Changes**:
1. Add infrastructure generation step before calling executecosimulation.py
2. Pass infrastructure file path to executecosimulation.py

```bash
# Before running executecosimulation.py
INFRA_FILE="${OUT_DIR}/infrastructure.json"
if [ ! -f "${INFRA_FILE}" ]; then
    echo "[${DID}] Generating deterministic infrastructure..."
    python -m src.generate_infrastructure \
        --config "${TMP_CFG}" \
        --output "${INFRA_FILE}" \
        --seed "${seed}"
fi

# Pass infrastructure file to executecosimulation.py
python -m src.executecosimulation --brute-force \
    --infrastructure "${INFRA_FILE}" \
    > '${TMP_LOG}' 2>&1
```

## Benefits

1. **Determinism**: All simulation runs use identical infrastructure
2. **Reproducibility**: Same seed → same infrastructure → same results
3. **Performance**: Generate once, reuse many times
4. **Debugging**: Can inspect infrastructure.json to understand configuration
5. **Validation**: Can verify infrastructure is consistent across runs

## Migration Path

1. **Phase 1**: Implement infrastructure generator (new script)
2. **Phase 2**: Modify executecosimulation.py to accept infrastructure file
3. **Phase 3**: Modify simulation.py to use deterministic values
4. **Phase 4**: Update generate_gnn_datasets.sh to generate infrastructure first
5. **Phase 5**: Test with small dataset to verify determinism
6. **Phase 6**: Re-run full dataset generation

## Verification

After implementation, verify determinism:

```bash
# Run comparison script
python scripts_analysis/compare_task_systemstates.py ds_00000 --limit 50

# Expected results:
# - Negative task counts: identical across all files
# - Negative task times: identical across all files
# - SystemStateResult: identical across all files
```

## Key Files to Modify

1. **NEW**: `src/generate_infrastructure.py` - Infrastructure generator
2. **MODIFY**: `src/executecosimulation.py` - Load infrastructure file
3. **MODIFY**: `src/placement/simulation.py` - Use deterministic values
4. **MODIFY**: `scripts_cosim/generate_gnn_datasets.sh` - Generate infrastructure first

## Random Number Generator Strategy

**Critical**: Use a single RNG instance with a fixed seed for all infrastructure generation:

```python
import random

# Create a single RNG instance with seed
infra_rng = random.Random(seed)

# Use this RNG for ALL infrastructure generation:
# - Network topology
# - Replica placements  
# - Queue distributions

# DO NOT create new Random() instances without seed
# DO NOT use global random module functions
```

## Queue Distribution Pre-Generation

For each platform that gets a replica:
1. Sample queue length using `sample_bounded_int()` with seeded RNG
2. Save to infrastructure.json: `{"node_name:platform_id": queue_length}`
3. Use exact value during simulation (no re-sampling)

## Replica Placement Pre-Generation

For each task type:
1. Determine which nodes should have replicas (based on preinit config)
2. For each node, determine how many replicas (based on per_client/per_server)
3. Select specific platforms deterministically (e.g., first N suitable platforms)
4. Save to infrastructure.json with exact platform IDs
5. Use exact placements during simulation (no random selection)

