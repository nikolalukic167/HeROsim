# Simulation Results: `initial_results_simple`

This document describes the structure and contents of the simulation result files generated in the `simulation_data/initial_results_simple/` directory. Each file (e.g., `simulation_1.json`, `simulation_2.json`, ...) contains the results of a single simulation run, including all relevant statistics, configuration, and detailed per-task, per-node, and per-application results.

---

## Top-Level Structure

Each result file is a JSON object with the following structure:

```json
{
  "status": "success",
  "config": {
    "infrastructure": { /* see below */ },
    "workload": { /* see below */ }
  },
  "sim_inputs": { /* see below */ },
  "stats": {
    "policy": { /* SimulationPolicy */ },
    "endTime": 12345.67,
    "traceFile": "data/nofs-ids/traces/workload-125-225.json",
    "unusedPlatforms": 0.0,
    "unusedNodes": 0.0,
    "averageOccupation": 0.0,
    "averageElapsedTime": 0.0,
    "averagePullTime": 0.0,
    "averageColdStartTime": 0.0,
    "averageExecutionTime": 0.0,
    "averageWaitTime": 0.0,
    "averageQueueTime": 0.0,
    "averageInitializationTime": 0.0,
    "averageComputeTime": 0.0,
    "averageCommunicationsTime": 0.0,
    "penaltyProportion": 0.0,
    "localDependenciesProportion": 0.0,
    "localCommunicationsProportion": 0.0,
    "nodeCacheHitsProportion": 0.0,
    "taskCacheHitsProportion": 0.0,
    "coldStartProportion": 0.0,
    "taskResponseTimeDistribution": [0.0, ...],
    "applicationResponseTimeDistribution": [0.0, ...],
    "penaltyDistributionOverTime": [[123.0, 0.0], ...],
    "energy": 0.0,
    "reclaimableEnergy": 0.0,
    "applicationResults": [ /* ApplicationResult[] */ ],
    "nodeResults": [ /* NodeResult[] */ ],
    "taskResults": [ /* TaskResult[] */ ],
    "scaleEvents": [ /* ScaleEvent[] */ ],
    "systemEvents": [ /* SystemEvent[] */ ],
    "averageNetworkLatency": 0.0,
    "nodePairLatencies": { "nodeA->nodeB": 0.0, ... },
    "networkTopology": { "nodeA": { "nodeB": 0.1, ... }, ... },
    "offloadingRate": 0.0,
    "systemStateResults": [ /* SystemStateResult[] */ ]
  },
  "sample": {
    "apps": ["nofs-dnn1", "nofs-dnn2"],
    "sample": [0.5, 0.3, ...],
    "mapping": {"0": "network_bandwidth", ...},
    "infra_config": { /* see infrastructure below */ },
    "workload_base": { /* base workload */ },
    "sim_inputs": { /* see below */ },
    "scheduling_strategy": "gnn_gnn",
    "cache_policy": "fifo",
    "task_priority": "fifo",
    "keep_alive": 30,
    "queue_length": 100
  }
}
```

---

## Key Sections and Placeholders

### 1. `config.infrastructure`
Infrastructure configuration for the simulation, e.g.:
```json
{
  "network": { "bandwidth": 100.0 },
  "nodes": [
    {
      "node_name": "node0",
      "type": "rpi",
      "memory": 8,
      "platforms": ["rpiCpu", ...],
      "storage": ["flashCard", ...],
      "network_map": { "node1": 0.12, ... }
    },
    ...
  ]
}
```

### 2. `config.workload`
Flattened workload events and stats, e.g.:
```json
{
  "rps": 10,
  "duration": 1200,
  "events": [
    {
      "timestamp": 0.0,
      "application": { "name": "nofs-dnn1", "dag": { ... } },
      "qos": { "name": "qos1", "maxDurationDeviation": 1.0 },
      "node_name": "node0"
    },
    ...
  ]
}
```

### 3. `sim_inputs`
All static simulation input files (platform types, storage types, etc.), e.g.:
```json
{
  "platform_types": { "rpiCpu": { ... }, ... },
  "storage_types": { ... },
  "qos_types": { ... },
  "application_types": { ... },
  "task_types": { ... }
}
```

### 4. `stats` (SimulationStats)
A detailed object with all simulation statistics and results. See below for a faithful placeholder:

```json
{
  "policy": { "priority": { "tasks": "fifo" }, "scheduling": "gnn_gnn", ... },
  "endTime": 12345.67,
  "traceFile": "data/nofs-ids/traces/workload-125-225.json",
  "unusedPlatforms": 0.0,
  "unusedNodes": 0.0,
  "averageOccupation": 0.0,
  "averageElapsedTime": 0.0,
  "averagePullTime": 0.0,
  "averageColdStartTime": 0.0,
  "averageExecutionTime": 0.0,
  "averageWaitTime": 0.0,
  "averageQueueTime": 0.0,
  "averageInitializationTime": 0.0,
  "averageComputeTime": 0.0,
  "averageCommunicationsTime": 0.0,
  "penaltyProportion": 0.0,
  "localDependenciesProportion": 0.0,
  "localCommunicationsProportion": 0.0,
  "nodeCacheHitsProportion": 0.0,
  "taskCacheHitsProportion": 0.0,
  "coldStartProportion": 0.0,
  "taskResponseTimeDistribution": [0.0, ...],
  "applicationResponseTimeDistribution": [0.0, ...],
  "penaltyDistributionOverTime": [[123.0, 0.0], ...],
  "energy": 0.0,
  "reclaimableEnergy": 0.0,
  "applicationResults": [
    {
      "applicationId": 0,
      "dispatchedTime": 0.0,
      "elapsedTime": 0.0,
      "pullTime": 0.0,
      "coldStartTime": 0.0,
      "executionTime": 0.0,
      "communicationsTime": 0.0,
      "penalty": false,
      "type": "nofs-dnn1",
      "platform_type": "rpiCpu"
    }, ...
  ],
  "nodeResults": [
    {
      "nodeId": 0,
      "unused": false,
      "energy": { "rpiCpu": 0.0, ... },
      "energyIdle": { "rpiCpu": 0.0, ... },
      "idleTime": { "rpiCpu": 0.0, ... },
      "schedulingTime": 0.0,
      "storageTime": 0.0,
      "localDependencies": 0,
      "cacheHits": 0,
      "platformResults": [ /* PlatformResult[] */ ],
      "storageResults": [ /* StorageResult[] */ ]
    }, ...
  ],
  "taskResults": [
    {
      "taskId": 0,
      "dispatchedTime": 0.0,
      "scheduledTime": 0.0,
      "arrivedTime": 0.0,
      "startedTime": 0.0,
      "doneTime": 0.0,
      "applicationType": { "name": "nofs-dnn1", "dag": { ... } },
      "taskType": { "name": "dnn1", ... },
      "platform": { "shortName": "rpiCpu", ... },
      "elapsedTime": 0.0,
      "pullTime": 0.0,
      "coldStartTime": 0.0,
      "executionTime": 0.0,
      "waitTime": 0.0,
      "queueTime": 0.0,
      "initializationTime": 0.0,
      "computeTime": 0.0,
      "communicationsTime": 0.0,
      "coldStarted": false,
      "cacheHit": false,
      "localDependencies": false,
      "localCommunications": false,
      "energy": 0.0,
      "networkLatency": 0.0,
      "sourceNode": "node0",
      "executionNode": "node1",
      "executionPlatform": "rpiCpu",
      "gnn_decision_time": 0.0
    }, ...
  ],
  "scaleEvents": [
    { "name": "scaleUp", "timestamp": 0.0, "action": "add", "count": 1, "average_queue_length": 0.0, "platform_type": "rpiCpu" }, ...
  ],
  "systemEvents": [
    { "name": "eventName", "timestamp": 0.0, "count": 0, "average_queue_length": 0.0 }, ...
  ],
  "averageNetworkLatency": 0.0,
  "nodePairLatencies": { "node0->node1": 0.0, ... },
  "networkTopology": { "node0": { "node1": 0.1, ... }, ... },
  "offloadingRate": 0.0,
  "systemStateResults": [
    {
      "timestamp": 0.0,
      "scheduler_state": { ... },
      "available_resources": { "node0": [0, 1], ... },
      "replicas": { "dnn1": [["node0", 0], ...], ... }
    }, ...
  ]
}
```

---

## Notes
- All numeric values are placeholders and will be filled with real simulation data.
- The actual number of nodes, tasks, applications, and events will depend on the simulation configuration and workload.
- The `networkTopology` field provides the full network map used in the simulation.
- The `systemStateResults` array contains snapshots of the system state at various points in the simulation.

---

## See Also
- `src/placement/model.py` for full type definitions
- `src/placement/orchestrator.py` for result construction logic
- `simulation_data/space_with_network.json` for infrastructure config examples 