#!/usr/bin/env bash
set -uo pipefail

# ============================================================================
# MINIMAL GNN Dataset Test Script
# ============================================================================
# Purpose: Run a single valid dataset generation to analyze the output
# - Uses 1 network topology (connection probability 0.50 - moderate connectivity)
# - Uses 1 replica config (moderate preinit)
# - Uses 1 queue distribution (zero - simplest case)
# - Stops after first successful result
# ============================================================================

BASE="/root/projects/my-herosim"
CFG_PATH="${BASE}/simulation_data/space_with_network.json"
OUT_BASE="${BASE}/simulation_data/gnn_datasets_analysis"
RESULTS_DIR="${BASE}/simulation_data/initial_results_simple"
WORKLOAD_DIR="${BASE}/data/nofs-ids/traces"
WORKLOAD_TEMPLATES_DIR="${BASE}/data/nofs-ids/traces/gnn_templates_analysis"
PROGRESS_LOG="${BASE}/logs/progress_analysis.txt"
mkdir -p "${OUT_BASE}" "${RESULTS_DIR}" "${BASE}/logs" "${WORKLOAD_TEMPLATES_DIR}"

# ============================================================================
# MINIMAL CONFIG: Single network topology, single replica config, single queue dist
# ============================================================================
PROBS=(0.50)  # Single moderate connectivity (50% connection probability)

# Single replica config: moderate warm start (50% preinit on clients/servers, 2 replicas each)
REPLICA_CFGS=(
  "2 2 0.5 0.8"
)

SEEDS=(101)  # Single seed

# Single queue distribution: zero queues (simplest case to analyze)
QUEUE_DISTS=(
  "zero constant 0 0 0 0 0"
)

# Workload: 5 tasks with 50-50 dnn1/dnn2 ratio
NUM_TASKS=5
NUM_CLIENT_NODES=10
TASK_TYPE_RATIOS=("50-50")  # Single ratio for simplicity

# Generate single workload template
generate_workload_template() {
  local base_trace="${WORKLOAD_DIR}/workload-10.json"
  
  if [ ! -f "${base_trace}" ]; then
    echo "ERROR: Base workload trace not found: ${base_trace}"
    exit 1
  fi
  
  local rps=$(jq -r '.rps' "${base_trace}")
  local duration=$(jq -r '.duration' "${base_trace}")
  local base_events=$(jq '.events' "${base_trace}")
  
  echo "Generating workload template with ${NUM_TASKS} tasks..."
  
  local ratio="${TASK_TYPE_RATIOS[0]}"
  IFS='-' read -r dnn1_pct dnn2_pct <<< "${ratio}"
  
  local num_dnn1=$((NUM_TASKS * dnn1_pct / 100))
  local num_dnn2=$((NUM_TASKS - num_dnn1))
  
  local workload_file="${WORKLOAD_TEMPLATES_DIR}/workload_template_0.json"
  
  # Create array of task types
  local task_types=()
  for ((i=0; i<num_dnn1; i++)); do task_types+=("dnn1"); done
  for ((i=0; i<num_dnn2; i++)); do task_types+=("dnn2"); done
  
  # Fixed client node assignments for reproducibility
  local client_nodes=(0 1 2 3 4)  # Tasks from 5 different client nodes
  
  # Use jq to construct the workload JSON
  jq -n \
    --argjson rps "${rps}" \
    --argjson duration "${duration}" \
    --argjson base_events "${base_events}" \
    --argjson num_tasks "${NUM_TASKS}" \
    '{
      rps: $rps,
      duration: $duration,
      events: [
        range($num_tasks) as $i | $base_events[$i % ($base_events | length)] | 
        . + {timestamp: .timestamp}
      ]
    }' > "${workload_file}.tmp"
  
  # Replace task types and client nodes
  local client_nodes_csv=$(IFS=,; echo "${client_nodes[*]}")
  python3 << EOF
import json

task_types = "${task_types[@]}".split()
client_nodes = [${client_nodes_csv}]

with open("${workload_file}.tmp", 'r') as f:
    workload = json.load(f)

for idx, event in enumerate(workload['events'][:${NUM_TASKS}]):
    task_type = task_types[idx]
    client_node = client_nodes[idx]
    
    event['application']['name'] = f"nofs-{task_type}"
    event['application']['dag'] = {task_type: []}
    event['node_name'] = f"client_node{client_node}"

workload['events'] = workload['events'][:${NUM_TASKS}]

with open("${workload_file}", 'w') as f:
    json.dump(workload, f, indent=2)
EOF
  
  rm -f "${workload_file}.tmp"
  
  echo "  Template: ${num_dnn1} dnn1 + ${num_dnn2} dnn2, clients: [${client_nodes[@]}]"
  echo "Workload template generation complete."
}

# Backup original config + restore on exit
ORIG_BAK="${CFG_PATH}.bak.$(date +%s)"
cp "${CFG_PATH}" "${ORIG_BAK}"
restore_and_cleanup() { 
    cp "${ORIG_BAK}" "${CFG_PATH}" 2>/dev/null || true
    rm -f "${ORIG_BAK}" 2>/dev/null || true
}
trap restore_and_cleanup EXIT INT TERM

# Generate workload template
generate_workload_template

# Single iteration
connection_probability="${PROBS[0]}"
read -r replicas_per_client replicas_per_server preinit_client_pct preinit_server_pct <<< "${REPLICA_CFGS[0]}"
seed="${SEEDS[0]}"
read -r qname qtype qp1 qp2 qmin qmax qstep <<< "${QUEUE_DISTS[0]}"

DID="ds_analysis_00000"
OUT_DIR="${OUT_BASE}/${DID}"

WORKLOAD_TEMPLATE="${WORKLOAD_TEMPLATES_DIR}/workload_template_0.json"

echo ""
echo "=== Running Single GNN Dataset Generation ==="
echo "[${DID}] conn=${connection_probability} seed=${seed} rpc=${replicas_per_client} rps=${replicas_per_server} cpct=${preinit_client_pct} spct=${preinit_server_pct} q=${qname}"

TMP_CFG="$(mktemp)"
jq --argjson p "${connection_probability}" \
   --argjson pc "${replicas_per_client}" \
   --argjson ps "${replicas_per_server}" \
   --argjson cp "${preinit_client_pct}" \
   --argjson sp "${preinit_server_pct}" \
   --argjson seed "${seed}" \
   --arg qtype "${qtype}" \
   --argjson qp1 "${qp1}" \
   --argjson qp2 "${qp2}" \
   --argjson qmin "${qmin}" \
   --argjson qmax "${qmax}" \
   --argjson qstep "${qstep}" \
   '.network.topology.connection_probability=$p
    | .network.topology.seed=$seed
    | .preinit.client_percentage=$cp
    | .preinit.server_percentage=$sp
    | .replicas.dnn1.per_client=$pc
    | .replicas.dnn1.per_server=$ps
    | .replicas.dnn2.per_client=$pc
    | .replicas.dnn2.per_server=$ps
    | .prewarm.dnn1 = {
        distribution: "none",
        queue_distribution: "statistical",
        queue_distribution_params: (
          if $qtype == "constant" then {type:"constant", value:$qp1, min:$qmin, max:$qmax, step:$qstep}
          elif $qtype == "poisson" then {type:$qtype, lambda:$qp1, min:$qmin, max:$qmax, step:$qstep}
          elif $qtype == "normal" then {type:$qtype, mean:$qp1, stddev:($qp2|if .==0 then 1 else . end), min:$qmin, max:$qmax, step:$qstep}
          elif $qtype == "uniform" then {type:$qtype, low:$qp1, high:$qp2, min:$qmin, max:$qmax, step:$qstep}
          else {type:"poisson", lambda:4, min:$qmin, max:$qmax, step:$qstep} end)
      }
    | .prewarm.dnn2 = {
        distribution: "none",
        queue_distribution: "statistical",
        queue_distribution_params: (
          if $qtype == "constant" then {type:"constant", value:$qp1, min:$qmin, max:$qmax, step:$qstep}
          elif $qtype == "poisson" then {type:$qtype, lambda:$qp1, min:$qmin, max:$qmax, step:$qstep}
          elif $qtype == "normal" then {type:$qtype, mean:$qp1, stddev:($qp2|if .==0 then 1 else . end), min:$qmin, max:$qmax, step:$qstep}
          elif $qtype == "uniform" then {type:$qtype, low:$qp1, high:$qp2, min:$qmin, max:$qmax, step:$qstep}
          else {type:"poisson", lambda:4, min:$qmin, max:$qmax, step:$qstep} end)
      }' \
   "${CFG_PATH}" > "${TMP_CFG}"

# Apply the temporary config
cp "${TMP_CFG}" "${CFG_PATH}"

# Copy workload template
cp "${WORKLOAD_TEMPLATE}" "${WORKLOAD_DIR}/workload-10.json"

# Prepare dataset directory
mkdir -p "${OUT_DIR}"

# Generate deterministic infrastructure
INFRA_FILE="${OUT_DIR}/infrastructure.json"
echo "[${DID}] Generating deterministic infrastructure..."
cd "${BASE}" && \
pipenv run python -m src.generate_infrastructure \
  --config "${TMP_CFG}" \
  --sim-input "${BASE}/data/nofs-ids" \
  --output "${INFRA_FILE}" \
  --seed "${seed}" || {
  echo "[${DID}] ERROR: Infrastructure generation failed"
  exit 1
}

echo ""
echo "=== Infrastructure Generated ==="
echo "File: ${INFRA_FILE}"
echo ""

# Run the brute-force co-simulation
echo "=== Running Brute-Force Co-Simulation ==="
start_time=$(date +%s)

cd "${BASE}" && \
pipenv run python -m src.executecosimulation --brute-force \
  --infrastructure "${INFRA_FILE}" \
  2>&1 | tee "${OUT_DIR}/run.log" || {
  echo "[${DID}] Simulation failed"
}

end_time=$(date +%s)
duration=$((end_time - start_time))

# Check results
if [ -f "${RESULTS_DIR}/best.json" ]; then
  cp "${RESULTS_DIR}/best.json" "${OUT_DIR}/best.json"
  
  OPTIMAL_FILE=$(jq -r '.file' "${OUT_DIR}/best.json")
  OPTIMAL_RTT=$(jq -r '.rtt' "${OUT_DIR}/best.json")
  
  if [ -f "${RESULTS_DIR}/${OPTIMAL_FILE}" ]; then
    cp "${RESULTS_DIR}/${OPTIMAL_FILE}" "${OUT_DIR}/optimal_result.json"
    echo ""
    echo "=== OPTIMAL RESULT ==="
    echo "File: ${OPTIMAL_FILE}"
    echo "RTT: ${OPTIMAL_RTT}s"
  fi

  # Copy placement summaries
  if [ -f "${RESULTS_DIR}/placements.jsonl" ]; then
    mkdir -p "${OUT_DIR}/placements"
    cp "${RESULTS_DIR}/placements.jsonl" "${OUT_DIR}/placements/placements.jsonl"
    PLACEMENT_COUNT=$(wc -l < "${OUT_DIR}/placements/placements.jsonl" 2>/dev/null || echo "0")
    echo "Placements evaluated: ${PLACEMENT_COUNT}"
  fi

  # Save config and workload
  cp "${TMP_CFG}" "${OUT_DIR}/space_with_network.json"
  cp "${WORKLOAD_TEMPLATE}" "${OUT_DIR}/workload.json"
  
  echo "${DID} SUCCESS $(date '+%Y-%m-%d %H:%M:%S') ${duration}s RTT=${OPTIMAL_RTT}s" >> "${PROGRESS_LOG}"
  
  # Clean up
  rm -f "${RESULTS_DIR}"/simulation_*.json
  rm -f "${RESULTS_DIR}/best.json"
  rm -f "${RESULTS_DIR}/placements.jsonl"
  
  echo ""
  echo "=== SUCCESS ==="
  echo "Duration: ${duration}s"
  echo "Output directory: ${OUT_DIR}"
else
  echo "[${DID}] No optimal result found"
  echo "${DID} FAILED $(date '+%Y-%m-%d %H:%M:%S') ${duration}s" >> "${PROGRESS_LOG}"
fi

rm -f "${TMP_CFG}"

echo ""
echo "=== Analysis Complete ==="
echo "Output directory: ${OUT_DIR}"
echo "  - infrastructure.json: Network topology, replica placements, queue distributions"
echo "  - optimal_result.json: Best placement result"
echo "  - placements/placements.jsonl: All placement combinations with RTTs"
echo "  - workload.json: Workload template used"
echo "  - space_with_network.json: Configuration used"
echo "  - run.log: Full simulation log"
