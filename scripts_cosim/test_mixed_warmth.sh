#!/usr/bin/env bash
set -uo pipefail

# ============================================================================
# MIXED WARMTH TEST - Test cold start influence on optimal placement
# ============================================================================
# Uses Poisson queue distribution (lambda=4) to create mix of warm/cold:
# - Platforms with queue > 0: WARM (have executed tasks)
# - Platforms with queue = 0: COLD (initialized but never executed)
# ============================================================================

BASE="/root/projects/my-herosim"
CFG_PATH="${BASE}/simulation_data/space_with_network.json"
OUT_BASE="${BASE}/simulation_data/gnn_datasets_mixed_warmth"
RESULTS_DIR="${BASE}/simulation_data/initial_results_simple"
WORKLOAD_DIR="${BASE}/data/nofs-ids/traces"
WORKLOAD_TEMPLATES_DIR="${BASE}/data/nofs-ids/traces/gnn_templates_mixed"
PROGRESS_LOG="${BASE}/logs/progress_mixed.txt"
mkdir -p "${OUT_BASE}" "${RESULTS_DIR}" "${BASE}/logs" "${WORKLOAD_TEMPLATES_DIR}"

# Single config with Poisson queue distribution
PROBS=(0.50)
REPLICA_CFGS=("2 2 0.5 0.8")
SEEDS=(101)
# Poisson lambda=4 creates mix: ~2% zero, ~7% one, ~15% two, ...
QUEUE_DISTS=("pois4 poisson 4 0 0 12 1")

NUM_TASKS=5
NUM_CLIENT_NODES=10
TASK_TYPE_RATIOS=("50-50")

# Generate workload template
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
  
  local task_types=()
  for ((i=0; i<num_dnn1; i++)); do task_types+=("dnn1"); done
  for ((i=0; i<num_dnn2; i++)); do task_types+=("dnn2"); done
  
  local client_nodes=(0 1 2 3 4)
  
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
}

# Backup and restore
ORIG_BAK="${CFG_PATH}.bak.$(date +%s)"
cp "${CFG_PATH}" "${ORIG_BAK}"
restore_and_cleanup() { 
    cp "${ORIG_BAK}" "${CFG_PATH}" 2>/dev/null || true
    rm -f "${ORIG_BAK}" 2>/dev/null || true
}
trap restore_and_cleanup EXIT INT TERM

generate_workload_template

connection_probability="${PROBS[0]}"
read -r replicas_per_client replicas_per_server preinit_client_pct preinit_server_pct <<< "${REPLICA_CFGS[0]}"
seed="${SEEDS[0]}"
read -r qname qtype qp1 qp2 qmin qmax qstep <<< "${QUEUE_DISTS[0]}"

DID="ds_mixed_warmth"
OUT_DIR="${OUT_BASE}/${DID}"
WORKLOAD_TEMPLATE="${WORKLOAD_TEMPLATES_DIR}/workload_template_0.json"

echo ""
echo "=== MIXED WARMTH TEST ==="
echo "Queue distribution: ${qname} (${qtype}, lambda=${qp1})"
echo "This will create a mix of WARM (queue>0) and COLD (queue=0) platforms"
echo ""

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
        queue_distribution_params: {type:$qtype, lambda:$qp1, min:$qmin, max:$qmax, step:$qstep}
      }
    | .prewarm.dnn2 = {
        distribution: "none",
        queue_distribution: "statistical",
        queue_distribution_params: {type:$qtype, lambda:$qp1, min:$qmin, max:$qmax, step:$qstep}
      }' \
   "${CFG_PATH}" > "${TMP_CFG}"

cp "${TMP_CFG}" "${CFG_PATH}"
cp "${WORKLOAD_TEMPLATE}" "${WORKLOAD_DIR}/workload-10.json"

mkdir -p "${OUT_DIR}"

INFRA_FILE="${OUT_DIR}/infrastructure.json"
echo "[${DID}] Generating infrastructure with Poisson queue distribution..."
cd "${BASE}" && \
pipenv run python -m src.generate_infrastructure \
  --config "${TMP_CFG}" \
  --sim-input "${BASE}/data/nofs-ids" \
  --output "${INFRA_FILE}" \
  --seed "${seed}" || exit 1

# Show queue distribution stats
echo ""
echo "=== Queue Distribution Analysis ==="
pipenv run python3 -c "
import json
with open('${INFRA_FILE}', 'r') as f:
    infra = json.load(f)

for task_type, queues in infra['queue_distributions'].items():
    zero_count = sum(1 for v in queues.values() if v == 0)
    nonzero_count = sum(1 for v in queues.values() if v > 0)
    total_queue = sum(queues.values())
    print(f'{task_type}: {zero_count} COLD (0), {nonzero_count} WARM (>0), total queue={total_queue}')
"

echo ""
echo "=== Running Brute-Force Co-Simulation ==="
start_time=$(date +%s)

cd "${BASE}" && \
pipenv run python -m src.executecosimulation --brute-force \
  --infrastructure "${INFRA_FILE}" \
  2>&1 | tail -20

end_time=$(date +%s)
duration=$((end_time - start_time))

if [ -f "${RESULTS_DIR}/best.json" ]; then
  cp "${RESULTS_DIR}/best.json" "${OUT_DIR}/best.json"
  
  OPTIMAL_FILE=$(jq -r '.file' "${OUT_DIR}/best.json")
  OPTIMAL_RTT=$(jq -r '.rtt' "${OUT_DIR}/best.json")
  
  if [ -f "${RESULTS_DIR}/${OPTIMAL_FILE}" ]; then
    cp "${RESULTS_DIR}/${OPTIMAL_FILE}" "${OUT_DIR}/optimal_result.json"
  fi

  if [ -f "${RESULTS_DIR}/placements.jsonl" ]; then
    mkdir -p "${OUT_DIR}/placements"
    cp "${RESULTS_DIR}/placements.jsonl" "${OUT_DIR}/placements/placements.jsonl"
    PLACEMENT_COUNT=$(wc -l < "${OUT_DIR}/placements/placements.jsonl" 2>/dev/null || echo "0")
  fi

  cp "${TMP_CFG}" "${OUT_DIR}/space_with_network.json"
  cp "${WORKLOAD_TEMPLATE}" "${OUT_DIR}/workload.json"
  
  rm -f "${RESULTS_DIR}"/simulation_*.json
  rm -f "${RESULTS_DIR}/best.json"
  rm -f "${RESULTS_DIR}/placements.jsonl"
  
  echo ""
  echo "=== RESULTS ==="
  echo "Optimal RTT: ${OPTIMAL_RTT}s"
  echo "Placements evaluated: ${PLACEMENT_COUNT}"
  echo "Duration: ${duration}s"
  echo ""
  
  # Analyze cold start breakdown
  echo "=== Cold Start Analysis ==="
  pipenv run python3 -c "
import json

with open('${OUT_DIR}/optimal_result.json', 'r') as f:
    result = json.load(f)

task_results = result['stats']['taskResults']

cold_count = sum(1 for tr in task_results if tr['coldStarted'])
total = len(task_results)
total_cold = sum(tr['coldStartTime'] for tr in task_results)

print(f'Cold starts: {cold_count}/{total} tasks')
print(f'Total cold start time: {total_cold:.4f}s')
print()
print('Per-task breakdown:')
for tr in task_results:
    status = 'COLD' if tr['coldStarted'] else 'WARM'
    print(f\"  Task {tr['taskId']} ({tr['taskType']['name']}): {status} cold_start={tr['coldStartTime']:.3f}s\")
"

else
  echo "[${DID}] No optimal result found"
fi

rm -f "${TMP_CFG}"

echo ""
echo "=== Test Complete ==="
echo "Output: ${OUT_DIR}"
