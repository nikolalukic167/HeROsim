#!/usr/bin/env bash
set -uo pipefail  # -u: undefined variables error, -o pipefail: pipeline failures

# ============================================================================
# GNN Dataset Generation Script
# ============================================================================
# Generate diverse datasets by systematically varying:
#   1. Network topology (connection probability): 19 levels
#   2. Replica configuration (per_client, per_server, preinit percentages): 15 configs
#   3. Pre-warm queue levels (initial backlog per replica): 11 levels
#   4. Workload patterns (task type ratios, client node distribution): 10 templates
#
# This creates up to 20,000 unique datasets (capped from 19×15×11×10 = 31,350 possible)
# for training GNN-based schedulers with production-realistic scenarios.
#
# Each dataset contains:
#   - space_with_network.json: Infrastructure configuration
#   - workload.json: Workload trace with task types and client assignments
#   - run.log: Execution log
#   - best.json: Metadata about optimal result
#   - optimal_result.json: Best placement configuration found
# ============================================================================

BASE="/root/projects/my-herosim"
CFG_PATH="${BASE}/simulation_data/space_with_network.json"
OUT_BASE="${BASE}/simulation_data/gnn_datasets"
RESULTS_DIR="${BASE}/simulation_data/initial_results_simple"
WORKLOAD_DIR="${BASE}/data/nofs-ids/traces"
WORKLOAD_TEMPLATES_DIR="${BASE}/data/nofs-ids/traces/gnn_templates"
PROGRESS_LOG="${BASE}/logs/progress.txt"
mkdir -p "${OUT_BASE}" "${RESULTS_DIR}" "${BASE}/logs" "${WORKLOAD_TEMPLATES_DIR}"

# Grid definition
PROBS=(0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90)

# Replica config matrix: (per_client per_server client_percentage server_percentage) → 15 configs
REPLICA_CFGS=(
  "3 3 0.5 0.7"
  "2 3 0.6 0.8"
  "2 1 0.6 0.9"
  "1 3 0.3 0.7"
  "0 4 0.4 0.5"
  "2 2 0.5 0.8"
  "1 4 0.6 0.6"
  "2 2 0.6 0.8"
  "1 4 0.4 0.6"
  "2 2 0.5 0.7"
  "3 2 0.7 0.8"
  "1 2 0.5 0.9"
  "2 4 0.8 0.7"
  "3 1 0.6 0.8"
  "1 1 0.4 0.8"
)

# Additional random seeds to scale dataset count (~PROBS*CFG*QDISTS*SEEDS)
# Target ~10k datasets: 19*15*8*5 ≈ 11,400

# SEEDS=(101 202 303 404 505)
SEEDS=(101 202) 


# Queue distribution variants (type mean/std or lambda, min, max, step-meta)
# Format: name type param1 param2 min max step
# Expanded set to improve diversity while staying bounded [0,32]
QUEUE_DISTS=(
  "pois4 poisson 4 0 0 32 1"
  "pois6 poisson 6 0 0 32 1"
  "pois8 poisson 8 0 0 32 1"
  "pois12 poisson 12 0 0 32 1"
  "pois16 poisson 16 0 0 32 1"
  "pois20 poisson 20 0 0 32 1"
  "norm8 normal 8 3 0 32 1"
  "norm12 normal 12 4 0 32 1"
  "norm16 normal 16 5 0 32 1"
  "norm20 normal 20 6 0 32 1"
  "unif024 uniform 0 24 0 24 1"
  "unif028 uniform 0 28 0 28 1"
  "unif032 uniform 0 32 0 32 1"
)

# Workload generation parameters
NUM_TASKS=5  # Number of tasks per workload
# NUM_TASKS_MIN=3  # Uncomment to enable variable workload sizes
# NUM_TASKS_MAX=10
NUM_CLIENT_NODES=10  # client_node0 to client_node9
TASK_TYPE_RATIOS=("0-100" "10-90" "20-80" "30-70" "40-60" "50-50" "60-40" "70-30" "80-20" "90-10" "100-0")  # dnn1%-dnn2% ratios

# Generate 10 diverse workload templates
# Strategy: Cycle through task type ratios, with multiple random client distributions per ratio
generate_workload_templates() {
  local base_trace="${WORKLOAD_DIR}/workload-10.json"
  
  if [ ! -f "${base_trace}" ]; then
    echo "ERROR: Base workload trace not found: ${base_trace}"
    exit 1
  fi
  
  # Read the base workload to extract template structure
  local rps=$(jq -r '.rps' "${base_trace}")
  local duration=$(jq -r '.duration' "${base_trace}")
  local base_events=$(jq '.events' "${base_trace}")
  
  echo "Generating 10 workload templates with ${NUM_TASKS} tasks each..."
  
  for template_idx in {0..9}; do
    local ratio="${TASK_TYPE_RATIOS[$((template_idx % ${#TASK_TYPE_RATIOS[@]}))]}"
    IFS='-' read -r dnn1_pct dnn2_pct <<< "${ratio}"
    
    local num_dnn1=$((NUM_TASKS * dnn1_pct / 100))
    local num_dnn2=$((NUM_TASKS - num_dnn1))
    
    local workload_file="${WORKLOAD_TEMPLATES_DIR}/workload_template_${template_idx}.json"
    
    # Create array of task types
    local task_types=()
    for ((i=0; i<num_dnn1; i++)); do task_types+=("dnn1"); done
    for ((i=0; i<num_dnn2; i++)); do task_types+=("dnn2"); done
    
    # Generate random client node assignments for each task
    local client_nodes=()
    for ((i=0; i<NUM_TASKS; i++)); do
      client_nodes+=($((RANDOM % NUM_CLIENT_NODES)))
    done
    
    # Use jq to construct the workload JSON with randomized tasks
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
    
    # Now replace task types and client nodes using Python for better array handling
    local client_nodes_csv=$(IFS=,; echo "${client_nodes[*]}")
    python3 << EOF
import json

task_types = "${task_types[@]}".split()
client_nodes = [${client_nodes_csv}]

with open("${workload_file}.tmp", 'r') as f:
    workload = json.load(f)

# Update each event with new task type and client node
for idx, event in enumerate(workload['events'][:${NUM_TASKS}]):
    task_type = task_types[idx]
    client_node = client_nodes[idx]
    
    event['application']['name'] = f"nofs-{task_type}"
    event['application']['dag'] = {task_type: []}
    event['node_name'] = f"client_node{client_node}"

# Keep only NUM_TASKS events
workload['events'] = workload['events'][:${NUM_TASKS}]

with open("${workload_file}", 'w') as f:
    json.dump(workload, f, indent=2)
EOF
    
    rm -f "${workload_file}.tmp"
    
    echo "  Template ${template_idx}: ${num_dnn1} dnn1 + ${num_dnn2} dnn2, clients: [${client_nodes[@]}]"
  done
  
  echo "Workload template generation complete."
}

# Backup original config + ensure restoration on exit
ORIG_BAK="${CFG_PATH}.bak.$(date +%s)"
cp "${CFG_PATH}" "${ORIG_BAK}"
restore_cfg() { cp "${ORIG_BAK}" "${CFG_PATH}" 2>/dev/null || true; }
trap restore_cfg EXIT INT TERM

# Generate workload templates before starting dataset generation
generate_workload_templates

i=0
workload_template_idx=0  # Cycle through workload templates
for connection_probability in "${PROBS[@]}"; do
  for replica_cfg in "${REPLICA_CFGS[@]}"; do
    read -r replicas_per_client replicas_per_server preinit_client_pct preinit_server_pct <<< "${replica_cfg}"
    for seed in "${SEEDS[@]}"; do
      for qdist in "${QUEUE_DISTS[@]}"; do
        read -r qname qtype qp1 qp2 qmin qmax qstep <<< "${qdist}"

        DID=$(printf "ds_%05d" "$i")  # 0..99999 supports up to 100k
        OUT_DIR="${OUT_BASE}/${DID}"

        # Select workload template (cycle through 10 templates)
        WORKLOAD_TEMPLATE="${WORKLOAD_TEMPLATES_DIR}/workload_template_${workload_template_idx}.json"
        
        # Validate workload template exists
        if [ ! -f "${WORKLOAD_TEMPLATE}" ]; then
          echo "ERROR: Workload template not found: ${WORKLOAD_TEMPLATE}"
          exit 1
        fi

        echo "[${DID}] wl=${workload_template_idx} conn=${connection_probability} seed=${seed} rpc=${replicas_per_client} rps=${replicas_per_server} cpct=${preinit_client_pct} spct=${preinit_server_pct} q=${qname}"

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
            # Disable replica sampling; use fixed per_server/per_client
            | .prewarm.dnn1 = {
                distribution: "none",
                queue_distribution: "statistical",
                queue_distribution_params: (
                  if $qtype == "poisson" then {type:$qtype, lambda:$qp1, min:$qmin, max:$qmax, step:$qstep}
                  elif $qtype == "normal" then {type:$qtype, mean:$qp1, stddev:($qp2|if .==0 then 1 else . end), min:$qmin, max:$qmax, step:$qstep}
                  elif $qtype == "uniform" then {type:$qtype, low:$qp1, high:$qp2, min:$qmin, max:$qmax, step:$qstep}
                  else {type:"poisson", lambda:4, min:$qmin, max:$qmax, step:$qstep} end)
              }
            | .prewarm.dnn2 = {
                distribution: "none",
                queue_distribution: "statistical",
                queue_distribution_params: (
                  if $qtype == "poisson" then {type:$qtype, lambda:$qp1, min:$qmin, max:$qmax, step:$qstep}
                  elif $qtype == "normal" then {type:$qtype, mean:$qp1, stddev:($qp2|if .==0 then 1 else . end), min:$qmin, max:$qmax, step:$qstep}
                  elif $qtype == "uniform" then {type:$qtype, low:$qp1, high:$qp2, min:$qmin, max:$qmax, step:$qstep}
                  else {type:"poisson", lambda:4, min:$qmin, max:$qmax, step:$qstep} end)
              }' \
           "${CFG_PATH}" > "${TMP_CFG}"

        # Apply the temporary config to the hardcoded path (sequential to avoid races)
        cp "${TMP_CFG}" "${CFG_PATH}"
        
        # Copy workload template to the expected location for this run
        cp "${WORKLOAD_TEMPLATE}" "${WORKLOAD_DIR}/workload-10.json"

        # Run executor (sequential). Let brute-force run fully (no cap)
        # Use temporary log file until we know if it succeeds
        TMP_LOG="$(mktemp)"
        start_time=$(date +%s)
        
        # Run with timeout to prevent infinite hangs (3600s = 1 hour per dataset)
        timeout 3600 bash -c "
          cd '${BASE}' && \
          python -m src.executecosimulation --brute-force \
            > '${TMP_LOG}' 2>&1
        " || {
          EXIT_CODE=$?
          if [ $EXIT_CODE -eq 124 ]; then
            echo "[${DID}] TIMEOUT after 3600s" >> "${TMP_LOG}"
          else
            echo "[${DID}] FAIL exit_code=$EXIT_CODE" >> "${TMP_LOG}"
          fi
        }
        
        end_time=$(date +%s)
        duration=$((end_time - start_time))

        # Prepare dataset directory
        mkdir -p "${OUT_DIR}"

        # Check if simulation found an optimal result
        if [ -f "${RESULTS_DIR}/best.json" ]; then
          # Copy best.json to dataset directory before next run overwrites it
          cp "${RESULTS_DIR}/best.json" "${OUT_DIR}/best.json"
          
          OPTIMAL_FILE=$(jq -r '.file' "${OUT_DIR}/best.json")
          OPTIMAL_RTT=$(jq -r '.rtt' "${OUT_DIR}/best.json")
          
          if [ -f "${RESULTS_DIR}/${OPTIMAL_FILE}" ]; then
            cp "${RESULTS_DIR}/${OPTIMAL_FILE}" "${OUT_DIR}/optimal_result.json"
            echo "[${DID}] Optimal: ${OPTIMAL_FILE} (RTT: ${OPTIMAL_RTT}s)"
          fi

          # Index whatever results are present (heavy results kept if not pruned)
          ls -1 "${RESULTS_DIR}"/simulation_*.json 2>/dev/null | xargs -I{} basename {} > "${OUT_DIR}/results_index.txt" || true

          # Copy lightweight placement summaries (placement + RTT) into a subfolder
          mkdir -p "${OUT_DIR}/placements"
          ls -1 "${RESULTS_DIR}"/placement_summary_*.json 2>/dev/null | while read -r SUM; do
            cp "$SUM" "${OUT_DIR}/placements/" || true
            rm -f "$SUM" || true
          done

          # Save configuration files (reuse TMP_CFG instead of regenerating)
          cp "${TMP_CFG}" "${OUT_DIR}/space_with_network.json"
          cp "${WORKLOAD_TEMPLATE}" "${OUT_DIR}/workload.json"
          mv "${TMP_LOG}" "${OUT_DIR}/run.log"
          
          # Log progress with time
          echo "${DID} SUCCESS $(date '+%Y-%m-%d %H:%M:%S') ${duration}s RTT=${OPTIMAL_RTT}s qdist=${qname} seed=${seed}" >> "${PROGRESS_LOG}"
          
          # Clean up result files from shared directory
          rm -f "${RESULTS_DIR}/${OPTIMAL_FILE}"
          rm -f "${RESULTS_DIR}/best.json"
        else
          echo "[${DID}] No optimal result found - continuing (no skip)"
          # Log failure with time
          echo "${DID} FAILED $(date '+%Y-%m-%d %H:%M:%S') ${duration}s conn=${connection_probability} repl='${replica_cfg}' wl=${workload_template_idx} q=${qname} seed=${seed}" >> "${PROGRESS_LOG}"
          # Keep logs/configs for debugging this combo
          cp "${TMP_CFG}" "${OUT_DIR}/space_with_network.json"
          cp "${WORKLOAD_TEMPLATE}" "${OUT_DIR}/workload.json"
          mv "${TMP_LOG}" "${OUT_DIR}/run.log"
        fi

        # Clean tmp cfg
        rm -f "${TMP_CFG}"

        i=$((i+1))
        # Cycle through workload templates (10 templates total)
        workload_template_idx=$(( (workload_template_idx + 1) % 10 ))
        
        # Stop when reaching ~100000 datasets
        if [ "$i" -ge 10 ]; then
          break 5
        fi
      done
    done
  done
done

# Original config is restored by trap
echo ""
echo "=== Generation Complete ==="
DATASET_COUNT=$(ls -1d ${OUT_BASE}/ds_* 2>/dev/null | wc -l)
echo "Total datasets generated: ${DATASET_COUNT}"
echo "Dataset directory: ${OUT_BASE}"
echo "Progress log: ${PROGRESS_LOG}"
echo ""
echo "Done!"
