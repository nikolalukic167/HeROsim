#!/usr/bin/env bash
set -euo pipefail

# Generate 100 diverse datasets by varying network topology and replica preinit.
# This script edits the repo's config in-place per dataset (sequentially),
# runs the brute-force executor, and stores the exact config and log per dataset.

BASE="/root/projects/my-herosim"
CFG_PATH="${BASE}/simulation_data/space_with_network.json"
OUT_BASE="${BASE}/simulation_data/gnn_datasets"
RESULTS_DIR="${BASE}/simulation_data/initial_results_simple"
PROGRESS_LOG="${BASE}/logs/progress.txt"
mkdir -p "${OUT_BASE}" "${RESULTS_DIR}" "${BASE}/logs"

# Grid definition
# Note: Do not vary seeds for now; keep the config's seed. Limit connectivity ≤ 0.50.
PROBS=(0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50)

# Prefilled-queue sweep (10 levels) → multiplies datasets by 10
# Controls initial backlog per replica for dnn1 and dnn2
# PREWARM_Q=(0 1 2 3 4 6 8 12 16 24)
PREWARM_Q=(24 16 12 8 6 4 3 2 1 0)

# Replica config matrix: (per_client per_server client_percentage server_percentage) → 10 configs
REPLICA_CFGS=(
  "0 1 0.2 0.5"
  "1 2 0.4 0.8"
  "0 2 0.3 0.6"
  "1 1 0.5 0.7"
  "0 3 0.2 0.4"
  "2 1 0.6 0.9"
  "1 3 0.3 0.7"
  "0 4 0.4 0.5"
  "2 2 0.5 0.8"
  "1 4 0.2 0.6"
)

# Backup original config once
ORIG_BAK="${CFG_PATH}.bak.$(date +%s)"
cp "${CFG_PATH}" "${ORIG_BAK}"

i=0
for connection_probability in "${PROBS[@]}"; do
  for replica_cfg in "${REPLICA_CFGS[@]}"; do
    read -r replicas_per_client replicas_per_server preinit_client_pct preinit_server_pct <<< "${replica_cfg}"
    for prewarm_initial_queue in "${PREWARM_Q[@]}"; do

      DID=$(printf "ds_%04d" "$i")
      OUT_DIR="${OUT_BASE}/${DID}"
      mkdir -p "${OUT_DIR}"

      echo "[${DID}] conn_prob=${connection_probability} replicas_per_client=${replicas_per_client} replicas_per_server=${replicas_per_server} preinit_client_pct=${preinit_client_pct} preinit_server_pct=${preinit_server_pct} prewarm.initial_queue=${prewarm_initial_queue}"

      TMP_CFG="$(mktemp)"
      jq --argjson p "${connection_probability}" \
         --argjson pc "${replicas_per_client}" \
         --argjson ps "${replicas_per_server}" \
         --argjson cp "${preinit_client_pct}" \
         --argjson sp "${preinit_server_pct}" \
         --argjson iq "${prewarm_initial_queue}" \
         '.network.topology.connection_probability=$p
          | .preinit.client_percentage=$cp
          | .preinit.server_percentage=$sp
          | .replicas.dnn1.per_client=$pc
          | .replicas.dnn1.per_server=$ps
          | .replicas.dnn2.per_client=$pc
          | .replicas.dnn2.per_server=$ps
          | .prewarm.dnn1.initial_queue=$iq
          | .prewarm.dnn2.initial_queue=$iq' \
         "${CFG_PATH}" > "${TMP_CFG}"

      # Apply the temporary config to the hardcoded path (sequential to avoid races)
      cp "${TMP_CFG}" "${CFG_PATH}"
      cp "${TMP_CFG}" "${OUT_DIR}/space_with_network.json"
      rm -f "${TMP_CFG}"

      # Run executor (sequential). Let brute-force run fully (no cap)
      start_time=$(date +%s)
      (
        cd "${BASE}" && \
        python -m src.executecosimulation --brute-force \
          > "${OUT_DIR}/run.log" 2>&1
      ) || echo "FAIL ${DID}" | tee -a "${OUT_DIR}/run.log"
      end_time=$(date +%s)
      duration=$((end_time - start_time))

      # Log progress with time
      echo "${DID} $(date '+%Y-%m-%d %H:%M:%S') ${duration}s" >> "${PROGRESS_LOG}"

      # Optionally snapshot current results listing to track which files arrived in this run
      ls -1 "${RESULTS_DIR}" > "${OUT_DIR}/results_index.txt" || true

      i=$((i+1))
      # Stop at 1000 datasets
      if [ "$i" -ge 1000 ]; then
        break 3
      fi
    done
  done
done

# Restore original config
cp "${ORIG_BAK}" "${CFG_PATH}" || true
echo "Done. Datasets in: ${OUT_BASE}"


