#!/usr/bin/env bash
set -euo pipefail

# Generate 100 diverse datasets by varying network topology and replica preinit.
# This script edits the repo's config in-place per dataset (sequentially),
# runs the brute-force executor, and stores the exact config and log per dataset.

BASE="/root/projects/my-herosim"
CFG_PATH="${BASE}/simulation_data/space_with_network.json"
OUT_BASE="${BASE}/simulation_data/gnn_datasets"
RESULTS_DIR="${BASE}/simulation_data/initial_results_simple"
mkdir -p "${OUT_BASE}" "${RESULTS_DIR}"

# Grid definition
SEEDS=(11 23 37 41 53)
PROBS=(0.20 0.35 0.50 0.65 0.80)

# Replica config matrix: (per_client per_server client_percentage server_percentage)
REPLICA_CFGS=(
  "0 1 0.2 0.5"
  "1 2 0.4 0.8"
)

# Backup original config once
ORIG_BAK="${CFG_PATH}.bak.$(date +%s)"
cp "${CFG_PATH}" "${ORIG_BAK}"

i=0
for seed in "${SEEDS[@]}"; do
  for p in "${PROBS[@]}"; do
    for cfg in "${REPLICA_CFGS[@]}"; do
      read -r pc ps cpct spct <<< "${cfg}"

      DID=$(printf "ds_%03d" "$i")
      OUT_DIR="${OUT_BASE}/${DID}"
      mkdir -p "${OUT_DIR}"

      echo "[${DID}] seed=${seed} p=${p} per_client=${pc} per_server=${ps} client_pct=${cpct} server_pct=${spct}"

      TMP_CFG="$(mktemp)"
      jq --argjson seed "${seed}" \
         --argjson p "${p}" \
         --argjson pc "${pc}" \
         --argjson ps "${ps}" \
         --argjson cp "${cpct}" \
         --argjson sp "${spct}" \
         '.network.topology.seed=$seed
          | .network.topology.connection_probability=$p
          | .preinit.client_percentage=$cp
          | .preinit.server_percentage=$sp
          | .replicas.dnn1.per_client=$pc
          | .replicas.dnn1.per_server=$ps
          | .replicas.dnn2.per_client=$pc
          | .replicas.dnn2.per_server=$ps' \
         "${CFG_PATH}" > "${TMP_CFG}"

      # Apply the temporary config to the hardcoded path (sequential to avoid races)
      cp "${TMP_CFG}" "${CFG_PATH}"
      cp "${TMP_CFG}" "${OUT_DIR}/space_with_network.json"
      rm -f "${TMP_CFG}"

      # Run executor (sequential). Adjust the cap as needed.
      (
        cd "${BASE}" && \
        python -m src.executecosimulation --brute-force 50 \
          > "${OUT_DIR}/run.log" 2>&1
      ) || echo "FAIL ${DID}" | tee -a "${OUT_DIR}/run.log"

      # Optionally snapshot current results listing to track which files arrived in this run
      ls -1 "${RESULTS_DIR}" > "${OUT_DIR}/results_index.txt" || true

      i=$((i+1))
    done
  done
done

# Restore original config
cp "${ORIG_BAK}" "${CFG_PATH}" || true
echo "Done. Datasets in: ${OUT_BASE}"


