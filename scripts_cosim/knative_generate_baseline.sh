#!/usr/bin/env bash
set -uo pipefail

# ============================================================================
# Knative Baseline Generation Script
# ============================================================================
# Iterates over existing ds_* directories from the co-simulation and runs
# the knative_network scheduler to capture system state with queue occupancy.
#
# For each dataset with infrastructure.json and workload.json:
#   1. Runs knative_network scheduler simulation
#   2. Saves system_state_captured.json with queue occupancy at scheduling time
#
# Usage:
#   ./knative_generate_baseline.sh [datasets_base_path]
#
# Default datasets_base_path: /root/projects/my-herosim/simulation_data/gnn_datasets
# ============================================================================

BASE="/root/projects/my-herosim"
DEFAULT_DATASETS_BASE="${BASE}/simulation_data/artifacts/run2000/gnn_datasets"
PROGRESS_LOG="${BASE}/logs/knative_baseline_unique_progress.txt"

# Allow override via command line argument
DATASETS_BASE="${1:-${DEFAULT_DATASETS_BASE}}"

mkdir -p "${BASE}/logs"

echo "=== Knative Baseline Generation ==="
echo "Datasets base: ${DATASETS_BASE}"
echo "Progress log: ${PROGRESS_LOG}"
echo ""

# Check if datasets directory exists
if [ ! -d "${DATASETS_BASE}" ]; then
    echo "ERROR: Datasets directory not found: ${DATASETS_BASE}"
    exit 1
fi

# Find all ds_* directories
DS_DIRS=$(find "${DATASETS_BASE}" -maxdepth 1 -type d -name "ds_*" | sort)

if [ -z "${DS_DIRS}" ]; then
    echo "No ds_* directories found in ${DATASETS_BASE}"
    exit 1
fi

# Count total datasets
TOTAL_COUNT=$(echo "${DS_DIRS}" | wc -l)
echo "Found ${TOTAL_COUNT} datasets to process"
echo ""

# Counters
SUCCESS_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0
ALREADY_COUNT=0

# Process each dataset
i=0
for DS_DIR in ${DS_DIRS}; do
    i=$((i+1))
    DID=$(basename "${DS_DIR}")
    
    # Check if already processed (unique version)
    if [ -f "${DS_DIR}/system_state_captured_unique.json" ]; then
        echo "[${i}/${TOTAL_COUNT}] ${DID}: Already processed, skipping"
        ALREADY_COUNT=$((ALREADY_COUNT+1))
        continue
    fi
    
    # Check required files
    INFRA_FILE="${DS_DIR}/infrastructure.json"
    WORKLOAD_FILE="${DS_DIR}/workload.json"
    SPACE_FILE="${DS_DIR}/space_with_network.json"
    
    if [ ! -f "${INFRA_FILE}" ]; then
        echo "[${i}/${TOTAL_COUNT}] ${DID}: Missing infrastructure.json, skipping"
        SKIP_COUNT=$((SKIP_COUNT+1))
        echo "${DID} SKIP $(date '+%Y-%m-%d %H:%M:%S') missing_infrastructure" >> "${PROGRESS_LOG}"
        continue
    fi
    
    if [ ! -f "${WORKLOAD_FILE}" ]; then
        echo "[${i}/${TOTAL_COUNT}] ${DID}: Missing workload.json, skipping"
        SKIP_COUNT=$((SKIP_COUNT+1))
        echo "${DID} SKIP $(date '+%Y-%m-%d %H:%M:%S') missing_workload" >> "${PROGRESS_LOG}"
        continue
    fi
    
    if [ ! -f "${SPACE_FILE}" ]; then
        echo "[${i}/${TOTAL_COUNT}] ${DID}: Missing space_with_network.json, skipping"
        SKIP_COUNT=$((SKIP_COUNT+1))
        echo "${DID} SKIP $(date '+%Y-%m-%d %H:%M:%S') missing_space_config" >> "${PROGRESS_LOG}"
        continue
    fi
    
    echo "[${i}/${TOTAL_COUNT}] ${DID}: Running knative_network simulation..."
    
    start_time=$(date +%s)
    
    # Run the knative baseline simulation
    # Timeout after 300 seconds (5 minutes) per dataset
    cd "${BASE}" && timeout 300 python -m src.executeknativecosim \
        --dataset-dir "${DS_DIR}" \
        2>&1 | tail -5
    
    EXIT_CODE=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    if [ $EXIT_CODE -eq 0 ] && [ -f "${DS_DIR}/system_state_captured_unique.json" ]; then
        # Extract RTT from the captured state
        RTT=$(jq -r '.total_rtt // "N/A"' "${DS_DIR}/system_state_captured_unique.json" 2>/dev/null || echo "N/A")
        echo "[${i}/${TOTAL_COUNT}] ${DID}: SUCCESS (RTT: ${RTT}s, ${duration}s)"
        SUCCESS_COUNT=$((SUCCESS_COUNT+1))
        echo "${DID} SUCCESS $(date '+%Y-%m-%d %H:%M:%S') ${duration}s RTT=${RTT}" >> "${PROGRESS_LOG}"
    elif [ $EXIT_CODE -eq 124 ]; then
        echo "[${i}/${TOTAL_COUNT}] ${DID}: TIMEOUT after 300s"
        FAIL_COUNT=$((FAIL_COUNT+1))
        echo "${DID} TIMEOUT $(date '+%Y-%m-%d %H:%M:%S') ${duration}s" >> "${PROGRESS_LOG}"
    else
        echo "[${i}/${TOTAL_COUNT}] ${DID}: FAILED (exit code: ${EXIT_CODE})"
        FAIL_COUNT=$((FAIL_COUNT+1))
        echo "${DID} FAILED $(date '+%Y-%m-%d %H:%M:%S') ${duration}s exit_code=${EXIT_CODE}" >> "${PROGRESS_LOG}"
    fi
done

echo ""
echo "=== Complete ==="
echo "Total datasets: ${TOTAL_COUNT}"
echo "Already processed: ${ALREADY_COUNT}"
echo "Successfully processed: ${SUCCESS_COUNT}"
echo "Skipped - missing files: ${SKIP_COUNT}"
echo "Failed: ${FAIL_COUNT}"
echo ""
echo "Progress log: ${PROGRESS_LOG}"
