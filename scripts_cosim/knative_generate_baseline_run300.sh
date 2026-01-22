#!/usr/bin/env bash
set -uo pipefail

# ============================================================================
# Knative Baseline Generation Script for run300
# ============================================================================
# Safely runs knative baseline generation on run300 datasets without
# interfering with the running co-simulation (generate_gnn_datasets_fast.py).
#
# Safety features:
# 1. Checks for file locks before processing
# 2. Uses separate progress log file (run300 specific)
# 3. Processes datasets sequentially to avoid resource contention
# 4. Optionally runs determinism test to verify results
#
# Usage:
#   ./knative_generate_baseline_run300.sh [--test-determinism] [--test-runs N]
#
# Options:
#   --test-determinism: Run determinism test after each successful capture
#   --test-runs N: Number of runs for determinism test (default: 3)
# ============================================================================

BASE="/root/projects/my-herosim"
DATASETS_BASE="${BASE}/simulation_data/artifacts/run300/gnn_datasets"
PROGRESS_LOG="${BASE}/logs/knative_baseline_run300_progress.txt"

# Parse arguments
TEST_DETERMINISM=false
TEST_RUNS=3
while [[ $# -gt 0 ]]; do
    case $1 in
        --test-determinism)
            TEST_DETERMINISM=true
            shift
            ;;
        --test-runs)
            TEST_RUNS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--test-determinism] [--test-runs N]"
            exit 1
            ;;
    esac
done

# No arguments needed - processes all ready datasets

mkdir -p "${BASE}/logs"

echo "=== Knative Baseline Generation for run300 ==="
echo "Datasets base: ${DATASETS_BASE}"
echo "Progress log: ${PROGRESS_LOG}"
if [ "$TEST_DETERMINISM" = true ]; then
    echo "Determinism testing: ENABLED (${TEST_RUNS} runs per dataset)"
fi
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
echo "Found ${TOTAL_COUNT} datasets to check"
echo ""

# Counters
SUCCESS_COUNT=0
SKIP_COUNT=0
SKIP_IN_PROGRESS_COUNT=0
FAIL_COUNT=0
ALREADY_COUNT=0
NOT_READY_COUNT=0

# Process each dataset
i=0
for DS_DIR in ${DS_DIRS}; do
    i=$((i+1))
    DID=$(basename "${DS_DIR}")
    
    # Skip if already processed
    if [ -f "${DS_DIR}/system_state_captured_unique.json" ]; then
        ALREADY_COUNT=$((ALREADY_COUNT+1))
        continue
    fi
    
    # Check required files
    if [ ! -f "${DS_DIR}/infrastructure.json" ] || \
       [ ! -f "${DS_DIR}/workload.json" ] || \
       [ ! -f "${DS_DIR}/space_with_network.json" ]; then
        NOT_READY_COUNT=$((NOT_READY_COUNT+1))
        continue
    fi
    
    echo "[${i}/${TOTAL_COUNT}] ${DID}: Running knative_network simulation..."
    
    start_time=$(date +%s)
    
    # Timeout after 60 seconds (1 minute) per dataset
    cd "${BASE}" && timeout 60 python3 -m src.executeknativecosim \
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
        
        # Optional determinism test
        if [ "$TEST_DETERMINISM" = true ]; then
            echo "  Running determinism test (${TEST_RUNS} runs)..."
            cd "${BASE}" && python3 -m src.executeknativecosim_determinism_test \
                "${DS_DIR}" "${TEST_RUNS}" 2>&1 | grep -E "(deterministic|SIMULATION IS)" | head -2 || true
        fi
    elif [ $EXIT_CODE -eq 124 ]; then
        echo "[${i}/${TOTAL_COUNT}] ${DID}: TIMEOUT after 60s"
        FAIL_COUNT=$((FAIL_COUNT+1))
        echo "${DID} TIMEOUT $(date '+%Y-%m-%d %H:%M:%S') ${duration}s" >> "${PROGRESS_LOG}"
    else
        echo "[${i}/${TOTAL_COUNT}] ${DID}: FAILED (exit code: ${EXIT_CODE})"
        FAIL_COUNT=$((FAIL_COUNT+1))
        echo "${DID} FAILED $(date '+%Y-%m-%d %H:%M:%S') ${duration}s exit_code=${EXIT_CODE}" >> "${PROGRESS_LOG}"
    fi
    
    # Small delay to avoid hammering the system
    sleep 0.5
done

echo ""
echo ""
echo "=== Complete ==="
echo "Total datasets: ${TOTAL_COUNT}"
echo "Already processed: ${ALREADY_COUNT}"
echo "Successfully processed: ${SUCCESS_COUNT}"
echo "Skipped - in progress (files open): ${SKIP_IN_PROGRESS_COUNT}"
echo "Skipped - not ready (missing/incomplete files): ${NOT_READY_COUNT}"
echo "Failed: ${FAIL_COUNT}"
echo ""
echo "Progress log: ${PROGRESS_LOG}"
