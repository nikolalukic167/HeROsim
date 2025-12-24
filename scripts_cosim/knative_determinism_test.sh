#!/usr/bin/env bash
set -uo pipefail

# ============================================================================
# Knative Determinism Test Script
# ============================================================================
# Tests if the knative_network simulation is deterministic by running
# the same dataset multiple times and comparing outputs.
#
# Usage:
#   ./knative_determinism_test.sh [dataset_dir] [num_runs]
#
# Default: Tests ds_00014 with 5 runs
# ============================================================================

BASE="/root/projects/my-herosim"
DEFAULT_DATASET="${BASE}/simulation_data/artifacts/run1650/gnn_datasets/ds_00014"
DEFAULT_RUNS=5

# Parse arguments
DATASET_DIR="${1:-${DEFAULT_DATASET}}"
NUM_RUNS="${2:-${DEFAULT_RUNS}}"

# Output directory (separate from main baseline to avoid interference)
OUTPUT_DIR="${BASE}/logs/determinism_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR}"

echo "=== Knative Determinism Test ==="
echo "Dataset: ${DATASET_DIR}"
echo "Number of runs: ${NUM_RUNS}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Check if dataset exists
if [ ! -d "${DATASET_DIR}" ]; then
    echo "ERROR: Dataset directory not found: ${DATASET_DIR}"
    exit 1
fi

# Check required files
for FILE in infrastructure.json workload.json space_with_network.json; do
    if [ ! -f "${DATASET_DIR}/${FILE}" ]; then
        echo "ERROR: Missing ${FILE}"
        exit 1
    fi
done

cd "${BASE}"

# Run the Python determinism test
echo "Running determinism test..."
echo ""

pipenv run python3 -m src.executeknativecosim_determinism_test \
    "${DATASET_DIR}" \
    "${NUM_RUNS}" \
    2>&1 | tee "${OUTPUT_DIR}/test_output.txt"

EXIT_CODE=$?

echo ""
echo "Output saved to: ${OUTPUT_DIR}/test_output.txt"

exit $EXIT_CODE

