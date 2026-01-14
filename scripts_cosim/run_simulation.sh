#!/usr/bin/env bash
set -uo pipefail

# ============================================================================
# Simulation Runner Script with Policy Selection
# ============================================================================
# Runs simulations with different policies (vanilla knative or vanilla gnn)
# for real simulation (full workload, no warmup tasks, autoscaling from zero).
#
# Usage:
#   ./scripts_cosim/run_simulation.sh --knative [--timeout N] [--seed N]
#   ./scripts_cosim/run_simulation.sh --gnn [--timeout N] [--seed N]
#   ./scripts_cosim/run_simulation.sh --roundrobin [--timeout N] [--seed N]
#   ./scripts_cosim/run_simulation.sh --knative_no_batch [--timeout N] [--seed N]
#   ./scripts_cosim/run_simulation.sh --multiloop [--timeout N] [--seed N]
#
# Options:
#   --knative         Run with vanilla knative policy (kn_network_kn_network)
#   --gnn             Run with vanilla gnn policy (gnn_gnn)
#   --roundrobin      Run with roundrobin network policy (rr_network_rr_network)
#   --knative_no_batch Run with knative no batch network policy (kn_network_no_batch_kn_network_no_batch)
#   --multiloop       Run with multiloop policy (multiloop_multiloop)
#   --timeout N       Timeout in seconds (default: 3600)
#   --seed N          Random seed for deterministic network topology (optional)
#
# Files used:
#   Config: simulation_data/space_with_network.json
#   Workload: data/nofs-ids/traces/workload-50-50.json
# ============================================================================

BASE="/root/projects/my-herosim"
CONFIG_FILE="${BASE}/simulation_data/space_with_network.json"
WORKLOAD_FILE="${BASE}/data/nofs-ids/traces/workload-5-5.json"
OUTPUT_DIR="${BASE}/simulation_data/results"
DEFAULT_TIMEOUT=3600

# Parse arguments
POLICY=""
TIMEOUT="${DEFAULT_TIMEOUT}"
SEED=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --knative)
            POLICY="knative"
            shift
            ;;
        --gnn)
            POLICY="gnn"
            shift
            ;;
        --roundrobin)
            POLICY="roundrobin"
            shift
            ;;
        --knative_no_batch)
            POLICY="knative_no_batch"
            shift
            ;;
        --multiloop)
            POLICY="multiloop"
            shift
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate policy selection
if [ -z "$POLICY" ]; then
    echo "ERROR: Must specify either --knative, --gnn, --roundrobin, --knative_no_batch, or --multiloop"
    echo ""
    echo "Usage:"
    echo "  $0 --knative [--timeout N] [--seed N]"
    echo "  $0 --gnn [--timeout N] [--seed N]"
    echo "  $0 --roundrobin [--timeout N] [--seed N]"
    echo "  $0 --knative_no_batch [--timeout N] [--seed N]"
    echo "  $0 --multiloop [--timeout N] [--seed N]"
    exit 1
fi

# Set policy-specific variables
if [ "$POLICY" = "knative" ]; then
    PROGRESS_LOG="${BASE}/logs/knative_simulation_progress.txt"
    POLICY_NAME="vanilla knative"
    SCHEDULING_STRATEGY="kn_kn"
    OUTPUT_FILE="${OUTPUT_DIR}/simulation_result_knative.json"
elif [ "$POLICY" = "gnn" ]; then
    PROGRESS_LOG="${BASE}/logs/gnn_simulation_progress.txt"
    POLICY_NAME="vanilla gnn"
    SCHEDULING_STRATEGY="gnn_gnn"
    OUTPUT_FILE="${OUTPUT_DIR}/simulation_result_gnn.json"
elif [ "$POLICY" = "roundrobin" ]; then
    PROGRESS_LOG="${BASE}/logs/roundrobin_simulation_progress.txt"
    POLICY_NAME="roundrobin network"
    SCHEDULING_STRATEGY="rr_network_rr_network"
    OUTPUT_FILE="${OUTPUT_DIR}/simulation_result_roundrobin.json"
elif [ "$POLICY" = "knative_no_batch" ]; then
    PROGRESS_LOG="${BASE}/logs/knative_no_batch_simulation_progress.txt"
    POLICY_NAME="knative no batch network"
    SCHEDULING_STRATEGY="kn_network_no_batch_kn_network_no_batch"
    OUTPUT_FILE="${OUTPUT_DIR}/simulation_result_knative_no_batch.json"
elif [ "$POLICY" = "multiloop" ]; then
    PROGRESS_LOG="${BASE}/logs/multiloop_simulation_progress.txt"
    POLICY_NAME="multiloop"
    SCHEDULING_STRATEGY="multiloop_multiloop"
    OUTPUT_FILE="${OUTPUT_DIR}/simulation_result_multiloop.json"
fi

mkdir -p "${BASE}/logs" "${OUTPUT_DIR}"

echo "=== Simulation Runner: ${POLICY_NAME} ==="
echo "Config file: ${CONFIG_FILE}"
echo "Workload file: ${WORKLOAD_FILE}"
echo "Output file: ${OUTPUT_FILE}"
echo "Scheduling strategy: ${SCHEDULING_STRATEGY}"
echo "Timeout: ${TIMEOUT}s"
if [ -n "${SEED}" ]; then
    echo "Seed: ${SEED}"
fi
echo "Progress log: ${PROGRESS_LOG}"
echo ""

# Check if required files exist
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

if [ ! -f "${WORKLOAD_FILE}" ]; then
    echo "ERROR: Workload file not found: ${WORKLOAD_FILE}"
    exit 1
fi

echo "Starting simulation..."
start_time=$(date +%s)

# Build command
CMD="cd '${BASE}' && timeout '${TIMEOUT}' pipenv run python -m src.executesimulation \
    --config '${CONFIG_FILE}' \
    --workload '${WORKLOAD_FILE}' \
    --policy '${POLICY}' \
    --output '${OUTPUT_FILE}'"

if [ -n "${SEED}" ]; then
    CMD="${CMD} --seed '${SEED}'"
fi

# Run the simulation
eval "${CMD}" 2>&1

EXIT_CODE=$?
end_time=$(date +%s)
duration=$((end_time - start_time))

if [ $EXIT_CODE -eq 0 ] && [ -f "${OUTPUT_FILE}" ]; then
    # Extract RTT from the result if available
    RTT=$(jq -r '.total_rtt // "N/A"' "${OUTPUT_FILE}" 2>/dev/null || echo "N/A")
    echo ""
    echo "=== SUCCESS ==="
    echo "Duration: ${duration}s"
    echo "Total RTT: ${RTT}s"
    echo "Output file: ${OUTPUT_FILE}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') SUCCESS ${duration}s RTT=${RTT}" >> "${PROGRESS_LOG}"
    exit 0
elif [ $EXIT_CODE -eq 124 ]; then
    echo ""
    echo "=== TIMEOUT ==="
    echo "Simulation timed out after ${TIMEOUT}s"
    echo "$(date '+%Y-%m-%d %H:%M:%S') TIMEOUT ${duration}s" >> "${PROGRESS_LOG}"
    exit 1
else
    echo ""
    echo "=== FAILED ==="
    echo "Exit code: ${EXIT_CODE}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') FAILED ${duration}s exit_code=${EXIT_CODE}" >> "${PROGRESS_LOG}"
    exit 1
fi

