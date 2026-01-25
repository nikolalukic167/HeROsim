#!/bin/bash
# Generate GNN baseline placements for all datasets in a run directory
#
# Usage:
#   ./scripts_cosim/gnn_generate_baseline.sh [run_dir]
#
# Example:
#   ./scripts_cosim/gnn_generate_baseline.sh simulation_data/artifacts/run2000/gnn_datasets
#
# This script runs the GNN scheduler on each ds_* dataset and saves
# system_state_gnn.json with the GNN's placement decisions.
# Note: Datasets without system_state_captured_unique.json (knative baseline)
# will be automatically skipped.
# Each dataset has a 30s timeout - if it doesn't complete, it's skipped.

# Default run directory
RUN_DIR="${1:-simulation_data/artifacts/run2000/gnn_datasets}"

# Check if directory exists
if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: Directory not found: $RUN_DIR"
    exit 1
fi

# Count datasets
NUM_DATASETS=$(find "$RUN_DIR" -maxdepth 1 -type d -name 'ds_*' | wc -l)
echo "Found $NUM_DATASETS datasets in $RUN_DIR"

# Check if model exists
MODEL_PATH="src/notebooks/new/best_gnn_regret_model.pt"
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: GNN model not found at $MODEL_PATH"
    echo "Please train the model first using src/notebooks/new/23-12-17-52_regret.py"
    exit 1
fi

echo "Using GNN model: $MODEL_PATH"
echo ""
echo "Starting GNN baseline generation..."
echo "=================================================="

# Track progress
PROGRESS_FILE="$RUN_DIR/gnn_progress.log"
echo "Progress log: $PROGRESS_FILE"
echo ""

# Run the executor with 30s timeout per dataset
# Model is loaded once, then all datasets are processed with timeout
START_TIME=$(date +%s)

pipenv run python -m src.executegnncosim --datasets-base "$RUN_DIR" --timeout 30 2>&1 | tee "$PROGRESS_FILE"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=================================================="
echo "GNN baseline generation complete!"
echo "Total time: ${ELAPSED}s"
echo ""

# Count successful outputs
NUM_GNN_FILES=$(find "$RUN_DIR" -name 'system_state_gnn.json' | wc -l)
echo "Generated $NUM_GNN_FILES system_state_gnn.json files"
