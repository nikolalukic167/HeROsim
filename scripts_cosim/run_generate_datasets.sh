#!/bin/bash
# Script to generate GNN datasets for both 3-task and 2-task configurations
# Based on improved co-simulation settings
#
# Usage:
#   ./run_generate_datasets.sh [max_datasets] [quiet_flag]
#   nohup ./run_generate_datasets.sh 10000 --quiet > /dev/null 2>&1 &

set -e

PROJECT_ROOT="/root/projects/my-herosim"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "GNN Dataset Generation (Improved Co-Sim)"
echo "=========================================="
echo ""

# Configuration
MAX_DATASETS=${1:-10000}  # Default to 10000 if not provided
QUIET_FLAG=${2:-"--quiet"}  # Default to quiet mode

echo "Configuration:"
echo "  Max datasets per task count: $MAX_DATASETS"
echo "  Quiet mode: $QUIET_FLAG"
echo "  Started at: $(date)"
echo ""

# Step 1: Generate 3-task datasets (run first)
echo "=========================================="
echo "Step 1: Generating 3-task datasets"
echo "=========================================="
echo ""

pipenv run python scripts_cosim/generate_gnn_datasets_fast.py \
    --num-tasks 3 \
    --max-datasets "$MAX_DATASETS" \
    $QUIET_FLAG

echo ""
echo "✓ 3-task datasets generation complete at $(date)"
echo ""

# Step 2: Generate 2-task datasets (run second)
echo "=========================================="
echo "Step 2: Generating 2-task datasets"
echo "=========================================="
echo ""

pipenv run python scripts_cosim/generate_gnn_datasets_fast.py \
    --num-tasks 2 \
    --max-datasets "$MAX_DATASETS" \
    $QUIET_FLAG

echo ""
echo "✓ 2-task datasets generation complete at $(date)"
echo ""

echo "=========================================="
echo "All datasets generated successfully!"
echo "Completed at: $(date)"
echo "=========================================="
echo ""
echo "Output directories:"
echo "  - simulation_data/gnn_datasets_3tasks/"
echo "  - simulation_data/gnn_datasets_2tasks/"
echo ""
echo "Progress logs:"
echo "  - logs/progress_3tasks.txt"
echo "  - logs/progress_2tasks.txt"
echo ""
