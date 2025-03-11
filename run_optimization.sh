#!/bin/bash

# Default maximum number of parallel pipelines
MAX_PARALLEL=${1:-2}

# Define the space IDs
spaces=("R1-49" "R1-817" "R1-1233" "R1-1358" "R1-1412" "R1-1437" "R1-1465" "R1-2119" "R1-351")

# Function to run the optimization pipeline for a space
run_pipeline() {
    local space=$1
    local config_file="data/nofs-ids/workload-configs/${space}-7500-20.json"

    echo "Starting pipeline for ${space}"

    # Run the four commands in sequence for this space
    python -m src.optimization.generate "spaces/${space}" "${config_file}" 1 nofs-dnn1
    python -m src.optimization.sample "spaces/${space}" 15
    python -m src.optimization.initial "spaces/${space}" 4 nofs-dnn1
    python -m src.optimization.optimization 1 25 5 90 1 "spaces/${space}"

    echo "Completed pipeline for ${space}"
}

# Track running processes
running=0

# Run the pipelines with a limit on parallelism
for space in "${spaces[@]}"; do
    # If we've reached the maximum number of parallel processes, wait for one to finish
    if [ $running -ge $MAX_PARALLEL ]; then
        wait -n  # Wait for any child process to exit
        ((running--))
    fi

    # Start a new pipeline
    run_pipeline "${space}" &
    ((running++))

    echo "Started pipeline for ${space} (${running}/${MAX_PARALLEL} slots used)"
done

# Wait for all remaining processes to complete
wait

echo "All optimization pipelines completed"
