#!/bin/bash

# Default maximum number of parallel pipelines
MAX_PARALLEL=${1:-3}

# Define the space IDs
spaces=("R1-49" "R1-817" "R1-1233")

# Function to run the optimization pipeline for a space
run_pipeline() {
    local space=$1
    local config_file="data/nofs-ids/workload-configs/${space}-9500-20.json"

    # Generate a unique ID using timestamp and random string
    local unique_id="pipeline-${space}-$(date +%Y%m%d%H%M%S)"

    echo "Starting pipeline for ${space}"

    # Run the four commands in sequence for this space
    python -m src.optimization.generate "spaces/${space}" "${config_file}" 1 nofs-dnn1
    python -m src.optimization.sample "spaces/${space}" 15
    python -m src.optimization.initial "spaces/${space}" 5 nofs-dnn1
    python -m src.optimization.initialproactive "spaces/${space}" 5 nofs-dnn1
    python -m src.optimization.optimization 1 10 5 30 1 "spaces/${space}" "${unique_id}"
    python -m src.optimization.finetune "spaces/${space}" "${unique_id}"
    python -m src.optimization.validate "spaces/${space}" "${unique_id}" "${config_file}" 4

    echo "Completed pipeline for ${space} with ID: ${unique_id}"
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
