#!/bin/bash

# Define arrays to store the command modules and their corresponding arguments
modules=("src.motivational.reactiveandproactiveparalleldiffworkloads" "src.motivational.reactiveandproactiveparalleldiffworkloadspart2" "src.motivational.reactiveandproactiveparalleldiffworkloadspart2")
# Define the last argument for each module execution
last_args=("" "one_day" "first" "first_second" "first_second_third")

output_prefixes=("R1-1233-output" "R1-1358-output" "R1-1412-output" "R1-1437-output" "R1-1465-output" "R1-2119-output" "R1-351-output" "R1-49-output" "R1-817-output")
config_files=("./data/nofs-ids/workload-configs/R1-1233-9500-20.json" "./data/nofs-ids/workload-configs/R1-1358-9500-20.json" "./data/nofs-ids/workload-configs/R1-1412-9500-20.json" "./data/nofs-ids/workload-configs/R1-1437-9500-20.json" "./data/nofs-ids/workload-configs/R1-1465-9500-20.json" "./data/nofs-ids/workload-configs/R1-2119-9500-20.json" "./data/nofs-ids/workload-configs/R1-351-9500-20.json" "./data/nofs-ids/workload-configs/R1-49-9500-20.json" "./data/nofs-ids/workload-configs/R1-817-9500-20.json")
r1_values=("1233" "1358" "1412" "1437" "1465" "2119" "351" "49" "817")

# Function to run a specific module with all arguments
run_module() {
    local module=$1
    local last_arg=$2
    local max_parallel=4
    local running=0

    # Loop through all the arguments
    for i in "${!output_prefixes[@]}"; do
        # Build the command based on whether we need the last argument
        if [ -z "$last_arg" ]; then
            # First module doesn't need the extra argument
            python -m "$module" "${output_prefixes[$i]}" 1 "${config_files[$i]}" 1 4 R1 "${r1_values[$i]}" &
        else
            # Part2 modules need the extra argument
            python -m "$module" "${output_prefixes[$i]}" 1 "${config_files[$i]}" 1 4 R1 "${r1_values[$i]}" "$last_arg" &
        fi

        # Increment the counter of running processes
        ((running++))

        # If we've reached the maximum number of parallel processes, wait for one to finish
        if [ $running -ge $max_parallel ]; then
            wait -n  # Wait for any child process to exit
            ((running--))
        fi
    done

    # Wait for all remaining processes to finish before starting the next module
    wait
}

# Loop through each module sequentially, but run parallel jobs within each module
for i in "${!modules[@]}"; do
    echo "Running module: ${modules[$i]} with last_arg: ${last_args[$i]}"
    run_module "${modules[$i]}" "${last_args[$i]}"
    echo "Completed module: ${modules[$i]} with last_arg: ${last_args[$i]}"
done
