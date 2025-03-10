#!/bin/bash

# Define arrays to store the command modules and their corresponding arguments
modules=("src.motivational.reactiveandproactiveparalleldiffworkloads" "src.motivational.reactiveandproactiveparalleldiffworkloadspart2")
output_prefixes=("R1-1233-output" "R1-1358-output" "R1-1412-output" "R1-1437-output" "R1-1465-output" "R1-2119-output" "R1-351-output" "R1-49-output" "R1-817-output")
config_files=("./data/nofs-ids/workload-configs/R1-1233-7500-20.json" "./data/nofs-ids/workload-configs/R1-1358-7500-20.json" "./data/nofs-ids/workload-configs/R1-1412-7500-20.json" "./data/nofs-ids/workload-configs/R1-1437-7500-20.json" "./data/nofs-ids/workload-configs/R1-1465-7500-20.json" "./data/nofs-ids/workload-configs/R1-2119-7500-20.json" "./data/nofs-ids/workload-configs/R1-351-7500-20.json" "./data/nofs-ids/workload-configs/R1-49-7500-20.json" "./data/nofs-ids/workload-configs/R1-817-7500-20.json")
r1_values=("1233" "1358" "1412" "1437" "1465" "2119" "351" "49" "817")

# Loop through each module
for module in "${modules[@]}"; do
    # Loop through all the arguments
    for i in "${!output_prefixes[@]}"; do
        python -m "$module" "${output_prefixes[$i]}" 1 "${config_files[$i]}" 1 4 R1 "${r1_values[$i]}"
    done
done
