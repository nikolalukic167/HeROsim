#!/bin/bash

LOG_FILE="logs_sim/gnn_wait_time.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: $LOG_FILE not found."
    exit 1
fi

awk '
/\[GNN Batch\]/ {
    # Based on your input:
    # [GNN Batch] is $1 and $2
    # Batch is $3
    # size: is $4
    # 3 is $5 (we strip the comma)
    # tasks, is $6
    # wait is $7
    # time: is $8
    # 87.67ms is $9

    # 1. Get batch size (remove the comma if present)
    size = $5;
    gsub(/,/, "", size);
    
    # 2. Get wait time (remove the ms)
    time_str = $9;
    gsub(/ms/, "", time_str);
    wait_time = time_str + 0;

    # 3. Accumulate
    if (size > 0) {
        sum[size] += wait_time;
        count[size]++;
    }
}

END {
    print "-----------------------------------------------";
    printf "%-15s | %-13s | %-10s\n", "Batch Size", "Avg Wait (ms)", "Samples";
    print "-----------------------------------------------";
    
    # Check for sizes 1 through 3
    for (i = 1; i <= 4; i++) {
        if (count[i] > 0) {
            printf "Size: %-9d | %-13.2f | %-10d\n", i, sum[i]/count[i], count[i];
        } else {
            printf "Size: %-9d | %-13s | %-10d\n", i, "0.00", 0;
        }
    }
    print "-----------------------------------------------";
}' "$LOG_FILE"