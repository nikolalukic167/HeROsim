#!/bin/bash
ds_number="$(ls -Art simulation_data/gnn_datasets | tail -n 1)"

echo "ds: simulation_data/gnn_datasets/${ds_number}/placement_progress.txt"

echo "time: $(date)"

cat simulation_data/gnn_datasets/${ds_number}/placement_progress.txt
