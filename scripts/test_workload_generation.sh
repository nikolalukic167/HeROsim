#!/usr/bin/env bash
set -euo pipefail

# Test script for workload template generation
# This verifies the workload generation logic before running the full dataset generation

BASE="/root/projects/my-herosim"
WORKLOAD_DIR="${BASE}/data/nofs-ids/traces"
WORKLOAD_TEMPLATES_DIR="${BASE}/data/nofs-ids/traces/test_templates"
mkdir -p "${WORKLOAD_TEMPLATES_DIR}"

# Workload generation parameters
NUM_TASKS=5
NUM_CLIENT_NODES=10
TASK_TYPE_RATIOS=("0-100" "20-80" "40-60" "60-40" "80-20" "100-0")

# Generate 10 diverse workload templates
generate_workload_templates() {
  local base_trace="${WORKLOAD_DIR}/workload-10.json"
  
  if [ ! -f "${base_trace}" ]; then
    echo "ERROR: Base workload trace not found: ${base_trace}"
    exit 1
  fi
  
  # Read the base workload to extract template structure
  local rps=$(jq -r '.rps' "${base_trace}")
  local duration=$(jq -r '.duration' "${base_trace}")
  local base_events=$(jq '.events' "${base_trace}")
  
  echo "Generating 10 workload templates with ${NUM_TASKS} tasks each..."
  echo ""
  
  for template_idx in {0..9}; do
    local ratio="${TASK_TYPE_RATIOS[$((template_idx % ${#TASK_TYPE_RATIOS[@]}))]}"
    IFS='-' read -r dnn1_pct dnn2_pct <<< "${ratio}"
    
    local num_dnn1=$((NUM_TASKS * dnn1_pct / 100))
    local num_dnn2=$((NUM_TASKS - num_dnn1))
    
    local workload_file="${WORKLOAD_TEMPLATES_DIR}/workload_template_${template_idx}.json"
    
    # Create array of task types
    local task_types=()
    for ((i=0; i<num_dnn1; i++)); do task_types+=("dnn1"); done
    for ((i=0; i<num_dnn2; i++)); do task_types+=("dnn2"); done
    
    # Generate random client node assignments for each task
    local client_nodes=()
    for ((i=0; i<NUM_TASKS; i++)); do
      client_nodes+=($((RANDOM % NUM_CLIENT_NODES)))
    done
    
    # Use jq to construct the workload JSON with randomized tasks
    jq -n \
      --argjson rps "${rps}" \
      --argjson duration "${duration}" \
      --argjson base_events "${base_events}" \
      --argjson num_tasks "${NUM_TASKS}" \
      '{
        rps: $rps,
        duration: $duration,
        events: [
          range($num_tasks) as $i | $base_events[$i % ($base_events | length)] | 
          . + {timestamp: .timestamp}
        ]
      }' > "${workload_file}.tmp"
    
    # Now replace task types and client nodes using Python for better array handling
    local client_nodes_csv=$(IFS=,; echo "${client_nodes[*]}")
    python3 << EOF
import json

task_types = "${task_types[@]}".split()
client_nodes = [${client_nodes_csv}]

with open("${workload_file}.tmp", 'r') as f:
    workload = json.load(f)

# Update each event with new task type and client node
for idx, event in enumerate(workload['events'][:${NUM_TASKS}]):
    task_type = task_types[idx]
    client_node = client_nodes[idx]
    
    event['application']['name'] = f"nofs-{task_type}"
    event['application']['dag'] = {task_type: []}
    event['node_name'] = f"client_node{client_node}"

# Keep only NUM_TASKS events
workload['events'] = workload['events'][:${NUM_TASKS}]

with open("${workload_file}", 'w') as f:
    json.dump(workload, f, indent=2)
EOF
    
    rm -f "${workload_file}.tmp"
    
    echo "Template ${template_idx}: ${num_dnn1} dnn1 + ${num_dnn2} dnn2"
    echo "  Ratio: ${ratio} | Clients: [${client_nodes[@]}]"
    
    # Validate the generated workload
    if [ -f "${workload_file}" ]; then
      local event_count=$(jq '.events | length' "${workload_file}")
      echo "  ✓ Generated ${event_count} events"
      
      # Show task type distribution
      local actual_dnn1=$(jq '[.events[].application.name] | map(select(. == "nofs-dnn1")) | length' "${workload_file}")
      local actual_dnn2=$(jq '[.events[].application.name] | map(select(. == "nofs-dnn2")) | length' "${workload_file}")
      echo "  ✓ Actual distribution: ${actual_dnn1} dnn1 + ${actual_dnn2} dnn2"
      
      # Show first event as sample
      echo "  Sample event:"
      jq -c '.events[0] | {app: .application.name, node: .node_name}' "${workload_file}"
    else
      echo "  ✗ Failed to generate workload file"
    fi
    echo ""
  done
  
  echo "Workload template generation complete."
  echo "Templates saved to: ${WORKLOAD_TEMPLATES_DIR}"
}

# Run the test
generate_workload_templates

# Show summary statistics
echo ""
echo "=== Summary ==="
for template_idx in {0..9}; do
  workload_file="${WORKLOAD_TEMPLATES_DIR}/workload_template_${template_idx}.json"
  if [ -f "${workload_file}" ]; then
    dnn1_count=$(jq '[.events[].application.name] | map(select(. == "nofs-dnn1")) | length' "${workload_file}")
    dnn2_count=$(jq '[.events[].application.name] | map(select(. == "nofs-dnn2")) | length' "${workload_file}")
    clients=$(jq -r '[.events[].node_name] | unique | join(", ")' "${workload_file}")
    echo "Template $template_idx: ${dnn1_count}×dnn1 ${dnn2_count}×dnn2 | Clients: $clients"
  fi
done

echo ""
echo "Test complete! You can inspect the generated templates in:"
echo "  ${WORKLOAD_TEMPLATES_DIR}"

