#!/usr/bin/env bash
#
# Run simulations sequentially: policies (knative, gnn, random, offload) x workloads.
# Sequential to avoid OOM. Safe to run under nohup and log out (keeps running).
#
# Usage:
#   ./scripts_cosim/run_all_sequential.sh
#   nohup ./scripts_cosim/run_all_sequential.sh > run_all_sequential.log 2>&1 &
#
# Results: simulation_data/results/<workload>/simulation_result_<policy>.json
# Logs:    logs_sim/<workload>/<policy>.log
# Master:  run_all_sequential.log (or whatever you redirect to)
#

set -e

BASE_DIR="${BASE_DIR:-/root/projects/my-herosim}"
cd "$BASE_DIR" || exit 1

CONFIG="$BASE_DIR/simulation_data/space_with_network.json"
TRACES_DIR="$BASE_DIR/data/nofs-ids/traces"
RESULTS_DIR="$BASE_DIR/simulation_data/results"
LOGS_DIR="$BASE_DIR/logs_sim"
TIMEOUT="${TIMEOUT:-3600}"
SEED="${SEED:-42}"

WORKLOADS="100-100"
# policy flag for run_simulation.py -> output filename stem
POLICIES="offload_network:offload"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" ; }

# Create dirs
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"
for w in $WORKLOADS; do
  mkdir -p "$RESULTS_DIR/$w" "$LOGS_DIR/$w"
done

log "Starting 9 sequential runs (workloads: $WORKLOADS; timeout=${TIMEOUT}s; seed=$SEED)"
log "Results: $RESULTS_DIR/<workload>/simulation_result_<policy>.json"
log "Logs:    $LOGS_DIR/<workload>/<policy>.log"
log ""

run_count=0
for workload in $WORKLOADS; do
  workload_file="$TRACES_DIR/workload-${workload}.json"
  if [[ ! -f "$workload_file" ]]; then
    log "SKIP workload $workload: $workload_file not found"
    continue
  fi
  for entry in $POLICIES; do
    policy="${entry%%:*}"
    short="${entry##*:}"
    run_count=$((run_count + 1))
    log "--- Run $run_count/9: workload=$workload policy=$policy ---"
    out_file="$RESULTS_DIR/$workload/simulation_result_${policy}.json"
    run_log="$LOGS_DIR/$workload/${short}.log"
    case "$policy" in
      knative_network)  flag="--knative_network" ;;
      gnn)              flag="--gnn" ;;
      random_network)   flag="--random_network" ;;
      offload_network)  flag="--offload_network" ;;
      *)                log "Unknown policy $policy"; exit 1 ;;
    esac
    if pipenv run python scripts_cosim/run_simulation.py \
      $flag \
      --timeout "$TIMEOUT" \
      --seed "$SEED" \
      --workload "$workload_file" \
      --output "$out_file" \
      > "$run_log" 2>&1 ; then
      log "OK $workload $policy -> $out_file (log: $run_log)"
    else
      code=$?
      log "FAILED $workload $policy (exit $code). Log: $run_log"
      # Continue to next run so one OOM/failure doesn't stop the rest
    fi
    # Let the process fully exit and OS reclaim memory before next run
    sleep 2
  done
done

log ""
log "Done. $run_count runs completed. Check $RESULTS_DIR and $LOGS_DIR."
