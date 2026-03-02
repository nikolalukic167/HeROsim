#!/usr/bin/env python3
"""
Simulation Runner Script with Policy Selection (Python Wrapper)

Runs simulations with different policies (gnn, knative_network, herocache_network, etc.)
for real simulation (full workload, no warmup tasks, autoscaling from zero).

Usage:
    python scripts_cosim/run_simulation.py --gnn [--timeout N] [--seed N]
    python scripts_cosim/run_simulation.py --roundrobin [--timeout N] [--seed N]
    python scripts_cosim/run_simulation.py --knative_network [--timeout N] [--seed N]
    python scripts_cosim/run_simulation.py --herocache_network [--timeout N] [--seed N]
    python scripts_cosim/run_simulation.py --random_network [--timeout N] [--seed N]

Options:
    --gnn             Run with vanilla gnn policy (gnn_gnn)
    --roundrobin      Run with roundrobin network policy (rr_network_rr_network)
    --knative_network Run with knative network policy (kn_network_kn_network)
    --herocache_network Run with herocache network policy (hrc_network_hrc_network)
    --random_network Run with random network-aware policy (rp_network_rp_network)
    --timeout N       Timeout in seconds (default: 3600)
    --seed N          Random seed for deterministic network topology (optional)

Files used:
    Config: simulation_data/space_with_network.json
    Workload: data/nofs-ids/traces/workload-xy-xy.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Tuple


# Constants
BASE_DIR = Path("/root/projects/my-herosim")
CONFIG_FILE = BASE_DIR / "simulation_data/space_with_network.json"
WORKLOAD_FILE = BASE_DIR / "data/nofs-ids/traces/workload-200-200.json"
OUTPUT_DIR = BASE_DIR / "simulation_data/results"
DEFAULT_TIMEOUT = 3600

# Policy configuration mapping
POLICY_CONFIG: Dict[str, Dict[str, str]] = {
    "gnn": {
        "progress_log": BASE_DIR / "logs/gnn_simulation_progress.txt",
        "policy_name": "vanilla gnn",
        "scheduling_strategy": "gnn_gnn",
        "output_file": OUTPUT_DIR / "simulation_result_gnn.json",
    },
    "roundrobin": {
        "progress_log": BASE_DIR / "logs/roundrobin_simulation_progress.txt",
        "policy_name": "roundrobin network",
        "scheduling_strategy": "rr_network_rr_network",
        "output_file": OUTPUT_DIR / "simulation_result_roundrobin.json",
    },
    "knative_network": {
        "progress_log": BASE_DIR / "logs/knative_network_simulation_progress.txt",
        "policy_name": "knative network",
        "scheduling_strategy": "kn_network_kn_network",
        "output_file": OUTPUT_DIR / "simulation_result_knative_network.json",
    },
    "herocache_network": {
        "progress_log": BASE_DIR / "logs/herocache_network_simulation_progress.txt",
        "policy_name": "herocache network",
        "scheduling_strategy": "hrc_network_hrc_network",
        "output_file": OUTPUT_DIR / "simulation_result_herocache_network.json",
    },
    "random_network": {
        "progress_log": BASE_DIR / "logs/random_network_simulation_progress.txt",
        "policy_name": "random network",
        "scheduling_strategy": "rp_network_rp_network",
        "output_file": OUTPUT_DIR / "simulation_result_random_network.json",
    },
    "offload_network": {
        "progress_log": BASE_DIR / "logs/offload_network_simulation_progress.txt",
        "policy_name": "offload network",
        "scheduling_strategy": "offload_network_offload_network",
        "output_file": OUTPUT_DIR / "simulation_result_offload_network.json",
    },
}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run simulation with different scheduling policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Policy selection (mutually exclusive group)
    policy_group = parser.add_mutually_exclusive_group(required=True)
    policy_group.add_argument("--gnn", action="store_const", const="gnn", dest="policy",
                             help="Run with vanilla gnn policy")
    policy_group.add_argument("--roundrobin", action="store_const", const="roundrobin", dest="policy",
                             help="Run with roundrobin network policy")
    policy_group.add_argument("--knative_network", action="store_const", const="knative_network", dest="policy",
                             help="Run with knative network policy")
    policy_group.add_argument("--herocache_network", action="store_const", const="herocache_network", dest="policy",
                             help="Run with herocache network policy")
    policy_group.add_argument("--random_network", action="store_const", const="random_network", dest="policy",
                             help="Run with random network-aware policy")
    policy_group.add_argument("--offload_network", action="store_const", const="offload_network", dest="policy",
                             help="Run with offload-to-server network-aware policy")
    
    # Optional arguments
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                       help=f"Timeout in seconds (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for deterministic network topology")
    parser.add_argument("--workload", type=str, default=None,
                       help="Path to workload JSON (default: workload-200-200.json)")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to result JSON (default: simulation_data/results/simulation_result_<policy>.json)")
    
    return parser.parse_args()


def validate_files(config_file: Path, workload_file: Path) -> None:
    """Validate that required files exist."""
    if not config_file.exists():
        print(f"ERROR: Config file not found: {config_file}", file=sys.stderr)
        sys.exit(1)
    
    if not workload_file.exists():
        print(f"ERROR: Workload file not found: {workload_file}", file=sys.stderr)
        sys.exit(1)


def extract_rtt(output_file: Path) -> Optional[float]:
    """Extract RTT from simulation result JSON file."""
    try:
        with open(output_file, 'r') as f:
            result = json.load(f)
            return result.get('total_rtt')
    except (json.JSONDecodeError, IOError, KeyError):
        return None


def run_simulation(
    policy: str,
    config_file: Path,
    workload_file: Path,
    output_file: Path,
    timeout: int,
    seed: Optional[int] = None
) -> Tuple[int, float]:
    """
    Run the simulation and return exit code and duration.
    
    Returns:
        Tuple of (exit_code, duration_seconds)
    """
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        "pipenv", "run", "python", "-u", "-m", "src.executesimulation",
        "--config", str(config_file),
        "--workload", str(workload_file),
        "--policy", policy,
        "--output", str(output_file),
    ]
    
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    
    # Set environment
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    # Run simulation with timeout
    # Note: Not specifying stdout/stderr allows proper redirection with > or >>
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            env=env,
            timeout=timeout,
            check=False,
        )
        exit_code = result.returncode
    except subprocess.TimeoutExpired:
        exit_code = 124  # Timeout exit code (matches bash timeout)
    finally:
        duration = time.time() - start_time
    
    return exit_code, duration


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Get policy configuration
    if args.policy not in POLICY_CONFIG:
        print(f"ERROR: Unknown policy: {args.policy}", file=sys.stderr)
        sys.exit(1)
    
    config = POLICY_CONFIG[args.policy]
    progress_log = config["progress_log"]
    policy_name = config["policy_name"]
    scheduling_strategy = config["scheduling_strategy"]
    output_file = Path(config["output_file"]) if args.output is None else Path(args.output)
    workload_file = WORKLOAD_FILE if args.workload is None else Path(args.workload)
    
    # Create necessary directories
    (BASE_DIR / "logs").mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Print configuration (flush so nohup logs show header first, not after child output)
    def log(msg: str) -> None:
        print(msg, flush=True)
    log(f"=== Simulation Runner: {policy_name} ===")
    log(f"Config file: {CONFIG_FILE}")
    log(f"Workload file: {workload_file}")
    log(f"Output file: {output_file}")
    log(f"Scheduling strategy: {scheduling_strategy}")
    log(f"Timeout: {args.timeout}s")
    if args.seed is not None:
        log(f"Seed: {args.seed}")
    log(f"Progress log: {progress_log}")
    log("")
    validate_files(CONFIG_FILE, workload_file)
    log("Starting simulation...")
    exit_code, duration = run_simulation(
        policy=args.policy,
        config_file=CONFIG_FILE,
        workload_file=workload_file,
        output_file=output_file,
        timeout=args.timeout,
        seed=args.seed,
    )
    
    # Handle results
    if exit_code == 0 and output_file.exists():
        rtt = extract_rtt(output_file)
        rtt_str = f"{rtt}s" if rtt is not None else "N/A"
        log("")
        log("=== SUCCESS ===")
        log(f"Duration: {duration:.1f}s")
        log(f"Total RTT: {rtt_str}")
        log(f"Output file: {output_file}")
        sys.exit(0)
    elif exit_code == 124:
        log("")
        log("=== TIMEOUT ===")
        log(f"Simulation timed out after {args.timeout}s")
        sys.exit(1)
    else:
        log("")
        log("=== FAILED ===")
        log(f"Exit code: {exit_code}")
        if exit_code == -9:
            log("(Exit -9 usually means SIGKILL, e.g. out-of-memory killer.)")
        sys.exit(1)


if __name__ == "__main__":
    main()
