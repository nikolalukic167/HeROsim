#!/usr/bin/env python3
"""
Simulation Runner Script with Policy Selection (Python Wrapper)

Runs simulations with different policies (vanilla knative or vanilla gnn)
for real simulation (full workload, no warmup tasks, autoscaling from zero).

Usage:
    python scripts_cosim/run_simulation.py --knative [--timeout N] [--seed N]
    python scripts_cosim/run_simulation.py --gnn [--timeout N] [--seed N]
    python scripts_cosim/run_simulation.py --roundrobin [--timeout N] [--seed N]
    python scripts_cosim/run_simulation.py --knative_network [--timeout N] [--seed N]
    python scripts_cosim/run_simulation.py --herocache_network [--timeout N] [--seed N]
    python scripts_cosim/run_simulation.py --herocache_network_batch [--timeout N] [--seed N]

Options:
    --knative         Run with vanilla knative policy (kn_network_kn_network)
    --gnn             Run with vanilla gnn policy (gnn_gnn)
    --roundrobin      Run with roundrobin network policy (rr_network_rr_network)
    --knative_network Run with knative network policy (no batching) (kn_network_kn_network)
    --herocache_network Run with herocache network policy (hrc_network_hrc_network)
    --herocache_network_batch Run with herocache network batch policy (hrc_network_batch_hrc_network_batch)
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
WORKLOAD_FILE = BASE_DIR / "data/nofs-ids/traces/workload-35-35.json"
OUTPUT_DIR = BASE_DIR / "simulation_data/results"
DEFAULT_TIMEOUT = 3600

# Policy configuration mapping
POLICY_CONFIG: Dict[str, Dict[str, str]] = {
    "knative": {
        "progress_log": BASE_DIR / "logs/knative_simulation_progress.txt",
        "policy_name": "vanilla knative",
        "scheduling_strategy": "kn_kn",
        "output_file": OUTPUT_DIR / "simulation_result_knative.json",
    },
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
    "herocache_network_batch": {
        "progress_log": BASE_DIR / "logs/herocache_network_batch_simulation_progress.txt",
        "policy_name": "herocache network batch",
        "scheduling_strategy": "hrc_network_batch_hrc_network_batch",
        "output_file": OUTPUT_DIR / "simulation_result_herocache_network_batch.json",
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
    policy_group.add_argument("--knative", action="store_const", const="knative", dest="policy",
                             help="Run with vanilla knative policy")
    policy_group.add_argument("--gnn", action="store_const", const="gnn", dest="policy",
                             help="Run with vanilla gnn policy")
    policy_group.add_argument("--roundrobin", action="store_const", const="roundrobin", dest="policy",
                             help="Run with roundrobin network policy")
    policy_group.add_argument("--knative_network", action="store_const", const="knative_network", dest="policy",
                             help="Run with knative network policy (no batching)")
    policy_group.add_argument("--herocache_network", action="store_const", const="herocache_network", dest="policy",
                             help="Run with herocache network policy")
    policy_group.add_argument("--herocache_network_batch", action="store_const", const="herocache_network_batch", dest="policy",
                             help="Run with herocache network batch policy")
    
    # Optional arguments
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                       help=f"Timeout in seconds (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for deterministic network topology")
    
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
    output_file = config["output_file"]
    
    # Create necessary directories
    (BASE_DIR / "logs").mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print(f"=== Simulation Runner: {policy_name} ===")
    print(f"Config file: {CONFIG_FILE}")
    print(f"Workload file: {WORKLOAD_FILE}")
    print(f"Output file: {output_file}")
    print(f"Scheduling strategy: {scheduling_strategy}")
    print(f"Timeout: {args.timeout}s")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print(f"Progress log: {progress_log}")
    print()
    
    # Validate files
    validate_files(CONFIG_FILE, WORKLOAD_FILE)
    
    # Run simulation
    print("Starting simulation...")
    exit_code, duration = run_simulation(
        policy=args.policy,
        config_file=CONFIG_FILE,
        workload_file=WORKLOAD_FILE,
        output_file=output_file,
        timeout=args.timeout,
        seed=args.seed,
    )
    
    # Handle results
    if exit_code == 0 and output_file.exists():
        rtt = extract_rtt(output_file)
        rtt_str = f"{rtt}s" if rtt is not None else "N/A"
        print()
        print("=== SUCCESS ===")
        print(f"Duration: {duration:.1f}s")
        print(f"Total RTT: {rtt_str}")
        print(f"Output file: {output_file}")
        sys.exit(0)
    elif exit_code == 124:
        print()
        print("=== TIMEOUT ===")
        print(f"Simulation timed out after {args.timeout}s")
        sys.exit(1)
    else:
        print()
        print("=== FAILED ===")
        print(f"Exit code: {exit_code}")
        sys.exit(1)


if __name__ == "__main__":
    main()
