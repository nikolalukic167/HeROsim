#!/usr/bin/env python3
"""
Test script to compare fast-forward warmup vs normal execution.

Runs 3 placements with fast-forward enabled and 3 without,
then compares the results.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

def run_test_placement(placement_idx: int, fast_forward: bool, output_base: Path):
    """Run a single test placement."""
    print(f"\n{'='*100}")
    print(f"Test Placement {placement_idx}: {'WITH' if fast_forward else 'WITHOUT'} Fast-Forward Warmup")
    print(f"{'='*100}\n")
    
    # Create output directory
    test_dir = output_base / f"test_placement_{placement_idx}_{'ff' if fast_forward else 'normal'}"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Use existing config file as base
    base_config_path = PROJECT_ROOT / "simulation_data" / "space_with_network.json"
    if not base_config_path.exists():
        print(f"ERROR: Base config file not found: {base_config_path}")
        return {
            'success': False,
            'duration': 0,
            'rtt': float('inf'),
            'error': f"Config file not found: {base_config_path}"
        }
    
    with open(base_config_path, 'r') as f:
        config = json.load(f)
    
    # Override prewarm config to create LARGE queues for testing fast-forward
    # mean=150 ensures most platforms have >100 tasks to trigger fast-forward
    config['prewarm'] = {
        "dnn1": {
            "distribution": "none",
            "queue_distribution": "statistical",
            "queue_distribution_params": {
                "type": "normal",
                "mean": 150,
                "stddev": 20,
                "min": 100,
                "max": 200,
                "step": 1
            }
        },
        "dnn2": {
            "distribution": "none",
            "queue_distribution": "statistical",
            "queue_distribution_params": {
                "type": "normal",
                "mean": 150,
                "stddev": 20,
                "min": 100,
                "max": 200,
                "step": 1
            }
        }
    }
    
    # Save config
    config_path = test_dir / "space_with_network.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create minimal workload (5 tasks) - use proper structure
    workload = {
        "rps": 10,
        "duration": 10,
        "events": [
            {
                "timestamp": i,
                "application": {
                    "name": "nofs-dnn1" if i % 2 == 0 else "nofs-dnn2",
                    "dag": {"dnn1": []} if i % 2 == 0 else {"dnn2": []}
                },
                "node_name": f"client_node{i % 10}",
                "qos": "medium"  # Required field
            }
            for i in range(5)
        ]
    }
    
    workload_path = test_dir / "workload.json"
    with open(workload_path, 'w') as f:
        json.dump(workload, f, indent=2)
    
    # Prepare paths
    sim_input_path = PROJECT_ROOT / "data" / "nofs-ids"
    samples_file = PROJECT_ROOT / "simulation_data" / "lhs_samples_simple.npy"
    mapping_file = PROJECT_ROOT / "simulation_data" / "lhs_samples_simple_mapping.pkl"
    results_dir = test_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Generate infrastructure
    infra_file = test_dir / "infrastructure.json"
    from src.generate_infrastructure import generate_deterministic_infrastructure
    generate_deterministic_infrastructure(
        str(config_path),
        sim_input_path,
        str(infra_file),
        101
    )
    
    # Load sample
    import numpy as np
    samples = np.load(samples_file)
    sample = samples[0]
    
    # Run simulation
    from src.executecosimulation import execute_brute_force_optimized
    
    start_time = time.time()
    try:
        result_paths = execute_brute_force_optimized(
            apps=list(config['wsc'].keys()),
            config_file=str(config_path),
            mapping_file=str(mapping_file),
            output_dir=results_dir,
            sample=sample,
            sim_input_path=sim_input_path,
            workload_base_file=str(workload_path),
            max_workers=1,
            infrastructure_file=infra_file,
            quiet=False,
            final_dataset_dir=test_dir,
            fast_forward_warmup=fast_forward,
            fast_forward_threshold=100
        )
        duration = time.time() - start_time
        
        # Load best result
        best_json = results_dir / "best.json"
        if best_json.exists():
            with open(best_json, 'r') as f:
                best_info = json.load(f)
            rtt = best_info.get('rtt', float('inf'))
        else:
            rtt = float('inf')
        
        return {
            'success': True,
            'duration': duration,
            'rtt': rtt,
            'result_paths': result_paths
        }
    except Exception as e:
        duration = time.time() - start_time
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'duration': duration,
            'rtt': float('inf'),
            'error': str(e)
        }

def compare_results(results_ff: list, results_normal: list):
    """Compare fast-forward vs normal results."""
    print(f"\n{'='*100}")
    print("RESULTS COMPARISON")
    print(f"{'='*100}\n")
    
    print("Fast-Forward Warmup Results:")
    print("-" * 100)
    for i, result in enumerate(results_ff, 1):
        status = "✓" if result['success'] else "✗"
        print(f"  Placement {i}: {status} RTT={result['rtt']:.3f}s, Duration={result['duration']:.1f}s")
        if not result['success']:
            print(f"    Error: {result.get('error', 'Unknown')}")
    
    print("\nNormal Execution Results:")
    print("-" * 100)
    for i, result in enumerate(results_normal, 1):
        status = "✓" if result['success'] else "✗"
        print(f"  Placement {i}: {status} RTT={result['rtt']:.3f}s, Duration={result['duration']:.1f}s")
        if not result['success']:
            print(f"    Error: {result.get('error', 'Unknown')}")
    
    # Calculate averages
    ff_successful = [r for r in results_ff if r['success']]
    normal_successful = [r for r in results_normal if r['success']]
    
    if ff_successful and normal_successful:
        avg_rtt_ff = sum(r['rtt'] for r in ff_successful) / len(ff_successful)
        avg_duration_ff = sum(r['duration'] for r in ff_successful) / len(ff_successful)
        avg_rtt_normal = sum(r['rtt'] for r in normal_successful) / len(normal_successful)
        avg_duration_normal = sum(r['duration'] for r in normal_successful) / len(normal_successful)
        
        print(f"\n{'='*100}")
        print("SUMMARY")
        print(f"{'='*100}\n")
        print(f"Fast-Forward Warmup:")
        print(f"  Average RTT: {avg_rtt_ff:.3f}s")
        print(f"  Average Duration: {avg_duration_ff:.1f}s")
        print(f"  Success Rate: {len(ff_successful)}/{len(results_ff)}")
        print(f"\nNormal Execution:")
        print(f"  Average RTT: {avg_rtt_normal:.3f}s")
        print(f"  Average Duration: {avg_duration_normal:.1f}s")
        print(f"  Success Rate: {len(normal_successful)}/{len(results_normal)}")
        
        speedup = avg_duration_normal / avg_duration_ff if avg_duration_ff > 0 else 0
        rtt_diff = avg_rtt_ff - avg_rtt_normal
        rtt_diff_pct = (rtt_diff / avg_rtt_normal * 100) if avg_rtt_normal > 0 else 0
        
        print(f"\nComparison:")
        print(f"  Speedup: {speedup:.2f}x {'(faster)' if speedup > 1 else '(slower)'}")
        print(f"  RTT Difference: {rtt_diff:+.3f}s ({rtt_diff_pct:+.2f}%)")
        print(f"  {'✓ RTT matches' if abs(rtt_diff) < 0.01 else '⚠ RTT differs significantly'}")

def main():
    """Main test function."""
    output_base = PROJECT_ROOT / "simulation_data" / "fast_forward_test"
    output_base.mkdir(parents=True, exist_ok=True)
    
    print("="*100)
    print("FAST-FORWARD WARMUP TEST")
    print("="*100)
    print("\nThis test will:")
    print("  1. Run 3 placements WITH fast-forward warmup")
    print("  2. Run 3 placements WITHOUT fast-forward warmup")
    print("  3. Compare results (RTT accuracy and speedup)")
    print(f"\nOutput directory: {output_base}\n")
    
    # Run test with fast-forward (1 dataset only)
    print("\n" + "="*100)
    print("PHASE 1: Running WITH Fast-Forward Warmup")
    print("="*100)
    results_ff = []
    result = run_test_placement(1, fast_forward=True, output_base=output_base)
    results_ff.append(result)
    
    # Run test without fast-forward (1 dataset only)
    print("\n" + "="*100)
    print("PHASE 2: Running WITHOUT Fast-Forward Warmup")
    print("="*100)
    results_normal = []
    result = run_test_placement(1, fast_forward=False, output_base=output_base)
    results_normal.append(result)
    
    # Compare results
    compare_results(results_ff, results_normal)
    
    print(f"\n{'='*100}")
    print("Test complete! Results saved to:", output_base)
    print(f"{'='*100}\n")

if __name__ == "__main__":
    main()
