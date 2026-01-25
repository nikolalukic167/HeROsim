#!/usr/bin/env python3
"""
Simple test to verify fast-forward warmup works.
Creates a scenario where one platform has > 100 warmup tasks.
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_fast_forward():
    """Test fast-forward with a simple scenario."""
    print("="*100)
    print("SIMPLE FAST-FORWARD WARMUP TEST")
    print("="*100 + "\n")
    
    # Use existing infrastructure generation to create a scenario
    # where one platform has > 100 warmup tasks
    from src.generate_infrastructure import generate_deterministic_infrastructure
    
    base_config_path = PROJECT_ROOT / "simulation_data" / "space_with_network.json"
    if not base_config_path.exists():
        print(f"ERROR: Config file not found: {base_config_path}")
        return
    
    with open(base_config_path, 'r') as f:
        config = json.load(f)
    
    # Modify prewarm to create a large queue on one platform
    # Use a high mean so at least one platform gets > 100 tasks
    config['prewarm'] = {
        "dnn1": {
            "distribution": "none",
            "queue_distribution": "statistical",
            "queue_distribution_params": {
                "type": "normal",
                "mean": 150,  # High mean to ensure some platforms get > 100
                "stddev": 20,
                "min": 0,
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
                "min": 0,
                "max": 200,
                "step": 1
            }
        }
    }
    
    test_dir = PROJECT_ROOT / "simulation_data" / "ff_simple_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = test_dir / "space_with_network.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    sim_input_path = PROJECT_ROOT / "data" / "nofs-ids"
    infra_file = test_dir / "infrastructure.json"
    
    print("Generating infrastructure with large queues...")
    generate_deterministic_infrastructure(
        str(config_path),
        sim_input_path,
        str(infra_file),
        101
    )
    
    # Check infrastructure to see queue distribution
    with open(infra_file, 'r') as f:
        infra_data = json.load(f)
    
    queue_dists = infra_data.get('queue_distributions', {})
    max_queue = 0
    platforms_with_large_queues = 0
    
    for task_type, queues in queue_dists.items():
        for queue_key, queue_len in queues.items():
            max_queue = max(max_queue, queue_len)
            if queue_len > 100:
                platforms_with_large_queues += 1
    
    print(f"\nQueue Distribution Analysis:")
    print(f"  Max queue length: {max_queue}")
    print(f"  Platforms with > 100 tasks: {platforms_with_large_queues}")
    
    if max_queue < 100:
        print(f"\n⚠️  WARNING: No platforms have > 100 tasks. Fast-forward won't trigger.")
        print(f"   Consider lowering threshold or increasing queue mean.")
    else:
        print(f"\n✓ Found {platforms_with_large_queues} platforms with > 100 tasks")
        print(f"  Fast-forward should trigger for these platforms")
    
    print(f"\nTest infrastructure saved to: {test_dir}")
    print(f"To test fast-forward, run a simulation with --fast-forward-warmup")

if __name__ == "__main__":
    test_fast_forward()
