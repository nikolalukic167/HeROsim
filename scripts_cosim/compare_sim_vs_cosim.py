#!/usr/bin/env python3
"""
Compare Real Simulation Results vs Co-Simulation Training Data

This script analyzes the differences between:
1. Real simulation results (from running GNN/Knative/HRC schedulers)
2. Co-simulation generated training data (from generate_gnn_datasets_fast.py)

The goal is to identify discrepancies that could affect GNN training quality.

Usage:
    python scripts_cosim/compare_sim_vs_cosim.py [--results-dir DIR] [--cosim-dir DIR]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import statistics

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_simulation_result(filepath: Path) -> Optional[Dict]:
    """Load a simulation result JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def load_cosim_dataset(dataset_dir: Path) -> Optional[Dict]:
    """Load a co-simulation dataset (system_state_captured_unique.json)."""
    state_file = dataset_dir / "system_state_captured_unique.json"
    if not state_file.exists():
        return None
    try:
        with open(state_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {state_file}: {e}")
        return None


def analyze_queue_distributions(task_results: List[Dict]) -> Dict[str, Any]:
    """Analyze queue length distributions from real simulation."""
    all_queues = []
    queue_snapshots_at_scheduling = []
    full_queue_snapshots = []
    
    for tr in task_results:
        # Queue snapshot at scheduling time (valid replicas only)
        qs = tr.get('queueSnapshotAtScheduling')
        if qs:
            queue_snapshots_at_scheduling.extend(qs.values())
        
        # Full queue snapshot (all platforms)
        fqs = tr.get('fullQueueSnapshot')
        if fqs:
            full_queue_snapshots.extend(fqs.values())
    
    result = {}
    
    if queue_snapshots_at_scheduling:
        result['valid_replicas_queue'] = {
            'count': len(queue_snapshots_at_scheduling),
            'mean': statistics.mean(queue_snapshots_at_scheduling),
            'median': statistics.median(queue_snapshots_at_scheduling),
            'max': max(queue_snapshots_at_scheduling),
            'min': min(queue_snapshots_at_scheduling),
            'stdev': statistics.stdev(queue_snapshots_at_scheduling) if len(queue_snapshots_at_scheduling) > 1 else 0,
            'zeros_pct': 100 * queue_snapshots_at_scheduling.count(0) / len(queue_snapshots_at_scheduling),
        }
    
    if full_queue_snapshots:
        result['all_platforms_queue'] = {
            'count': len(full_queue_snapshots),
            'mean': statistics.mean(full_queue_snapshots),
            'median': statistics.median(full_queue_snapshots),
            'max': max(full_queue_snapshots),
            'min': min(full_queue_snapshots),
            'stdev': statistics.stdev(full_queue_snapshots) if len(full_queue_snapshots) > 1 else 0,
            'zeros_pct': 100 * full_queue_snapshots.count(0) / len(full_queue_snapshots),
        }
    
    return result


def analyze_temporal_state(task_results: List[Dict]) -> Dict[str, Any]:
    """Analyze temporal state (remaining times) from real simulation."""
    current_task_remaining = []
    cold_start_remaining = []
    comm_remaining = []
    
    for tr in task_results:
        ts = tr.get('temporalStateAtScheduling')
        if ts:
            for platform_state in ts.values():
                if isinstance(platform_state, dict):
                    current_task_remaining.append(platform_state.get('current_task_remaining', 0))
                    cold_start_remaining.append(platform_state.get('cold_start_remaining', 0))
                    comm_remaining.append(platform_state.get('comm_remaining', 0))
    
    result = {}
    
    if current_task_remaining:
        non_zero_task = [x for x in current_task_remaining if x > 0]
        result['current_task_remaining'] = {
            'count': len(current_task_remaining),
            'mean': statistics.mean(current_task_remaining),
            'max': max(current_task_remaining),
            'non_zero_count': len(non_zero_task),
            'non_zero_mean': statistics.mean(non_zero_task) if non_zero_task else 0,
        }
    
    if cold_start_remaining:
        non_zero_cold = [x for x in cold_start_remaining if x > 0]
        result['cold_start_remaining'] = {
            'count': len(cold_start_remaining),
            'mean': statistics.mean(cold_start_remaining),
            'max': max(cold_start_remaining),
            'non_zero_count': len(non_zero_cold),
            'non_zero_mean': statistics.mean(non_zero_cold) if non_zero_cold else 0,
        }
    
    if comm_remaining:
        non_zero_comm = [x for x in comm_remaining if x > 0]
        result['comm_remaining'] = {
            'count': len(comm_remaining),
            'mean': statistics.mean(comm_remaining),
            'max': max(comm_remaining),
            'non_zero_count': len(non_zero_comm),
            'non_zero_mean': statistics.mean(non_zero_comm) if non_zero_comm else 0,
        }
    
    return result


def analyze_replica_counts(task_results: List[Dict]) -> Dict[str, Any]:
    """Analyze replica counts from real simulation."""
    valid_replica_counts = []
    
    for tr in task_results:
        qs = tr.get('queueSnapshotAtScheduling')
        if qs:
            valid_replica_counts.append(len(qs))
    
    if valid_replica_counts:
        return {
            'mean': statistics.mean(valid_replica_counts),
            'median': statistics.median(valid_replica_counts),
            'max': max(valid_replica_counts),
            'min': min(valid_replica_counts),
            'stdev': statistics.stdev(valid_replica_counts) if len(valid_replica_counts) > 1 else 0,
        }
    return {}


def analyze_cosim_dataset(dataset: Dict) -> Dict[str, Any]:
    """Analyze a co-simulation dataset."""
    result = {}
    
    # Replica counts
    replicas = dataset.get('replicas', {})
    total_replicas = sum(len(r) for r in replicas.values())
    result['total_replicas'] = total_replicas
    result['replicas_per_type'] = {k: len(v) for k, v in replicas.items()}
    
    # Queue analysis from task_placements
    task_placements = dataset.get('task_placements', [])
    all_queues = []
    full_queues = []
    temporal_remaining = []
    
    for tp in task_placements:
        qs = tp.get('queue_snapshot_at_scheduling', {})
        if qs:
            all_queues.extend(qs.values())
        
        fqs = tp.get('full_queue_snapshot', {})
        if fqs:
            full_queues.extend(fqs.values())
        
        ts = tp.get('temporal_state_at_scheduling', {})
        if ts:
            for ps in ts.values():
                if isinstance(ps, dict):
                    temporal_remaining.append(ps.get('current_task_remaining', 0))
    
    if all_queues:
        result['valid_replicas_queue'] = {
            'count': len(all_queues),
            'mean': statistics.mean(all_queues),
            'max': max(all_queues),
            'zeros_pct': 100 * all_queues.count(0) / len(all_queues),
        }
    
    if full_queues:
        result['all_platforms_queue'] = {
            'count': len(full_queues),
            'mean': statistics.mean(full_queues),
            'max': max(full_queues),
            'zeros_pct': 100 * full_queues.count(0) / len(full_queues),
        }
    
    if temporal_remaining:
        non_zero = [x for x in temporal_remaining if x > 0]
        result['temporal_state'] = {
            'count': len(temporal_remaining),
            'mean': statistics.mean(temporal_remaining),
            'non_zero_count': len(non_zero),
        }
    
    return result


def compare_distributions(sim_stats: Dict, cosim_stats: List[Dict]) -> Dict[str, Any]:
    """Compare distributions between simulation and co-simulation."""
    comparison = {}
    
    # Aggregate co-sim stats
    cosim_queue_means = []
    cosim_queue_maxes = []
    cosim_zeros_pcts = []
    
    for cs in cosim_stats:
        if 'valid_replicas_queue' in cs:
            cosim_queue_means.append(cs['valid_replicas_queue'].get('mean', 0))
            cosim_queue_maxes.append(cs['valid_replicas_queue'].get('max', 0))
            cosim_zeros_pcts.append(cs['valid_replicas_queue'].get('zeros_pct', 100))
    
    if cosim_queue_means:
        comparison['cosim_queue_mean'] = {
            'mean': statistics.mean(cosim_queue_means),
            'max': max(cosim_queue_means),
            'min': min(cosim_queue_means),
        }
        comparison['cosim_queue_max'] = {
            'mean': statistics.mean(cosim_queue_maxes),
            'max': max(cosim_queue_maxes),
        }
        comparison['cosim_zeros_pct'] = {
            'mean': statistics.mean(cosim_zeros_pcts),
        }
    
    if 'valid_replicas_queue' in sim_stats:
        comparison['sim_queue'] = sim_stats['valid_replicas_queue']
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Compare simulation vs co-simulation data")
    parser.add_argument('--results-dir', type=str, 
                        default='/root/projects/my-herosim/simulation_data/results',
                        help='Directory with simulation results')
    parser.add_argument('--cosim-dir', type=str,
                        default='/root/projects/my-herosim/simulation_data/artifacts/run_queue_big',
                        help='Directory with co-simulation datasets')
    parser.add_argument('--max-cosim-datasets', type=int, default=10000,
                        help='Maximum co-sim datasets to analyze')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    cosim_dir = Path(args.cosim_dir)
    
    print("=" * 80)
    print("COMPARISON: Real Simulation vs Co-Simulation Training Data")
    print("=" * 80)
    
    # Load and analyze simulation results
    sim_files = {
        'gnn': results_dir / 'simulation_result_gnn.json',
        'herocache_network': results_dir / 'simulation_result_herocache_network.json',
        'knative': results_dir / 'simulation_result_knative.json',
    }
    
    sim_analyses = {}
    
    for name, filepath in sim_files.items():
        if filepath.exists():
            print(f"\n{'='*40}")
            print(f"Analyzing {name.upper()} simulation results...")
            print(f"{'='*40}")
            
            data = load_simulation_result(filepath)
            if data:
                stats = data.get('stats', {})
                task_results = stats.get('taskResults', [])
                
                print(f"  Total tasks: {len(task_results)}")
                print(f"  Total RTT: {data.get('total_rtt', 'N/A')}")
                
                # Check if state capture is present
                sample_tr = task_results[0] if task_results else {}
                has_state = bool(sample_tr.get('queueSnapshotAtScheduling'))
                print(f"  State capture present: {has_state}")
                
                if has_state:
                    queue_stats = analyze_queue_distributions(task_results)
                    temporal_stats = analyze_temporal_state(task_results)
                    replica_stats = analyze_replica_counts(task_results)
                    
                    sim_analyses[name] = {
                        'queue': queue_stats,
                        'temporal': temporal_stats,
                        'replicas': replica_stats,
                    }
                    
                    print(f"\n  Queue Statistics (valid replicas at scheduling):")
                    if 'valid_replicas_queue' in queue_stats:
                        qs = queue_stats['valid_replicas_queue']
                        print(f"    Mean: {qs['mean']:.2f}")
                        print(f"    Median: {qs['median']:.2f}")
                        print(f"    Max: {qs['max']}")
                        print(f"    StdDev: {qs['stdev']:.2f}")
                        print(f"    Zeros %: {qs['zeros_pct']:.1f}%")
                    
                    print(f"\n  Temporal State (current_task_remaining):")
                    if 'current_task_remaining' in temporal_stats:
                        ts = temporal_stats['current_task_remaining']
                        print(f"    Mean: {ts['mean']:.4f}")
                        print(f"    Max: {ts['max']:.4f}")
                        print(f"    Non-zero count: {ts['non_zero_count']} / {ts['count']}")
                    
                    print(f"\n  Valid Replica Counts per Task:")
                    if replica_stats:
                        print(f"    Mean: {replica_stats['mean']:.1f}")
                        print(f"    Max: {replica_stats['max']}")
                        print(f"    Min: {replica_stats['min']}")
    
    # Analyze co-simulation datasets
    print(f"\n{'='*80}")
    print("Analyzing Co-Simulation Training Data...")
    print(f"{'='*80}")
    
    cosim_analyses = []
    
    for task_count in ['2tasks', '3tasks']:
        task_dir = cosim_dir / f'gnn_datasets_{task_count}'
        if not task_dir.exists():
            continue
        
        print(f"\n  Dataset: {task_count}")
        
        dataset_dirs = sorted(task_dir.glob('ds_*'))[:args.max_cosim_datasets]
        loaded = 0
        
        for ds_dir in dataset_dirs:
            dataset = load_cosim_dataset(ds_dir)
            if dataset:
                analysis = analyze_cosim_dataset(dataset)
                analysis['task_count'] = task_count
                analysis['dataset_id'] = ds_dir.name
                cosim_analyses.append(analysis)
                loaded += 1
        
        print(f"    Loaded: {loaded} datasets")
    
    # Aggregate co-sim statistics
    if cosim_analyses:
        print(f"\n  Co-Simulation Aggregate Statistics:")
        
        queue_means = [cs['valid_replicas_queue']['mean'] for cs in cosim_analyses if 'valid_replicas_queue' in cs]
        queue_maxes = [cs['valid_replicas_queue']['max'] for cs in cosim_analyses if 'valid_replicas_queue' in cs]
        zeros_pcts = [cs['valid_replicas_queue']['zeros_pct'] for cs in cosim_analyses if 'valid_replicas_queue' in cs]
        
        if queue_means:
            print(f"    Queue Mean (across datasets): {statistics.mean(queue_means):.2f}")
            print(f"    Queue Max (across datasets): {max(queue_maxes)}")
            print(f"    Zeros % (across datasets): {statistics.mean(zeros_pcts):.1f}%")
    
    # Comparison and recommendations
    print(f"\n{'='*80}")
    print("COMPARISON AND RECOMMENDATIONS")
    print(f"{'='*80}")
    
    for name, sim_data in sim_analyses.items():
        print(f"\n{name.upper()} vs Co-Simulation:")
        
        if 'queue' in sim_data and 'valid_replicas_queue' in sim_data['queue']:
            sim_q = sim_data['queue']['valid_replicas_queue']
            
            if queue_means:
                cosim_q_mean = statistics.mean(queue_means)
                cosim_q_max = max(queue_maxes)
                
                print(f"\n  Queue Length Comparison:")
                print(f"    Simulation Mean: {sim_q['mean']:.2f}")
                print(f"    Co-Sim Mean:     {cosim_q_mean:.2f}")
                print(f"    Difference:      {sim_q['mean'] - cosim_q_mean:.2f}")
                
                print(f"\n    Simulation Max:  {sim_q['max']}")
                print(f"    Co-Sim Max:      {cosim_q_max}")
                
                print(f"\n    Simulation Zeros %: {sim_q['zeros_pct']:.1f}%")
                print(f"    Co-Sim Zeros %:     {statistics.mean(zeros_pcts):.1f}%")
                
                # Recommendations
                if sim_q['mean'] > cosim_q_mean * 2:
                    print(f"\n  [!] ISSUE: Real simulation has much higher queue lengths!")
                    print(f"      Recommendation: Increase queue distribution parameters in co-sim")
                    print(f"      Current co-sim max queue ~{cosim_q_max}, real sim max {sim_q['max']}")
                
                if sim_q['zeros_pct'] < zeros_pcts[0] * 0.5 if zeros_pcts else False:
                    print(f"\n  [!] ISSUE: Real simulation has fewer empty queues!")
                    print(f"      Recommendation: Use higher initial queue values in co-sim")
    
    # Specific recommendations for generate_gnn_datasets_fast.py
    print(f"\n{'='*80}")
    print("SPECIFIC RECOMMENDATIONS FOR CO-SIMULATION")
    print(f"{'='*80}")
    
    if sim_analyses:
        # Get the highest queue stats from any simulation
        max_sim_queue_mean = max(
            sa['queue']['valid_replicas_queue']['mean'] 
            for sa in sim_analyses.values() 
            if 'queue' in sa and 'valid_replicas_queue' in sa['queue']
        )
        max_sim_queue_max = max(
            sa['queue']['valid_replicas_queue']['max'] 
            for sa in sim_analyses.values() 
            if 'queue' in sa and 'valid_replicas_queue' in sa['queue']
        )
        
        print(f"\n1. QUEUE DISTRIBUTION PARAMETERS:")
        print(f"   Real simulation queue mean: {max_sim_queue_mean:.2f}")
        print(f"   Real simulation queue max:  {max_sim_queue_max}")
        
        if queue_means:
            print(f"   Co-sim queue mean:          {statistics.mean(queue_means):.2f}")
            print(f"   Co-sim queue max:           {max(queue_maxes)}")
        
        print(f"\n   Suggested changes to QUEUE_DISTRIBUTIONS in generate_gnn_datasets_fast.py:")
        print(f"   - Add higher Poisson lambda values (e.g., 50, 100, 200)")
        print(f"   - Add normal distributions with mean ~{max_sim_queue_mean/2:.0f} to {max_sim_queue_mean:.0f}")
        print(f"   - Increase max queue bounds to at least {max_sim_queue_max}")
        
        print(f"\n2. WORKLOAD DURATION:")
        print(f"   Current WORKLOAD_DURATION: 30 seconds")
        print(f"   Consider increasing to 60-120 seconds to allow queue buildup")
        
        print(f"\n3. REPLICA CONFIGURATIONS:")
        print(f"   Ensure cold start (0% preinit) scenarios are well represented")
        print(f"   These create more realistic queue buildup patterns")
        
        print(f"\n4. QUEUE NORMALIZATION FACTOR:")
        print(f"   In prepare_graphs_cache.py, QUEUE_NORM_FACTOR should be:")
        print(f"   - At least {max_sim_queue_max} to handle real simulation max queues")
        print(f"   - Recommended: {max(100, max_sim_queue_max * 1.5):.0f}")


if __name__ == "__main__":
    main()
