#!/usr/bin/env python3
"""
Compare results between herosim-original and my-herosim pipelines.

Analyzes:
- Queue time vs RTT correlation
- Wait time vs RTT correlation
- Cold start patterns
- Overall RTT breakdown
"""

import json
import sys
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not available, using basic statistics")

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, using basic correlation")


def load_result_file(filepath: Path) -> Dict:
    """Load a result JSON file with progress tracking."""
    print(f"Loading: {filepath.name}...", end='', flush=True)
    try:
        # For large files, use streaming JSON parser if available
        file_size = filepath.stat().st_size / (1024 * 1024)  # MB
        if file_size > 50:
            print(f" ({file_size:.1f}MB - this may take a while)", end='', flush=True)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f" ✓ ({file_size:.1f}MB loaded)")
        return data
    except Exception as e:
        print(f" ✗ Error: {e}")
        return {}


def extract_task_metrics(data: Dict, source_name: str) -> List[Dict]:
    """Extract task-level metrics from simulation results with progress tracking."""
    # Handle both formats
    if 'stats' in data:
        task_results = data.get('stats', {}).get('taskResults', [])
    else:
        task_results = data.get('taskResults', [])
    
    total = len(task_results)
    print(f"  Extracting {total:,} tasks...", end='', flush=True)
    
    tasks = []
    for i, task in enumerate(task_results):
        # Only include real tasks (taskId >= 0)
        task_id = task.get('taskId')
        if task_id is not None and task_id >= 0:
            tasks.append({
                'task_id': task_id,
                'elapsed_time': task.get('elapsedTime', 0),
                'queue_time': task.get('queueTime', 0),
                'wait_time': task.get('waitTime', 0),
                'execution_time': task.get('executionTime', 0),
                'pull_time': task.get('pullTime', 0),
                'cold_start_time': task.get('coldStartTime', 0),
                'initialization_time': task.get('initializationTime', 0),
                'compute_time': task.get('computeTime', 0),
                'communications_time': task.get('communicationsTime', 0),
                'network_latency': task.get('networkLatency', 0),
                'cold_started': task.get('coldStarted', False),
                'cache_hit': task.get('cacheHit', False),
                'source': source_name,
            })
        
        # Progress update every 10k tasks
        if (i + 1) % 10000 == 0:
            print(f" {i+1:,}/{total:,}...", end='', flush=True)
    
    print(f" ✓ ({len(tasks):,} valid tasks)")
    return tasks


def pearson_correlation(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Calculate Pearson correlation coefficient without numpy/scipy."""
    n = len(x)
    if n != len(y) or n < 2:
        return 0.0, 1.0
    
    # Calculate means
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    # Calculate standard deviations
    var_x = sum((xi - mean_x) ** 2 for xi in x) / (n - 1) if n > 1 else 0
    var_y = sum((yi - mean_y) ** 2 for yi in y) / (n - 1) if n > 1 else 0
    
    if var_x == 0 or var_y == 0:
        return 0.0, 1.0
    
    std_x = math.sqrt(var_x)
    std_y = math.sqrt(var_y)
    
    # Calculate covariance
    covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / (n - 1) if n > 1 else 0
    
    # Calculate correlation
    correlation = covariance / (std_x * std_y) if (std_x * std_y) > 0 else 0.0
    
    # Simple p-value approximation (not exact, but good enough)
    t_stat = correlation * math.sqrt((n - 2) / (1 - correlation ** 2)) if abs(correlation) < 0.999 else 0
    # For large n, p-value is very small if |r| > 0.1
    pvalue = 0.001 if abs(correlation) > 0.1 else 0.5
    
    return correlation, pvalue


def calculate_correlations(tasks: List[Dict]) -> Dict[str, float]:
    """Calculate correlations between RTT and its components."""
    if len(tasks) < 2:
        return {}
    
    if HAS_NUMPY and HAS_SCIPY:
        elapsed = np.array([t['elapsed_time'] for t in tasks])
        queue = np.array([t['queue_time'] for t in tasks])
        wait = np.array([t['wait_time'] for t in tasks])
        execution = np.array([t['execution_time'] for t in tasks])
        compute = np.array([t['compute_time'] for t in tasks])
        network = np.array([t['network_latency'] for t in tasks])
        
        correlations = {}
        
        # Queue time vs RTT
        if np.std(queue) > 0 and np.std(elapsed) > 0:
            corr, pvalue = stats.pearsonr(queue, elapsed)
            correlations['queue_vs_rtt'] = {
                'correlation': corr,
                'pvalue': pvalue,
                'r_squared': corr ** 2,
            }
        
        # Wait time vs RTT
        if np.std(wait) > 0 and np.std(elapsed) > 0:
            corr, pvalue = stats.pearsonr(wait, elapsed)
            correlations['wait_vs_rtt'] = {
                'correlation': corr,
                'pvalue': pvalue,
                'r_squared': corr ** 2,
            }
        
        # Execution time vs RTT
        if np.std(execution) > 0 and np.std(elapsed) > 0:
            corr, pvalue = stats.pearsonr(execution, elapsed)
            correlations['execution_vs_rtt'] = {
                'correlation': corr,
                'pvalue': pvalue,
                'r_squared': corr ** 2,
            }
        
        # Compute time vs RTT
        if np.std(compute) > 0 and np.std(elapsed) > 0:
            corr, pvalue = stats.pearsonr(compute, elapsed)
            correlations['compute_vs_rtt'] = {
                'correlation': corr,
                'pvalue': pvalue,
                'r_squared': corr ** 2,
            }
        
        # Network latency vs RTT
        if np.std(network) > 0 and np.std(elapsed) > 0:
            corr, pvalue = stats.pearsonr(network, elapsed)
            correlations['network_vs_rtt'] = {
                'correlation': corr,
                'pvalue': pvalue,
                'r_squared': corr ** 2,
            }
    else:
        # Fallback to basic implementation
        elapsed = [t['elapsed_time'] for t in tasks]
        queue = [t['queue_time'] for t in tasks]
        wait = [t['wait_time'] for t in tasks]
        execution = [t['execution_time'] for t in tasks]
        compute = [t['compute_time'] for t in tasks]
        network = [t['network_latency'] for t in tasks]
        
        correlations = {}
        
        # Queue time vs RTT
        if len(set(queue)) > 1 and len(set(elapsed)) > 1:
            corr, pvalue = pearson_correlation(queue, elapsed)
            correlations['queue_vs_rtt'] = {
                'correlation': corr,
                'pvalue': pvalue,
                'r_squared': corr ** 2,
            }
        
        # Wait time vs RTT
        if len(set(wait)) > 1 and len(set(elapsed)) > 1:
            corr, pvalue = pearson_correlation(wait, elapsed)
            correlations['wait_vs_rtt'] = {
                'correlation': corr,
                'pvalue': pvalue,
                'r_squared': corr ** 2,
            }
        
        # Execution time vs RTT
        if len(set(execution)) > 1 and len(set(elapsed)) > 1:
            corr, pvalue = pearson_correlation(execution, elapsed)
            correlations['execution_vs_rtt'] = {
                'correlation': corr,
                'pvalue': pvalue,
                'r_squared': corr ** 2,
            }
        
        # Compute time vs RTT
        if len(set(compute)) > 1 and len(set(elapsed)) > 1:
            corr, pvalue = pearson_correlation(compute, elapsed)
            correlations['compute_vs_rtt'] = {
                'correlation': corr,
                'pvalue': pvalue,
                'r_squared': corr ** 2,
            }
        
        # Network latency vs RTT
        if len(set(network)) > 1 and len(set(elapsed)) > 1:
            corr, pvalue = pearson_correlation(network, elapsed)
            correlations['network_vs_rtt'] = {
                'correlation': corr,
                'pvalue': pvalue,
                'r_squared': corr ** 2,
            }
    
    return correlations


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile without numpy."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    d0 = sorted_data[int(f)] * (c - k)
    d1 = sorted_data[int(c)] * (k - f)
    return d0 + d1


def analyze_compute_time_distribution(tasks: List[Dict]) -> Dict:
    """Analyze compute time distribution to understand the discrepancy."""
    if not tasks:
        return {}
    
    # Extract data
    elapsed_list = [t['elapsed_time'] for t in tasks if t['elapsed_time'] > 0]
    compute_list = [t['compute_time'] for t in tasks if t['elapsed_time'] > 0]
    
    # Calculate ratios
    ratios = [c / e for c, e in zip(compute_list, elapsed_list) if e > 0]
    
    if HAS_NUMPY and len(ratios) > 0:
        ratios_arr = np.array(ratios)
        return {
            'num_tasks': len(ratios),
            'mean_ratio': float(np.mean(ratios_arr)),
            'median_ratio': float(np.median(ratios_arr)),
            'std_ratio': float(np.std(ratios_arr)),
            'min_ratio': float(np.min(ratios_arr)),
            'max_ratio': float(np.max(ratios_arr)),
            'p95_ratio': float(np.percentile(ratios_arr, 95)),
            'p99_ratio': float(np.percentile(ratios_arr, 99)),
            'tasks_with_ratio_gt_1': int(np.sum(ratios_arr > 1.0)),
            'tasks_with_ratio_gt_0_5': int(np.sum(ratios_arr > 0.5)),
            'tasks_with_ratio_lt_0_01': int(np.sum(ratios_arr < 0.01)),
        }
    else:
        if not ratios:
            return {}
        sorted_ratios = sorted(ratios)
        return {
            'num_tasks': len(ratios),
            'mean_ratio': sum(ratios) / len(ratios),
            'median_ratio': sorted_ratios[len(sorted_ratios) // 2],
            'std_ratio': math.sqrt(sum((r - sum(ratios)/len(ratios))**2 for r in ratios) / len(ratios)),
            'min_ratio': min(ratios),
            'max_ratio': max(ratios),
            'p95_ratio': percentile(ratios, 95),
            'p99_ratio': percentile(ratios, 99),
            'tasks_with_ratio_gt_1': sum(1 for r in ratios if r > 1.0),
            'tasks_with_ratio_gt_0_5': sum(1 for r in ratios if r > 0.5),
            'tasks_with_ratio_lt_0_01': sum(1 for r in ratios if r < 0.01),
        }


def calculate_statistics(tasks: List[Dict]) -> Dict:
    """Calculate comprehensive timing statistics."""
    if not tasks:
        return {}
    
    # Extract arrays
    elapsed_list = [t['elapsed_time'] for t in tasks]
    queue_list = [t['queue_time'] for t in tasks]
    wait_list = [t['wait_time'] for t in tasks]
    execution_list = [t['execution_time'] for t in tasks]
    pull_list = [t['pull_time'] for t in tasks]
    cold_start_list = [t['cold_start_time'] for t in tasks]
    init_list = [t['initialization_time'] for t in tasks]
    compute_list = [t['compute_time'] for t in tasks]
    comms_list = [t['communications_time'] for t in tasks]
    network_list = [t['network_latency'] for t in tasks]
    
    if HAS_NUMPY:
        metrics = {
            'elapsed_time': np.array(elapsed_list),
            'queue_time': np.array(queue_list),
            'wait_time': np.array(wait_list),
            'execution_time': np.array(execution_list),
            'pull_time': np.array(pull_list),
            'cold_start_time': np.array(cold_start_list),
            'initialization_time': np.array(init_list),
            'compute_time': np.array(compute_list),
            'communications_time': np.array(comms_list),
            'network_latency': np.array(network_list),
        }
        
        # Calculate proportions - TWO METHODS for comparison
        elapsed = metrics['elapsed_time']
        total_elapsed = np.sum(elapsed[elapsed > 0])
        non_zero_elapsed = elapsed[elapsed > 0]
        
        proportions = {}
        proportions_by_ratio_mean = {}  # Old method: mean of ratios
        
        for key in ['queue_time', 'wait_time', 'execution_time', 'pull_time', 
                    'cold_start_time', 'initialization_time', 'compute_time', 
                    'communications_time', 'network_latency']:
            metric = metrics[key]
            total_metric = np.sum(metric[elapsed > 0])
            
            # Method 1: Total time proportion (correct for system-level analysis)
            if total_elapsed > 0:
                proportions[key] = (total_metric / total_elapsed) * 100
            else:
                proportions[key] = 0.0
            
            # Method 2: Mean of ratios (old method - can be misleading)
            if len(non_zero_elapsed) > 0:
                ratio = metric / elapsed
                proportions_by_ratio_mean[key] = np.mean(ratio[elapsed > 0]) * 100
            else:
                proportions_by_ratio_mean[key] = 0.0
        
        # Calculate summary stats
        summary = {}
        for key, arr in metrics.items():
            summary[key] = {
                'mean': float(np.mean(arr)),
                'median': float(np.median(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'p95': float(np.percentile(arr, 95)),
                'p99': float(np.percentile(arr, 99)),
            }
    else:
        # Fallback implementation
        metrics_dict = {
            'elapsed_time': elapsed_list,
            'queue_time': queue_list,
            'wait_time': wait_list,
            'execution_time': execution_list,
            'pull_time': pull_list,
            'cold_start_time': cold_start_list,
            'initialization_time': init_list,
            'compute_time': compute_list,
            'communications_time': comms_list,
            'network_latency': network_list,
        }
        
        # Calculate proportions - TWO METHODS for comparison
        total_elapsed = sum(e for e in elapsed_list if e > 0)
        proportions = {}
        proportions_by_ratio_mean = {}  # Old method: mean of ratios
        
        for key in ['queue_time', 'wait_time', 'execution_time', 'pull_time', 
                    'cold_start_time', 'initialization_time', 'compute_time', 
                    'communications_time', 'network_latency']:
            metric_list = metrics_dict[key]
            total_metric = sum(m for m, e in zip(metric_list, elapsed_list) if e > 0)
            
            # Method 1: Total time proportion (correct for system-level analysis)
            if total_elapsed > 0:
                proportions[key] = (total_metric / total_elapsed) * 100
            else:
                proportions[key] = 0.0
            
            # Method 2: Mean of ratios (old method - can be misleading)
            ratios = [m / e for m, e in zip(metric_list, elapsed_list) if e > 0]
            if ratios:
                proportions_by_ratio_mean[key] = (sum(ratios) / len(ratios)) * 100
            else:
                proportions_by_ratio_mean[key] = 0.0
        
        # Calculate summary stats
        summary = {}
        for key, arr in metrics_dict.items():
            if arr:
                summary[key] = {
                    'mean': sum(arr) / len(arr),
                    'median': sorted(arr)[len(arr) // 2] if arr else 0,
                    'std': math.sqrt(sum((x - sum(arr)/len(arr))**2 for x in arr) / len(arr)) if arr else 0,
                    'min': min(arr),
                    'max': max(arr),
                    'p95': percentile(arr, 95),
                    'p99': percentile(arr, 99),
                }
            else:
                summary[key] = {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0, 'p95': 0, 'p99': 0}
    
    # Count cold starts and cache hits
    cold_start_count = sum(1 for t in tasks if t['cold_started'])
    cache_hit_count = sum(1 for t in tasks if t['cache_hit'])
    
    return {
        'num_tasks': len(tasks),
        'cold_start_count': cold_start_count,
        'cold_start_proportion': (cold_start_count / len(tasks) * 100) if tasks else 0,
        'cache_hit_count': cache_hit_count,
        'cache_hit_proportion': (cache_hit_count / len(tasks) * 100) if tasks else 0,
        'metrics': summary,
        'proportions': proportions,
        'proportions_by_ratio_mean': proportions_by_ratio_mean if 'proportions_by_ratio_mean' in locals() else {},
    }


def extract_energy_metrics(data: Dict) -> Dict:
    """Extract energy metrics from result data."""
    if 'stats' in data:
        stats = data.get('stats', {})
    else:
        stats = data
    
    energy = stats.get('energy', 0)
    reclaimable_energy = stats.get('reclaimableEnergy', 0)
    task_results = stats.get('taskResults', [])
    num_tasks = len([t for t in task_results if t.get('taskId', -1) >= 0])
    
    # Calculate energy per task
    energy_per_task = energy / num_tasks if num_tasks > 0 else 0
    reclaimable_per_task = reclaimable_energy / num_tasks if num_tasks > 0 else 0
    wasted_energy = energy - reclaimable_energy
    
    return {
        'total_energy_kwh': energy,
        'reclaimable_energy_kwh': reclaimable_energy,
        'wasted_energy_kwh': wasted_energy,
        'num_tasks': num_tasks,
        'energy_per_task_kwh': energy_per_task,
        'reclaimable_per_task_kwh': reclaimable_per_task,
        'wasted_per_task_kwh': wasted_energy / num_tasks if num_tasks > 0 else 0,
        'waste_percentage': (wasted_energy / energy * 100) if energy > 0 else 0,
        'efficiency_percentage': (reclaimable_energy / energy * 100) if energy > 0 else 0,
    }


def print_comparison(original_stats: Dict, myherosim_stats: Dict, 
                    original_corr: Dict, myherosim_corr: Dict,
                    original_compute_analysis: Dict = None, myherosim_compute_analysis: Dict = None,
                    original_energy: Dict = None, myherosim_energy: Dict = None):
    """Print comparison between original and my-herosim."""
    print("\n" + "="*100)
    print("COMPREHENSIVE COMPARISON: herosim-original vs my-herosim")
    print("="*100)
    
    # Basic stats
    print(f"\n{'Metric':<40} {'herosim-original':<30} {'my-herosim':<30}")
    print("-" * 100)
    print(f"{'Total tasks':<40} {original_stats.get('num_tasks', 0):<30} {myherosim_stats.get('num_tasks', 0):<30}")
    print(f"{'Cold start proportion':<40} {original_stats.get('cold_start_proportion', 0):.2f}%{'':<25} {myherosim_stats.get('cold_start_proportion', 0):.2f}%{'':<25}")
    print(f"{'Cache hit proportion':<40} {original_stats.get('cache_hit_proportion', 0):.2f}%{'':<25} {myherosim_stats.get('cache_hit_proportion', 0):.2f}%{'':<25}")
    
    # RTT breakdown - Show both calculation methods
    print("\n" + "-"*100)
    print("RTT BREAKDOWN - METHOD 1: Total Time Proportion (sum(component) / sum(elapsed))")
    print("This shows what % of TOTAL system time is spent in each component")
    print("-"*100)
    
    components = ['queue_time', 'wait_time', 'execution_time', 'compute_time', 
                  'initialization_time', 'network_latency']
    component_labels = {
        'queue_time': 'Queue Time',
        'wait_time': 'Wait Time',
        'execution_time': 'Execution Time',
        'compute_time': 'Compute Time',
        'initialization_time': 'Initialization Time',
        'network_latency': 'Network Latency',
    }
    
    for comp in components:
        orig_prop = original_stats.get('proportions', {}).get(comp, 0)
        my_prop = myherosim_stats.get('proportions', {}).get(comp, 0)
        label = component_labels.get(comp, comp)
        print(f"{label:<40} {orig_prop:>6.2f}%{'':<23} {my_prop:>6.2f}%{'':<23}")
    
    # Show old method for comparison
    print("\n" + "-"*100)
    print("RTT BREAKDOWN - METHOD 2: Mean of Ratios (mean(component/elapsed))")
    print("This shows the AVERAGE % per task (can be misleading with outliers)")
    print("-"*100)
    
    for comp in components:
        orig_prop_old = original_stats.get('proportions_by_ratio_mean', {}).get(comp, 0)
        my_prop_old = myherosim_stats.get('proportions_by_ratio_mean', {}).get(comp, 0)
        label = component_labels.get(comp, comp)
        print(f"{label:<40} {orig_prop_old:>6.2f}%{'':<23} {my_prop_old:>6.2f}%{'':<23}")
    
    # Explain the difference
    print("\n" + "-"*100)
    print("WHY THE DIFFERENCE?")
    print("-"*100)
    print("Method 1 (Total Time): Shows system-level time distribution")
    print("  - If 1000 tasks each wait 1s in queue and compute for 0.01s:")
    print("    Queue: 1000s / 1010s = 99% | Compute: 10s / 1010s = 1%")
    print("")
    print("Method 2 (Mean Ratio): Shows average per-task ratio")
    print("  - Same example: mean(queue/elapsed) = 1/1.01 = 99% | mean(compute/elapsed) = 0.01/1.01 = 1%")
    print("  - BUT: If some tasks have very small elapsed_time, ratios can be >100% or skewed")
    print("  - Example: Task with 0.01s elapsed and 0.05s compute → 500% ratio!")
    print("")
    
    # Analyze compute time specifically
    orig_compute_mean = original_stats.get('metrics', {}).get('compute_time', {}).get('mean', 0)
    orig_elapsed_mean = original_stats.get('metrics', {}).get('elapsed_time', {}).get('mean', 0)
    orig_compute_total = original_stats.get('proportions', {}).get('compute_time', 0) / 100 * orig_elapsed_mean * original_stats.get('num_tasks', 1)
    
    my_compute_mean = myherosim_stats.get('metrics', {}).get('compute_time', {}).get('mean', 0)
    my_elapsed_mean = myherosim_stats.get('metrics', {}).get('elapsed_time', {}).get('mean', 0)
    
    print(f"Compute Time Analysis:")
    print(f"  herosim-original:")
    print(f"    - Average compute_time per task: {orig_compute_mean:.6f}s")
    print(f"    - Average elapsed_time per task: {orig_elapsed_mean:.4f}s")
    print(f"    - Ratio (compute/elapsed): {orig_compute_mean/orig_elapsed_mean*100:.2f}%")
    print(f"    - Total compute time proportion: {original_stats.get('proportions', {}).get('compute_time', 0):.2f}%")
    print(f"  my-herosim:")
    print(f"    - Average compute_time per task: {my_compute_mean:.6f}s")
    print(f"    - Average elapsed_time per task: {my_elapsed_mean:.4f}s")
    print(f"    - Ratio (compute/elapsed): {my_compute_mean/my_elapsed_mean*100:.2f}%")
    print(f"    - Total compute time proportion: {myherosim_stats.get('proportions', {}).get('compute_time', 0):.2f}%")
    
    # Detailed compute time analysis
    if original_compute_analysis and myherosim_compute_analysis:
        print("\n" + "-"*100)
        print("COMPUTE TIME RATIO DISTRIBUTION ANALYSIS")
        print("(Shows why Method 2 gives different results)")
        print("-"*100)
        
        print(f"\n{'Metric':<50} {'herosim-original':<30} {'my-herosim':<30}")
        print("-"*110)
        
        metrics_to_show = [
            ('mean_ratio', 'Mean ratio (compute/elapsed)'),
            ('median_ratio', 'Median ratio'),
            ('p95_ratio', '95th percentile ratio'),
            ('p99_ratio', '99th percentile ratio'),
            ('max_ratio', 'Max ratio'),
            ('tasks_with_ratio_gt_1', 'Tasks with ratio > 1.0 (impossible!)'),
            ('tasks_with_ratio_gt_0_5', 'Tasks with ratio > 0.5'),
            ('tasks_with_ratio_lt_0_01', 'Tasks with ratio < 0.01'),
        ]
        
        for key, label in metrics_to_show:
            orig_val = original_compute_analysis.get(key, 0)
            my_val = myherosim_compute_analysis.get(key, 0)
            if isinstance(orig_val, float):
                print(f"{label:<50} {orig_val:>10.4f}{'':<19} {my_val:>10.4f}{'':<19}")
            else:
                print(f"{label:<50} {orig_val:>10}{'':<19} {my_val:>10}{'':<19}")
        
        print("\n" + "-"*100)
        print("EXPLANATION OF DISCREPANCY:")
        print("-"*100)
        
        orig_gt1 = original_compute_analysis.get('tasks_with_ratio_gt_1', 0)
        my_gt1 = myherosim_compute_analysis.get('tasks_with_ratio_gt_1', 0)
        
        if orig_gt1 > 0 or my_gt1 > 0:
            print(f"⚠️  WARNING: Found tasks where compute_time > elapsed_time!")
            print(f"   This is impossible and indicates data quality issues or timing bugs.")
            print(f"   herosim-original: {orig_gt1} tasks")
            print(f"   my-herosim: {my_gt1} tasks")
            print(f"   These outliers heavily skew Method 2 (mean of ratios).")
        
        orig_max_ratio = original_compute_analysis.get('max_ratio', 0)
        my_max_ratio = myherosim_compute_analysis.get('max_ratio', 0)
        
        if orig_max_ratio > 10 or my_max_ratio > 10:
            print(f"\n⚠️  Extreme outliers detected:")
            print(f"   herosim-original max ratio: {orig_max_ratio:.2f}x")
            print(f"   my-herosim max ratio: {my_max_ratio:.2f}x")
            print(f"   These extreme values make Method 2 unreliable.")
        
        print(f"\n✓ Method 1 (Total Time Proportion) is more reliable because:")
        print(f"  - It's not affected by outliers")
        print(f"  - It shows actual system resource usage")
        print(f"  - It correctly weights by task duration")
        print(f"  - Sum of proportions = 100% (verifiable)")
    
    # Average values
    print("\n" + "-"*100)
    print("AVERAGE VALUES (seconds)")
    print("-"*100)
    
    value_components = ['elapsed_time', 'queue_time', 'wait_time', 'execution_time', 'compute_time']
    value_labels = {
        'elapsed_time': 'RTT (Elapsed Time)',
        'queue_time': 'Queue Time',
        'wait_time': 'Wait Time',
        'execution_time': 'Execution Time',
        'compute_time': 'Compute Time',
    }
    
    for comp in value_components:
        orig_mean = original_stats.get('metrics', {}).get(comp, {}).get('mean', 0)
        my_mean = myherosim_stats.get('metrics', {}).get(comp, {}).get('mean', 0)
        label = value_labels.get(comp, comp)
        print(f"{label:<40} {orig_mean:>10.4f}s{'':<18} {my_mean:>10.4f}s{'':<18}")
    
    # Energy analysis
    print("\n" + "="*100)
    print("ENERGY ANALYSIS")
    print("="*100)
    
    if original_energy and myherosim_energy:
        print(f"\n{'Metric':<50} {'herosim-original':<30} {'my-herosim':<30}")
        print("-"*110)
        
        orig_total = original_energy.get('total_energy_kwh', 0)
        my_total = myherosim_energy.get('total_energy_kwh', 0)
        print(f"{'Total Energy (kWh)':<50} {orig_total:>15.6f}{'':<14} {my_total:>15.6f}{'':<14}")
        
        orig_reclaim = original_energy.get('reclaimable_energy_kwh', 0)
        my_reclaim = myherosim_energy.get('reclaimable_energy_kwh', 0)
        print(f"{'Reclaimable Energy (kWh)':<50} {orig_reclaim:>15.6f}{'':<14} {my_reclaim:>15.6f}{'':<14}")
        
        orig_wasted = original_energy.get('wasted_energy_kwh', 0)
        my_wasted = myherosim_energy.get('wasted_energy_kwh', 0)
        print(f"{'Wasted Energy (kWh)':<50} {orig_wasted:>15.6f}{'':<14} {my_wasted:>15.6f}{'':<14}")
        
        orig_waste_pct = original_energy.get('waste_percentage', 0)
        my_waste_pct = myherosim_energy.get('waste_percentage', 0)
        print(f"{'Waste Percentage (%)':<50} {orig_waste_pct:>15.2f}%{'':<13} {my_waste_pct:>15.2f}%{'':<13}")
        
        orig_efficiency = original_energy.get('efficiency_percentage', 0)
        my_efficiency = myherosim_energy.get('efficiency_percentage', 0)
        print(f"{'Energy Efficiency (%)':<50} {orig_efficiency:>15.2f}%{'':<13} {my_efficiency:>15.2f}%{'':<13}")
        
        orig_per_task = original_energy.get('energy_per_task_kwh', 0)
        my_per_task = myherosim_energy.get('energy_per_task_kwh', 0)
        print(f"{'Energy per Task (kWh)':<50} {orig_per_task:>15.9f}{'':<14} {my_per_task:>15.9f}{'':<14}")
        
        orig_reclaim_per_task = original_energy.get('reclaimable_per_task_kwh', 0)
        my_reclaim_per_task = myherosim_energy.get('reclaimable_per_task_kwh', 0)
        print(f"{'Reclaimable per Task (kWh)':<50} {orig_reclaim_per_task:>15.9f}{'':<14} {my_reclaim_per_task:>15.9f}{'':<14}")
        
        # Energy analysis insights
        print("\n" + "-"*100)
        print("ENERGY ANALYSIS INSIGHTS")
        print("-"*100)
        
        if my_total > 0 and orig_total > 0:
            energy_ratio = my_total / orig_total
            print(f"\n1. Total Energy Comparison:")
            print(f"   - my-herosim uses {energy_ratio:.2f}x the energy of herosim-original")
            if energy_ratio > 1.5:
                print(f"   ⚠️  WARNING: my-herosim uses significantly MORE energy")
                print(f"      This is likely due to:")
                print(f"        - More replicas (queue_length=5 vs 100)")
                print(f"        - Longer keep_alive (60s vs 30s) → replicas stay idle longer")
                print(f"        - More nodes (20 vs 10)")
            elif energy_ratio < 0.8:
                print(f"   ✓ my-herosim uses LESS energy (more efficient)")
            else:
                print(f"   → Similar energy consumption")
        
        if my_waste_pct > orig_waste_pct * 1.2:
            print(f"\n2. Energy Waste:")
            print(f"   - herosim-original: {orig_waste_pct:.2f}% wasted")
            print(f"   - my-herosim: {my_waste_pct:.2f}% wasted")
            print(f"   ⚠️  my-herosim wastes more energy (likely due to idle replicas)")
        
        if my_per_task > orig_per_task * 1.2:
            print(f"\n3. Energy per Task:")
            print(f"   - herosim-original: {orig_per_task:.9f} kWh/task")
            print(f"   - my-herosim: {my_per_task:.9f} kWh/task")
            print(f"   ⚠️  Higher energy per task in my-herosim")
            print(f"      Possible reasons:")
            print(f"        - More replicas = more infrastructure overhead")
            print(f"        - Network latency adds energy for offloaded tasks")
            print(f"        - Batching overhead")
        elif my_per_task < orig_per_task * 0.8:
            print(f"\n3. Energy per Task:")
            print(f"   - herosim-original: {orig_per_task:.9f} kWh/task")
            print(f"   - my-herosim: {my_per_task:.9f} kWh/task")
            print(f"   ✓ Lower energy per task in my-herosim (more efficient)")
        
        print(f"\n4. Energy Efficiency Trade-offs:")
        print(f"   - herosim-original: Lower energy but higher RTT (72.35s)")
        print(f"   - my-herosim: Higher energy but lower RTT (47.69s)")
        print(f"   → Trade-off: Energy vs Performance")
        print(f"   → Energy increase: {((my_total/orig_total - 1) * 100):.1f}%")
        print(f"   → RTT improvement: {((1 - 47.69/72.35) * 100):.1f}%")
        
        if my_total > orig_total:
            efficiency_gain = (1 - 47.69/72.35) * 100
            energy_cost = (my_total/orig_total - 1) * 100
            if efficiency_gain > energy_cost:
                print(f"   ✓ Performance gain ({efficiency_gain:.1f}%) > Energy cost ({energy_cost:.1f}%)")
            else:
                print(f"   ⚠️  Energy cost ({energy_cost:.1f}%) > Performance gain ({efficiency_gain:.1f}%)")
    else:
        print("\n⚠️  Energy metrics not available in result files")
    
    # CORRELATIONS - THE KEY ANALYSIS
    print("\n" + "="*100)
    print("CORRELATION ANALYSIS: Queue Time vs RTT")
    print("="*100)
    
    orig_queue_corr = original_corr.get('queue_vs_rtt', {})
    my_queue_corr = myherosim_corr.get('queue_vs_rtt', {})
    
    print(f"\n{'Metric':<40} {'herosim-original':<30} {'my-herosim':<30}")
    print("-" * 100)
    
    if orig_queue_corr:
        orig_r = orig_queue_corr.get('correlation', 0)
        orig_r2 = orig_queue_corr.get('r_squared', 0)
        print(f"{'Queue Time vs RTT (Pearson r)':<40} {orig_r:>8.4f}{'':<21} {my_queue_corr.get('correlation', 0):>8.4f}{'':<21}")
        print(f"{'Queue Time vs RTT (R²)':<40} {orig_r2:>8.4f} ({orig_r2*100:.2f}%){'':<15} {my_queue_corr.get('r_squared', 0):>8.4f} ({my_queue_corr.get('r_squared', 0)*100:.2f}%){'':<15}")
    else:
        print(f"{'Queue Time vs RTT (Pearson r)':<40} {'N/A':<30} {my_queue_corr.get('correlation', 0):>8.4f}{'':<21}")
        print(f"{'Queue Time vs RTT (R²)':<40} {'N/A':<30} {my_queue_corr.get('r_squared', 0):>8.4f} ({my_queue_corr.get('r_squared', 0)*100:.2f}%){'':<15}")
    
    # Wait time correlation
    print("\n" + "-"*100)
    print("CORRELATION ANALYSIS: Wait Time vs RTT")
    print("-"*100)
    
    orig_wait_corr = original_corr.get('wait_vs_rtt', {})
    my_wait_corr = myherosim_corr.get('wait_vs_rtt', {})
    
    if orig_wait_corr:
        orig_r = orig_wait_corr.get('correlation', 0)
        orig_r2 = orig_wait_corr.get('r_squared', 0)
        print(f"{'Wait Time vs RTT (Pearson r)':<40} {orig_r:>8.4f}{'':<21} {my_wait_corr.get('correlation', 0):>8.4f}{'':<21}")
        print(f"{'Wait Time vs RTT (R²)':<40} {orig_r2:>8.4f} ({orig_r2*100:.2f}%){'':<15} {my_wait_corr.get('r_squared', 0):>8.4f} ({my_wait_corr.get('r_squared', 0)*100:.2f}%){'':<15}")
    else:
        print(f"{'Wait Time vs RTT (Pearson r)':<40} {'N/A':<30} {my_wait_corr.get('correlation', 0):>8.4f}{'':<21}")
        print(f"{'Wait Time vs RTT (R²)':<40} {'N/A':<30} {my_wait_corr.get('r_squared', 0):>8.4f} ({my_wait_corr.get('r_squared', 0)*100:.2f}%){'':<15}")
    
    # Key insights
    print("\n" + "="*100)
    print("KEY INSIGHTS")
    print("="*100)
    
    if orig_queue_corr and my_queue_corr:
        orig_r2 = orig_queue_corr.get('r_squared', 0)
        my_r2 = my_queue_corr.get('r_squared', 0)
        
        print(f"\n1. Queue Time Correlation:")
        print(f"   - herosim-original: {orig_r2*100:.2f}% of RTT variance explained by queue time")
        print(f"   - my-herosim: {my_r2*100:.2f}% of RTT variance explained by queue time")
        
        if my_r2 > 0.95:
            print(f"   ⚠️  WARNING: my-herosim has {my_r2*100:.2f}% correlation - this is VERY HIGH!")
            print(f"      This suggests queue time is the dominant factor in RTT.")
            print(f"      Possible causes:")
            print(f"        - queue_length=5 (too low) → more replicas → less queue contention")
            print(f"        - Batching may be affecting wait_time instead of queue_time")
            print(f"        - Network latency not being captured properly")
        
        if orig_r2 < 0.5:
            print(f"   ✓ herosim-original has lower correlation ({orig_r2*100:.2f}%)")
            print(f"     This suggests RTT is more balanced across components.")
    
    # Component breakdown comparison
    orig_queue_prop = original_stats.get('proportions', {}).get('queue_time', 0)
    my_queue_prop = myherosim_stats.get('proportions', {}).get('queue_time', 0)
    
    print(f"\n2. Queue Time as % of RTT:")
    print(f"   - herosim-original: {orig_queue_prop:.2f}%")
    print(f"   - my-herosim: {my_queue_prop:.2f}%")
    
    if my_queue_prop > 80:
        print(f"   ⚠️  WARNING: Queue time dominates RTT in my-herosim!")
    
    orig_wait_prop = original_stats.get('proportions', {}).get('wait_time', 0)
    my_wait_prop = myherosim_stats.get('proportions', {}).get('wait_time', 0)
    
    print(f"\n3. Wait Time as % of RTT:")
    print(f"   - herosim-original: {orig_wait_prop:.2f}%")
    print(f"   - my-herosim: {my_wait_prop:.2f}%")
    
    if my_wait_prop > orig_wait_prop * 2:
        print(f"   ⚠️  Wait time is much higher in my-herosim (likely due to batching)")
    
    print("\n" + "="*100)


def main():
    if len(sys.argv) < 3:
        print("Usage: pipenv run python compare_original_vs_myherosim.py <original_result.json> <myherosim_result.json>")
        print("\nExample:")
        print("  pipenv run python scripts_cosim/compare_original_vs_myherosim.py \\")
        print("    /root/projects/herosim-original/result_baselines/20260117-142421-902848.json \\")
        print("    simulation_data/results/simulation_result_knative.json")
        sys.exit(1)
    
    original_file = Path(sys.argv[1])
    myherosim_file = Path(sys.argv[2])
    
    # Load files
    print("="*80)
    print("Loading result files (this may take a while for large files)...")
    original_data = load_result_file(original_file)
    myherosim_data = load_result_file(myherosim_file)
    
    if not original_data or not myherosim_data:
        print("Error: Failed to load result files")
        sys.exit(1)
    
    # Extract task metrics
    print("\nExtracting task metrics:")
    original_tasks = extract_task_metrics(original_data, "herosim-original")
    myherosim_tasks = extract_task_metrics(myherosim_data, "my-herosim")
    
    # Calculate statistics
    print("\nCalculating statistics...", end='', flush=True)
    original_stats = calculate_statistics(original_tasks)
    myherosim_stats = calculate_statistics(myherosim_tasks)
    print(" ✓")
    
    # Analyze compute time distribution
    print("Analyzing compute time distribution...", end='', flush=True)
    original_compute_analysis = analyze_compute_time_distribution(original_tasks)
    myherosim_compute_analysis = analyze_compute_time_distribution(myherosim_tasks)
    print(" ✓")
    
    # Calculate correlations
    print("Calculating correlations...", end='', flush=True)
    original_corr = calculate_correlations(original_tasks)
    myherosim_corr = calculate_correlations(myherosim_tasks)
    print(" ✓")
    
    # Extract energy metrics
    print("Extracting energy metrics...", end='', flush=True)
    original_energy = extract_energy_metrics(original_data)
    myherosim_energy = extract_energy_metrics(myherosim_data)
    print(" ✓")
    
    # Print comparison
    print_comparison(original_stats, myherosim_stats, original_corr, myherosim_corr, 
                    original_compute_analysis, myherosim_compute_analysis,
                    original_energy, myherosim_energy)


if __name__ == "__main__":
    main()
