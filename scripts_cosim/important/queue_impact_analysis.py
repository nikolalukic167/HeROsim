#!/usr/bin/env python3
"""
Analysis of queue impact on co-simulation duration and optimization strategies.

Key Findings:
1. Queue distributions directly impact simulation time (6.35ms per warmup task)
2. Large queues (norm16, pois12) can create 100,000+ warmup tasks → hours of simulation
3. Fixed queue normalization (QUEUE_NORM_FACTOR=10) doesn't adapt to system state
4. Every warmup task must execute fully (network + cold start + execution + comm)

Optimization Strategies:
1. Adaptive queue normalization based on current system state
2. Batch warmup task execution with time compression
3. Selective warmup (only execute representative samples)
"""

import re
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path("/root/projects/my-herosim")


def analyze_queue_impact():
    """Analyze how queue distributions affect simulation duration."""
    queue_times = defaultdict(list)
    queue_placements = defaultdict(list)
    
    progress_file = BASE_DIR / "logs/progress.txt"
    if not progress_file.exists():
        print(f"ERROR: {progress_file} not found")
        return
    
    with open(progress_file, 'r') as f:
        for line in f:
            if 'SUCCESS' in line:
                # Extract queue type, duration
                match = re.search(r'q=(\w+)', line)
                if match:
                    q_type = match.group(1)
                    duration_match = re.search(r'(\d+\.\d+)s.*?RTT', line)
                    if duration_match:
                        duration = float(duration_match.group(1))
                        queue_times[q_type].append(duration)
                
                # Extract placements count if present
                placements_match = re.search(r'placements:\s*(\d+)', line)
                if placements_match:
                    placements = int(placements_match.group(1))
                    q_match = re.search(r'q=(\w+)', line)
                    if q_match:
                        q_type = q_match.group(1)
                        queue_placements[q_type].append(placements)
    
    print("="*100)
    print("QUEUE IMPACT ANALYSIS")
    print("="*100 + "\n")
    
    print("Average Duration by Queue Type:")
    print("-"*100)
    for q_type in sorted(queue_times.keys()):
        times = queue_times[q_type]
        if times:
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            print(f"  {q_type:12s}: avg={avg_time:7.1f}s, min={min_time:6.1f}s, max={max_time:7.1f}s, count={len(times)}")
    
    print("\nPlacements Count by Queue Type:")
    print("-"*100)
    for q_type in sorted(queue_placements.keys()):
        placements = queue_placements[q_type]
        if placements:
            avg_placements = sum(placements) / len(placements)
            max_placements = max(placements)
            min_placements = min(placements)
            print(f"  {q_type:12s}: avg={avg_placements:10.0f}, min={min_placements:8d}, max={max_placements:10d}, count={len(placements)}")
    
    # Correlation analysis
    print("\n" + "="*100)
    print("CORRELATION: Placements vs Duration")
    print("="*100)
    
    with open(progress_file, 'r') as f:
        pairs = []
        for line in f:
            if 'SUCCESS' in line and 'placements:' in line:
                duration_match = re.search(r'(\d+\.\d+)s.*?RTT', line)
                placements_match = re.search(r'placements:\s*(\d+)', line)
                if duration_match and placements_match:
                    duration = float(duration_match.group(1))
                    placements = int(placements_match.group(1))
                    pairs.append((placements, duration))
        
        if pairs:
            pairs.sort()
            print(f"\nFound {len(pairs)} datasets with placement counts")
            print("\nTop 10 by placements:")
            for placements, duration in pairs[-10:]:
                print(f"  {placements:10d} placements → {duration:8.1f}s ({duration/60:.1f} min)")
            
            if len(pairs) > 1:
                avg_placements = sum(p for p, d in pairs) / len(pairs)
                avg_duration = sum(d for p, d in pairs) / len(pairs)
                print(f"\nAverage: {avg_placements:.0f} placements → {avg_duration:.1f}s")
                print(f"Ratio: {avg_duration/avg_placements*1000:.3f}ms per placement")
                print(f"\n⚠️  CRITICAL: Each warmup task takes ~{avg_duration/avg_placements*1000:.1f}ms to execute")
                print(f"   With 100,000 warmup tasks: {100000 * avg_duration/avg_placements / 60:.1f} minutes")


def main():
    analyze_queue_impact()
    
    print("\n" + "="*100)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*100 + "\n")
    
    print("""
1. ADAPTIVE QUEUE NORMALIZATION
   Current: Fixed QUEUE_NORM_FACTOR = 10.0
   Problem: Doesn't adapt to actual queue distribution in system
   
   Solution: Calculate normalization factor dynamically based on:
   - Current max queue length in system
   - Percentile-based normalization (e.g., 90th percentile)
   - Per-batch normalization (normalize within each batch's context)
   
   Implementation:
   - Calculate QUEUE_NORM_FACTOR per batch from queue snapshot
   - Use percentile-based normalization (robust to outliers)
   - Fallback to fixed factor if no queues present


2. WARMUP TASK EXECUTION OPTIMIZATION
   Current: Every warmup task executes fully (network + cold start + execution + comm)
   Problem: 100,000+ warmup tasks → hours of simulation time
   
   Solution Options:
   
   A. BATCH WARMUP EXECUTION (Time Compression)
      - Group warmup tasks by platform
      - Execute them in compressed time (skip intermediate timeouts)
      - Still calculate real execution times, but execute sequentially without delays
      - Mark all as completed at once after calculating total time
   
   B. SELECTIVE WARMUP (Representative Sampling)
      - Only execute a sample of warmup tasks (e.g., first N per platform)
      - Calculate remaining queue time based on sample
      - Use statistical estimation for remaining tasks
      - Risk: May miss edge cases in queue behavior
   
   C. FAST-FORWARD WARMUP (Time Jump)
      - Calculate total warmup time for all tasks on a platform
      - Jump simulation time forward by that amount
      - Mark all warmup tasks as completed
      - Still accurate for timing, but skips intermediate events
   
   D. PARALLEL WARMUP (Concurrent Execution)
      - Execute warmup tasks in parallel (if platform supports it)
      - Reduce wall-clock time while maintaining simulation accuracy
      - Requires platform concurrency modeling


3. HYBRID APPROACH (Recommended)
   - Use adaptive queue normalization for GNN inference
   - Use fast-forward warmup for large queues (>1000 tasks)
   - Use full execution for small queues (<100 tasks) to maintain accuracy
   - Calculate real execution times, but skip intermediate SimPy events
    """)


if __name__ == "__main__":
    main()
