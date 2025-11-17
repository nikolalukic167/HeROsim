#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

base_path = Path("simulation_data/artifacts/run9_all/gnn_datasets")

print(f"Scanning directory: {base_path}")
print("=" * 60)

results = []
total_datasets = 0
total_processed = 0
stats = {
    "total_tasks": 0,
    "positive_task_ids": 0,
    "cold_start_tasks": 0,
    "positive_cold_start_tasks": 0,
    "task_id_distribution": defaultdict(int)
}

for ds_dir in sorted(base_path.iterdir()):
    if not ds_dir.is_dir() or not ds_dir.name.startswith("ds_"):
        continue
    
    total_datasets += 1
    json_path = ds_dir / "optimal_result.json"
    if not json_path.exists():
        print(f"[SKIP] {ds_dir.name}: optimal_result.json not found")
        continue
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        stats_data = data.get("stats", {})
        task_results = stats_data.get("taskResults", []) if isinstance(stats_data, dict) else []
        stats["total_tasks"] += len(task_results)
        
        positive_cold_start_ids = []
        positive_task_ids = []
        all_cold_start_ids = []
        
        for task in task_results:
            task_id = task.get("taskId")
            cold_started = task.get("coldStarted", False)
            
            if task_id is not None:
                stats["task_id_distribution"][task_id] += 1
            
            if task_id is not None and 0 <= task_id <= 4:
                positive_task_ids.append(task_id)
                stats["positive_task_ids"] += 1
            
            if cold_started:
                all_cold_start_ids.append(task_id)
                stats["cold_start_tasks"] += 1
            
            if task_id is not None and 0 <= task_id <= 4 and cold_started:
                positive_cold_start_ids.append(task_id)
                stats["positive_cold_start_tasks"] += 1
        
        if positive_cold_start_ids:
            unique_ids = sorted(set(positive_cold_start_ids))
            results.append({
                "dataset": ds_dir.name,
                "cold_start_task_ids": unique_ids,
                "total_tasks": len(task_results),
                "positive_tasks": len(positive_task_ids),
                "all_cold_starts": len(all_cold_start_ids)
            })
            print(f"[FOUND] {ds_dir.name}: {unique_ids} (tasks: {len(task_results)}, positive: {len(positive_task_ids)}, cold_starts: {len(all_cold_start_ids)})")
        else:
            if total_processed % 100 == 0:
                print(f"[CHECK] {ds_dir.name}: no positive cold start tasks (tasks: {len(task_results)})")
        
        total_processed += 1
    
    except Exception as e:
        print(f"[ERROR] {json_path}: {e}", file=sys.stderr)

print("=" * 60)
print("\nSUMMARY:")
print(f"Total datasets found: {total_datasets}")
print(f"Total datasets processed: {total_processed}")
print(f"Datasets with positive cold start tasks (0-4): {len(results)}")
print(f"\nStatistics:")
print(f"  Total tasks analyzed: {stats['total_tasks']}")
print(f"  Tasks with positive IDs (0-4): {stats['positive_task_ids']}")
print(f"  Cold start tasks: {stats['cold_start_tasks']}")
print(f"  Positive cold start tasks (0-4): {stats['positive_cold_start_tasks']}")

if results:
    print("\nDatasets with positive cold start tasks:")
    for result in results:
        print(f"  {result['dataset']}: task IDs {result['cold_start_task_ids']}")
else:
    print("\nNo datasets found with positive cold start tasks (0-4)")

