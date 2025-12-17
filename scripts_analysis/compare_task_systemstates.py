#!/usr/bin/env python3
"""
Compare systemStateResult across tasks 0-4 in all simulation_*.json files in results/ directory.

For each dataset, loads all simulation files from results/ directory and compares
all systemStateResults from tasks with taskId 0, 1, 2, 3, 4 across all files.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

BASE_DIR = Path("/root/projects/my-herosim/simulation_data/gnn_datasets")
TARGET_TASK_IDS = [0, 1, 2, 3, 4]


def normalize_replicas(replicas: Dict) -> Dict[str, Set[Tuple[str, int]]]:
    """
    Normalize replicas dict to sets for comparison.
    
    Converts: {"dnn1": [["node0", 1], ["node1", 2]]}
    To: {"dnn1": {("node0", 1), ("node1", 2)}}
    """
    normalized = {}
    for task_type, replica_list in replicas.items():
        replica_set = set()
        for replica in replica_list:
            if isinstance(replica, list) and len(replica) >= 2:
                node_name, platform_id = replica[0], replica[1]
                replica_set.add((node_name, platform_id))
        normalized[task_type] = replica_set
    return normalized


def normalize_available_resources(available: Dict) -> Dict[str, Set[int]]:
    """
    Normalize available_resources dict to sets for comparison.
    
    Converts: {"node0": [1, 2, 3], "node1": [4, 5]}
    To: {"node0": {1, 2, 3}, "node1": {4, 5}}
    """
    normalized = {}
    for node_name, platform_list in available.items():
        normalized[node_name] = set(platform_list)
    return normalized


def compare_replicas(replicas1: Dict, replicas2: Dict) -> List[str]:
    """Compare two replicas dictionaries."""
    differences = []
    norm1 = normalize_replicas(replicas1)
    norm2 = normalize_replicas(replicas2)
    
    all_task_types = set(norm1.keys()) | set(norm2.keys())
    
    for task_type in all_task_types:
        set1 = norm1.get(task_type, set())
        set2 = norm2.get(task_type, set())
        
        if set1 != set2:
            only_in_1 = set1 - set2
            only_in_2 = set2 - set1
            if only_in_1:
                differences.append(f"replicas.{task_type}: Only in first: {sorted(only_in_1)}")
            if only_in_2:
                differences.append(f"replicas.{task_type}: Only in second: {sorted(only_in_2)}")
    
    return differences


def compare_available_resources(avail1: Dict, avail2: Dict) -> List[str]:
    """Compare two available_resources dictionaries."""
    differences = []
    norm1 = normalize_available_resources(avail1)
    norm2 = normalize_available_resources(avail2)
    
    all_nodes = set(norm1.keys()) | set(norm2.keys())
    
    for node_name in all_nodes:
        set1 = norm1.get(node_name, set())
        set2 = norm2.get(node_name, set())
        
        if set1 != set2:
            only_in_1 = set1 - set2
            only_in_2 = set2 - set1
            if only_in_1:
                differences.append(f"available_resources.{node_name}: Only in first: {sorted(only_in_1)}")
            if only_in_2:
                differences.append(f"available_resources.{node_name}: Only in second: {sorted(only_in_2)}")
    
    return differences


def load_simulation_file(file_path: Path) -> Optional[Dict]:
    """Load a simulation JSON file and return its data."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return None


def extract_target_tasks_from_file(file_path: Path) -> Dict[int, Dict]:
    """
    Extract tasks with taskId 0-4 from a simulation file.
    
    Returns: {taskId: task_result_dict}
    """
    data = load_simulation_file(file_path)
    if not data:
        return {}
    
    stats = data.get("stats", {})
    task_results = stats.get("taskResults", [])
    
    # Build a dict by taskId
    tasks_by_id = {}
    for task in task_results:
        task_id = task.get("taskId")
        if task_id in TARGET_TASK_IDS:
            tasks_by_id[task_id] = task
    
    return tasks_by_id


def extract_negative_tasks_from_file(file_path: Path) -> List[Dict]:
    """
    Extract all tasks with negative taskId (warmup tasks) from a simulation file.
    
    Returns: List of task dictionaries with negative taskIds
    """
    data = load_simulation_file(file_path)
    if not data:
        return []
    
    stats = data.get("stats", {})
    task_results = stats.get("taskResults", [])
    
    # Filter tasks with negative taskId
    negative_tasks = []
    for task in task_results:
        task_id = task.get("taskId")
        if task_id is not None and task_id < 0:
            negative_tasks.append(task)
    
    return negative_tasks


def analyze_results_directory(results_dir: Path, file_limit: Optional[int] = None) -> Dict:
    """
    Analyze all simulation_*.json files in results directory.
    
    Extracts all systemStateResults from tasks 0-4 and compares them.
    Also analyzes negative tasks (warmup tasks).
    """
    if not results_dir.exists():
        return {"error": f"Results directory not found: {results_dir}"}
    
    # Find all simulation files
    print(f"  Finding simulation files in {results_dir.name}...")
    simulation_files = sorted(results_dir.glob("simulation_*.json"))
    
    if not simulation_files:
        return {"error": f"No simulation_*.json files found in {results_dir}"}
    
    # Apply file limit if specified
    original_count = len(simulation_files)
    if file_limit and file_limit > 0:
        simulation_files = simulation_files[:file_limit]
        print(f"  Found {original_count} simulation files (limiting to {len(simulation_files)})")
    else:
        print(f"  Found {len(simulation_files)} simulation files")
    
    print(f"  Loading tasks 0-4 from all files...")
    
    # Extract all target tasks from all files
    all_tasks = []  # List of (file_name, task_id, systemStateResult)
    missing_tasks_by_file = defaultdict(list)
    
    for idx, sim_file in enumerate(simulation_files):
        if (idx + 1) % 10 == 0 or (idx + 1) == len(simulation_files):
            print(f"    Loading file {idx + 1}/{len(simulation_files)}: {sim_file.name[:60]}...")
        
        tasks = extract_target_tasks_from_file(sim_file)
        # Check which target task IDs are missing
        found_ids = set(tasks.keys())
        missing_ids = set(TARGET_TASK_IDS) - found_ids
        if missing_ids:
            missing_tasks_by_file[sim_file.name] = sorted(missing_ids)
        
        # Extract systemStateResult for each found task
        for task_id, task_data in tasks.items():
            system_state = task_data.get("systemStateResult")
            if system_state is None:
                missing_tasks_by_file[sim_file.name].append(f"task_{task_id}_no_systemStateResult")
            else:
                all_tasks.append({
                    "file": sim_file.name,
                    "task_id": task_id,
                    "systemStateResult": system_state
                })
    
    print(f"  Loaded {len(all_tasks)} tasks with systemStateResult")
    
    if not all_tasks:
        return {"error": "No systemStateResults found in any simulation files"}
    
    # Extract replicas and available_resources from all tasks
    print(f"  Extracting replicas and available_resources...")
    replicas_list = []
    available_list = []
    task_info = []
    
    for task_entry in all_tasks:
        state = task_entry["systemStateResult"]
        replicas_list.append(state.get("replicas", {}))
        available_list.append(state.get("available_resources", {}))
        task_info.append(f"{task_entry['file']}:task_{task_entry['task_id']}")
    
    # Compare all pairs
    total_comparisons = len(replicas_list) * (len(replicas_list) - 1) // 2
    print(f"  Comparing {total_comparisons} pairs of replicas...")
    replicas_identical = True
    replicas_differences = []
    for i in range(len(replicas_list)):
        if (i + 1) % 50 == 0 or (i + 1) == len(replicas_list):
            print(f"    Comparing replicas: {i + 1}/{len(replicas_list)} tasks processed...")
        for j in range(i + 1, len(replicas_list)):
            diff = compare_replicas(replicas_list[i], replicas_list[j])
            if diff:
                replicas_identical = False
                replicas_differences.append({
                    "task1": task_info[i],
                    "task2": task_info[j],
                    "differences": diff
                })
    
    print(f"  Comparing {total_comparisons} pairs of available_resources...")
    available_identical = True
    available_differences = []
    for i in range(len(available_list)):
        if (i + 1) % 50 == 0 or (i + 1) == len(available_list):
            print(f"    Comparing available_resources: {i + 1}/{len(available_list)} tasks processed...")
        for j in range(i + 1, len(available_list)):
            diff = compare_available_resources(available_list[i], available_list[j])
            if diff:
                available_identical = False
                available_differences.append({
                    "task1": task_info[i],
                    "task2": task_info[j],
                    "differences": diff
                })
    
    all_identical = replicas_identical and available_identical
    
    # Count unique states
    print(f"  Counting unique states...")
    unique_replicas = set()
    unique_available = set()
    for state in all_tasks:
        replicas = state["systemStateResult"].get("replicas", {})
        available = state["systemStateResult"].get("available_resources", {})
        
        # Normalize and convert to hashable format
        norm_replicas = normalize_replicas(replicas)
        replicas_tuple = tuple(sorted(
            (task_type, tuple(sorted(replica_set)))
            for task_type, replica_set in norm_replicas.items()
        ))
        
        norm_available = normalize_available_resources(available)
        available_tuple = tuple(sorted(
            (node_name, tuple(sorted(platform_set)))
            for node_name, platform_set in norm_available.items()
        ))
        
        unique_replicas.add(replicas_tuple)
        unique_available.add(available_tuple)
    
    print(f"  Analysis complete: {len(unique_replicas)} unique replicas states, {len(unique_available)} unique available_resources states")
    
    # Analyze negative tasks (warmup tasks)
    print(f"  Analyzing negative tasks (warmup tasks)...")
    negative_tasks_by_file = {}
    negative_task_times = []  # List of (file, task_id, arrivedTime, doneTime)
    
    for idx, sim_file in enumerate(simulation_files):
        if (idx + 1) % 10 == 0 or (idx + 1) == len(simulation_files):
            print(f"    Analyzing negative tasks: {idx + 1}/{len(simulation_files)} files processed...")
        
        negative_tasks = extract_negative_tasks_from_file(sim_file)
        negative_tasks_by_file[sim_file.name] = negative_tasks
        
        # Extract times for each negative task
        for task in negative_tasks:
            task_id = task.get("taskId")
            arrived_time = task.get("arrivedTime")
            done_time = task.get("doneTime")
            negative_task_times.append({
                "file": sim_file.name,
                "task_id": task_id,
                "arrivedTime": arrived_time,
                "doneTime": done_time,
                "duration": done_time - arrived_time if (arrived_time is not None and done_time is not None) else None
            })
    
    # Analyze negative task counts
    negative_task_counts = [len(tasks) for tasks in negative_tasks_by_file.values()]
    negative_task_counts_identical = len(set(negative_task_counts)) == 1
    
    # Analyze negative task times
    if negative_task_times:
        # Group by task_id
        times_by_task_id = defaultdict(list)
        for entry in negative_task_times:
            times_by_task_id[entry["task_id"]].append(entry)
        
        # Compare times across files for each task_id
        time_differences = []
        for task_id, entries in times_by_task_id.items():
            arrived_times = [e["arrivedTime"] for e in entries if e["arrivedTime"] is not None]
            done_times = [e["doneTime"] for e in entries if e["doneTime"] is not None]
            durations = [e["duration"] for e in entries if e["duration"] is not None]
            
            if arrived_times:
                arrived_min, arrived_max = min(arrived_times), max(arrived_times)
                arrived_diff = arrived_max - arrived_min
            else:
                arrived_min = arrived_max = arrived_diff = None
            
            if done_times:
                done_min, done_max = min(done_times), max(done_times)
                done_diff = done_max - done_min
            else:
                done_min = done_max = done_diff = None
            
            if durations:
                duration_min, duration_max = min(durations), max(durations)
                duration_diff = duration_max - duration_min
            else:
                duration_min = duration_max = duration_diff = None
            
            time_differences.append({
                "task_id": task_id,
                "arrivedTime": {
                    "min": arrived_min,
                    "max": arrived_max,
                    "difference": arrived_diff,
                    "identical": arrived_diff == 0 if arrived_diff is not None else None
                },
                "doneTime": {
                    "min": done_min,
                    "max": done_max,
                    "difference": done_diff,
                    "identical": done_diff == 0 if done_diff is not None else None
                },
                "duration": {
                    "min": duration_min,
                    "max": duration_max,
                    "difference": duration_diff,
                    "identical": duration_diff == 0 if duration_diff is not None else None
                },
                "count": len(entries)
            })
        
        if time_differences:
            all_times_identical = all(
                (td["arrivedTime"]["identical"] and td["doneTime"]["identical"])
                for td in time_differences
                if td["arrivedTime"]["identical"] is not None and td["doneTime"]["identical"] is not None
            )
        else:
            all_times_identical = True
    else:
        time_differences = []
        all_times_identical = True
    
    print(f"  Negative tasks analysis complete: {len(negative_task_times)} negative tasks found")
    
    return {
        "dataset": results_dir.parent.name,
        "results_dir": str(results_dir),
        "total_files": len(simulation_files),
        "total_tasks_found": len(all_tasks),
        "expected_tasks": len(simulation_files) * len(TARGET_TASK_IDS),
        "missing_tasks_by_file": dict(missing_tasks_by_file),
        "all_identical": all_identical,
        "replicas_identical": replicas_identical,
        "available_resources_identical": available_identical,
        "unique_replicas_states": len(unique_replicas),
        "unique_available_states": len(unique_available),
        "replicas_differences": replicas_differences[:50],  # Limit for readability
        "available_resources_differences": available_differences[:50],
        "total_replicas_differences": len(replicas_differences),
        "total_available_differences": len(available_differences),
        "sample_replicas": normalize_replicas(replicas_list[0]) if replicas_list else {},
        "sample_available": normalize_available_resources(available_list[0]) if available_list else {},
        # Negative tasks analysis
        "negative_task_counts": negative_task_counts,
        "negative_task_counts_identical": negative_task_counts_identical,
        "negative_task_count_min": min(negative_task_counts) if negative_task_counts else 0,
        "negative_task_count_max": max(negative_task_counts) if negative_task_counts else 0,
        "negative_task_count_avg": sum(negative_task_counts) / len(negative_task_counts) if negative_task_counts else 0,
        "negative_task_times": time_differences,
        "negative_times_identical": all_times_identical,
        "total_negative_tasks": len(negative_task_times),
    }


def main():
    """Main function to analyze all datasets."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare systemStateResult across tasks 0-4 in all simulation files in results/ directory"
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        nargs="?",
        help="Path to dataset directory (e.g., ds_00000). If not provided, analyzes all datasets."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of simulation files to analyze (e.g., --limit 50)"
    )
    args = parser.parse_args()
    
    if not BASE_DIR.exists():
        print(f"Error: Base directory not found: {BASE_DIR}")
        return
    
    # Determine which datasets to analyze
    if args.dataset_dir:
        dataset_path = Path(args.dataset_dir)
        if not dataset_path.is_absolute():
            dataset_path = BASE_DIR / args.dataset_dir
        dataset_dirs = [dataset_path] if dataset_path.exists() else []
    else:
        dataset_dirs = sorted([d for d in BASE_DIR.iterdir() if d.is_dir() and d.name.startswith("ds_")])
    
    if not dataset_dirs:
        print(f"No dataset directories found")
        return
    
    print(f"Analyzing {len(dataset_dirs)} dataset(s)...\n")
    
    if args.limit:
        print(f"File limit: {args.limit} files per dataset\n")
    
    all_analyses = []
    for idx, dataset_dir in enumerate(dataset_dirs):
        print(f"\n[{idx + 1}/{len(dataset_dirs)}] Analyzing dataset: {dataset_dir.name}")
        results_dir = dataset_dir / "results"
        analysis = analyze_results_directory(results_dir, file_limit=args.limit)
        if "error" not in analysis:
            all_analyses.append(analysis)
            print(f"  ✓ Completed analysis for {dataset_dir.name}")
        elif "error" in analysis:
            print(f"  ⚠️  Warning: {analysis['error']}")
    
    if not all_analyses:
        print("No valid analyses found")
        return
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Summary")
    print(f"{'='*80}")
    
    for analysis in all_analyses:
        print(f"\nDataset: {analysis['dataset']}")
        print(f"  Results directory: {analysis['results_dir']}")
        print(f"  Total simulation files: {analysis['total_files']}")
        print(f"  Tasks found: {analysis['total_tasks_found']} (expected: {analysis['expected_tasks']})")
        
        if analysis['missing_tasks_by_file']:
            missing_count = sum(len(v) for v in analysis['missing_tasks_by_file'].values())
            print(f"  ⚠️  Missing tasks: {missing_count} across {len(analysis['missing_tasks_by_file'])} files")
            if len(analysis['missing_tasks_by_file']) <= 5:
                for file_name, missing in analysis['missing_tasks_by_file'].items():
                    print(f"    {file_name}: {missing}")
        
        print(f"  Unique replicas states: {analysis['unique_replicas_states']}")
        print(f"  Unique available_resources states: {analysis['unique_available_states']}")
        print(f"  All identical: {analysis['all_identical']}")
        
        if analysis['all_identical']:
            print(f"  ✓ All systemStateResults are identical across all tasks and files!")
        else:
            print(f"  ⚠️  Differences found:")
            print(f"    Replicas differences: {analysis['total_replicas_differences']} pairs")
            print(f"    Available resources differences: {analysis['total_available_differences']} pairs")
            
            # Show sample differences
            if analysis['replicas_differences']:
                print(f"\n    Sample Replicas Differences (first 3):")
                for diff in analysis['replicas_differences'][:3]:
                    print(f"      {diff['task1']} vs {diff['task2']}:")
                    for d in diff['differences'][:2]:
                        print(f"        - {d}")
            
            if analysis['available_resources_differences']:
                print(f"\n    Sample Available Resources Differences (first 3):")
                for diff in analysis['available_resources_differences'][:3]:
                    print(f"      {diff['task1']} vs {diff['task2']}:")
                    for d in diff['differences'][:2]:
                        print(f"        - {d}")
        
        # Show sample state structure
        if analysis.get('sample_replicas'):
            print(f"\n  Sample Replicas (from first task):")
            for task_type, replica_set in analysis['sample_replicas'].items():
                print(f"    {task_type}: {len(replica_set)} replicas")
        
        if analysis.get('sample_available'):
            print(f"\n  Sample Available Resources (from first task):")
            total_available = sum(len(platforms) for platforms in analysis['sample_available'].values())
            print(f"    Total available platforms: {total_available}")
            for node_name, platforms in sorted(analysis['sample_available'].items())[:5]:
                print(f"    {node_name}: {len(platforms)} platforms")
        
        # Negative tasks analysis
        print(f"\n  Negative Tasks (Warmup) Analysis:")
        print(f"    Total negative tasks found: {analysis.get('total_negative_tasks', 0)}")
        print(f"    Negative task counts identical across files: {analysis.get('negative_task_counts_identical', False)}")
        if analysis.get('negative_task_counts'):
            counts = analysis['negative_task_counts']
            print(f"    Negative task count: min={analysis.get('negative_task_count_min')}, max={analysis.get('negative_task_count_max')}, avg={analysis.get('negative_task_count_avg', 0):.1f}")
            if not analysis.get('negative_task_counts_identical'):
                unique_counts = sorted(set(counts))
                print(f"    Unique counts: {unique_counts}")
        
        if analysis.get('negative_task_times'):
            times_identical = analysis.get('negative_times_identical', False)
            print(f"    Negative task times identical: {times_identical}")
            
            if not times_identical:
                print(f"\n    Time Differences (showing first 5 task IDs):")
                for time_diff in analysis['negative_task_times'][:5]:
                    task_id = time_diff['task_id']
                    print(f"      Task {task_id}:")
                    if time_diff['arrivedTime']['difference'] is not None:
                        print(f"        arrivedTime: min={time_diff['arrivedTime']['min']:.6f}, max={time_diff['arrivedTime']['max']:.6f}, diff={time_diff['arrivedTime']['difference']:.6f}")
                    if time_diff['doneTime']['difference'] is not None:
                        print(f"        doneTime: min={time_diff['doneTime']['min']:.6f}, max={time_diff['doneTime']['max']:.6f}, diff={time_diff['doneTime']['difference']:.6f}")
                    if time_diff['duration']['difference'] is not None:
                        print(f"        duration: min={time_diff['duration']['min']:.6f}, max={time_diff['duration']['max']:.6f}, diff={time_diff['duration']['difference']:.6f}")


if __name__ == "__main__":
    main()
