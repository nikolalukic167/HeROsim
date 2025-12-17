#!/usr/bin/env python3
"""
Validate placements from placements.jsonl against system state from optimal_result.json.
Checks if each placement is possible given the available replicas in the system state.
Also checks if all possible combinations (with uniqueness constraint) are present.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from itertools import product

def load_placements(jsonl_path: Path) -> List[Dict]:
    """Load all placements from JSONL file."""
    placements = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                placements.append(data)
            except json.JSONDecodeError:
                continue
    return placements

def load_system_state(optimal_result_path: Path) -> Dict:
    """Load system state from optimal_result.json."""
    with open(optimal_result_path, 'r') as f:
        result = json.load(f)
    
    # Get the last (most recent) system state result
    stats = result.get("stats", {})
    system_state_results = stats.get("systemStateResults", [])
    
    if not system_state_results:
        return None
    
    # Get the last non-null system state
    for state in reversed(system_state_results):
        if state and state.get("replicas"):
            return state
    
    return None

def get_replicas_set(system_state: Dict) -> Dict[str, Set[Tuple[str, int]]]:
    """Extract replicas as sets of (node_name, platform_id) tuples."""
    replicas_by_task = {}
    replicas_dict = system_state.get("replicas", {})
    
    for task_type, replica_list in replicas_dict.items():
        replica_set = set()
        for replica in replica_list:
            if isinstance(replica, list) and len(replica) >= 2:
                node_name = replica[0]
                platform_id = replica[1]
                replica_set.add((node_name, platform_id))
        replicas_by_task[task_type] = replica_set
    
    return replicas_by_task

def get_node_mapping(optimal_result_path: Path) -> Dict[int, str]:
    """Get mapping from node_id to node_name."""
    with open(optimal_result_path, 'r') as f:
        result = json.load(f)
    
    infra_nodes = result.get("config", {}).get("infrastructure", {}).get("nodes", [])
    node_mapping = {}
    
    for i, node in enumerate(infra_nodes):
        node_id = i
        node_name = node.get("node_name", f"node_{i}")
        node_mapping[node_id] = node_name
    
    return node_mapping

def validate_placement(
    placement_plan: Dict,
    replicas_by_task: Dict[str, Set[Tuple[str, int]]],
    node_mapping: Dict[int, str],
    task_results: List[Dict]
) -> Tuple[bool, List[str]]:
    """
    Validate if a placement is possible given the system state.
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Build task_id -> task_type mapping from task_results
    task_id_to_type = {}
    task_id_to_source = {}
    for task_result in task_results:
        task_id = task_result.get("taskId")
        if task_id is not None:
            task_type = task_result.get("taskType", {}).get("name", "unknown")
            source_node = task_result.get("sourceNode", "")
            task_id_to_type[task_id] = task_type
            task_id_to_source[task_id] = source_node
    
    # Validate each task placement
    for task_id_str, placement in placement_plan.items():
        try:
            task_id = int(task_id_str)
        except (ValueError, TypeError):
            errors.append(f"Invalid task_id: {task_id_str}")
            continue
        
        if not isinstance(placement, list) or len(placement) < 2:
            errors.append(f"Task {task_id}: Invalid placement format: {placement}")
            continue
        
        node_id, platform_id = placement[0], placement[1]
        
        # Get node name
        node_name = node_mapping.get(node_id)
        if node_name is None:
            errors.append(f"Task {task_id}: Node ID {node_id} not found in infrastructure")
            continue
        
        # Get task type
        task_type = task_id_to_type.get(task_id)
        if task_type is None:
            errors.append(f"Task {task_id}: Task type not found")
            continue
        
        # Get source node
        source_node = task_id_to_source.get(task_id, "")
        
        # Check if replica exists for this task type
        replicas = replicas_by_task.get(task_type, set())
        replica_key = (node_name, platform_id)
        
        if replica_key not in replicas:
            errors.append(
                f"Task {task_id} ({task_type}): Replica ({node_name}, {platform_id}) not found in system state. "
                f"Available replicas for {task_type}: {len(replicas)}"
            )
            continue
        
        # Check network connectivity (local execution is always allowed)
        if source_node and node_name != source_node:
            # For remote execution, we'd need to check network_map, but for simplicity
            # we just note it's a remote placement
            pass
    
    return len(errors) == 0, errors

def check_uniqueness_constraint(placement_plan: Dict) -> Tuple[bool, List[str]]:
    """
    Check if placement satisfies uniqueness constraint (no two tasks share the same replica).
    Returns: (is_unique, list_of_violations)
    """
    violations = []
    used_replicas = set()
    
    for task_id_str, placement in placement_plan.items():
        if not isinstance(placement, list) or len(placement) < 2:
            continue
        
        node_id, platform_id = placement[0], placement[1]
        replica = (node_id, platform_id)
        
        if replica in used_replicas:
            violations.append(f"Task {task_id_str} uses replica ({node_id}, {platform_id}) which is already used by another task")
        else:
            used_replicas.add(replica)
    
    return len(violations) == 0, violations

def generate_all_possible_combinations(
    workload_events: List[Dict],
    task_types: Dict,
    replicas_by_task: Dict[str, Set[Tuple[str, int]]],
    node_mapping: Dict[int, str],
    infra_nodes: List[Dict]
) -> Set[Tuple[Tuple[int, int], ...]]:
    """
    Generate all possible placement combinations with uniqueness constraint.
    This matches the logic in executecosimulation.py's generate_all_combinations_with_unique_replicas.
    
    Args:
        workload_events: List of workload events (from config.workload.events)
        task_types: Dict of task type configurations
        replicas_by_task: Dict mapping task_type -> set of (node_name, platform_id) tuples
        node_mapping: Dict mapping node_id -> node_name
        infra_nodes: List of infrastructure node configs
    """
    # Build feasible platforms per task (matching executecosimulation.py logic)
    tasks = []
    task_id = 0
    
    for event in workload_events:
        application = event.get('application', {})
        dag = application.get('dag', {})
        
        # Handle both list and dict DAG formats (matching executecosimulation.py)
        if isinstance(dag, list):
            task_type_names = dag
        elif isinstance(dag, dict):
            task_type_names = list(dag.keys())
        else:
            task_type_names = []
        
        source_node_name = event.get('node_name', '')
        
        for task_type_name in task_type_names:
            if task_type_name not in task_types:
                continue
            
            # Get feasible replicas for this task type (from replicas_by_task)
            replicas = replicas_by_task.get(task_type_name, set())
            if not replicas:
                continue
            
            # Filter by network connectivity (matching executecosimulation.py lines 1123-1147)
            feasible_platforms = []
            for node_name, platform_id in replicas:
                # Convert node_name to node_id
                node_id = next((i for i, n in enumerate(infra_nodes) if n.get("node_name") == node_name), None)
                if node_id is None:
                    continue
                
                # Rule 1: Local execution always allowed
                if source_node_name == node_name:
                    feasible_platforms.append((node_id, platform_id))
                    continue
                
                # Rule 2 & 3: Server nodes with network connectivity, reject other client nodes
                if node_name.startswith('client_node'):
                    continue  # Reject other client nodes
                
                # Check network connectivity for server nodes
                node_config = next((n for n in infra_nodes if n.get("node_name") == node_name), None)
                if node_config and source_node_name in node_config.get('network_map', {}):
                    feasible_platforms.append((node_id, platform_id))
            
            if feasible_platforms:
                tasks.append({
                    'task_id': task_id,
                    'feasible_platforms': feasible_platforms
                })
                task_id += 1
    
    # Generate all combinations with uniqueness constraint (recursive)
    all_combinations = set()
    
    def generate_recursive(task_index: int, current_placement: Dict[int, Tuple[int, int]], used_replicas: set):
        if task_index >= len(tasks):
            # All tasks assigned - create sorted tuple
            sorted_tasks = sorted(current_placement.keys())
            combo = tuple((current_placement[t_id]) for t_id in sorted_tasks)
            all_combinations.add(combo)
            return
        
        task = tasks[task_index]
        found_valid = False
        
        for replica in task['feasible_platforms']:
            if replica in used_replicas:
                continue  # Skip if already used (uniqueness constraint)
            
            current_placement[task['task_id']] = replica
            used_replicas.add(replica)
            found_valid = True
            
            generate_recursive(task_index + 1, current_placement, used_replicas)
            
            # Backtrack
            del current_placement[task['task_id']]
            used_replicas.remove(replica)
        
        if not found_valid:
            return  # No valid replica for this task
    
    generate_recursive(0, {}, set())
    return all_combinations

def main():
    # Paths
    base_dir = Path("/root/projects/my-herosim/simulation_data/artifacts/run1100/gnn_datasets/ds_00003")
    placements_jsonl = base_dir / "placements" / "placements.jsonl"
    optimal_result = base_dir / "optimal_result.json"
    
    if not placements_jsonl.exists():
        print(f"Error: {placements_jsonl} not found")
        return
    
    if not optimal_result.exists():
        print(f"Error: {optimal_result} not found")
        return
    
    # Load data
    print("Loading placements from JSONL...")
    placements = load_placements(placements_jsonl)
    print(f"Loaded {len(placements)} placements")
    
    print("\nLoading system state from optimal_result.json...")
    system_state = load_system_state(optimal_result)
    if not system_state:
        print("Error: No valid system state found in optimal_result.json")
        return
    
    # Get workload events and task types (matching executecosimulation.py)
    with open(optimal_result, 'r') as f:
        result = json.load(f)
    task_results = result.get("stats", {}).get("taskResults", [])
    infra_nodes = result.get("config", {}).get("infrastructure", {}).get("nodes", [])
    workload_events = result.get("config", {}).get("workload", {}).get("events", [])
    # Task types are in sample.sim_inputs, not config.simInputs
    sample = result.get("sample", {})
    sim_inputs = sample.get("sim_inputs", {}) if sample else {}
    task_types = sim_inputs.get("task_types", {})
    
    print(f"\nWorkload events: {len(workload_events)} events")
    print(f"Task types: {list(task_types.keys())}")
    
    # Extract replicas
    replicas_by_task = get_replicas_set(system_state)
    print(f"\nReplicas in system state:")
    for task_type, replicas in replicas_by_task.items():
        print(f"  {task_type}: {len(replicas)} replicas")
    
    # Get node mapping
    node_mapping = get_node_mapping(optimal_result)
    print(f"\nNode mapping: {len(node_mapping)} nodes")
    
    # Validate each placement
    print("\n" + "="*80)
    print("VALIDATING PLACEMENTS")
    print("="*80)
    
    valid_count = 0
    invalid_count = 0
    uniqueness_violations = 0
    
    # Convert placements to set of combos for comparison
    placements_combos = set()
    
    for i, placement_data in enumerate(placements):
        placement_plan = placement_data.get("placement_plan", {})
        rtt = placement_data.get("rtt", "N/A")
        
        # Check uniqueness constraint
        is_unique, violations = check_uniqueness_constraint(placement_plan)
        if not is_unique:
            uniqueness_violations += 1
            if uniqueness_violations <= 3:
                print(f"\n⚠️  Placement {i+1} (RTT: {rtt}): UNIQUENESS VIOLATION")
                for violation in violations:
                    print(f"    {violation}")
        
        # Check validity against system state
        is_valid, errors = validate_placement(
            placement_plan,
            replicas_by_task,
            node_mapping,
            task_results
        )
        
        if is_valid and is_unique:
            valid_count += 1
            # Convert to combo format for comparison
            sorted_tasks = sorted(placement_plan.keys(), key=lambda x: int(x))
            combo = tuple(
                (int(placement_plan[t][0]), int(placement_plan[t][1]))
                for t in sorted_tasks
                if isinstance(placement_plan[t], list) and len(placement_plan[t]) >= 2
            )
            if combo:
                placements_combos.add(combo)
        else:
            invalid_count += 1
            if i < 5:  # Show first 5 invalid placements
                print(f"\n✗ Placement {i+1} (RTT: {rtt}): INVALID")
                for error in errors:
                    print(f"    {error}")
    
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"Total placements: {len(placements)}")
    print(f"Valid (system state + uniqueness): {valid_count} ({valid_count/len(placements)*100:.1f}%)")
    print(f"Invalid: {invalid_count} ({invalid_count/len(placements)*100:.1f}%)")
    print(f"Uniqueness violations: {uniqueness_violations} ({uniqueness_violations/len(placements)*100:.1f}%)")
    
    # Check completeness: generate all possible combinations and compare
    print("\n" + "="*80)
    print("CHECKING COMPLETENESS")
    print("="*80)
    print("Generating all possible combinations (with uniqueness constraint)...")
    
    try:
        all_possible_combos = generate_all_possible_combinations(
            workload_events,
            task_types,
            replicas_by_task,
            node_mapping,
            infra_nodes
        )
        print(f"Generated {len(all_possible_combos)} possible combinations")
        print(f"Found {len(placements_combos)} unique combinations in placements.jsonl")
        
        missing_combos = all_possible_combos - placements_combos
        extra_combos = placements_combos - all_possible_combos
        
        print(f"\nMissing combinations: {len(missing_combos)}")
        if missing_combos:
            print("  Sample missing combinations (first 5):")
            for combo in list(missing_combos)[:5]:
                print(f"    {combo}")
        
        print(f"\nExtra combinations (in placements but not in generated): {len(extra_combos)}")
        if extra_combos:
            print("  Sample extra combinations (first 5):")
            for combo in list(extra_combos)[:5]:
                print(f"    {combo}")
        
        if len(missing_combos) == 0 and len(extra_combos) == 0:
            print("\n✓ COMPLETE: All possible combinations are present in placements.jsonl")
        else:
            print(f"\n⚠️  INCOMPLETE: {len(missing_combos)} combinations missing, {len(extra_combos)} extra")
            print("  This means the model can predict valid placements that weren't simulated!")
    except Exception as e:
        print(f"\n⚠️  Error generating all combinations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
