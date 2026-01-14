#!/usr/bin/env python3
"""
Compare GNN and Knative scheduler placements against brute-force optimal placements.

1. Analyze both GNN and Knative placements for each dataset
2. Compare their performance (regret, rank, RTT)
3. Show side-by-side statistics and improvement metrics

Usage:
    python compare_gnn_knative.py <datasets_base>          # Compare all ds_* in directory
    python compare_gnn_knative.py <ds_XXXXX>               # Compare single dataset
    python compare_gnn_knative.py --json <datasets_base>   # Output JSON summary
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import statistics


@dataclass
class AnalysisResult:
    dataset_dir: str
    # Matching
    placement_found_in_bruteforce: bool
    matching_rtt_from_bruteforce: Optional[float]  # "True RTT" - RTT from JSONL lookup
    knative_rtt: float  # RTT from knative simulation
    rtt_difference: Optional[float]  # knative_rtt - matching_rtt_from_bruteforce (scheduler timing diff)
    # Correctness
    all_placements_valid: bool
    validity_errors: List[str]
    # System state comparison
    system_state_matches: bool
    system_state_errors: List[str]
    # Queue/Replica consistency
    queue_replica_consistent: bool
    queue_replica_errors: List[str]
    # Regret stats (using "true RTT" from JSONL when available)
    optimal_rtt: float  # Best RTT from best.json
    true_rtt: Optional[float]  # RTT from JSONL lookup (same as matching_rtt_from_bruteforce)
    regret: float  # true_rtt - optimal_rtt (using JSONL RTT, not knative RTT)
    regret_percent: float  # (true_rtt - optimal_rtt) / optimal_rtt * 100
    rank: int  # 1 = optimal, 2 = second best, etc.
    total_placements: int
    # Optimal placement match
    matches_bf_optimal_placement: bool  # True if knative placement == BF optimal placement
    bf_optimal_rtt: float  # RTT from best.json


@dataclass
class ComparisonResult:
    dataset_dir: str
    gnn_result: Optional[AnalysisResult]
    knative_result: Optional[AnalysisResult]
    # Comparison metrics
    gnn_better: Optional[bool]  # True if GNN has lower regret than Knative
    regret_improvement: Optional[float]  # knative_regret - gnn_regret (positive = GNN better)
    regret_improvement_pct: Optional[float]  # (knative_regret - gnn_regret) / knative_regret * 100
    rank_improvement: Optional[int]  # knative_rank - gnn_rank (positive = GNN better)
    both_found_optimal: Optional[bool]  # True if both found optimal placement
    gnn_found_optimal: Optional[bool]  # True if GNN found optimal
    knative_found_optimal: Optional[bool]  # True if Knative found optimal


def load_json(path: Path) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[dict]:
    results = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def build_node_id_to_name_map(space_config: dict) -> Dict[int, str]:
    """Build mapping from node ID to node name based on space config."""
    node_map = {}
    client_count = space_config['nodes']['client_nodes']['count']
    server_count = space_config['nodes']['server_nodes']['count']
    
    # Client nodes come first (IDs 0 to client_count-1)
    for i in range(client_count):
        node_map[i] = f"client_node{i}"
    
    # Server nodes next (IDs client_count to client_count+server_count-1)
    for i in range(server_count):
        node_map[client_count + i] = f"node{i}"
    
    return node_map


def extract_knative_placement(captured_state: dict) -> Dict[int, Tuple[str, int]]:
    """Extract placement as {task_id: (node_name, platform_id)}"""
    placement = {}
    for tp in captured_state['task_placements']:
        task_id = tp['task_id']
        node_name = tp['execution_node']
        platform_id = int(tp['execution_platform'])
        placement[task_id] = (node_name, platform_id)
    return placement


def convert_bruteforce_placement(bf_placement: dict, node_map: Dict[int, str]) -> Dict[int, Tuple[str, int]]:
    """Convert brute-force placement to {task_id: (node_name, platform_id)}"""
    result = {}
    for task_id_str, (node_id, platform_id) in bf_placement.items():
        task_id = int(task_id_str)
        node_name = node_map.get(node_id, f"unknown_{node_id}")
        result[task_id] = (node_name, platform_id)
    return result


def placements_match(p1: Dict[int, Tuple[str, int]], p2: Dict[int, Tuple[str, int]]) -> bool:
    """Check if two placements are identical."""
    if set(p1.keys()) != set(p2.keys()):
        return False
    for task_id in p1:
        if p1[task_id] != p2[task_id]:
            return False
    return True


# =============================================================================
# Part 1: Check if placement matches any in placements.jsonl
# =============================================================================
def check_placement_match(
    knative_placement: Dict[int, Tuple[str, int]],
    bruteforce_placements: List[dict],
    node_map: Dict[int, str],
    knative_rtt: float
) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    Check if knative placement matches any brute-force placement.
    Returns: (found, matching_rtt, rtt_difference)
    """
    for bf in bruteforce_placements:
        bf_placement = convert_bruteforce_placement(bf['placement_plan'], node_map)
        if placements_match(knative_placement, bf_placement):
            bf_rtt = bf['rtt']
            return True, bf_rtt, knative_rtt - bf_rtt
    return False, None, None


# =============================================================================
# Part 2: Verify placement correctness
# =============================================================================
def verify_placement_correctness(
    captured_state: dict,
    infrastructure: dict,
    space_config: dict
) -> Tuple[bool, List[str]]:
    """
    Verify that each task's placement is valid:
    1. The platform is a valid replica for the task type
    2. Network connectivity exists (or it's local execution)
    3. The chosen platform had the minimum queue among valid replicas
    
    Returns: (all_valid, list_of_errors)
    """
    errors = []
    
    # Build replica map: {task_type -> set of (node_name, platform_id)}
    replica_map = {}
    for task_type, placements in infrastructure['replica_placements'].items():
        replica_map[task_type] = set()
        for p in placements:
            replica_map[task_type].add((p['node_name'], p['platform_id']))
    
    # Build network map: {node_name -> {connected_node_name -> latency}}
    network_maps = infrastructure['network_maps']
    
    for tp in captured_state['task_placements']:
        task_id = tp['task_id']
        task_type = tp['task_type']
        source_node = tp['source_node']
        exec_node = tp['execution_node']
        exec_platform = int(tp['execution_platform'])
        queue_snapshot = tp.get('queue_snapshot_at_scheduling', {})
        
        # Check 1: Is the platform a valid replica?
        if task_type not in replica_map:
            errors.append(f"Task {task_id}: Unknown task type '{task_type}'")
            continue
        
        if (exec_node, exec_platform) not in replica_map[task_type]:
            errors.append(f"Task {task_id}: ({exec_node}, {exec_platform}) is not a valid replica for {task_type}")
            continue
        
        # Check 2: Network connectivity
        is_local = (source_node == exec_node)
        is_server = not exec_node.startswith('client_node')
        
        if not is_local:
            if not is_server:
                # Placed on another client node - not allowed
                errors.append(f"Task {task_id}: Placed on another client node {exec_node} (source: {source_node})")
                continue
            
            # Check network connectivity from server to source
            if exec_node not in network_maps:
                errors.append(f"Task {task_id}: Execution node {exec_node} has no network map")
                continue
            
            if source_node not in network_maps[exec_node]:
                errors.append(f"Task {task_id}: No network connectivity from {exec_node} to {source_node}")
                continue
        
        # Check 3: Queue occupancy - verify the chosen platform exists in the snapshot
        # Note: We can't verify "minimum queue" because the snapshot is taken at batch start,
        # but placements happen sequentially within the batch, changing queue states.
        # We just verify the platform was a valid option.
        if queue_snapshot:
            exec_key = f"{exec_node}:{exec_platform}"
            if exec_key not in queue_snapshot:
                errors.append(
                    f"Task {task_id}: Execution platform {exec_key} not in valid replicas snapshot"
                )
    
    return len(errors) == 0, errors


# =============================================================================
# Part 2b: Compare system state with optimal_result.json
# =============================================================================
def compare_system_states(
    captured_state: dict,
    dataset_dir: Path
) -> Tuple[bool, List[str]]:
    """
    Compare system state from system_state_captured.json with systemStateResult
    from optimal_result.json. They should match (same replicas, available_resources).
    
    Returns: (matches, list_of_errors)
    """
    errors = []
    
    optimal_file = dataset_dir / "optimal_result.json"
    if not optimal_file.exists():
        return True, []  # Skip if no optimal_result.json
    
    try:
        with open(optimal_file, 'r') as f:
            optimal_data = json.load(f)
    except Exception as e:
        errors.append(f"Failed to load optimal_result.json: {e}")
        return False, errors
    
    # Find first non-null systemStateResult in taskResults
    task_results = optimal_data.get('stats', {}).get('taskResults', [])
    optimal_ssr = None
    for tr in task_results:
        ssr = tr.get('systemStateResult')
        if ssr:
            optimal_ssr = ssr
            break
    
    if not optimal_ssr:
        return True, []  # No systemStateResult to compare
    
    # Compare replicas
    captured_replicas = captured_state.get('replicas', {})
    optimal_replicas = optimal_ssr.get('replicas', {})
    
    # Normalize replicas to sets of tuples for comparison
    def normalize_replicas(replicas_dict):
        result = {}
        for task_type, replica_list in replicas_dict.items():
            result[task_type] = set(tuple(r) for r in replica_list)
        return result
    
    captured_norm = normalize_replicas(captured_replicas)
    optimal_norm = normalize_replicas(optimal_replicas)
    
    # Check task types match
    if set(captured_norm.keys()) != set(optimal_norm.keys()):
        errors.append(
            f"Task types mismatch: captured={set(captured_norm.keys())}, "
            f"optimal={set(optimal_norm.keys())}"
        )
    else:
        # Check each task type's replicas
        for task_type in captured_norm:
            if captured_norm[task_type] != optimal_norm[task_type]:
                captured_only = captured_norm[task_type] - optimal_norm[task_type]
                optimal_only = optimal_norm[task_type] - captured_norm[task_type]
                if captured_only:
                    errors.append(f"Replicas for {task_type}: captured has extra {captured_only}")
                if optimal_only:
                    errors.append(f"Replicas for {task_type}: optimal has extra {optimal_only}")
    
    # Compare available_resources
    captured_resources = captured_state.get('available_resources', {})
    optimal_resources = optimal_ssr.get('available_resources', {})
    
    # Normalize to sets for comparison
    def normalize_resources(resources_dict):
        result = {}
        for node, platforms in resources_dict.items():
            result[node] = set(platforms)
        return result
    
    captured_res_norm = normalize_resources(captured_resources)
    optimal_res_norm = normalize_resources(optimal_resources)
    
    # Check nodes match
    if set(captured_res_norm.keys()) != set(optimal_res_norm.keys()):
        captured_only = set(captured_res_norm.keys()) - set(optimal_res_norm.keys())
        optimal_only = set(optimal_res_norm.keys()) - set(captured_res_norm.keys())
        if captured_only:
            errors.append(f"Available resources: captured has extra nodes {captured_only}")
        if optimal_only:
            errors.append(f"Available resources: optimal has extra nodes {optimal_only}")
    else:
        # Check each node's platforms
        for node in captured_res_norm:
            if captured_res_norm[node] != optimal_res_norm[node]:
                captured_only = captured_res_norm[node] - optimal_res_norm[node]
                optimal_only = optimal_res_norm[node] - captured_res_norm[node]
                if captured_only or optimal_only:
                    errors.append(
                        f"Available resources for {node}: "
                        f"captured={sorted(captured_res_norm[node])}, "
                        f"optimal={sorted(optimal_res_norm[node])}"
                    )
    
    return len(errors) == 0, errors


# =============================================================================
# Part 2c: Check queue/replica consistency
# =============================================================================
def check_queue_replica_consistency(
    captured_state: dict,
    infrastructure: dict
) -> Tuple[bool, List[str]]:
    """
    Check that:
    1. All replicas have non-empty queues (queue_distributions)
    2. All queues in queue_distributions have corresponding replicas
    
    Returns: (consistent, list_of_errors)
    """
    errors = []
    
    queue_distributions = infrastructure.get('queue_distributions', {})
    replica_placements = infrastructure.get('replica_placements', {})
    
    # Build set of all replica keys: {task_type -> set of "node_name:platform_id"}
    replica_keys_by_type = {}
    for task_type, placements in replica_placements.items():
        replica_keys_by_type[task_type] = set()
        for p in placements:
            key = f"{p['node_name']}:{p['platform_id']}"
            replica_keys_by_type[task_type].add(key)
    
    # Build set of all queue keys: {task_type -> set of "node_name:platform_id"}
    queue_keys_by_type = {}
    for task_type, queues in queue_distributions.items():
        queue_keys_by_type[task_type] = set(queues.keys())
    
    # Check 1: All replicas should have queues
    for task_type, replica_keys in replica_keys_by_type.items():
        queue_keys = queue_keys_by_type.get(task_type, set())
        replicas_without_queues = replica_keys - queue_keys
        if replicas_without_queues:
            errors.append(
                f"{task_type}: {len(replicas_without_queues)} replicas without queues: "
                f"{list(replicas_without_queues)[:3]}..."
            )
    
    # Check 2: All queues should have replicas
    for task_type, queue_keys in queue_keys_by_type.items():
        replica_keys = replica_keys_by_type.get(task_type, set())
        queues_without_replicas = queue_keys - replica_keys
        if queues_without_replicas:
            errors.append(
                f"{task_type}: {len(queues_without_replicas)} queues without replicas: "
                f"{list(queues_without_replicas)[:3]}..."
            )
    
    # Check 3: All queues should be non-empty (have queue length > 0)
    empty_queues = []
    for task_type, queues in queue_distributions.items():
        for key, length in queues.items():
            if length == 0:
                empty_queues.append(f"{task_type}:{key}")
    
    if empty_queues:
        errors.append(f"{len(empty_queues)} queues with length 0: {empty_queues[:5]}...")
    
    return len(errors) == 0, errors


# =============================================================================
# Part 3: Calculate regret and stats
# =============================================================================
def get_optimal_rtt(dataset_dir: Path, bruteforce_placements: List[dict]) -> float:
    """
    Get optimal RTT from best.json (preferred) or minimum from placements.jsonl.
    """
    best_file = dataset_dir / "best.json"
    if best_file.exists():
        try:
            with open(best_file, 'r') as f:
                best_data = json.load(f)
                return best_data['rtt']
        except Exception:
            pass
    
    # Fallback to minimum from placements
    if bruteforce_placements:
        return min(p['rtt'] for p in bruteforce_placements)
    
    return 0.0


def get_bf_optimal_placement(dataset_dir: Path, bruteforce_placements: List[dict]) -> Optional[dict]:
    """
    Get the optimal placement from best.json or minimum RTT from placements.jsonl.
    Returns the placement dict with 'placement_plan' and 'rtt'.
    """
    best_file = dataset_dir / "best.json"
    if best_file.exists():
        try:
            with open(best_file, 'r') as f:
                best_data = json.load(f)
                best_filename = best_data.get('file', '')
                best_rtt = best_data.get('rtt', 0)
                
                # Find the matching placement in placements.jsonl by RTT
                for p in bruteforce_placements:
                    if abs(p['rtt'] - best_rtt) < 0.0001:
                        return p
        except Exception:
            pass
    
    # Fallback to minimum RTT placement
    if bruteforce_placements:
        return min(bruteforce_placements, key=lambda p: p['rtt'])
    
    return None


def check_matches_bf_optimal(
    knative_placement: Dict[int, Tuple[str, int]],
    dataset_dir: Path,
    bruteforce_placements: List[dict],
    node_map: Dict[int, str]
) -> bool:
    """Check if knative placement matches the BF optimal placement."""
    bf_optimal = get_bf_optimal_placement(dataset_dir, bruteforce_placements)
    if bf_optimal is None:
        return False
    
    bf_optimal_placement = convert_bruteforce_placement(bf_optimal['placement_plan'], node_map)
    return placements_match(knative_placement, bf_optimal_placement)


def calculate_regret_stats(
    true_rtt: Optional[float],
    bruteforce_placements: List[dict],
    dataset_dir: Path
) -> Tuple[float, float, float, int, int]:
    """
    Calculate regret statistics using true_rtt (from JSONL lookup) and optimal RTT from best.json.
    Returns: (optimal_rtt, regret, regret_percent, rank, total_placements)
    
    Rank is 1-indexed position based on true_rtt in sorted BF RTTs.
    Rank 1 = best (true_rtt is <= all BF RTTs)
    """
    if not bruteforce_placements or true_rtt is None:
        optimal_rtt = get_optimal_rtt(dataset_dir, bruteforce_placements) if bruteforce_placements else 0.0
        return optimal_rtt, 0.0, 0.0, 1, len(bruteforce_placements) if bruteforce_placements else 0
    
    # Get optimal RTT from best.json (or minimum from placements)
    optimal_rtt = get_optimal_rtt(dataset_dir, bruteforce_placements)
    
    # Calculate regret using true_rtt (from JSONL lookup)
    regret = true_rtt - optimal_rtt
    regret_percent = (regret / optimal_rtt) * 100 if optimal_rtt > 0 else 0.0
    
    # Sort by RTT to find rank
    sorted_rtts = sorted(p['rtt'] for p in bruteforce_placements)
    
    # Find rank (1-indexed) - count how many BF placements are better than true_rtt
    # Rank 1 means placement is best or tied for best
    rank = 1
    for rtt in sorted_rtts:
        if rtt < true_rtt - 0.0001:  # BF placement is clearly better
            rank += 1
        else:
            break  # true_rtt is better or equal to this and all remaining
    
    return optimal_rtt, regret, regret_percent, rank, len(bruteforce_placements)


def compare_results(gnn_result: Optional[AnalysisResult], knative_result: Optional[AnalysisResult], dataset_dir: str) -> ComparisonResult:
    """Compare GNN and Knative results for a dataset."""
    if gnn_result is None and knative_result is None:
        return ComparisonResult(
            dataset_dir=dataset_dir,
            gnn_result=None,
            knative_result=None,
            gnn_better=None,
            regret_improvement=None,
            regret_improvement_pct=None,
            rank_improvement=None,
            both_found_optimal=None,
            gnn_found_optimal=None,
            knative_found_optimal=None
        )
    
    # Calculate comparison metrics
    gnn_better = None
    regret_improvement = None
    regret_improvement_pct = None
    rank_improvement = None
    both_found_optimal = None
    gnn_found_optimal = None
    knative_found_optimal = None
    
    if gnn_result and knative_result:
        # Both have results - compare
        if gnn_result.true_rtt is not None and knative_result.true_rtt is not None:
            # Compare regret
            gnn_better = gnn_result.regret < knative_result.regret
            regret_improvement = knative_result.regret - gnn_result.regret
            
            # Calculate percentage improvement with proper handling of edge cases
            # Use a threshold to avoid division by very small numbers
            THRESHOLD = 0.001  # 1ms threshold
            if abs(knative_result.regret) > THRESHOLD:
                # Normal case: calculate percentage
                regret_improvement_pct = (regret_improvement / knative_result.regret) * 100
                # Cap at reasonable values to avoid absurd percentages
                regret_improvement_pct = max(-1000.0, min(1000.0, regret_improvement_pct))
            elif abs(regret_improvement) < THRESHOLD:
                # Both regrets are very small and similar - essentially tied
                regret_improvement_pct = 0.0
            elif regret_improvement > 0:
                # GNN is better, but knative regret is very small
                # Use absolute improvement instead of percentage
                regret_improvement_pct = 100.0  # Cap at 100% improvement
            else:
                # GNN is worse, but knative regret is very small
                regret_improvement_pct = -100.0  # Cap at -100% (worse)
            
            # Compare rank
            rank_improvement = knative_result.rank - gnn_result.rank
        
        # Check optimal placement matches
        gnn_found_optimal = gnn_result.matches_bf_optimal_placement
        knative_found_optimal = knative_result.matches_bf_optimal_placement
        both_found_optimal = gnn_found_optimal and knative_found_optimal
    elif gnn_result:
        gnn_found_optimal = gnn_result.matches_bf_optimal_placement
    elif knative_result:
        knative_found_optimal = knative_result.matches_bf_optimal_placement
    
    return ComparisonResult(
        dataset_dir=dataset_dir,
        gnn_result=gnn_result,
        knative_result=knative_result,
        gnn_better=gnn_better,
        regret_improvement=regret_improvement,
        regret_improvement_pct=regret_improvement_pct,
        rank_improvement=rank_improvement,
        both_found_optimal=both_found_optimal,
        gnn_found_optimal=gnn_found_optimal,
        knative_found_optimal=knative_found_optimal
    )


def analyze_dataset(dataset_dir: Path, quiet: bool = False, scheduler_type: str = "knative") -> Optional[AnalysisResult]:
    """Analyze a single dataset.
    
    Args:
        dataset_dir: Path to the dataset directory
        quiet: If True, suppress skip messages
        scheduler_type: "knative" or "gnn" - determines which system_state file to use
    """
    # Select the appropriate system state file based on scheduler type
    if scheduler_type == "gnn":
        captured_file = dataset_dir / "system_state_gnn.json"
    else:
        # Default to knative (unique version)
        captured_file = dataset_dir / "system_state_captured_unique.json"
    
    placements_file = dataset_dir / "placements" / "placements.jsonl"
    infra_file = dataset_dir / "infrastructure.json"
    space_file = dataset_dir / "space_with_network.json"
    
    # Check required files
    if not captured_file.exists():
        if not quiet:
            print(f"  Skipping {dataset_dir.name}: No {captured_file.name}", file=sys.stderr)
        return None
    if not placements_file.exists():
        if not quiet:
            print(f"  Skipping {dataset_dir.name}: No placements.jsonl", file=sys.stderr)
        return None
    if not infra_file.exists():
        if not quiet:
            print(f"  Skipping {dataset_dir.name}: No infrastructure.json", file=sys.stderr)
        return None
    if not space_file.exists():
        if not quiet:
            print(f"  Skipping {dataset_dir.name}: No space_with_network.json", file=sys.stderr)
        return None
    
    # Load data
    captured_state = load_json(captured_file)
    bruteforce_placements = load_jsonl(placements_file)
    infrastructure = load_json(infra_file)
    space_config = load_json(space_file)
    
    knative_rtt = captured_state['total_rtt']
    knative_placement = extract_knative_placement(captured_state)
    node_map = build_node_id_to_name_map(space_config)
    
    # Part 1: Check placement match
    found, bf_rtt, rtt_diff = check_placement_match(
        knative_placement, bruteforce_placements, node_map, knative_rtt
    )
    
    # Part 2: Verify correctness
    all_valid, errors = verify_placement_correctness(captured_state, infrastructure, space_config)
    
    # Part 2b: Compare system state with optimal_result.json
    state_matches, state_errors = compare_system_states(captured_state, dataset_dir)
    
    # Part 2c: Check queue/replica consistency
    queue_consistent, queue_errors = check_queue_replica_consistency(captured_state, infrastructure)
    
    # Part 3: Calculate regret using true_rtt (from JSONL lookup)
    # bf_rtt is the "true RTT" - the RTT from the JSONL for this exact placement
    optimal_rtt, regret, regret_pct, rank, total = calculate_regret_stats(
        bf_rtt, bruteforce_placements, dataset_dir  # Use bf_rtt (true RTT from JSONL)
    )
    
    # Check if knative found the BF optimal placement
    matches_bf_optimal = check_matches_bf_optimal(
        knative_placement, dataset_dir, bruteforce_placements, node_map
    )
    bf_optimal_rtt = get_optimal_rtt(dataset_dir, bruteforce_placements)
    
    return AnalysisResult(
        dataset_dir=dataset_dir.name,
        placement_found_in_bruteforce=found,
        matching_rtt_from_bruteforce=bf_rtt,  # True RTT from JSONL
        knative_rtt=knative_rtt,  # RTT from knative simulation
        rtt_difference=rtt_diff,  # Scheduler timing difference
        all_placements_valid=all_valid,
        validity_errors=errors,
        system_state_matches=state_matches,
        system_state_errors=state_errors,
        queue_replica_consistent=queue_consistent,
        queue_replica_errors=queue_errors,
        optimal_rtt=optimal_rtt,
        true_rtt=bf_rtt,  # RTT from JSONL lookup
        regret=regret,  # true_rtt - optimal_rtt
        regret_percent=regret_pct,
        rank=rank,
        total_placements=total,
        matches_bf_optimal_placement=matches_bf_optimal,
        bf_optimal_rtt=bf_optimal_rtt
    )


def print_complexity_analysis(results: List[AnalysisResult]):
    """Print analysis of how knative performance varies with problem complexity."""
    results_with_true_rtt = [r for r in results if r.true_rtt is not None]
    if len(results_with_true_rtt) < 10:
        return  # Not enough data
    
    print("\n" + "=" * 80)
    print("COMPLEXITY ANALYSIS: How knative scales with problem size")
    print("=" * 80)
    
    # Sort by placement space size
    sorted_results = sorted(results_with_true_rtt, key=lambda r: r.total_placements)
    
    # Divide into quartiles by complexity
    n = len(sorted_results)
    q1 = sorted_results[:n//4]
    q2 = sorted_results[n//4:n//2]
    q3 = sorted_results[n//2:3*n//4]
    q4 = sorted_results[3*n//4:]
    
    quartiles = [
        ("Q1 (Simplest)", q1),
        ("Q2", q2),
        ("Q3", q3),
        ("Q4 (Most Complex)", q4),
    ]
    
    print("\n┌─────────────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐")
    print("│ Complexity Quartile │ Placements │ Mean Regret│ Regret %   │ Optimal %  │ Top-10 %   │")
    print("├─────────────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤")
    
    for name, quartile in quartiles:
        if not quartile:
            continue
        avg_placements = statistics.mean(r.total_placements for r in quartile)
        mean_regret = statistics.mean(r.regret for r in quartile)
        mean_regret_pct = statistics.mean(r.regret_percent for r in quartile)
        optimal_pct = sum(1 for r in quartile if r.rank == 1) / len(quartile) * 100
        top10_pct = sum(1 for r in quartile if r.rank <= 10) / len(quartile) * 100
        
        print(f"│ {name:<19} │ {avg_placements:>10.0f} │ {mean_regret:>9.3f}s │ {mean_regret_pct:>9.1f}% │ {optimal_pct:>9.1f}% │ {top10_pct:>9.1f}% │")
    
    print("└─────────────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘")
    
    # Correlation analysis
    placements = [r.total_placements for r in results_with_true_rtt]
    regrets = [r.regret for r in results_with_true_rtt]
    regret_pcts = [r.regret_percent for r in results_with_true_rtt]
    ranks = [r.rank for r in results_with_true_rtt]
    
    # Simple correlation (Pearson)
    def correlation(x, y):
        n = len(x)
        if n < 2:
            return 0
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        den_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
        den_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5
        if den_x == 0 or den_y == 0:
            return 0
        return num / (den_x * den_y)
    
    corr_regret = correlation(placements, regrets)
    corr_regret_pct = correlation(placements, regret_pcts)
    corr_rank = correlation(placements, ranks)
    
    print(f"\nCorrelation with placement space size:")
    print(f"  Regret (seconds):  {corr_regret:+.3f}")
    print(f"  Regret (%):        {corr_regret_pct:+.3f}")
    print(f"  Rank:              {corr_rank:+.3f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if corr_regret > 0.3:
        print(f"  ⚠️  Knative regret INCREASES with problem complexity (r={corr_regret:.2f})")
        print(f"     → GNN has more room to improve on complex problems")
    elif corr_regret < -0.3:
        print(f"  ✓  Knative regret DECREASES with problem complexity (r={corr_regret:.2f})")
    else:
        print(f"  →  Knative regret is relatively stable across complexity (r={corr_regret:.2f})")
    
    # Show worst cases (high complexity, high regret)
    high_complexity_high_regret = [
        r for r in results_with_true_rtt 
        if r.total_placements > statistics.median(placements) and r.regret_percent > 50
    ]
    if high_complexity_high_regret:
        print(f"\n  High-complexity cases with >50% regret: {len(high_complexity_high_regret)}")
        print(f"     These are prime targets for GNN improvement")


def print_summary(results: List[AnalysisResult]):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    n = len(results)
    print(f"\nTotal datasets analyzed: {n}")
    
    # Part 1: Placement matching
    matched = sum(1 for r in results if r.placement_found_in_bruteforce)
    print(f"\n--- Part 1: Placement Matching ---")
    print(f"Placements found in brute-force: {matched}/{n} ({matched/n*100:.1f}%)")
    
    if matched > 0:
        rtt_diffs = [r.rtt_difference for r in results if r.rtt_difference is not None]
        print(f"Scheduler timing diff (knative_sim_rtt - true_rtt):")
        print(f"  Mean: {statistics.mean(rtt_diffs):.6f}s")
        print(f"  Min:  {min(rtt_diffs):.6f}s")
        print(f"  Max:  {max(rtt_diffs):.6f}s")
    
    # Part 2: Validity
    valid = sum(1 for r in results if r.all_placements_valid)
    print(f"\n--- Part 2: Placement Validity ---")
    print(f"All placements valid: {valid}/{n} ({valid/n*100:.1f}%)")
    
    if valid < n:
        print("\nDatasets with validity errors:")
        for r in results:
            if not r.all_placements_valid:
                print(f"  {r.dataset_dir}:")
                for err in r.validity_errors[:3]:
                    print(f"    - {err}")
                if len(r.validity_errors) > 3:
                    print(f"    ... and {len(r.validity_errors) - 3} more errors")
    
    # Part 2b: System state comparison
    state_match = sum(1 for r in results if r.system_state_matches)
    print(f"\n--- Part 2b: System State Comparison ---")
    print(f"System state matches optimal_result.json: {state_match}/{n} ({state_match/n*100:.1f}%)")
    
    if state_match < n:
        print("\nDatasets with system state mismatches:")
        for r in results:
            if not r.system_state_matches:
                print(f"  {r.dataset_dir}:")
                for err in r.system_state_errors[:3]:
                    print(f"    - {err}")
    
    # Part 2c: Queue/Replica consistency (simplified - just percentage)
    queue_consistent = sum(1 for r in results if r.queue_replica_consistent)
    print(f"\n--- Part 2c: Queue/Replica Consistency ---")
    print(f"Queue/Replica consistent: {queue_consistent}/{n} ({queue_consistent/n*100:.1f}%)")
    
    # Part 3: Regret stats (using TRUE RTT from JSONL lookup)
    print(f"\n--- Part 3: Regret Statistics (using True RTT from JSONL) ---")
    
    # Filter to only results with true_rtt available
    results_with_true_rtt = [r for r in results if r.true_rtt is not None]
    if not results_with_true_rtt:
        print("  No results with true RTT available")
        return
    
    regrets = [r.regret for r in results_with_true_rtt]
    regret_pcts = [r.regret_percent for r in results_with_true_rtt]
    ranks = [r.rank for r in results_with_true_rtt]
    totals = [r.total_placements for r in results_with_true_rtt]
    n_true = len(results_with_true_rtt)
    
    # Count how many times knative found the exact BF optimal placement
    optimal_placement_count = sum(1 for r in results_with_true_rtt if r.matches_bf_optimal_placement)
    print(f"Knative found BF optimal placement: {optimal_placement_count}/{n_true} ({optimal_placement_count/n_true*100:.1f}%)")
    
    # Zero regret = found optimal
    zero_regret_count = sum(1 for r in regrets if abs(r) < 0.001)  # Within 1ms of optimal
    
    print(f"\nRegret (true_rtt - optimal_rtt):")
    print(f"  Mean: {statistics.mean(regrets):.6f}s")
    print(f"  Std:  {statistics.stdev(regrets) if len(regrets) > 1 else 0:.6f}s")
    print(f"  Min:  {min(regrets):.6f}s")
    print(f"  Max:  {max(regrets):.6f}s")
    print(f"  Zero regret (within 1ms): {zero_regret_count}/{n_true} ({zero_regret_count/n_true*100:.1f}%)")
    
    print(f"\nRegret %:")
    print(f"  Mean: {statistics.mean(regret_pcts):.2f}%")
    print(f"  Std:  {statistics.stdev(regret_pcts) if len(regret_pcts) > 1 else 0:.2f}%")
    print(f"  Min:  {min(regret_pcts):.2f}%")
    print(f"  Max:  {max(regret_pcts):.2f}%")
    
    optimal_count = sum(1 for r in results_with_true_rtt if r.rank == 1)
    top5_count = sum(1 for r in results_with_true_rtt if r.rank <= 5)
    top10_count = sum(1 for r in results_with_true_rtt if r.rank <= 10)
    
    print(f"\nRank distribution:")
    print(f"  Optimal (rank=1): {optimal_count}/{n_true} ({optimal_count/n_true*100:.1f}%)")
    print(f"  Top 5:            {top5_count}/{n_true} ({top5_count/n_true*100:.1f}%)")
    print(f"  Top 10:           {top10_count}/{n_true} ({top10_count/n_true*100:.1f}%)")
    print(f"  Mean rank:        {statistics.mean(ranks):.1f}")
    print(f"  Median rank:      {statistics.median(ranks):.1f}")
    
    if totals:
        print(f"\nPlacement space size:")
        print(f"  Mean:   {statistics.mean(totals):.0f} placements")
        print(f"  Median: {statistics.median(totals):.0f} placements")
        print(f"  Min:    {min(totals)} placements")
        print(f"  Max:    {max(totals)} placements")
    
    # Percentile rank (how well knative does relative to total options)
    percentile_ranks = [(r.rank / r.total_placements * 100) if r.total_placements > 0 else 0 
                        for r in results_with_true_rtt]
    print(f"\nPercentile rank (lower is better):")
    print(f"  Mean:   {statistics.mean(percentile_ranks):.2f}%")
    print(f"  Median: {statistics.median(percentile_ranks):.2f}%")


def analyze_and_compare_dataset(dataset_dir: Path, quiet: bool = False) -> Optional[ComparisonResult]:
    """Analyze both GNN and Knative for a dataset and compare them."""
    gnn_result = analyze_dataset(dataset_dir, quiet=quiet, scheduler_type="gnn")
    knative_result = analyze_dataset(dataset_dir, quiet=quiet, scheduler_type="knative")
    
    if gnn_result is None and knative_result is None:
        return None
    
    return compare_results(gnn_result, knative_result, dataset_dir.name)


def print_comparison_summary(comparisons: List[ComparisonResult]):
    """Print summary comparing GNN vs Knative."""
    print("\n" + "=" * 80)
    print("GNN vs KNAIVE COMPARISON SUMMARY")
    print("=" * 80)
    
    n = len(comparisons)
    print(f"\nTotal datasets: {n}")
    
    # Count datasets with both results
    both_results = [c for c in comparisons if c.gnn_result and c.knative_result]
    n_both = len(both_results)
    print(f"Datasets with both GNN and Knative results: {n_both}")
    
    if n_both == 0:
        print("\nNo datasets have both GNN and Knative results to compare.")
        return
    
    # Count GNN-only and Knative-only
    gnn_only = [c for c in comparisons if c.gnn_result and not c.knative_result]
    knative_only = [c for c in comparisons if c.knative_result and not c.gnn_result]
    
    if gnn_only:
        print(f"GNN-only datasets: {len(gnn_only)}")
    if knative_only:
        print(f"Knative-only datasets: {len(knative_only)}")
    
    # Comparison metrics (only for datasets with both)
    gnn_better_count = sum(1 for c in both_results if c.gnn_better is True)
    knative_better_count = sum(1 for c in both_results if c.gnn_better is False)
    tie_count = sum(1 for c in both_results if c.gnn_better is None)
    
    print(f"\n--- Performance Comparison ---")
    print(f"GNN better: {gnn_better_count}/{n_both} ({gnn_better_count/n_both*100:.1f}%)")
    print(f"Knative better: {knative_better_count}/{n_both} ({knative_better_count/n_both*100:.1f}%)")
    if tie_count > 0:
        print(f"Tie/Unknown: {tie_count}/{n_both} ({tie_count/n_both*100:.1f}%)")
    
    # Regret improvement statistics
    regret_improvements = [c.regret_improvement for c in both_results if c.regret_improvement is not None]
    regret_improvements_pct = [c.regret_improvement_pct for c in both_results if c.regret_improvement_pct is not None]
    
    if regret_improvements:
        print(f"\n--- Regret Improvement (Knative - GNN, positive = GNN better) ---")
        print(f"Mean improvement: {statistics.mean(regret_improvements):.6f}s ({statistics.mean(regret_improvements_pct):.2f}%)")
        print(f"Median improvement: {statistics.median(regret_improvements):.6f}s ({statistics.median(regret_improvements_pct):.2f}%)")
        if len(regret_improvements) > 1:
            print(f"Std dev: {statistics.stdev(regret_improvements):.6f}s ({statistics.stdev(regret_improvements_pct):.2f}%)")
        print(f"Min improvement: {min(regret_improvements):.6f}s ({min(regret_improvements_pct):.2f}%)")
        print(f"Max improvement: {max(regret_improvements):.6f}s ({max(regret_improvements_pct):.2f}%)")
        
        # Count improvements > 0 (GNN better)
        positive_improvements = sum(1 for imp in regret_improvements if imp > 0.0001)
        negative_improvements = sum(1 for imp in regret_improvements if imp < -0.0001)
        tied_improvements = len(regret_improvements) - positive_improvements - negative_improvements
        print(f"GNN better: {positive_improvements}/{len(regret_improvements)} ({positive_improvements/len(regret_improvements)*100:.1f}%)")
        print(f"Knative better: {negative_improvements}/{len(regret_improvements)} ({negative_improvements/len(regret_improvements)*100:.1f}%)")
        if tied_improvements > 0:
            print(f"Tied: {tied_improvements}/{len(regret_improvements)} ({tied_improvements/len(regret_improvements)*100:.1f}%)")
    
    # Rank improvement
    rank_improvements = [c.rank_improvement for c in both_results if c.rank_improvement is not None]
    if rank_improvements:
        print(f"\n--- Rank Improvement (Knative rank - GNN rank, positive = GNN better) ---")
        print(f"Mean rank improvement: {statistics.mean(rank_improvements):.1f}")
        print(f"Median rank improvement: {statistics.median(rank_improvements):.1f}")
        if len(rank_improvements) > 1:
            print(f"Std dev: {statistics.stdev(rank_improvements):.1f}")
        print(f"Min rank improvement: {min(rank_improvements)}")
        print(f"Max rank improvement: {max(rank_improvements)}")
        positive_rank_improvements = sum(1 for imp in rank_improvements if imp > 0)
        negative_rank_improvements = sum(1 for imp in rank_improvements if imp < 0)
        tied_rank_improvements = len(rank_improvements) - positive_rank_improvements - negative_rank_improvements
        print(f"GNN better rank: {positive_rank_improvements}/{len(rank_improvements)} ({positive_rank_improvements/len(rank_improvements)*100:.1f}%)")
        print(f"Knative better rank: {negative_rank_improvements}/{len(rank_improvements)} ({negative_rank_improvements/len(rank_improvements)*100:.1f}%)")
        if tied_rank_improvements > 0:
            print(f"Tied rank: {tied_rank_improvements}/{len(rank_improvements)} ({tied_rank_improvements/len(rank_improvements)*100:.1f}%)")
    
    # Optimal placement finding
    gnn_optimal_count = sum(1 for c in both_results if c.gnn_found_optimal is True)
    knative_optimal_count = sum(1 for c in both_results if c.knative_found_optimal is True)
    both_optimal_count = sum(1 for c in both_results if c.both_found_optimal is True)
    
    print(f"\n--- Optimal Placement Finding ---")
    print(f"GNN found optimal: {gnn_optimal_count}/{n_both} ({gnn_optimal_count/n_both*100:.1f}%)")
    print(f"Knative found optimal: {knative_optimal_count}/{n_both} ({knative_optimal_count/n_both*100:.1f}%)")
    print(f"Both found optimal: {both_optimal_count}/{n_both} ({both_optimal_count/n_both*100:.1f}%)")
    
    # Individual statistics
    print(f"\n--- Individual Statistics ---")
    
    # GNN statistics
    gnn_results = [c.gnn_result for c in comparisons if c.gnn_result]
    if gnn_results:
        gnn_with_rtt = [r for r in gnn_results if r.true_rtt is not None]
        if gnn_with_rtt:
            gnn_regrets = [r.regret for r in gnn_with_rtt]
            gnn_regret_pcts = [r.regret_percent for r in gnn_with_rtt]
            gnn_ranks = [r.rank for r in gnn_with_rtt]
            print(f"\nGNN (n={len(gnn_with_rtt)}):")
            print(f"  Regret: mean={statistics.mean(gnn_regrets):.6f}s ({statistics.mean(gnn_regret_pcts):.2f}%), median={statistics.median(gnn_regrets):.6f}s ({statistics.median(gnn_regret_pcts):.2f}%)")
            if len(gnn_regrets) > 1:
                print(f"  Regret std dev: {statistics.stdev(gnn_regrets):.6f}s ({statistics.stdev(gnn_regret_pcts):.2f}%)")
            print(f"  Rank: mean={statistics.mean(gnn_ranks):.1f}, median={statistics.median(gnn_ranks):.1f}")
            if len(gnn_ranks) > 1:
                print(f"  Rank std dev: {statistics.stdev(gnn_ranks):.1f}")
            print(f"  Optimal count: {sum(1 for r in gnn_with_rtt if r.matches_bf_optimal_placement)}/{len(gnn_with_rtt)} ({sum(1 for r in gnn_with_rtt if r.matches_bf_optimal_placement)/len(gnn_with_rtt)*100:.1f}%)")
    
    # Knative statistics
    knative_results = [c.knative_result for c in comparisons if c.knative_result]
    if knative_results:
        knative_with_rtt = [r for r in knative_results if r.true_rtt is not None]
        if knative_with_rtt:
            knative_regrets = [r.regret for r in knative_with_rtt]
            knative_regret_pcts = [r.regret_percent for r in knative_with_rtt]
            knative_ranks = [r.rank for r in knative_with_rtt]
            print(f"\nKnative (n={len(knative_with_rtt)}):")
            print(f"  Regret: mean={statistics.mean(knative_regrets):.6f}s ({statistics.mean(knative_regret_pcts):.2f}%), median={statistics.median(knative_regrets):.6f}s ({statistics.median(knative_regret_pcts):.2f}%)")
            if len(knative_regrets) > 1:
                print(f"  Regret std dev: {statistics.stdev(knative_regrets):.6f}s ({statistics.stdev(knative_regret_pcts):.2f}%)")
            print(f"  Rank: mean={statistics.mean(knative_ranks):.1f}, median={statistics.median(knative_ranks):.1f}")
            if len(knative_ranks) > 1:
                print(f"  Rank std dev: {statistics.stdev(knative_ranks):.1f}")
            print(f"  Optimal count: {sum(1 for r in knative_with_rtt if r.matches_bf_optimal_placement)}/{len(knative_with_rtt)} ({sum(1 for r in knative_with_rtt if r.matches_bf_optimal_placement)/len(knative_with_rtt)*100:.1f}%)")


def main():
    # Parse arguments
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    json_output = '--json' in sys.argv
    
    if len(args) < 1:
        datasets_base = Path("/root/projects/my-herosim/simulation_data/artifacts/run2000/gnn_datasets")
    else:
        datasets_base = Path(args[0])
    
    if not datasets_base.exists():
        print(f"ERROR: Directory not found: {datasets_base}", file=sys.stderr)
        sys.exit(1)
    
    # Check if it's a single dataset or a base directory
    if datasets_base.name.startswith('ds_'):
        # Single dataset mode
        if not json_output:
            print(f"Comparing GNN vs Knative for dataset: {datasets_base.name}")
            print("=" * 80)
        comparison = analyze_and_compare_dataset(datasets_base)
        if comparison:
            if json_output:
                print(json.dumps(asdict(comparison), indent=2))
            else:
                print_single_comparison(comparison)
        return
    
    # Find all ds_* directories
    ds_dirs = sorted([d for d in datasets_base.iterdir() if d.is_dir() and d.name.startswith('ds_')])
    
    if not ds_dirs:
        print(f"No ds_* directories found in {datasets_base}", file=sys.stderr)
        sys.exit(1)
    
    if not json_output:
        print(f"Found {len(ds_dirs)} datasets to compare (GNN vs Knative)")
        print("=" * 80)
    
    comparisons = []
    for ds_dir in ds_dirs:
        if not json_output:
            print(f"\n[{ds_dir.name}]")
        comparison = analyze_and_compare_dataset(ds_dir, quiet=json_output)
        if comparison:
            comparisons.append(comparison)
            
            if not json_output:
                print_single_comparison(comparison)
    
    if comparisons:
        # Check for JSON output flag
        if '--json' in sys.argv:
            output_comparison_json(comparisons)
        else:
            print_comparison_summary(comparisons)
    else:
        print("\nNo datasets could be compared.")


def print_single_comparison(comparison: ComparisonResult):
    """Print comparison for a single dataset."""
    print(f"  Dataset: {comparison.dataset_dir}")
    
    if comparison.gnn_result:
        print(f"  GNN:")
        print(f"    Validity: {'✓' if comparison.gnn_result.all_placements_valid else '✗'}")
        print(f"    Placement found: {'YES' if comparison.gnn_result.placement_found_in_bruteforce else 'NO'}")
        print(f"    Matches optimal: {'YES ✓' if comparison.gnn_result.matches_bf_optimal_placement else 'NO'}")
        if comparison.gnn_result.true_rtt is not None:
            print(f"    True RTT: {comparison.gnn_result.true_rtt:.6f}s")
            print(f"    Regret: {comparison.gnn_result.regret:.6f}s ({comparison.gnn_result.regret_percent:.2f}%)")
            print(f"    Rank: {comparison.gnn_result.rank}/{comparison.gnn_result.total_placements}")
    else:
        print(f"  GNN: No result")
    
    if comparison.knative_result:
        print(f"  Knative:")
        print(f"    Validity: {'✓' if comparison.knative_result.all_placements_valid else '✗'}")
        print(f"    Placement found: {'YES' if comparison.knative_result.placement_found_in_bruteforce else 'NO'}")
        print(f"    Matches optimal: {'YES ✓' if comparison.knative_result.matches_bf_optimal_placement else 'NO'}")
        if comparison.knative_result.true_rtt is not None:
            print(f"    True RTT: {comparison.knative_result.true_rtt:.6f}s")
            print(f"    Regret: {comparison.knative_result.regret:.6f}s ({comparison.knative_result.regret_percent:.2f}%)")
            print(f"    Rank: {comparison.knative_result.rank}/{comparison.knative_result.total_placements}")
    else:
        print(f"  Knative: No result")
    
    if comparison.gnn_result and comparison.knative_result:
        print(f"  Comparison:")
        if comparison.gnn_better is not None:
            winner = "GNN" if comparison.gnn_better else "Knative"
            print(f"    Winner: {winner}")
        if comparison.regret_improvement is not None:
            print(f"    Regret improvement: {comparison.regret_improvement:.6f}s ({comparison.regret_improvement_pct:.2f}%)")
        if comparison.rank_improvement is not None:
            print(f"    Rank improvement: {comparison.rank_improvement}")


def output_comparison_json(comparisons: List[ComparisonResult]):
    """Output comparison summary as JSON."""
    both_results = [c for c in comparisons if c.gnn_result and c.knative_result]
    n_both = len(both_results)
    
    regret_improvements = [c.regret_improvement for c in both_results if c.regret_improvement is not None]
    regret_improvements_pct = [c.regret_improvement_pct for c in both_results if c.regret_improvement_pct is not None]
    rank_improvements = [c.rank_improvement for c in both_results if c.rank_improvement is not None]
    
    summary = {
        "total_datasets": len(comparisons),
        "datasets_with_both": n_both,
        "gnn_better_count": sum(1 for c in both_results if c.gnn_better is True),
        "knative_better_count": sum(1 for c in both_results if c.gnn_better is False),
        "regret_improvement": {
            "mean": statistics.mean(regret_improvements) if regret_improvements else 0,
            "mean_pct": statistics.mean(regret_improvements_pct) if regret_improvements_pct else 0,
            "median": statistics.median(regret_improvements) if regret_improvements else 0,
        },
        "rank_improvement": {
            "mean": statistics.mean(rank_improvements) if rank_improvements else 0,
            "median": statistics.median(rank_improvements) if rank_improvements else 0,
        },
        "gnn_optimal_count": sum(1 for c in both_results if c.gnn_found_optimal is True),
        "knative_optimal_count": sum(1 for c in both_results if c.knative_found_optimal is True),
        "comparisons": [asdict(c) for c in comparisons],
    }
    
    print(json.dumps(summary, indent=2))


def output_json_summary(results: List[AnalysisResult]):
    """Output summary as JSON."""
    n = len(results)
    matched = sum(1 for r in results if r.placement_found_in_bruteforce)
    valid = sum(1 for r in results if r.all_placements_valid)
    
    # Filter to results with true_rtt
    results_with_true_rtt = [r for r in results if r.true_rtt is not None]
    n_true = len(results_with_true_rtt)
    
    optimal_count = sum(1 for r in results_with_true_rtt if r.rank == 1)
    bf_optimal_match_count = sum(1 for r in results_with_true_rtt if r.matches_bf_optimal_placement)
    
    regrets = [r.regret for r in results_with_true_rtt]
    regret_pcts = [r.regret_percent for r in results_with_true_rtt]
    ranks = [r.rank for r in results_with_true_rtt]
    
    state_match = sum(1 for r in results if r.system_state_matches)
    
    summary = {
        "total_datasets": n,
        "datasets_with_true_rtt": n_true,
        "placement_matching": {
            "matched": matched,
            "match_rate": matched / n if n > 0 else 0,
        },
        "bf_optimal_match": {
            "matched": bf_optimal_match_count,
            "match_rate": bf_optimal_match_count / n_true if n_true > 0 else 0,
        },
        "validity": {
            "valid": valid,
            "valid_rate": valid / n if n > 0 else 0,
        },
        "system_state_match": {
            "matched": state_match,
            "match_rate": state_match / n if n > 0 else 0,
        },
        "queue_replica_consistency": {
            "consistent": sum(1 for r in results if r.queue_replica_consistent),
            "consistent_rate": sum(1 for r in results if r.queue_replica_consistent) / n if n > 0 else 0,
        },
        "regret": {
            "mean": statistics.mean(regrets) if regrets else 0,
            "std": statistics.stdev(regrets) if len(regrets) > 1 else 0,
            "min": min(regrets) if regrets else 0,
            "max": max(regrets) if regrets else 0,
        },
        "regret_percent": {
            "mean": statistics.mean(regret_pcts) if regret_pcts else 0,
            "std": statistics.stdev(regret_pcts) if len(regret_pcts) > 1 else 0,
            "min": min(regret_pcts) if regret_pcts else 0,
            "max": max(regret_pcts) if regret_pcts else 0,
        },
        "rank": {
            "optimal_count": optimal_count,
            "optimal_rate": optimal_count / n_true if n_true > 0 else 0,
            "mean": statistics.mean(ranks) if ranks else 0,
            "median": statistics.median(ranks) if ranks else 0,
        },
        "datasets": [asdict(r) for r in results],
    }
    
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

