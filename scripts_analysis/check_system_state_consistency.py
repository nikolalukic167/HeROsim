#!/usr/bin/env python3
"""
Script to check if all non-null systemStateResult entries have the same
available_resources and replicas as a reference entry across multiple datasets.

Usage:
    python check_system_state_consistency.py <parent_dir>
    
Example:
    python check_system_state_consistency.py simulation_data/artifacts/run1100/gnn_datasets
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def normalize_list_for_comparison(lst: List) -> List:
    """Sort lists for comparison (order doesn't matter for available_resources)."""
    if isinstance(lst, list):
        return sorted(lst)
    return lst


def normalize_dict_for_comparison(d: Dict) -> Dict:
    """Normalize dictionary for comparison."""
    if not isinstance(d, dict):
        return d
    
    normalized = {}
    for key, value in sorted(d.items()):
        if isinstance(value, list):
            normalized[key] = normalize_list_for_comparison(value)
        elif isinstance(value, dict):
            normalized[key] = normalize_dict_for_comparison(value)
        else:
            normalized[key] = value
    return normalized


def compare_available_resources(ref: Dict, other: Dict) -> bool:
    """Compare available_resources dictionaries."""
    ref_norm = normalize_dict_for_comparison(ref)
    other_norm = normalize_dict_for_comparison(other)
    return ref_norm == other_norm


def compare_replicas(ref: Dict, other: Dict) -> bool:
    """Compare replicas dictionaries."""
    # Replicas structure: {"dnn1": [[node, platform_id], ...], "dnn2": [...]}
    # Need to compare sets of tuples for each task type
    if set(ref.keys()) != set(other.keys()):
        return False
    
    for task_type in ref.keys():
        ref_replicas = ref[task_type]
        other_replicas = other[task_type]
        
        # Convert to sets of tuples for comparison
        ref_set = {tuple(replica) for replica in ref_replicas}
        other_set = {tuple(replica) for replica in other_replicas}
        
        if ref_set != other_set:
            return False
    
    return True


def load_reference_from_file(file_path: Path) -> Optional[Dict]:
    """Load the reference systemStateResult by parsing the full JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Find the systemStateResult at the specified location
        # It should be in task results
        task_results = data.get('stats', {}).get('taskResults', [])
        
        for task_result in task_results:
            system_state = task_result.get('systemStateResult')
            if system_state is not None:
                # Check if this matches the reference (has the expected structure)
                if 'available_resources' in system_state and 'replicas' in system_state:
                    return system_state
        
        # If not found in task results, try to find it directly
        # Read the file and extract from the specific line range
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse the full JSON
        data = json.loads(content)
        
        # Search recursively for systemStateResult
        def find_system_state(obj, path=""):
            if isinstance(obj, dict):
                if 'systemStateResult' in obj and obj['systemStateResult'] is not None:
                    system_state = obj['systemStateResult']
                    if isinstance(system_state, dict) and 'available_resources' in system_state:
                        return system_state
                for key, value in obj.items():
                    result = find_system_state(value, f"{path}.{key}")
                    if result:
                        return result
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    result = find_system_state(item, f"{path}[{i}]")
                    if result:
                        return result
            return None
        
        system_state = find_system_state(data)
        if system_state:
            return system_state
        
        return None
        
    except Exception as e:
        logger.error(f"Error loading reference file {file_path}: {e}")
        return None


def find_all_system_states(file_path: Path) -> List[Dict[str, Any]]:
    """Find all non-null systemStateResult entries in a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        system_states = []
        
        # Search in task results
        task_results = data.get('stats', {}).get('taskResults', [])
        for task_result in task_results:
            system_state = task_result.get('systemStateResult')
            if system_state is not None and isinstance(system_state, dict):
                system_states.append({
                    'task_id': task_result.get('taskId'),
                    'file': file_path.name,
                    'system_state': system_state
                })
        
        return system_states
        
    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {e}")
        return []


def process_dataset(dataset_dir: Path) -> Dict[str, Any]:
    """Process a single dataset directory and return statistics."""
    results_dir = dataset_dir / 'results'
    
    if not results_dir.exists():
        return {
            'dataset': dataset_dir.name,
            'error': f"Results directory not found: {results_dir}",
            'total_entries': 0,
            'matching_entries': 0,
            'mismatched_entries': 0,
            'files_with_mismatches': set()
        }
    
    # Find all JSON files in results directory
    json_files = sorted(results_dir.glob('*.json'))
    
    if not json_files:
        return {
            'dataset': dataset_dir.name,
            'error': f"No JSON files found in {results_dir}",
            'total_entries': 0,
            'matching_entries': 0,
            'mismatched_entries': 0,
            'files_with_mismatches': set()
        }
    
    # Use the first file as the reference
    reference_file = json_files[0]
    
    # Load reference systemStateResult
    reference_state = load_reference_from_file(reference_file)
    if reference_state is None:
        return {
            'dataset': dataset_dir.name,
            'error': "Failed to load reference systemStateResult from first file",
            'total_entries': 0,
            'matching_entries': 0,
            'mismatched_entries': 0,
            'files_with_mismatches': set()
        }
    
    ref_available_resources = reference_state.get('available_resources', {})
    ref_replicas = reference_state.get('replicas', {})
    
    # Statistics
    total_entries = 0
    matching_entries = 0
    mismatched_entries = 0
    files_with_mismatches = set()
    
    # Check each file (including the reference file itself)
    for json_file in json_files:
        system_states = find_all_system_states(json_file)
        
        for entry in system_states:
            total_entries += 1
            system_state = entry['system_state']
            file_name = entry['file']
            
            available_resources = system_state.get('available_resources', {})
            replicas = system_state.get('replicas', {})
            
            # Compare available_resources
            resources_match = compare_available_resources(ref_available_resources, available_resources)
            
            # Compare replicas
            replicas_match = compare_replicas(ref_replicas, replicas)
            
            if resources_match and replicas_match:
                matching_entries += 1
            else:
                mismatched_entries += 1
                files_with_mismatches.add(f"{dataset_dir.name}/{file_name}")
    
    return {
        'dataset': dataset_dir.name,
        'error': None,
        'total_entries': total_entries,
        'matching_entries': matching_entries,
        'mismatched_entries': mismatched_entries,
        'files_with_mismatches': files_with_mismatches
    }


def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python check_system_state_consistency.py <parent_dir>")
        logger.error("Example: python check_system_state_consistency.py simulation_data/artifacts/run1100/gnn_datasets")
        sys.exit(1)
    
    parent_dir = Path(sys.argv[1])
    
    if not parent_dir.exists():
        logger.error(f"Parent directory not found: {parent_dir}")
        sys.exit(1)
    
    # Find all ds_* directories
    dataset_dirs = sorted([d for d in parent_dir.iterdir() if d.is_dir() and d.name.startswith('ds_')])
    
    if not dataset_dirs:
        logger.error(f"No ds_* directories found in {parent_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(dataset_dirs)} dataset directories to check")
    
    # Aggregate statistics
    all_total_entries = 0
    all_matching_entries = 0
    all_mismatched_entries = 0
    all_files_with_mismatches = set()
    datasets_with_errors = []
    datasets_with_mismatches = []
    
    # Process each dataset with progress bar
    for dataset_dir in tqdm(dataset_dirs, desc="Processing datasets"):
        result = process_dataset(dataset_dir)
        
        if result['error']:
            datasets_with_errors.append((result['dataset'], result['error']))
        else:
            all_total_entries += result['total_entries']
            all_matching_entries += result['matching_entries']
            all_mismatched_entries += result['mismatched_entries']
            all_files_with_mismatches.update(result['files_with_mismatches'])
            
            if result['mismatched_entries'] > 0:
                datasets_with_mismatches.append(result['dataset'])
    
    # Summary
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total datasets processed: {len(dataset_dirs)}")
    logger.info(f"Total systemStateResult entries checked: {all_total_entries}")
    logger.info(f"  Matching entries: {all_matching_entries}")
    logger.info(f"  Mismatched entries: {all_mismatched_entries}")
    logger.info(f"Datasets with mismatches: {len(datasets_with_mismatches)}")
    logger.info(f"Files with mismatches: {len(all_files_with_mismatches)}")
    
    if datasets_with_errors:
        logger.warning(f"Datasets with errors: {len(datasets_with_errors)}")
        for dataset, error in datasets_with_errors:
            logger.warning(f"  - {dataset}: {error}")
    
    if datasets_with_mismatches:
        logger.warning("Datasets with mismatches:")
        for dataset in sorted(datasets_with_mismatches):
            logger.warning(f"  - {dataset}")
    
    if all_files_with_mismatches:
        logger.warning("Sample files with mismatches (first 20):")
        for file_name in sorted(all_files_with_mismatches)[:20]:
            logger.warning(f"  - {file_name}")
        if len(all_files_with_mismatches) > 20:
            logger.warning(f"  ... and {len(all_files_with_mismatches) - 20} more")
    
    if all_mismatched_entries == 0 and not datasets_with_errors:
        logger.info("✓ All systemStateResult entries match the reference across all datasets!")
        return 0
    else:
        logger.warning(f"✗ Found {all_mismatched_entries} mismatched entries across {len(datasets_with_mismatches)} datasets")
        return 1


if __name__ == '__main__':
    sys.exit(main())

