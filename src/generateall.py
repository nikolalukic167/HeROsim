import json
import pickle

import numpy as np
from itertools import product

def generate_range(min_val, max_val, step):
    """Generate a range of values from min to max with step."""
    return np.arange(min_val, max_val + step, step)

def generate_device_proportions(devices_config):
    """Generate valid device proportion combinations."""
    device_ranges = {}

    # Generate ranges for each device
    for device, config in devices_config.items():
        device_ranges[device] = generate_range(
            config['min'],
            config['max'],
            config['step']
        )

    # Generate all possible combinations
    devices = list(device_ranges.keys())
    combinations = product(*[device_ranges[d] for d in devices])

    # Filter valid combinations (sum to 1)
    valid_combinations = []
    for combo in combinations:
        if np.isclose(sum(combo), 1.0, rtol=1e-10):
            valid_combinations.append(dict(zip(devices, combo)))

    return valid_combinations

def generate_workload_combinations(wsc_config):
    """Generate all workload combinations."""
    workload_ranges = {}

    # Generate ranges for each application
    for app, config in wsc_config.items():
        workload_ranges[app] = generate_range(
            config['min'],
            config['max'],
            config['step']
        )

    # Generate all possible combinations
    apps = sorted(workload_ranges.keys())  # Sort for consistent order
    combinations = product(*[workload_ranges[app] for app in apps])

    return [dict(zip(apps, combo)) for combo in combinations]

def generate_infrastructure_combinations(config):
    """Generate all valid infrastructure and workload combinations."""

    # Generate network bandwidth options
    nwc_values = generate_range(
        config['nwc']['min'],
        config['nwc']['max'],
        config['nwc']['step']
    )

    # Generate cluster size options
    csc_values = generate_range(
        config['csc']['min'],
        config['csc']['max'],
        config['csc']['step']
    )

    # Generate valid device proportions
    device_combinations = generate_device_proportions(config['pci'])

    # Generate workload combinations
    workload_combinations = generate_workload_combinations(config['wsc'])

    # Generate all possible combinations
    all_combinations = []
    for nwc, csc, dev_props, workload in product(
            nwc_values,
            csc_values,
            device_combinations,
            workload_combinations
    ):
        combination = {
            'network_bandwidth': nwc,
            'cluster_size': csc,
            'device_proportions': dev_props,
            'workloads': workload
        }
        all_combinations.append(combination)

    return all_combinations


def generate_structure_mapping(config):
    """Generate the mapping of array indices to their meaning."""
    structure = ["network_bandwidth", "cluster_size"]
    structure.extend(f"device_prop_{dev}" for dev in sorted(config['pci'].keys()))
    structure.extend(f"workload_{app}" for app in sorted(config['wsc'].keys()))
    return {i: name for i, name in enumerate(structure)}

def save_combinations(arrays, mapping, output_prefix="combinations"):
    """Save combinations and mapping."""
    # Save the combinations array
    np.save(f"{output_prefix}.npy", arrays)

    # Save the mapping
    with open(f"{output_prefix}_mapping.pkl", 'wb') as f:
        pickle.dump(mapping, f)


def combination_to_array(combination, config):
    """Convert a combination to a 1-D numpy array."""
    values = []

    # Add network bandwidth
    values.append(combination['network_bandwidth'])

    # Add cluster size
    values.append(combination['cluster_size'])

    # Add device proportions in a fixed order
    for device in sorted(config['pci'].keys()):
        values.append(combination['device_proportions'][device])

    # Add workload values in a fixed order
    for app in sorted(config['wsc'].keys()):
        values.append(combination['workloads'][app])

    return np.array(values)

if __name__ == "__main__":
    config_file = "simulation_data/space_simple.json"
    output_prefix = "simulation_data/combinations_simple"
    try:
        # Load configuration
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Generate combinations
        combinations = generate_infrastructure_combinations(config)

        # Convert to arrays
        arrays = np.array([combination_to_array(combo, config) for combo in combinations])

        # Generate and save mapping
        mapping = generate_structure_mapping(config)

        # Save everything
        save_combinations(arrays, mapping, output_prefix)

        print(f"Generated {len(arrays)} valid combinations")
        print("\nArray structure:")
        for idx, name in mapping.items():
            print(f"Index {idx}: {name}")

        # Show first few combinations
        print("\nFirst few combinations:")
        for i, arr in enumerate(arrays[:3]):
            print(f"\nCombination {i+1}:")
            print(arr)

    except FileNotFoundError:
        print(f"Error: Config file {config_file} not found")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in config file")
