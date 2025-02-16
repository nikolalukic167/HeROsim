import json
import numpy as np
from itertools import product
import itertools

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

def generate_infrastructure_combinations(config_file):
    """Generate all valid infrastructure combinations."""
    # Read configuration
    with open(config_file, 'r') as f:
        config = json.load(f)

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

    # Generate all possible combinations
    all_combinations = []
    for nwc, csc, dev_props in product(nwc_values, csc_values, device_combinations):
        combination = {
            'network_bandwidth': nwc,
            'cluster_size': csc,
            'device_proportions': dev_props
        }
        all_combinations.append(combination)

    return all_combinations

def combination_to_array(combination, config):
    """Convert a combination to a 1-D numpy array."""
    # Extract values in a fixed order
    values = []

    # Add network bandwidth
    values.append(combination['network_bandwidth'])

    # Add cluster size
    values.append(combination['cluster_size'])

    # Add device proportions in a fixed order
    for device in sorted(config['pci'].keys()):
        values.append(combination['device_proportions'][device])

    return np.array(values)

# Example usage
if __name__ == "__main__":
    config_file = "data/ids/space.json"

    try:
        # Load configuration
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Generate combinations
        combinations = generate_infrastructure_combinations(config_file)

        # Convert to arrays
        arrays = [combination_to_array(combo, config) for combo in combinations]

        print(f"Generated {len(arrays)} valid combinations")

        # Example output
        for i, arr in enumerate(arrays[:3]):  # Show first 3 combinations
            print(f"\nCombination {i+1}:")
            print(arr)

    except FileNotFoundError:
        print(f"Error: Config file {config_file} not found")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in config file")
