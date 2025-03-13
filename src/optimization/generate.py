import json
import os
import sys
import multiprocessing as mp
from pathlib import Path

import numpy as np

from src.executeinitial import setup_logging

from collections import Counter

from src.generateall import generate_infrastructure_combinations, combination_to_array, generate_structure_mapping, \
    save_combinations


def count_requests_per_second_with_stats(events):
    # Extract timestamps and convert them to seconds
    timestamps = [event['timestamp'] for event in events]
    seconds = [int(ts) for ts in timestamps]

    # Count occurrences per second
    counts = Counter(seconds)

    # Calculate min, max, and average values
    min_requests = min(counts.values()) if counts else 0
    max_requests = max(counts.values()) if counts else 0
    average_requests = sum(counts.values()) / len(counts) if counts else 0
    return counts, min_requests, max_requests, average_requests


def main():
    output_dir = sys.argv[1]
    workload_config_file = sys.argv[2]
    days = sys.argv[3]  # determines how many days are used to create space for function
    app = sys.argv[4]

    with open(workload_config_file, 'r') as fd:
        workload_configs = json.load(fd)

    logger = setup_logging(Path("data/nofs-ids"))
    base_dir = Path("data/nofs-ids")
    sim_input_path = Path("data/nofs-ids")

    base_space_file = "simulation_data/space_simple.json"

    os.makedirs(output_dir, exist_ok=True)
    workload = workload_configs[0]  # first week
    with open(workload, 'r') as fd:
        workload = json.load(fd)
        duration_per_day = workload['duration'] / 7
        seconds = 0
        requests_in_a_day = []
        for event in workload['events']:
            if event['timestamp'] < duration_per_day:
                requests_in_a_day.append(event)
            else:
                break
        counts, min_requests, max_requests, average_requests = count_requests_per_second_with_stats(requests_in_a_day)

    print(max_requests)
    with open(base_space_file, 'r') as fd:
        base_space = json.load(fd)
        base_space['wsc'][app] = {
            'average': average_requests,
            'min': (min_requests / average_requests) * 0.5,
            'max': (max_requests / average_requests) * 1.5,
            'step': 0.05
        }
        # Generate combinations
        combinations = generate_infrastructure_combinations(base_space)

        # Convert to arrays
        arrays = np.array([combination_to_array(combo, base_space) for combo in combinations])

        # Generate and save mapping
        mapping = generate_structure_mapping(base_space)

        # Save everything
        save_combinations(arrays, mapping, f'{output_dir}/combinations')

    with open(f'{output_dir}/space.json', 'w') as fd:
        json.dump(base_space, fd, indent=2)


if __name__ == '__main__':
    main()
