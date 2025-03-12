import json
import logging
import math
import os
import pickle
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import xgboost

# Assuming increase_events is imported from your previous script
from src.eventgenerator import increase_events_of_app
from src.motivational.constants import KEEP_ALIVE, QUEUE_LENGTH
from src.placement.executor import execute_sim
from src.placement.model import SimulationData, DataclassJSONEncoder
from src.train import train_model, save_models

REQUIRED_SIM_FILES = [
    'application-types.json',
    'platform-types.json',
    'qos-types.json',
    'storage-types.json',
    'task-types.json'
]


def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger('simulation')
    logger.setLevel(logging.INFO)

    # File handler
    # fh = logging.FileHandler(output_dir / 'simulation.log')
    # fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def load_simulation_inputs(sim_input_path: Path) -> Dict[str, Any]:
    """Load all required simulation input files."""
    sim_inputs = {}

    # Verify all required files exist
    missing_files = []
    for filename in REQUIRED_SIM_FILES:
        if not (sim_input_path / filename).exists():
            missing_files.append(filename)

    if missing_files:
        raise FileNotFoundError(
            f"Missing required simulation input files: {', '.join(missing_files)}"
        )

    # Load all files
    for filename in REQUIRED_SIM_FILES:
        file_path = sim_input_path / filename
        with open(file_path, 'r') as f:
            # Use filename without extension as key
            key = filename.replace('.json', '').replace('-', '_')
            sim_inputs[key] = json.load(f)

    return sim_inputs


def prepare_workloads(
        sample: np.ndarray,
        mapping: Dict[int, str],
        base_workload: Dict,
        apps: List[str]
) -> Dict[str, Dict]:
    """Prepare workloads based on sample values."""
    reverse_mapping = {name: idx for idx, name in mapping.items()}
    prepared_workloads = {}

    # Process each application
    for app_name in apps:
        # Get the workload factor from sample
        workload_key = f'workload_{app_name}'
        if workload_key in reverse_mapping:
            # logging.warning('read factor from sample, currently set to 1 for debugging purposes')
            factor = sample[int(reverse_mapping[workload_key])]
            # factor = 1
            # Create a deep copy of base workload
            workload_copy = deepcopy(base_workload)
            # Apply increase_events with the factor
            prepared_workloads[app_name] = increase_events_of_app(workload_copy['events'], factor, app_name)

    return prepared_workloads


def load_samples(prefix="lhs_samples"):
    """Load LHS samples and their mapping."""
    samples = np.load(f"{prefix}.npy")
    with open(f"{prefix}_mapping.pkl", 'rb') as f:
        mapping = pickle.load(f)
    return samples, mapping


def load_config(config_file: str) -> dict:
    """Load original configuration file with specs."""
    with open(config_file, 'r') as f:
        return json.load(f)


def create_reverse_mapping(mapping: Dict[int, str]) -> Dict[str, int]:
    """Create reverse mapping from names to indices."""
    return {name: int(idx) for idx, name in mapping.items()}


def calculate_device_counts(cluster_size: int, proportions: Dict[str, float]) -> Dict[str, int]:
    """Calculate number of devices for each type."""
    device_counts = {}
    remaining_size = cluster_size

    # First round up all device counts
    for device, proportion in proportions.items():
        count = math.ceil(cluster_size * proportion)
        device_counts[device] = count
        remaining_size -= count

    # If we allocated too many devices, reduce counts one by one
    # from the device with the smallest proportion
    if remaining_size < 0:
        sorted_devices = sorted(proportions.items(), key=lambda x: x[1])
        idx = 0
        while remaining_size < 0:
            device = sorted_devices[idx][0]
            if device_counts[device] > 1:  # Ensure at least one device remains
                device_counts[device] -= 1
                remaining_size += 1
            idx = (idx + 1) % len(sorted_devices)

    return device_counts


def prepare_simulation_config(
        sample: np.ndarray,
        mapping: Dict[int, str],
        original_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Prepare simulation configuration from a sample."""
    reverse_mapping = create_reverse_mapping(mapping)

    # Extract network bandwidth
    network_bandwidth = sample[reverse_mapping['network_bandwidth']]

    # Extract cluster size
    cluster_size = int(sample[reverse_mapping['cluster_size']])

    # Extract device proportions
    device_proportions = {}
    for device in original_config['pci'].keys():
        prop_key = f'device_prop_{device}'
        device_proportions[device] = sample[reverse_mapping[prop_key]]

    # Calculate device counts
    device_counts = calculate_device_counts(cluster_size, device_proportions)

    # Prepare simulation configuration
    infrastructure_config = {
        "network": {
            "bandwidth": float(network_bandwidth)
        },
        "nodes": []
    }

    # Generate node list
    for device_type, count in device_counts.items():
        device_specs = original_config['pci'][device_type]['specs']
        for _ in range(count):
            node_config = device_specs.copy()
            node_config['type'] = device_type  # Add device type to specs
            infrastructure_config['nodes'].append(node_config)

    return infrastructure_config


def execute_simulation(
        config: Dict[str, Any],
        sim_inputs: Dict[str, Any],
        scheduling_strategy: str,
        model_locations: Dict[str, str] = None,
        models: Dict[str, xgboost.XGBRegressor] = None,
        cache_policy='fifo',
        task_priority='fifo',
        keep_alive=30,
        queue_length=100,
        reconcile_interval=1
) -> Dict[str, Any]:
    """Execute simulation with full configuration and simulation inputs."""

    simulation_data = SimulationData(
        platform_types=sim_inputs['platform_types'],
        storage_types=sim_inputs['storage_types'],
        qos_types=sim_inputs['qos_types'],
        application_types=sim_inputs['application_types'],
        task_types=sim_inputs['task_types'],
    )

    stats = execute_sim(simulation_data, config['infrastructure'], cache_policy, keep_alive, task_priority,
                        queue_length,
                        scheduling_strategy, config['workload'], 'workload-mine',
                        model_locations=model_locations, models=models, reconcile_interval=reconcile_interval)
    return {
        "status": "success",
        "config": config,
        "sim_inputs": sim_inputs,
        "stats": stats
    }


def calculate_workload_stats(events: List[Dict]) -> Dict[str, float]:
    """Calculate statistics for the flattened workload."""
    if not events:
        return {
            "average_rps": 0,
            "duration": 0,
            "total_events": 0
        }

    # Get timestamps as integers
    timestamps = [int(event['timestamp']) for event in events]
    min_timestamp = min(timestamps)
    max_timestamp = max(timestamps)

    # Calculate duration in seconds
    duration = max_timestamp - min_timestamp + 1  # +1 to include both start and end second

    # Calculate average RPS
    total_events = len(events)
    average_rps = total_events / duration if duration > 0 else 0

    return {
        "rps": average_rps,
        "duration": duration,
        "total_events": total_events,
        "start_timestamp": min_timestamp,
        "end_timestamp": max_timestamp
    }


def flatten_workloads(workloads: Dict[str, Dict]) -> Dict[str, Any]:
    """Flatten multiple workload events into a single sorted list with statistics."""
    # Collect all events
    all_events = []
    for app_name, workload in workloads.items():
        events = workload
        all_events.extend(events)

    # Sort events by timestamp
    sorted_events = sorted(all_events, key=lambda x: x['timestamp'])

    # Calculate statistics
    stats = calculate_workload_stats(sorted_events)

    return {
        "rps": stats['rps'],
        "duration": stats['duration'],
        "events": sorted_events
    }


def main():
    # Configuration paths
    base_dir = Path("simulation_data")
    sim_input_path = Path("data/nofs-ids")  # Base path for simulation input files
    samples_file = base_dir / "lhs_samples_simple.npy"
    mapping_file = base_dir / "lhs_samples_simple_mapping.pkl"
    config_file = base_dir / "space_simple.json"
    workload_base_file = "data/nofs-ids/traces/workload-125-250.json"
    output_dir = base_dir / "initial_results_simple"
    os.makedirs(output_dir, exist_ok=True)
    max_workers = int(sys.argv[1])
    # Setup logging
    logger = setup_logging(output_dir)
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting simulation preparation")

        # Load simulation inputs
        logger.info("Loading simulation input files")
        sim_inputs = load_simulation_inputs(sim_input_path)

        # Load samples and mapping
        logger.info("Loading samples and mapping")
        samples = np.load(samples_file)
        with open(mapping_file, 'rb') as f:
            mapping = pickle.load(f)

        # Load infrastructure config
        logger.info("Loading infrastructure configuration")
        with open(config_file, 'r') as f:
            infra_config = json.load(f)

        # Load workload base
        logger.info("Loading workload base")
        with open(workload_base_file, 'r') as f:
            workload_base = json.load(f)

        # Get list of applications from config
        apps = [app for app in infra_config['wsc'].keys()]

        # Process each sample
        # reactive_results_paths = execute_reactive_samples(apps, infra_config, logger, mapping, output_dir, samples,
        #                                                   sim_inputs, workload_base)
        reactive_results_paths = execute_reactive_samples_parallel(apps, config_file, mapping_file, output_dir, samples,
                                                                   sim_input_path, workload_base_file, max_workers)
        # reactive_results_paths = ['simulation_data/initial_results_simple/simulation_{x}.json'  for x in [1,2,3,4,5]]
        # reactive_results_paths = ['simulation_data/initial_results_simple/simulation_{x}.json'  for x in [1]]
        print(reactive_results_paths)
        logger.info("Completed all simulations")

        logger.info("Training model now...")
        models, eval_results = train_model(output_dir, samples, include_queue_length=False)
        model_paths = save_models(models, output_dir)

        logger.info('Finished model training')
        logger.info(f'All files can be found under {output_dir}')


    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise e


import concurrent.futures
import json
from pathlib import Path


def process_sample(args):
    i, sample, output_dir, sim_input_path, mapping_file, config_file, workload_base_file, apps = args
    logger = setup_logging(output_dir)
    logger.info(f"Processing sample {i + 1}")

    try:
        sim_inputs = load_simulation_inputs(sim_input_path)

        with open(mapping_file, 'rb') as f:
            mapping = pickle.load(f)

        # Load infrastructure config
        with open(config_file, 'r') as f:
            infra_config = json.load(f)

        with open(workload_base_file, 'r') as f:
            workload_base = json.load(f)

        # Prepare infrastructure configuration
        sim_config = prepare_simulation_config(sample, mapping, infra_config)

        # Prepare workloads
        workloads = prepare_workloads(sample, mapping, workload_base, apps)
        # Flatten workloads into single sorted list
        flattened_workloads = flatten_workloads(workloads)

        # Combine infrastructure and workload configurations
        full_config = {
            "infrastructure": sim_config,
            "workload": flattened_workloads
        }

        # Execute simulation with additional inputs
        cache_policy = 'fifo'
        task_priority = 'fifo'
        keep_alive = KEEP_ALIVE
        queue_length = QUEUE_LENGTH
        scheduling_strategy = 'kn_kn'
        result = execute_simulation(full_config, sim_inputs, scheduling_strategy)
        result['sample'] = {
            'apps': apps,
            'sample': sample.tolist(),
            'mapping': mapping,
            'infra_config': infra_config,
            'workload_base': workload_base,
            'sim_inputs': sim_inputs,
            'scheduling_strategy': scheduling_strategy,
            'cache_policy': cache_policy,
            'task_priority': task_priority,
            'keep_alive': keep_alive,
            'queue_length': queue_length
        }

        # Save result
        result_file = output_dir / f"simulation_{i + 1}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, cls=DataclassJSONEncoder)

        logger.info(f"Completed simulation {i + 1}")
        return result_file

    except Exception as e:
        logger.error(f"Error in simulation {i + 1}: {str(e)}")
        logger.exception(e)
        return None


def process_sample_proactive(args):
    i, sample, output_dir, sim_input_path, mapping_file, config_file, workload_base_file, apps, model_locations = args
    logger = setup_logging(output_dir)
    logger.info(f"Processing sample {i + 1}")

    try:
        sim_inputs = load_simulation_inputs(sim_input_path)

        with open(mapping_file, 'rb') as f:
            mapping = pickle.load(f)

        # Load infrastructure config
        with open(config_file, 'r') as f:
            infra_config = json.load(f)

        with open(workload_base_file, 'r') as f:
            workload_base = json.load(f)

        # Prepare infrastructure configuration
        sim_config = prepare_simulation_config(sample, mapping, infra_config)

        # Prepare workloads
        workloads = prepare_workloads(sample, mapping, workload_base, apps)
        # Flatten workloads into single sorted list
        flattened_workloads = flatten_workloads(workloads)

        # Combine infrastructure and workload configurations
        full_config = {
            "infrastructure": sim_config,
            "workload": flattened_workloads
        }

        # Execute simulation with additional inputs
        cache_policy = 'fifo'
        task_priority = 'fifo'
        keep_alive = KEEP_ALIVE
        queue_length = QUEUE_LENGTH
        scheduling_strategy = 'prokn_prokn'
        result = execute_simulation(full_config, sim_inputs, scheduling_strategy, model_locations=model_locations)
        result['sample'] = {
            'apps': apps,
            'sample': sample.tolist(),
            'mapping': mapping,
            'infra_config': infra_config,
            'workload_base': workload_base,
            'sim_inputs': sim_inputs,
            'scheduling_strategy': scheduling_strategy,
            'cache_policy': cache_policy,
            'task_priority': task_priority,
            'keep_alive': keep_alive,
            'queue_length': queue_length
        }

        # Save result
        result_file = output_dir / f"simulation_{i + 1}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, cls=DataclassJSONEncoder)

        logger.info(f"Completed simulation {i + 1}")
        return result_file

    except Exception as e:
        logger.error(f"Error in simulation {i + 1}: {str(e)}")
        logger.exception(e)
        return None


def execute_reactive_samples_parallel(apps, config_file, mapping_file, output_dir, samples, sim_input_path,
                                      workload_base_file, max_workers):
    result_paths = []

    # Use ProcessPoolExecutor for CPU-bound tasks, ThreadPoolExecutor for I/O-bound tasks
    # Adjust max_workers as needed based on your system's capabilities
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of (index, sample) tuples to process
        sample_tuples = [(i, sample, output_dir, sim_input_path, mapping_file, config_file, workload_base_file, apps)
                         for i, sample in enumerate(samples)]
        start_ts = time.time()
        # Submit all tasks and process results as they complete
        future_to_sample = {executor.submit(process_sample, sample_tuple): sample_tuple for sample_tuple in
                            sample_tuples}

        for future in concurrent.futures.as_completed(future_to_sample):
            result_file = future.result()
            if result_file is not None:
                result_paths.append(result_file)
        end_ts = time.time()
        print(f'Duration: {end_ts - start_ts}')
    return result_paths


def execute_proactive_samples_parallel(apps, config_file, mapping_file, output_dir, samples, sim_input_path,
                                       workload_base_file, max_workers, model_paths):
    result_paths = []

    # Use ProcessPoolExecutor for CPU-bound tasks, ThreadPoolExecutor for I/O-bound tasks
    # Adjust max_workers as needed based on your system's capabilities
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of (index, sample) tuples to process
        sample_tuples = [
            (i, sample, output_dir, sim_input_path, mapping_file, config_file, workload_base_file, apps, model_paths)
            for i, sample in enumerate(samples)]
        start_ts = time.time()
        # Submit all tasks and process results as they complete
        future_to_sample = {executor.submit(process_sample_proactive, sample_tuple): sample_tuple for sample_tuple in
                            sample_tuples}

        for future in concurrent.futures.as_completed(future_to_sample):
            result_file = future.result()
            if result_file is not None:
                result_paths.append(result_file)
        end_ts = time.time()
        print(f'Duration: {end_ts - start_ts}')
    return result_paths


def execute_reactive_samples(apps, infra_config, logger, mapping, output_dir, samples, sim_inputs, workload_base):
    result_paths = []
    for i, sample in enumerate(samples):
        logger.info(f"Processing sample {i + 1}/{len(samples)}")

        # Prepare infrastructure configuration
        sim_config = prepare_simulation_config(sample, mapping, infra_config)

        # Prepare workloads
        workloads = prepare_workloads(sample, mapping, workload_base, apps)
        # Flatten workloads into single sorted list
        flattened_workloads = flatten_workloads(workloads)

        # Combine infrastructure and workload configurations
        full_config = {
            "infrastructure": sim_config,
            "workload": flattened_workloads
        }

        # Execute simulation with additional inputs
        try:
            cache_policy = 'fifo'
            task_priority = 'fifo'
            keep_alive = KEEP_ALIVE
            queue_length = QUEUE_LENGTH
            scheduling_strategy = 'kn_kn'
            result = execute_simulation(full_config, sim_inputs, scheduling_strategy)
            result['sample'] = {
                'apps': apps,
                'sample': sample.tolist(),
                'mapping': mapping,
                'infra_config': infra_config,
                'workload_base': workload_base,
                'sim_inputs': sim_inputs,
                'scheduling_strategy': scheduling_strategy,
                'cache_policy': cache_policy,
                'task_priority': task_priority,
                'keep_alive': keep_alive,
                'queue_length': queue_length
            }

            # Save result
            result_file = output_dir / f"simulation_{i + 1}.json"
            result_paths.append(result_file)
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, cls=DataclassJSONEncoder)

            logger.info(f"Completed simulation {i + 1}")

        except Exception as e:
            logger.error(f"Error in simulation {i + 1}: {str(e)}")
            logger.exception(e)
    return result_paths


def execute_proactive_samples(apps, infra_config, logger, mapping, output_dir, samples, sim_inputs, workload_base,
                              models_locations):
    result_paths = []
    for i, sample in enumerate(samples):
        logger.info(f"Processing sample {i + 1}/{len(samples)}")

        # Prepare infrastructure configuration
        sim_config = prepare_simulation_config(sample, mapping, infra_config)

        # Prepare workloads
        workloads = prepare_workloads(sample, mapping, workload_base, apps)
        # Flatten workloads into single sorted list
        flattened_workloads = flatten_workloads(workloads)

        # Combine infrastructure and workload configurations
        full_config = {
            "infrastructure": sim_config,
            "workload": flattened_workloads
        }

        # Execute simulation with additional inputs
        try:
            cache_policy = 'fifo'
            task_priority = 'fifo'
            keep_alive = KEEP_ALIVE
            queue_length = QUEUE_LENGTH
            scheduling_strategy = 'prokn_prokn'
            result = execute_simulation(full_config, sim_inputs, scheduling_strategy, model_locations=models_locations,
                                        cache_policy=cache_policy,
                                        task_priority=task_priority,
                                        keep_alive=keep_alive,
                                        queue_length=queue_length)
            result['sample'] = {
                'apps': apps,
                'sample': sample.tolist(),
                'mapping': mapping,
                'infra_config': infra_config,
                'workload_base': workload_base,
                'models_locations': models_locations,
                'sim_inputs': sim_inputs,
                'scheduling_strategy': scheduling_strategy,
                'cache_policy': cache_policy,
                'task_priority': task_priority,
                'keep_alive': keep_alive,
                'queue_length': queue_length
            }

            # Save result
            result_file = output_dir / f"simulation_{i + 1}.json"
            result_paths.append(result_file)

            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, cls=DataclassJSONEncoder)

            logger.info(f"Completed simulation {i + 1}")

        except Exception as e:
            logger.error(f"Error in simulation {i + 1}: {str(e)}")
            logger.exception(e)
    return result_paths


if __name__ == "__main__":
    main()
