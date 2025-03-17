import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

from src.executeinitial import setup_logging, execute_reactive_samples_parallel, load_simulation_inputs, \
    execute_proactive_samples_parallel
from src.generator.traces import generate_time_series
from src.placement.model import DataclassJSONEncoder, TimeSeries, SimulationData
from src.train import train_model, save_models


def generate_base_workload(sim_input_path, app, rps, ts_path):
    sim_inputs = load_simulation_inputs(sim_input_path)
    simulation_data = SimulationData(
        platform_types=sim_inputs['platform_types'],
        storage_types=sim_inputs['storage_types'],
        qos_types=sim_inputs['qos_types'],
        application_types=sim_inputs['application_types'],
        task_types=sim_inputs['task_types'],
    )
    time_series: TimeSeries = generate_time_series(
        simulation_data, rps, 350, 'poisson-increasing', app, str(ts_path).replace('.json', ''), peaks=None
    )
    return time_series

def main():
    base_dir = Path(sys.argv[1])
    sim_input_path = Path("data/nofs-ids")  # Base path for simulation input files
    samples_file = base_dir / "lhs_samples.npy"
    mapping_file = base_dir / "lhs_samples_mapping.pkl"
    config_file = base_dir / "space.json"
    # workload_base_file = "data/nofs-ids/traces/workload-125-250.json"
    reactive_output_dir = base_dir / "initial_results_reactive"
    proactive_output_dir = base_dir / "initial_results"
    os.makedirs(reactive_output_dir, exist_ok=True)
    os.makedirs(proactive_output_dir, exist_ok=True)
    max_workers = int(sys.argv[2])
    app = sys.argv[3]
    # Setup logging
    logger = setup_logging(Path("/tmp"))

    logger.info("Starting simulation preparation")

    # Load samples and mapping
    logger.info("Loading samples and mapping")
    samples = np.load(samples_file)
    with open(mapping_file, 'rb') as f:
        mapping = pickle.load(f)

    logger.info(f'Testing {len(samples)}')
    # Load infrastructure config
    logger.info("Loading infrastructure configuration")
    with open(config_file, 'r') as f:
        space_config = json.load(f)

    ts_path = base_dir / 'baseline_workload.json'
    ts = generate_base_workload(sim_input_path, app, space_config['wsc'][app]['average'], ts_path)
    with open(ts_path, 'w') as fd:
        json.dump(ts, fd, indent=2, cls=DataclassJSONEncoder)

    apps = [app]
    reactive_results_paths = execute_reactive_samples_parallel(apps, config_file, mapping_file, reactive_output_dir, samples,
                                                               sim_input_path, ts_path, max_workers)
    print(reactive_results_paths)
    logger.info("Completed all simulations")

    logger.info("Training model now...")
    models, eval_results = train_model(reactive_output_dir, samples, include_queue_length=False)
    model_paths = save_models(models, proactive_output_dir)

    for result_path in reactive_results_paths:
        os.remove(result_path)

    # proactive_results_paths = execute_proactive_samples_parallel(apps, config_file, mapping_file, proactive_output_dir, samples,
    #                                                            sim_input_path, ts_path, max_workers, model_paths)

    logger.info('Finished model training')
    logger.info(f'All files can be found under {proactive_output_dir}')


if __name__ == '__main__':
    main()
